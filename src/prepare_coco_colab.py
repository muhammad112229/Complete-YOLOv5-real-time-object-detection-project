"""Reconstruct the verified COCO YOLOv5 dataset split inside Google Colab.

This script is designed for Colab. It downloads official COCO 2017 archives in
Colab storage, reconstructs the exact local image-level split from transferred
manifests, regenerates YOLO labels from COCO annotations, and validates counts
against local reference checksums.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import shutil
import sys
import urllib.request
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


LOGGER = logging.getLogger(__name__)

COCO_ARCHIVES = {
    "train2017": {
        "url": "http://images.cocodataset.org/zips/train2017.zip",
        "filename": "train2017.zip",
        "expected_size_bytes": 19_336_861_798,
        "top_level": "train2017",
        "expected_images": 118_287,
    },
    "val2017": {
        "url": "http://images.cocodataset.org/zips/val2017.zip",
        "filename": "val2017.zip",
        "expected_size_bytes": 815_585_330,
        "top_level": "val2017",
        "expected_images": 5_000,
    },
    "annotations_trainval2017": {
        "url": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
        "filename": "annotations_trainval2017.zip",
        "expected_size_bytes": 252_907_541,
        "top_level": "annotations",
    },
}

EXPECTED_COUNTS = {
    "train": 98_629,
    "val": 12_328,
    "test": 12_330,
    "accepted_annotations": 886_282,
    "excluded_crowd_annotations": 10_498,
    "rejected_invalid_boxes": 2,
    "classes": 80,
    "seed": 42,
}

CHUNK_SIZE = 8 * 1024 * 1024


@dataclass(frozen=True)
class ColabPaths:
    """Resolved Colab reconstruction paths."""

    workspace: Path
    storage_root: Path
    dataset_root: Path
    archive_root: Path
    coco_root: Path
    processed_root: Path
    manifests_dir: Path
    artifacts_root: Path


def utc_now() -> str:
    """Return an ISO UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def ensure_dir(path: Path) -> Path:
    """Create a directory and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, data: Any) -> None:
    """Write JSON with UTF-8 encoding."""
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def sha256_text(text: str) -> str:
    """Hash text with SHA-256."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path, chunk_size: int = CHUNK_SIZE) -> str:
    """Hash a file with SHA-256."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def paths_for(workspace: Path, storage_root: Path, storage_mode: str, manifests_dir: Path) -> ColabPaths:
    """Resolve Colab dataset and artifact paths."""
    workspace = workspace.resolve()
    storage_root = storage_root.resolve()
    if storage_mode not in {"runtime", "drive"}:
        raise ValueError("storage_mode must be 'runtime' or 'drive'")
    dataset_root = storage_root / "coco2017"
    return ColabPaths(
        workspace=workspace,
        storage_root=storage_root,
        dataset_root=dataset_root,
        archive_root=dataset_root / "archives",
        coco_root=dataset_root / "raw" / "coco2017",
        processed_root=dataset_root / "coco_yolo_exact_split",
        manifests_dir=manifests_dir.resolve(),
        artifacts_root=workspace / "artifacts",
    )


def download_with_resume(url: str, destination: Path, expected_size: int) -> dict[str, Any]:
    """Download a URL with simple Range resume support."""
    ensure_dir(destination.parent)
    part = destination.with_suffix(destination.suffix + ".part")
    if destination.exists() and destination.stat().st_size == expected_size:
        return {"path": str(destination), "status": "skipped_existing", "size_bytes": expected_size}
    downloaded = part.stat().st_size if part.exists() else 0
    headers = {"Range": f"bytes={downloaded}-"} if downloaded else {}
    request = urllib.request.Request(url, headers=headers)
    mode = "ab" if downloaded else "wb"
    LOGGER.info("Downloading %s to %s", url, destination)
    with urllib.request.urlopen(request) as response, part.open(mode + "") as output:  # noqa: S310 official COCO URL
        if downloaded and response.status == 200:
            output.seek(0)
            output.truncate()
            downloaded = 0
        while True:
            chunk = response.read(CHUNK_SIZE)
            if not chunk:
                break
            output.write(chunk)
    actual = part.stat().st_size
    if actual != expected_size:
        raise RuntimeError(f"Incomplete download for {destination}: {actual} != {expected_size}")
    part.replace(destination)
    return {"path": str(destination), "status": "downloaded", "size_bytes": actual}


def verify_zip(path: Path) -> None:
    """Validate a ZIP archive."""
    with zipfile.ZipFile(path) as archive:
        bad = archive.testzip()
    if bad:
        raise ValueError(f"Corrupt ZIP entry: {bad}")


def download_archives(paths: ColabPaths) -> dict[str, Any]:
    """Download official COCO archives if missing or invalid."""
    report: dict[str, Any] = {}
    ensure_dir(paths.archive_root)
    for name, meta in COCO_ARCHIVES.items():
        archive_path = paths.archive_root / str(meta["filename"])
        expected = int(meta["expected_size_bytes"])
        if archive_path.exists() and archive_path.stat().st_size == expected:
            verify_zip(archive_path)
            status = {"path": str(archive_path), "status": "skipped_verified", "size_bytes": expected}
        else:
            status = download_with_resume(str(meta["url"]), archive_path, expected)
            verify_zip(archive_path)
        report[name] = status
    return report


def extract_archives(paths: ColabPaths) -> dict[str, Any]:
    """Extract official COCO archives if needed."""
    ensure_dir(paths.coco_root)
    report: dict[str, Any] = {}
    for name, meta in COCO_ARCHIVES.items():
        archive_path = paths.archive_root / str(meta["filename"])
        top = paths.coco_root / str(meta["top_level"])
        if top.exists():
            report[name] = {"status": "skipped_existing", "target": str(top)}
            continue
        verify_zip(archive_path)
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(paths.coco_root)
        report[name] = {"status": "extracted", "target": str(top)}
    return report


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object."""
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def build_category_mapping(categories: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    """Build contiguous YOLO class mapping from COCO categories."""
    mapping: dict[int, dict[str, Any]] = {}
    for class_id, category in enumerate(sorted(categories, key=lambda item: int(item["id"]))):
        coco_id = int(category["id"])
        mapping[coco_id] = {
            "class_id": class_id,
            "coco_category_id": coco_id,
            "name": str(category["name"]),
            "supercategory": str(category.get("supercategory", "")),
        }
    return mapping


def clip_coco_bbox(bbox: list[float], image_width: int, image_height: int) -> tuple[float, float, float, float]:
    """Clip a COCO bbox and convert it to normalized YOLO xywh."""
    x_min, y_min, width, height = (float(value) for value in bbox)
    x1 = max(0.0, x_min)
    y1 = max(0.0, y_min)
    x2 = min(float(image_width), x_min + width)
    y2 = min(float(image_height), y_min + height)
    if x2 <= x1 or y2 <= y1:
        raise ValueError("bbox has zero or negative area after clipping")
    box_width = x2 - x1
    box_height = y2 - y1
    return (
        (x1 + box_width / 2.0) / image_width,
        (y1 + box_height / 2.0) / image_height,
        box_width / image_width,
        box_height / image_height,
    )


def load_coco_annotations(paths: ColabPaths) -> tuple[dict[str, dict[str, Any]], dict[int, dict[str, Any]]]:
    """Load train2017 and val2017 COCO annotations."""
    sources: dict[str, dict[str, Any]] = {}
    mapping: dict[int, dict[str, Any]] | None = None
    for source_split in ("train2017", "val2017"):
        annotation_path = paths.coco_root / "annotations" / f"instances_{source_split}.json"
        data = load_json(annotation_path)
        mapping = mapping or build_category_mapping(data.get("categories", []))
        images = {int(image["id"]): image for image in data.get("images", [])}
        annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for annotation in data.get("annotations", []):
            annotations_by_image[int(annotation["image_id"])].append(annotation)
        sources[source_split] = {"images": images, "annotations_by_image": annotations_by_image}
    if mapping is None:
        raise RuntimeError("No COCO categories loaded")
    return sources, mapping


def manifest_rows(manifest_path: Path) -> list[dict[str, str]]:
    """Read a split manifest CSV."""
    with manifest_path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def source_filename(row: dict[str, str]) -> str:
    """Get original COCO file name from a local manifest row without using Windows paths."""
    source_path = row.get("source_image_path", "")
    if "\\" in source_path:
        return source_path.split("\\")[-1]
    return Path(source_path).name


def project_filename(row: dict[str, str], field: str, suffix: str) -> str:
    """Get local project filename from a manifest row without using Windows paths."""
    value = row.get(field, "")
    filename = value.split("\\")[-1] if "\\" in value else Path(value).name
    if not filename:
        source_split = row["source_split"]
        original = Path(source_filename(row)).stem
        filename = f"{source_split}_{original}{suffix}"
    return filename


def link_or_copy(source: Path, destination: Path) -> str:
    """Create a hardlink or symlink to source, falling back to copy."""
    ensure_dir(destination.parent)
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        try:
            destination.symlink_to(source)
            return "symlink"
        except OSError:
            shutil.copy2(source, destination)
            return "copy_fallback"


def write_label(path: Path, labels: list[tuple[int, tuple[float, float, float, float]]]) -> None:
    """Write a YOLO label file."""
    ensure_dir(path.parent)
    lines = [f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}" for class_id, (x, y, w, h) in labels]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def split_identity_hash(rows: list[dict[str, str]]) -> str:
    """Hash split source identity keys."""
    keys = [f"{row['source_split']}:{int(row['image_id']):012d}" for row in rows]
    return sha256_text("\n".join(sorted(keys)) + "\n")


def write_dataset_yaml(path: Path, dataset_root: Path, names: dict[int, str]) -> None:
    """Write a Linux-compatible YOLOv5 dataset YAML."""
    lines = [
        f"path: {dataset_root.as_posix()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "nc: 80",
        "names:",
    ]
    lines.extend(f"  {index}: {name}" for index, name in sorted(names.items()))
    ensure_dir(path.parent)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def reconstruct_split(paths: ColabPaths, split: str, rows: list[dict[str, str]], sources: dict[str, Any], mapping: dict[int, dict[str, Any]]) -> dict[str, Any]:
    """Reconstruct one split using the transferred manifest rows."""
    image_dir = ensure_dir(paths.processed_root / "images" / split)
    label_dir = ensure_dir(paths.processed_root / "labels" / split)
    image_list: list[str] = []
    counters: Counter[str] = Counter()
    for row in rows:
        source_split = row["source_split"]
        image_id = int(row["image_id"])
        source_name = source_filename(row)
        image_info = sources[source_split]["images"].get(image_id)
        if not image_info:
            raise RuntimeError(f"Missing COCO image metadata for {source_split}:{image_id}")
        width = int(image_info["width"])
        height = int(image_info["height"])
        source_image = paths.coco_root / source_split / source_name
        if not source_image.exists():
            raise FileNotFoundError(f"Missing source image: {source_image}")
        dest_image = image_dir / project_filename(row, "project_image_path", ".jpg")
        dest_label = label_dir / project_filename(row, "project_label_path", ".txt")
        strategy = link_or_copy(source_image, dest_image)
        counters[f"image_{strategy}"] += 1
        labels: list[tuple[int, tuple[float, float, float, float]]] = []
        for annotation in sources[source_split]["annotations_by_image"].get(image_id, []):
            category_id = int(annotation.get("category_id", -1))
            if int(annotation.get("iscrowd", 0)) == 1:
                counters["excluded_crowd_annotations"] += 1
                continue
            if category_id not in mapping:
                counters["rejected_annotations"] += 1
                continue
            try:
                yolo_box = clip_coco_bbox(annotation["bbox"], width, height)
            except Exception:
                counters["rejected_invalid_boxes"] += 1
                continue
            labels.append((int(mapping[category_id]["class_id"]), yolo_box))
        write_label(dest_label, labels)
        counters["accepted_annotations"] += len(labels)
        image_list.append(dest_image.as_posix())
    list_path = paths.processed_root / f"{split}_images.txt"
    list_path.write_text("\n".join(image_list) + "\n", encoding="utf-8")
    return {
        "split": split,
        "image_count": len(rows),
        "label_count": len(list(label_dir.glob("*.txt"))),
        "image_list": str(list_path),
        "source_identity_sha256": split_identity_hash(rows),
        "counters": dict(counters),
    }


def reconstruct_dataset(paths: ColabPaths) -> dict[str, Any]:
    """Reconstruct all project splits and labels."""
    ensure_dir(paths.processed_root)
    sources, mapping = load_coco_annotations(paths)
    split_reports: dict[str, Any] = {}
    totals = Counter()
    for split in ("train", "val", "test"):
        rows = manifest_rows(paths.manifests_dir / f"{split}_manifest.csv")
        report = reconstruct_split(paths, split, rows, sources, mapping)
        split_reports[split] = report
        totals.update(report["counters"])
    class_names = {int(item["class_id"]): str(item["name"]) for item in mapping.values()}
    write_dataset_yaml(paths.processed_root / "coco_project.yaml", paths.processed_root, class_names)
    write_json(paths.processed_root / "class_mapping.json", {str(key): value for key, value in sorted(mapping.items())})
    (paths.processed_root / "class_names.txt").write_text(
        "\n".join(class_names[index] for index in range(len(class_names))) + "\n",
        encoding="utf-8",
    )
    return {
        "processed_root": str(paths.processed_root),
        "dataset_yaml": str(paths.processed_root / "coco_project.yaml"),
        "splits": split_reports,
        "totals": dict(totals),
        "class_count": len(class_names),
    }


def validate_extraction(paths: ColabPaths) -> dict[str, Any]:
    """Validate extracted source data counts and annotation JSON availability."""
    report = {
        "train2017_images": len(list((paths.coco_root / "train2017").glob("*.jpg"))),
        "val2017_images": len(list((paths.coco_root / "val2017").glob("*.jpg"))),
        "instances_train_exists": (paths.coco_root / "annotations" / "instances_train2017.json").exists(),
        "instances_val_exists": (paths.coco_root / "annotations" / "instances_val2017.json").exists(),
    }
    errors = []
    if report["train2017_images"] != 118_287:
        errors.append("train2017 image count mismatch")
    if report["val2017_images"] != 5_000:
        errors.append("val2017 image count mismatch")
    if not report["instances_train_exists"] or not report["instances_val_exists"]:
        errors.append("missing COCO instance annotation JSON")
    report["valid"] = not errors
    report["errors"] = errors
    return report


def load_reference(paths: ColabPaths) -> dict[str, Any]:
    """Load transferred local reference checksums."""
    return load_json(paths.workspace / "artifacts" / "colab_reference_checksums.json")


def compare_integrity(paths: ColabPaths, reconstruction: dict[str, Any]) -> dict[str, Any]:
    """Compare reconstructed dataset against local reference counts and hashes."""
    reference = load_reference(paths)
    rows = []
    for split in ("train", "val", "test"):
        expected_count = EXPECTED_COUNTS[split]
        actual_count = int(reconstruction["splits"][split]["image_count"])
        expected_hash = reference["split_identity"][split]["sha256"]
        actual_hash = reconstruction["splits"][split]["source_identity_sha256"]
        rows.append({"item": f"{split}_image_count", "expected": expected_count, "actual": actual_count, "status": "PASS" if expected_count == actual_count else "FAIL"})
        rows.append({"item": f"{split}_identity_sha256", "expected": expected_hash, "actual": actual_hash, "status": "PASS" if expected_hash == actual_hash else "FAIL"})
    total_accepted = int(reconstruction["totals"].get("accepted_annotations", 0))
    total_crowd = int(reconstruction["totals"].get("excluded_crowd_annotations", 0))
    total_invalid = int(reconstruction["totals"].get("rejected_invalid_boxes", 0))
    comparisons = [
        ("accepted_annotations", EXPECTED_COUNTS["accepted_annotations"], total_accepted),
        ("excluded_crowd_annotations", EXPECTED_COUNTS["excluded_crowd_annotations"], total_crowd),
        ("rejected_invalid_boxes", EXPECTED_COUNTS["rejected_invalid_boxes"], total_invalid),
        ("class_count", EXPECTED_COUNTS["classes"], int(reconstruction["class_count"])),
    ]
    for item, expected, actual in comparisons:
        rows.append({"item": item, "expected": expected, "actual": actual, "status": "PASS" if expected == actual else "FAIL"})
    return {"rows": rows, "valid": all(row["status"] == "PASS" for row in rows)}


def dataloader_smoke(paths: ColabPaths, yolov5_root: Path, imgsz: int = 640) -> dict[str, Any]:
    """Load one batch using YOLOv5's native dataloader."""
    if str(yolov5_root) not in sys.path:
        sys.path.insert(0, str(yolov5_root))
    from utils import dataloaders  # type: ignore

    dataloader, dataset = dataloaders.create_dataloader(
        path=str(paths.processed_root / "images" / "train"),
        imgsz=imgsz,
        batch_size=2,
        stride=32,
        single_cls=False,
        pad=0.5,
        rect=False,
        workers=0,
        prefix="colab-smoke: ",
    )
    images, labels, batch_paths, _ = next(iter(dataloader))
    return {
        "status": "passed",
        "batch_image_tensor_shape": list(images.shape),
        "batch_label_tensor_shape": list(labels.shape),
        "label_count": int(labels.shape[0]),
        "batch_paths": [str(path) for path in batch_paths],
        "dataset_length": len(dataset),
        "device": "cuda" if "torch" in sys.modules and sys.modules["torch"].cuda.is_available() else "cpu",
        "imgsz": imgsz,
    }


def run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    """Run requested Colab reconstruction stages."""
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(asctime)s | %(levelname)s | %(message)s")
    paths = paths_for(Path(args.workspace), Path(args.storage_root), args.storage_mode, Path(args.manifests_dir))
    ensure_dir(paths.artifacts_root)
    report: dict[str, Any] = {
        "generated_at_utc": utc_now(),
        "storage_mode": args.storage_mode,
        "paths": {key: str(value) for key, value in paths.__dict__.items()},
        "stages": {},
    }
    if args.download:
        report["stages"]["download"] = download_archives(paths)
    if args.extract:
        report["stages"]["extract"] = extract_archives(paths)
    if args.validate:
        report["stages"]["source_validation"] = validate_extraction(paths)
        if not report["stages"]["source_validation"]["valid"]:
            write_json(paths.artifacts_root / "colab_reconstruction_report.json", report)
            raise RuntimeError(f"Source validation failed: {report['stages']['source_validation']['errors']}")
    if args.reconstruct:
        reconstruction = reconstruct_dataset(paths)
        report["stages"]["reconstruct"] = reconstruction
        report["stages"]["integrity"] = compare_integrity(paths, reconstruction)
        if not report["stages"]["integrity"]["valid"]:
            write_json(paths.artifacts_root / "colab_reconstruction_report.json", report)
            raise RuntimeError("Integrity comparison failed")
    if args.dataloader_smoke:
        report["stages"]["dataloader_smoke"] = dataloader_smoke(paths, Path(args.yolov5_root), args.imgsz)
    write_json(paths.artifacts_root / "colab_reconstruction_report.json", report)
    return report


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Prepare exact COCO split reconstruction in Colab.")
    parser.add_argument("--workspace", default="/content/yolov5_project")
    parser.add_argument("--storage-root", default="/content/datasets")
    parser.add_argument("--storage-mode", choices=["runtime", "drive"], default="runtime")
    parser.add_argument("--manifests-dir", default="/content/yolov5_project/data/splits")
    parser.add_argument("--yolov5-root", default="/content/yolov5")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--extract", action="store_true")
    parser.add_argument("--reconstruct", action="store_true")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--dataloader-smoke", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> int:
    """CLI entrypoint."""
    report = run_pipeline(parse_args())
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
