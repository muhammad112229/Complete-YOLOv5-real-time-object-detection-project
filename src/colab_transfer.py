"""Build compact Google Colab transfer artifacts for the YOLOv5 COCO project."""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.common import ensure_dir, load_yaml, project_root


EXPECTED_REFERENCE = {
    "seed": 42,
    "train_image_count": 98_629,
    "val_image_count": 12_328,
    "test_image_count": 12_330,
    "accepted_annotation_count": 886_282,
    "excluded_crowd_annotation_count": 10_498,
    "rejected_invalid_box_count": 2,
    "class_count": 80,
}

PROHIBITED_PATTERNS = [
    ".venv/*",
    "data/raw/*",
    "data/processed/coco_yolo/images/*",
    "data/processed/coco_yolo/labels/*",
    "data/smoke/*",
    "external/yolov5/.git/*",
    "models/pretrained/*.pt",
    "models/trained/*",
    "models/optimized/*",
    "outputs/videos/*",
    "outputs/webcam/*",
    "results/yolov5s/smoke_test/*",
    "**/__pycache__/*",
    "**/*.cache",
    "**/.pytest_cache/*",
]


@dataclass(frozen=True)
class BundleValidation:
    """Validation result for a compact transfer bundle."""

    valid: bool
    errors: list[str]
    file_count: int
    size_bytes: int


def utc_now() -> str:
    """Return a UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, data: Any) -> None:
    """Write JSON with UTF-8 encoding."""
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Return the SHA-256 digest for a file."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    """Return the SHA-256 digest for normalized text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def should_exclude(path: str) -> bool:
    """Return whether a relative POSIX path is prohibited in the bundle."""
    normalized = path.replace("\\", "/")
    return any(fnmatch.fnmatch(normalized, pattern) for pattern in PROHIBITED_PATTERNS)


def required_bundle_files(root: Path) -> list[Path]:
    """Return the smallest portable file set needed for Colab reconstruction."""
    relative = [
        "README.md",
        "requirements.txt",
        "requirements-dev.txt",
        "pyproject.toml",
        "notebooks/YOLOv5_COCO_Training_Colab.ipynb",
        "configs/project.yaml",
        "configs/coco_project_colab.yaml",
        "configs/train_yolov5s.yaml",
        "configs/train_yolov5m.yaml",
        "configs/train_yolov5l.yaml",
        "configs/train_yolov5s_smoke.yaml",
        "configs/train_yolov5s_colab.yaml",
        "configs/train_yolov5m_colab.yaml",
        "configs/train_yolov5l_colab.yaml",
        "src/__init__.py",
        "src/common.py",
        "src/download_coco.py",
        "src/parse_coco_annotations.py",
        "src/prepare_coco_colab.py",
        "src/colab_transfer.py",
        "src/train_models.py",
        "data/splits/train_manifest.csv",
        "data/splits/val_manifest.csv",
        "data/splits/test_manifest.csv",
        "data/splits/train_images.txt",
        "data/splits/val_images.txt",
        "data/splits/test_images.txt",
        "data/splits/split_summary.json",
        "data/processed/coco_yolo/coco_project.yaml",
        "data/processed/coco_yolo/class_mapping.json",
        "data/processed/coco_yolo/class_names.txt",
        "artifacts/coco_dataset_statistics.json",
        "artifacts/coco_dataset_validation.json",
        "artifacts/colab_training_transfer_manifest.json",
        "artifacts/colab_training_transfer_manifest.md",
        "artifacts/colab_reference_checksums.json",
        "artifacts/colab_reference_checksums.md",
    ]
    files = [root / item for item in relative]
    missing = [str(path.relative_to(root)) for path in files if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Required bundle files are missing: {missing}")
    return files


def reference_checksum_files(root: Path) -> list[Path]:
    """Return files whose hashes define the local Colab reconstruction reference."""
    relative = [
        "data/splits/train_images.txt",
        "data/splits/val_images.txt",
        "data/splits/test_images.txt",
        "data/splits/train_manifest.csv",
        "data/splits/val_manifest.csv",
        "data/splits/test_manifest.csv",
        "data/splits/split_summary.json",
        "data/processed/coco_yolo/class_mapping.json",
        "data/processed/coco_yolo/class_names.txt",
        "configs/coco_project_colab.yaml",
        "src/prepare_coco_colab.py",
        "src/download_coco.py",
        "src/parse_coco_annotations.py",
        "src/colab_transfer.py",
        "notebooks/YOLOv5_COCO_Training_Colab.ipynb",
        "requirements.txt",
    ]
    return [root / item for item in relative]


def split_identity_hash(manifest_path: Path) -> dict[str, Any]:
    """Hash source split and image IDs from a split manifest."""
    import csv

    keys: list[str] = []
    with manifest_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            keys.append(f"{row['source_split']}:{int(row['image_id']):012d}")
    text = "\n".join(sorted(keys)) + "\n"
    return {"count": len(keys), "sha256": sha256_text(text)}


def dataset_yaml_summary(path: Path) -> dict[str, Any]:
    """Return a compact summary for a YOLO dataset YAML."""
    data = load_yaml(path)
    names = data.get("names", {})
    return {
        "path": data.get("path"),
        "train": data.get("train"),
        "val": data.get("val"),
        "test": data.get("test"),
        "nc": int(data.get("nc", 0)),
        "name_count": len(names),
    }


def build_reference_checksums(root: Path) -> dict[str, Any]:
    """Calculate reference checksums and expected counts from actual local files."""
    root = root.resolve()
    statistics = json.loads((root / "artifacts" / "coco_dataset_statistics.json").read_text(encoding="utf-8"))
    validation = json.loads((root / "artifacts" / "coco_dataset_validation.json").read_text(encoding="utf-8"))
    split_summary = json.loads((root / "data" / "splits" / "split_summary.json").read_text(encoding="utf-8"))
    files = {}
    for path in reference_checksum_files(root):
        rel = path.relative_to(root).as_posix()
        files[rel] = {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}
    split_identity = {
        "train": split_identity_hash(root / "data" / "splits" / "train_manifest.csv"),
        "val": split_identity_hash(root / "data" / "splits" / "val_manifest.csv"),
        "test": split_identity_hash(root / "data" / "splits" / "test_manifest.csv"),
    }
    reference = {
        "generated_at_utc": utc_now(),
        "strategy": "Download official COCO archives in Colab, then reconstruct exact split from transferred manifests.",
        "expected": {
            **EXPECTED_REFERENCE,
            "split_summary_counts": split_summary.get("counts"),
            "split_summary_annotation_counts": split_summary.get("annotation_counts"),
            "statistics_counts": {
                "total_source_images": statistics.get("total_source_images"),
                "accepted_annotations": statistics.get("accepted_annotations"),
                "excluded_crowd_annotations": statistics.get("excluded_crowd_annotations"),
                "invalid_boxes": statistics.get("invalid_boxes"),
            },
            "validation_final_readiness": validation.get("final_readiness"),
        },
        "files": files,
        "split_identity": split_identity,
        "dataset_yaml_template": dataset_yaml_summary(root / "configs" / "coco_project_colab.yaml"),
    }
    write_json(root / "artifacts" / "colab_reference_checksums.json", reference)
    lines = [
        "# Colab Reference Checksums",
        "",
        f"Generated UTC: `{reference['generated_at_utc']}`",
        "",
        "## Reference Counts",
        f"- Train images: {EXPECTED_REFERENCE['train_image_count']}",
        f"- Validation images: {EXPECTED_REFERENCE['val_image_count']}",
        f"- Test images: {EXPECTED_REFERENCE['test_image_count']}",
        f"- Accepted annotations: {EXPECTED_REFERENCE['accepted_annotation_count']}",
        f"- Excluded crowd annotations: {EXPECTED_REFERENCE['excluded_crowd_annotation_count']}",
        f"- Rejected invalid boxes: {EXPECTED_REFERENCE['rejected_invalid_box_count']}",
        f"- Classes: {EXPECTED_REFERENCE['class_count']}",
        f"- Seed: {EXPECTED_REFERENCE['seed']}",
        "",
        "## File SHA-256",
    ]
    for rel, item in files.items():
        lines.append(f"- `{rel}`: `{item['sha256']}` ({item['size_bytes']} bytes)")
    lines.extend(["", "## Split Identity Hashes"])
    for split, item in split_identity.items():
        lines.append(f"- {split}: count={item['count']}, sha256=`{item['sha256']}`")
    (root / "artifacts" / "colab_reference_checksums.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    return reference


def validate_bundle(zip_path: Path) -> BundleValidation:
    """Validate bundle integrity and exclusion rules."""
    errors: list[str] = []
    try:
        with zipfile.ZipFile(zip_path) as archive:
            bad = archive.testzip()
            if bad:
                errors.append(f"Corrupt ZIP member: {bad}")
            infos = archive.infolist()
            for info in infos:
                name = info.filename.replace("\\", "/")
                if name.startswith("/") or Path(name).is_absolute():
                    errors.append(f"Absolute path in ZIP: {name}")
                parts = Path(name).parts
                if ".." in parts:
                    errors.append(f"Path traversal entry in ZIP: {name}")
                if should_exclude(name):
                    errors.append(f"Prohibited path in ZIP: {name}")
                if name.lower().endswith((".pt", ".pth")):
                    errors.append(f"Model checkpoint must not be bundled: {name}")
                if name.lower().endswith((".jpg", ".jpeg", ".png", ".mp4", ".avi")) and name.startswith("data/"):
                    errors.append(f"Dataset media must not be bundled: {name}")
    except zipfile.BadZipFile as exc:
        errors.append(str(exc))
        infos = []
    return BundleValidation(
        valid=not errors,
        errors=errors,
        file_count=len(infos),
        size_bytes=zip_path.stat().st_size if zip_path.exists() else 0,
    )


def create_bundle(root: Path) -> dict[str, Any]:
    """Create and validate the compact Colab bundle."""
    root = root.resolve()
    transfer_root = ensure_dir(root / "transfer")
    bundle_path = transfer_root / "yolov5_colab_bundle.zip"
    contents_path = transfer_root / "yolov5_colab_bundle_contents.txt"
    sha_path = transfer_root / "yolov5_colab_bundle.sha256"
    files = required_bundle_files(root)
    with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in files:
            rel = path.relative_to(root).as_posix()
            if should_exclude(rel):
                raise ValueError(f"Refusing to bundle prohibited path: {rel}")
            archive.write(path, rel)
    validation = validate_bundle(bundle_path)
    if not validation.valid:
        raise RuntimeError(f"Bundle validation failed: {validation.errors}")
    digest = sha256_file(bundle_path)
    sha_path.write_text(f"{digest}  {bundle_path.name}\n", encoding="utf-8")
    with zipfile.ZipFile(bundle_path) as archive:
        names = sorted(info.filename for info in archive.infolist())
    contents_path.write_text("\n".join(names) + "\n", encoding="utf-8")
    report = {
        "generated_at_utc": utc_now(),
        "bundle_path": str(bundle_path),
        "sha256_path": str(sha_path),
        "contents_path": str(contents_path),
        "sha256": digest,
        "size_bytes": bundle_path.stat().st_size,
        "file_count": len(names),
        "validation": {
            "valid": validation.valid,
            "errors": validation.errors,
            "file_count": validation.file_count,
            "size_bytes": validation.size_bytes,
        },
        "included_files": names,
        "excluded_patterns": PROHIBITED_PATTERNS,
    }
    write_json(root / "artifacts" / "colab_bundle_report.json", report)
    return report


def main() -> int:
    """CLI entrypoint for checksum and bundle generation."""
    parser = argparse.ArgumentParser(description="Build Colab transfer checksums and bundle.")
    parser.add_argument("--write-checksums", action="store_true")
    parser.add_argument("--create-bundle", action="store_true")
    parser.add_argument("--validate-bundle", type=Path)
    args = parser.parse_args()
    root = project_root()
    if args.write_checksums:
        build_reference_checksums(root)
    if args.create_bundle:
        create_bundle(root)
    if args.validate_bundle:
        result = validate_bundle(args.validate_bundle)
        print(json.dumps(result.__dict__, indent=2))
        return 0 if result.valid else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
