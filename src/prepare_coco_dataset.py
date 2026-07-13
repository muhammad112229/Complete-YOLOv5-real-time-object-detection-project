"""Prepare COCO 2017 object-detection data for YOLOv5 training."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import shutil
import statistics
import sys
import time
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.common import ensure_dir, project_root, require_python_package, setup_logging, yolov5_root
from src.download_coco import COCO_2017_ARCHIVES, verify_zip
from src.parse_coco_annotations import build_category_mapping, coco_bbox_to_yolo


LOGGER = logging.getLogger(__name__)
SPLITS = ("train", "val", "test")
SOURCE_SPLITS = ("train2017", "val2017")


@dataclass(frozen=True)
class PreparedPaths:
    """Common COCO preparation paths."""

    root: Path
    raw_root: Path
    coco_root: Path
    archives_root: Path
    interim_root: Path
    processed_root: Path
    split_root: Path
    artifacts_root: Path
    results_root: Path
    validation_output_root: Path
    preprocessing_output_root: Path


def paths_for(root: Path) -> PreparedPaths:
    """Return standardized paths for the COCO pipeline."""
    return PreparedPaths(
        root=root,
        raw_root=root / "data" / "raw",
        coco_root=root / "data" / "raw" / "coco2017",
        archives_root=root / "data" / "raw" / "archives",
        interim_root=root / "data" / "interim",
        processed_root=root / "data" / "processed" / "coco_yolo",
        split_root=root / "data" / "splits",
        artifacts_root=root / "artifacts",
        results_root=root / "results" / "dataset_analysis",
        validation_output_root=root / "outputs" / "images" / "dataset_validation",
        preprocessing_output_root=root / "outputs" / "images" / "dataset_preprocessing",
    )


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object from disk."""
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def write_json(path: Path, data: Any, indent: int | None = 2) -> None:
    """Write JSON with UTF-8 encoding."""
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, indent=indent), encoding="utf-8")


def check_image_readable(path: Path) -> tuple[str, bool, str | None]:
    """Return whether an image can be opened by Pillow."""
    Image = require_python_package("PIL.Image", "Pillow")
    try:
        with Image.open(path) as image:
            image.verify()
        return str(path), True, None
    except Exception as exc:
        return str(path), False, str(exc)


def count_images(path: Path) -> int:
    """Count JPEG images under a directory."""
    return sum(1 for item in path.glob("*.jpg") if item.is_file())


def validate_raw_extraction(paths: PreparedPaths, max_workers: int = 8) -> dict[str, Any]:
    """Validate downloaded archives and extracted COCO data."""
    ensure_dir(paths.artifacts_root)
    archives: dict[str, Any] = {}
    for name, metadata in COCO_2017_ARCHIVES.items():
        archive_path = paths.archives_root / str(metadata["filename"])
        archive_report = {
            "path": str(archive_path),
            "exists": archive_path.exists(),
            "expected_size_bytes": metadata["expected_size_bytes"],
            "actual_size_bytes": archive_path.stat().st_size if archive_path.exists() else 0,
            "zip_integrity": "not_run",
            "error": None,
        }
        if archive_path.exists() and archive_path.stat().st_size > 0:
            marker = archive_path.with_suffix(archive_path.suffix + ".verified")
            if marker.exists():
                marker_data = load_json(marker)
                if marker_data.get("size_bytes") == archive_path.stat().st_size and marker_data.get("zip_integrity") == "passed":
                    archive_report["zip_integrity"] = "passed"
                    archive_report["verification_method"] = marker_data.get("method")
                else:
                    archive_report["zip_integrity"] = "failed"
                    archive_report["error"] = "verification marker does not match archive size"
            else:
                try:
                    verify_zip(archive_path)
                    archive_report["zip_integrity"] = "passed"
                    archive_report["verification_method"] = "zipfile.testzip"
                except Exception as exc:
                    archive_report["zip_integrity"] = "failed"
                    archive_report["error"] = str(exc)
        archives[name] = archive_report

    image_roots = {
        "train2017": paths.coco_root / "train2017",
        "val2017": paths.coco_root / "val2017",
    }
    annotation_paths = {
        "train2017": paths.coco_root / "annotations" / "instances_train2017.json",
        "val2017": paths.coco_root / "annotations" / "instances_val2017.json",
    }
    image_counts = {name: count_images(path) for name, path in image_roots.items()}

    json_reports: dict[str, Any] = {}
    missing_images: dict[str, list[str]] = {}
    duplicate_file_names: dict[str, int] = {}
    for split_name, annotation_path in annotation_paths.items():
        report = {"path": str(annotation_path), "exists": annotation_path.exists(), "valid_json": False, "pycocotools_load": False}
        if annotation_path.exists():
            try:
                data = load_json(annotation_path)
                report["valid_json"] = True
                report["images"] = len(data.get("images", []))
                report["annotations"] = len(data.get("annotations", []))
                report["categories"] = len(data.get("categories", []))
                filenames = [str(image["file_name"]) for image in data.get("images", [])]
                duplicate_file_names[split_name] = len(filenames) - len(set(filenames))
                missing_images[split_name] = [
                    file_name
                    for file_name in filenames
                    if not (image_roots[split_name] / file_name).exists()
                ]
                coco_module = require_python_package("pycocotools.coco", "pycocotools")
                coco_module.COCO(str(annotation_path))
                report["pycocotools_load"] = True
            except Exception as exc:
                report["error"] = str(exc)
        json_reports[split_name] = report

    corrupt_images: dict[str, list[dict[str, str]]] = {}
    for split_name, image_root in image_roots.items():
        image_paths = sorted(image_root.glob("*.jpg"))
        corrupt: list[dict[str, str]] = []
        LOGGER.info("Validating %d %s images for readability", len(image_paths), split_name)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for image_path, ok, error in executor.map(check_image_readable, image_paths):
                if not ok:
                    corrupt.append({"path": image_path, "error": error or "unreadable"})
        corrupt_images[split_name] = corrupt

    errors: list[str] = []
    for name, report in archives.items():
        if not report["exists"] or report["actual_size_bytes"] <= 0 or report["zip_integrity"] != "passed":
            errors.append(f"Archive validation failed: {name}")
    for split_name, report in json_reports.items():
        if not report["valid_json"] or not report["pycocotools_load"]:
            errors.append(f"Annotation validation failed: {split_name}")
        if missing_images.get(split_name):
            errors.append(f"Missing referenced images in {split_name}: {len(missing_images[split_name])}")
        if corrupt_images.get(split_name):
            errors.append(f"Corrupt images in {split_name}: {len(corrupt_images[split_name])}")

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "archives": archives,
        "extraction_root": str(paths.coco_root),
        "image_counts": image_counts,
        "annotation_files": json_reports,
        "missing_images": {key: len(value) for key, value in missing_images.items()},
        "missing_image_examples": {key: value[:20] for key, value in missing_images.items()},
        "duplicate_file_names": duplicate_file_names,
        "corrupt_images": {key: len(value) for key, value in corrupt_images.items()},
        "corrupt_image_examples": {key: value[:20] for key, value in corrupt_images.items()},
        "valid": not errors,
        "errors": errors,
    }
    write_json(paths.artifacts_root / "coco_extraction_validation.json", report)
    lines = [
        "# COCO Extraction Validation",
        "",
        f"Generated UTC: `{report['generated_at_utc']}`",
        f"Status: {'PASS' if report['valid'] else 'FAIL'}",
        "",
        "## Archives",
    ]
    for name, archive in archives.items():
        lines.append(
            f"- {name}: exists={archive['exists']}, size={archive['actual_size_bytes']}, "
            f"zip={archive['zip_integrity']}"
        )
    lines.extend(["", "## Extracted Images"])
    for split_name, count in image_counts.items():
        lines.append(f"- {split_name}: {count}")
    lines.extend(["", "## Annotation JSON"])
    for split_name, item in json_reports.items():
        lines.append(
            f"- {split_name}: json={item['valid_json']}, pycocotools={item['pycocotools_load']}, "
            f"images={item.get('images')}, annotations={item.get('annotations')}, categories={item.get('categories')}"
        )
    if errors:
        lines.extend(["", "## Errors"])
        lines.extend(f"- {error}" for error in errors)
    (paths.artifacts_root / "coco_extraction_validation.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    if errors:
        raise RuntimeError("Raw COCO extraction validation failed.")
    return report


def clip_coco_bbox(
    bbox: list[float],
    image_width: int,
    image_height: int,
) -> tuple[list[float], tuple[float, float, float, float]]:
    """Clip a COCO bbox and return clipped COCO + normalized YOLO boxes."""
    x_min, y_min, width, height = (float(value) for value in bbox)
    x1 = max(0.0, x_min)
    y1 = max(0.0, y_min)
    x2 = min(float(image_width), x_min + width)
    y2 = min(float(image_height), y_min + height)
    if x2 <= x1 or y2 <= y1:
        raise ValueError("bbox has zero or negative area after clipping")
    clipped = [x1, y1, x2 - x1, y2 - y1]
    return clipped, coco_bbox_to_yolo(clipped, image_width, image_height)


def parse_source_split(
    annotation_path: Path,
    image_root: Path,
    source_split: str,
    category_mapping: dict[int, dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[int, dict[str, Any]], dict[str, Any]]:
    """Parse one COCO source split and preserve accepted/rejected annotations."""
    data = load_json(annotation_path)
    mapping = category_mapping or build_category_mapping(data.get("categories", []))
    valid_category_ids = set(mapping)
    images = data.get("images", [])
    image_index = {int(image["id"]): image for image in images}
    annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    annotation_ids: set[int] = set()
    duplicate_annotation_ids = 0
    for annotation in data.get("annotations", []):
        annotation_id = int(annotation.get("id", -1))
        if annotation_id in annotation_ids:
            duplicate_annotation_ids += 1
        annotation_ids.add(annotation_id)
        annotations_by_image[int(annotation.get("image_id", -1))].append(annotation)

    image_records: list[dict[str, Any]] = []
    annotation_records: list[dict[str, Any]] = []
    rejected_reasons: Counter[str] = Counter()
    accepted_count = 0
    crowd_count = 0
    for image in images:
        image_id = int(image["id"])
        width = int(image.get("width", 0))
        height = int(image.get("height", 0))
        file_name = str(image["file_name"])
        internal_id = f"{source_split}_{image_id}"
        image_path = image_root / file_name
        accepted_objects: list[dict[str, Any]] = []
        has_crowd = False
        image_usable = image_path.exists() and width > 0 and height > 0
        for annotation in annotations_by_image.get(image_id, []):
            annotation_id = int(annotation.get("id", -1))
            category_id = int(annotation.get("category_id", -1))
            iscrowd = int(annotation.get("iscrowd", 0))
            status = "accepted"
            reason = ""
            clipped_bbox: list[float] | None = None
            yolo_bbox: tuple[float, float, float, float] | None = None
            if iscrowd:
                has_crowd = True
                crowd_count += 1
                status = "excluded"
                reason = "iscrowd_excluded_for_yolov5_compatibility"
            if status == "accepted" and not image_usable:
                status = "rejected"
                reason = "missing_image_or_invalid_dimensions"
            if status == "accepted" and image_id not in image_index:
                status = "rejected"
                reason = "image_id_not_found"
            if status == "accepted" and category_id not in valid_category_ids:
                status = "rejected"
                reason = "invalid_category_id"
            if status == "accepted" and "bbox" not in annotation:
                status = "rejected"
                reason = "missing_bbox"
            if status == "accepted":
                try:
                    clipped_bbox, yolo_bbox = clip_coco_bbox(annotation["bbox"], width, height)
                except Exception as exc:
                    status = "rejected"
                    reason = f"invalid_bbox:{exc}"
            mapping_item = mapping.get(category_id)
            record = {
                "internal_record_id": f"{internal_id}_{annotation_id}",
                "internal_image_id": internal_id,
                "source_split": source_split,
                "annotation_id": annotation_id,
                "image_id": image_id,
                "file_name": file_name,
                "category_id": category_id,
                "category_name": mapping_item["name"] if mapping_item else None,
                "class_id": mapping_item["class_id"] if mapping_item else None,
                "bbox_coco_original": annotation.get("bbox"),
                "bbox_coco_clipped": clipped_bbox,
                "bbox_yolo": list(yolo_bbox) if yolo_bbox else None,
                "area": annotation.get("area"),
                "iscrowd": iscrowd,
                "status": status,
                "reason": reason,
            }
            annotation_records.append(record)
            if status == "accepted":
                accepted_count += 1
                accepted_objects.append(record)
            elif reason:
                rejected_reasons[reason] += 1
        image_records.append(
            {
                "internal_record_id": internal_id,
                "image_id": image_id,
                "source_split": source_split,
                "file_name": file_name,
                "image_path": str(image_path),
                "width": width,
                "height": height,
                "valid_object_count": len(accepted_objects),
                "category_ids": sorted({int(item["category_id"]) for item in accepted_objects}),
                "category_names": sorted({str(item["category_name"]) for item in accepted_objects}),
                "has_crowd_annotations": has_crowd,
                "usable_for_training": image_usable,
            }
        )

    stats = {
        "source_split": source_split,
        "images": len(images),
        "annotations": len(data.get("annotations", [])),
        "accepted_annotations": accepted_count,
        "crowd_annotations": crowd_count,
        "duplicate_annotation_ids": duplicate_annotation_ids,
        "rejected_reasons": dict(rejected_reasons),
    }
    return image_records, annotation_records, mapping, stats


def write_combined_manifests(
    paths: PreparedPaths,
    image_records: list[dict[str, Any]],
    annotation_records: list[dict[str, Any]],
) -> None:
    """Write combined image and annotation manifests."""
    ensure_dir(paths.interim_root)
    image_csv = paths.interim_root / "coco_combined_image_manifest.csv"
    with image_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "internal_record_id",
                "image_id",
                "source_split",
                "file_name",
                "image_path",
                "width",
                "height",
                "valid_object_count",
                "category_ids",
                "category_names",
                "has_crowd_annotations",
                "usable_for_training",
            ],
        )
        writer.writeheader()
        for record in image_records:
            writer.writerow({**record, "category_ids": json.dumps(record["category_ids"]), "category_names": json.dumps(record["category_names"])})

    annotation_manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "iscrowd_handling": "Crowd annotations are preserved in this manifest but excluded from YOLO label files for YOLOv5 object-detection compatibility.",
        "annotations": annotation_records,
    }
    write_json(paths.interim_root / "coco_combined_annotation_manifest.json", annotation_manifest, indent=None)


def deterministic_image_split(
    image_records: list[dict[str, Any]],
    seed: int = 42,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> dict[str, list[dict[str, Any]]]:
    """Create a deterministic image-level 80/10/10 split."""
    usable = [record for record in image_records if record["usable_for_training"]]
    shuffled = list(usable)
    random.Random(seed).shuffle(shuffled)
    total = len(shuffled)
    train_count = int(total * train_ratio)
    val_count = int(total * val_ratio)
    return {
        "train": shuffled[:train_count],
        "val": shuffled[train_count : train_count + val_count],
        "test": shuffled[train_count + val_count :],
    }


def choose_link_strategy(paths: PreparedPaths) -> str:
    """Choose a Windows-compatible image strategy without duplicating data."""
    test_dir = ensure_dir(paths.processed_root / ".link_test")
    source = test_dir / "source.tmp"
    hardlink = test_dir / "hardlink.tmp"
    symlink = test_dir / "symlink.tmp"
    source.write_text("x", encoding="utf-8")
    try:
        if hardlink.exists():
            hardlink.unlink()
        os.link(source, hardlink)
        return "hardlink"
    except OSError:
        pass
    try:
        if symlink.exists():
            symlink.unlink()
        symlink.symlink_to(source)
        return "symlink"
    except OSError:
        pass
    finally:
        for path in (hardlink, symlink, source):
            try:
                if path.exists() or path.is_symlink():
                    path.unlink()
            except OSError:
                pass
    raise RuntimeError(
        "Neither hard links nor symbolic links are available. Refusing to duplicate the full COCO dataset automatically."
    )


def link_image(source: Path, destination: Path, strategy: str) -> None:
    """Link one image into the processed YOLO image tree."""
    ensure_dir(destination.parent)
    if destination.exists():
        return
    if strategy == "hardlink":
        os.link(source, destination)
    elif strategy == "symlink":
        destination.symlink_to(source)
    else:
        raise ValueError(f"Unsupported link strategy: {strategy}")


def write_dataset_yaml(path: Path, dataset_root: Path, class_names: list[str]) -> None:
    """Write a YOLOv5-compatible dataset YAML."""
    yaml_text = "\n".join(
        [
            f"path: {dataset_root.resolve().as_posix()}",
            "train: images/train",
            "val: images/val",
            "test: images/test",
            "nc: 80",
            "names:",
            *[f"  {index}: {name}" for index, name in enumerate(class_names)],
            "",
        ]
    )
    ensure_dir(path.parent)
    path.write_text(yaml_text, encoding="utf-8")


def write_colab_template(path: Path, class_names: list[str]) -> None:
    """Write a portable Colab dataset YAML template."""
    yaml_text = "\n".join(
        [
            "# Update path to the mounted Google Drive location before training in Colab.",
            "path: /content/drive/MyDrive/yolov5_coco_project/coco_yolo",
            "train: images/train",
            "val: images/val",
            "test: images/test",
            "nc: 80",
            "names:",
            *[f"  {index}: {name}" for index, name in enumerate(class_names)],
            "",
        ]
    )
    ensure_dir(path.parent)
    path.write_text(yaml_text, encoding="utf-8")


def create_yolo_layout(
    paths: PreparedPaths,
    splits: dict[str, list[dict[str, Any]]],
    annotation_records: list[dict[str, Any]],
    category_mapping: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    """Create linked image layout, YOLO labels, split manifests, and dataset YAML."""
    strategy = choose_link_strategy(paths)
    LOGGER.info("Selected image storage strategy: %s", strategy)
    accepted_by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rejected_annotations = [record for record in annotation_records if record["status"] != "accepted"]
    for record in annotation_records:
        if record["status"] == "accepted":
            accepted_by_image[str(record["internal_image_id"])].append(record)

    class_names = [item["name"] for _, item in sorted(category_mapping.items(), key=lambda pair: pair[1]["class_id"])]
    ensure_dir(paths.processed_root)
    write_json(paths.processed_root / "class_mapping.json", category_mapping)
    (paths.processed_root / "class_names.txt").write_text("\n".join(class_names) + "\n", encoding="utf-8")
    split_summary: dict[str, Any] = {
        "seed": 42,
        "strategy": "deterministic_random_image_level_split",
        "category_stratification": "Random image-level split; per-class distributions are reported rather than manipulated.",
        "image_storage_strategy": strategy,
        "counts": {},
        "percentages": {},
        "annotation_counts": {},
        "rejected_annotations": len(rejected_annotations),
    }

    total_images = sum(len(records) for records in splits.values())
    for split_name, records in splits.items():
        split_image_dir = paths.processed_root / "images" / split_name
        split_label_dir = paths.processed_root / "labels" / split_name
        ensure_dir(split_image_dir)
        ensure_dir(split_label_dir)
        image_txt = paths.split_root / f"{split_name}_images.txt"
        manifest_csv = paths.split_root / f"{split_name}_manifest.csv"
        image_lines: list[str] = []
        annotations_in_split = 0
        with manifest_csv.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(
                csv_file,
                fieldnames=[
                    "internal_record_id",
                    "source_split",
                    "image_id",
                    "source_image_path",
                    "project_image_path",
                    "project_label_path",
                    "width",
                    "height",
                    "valid_object_count",
                    "category_names",
                ],
            )
            writer.writeheader()
            for record in records:
                source_image = Path(record["image_path"])
                output_stem = f"{record['source_split']}_{source_image.stem}"
                destination_image = split_image_dir / f"{output_stem}{source_image.suffix.lower()}"
                destination_label = split_label_dir / f"{output_stem}.txt"
                link_image(source_image, destination_image, strategy)
                accepted = accepted_by_image.get(str(record["internal_record_id"]), [])
                label_lines = [
                    f"{int(item['class_id'])} "
                    f"{item['bbox_yolo'][0]:.8f} {item['bbox_yolo'][1]:.8f} "
                    f"{item['bbox_yolo'][2]:.8f} {item['bbox_yolo'][3]:.8f}"
                    for item in accepted
                ]
                destination_label.write_text("\n".join(label_lines) + ("\n" if label_lines else ""), encoding="utf-8")
                annotations_in_split += len(label_lines)
                image_lines.append(str(destination_image.resolve()))
                writer.writerow(
                    {
                        "internal_record_id": record["internal_record_id"],
                        "source_split": record["source_split"],
                        "image_id": record["image_id"],
                        "source_image_path": record["image_path"],
                        "project_image_path": str(destination_image.resolve()),
                        "project_label_path": str(destination_label.resolve()),
                        "width": record["width"],
                        "height": record["height"],
                        "valid_object_count": record["valid_object_count"],
                        "category_names": json.dumps(record["category_names"]),
                    }
                )
        image_txt.write_text("\n".join(image_lines) + "\n", encoding="utf-8")
        split_summary["counts"][split_name] = len(records)
        split_summary["percentages"][split_name] = round(len(records) / total_images * 100, 4) if total_images else 0
        split_summary["annotation_counts"][split_name] = annotations_in_split

    write_json(paths.split_root / "split_summary.json", split_summary)
    write_dataset_yaml(paths.processed_root / "coco_project.yaml", paths.processed_root, class_names)
    write_colab_template(paths.root / "configs" / "coco_project_colab.yaml", class_names)
    return split_summary


def load_split_manifest(path: Path) -> list[dict[str, str]]:
    """Load a split CSV manifest."""
    with path.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def summarize_distribution(values: list[float]) -> dict[str, float | int | None]:
    """Summarize numeric values."""
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None, "median": None}
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
    }


def generate_statistics_and_charts(
    paths: PreparedPaths,
    image_records: list[dict[str, Any]],
    annotation_records: list[dict[str, Any]],
    splits: dict[str, list[dict[str, Any]]],
    category_mapping: dict[int, dict[str, Any]],
    extraction_report: dict[str, Any],
) -> dict[str, Any]:
    """Generate dataset statistics and matplotlib charts."""
    matplotlib = require_python_package("matplotlib")
    matplotlib.use("Agg")
    pyplot = require_python_package("matplotlib.pyplot", "matplotlib")

    accepted = [record for record in annotation_records if record["status"] == "accepted"]
    excluded = [record for record in annotation_records if record["status"] == "excluded"]
    rejected = [record for record in annotation_records if record["status"] == "rejected"]
    class_object_counts: Counter[str] = Counter(str(record["category_name"]) for record in accepted)
    class_image_counts: Counter[str] = Counter()
    for image in image_records:
        for name in image["category_names"]:
            class_image_counts[str(name)] += 1
    widths = [float(record["bbox_yolo"][2]) for record in accepted]
    heights = [float(record["bbox_yolo"][3]) for record in accepted]
    areas = [float(record["bbox_coco_clipped"][2]) * float(record["bbox_coco_clipped"][3]) for record in accepted]
    size_buckets = Counter(
        "small" if area < 32**2 else "medium" if area < 96**2 else "large"
        for area in areas
    )
    split_class_counts: dict[str, Counter[str]] = {}
    split_annotation_counts: dict[str, int] = {}
    accepted_by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in accepted:
        accepted_by_image[str(record["internal_image_id"])].append(record)
    for split_name, records in splits.items():
        counter: Counter[str] = Counter()
        total = 0
        for image in records:
            for annotation in accepted_by_image.get(str(image["internal_record_id"]), []):
                counter[str(annotation["category_name"])] += 1
                total += 1
        split_class_counts[split_name] = counter
        split_annotation_counts[split_name] = total

    rejected_reasons = Counter(str(record["reason"]) for record in rejected + excluded)
    class_names = [item["name"] for _, item in sorted(category_mapping.items(), key=lambda pair: pair[1]["class_id"])]
    stats = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "total_source_images": len(image_records),
        "total_usable_images": sum(1 for record in image_records if record["usable_for_training"]),
        "total_annotations": len(annotation_records),
        "accepted_annotations": len(accepted),
        "rejected_annotations": len(rejected),
        "excluded_crowd_annotations": len(excluded),
        "crowd_annotations": len(excluded),
        "images_with_zero_accepted_objects": sum(1 for record in image_records if int(record["valid_object_count"]) == 0),
        "missing_images": extraction_report["missing_images"],
        "corrupt_or_unreadable_images": extraction_report["corrupt_images"],
        "invalid_boxes": sum(1 for record in rejected if str(record["reason"]).startswith("invalid_bbox")),
        "rejected_reasons": dict(rejected_reasons),
        "objects_per_image": summarize_distribution([int(record["valid_object_count"]) for record in image_records]),
        "class_wise_object_counts": {name: class_object_counts.get(name, 0) for name in class_names},
        "class_wise_image_counts": {name: class_image_counts.get(name, 0) for name in class_names},
        "bbox_width_distribution": summarize_distribution(widths),
        "bbox_height_distribution": summarize_distribution(heights),
        "bbox_area_distribution_pixels": summarize_distribution(areas),
        "object_size_counts": dict(size_buckets),
        "split_image_counts": {split_name: len(records) for split_name, records in splits.items()},
        "split_annotation_counts": split_annotation_counts,
        "split_class_distribution": {
            split_name: {name: counter.get(name, 0) for name in class_names}
            for split_name, counter in split_class_counts.items()
        },
    }
    write_json(paths.artifacts_root / "coco_dataset_statistics.json", stats)

    with (paths.artifacts_root / "coco_dataset_statistics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["class_id", "category_id", "category_name", "object_count", "image_count", "train", "val", "test"])
        writer.writeheader()
        for category_id, item in sorted(category_mapping.items(), key=lambda pair: pair[1]["class_id"]):
            name = item["name"]
            writer.writerow(
                {
                    "class_id": item["class_id"],
                    "category_id": category_id,
                    "category_name": name,
                    "object_count": class_object_counts.get(name, 0),
                    "image_count": class_image_counts.get(name, 0),
                    "train": split_class_counts["train"].get(name, 0),
                    "val": split_class_counts["val"].get(name, 0),
                    "test": split_class_counts["test"].get(name, 0),
                }
            )

    ensure_dir(paths.results_root)
    pyplot.figure(figsize=(18, 6))
    pyplot.bar(class_names, [class_object_counts.get(name, 0) for name in class_names])
    pyplot.xticks(rotation=90, fontsize=7)
    pyplot.ylabel("Objects")
    pyplot.tight_layout()
    pyplot.savefig(paths.results_root / "class_wise_object_distribution.png", dpi=150)
    pyplot.close()

    pyplot.figure(figsize=(6, 4))
    pyplot.bar(list(stats["split_image_counts"].keys()), list(stats["split_image_counts"].values()))
    pyplot.ylabel("Images")
    pyplot.tight_layout()
    pyplot.savefig(paths.results_root / "split_image_counts.png", dpi=150)
    pyplot.close()

    pyplot.figure(figsize=(6, 4))
    pyplot.bar(list(split_annotation_counts.keys()), list(split_annotation_counts.values()))
    pyplot.ylabel("Annotations")
    pyplot.tight_layout()
    pyplot.savefig(paths.results_root / "split_annotation_counts.png", dpi=150)
    pyplot.close()

    pyplot.figure(figsize=(8, 5))
    pyplot.hist([int(record["valid_object_count"]) for record in image_records], bins=50)
    pyplot.xlabel("Accepted objects per image")
    pyplot.ylabel("Images")
    pyplot.tight_layout()
    pyplot.savefig(paths.results_root / "objects_per_image_distribution.png", dpi=150)
    pyplot.close()

    pyplot.figure(figsize=(8, 5))
    pyplot.hist(areas, bins=80)
    pyplot.xlabel("Bounding-box area in pixels")
    pyplot.ylabel("Objects")
    pyplot.tight_layout()
    pyplot.savefig(paths.results_root / "bbox_area_distribution.png", dpi=150)
    pyplot.close()

    pyplot.figure(figsize=(18, 7))
    x = list(range(len(class_names)))
    bottoms = [0] * len(class_names)
    for split_name in SPLITS:
        values = [split_class_counts[split_name].get(name, 0) for name in class_names]
        pyplot.bar(x, values, bottom=bottoms, label=split_name)
        bottoms = [left + right for left, right in zip(bottoms, values, strict=True)]
    pyplot.xticks(x, class_names, rotation=90, fontsize=7)
    pyplot.ylabel("Objects")
    pyplot.legend()
    pyplot.tight_layout()
    pyplot.savefig(paths.results_root / "split_wise_class_distribution.png", dpi=150)
    pyplot.close()

    lines = [
        "# COCO Dataset Statistics",
        "",
        f"Generated UTC: `{stats['generated_at_utc']}`",
        f"- Total source images: {stats['total_source_images']}",
        f"- Total usable images: {stats['total_usable_images']}",
        f"- Total annotations: {stats['total_annotations']}",
        f"- Accepted annotations: {stats['accepted_annotations']}",
        f"- Rejected annotations: {stats['rejected_annotations']}",
        f"- Excluded crowd annotations: {stats['excluded_crowd_annotations']}",
        f"- Images with zero accepted objects: {stats['images_with_zero_accepted_objects']}",
        "",
        "## Split Counts",
    ]
    for split_name in SPLITS:
        lines.append(
            f"- {split_name}: {stats['split_image_counts'][split_name]} images, "
            f"{stats['split_annotation_counts'][split_name]} annotations"
        )
    lines.extend(["", "## Rejected / Excluded Reasons"])
    lines.extend(f"- {reason}: {count}" for reason, count in sorted(rejected_reasons.items()))
    (paths.artifacts_root / "coco_dataset_statistics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return stats


def draw_labels_on_image(image_path: Path, label_path: Path, output_path: Path, class_names: list[str], title: str | None = None) -> None:
    """Draw YOLO labels on an image."""
    Image = require_python_package("PIL.Image", "Pillow")
    ImageDraw = require_python_package("PIL.ImageDraw", "Pillow")
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size
    for line in label_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        class_id_text, x_text, y_text, w_text, h_text = line.split()
        class_id = int(class_id_text)
        x_center, y_center, box_width, box_height = map(float, [x_text, y_text, w_text, h_text])
        x1 = (x_center - box_width / 2) * width
        y1 = (y_center - box_height / 2) * height
        x2 = (x_center + box_width / 2) * width
        y2 = (y_center + box_height / 2) * height
        label = class_names[class_id]
        draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=3)
        draw.text((x1, max(0, y1 - 14)), label, fill=(255, 255, 0))
    if title:
        draw.text((8, 8), title, fill=(255, 255, 255))
    ensure_dir(output_path.parent)
    image.save(output_path)


def letterbox_transform(
    image_width: int,
    image_height: int,
    boxes_xyxy: list[tuple[float, float, float, float]],
    new_size: int = 640,
) -> tuple[float, int, int, list[tuple[float, float, float, float]]]:
    """Return YOLOv5-style letterbox scale, padding, and transformed boxes."""
    scale = min(new_size / image_width, new_size / image_height)
    resized_width = int(round(image_width * scale))
    resized_height = int(round(image_height * scale))
    pad_x = (new_size - resized_width) // 2
    pad_y = (new_size - resized_height) // 2
    transformed = [
        (x1 * scale + pad_x, y1 * scale + pad_y, x2 * scale + pad_x, y2 * scale + pad_y)
        for x1, y1, x2, y2 in boxes_xyxy
    ]
    return scale, pad_x, pad_y, transformed


def create_letterbox_visualization(image_path: Path, label_path: Path, output_path: Path, class_names: list[str], new_size: int = 640) -> None:
    """Save a composite original/letterboxed visualization with transformed boxes."""
    Image = require_python_package("PIL.Image", "Pillow")
    ImageDraw = require_python_package("PIL.ImageDraw", "Pillow")
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    boxes: list[tuple[int, tuple[float, float, float, float]]] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        class_id_text, x_text, y_text, w_text, h_text = line.split()
        class_id = int(class_id_text)
        x_center, y_center, box_width, box_height = map(float, [x_text, y_text, w_text, h_text])
        boxes.append(
            (
                class_id,
                (
                    (x_center - box_width / 2) * width,
                    (y_center - box_height / 2) * height,
                    (x_center + box_width / 2) * width,
                    (y_center + box_height / 2) * height,
                ),
            )
        )
    _, pad_x, pad_y, transformed = letterbox_transform(width, height, [box for _, box in boxes], new_size)
    scale = min(new_size / width, new_size / height)
    resized = image.resize((int(round(width * scale)), int(round(height * scale))))
    letterboxed = Image.new("RGB", (new_size, new_size), (114, 114, 114))
    letterboxed.paste(resized, (pad_x, pad_y))
    original_preview = image.copy()
    original_preview.thumbnail((new_size, new_size))
    original_canvas = Image.new("RGB", (new_size, new_size), (32, 32, 32))
    original_canvas.paste(original_preview, ((new_size - original_preview.width) // 2, (new_size - original_preview.height) // 2))
    original_draw = ImageDraw.Draw(original_canvas)
    letterbox_draw = ImageDraw.Draw(letterboxed)
    original_draw.text((8, 8), "original", fill=(255, 255, 255))
    letterbox_draw.text((8, 8), "letterbox 640x640", fill=(255, 255, 255))
    preview_scale = min(new_size / width, new_size / height)
    offset_x = (new_size - int(round(width * preview_scale))) // 2
    offset_y = (new_size - int(round(height * preview_scale))) // 2
    for (class_id, box), transformed_box in zip(boxes, transformed, strict=True):
        label = class_names[class_id]
        ox1, oy1, ox2, oy2 = box
        original_draw.rectangle(
            [ox1 * preview_scale + offset_x, oy1 * preview_scale + offset_y, ox2 * preview_scale + offset_x, oy2 * preview_scale + offset_y],
            outline=(0, 255, 0),
            width=3,
        )
        original_draw.text((ox1 * preview_scale + offset_x, max(0, oy1 * preview_scale + offset_y - 14)), label, fill=(255, 255, 0))
        letterbox_draw.rectangle(transformed_box, outline=(0, 255, 0), width=3)
        letterbox_draw.text((transformed_box[0], max(0, transformed_box[1] - 14)), label, fill=(255, 255, 0))
    composite = Image.new("RGB", (new_size * 2, new_size), (0, 0, 0))
    composite.paste(original_canvas, (0, 0))
    composite.paste(letterboxed, (new_size, 0))
    ensure_dir(output_path.parent)
    composite.save(output_path)


def create_visualizations(paths: PreparedPaths, class_names: list[str], seed: int = 42) -> dict[str, Any]:
    """Create deterministic dataset validation and letterbox examples."""
    rng = random.Random(seed)
    outputs: dict[str, list[str]] = {}
    sample_counts = {"train": 5, "val": 3, "test": 3}
    for split_name, count in sample_counts.items():
        records = load_split_manifest(paths.split_root / f"{split_name}_manifest.csv")
        rng.shuffle(records)
        outputs[split_name] = []
        for record in records[:count]:
            image_path = Path(record["project_image_path"])
            label_path = Path(record["project_label_path"])
            output_path = paths.validation_output_root / split_name / f"{image_path.stem}.jpg"
            draw_labels_on_image(
                image_path,
                label_path,
                output_path,
                class_names,
                title=f"{split_name} objects={record['valid_object_count']}",
            )
            outputs[split_name].append(str(output_path))

    letterbox_outputs: list[str] = []
    train_records = load_split_manifest(paths.split_root / "train_manifest.csv")
    rng.shuffle(train_records)
    for record in train_records[:3]:
        image_path = Path(record["project_image_path"])
        label_path = Path(record["project_label_path"])
        output_path = paths.preprocessing_output_root / f"letterbox_{image_path.stem}.jpg"
        create_letterbox_visualization(image_path, label_path, output_path, class_names)
        letterbox_outputs.append(str(output_path))
    return {"validation_samples": outputs, "letterbox_examples": letterbox_outputs}


def validate_dataset_yaml(yaml_path: Path) -> dict[str, Any]:
    """Validate a YOLOv5 dataset YAML file."""
    yaml = require_python_package("yaml", "PyYAML")
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    dataset_root = Path(str(data["path"]))
    names = data["names"]
    if isinstance(names, dict):
        name_count = len(names)
    else:
        name_count = len(list(names))
    paths_ok = {
        split: (dataset_root / str(data[split])).exists()
        for split in SPLITS
    }
    return {
        "path": str(yaml_path),
        "dataset_root": str(dataset_root),
        "name_count": name_count,
        "nc": int(data["nc"]),
        "paths_ok": paths_ok,
        "valid": name_count == 80 and int(data["nc"]) == 80 and all(paths_ok.values()),
    }


def validate_processed_dataset(
    paths: PreparedPaths,
    splits: dict[str, list[dict[str, Any]]],
    annotation_records: list[dict[str, Any]],
    extraction_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run automated validation rules for the processed dataset."""
    accepted_by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in annotation_records:
        if record["status"] == "accepted":
            accepted_by_image[str(record["internal_image_id"])].append(record)

    rules: list[dict[str, Any]] = []

    def add_rule(name: str, passed: bool, details: dict[str, Any] | None = None) -> None:
        rules.append({"name": name, "status": "PASS" if passed else "FAIL", "details": details or {}})

    split_keys: dict[str, set[str]] = {}
    split_paths: dict[str, set[str]] = {}
    split_label_paths: dict[str, set[str]] = {}
    label_errors: list[str] = []
    image_open_errors: list[str] = []
    reversible_failures = 0
    manifest_rows_total = 0
    for split_name in SPLITS:
        rows = load_split_manifest(paths.split_root / f"{split_name}_manifest.csv")
        manifest_rows_total += len(rows)
        split_keys[split_name] = {f"{row['source_split']}:{row['image_id']}" for row in rows}
        split_paths[split_name] = {str(Path(row["project_image_path"]).resolve()).lower() for row in rows}
        split_label_paths[split_name] = {str(Path(row["project_label_path"]).resolve()).lower() for row in rows}
        for row in rows:
            image_path = Path(row["project_image_path"])
            label_path = Path(row["project_label_path"])
            if not image_path.exists():
                label_errors.append(f"missing image {image_path}")
            if not label_path.exists():
                label_errors.append(f"missing label {label_path}")
                continue
            if extraction_report is None:
                image_check = check_image_readable(image_path)
                if not image_check[1]:
                    image_open_errors.append(f"{image_path}: {image_check[2]}")
            lines = label_path.read_text(encoding="utf-8").splitlines()
            accepted = accepted_by_image.get(row["internal_record_id"], [])
            if len(lines) != len(accepted):
                reversible_failures += 1
            for index, line in enumerate(lines):
                parts = line.split()
                if len(parts) != 5:
                    label_errors.append(f"{label_path}:{index + 1} expected 5 values")
                    continue
                try:
                    class_id = int(parts[0])
                    values = [float(value) for value in parts[1:]]
                except ValueError:
                    label_errors.append(f"{label_path}:{index + 1} non-numeric values")
                    continue
                if class_id < 0 or class_id > 79:
                    label_errors.append(f"{label_path}:{index + 1} class out of range")
                if not all(0 <= value <= 1 for value in values):
                    label_errors.append(f"{label_path}:{index + 1} normalized value out of range")
                if values[2] <= 0 or values[3] <= 0:
                    label_errors.append(f"{label_path}:{index + 1} non-positive width/height")
                if index < len(accepted):
                    expected = accepted[index]
                    expected_values = [float(value) for value in expected["bbox_yolo"]]
                    if class_id != int(expected["class_id"]) or any(abs(a - b) > 5e-7 for a, b in zip(values, expected_values, strict=True)):
                        reversible_failures += 1

    def pairwise_overlaps(items: dict[str, set[str]]) -> int:
        overlap_count = 0
        for left in SPLITS:
            for right in SPLITS:
                if left >= right:
                    continue
                overlap_count += len(items[left].intersection(items[right]))
        return overlap_count

    add_rule("no_image_source_key_leakage", pairwise_overlaps(split_keys) == 0, {"overlaps": pairwise_overlaps(split_keys)})
    add_rule("no_project_image_path_leakage", pairwise_overlaps(split_paths) == 0, {"overlaps": pairwise_overlaps(split_paths)})
    add_rule("no_label_path_leakage", pairwise_overlaps(split_label_paths) == 0, {"overlaps": pairwise_overlaps(split_label_paths)})
    add_rule("manifest_paths_exist", not any("missing image" in error or "missing label" in error for error in label_errors), {"errors": label_errors[:20]})
    add_rule("label_format_and_ranges", not label_errors, {"error_count": len(label_errors), "examples": label_errors[:20]})
    yaml_report = validate_dataset_yaml(paths.processed_root / "coco_project.yaml")
    add_rule("dataset_yaml_paths_resolve", yaml_report["valid"], yaml_report)
    if extraction_report is not None:
        corrupt_total = sum(int(value) for value in extraction_report.get("corrupt_images", {}).values())
        missing_total = sum(int(value) for value in extraction_report.get("missing_images", {}).values())
        add_rule(
            "all_usable_images_open",
            corrupt_total == 0 and missing_total == 0,
            {
                "method": "raw extraction validation opened every source image; processed images are hardlinks",
                "corrupt_images": extraction_report.get("corrupt_images", {}),
                "missing_images": extraction_report.get("missing_images", {}),
            },
        )
    else:
        add_rule("all_usable_images_open", not image_open_errors, {"error_count": len(image_open_errors), "examples": image_open_errors[:20]})
    add_rule("coco_to_yolo_reversible", reversible_failures == 0, {"failures": reversible_failures})
    total = sum(len(records) for records in splits.values())
    ratio_details = {split: len(records) / total if total else 0 for split, records in splits.items()}
    add_rule(
        "split_ratios_close_to_80_10_10",
        abs(ratio_details["train"] - 0.8) < 0.001 and abs(ratio_details["val"] - 0.1) < 0.001,
        ratio_details,
    )
    add_rule("source_traceability_present", manifest_rows_total == total, {"manifest_rows": manifest_rows_total, "split_total": total})
    placeholder_failures = []
    for path in [paths.processed_root / "coco_project.yaml", paths.root / "configs" / "coco_project_colab.yaml"]:
        text = path.read_text(encoding="utf-8")
        if "path\\to" in text.lower() or "placeholder" in text.lower():
            placeholder_failures.append(str(path))
    add_rule("no_placeholder_paths", not placeholder_failures, {"files": placeholder_failures})
    add_rule("all_labels_belong_to_single_split", pairwise_overlaps(split_label_paths) == 0)

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "rules": rules,
        "excluded_records": {
            "non_accepted_annotations": sum(1 for record in annotation_records if record["status"] != "accepted"),
            "reasons": dict(Counter(str(record["reason"]) for record in annotation_records if record["status"] != "accepted")),
        },
        "final_readiness": all(rule["status"] == "PASS" for rule in rules),
    }
    write_json(paths.artifacts_root / "coco_dataset_validation.json", report)
    lines = [
        "# COCO Dataset Validation",
        "",
        f"Generated UTC: `{report['generated_at_utc']}`",
        f"Final readiness: {'PASS' if report['final_readiness'] else 'FAIL'}",
        "",
        "## Rules",
    ]
    lines.extend(f"- {rule['name']}: {rule['status']} {rule['details']}" for rule in rules)
    lines.extend(["", "## Excluded Records"])
    lines.append(f"- Non-accepted annotations: {report['excluded_records']['non_accepted_annotations']}")
    for reason, count in report["excluded_records"]["reasons"].items():
        lines.append(f"- {reason}: {count}")
    (paths.artifacts_root / "coco_dataset_validation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def run_yolov5_dataset_loading_smoke(paths: PreparedPaths) -> dict[str, Any]:
    """Run a minimal YOLOv5 dataloader smoke check without training."""
    sys.path.insert(0, str(yolov5_root(paths.root)))
    yaml_report = validate_dataset_yaml(paths.processed_root / "coco_project.yaml")
    train_images = (paths.split_root / "train_images.txt").read_text(encoding="utf-8").splitlines()[:4]
    smoke_list = paths.artifacts_root / "coco_train_smoke_images.txt"
    smoke_list.write_text("\n".join(train_images) + "\n", encoding="utf-8")
    dataloaders = require_python_package("utils.dataloaders")
    dataloader, dataset = dataloaders.create_dataloader(
        path=str(smoke_list),
        imgsz=640,
        batch_size=2,
        stride=32,
        single_cls=False,
        pad=0.5,
        rect=False,
        workers=0,
        prefix="coco-smoke: ",
    )
    images, labels, paths_batch, _ = next(iter(dataloader))
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_yaml": yaml_report,
        "smoke_image_list": str(smoke_list),
        "batch_image_tensor_shape": list(images.shape),
        "batch_label_tensor_shape": list(labels.shape),
        "batch_paths": [str(path) for path in paths_batch],
        "dataset_length": len(dataset),
        "status": "passed",
    }
    write_json(paths.artifacts_root / "coco_dataset_loading_smoke.json", report)
    return report


def prepare_coco_dataset(root: Path, max_workers: int = 8) -> dict[str, Any]:
    """Run the complete COCO preprocessing pipeline after download/extraction."""
    started = time.perf_counter()
    paths = paths_for(root.resolve())
    for directory in [
        paths.interim_root,
        paths.processed_root,
        paths.split_root,
        paths.artifacts_root,
        paths.results_root,
        paths.validation_output_root,
        paths.preprocessing_output_root,
    ]:
        ensure_dir(directory)

    extraction_report = validate_raw_extraction(paths, max_workers=max_workers)
    train_ann = paths.coco_root / "annotations" / "instances_train2017.json"
    val_ann = paths.coco_root / "annotations" / "instances_val2017.json"
    train_images = paths.coco_root / "train2017"
    val_images = paths.coco_root / "val2017"
    train_records, train_annotations, mapping, train_stats = parse_source_split(train_ann, train_images, "train2017")
    val_records, val_annotations, mapping, val_stats = parse_source_split(val_ann, val_images, "val2017", mapping)
    image_records = train_records + val_records
    annotation_records = train_annotations + val_annotations
    write_combined_manifests(paths, image_records, annotation_records)
    splits = deterministic_image_split(image_records, seed=42)
    split_summary = create_yolo_layout(paths, splits, annotation_records, mapping)
    class_names = [item["name"] for _, item in sorted(mapping.items(), key=lambda pair: pair[1]["class_id"])]
    stats = generate_statistics_and_charts(paths, image_records, annotation_records, splits, mapping, extraction_report)
    visualization_report = create_visualizations(paths, class_names, seed=42)
    validation_report = validate_processed_dataset(paths, splits, annotation_records, extraction_report)
    loading_smoke_report = run_yolov5_dataset_loading_smoke(paths)
    duration = time.perf_counter() - started
    final_report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": round(duration, 2),
        "download_urls": {name: metadata["url"] for name, metadata in COCO_2017_ARCHIVES.items()},
        "source_stats": {"train2017": train_stats, "val2017": val_stats},
        "extraction": extraction_report,
        "split_summary": split_summary,
        "statistics": stats,
        "visualizations": visualization_report,
        "validation": validation_report,
        "dataset_loading_smoke": loading_smoke_report,
        "dataset_yaml": str(paths.processed_root / "coco_project.yaml"),
        "colab_yaml_template": str(paths.root / "configs" / "coco_project_colab.yaml"),
        "ready_for_training": bool(validation_report["final_readiness"] and loading_smoke_report["status"] == "passed"),
    }
    write_json(paths.artifacts_root / "coco_preprocessing_report.json", final_report)
    write_preprocessing_markdown(paths, final_report)
    return final_report


def load_splits_from_manifests(paths: PreparedPaths) -> dict[str, list[dict[str, Any]]]:
    """Load split records from existing CSV manifests."""
    splits: dict[str, list[dict[str, Any]]] = {}
    for split_name in SPLITS:
        records: list[dict[str, Any]] = []
        for row in load_split_manifest(paths.split_root / f"{split_name}_manifest.csv"):
            records.append(
                {
                    "internal_record_id": row["internal_record_id"],
                    "source_split": row["source_split"],
                    "image_id": int(row["image_id"]),
                    "image_path": row["source_image_path"],
                    "project_image_path": row["project_image_path"],
                    "project_label_path": row["project_label_path"],
                    "width": int(row["width"]),
                    "height": int(row["height"]),
                    "valid_object_count": int(row["valid_object_count"]),
                    "category_names": json.loads(row["category_names"]),
                    "usable_for_training": True,
                }
            )
        splits[split_name] = records
    return splits


def load_existing_annotation_records(paths: PreparedPaths) -> list[dict[str, Any]]:
    """Load annotation records from the combined annotation manifest."""
    manifest = load_json(paths.interim_root / "coco_combined_annotation_manifest.json")
    return list(manifest["annotations"])


def finalize_existing_coco_dataset(root: Path) -> dict[str, Any]:
    """Finalize validation/reporting from already generated COCO artifacts."""
    paths = paths_for(root.resolve())
    extraction_report = load_json(paths.artifacts_root / "coco_extraction_validation.json")
    split_summary = load_json(paths.split_root / "split_summary.json")
    stats = load_json(paths.artifacts_root / "coco_dataset_statistics.json")
    annotation_records = load_existing_annotation_records(paths)
    splits = load_splits_from_manifests(paths)
    validation_report = validate_processed_dataset(paths, splits, annotation_records, extraction_report)
    loading_smoke_report = run_yolov5_dataset_loading_smoke(paths)
    visualization_report = {
        "validation_samples_root": str(paths.validation_output_root),
        "letterbox_examples_root": str(paths.preprocessing_output_root),
    }
    final_report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": None,
        "download_urls": {name: metadata["url"] for name, metadata in COCO_2017_ARCHIVES.items()},
        "source_stats": {
            "train2017": stats.get("source_stats", {}).get("train2017"),
            "val2017": stats.get("source_stats", {}).get("val2017"),
        },
        "extraction": extraction_report,
        "split_summary": split_summary,
        "statistics": stats,
        "visualizations": visualization_report,
        "validation": validation_report,
        "dataset_loading_smoke": loading_smoke_report,
        "dataset_yaml": str(paths.processed_root / "coco_project.yaml"),
        "colab_yaml_template": str(paths.root / "configs" / "coco_project_colab.yaml"),
        "ready_for_training": bool(validation_report["final_readiness"] and loading_smoke_report["status"] == "passed"),
    }
    write_json(paths.artifacts_root / "coco_preprocessing_report.json", final_report)
    write_preprocessing_markdown(paths, final_report)
    return final_report


def write_preprocessing_markdown(paths: PreparedPaths, report: dict[str, Any]) -> None:
    """Write the final Markdown preprocessing report."""
    stats = report["statistics"]
    split_summary = report["split_summary"]
    archive_manifest_path = paths.artifacts_root / "coco_download_manifest.json"
    archive_manifest = load_json(archive_manifest_path) if archive_manifest_path.exists() else {"archives": {}}
    lines = [
        "# COCO Preprocessing Report",
        "",
        f"Generated UTC: `{report['generated_at_utc']}`",
        f"Processing duration: {report['duration_seconds']} seconds",
        "",
        "## Download URLs Used",
    ]
    lines.extend(f"- {name}: {url}" for name, url in report["download_urls"].items())
    lines.extend(["", "## Archive Sizes"])
    for name, item in archive_manifest.get("archives", {}).items():
        lines.append(
            f"- {name}: status={item.get('status')}, expected={item.get('expected_size_bytes')}, actual={item.get('actual_size_bytes')}, zip={item.get('zip_integrity')}"
        )
    lines.extend(
        [
            "",
            "## Extraction Results",
            f"- train2017 images: {report['extraction']['image_counts'].get('train2017')}",
            f"- val2017 images: {report['extraction']['image_counts'].get('val2017')}",
            f"- Extraction validation: {'PASS' if report['extraction']['valid'] else 'FAIL'}",
            "",
            "## Source Counts",
            f"- Total source images: {stats['total_source_images']}",
            f"- Total usable images: {stats['total_usable_images']}",
            f"- Total source annotations: {stats['total_annotations']}",
            f"- Accepted annotations: {stats['accepted_annotations']}",
            f"- Rejected annotations: {stats['rejected_annotations']}",
            f"- Crowd annotations excluded from YOLO labels: {stats['excluded_crowd_annotations']}",
            "",
            "## iscrowd Handling",
            "Crowd annotations are preserved in the combined annotation manifest but excluded from YOLO label files for YOLOv5 object-detection compatibility.",
            "",
            "## Split Strategy",
            "Deterministic random image-level split with seed 42. Exact multilabel stratification was not forced; per-class distributions are reported in the statistics artifacts.",
        ]
    )
    for split_name in SPLITS:
        lines.append(
            f"- {split_name}: {split_summary['counts'][split_name]} images ({split_summary['percentages'][split_name]}%), "
            f"{split_summary['annotation_counts'][split_name]} annotations"
        )
    lines.extend(
        [
            "",
            "## Class Mapping",
            "COCO category IDs are mapped to contiguous YOLO class indices 0-79 using sorted official COCO category IDs.",
            "",
            "## YOLO Conversion",
            "COCO `[x_min, y_min, width, height]` boxes are safely clipped to image bounds and converted to normalized YOLO `class x_center y_center width height` labels.",
            "",
            "## Image Storage Strategy",
            f"Selected strategy: `{split_summary['image_storage_strategy']}`. The full image dataset is not copied or permanently resized.",
            "",
            "## Letterbox Resizing",
            "YOLOv5 applies letterbox resizing during training and inference. Letterbox preserves aspect ratio, adds padding, and produces a 640 x 640 model input without permanently distorting source images.",
            "",
            "## Validation Results",
            f"- Dataset validation: {'PASS' if report['validation']['final_readiness'] else 'FAIL'}",
            f"- Dataset-loading smoke test: {report['dataset_loading_smoke']['status']}",
            "",
            "## Artifacts",
            "- `artifacts/coco_pre_download_check.json` / `.md`",
            "- `artifacts/coco_download_manifest.json`",
            "- `artifacts/coco_extraction_validation.json` / `.md`",
            "- `artifacts/coco_dataset_statistics.json` / `.csv` / `.md`",
            "- `artifacts/coco_dataset_validation.json` / `.md`",
            "- `artifacts/coco_dataset_loading_smoke.json`",
            "",
            "## Readiness",
            f"Ready for YOLOv5 training: {'YES' if report['ready_for_training'] else 'NO'}",
            "",
            "Next recommended action, not executed:",
            "",
            "```powershell",
            ".\\.venv\\Scripts\\python.exe -m src.train_models --config configs\\train_yolov5s.yaml --smoke-test --device cpu",
            "```",
        ]
    )
    (paths.artifacts_root / "coco_preprocessing_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    """CLI entrypoint for COCO preprocessing."""
    parser = argparse.ArgumentParser(description="Prepare COCO 2017 for YOLOv5.")
    parser.add_argument("--root", type=Path, default=project_root())
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--finalize-existing", action="store_true", help="Finalize from existing generated manifests.")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    setup_logging(args.log_level)
    if args.finalize_existing:
        report = finalize_existing_coco_dataset(args.root)
    else:
        report = prepare_coco_dataset(args.root, max_workers=args.max_workers)
    LOGGER.info("COCO preprocessing ready_for_training=%s", report["ready_for_training"])
    return 0 if report["ready_for_training"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
