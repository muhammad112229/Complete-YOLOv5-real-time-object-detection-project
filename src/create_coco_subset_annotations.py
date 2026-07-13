"""Create a COCO-format annotation JSON for the exact 2,500-image test subset."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.common import ensure_dir, project_root
from src.recreate_test_subset import label_path_for_image


DEFAULT_MANIFEST = Path("data") / "splits" / "test_subset_2500_seed42.txt"
DEFAULT_OUTPUT = (
    Path("data")
    / "processed"
    / "coco_yolo"
    / "annotations"
    / "instances_test_subset_2500_seed42.json"
)
DEFAULT_REPORT = Path("artifacts") / "test_subset_coco_annotation_verification.json"
DEFAULT_SOURCE_ANNOTATIONS = {
    "train2017": Path("data") / "raw" / "coco2017" / "annotations" / "instances_train2017.json",
    "val2017": Path("data") / "raw" / "coco2017" / "annotations" / "instances_val2017.json",
}
PROJECT_IMAGE_PATTERN = re.compile(r"^(train2017|val2017)_(\d{12})$")


def resolve_workspace_path(value: str | Path, root: Path | None = None) -> Path:
    """Resolve a workspace-relative path while preserving absolute paths."""
    base = root or project_root()
    path = Path(value)
    return path if path.is_absolute() else base / path


def relative_path(path: Path, root: Path | None = None) -> str:
    """Return a workspace-relative path when possible."""
    base = root or project_root()
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return str(path)


def sha256_file(path: Path) -> str:
    """Compute SHA256 for a file."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_subset_manifest(manifest_path: Path) -> list[Path]:
    """Load non-empty image paths from a text manifest in stored order."""
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Subset manifest not found: {manifest_path}")
    paths = [Path(line.strip()) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not paths:
        raise ValueError(f"Subset manifest contains no image paths: {manifest_path}")
    return paths


def parse_project_image_path(image_path: Path) -> tuple[str, int, str]:
    """Resolve a processed project image path to COCO source split, image ID, and file name."""
    match = PROJECT_IMAGE_PATTERN.match(image_path.stem)
    if not match:
        raise ValueError(
            f"Image name does not match expected '<train2017|val2017>_000000000000' pattern: {image_path.name}"
        )
    source_split = match.group(1)
    image_id = int(match.group(2))
    return source_split, image_id, f"{image_id:012d}{image_path.suffix.lower()}"


def clip_coco_bbox(bbox: list[Any], image_width: int, image_height: int) -> list[float]:
    """Clip a COCO bbox to image bounds using the same rule as preprocessing."""
    if len(bbox) != 4:
        raise ValueError("bbox must contain four values")
    x_min, y_min, width, height = (float(value) for value in bbox)
    x1 = max(0.0, x_min)
    y1 = max(0.0, y_min)
    x2 = min(float(image_width), x_min + width)
    y2 = min(float(image_height), y_min + height)
    if x2 <= x1 or y2 <= y1:
        raise ValueError("bbox has zero or negative area after clipping")
    return [round(x1, 3), round(y1, 3), round(x2 - x1, 3), round(y2 - y1, 3)]


def count_label_lines(label_path: Path) -> int:
    """Count non-empty YOLO label rows."""
    if not label_path.is_file():
        return 0
    return sum(1 for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip())


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object from disk."""
    if not path.is_file():
        raise FileNotFoundError(f"COCO annotation JSON not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"COCO annotation JSON root must be an object: {path}")
    return data


def validate_coco_subset_json(data: dict[str, Any]) -> list[str]:
    """Validate the generated subset JSON for COCOeval readiness."""
    errors: list[str] = []
    images = data.get("images", [])
    annotations = data.get("annotations", [])
    categories = data.get("categories", [])

    image_ids = [image.get("id") for image in images]
    annotation_ids = [annotation.get("id") for annotation in annotations]
    category_ids = {category.get("id") for category in categories}
    image_id_set = set(image_ids)

    if len(images) != 2500:
        errors.append(f"expected 2500 images, found {len(images)}")
    if len(image_ids) != len(image_id_set):
        errors.append("duplicate image IDs found")
    if len(categories) != 80:
        errors.append(f"expected 80 categories, found {len(categories)}")
    if len(annotation_ids) != len(set(annotation_ids)):
        errors.append("duplicate annotation IDs found")

    image_dimensions = {}
    for image in images:
        image_id = image.get("id")
        width = int(image.get("width", 0))
        height = int(image.get("height", 0))
        if width <= 0 or height <= 0:
            errors.append(f"image {image_id} has non-positive width or height")
        image_dimensions[image_id] = (width, height)

    for annotation in annotations:
        annotation_id = annotation.get("id")
        image_id = annotation.get("image_id")
        category_id = annotation.get("category_id")
        bbox = annotation.get("bbox", [])
        if annotation_id is None:
            errors.append("annotation with missing id found")
        if image_id not in image_id_set:
            errors.append(f"annotation {annotation_id} references missing image ID {image_id}")
        if category_id not in category_ids:
            errors.append(f"annotation {annotation_id} references missing category ID {category_id}")
        if not isinstance(bbox, list) or len(bbox) != 4:
            errors.append(f"annotation {annotation_id} has invalid bbox shape")
            continue
        x, y, width, height = (float(value) for value in bbox)
        if x < 0 or y < 0:
            errors.append(f"annotation {annotation_id} has negative bbox coordinate")
        if width <= 0 or height <= 0:
            errors.append(f"annotation {annotation_id} has non-positive bbox size")
        image_width, image_height = image_dimensions.get(image_id, (0, 0))
        if image_width and image_height and (x + width > image_width + 1e-6 or y + height > image_height + 1e-6):
            errors.append(f"annotation {annotation_id} bbox exceeds image bounds")

    return errors


def create_coco_subset_annotations(
    manifest_path: Path,
    output_json: Path,
    verification_json: Path,
    source_annotations: dict[str, Path] | None = None,
) -> dict[str, Any]:
    """Create and verify the COCO annotation JSON for the subset manifest."""
    root = project_root()
    manifest_path = resolve_workspace_path(manifest_path, root)
    output_json = resolve_workspace_path(output_json, root)
    verification_json = resolve_workspace_path(verification_json, root)
    source_annotations = {
        split: resolve_workspace_path(path, root)
        for split, path in (source_annotations or DEFAULT_SOURCE_ANNOTATIONS).items()
    }

    image_paths = load_subset_manifest(manifest_path)
    parsed_images: list[dict[str, Any]] = []
    selected_by_split: dict[str, set[int]] = defaultdict(set)
    missing_mapping_count = 0
    missing_label_count = 0
    label_line_count = 0

    for order, image_path in enumerate(image_paths):
        try:
            source_split, image_id, source_file_name = parse_project_image_path(image_path)
        except ValueError:
            missing_mapping_count += 1
            continue
        label_path = label_path_for_image(image_path)
        if not label_path.is_file():
            missing_label_count += 1
        label_line_count += count_label_lines(label_path)
        selected_by_split[source_split].add(image_id)
        parsed_images.append(
            {
                "order": order,
                "project_image_path": image_path,
                "source_split": source_split,
                "image_id": image_id,
                "source_file_name": source_file_name,
            }
        )

    image_id_counts = Counter(item["image_id"] for item in parsed_images)
    duplicate_image_id_count = sum(count - 1 for count in image_id_counts.values() if count > 1)

    source_data: dict[str, dict[str, Any]] = {}
    source_image_indexes: dict[str, dict[int, dict[str, Any]]] = {}
    source_annotations_by_image: dict[str, dict[int, list[dict[str, Any]]]] = {}
    categories: list[dict[str, Any]] | None = None
    source_files_used: list[str] = []

    for source_split in sorted(selected_by_split):
        annotation_path = source_annotations.get(source_split)
        if annotation_path is None:
            raise ValueError(f"No source annotation file configured for source split {source_split}")
        data = load_json(annotation_path)
        source_data[source_split] = data
        source_files_used.append(relative_path(annotation_path, root))
        if categories is None:
            categories = sorted(data.get("categories", []), key=lambda item: int(item["id"]))
        image_index = {int(image["id"]): image for image in data.get("images", [])}
        annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
        selected_ids = selected_by_split[source_split]
        for annotation in data.get("annotations", []):
            image_id = int(annotation.get("image_id", -1))
            if image_id in selected_ids:
                annotations_by_image[image_id].append(annotation)
        source_image_indexes[source_split] = image_index
        source_annotations_by_image[source_split] = annotations_by_image

    if categories is None:
        raise ValueError("No source categories were loaded.")
    categories = sorted(categories, key=lambda item: int(item["id"]))
    valid_category_ids = {int(category["id"]) for category in categories}

    missing_image_mapping_count = 0
    output_images: list[dict[str, Any]] = []
    for item in parsed_images:
        source_split = str(item["source_split"])
        image_id = int(item["image_id"])
        source_image = source_image_indexes[source_split].get(image_id)
        if source_image is None:
            missing_image_mapping_count += 1
            continue
        output_image = dict(source_image)
        output_image["id"] = image_id
        output_image["file_name"] = str(source_image.get("file_name", item["source_file_name"]))
        output_images.append(output_image)

    selected_image_ids = {int(image["id"]) for image in output_images}
    excluded_crowd_annotation_count = 0
    rejected_invalid_box_count = 0
    rejected_other_count = 0
    clipped_box_count = 0
    output_annotations: list[dict[str, Any]] = []
    seen_annotation_ids: set[int] = set()
    duplicate_annotation_id_count = 0

    for item in parsed_images:
        source_split = str(item["source_split"])
        image_id = int(item["image_id"])
        if image_id not in selected_image_ids:
            continue
        source_image = source_image_indexes[source_split][image_id]
        image_width = int(source_image.get("width", 0))
        image_height = int(source_image.get("height", 0))
        for annotation in source_annotations_by_image[source_split].get(image_id, []):
            annotation_id = int(annotation.get("id", -1))
            category_id = int(annotation.get("category_id", -1))
            if int(annotation.get("iscrowd", 0)) == 1:
                excluded_crowd_annotation_count += 1
                continue
            if category_id not in valid_category_ids or "bbox" not in annotation:
                rejected_other_count += 1
                continue
            try:
                clipped_bbox = clip_coco_bbox(annotation["bbox"], image_width, image_height)
            except Exception:
                rejected_invalid_box_count += 1
                continue
            original_bbox = [float(value) for value in annotation["bbox"]]
            if any(abs(a - b) > 1e-6 for a, b in zip(original_bbox, clipped_bbox, strict=True)):
                clipped_box_count += 1
            if annotation_id in seen_annotation_ids:
                duplicate_annotation_id_count += 1
            seen_annotation_ids.add(annotation_id)
            bbox_area = round(float(clipped_bbox[2]) * float(clipped_bbox[3]), 3)
            output_annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": clipped_bbox,
                    "area": bbox_area,
                    "iscrowd": 0,
                }
            )

    output_annotations.sort(key=lambda item: int(item["id"]))
    output_data = {
        "info": {
            "description": "COCO-format annotations for deterministic YOLOv5 test subset",
            "source_subset_manifest": relative_path(manifest_path, root),
        },
        "licenses": source_data[next(iter(source_data))].get("licenses", []),
        "images": output_images,
        "annotations": output_annotations,
        "categories": categories,
    }

    validation_errors = validate_coco_subset_json(output_data)
    if missing_mapping_count:
        validation_errors.append(f"{missing_mapping_count} images could not be parsed from project file names")
    if missing_image_mapping_count:
        validation_errors.append(f"{missing_image_mapping_count} images were missing from source COCO annotations")
    if duplicate_annotation_id_count:
        validation_errors.append(f"{duplicate_annotation_id_count} duplicate annotation IDs were found")

    ensure_dir(output_json.parent)
    output_json.write_text(json.dumps(output_data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    generated_sha256 = sha256_file(output_json)

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_annotation_files_used": source_files_used,
        "subset_manifest_path": relative_path(manifest_path, root),
        "selected_image_count": len(output_images),
        "selected_annotation_count": len(output_annotations),
        "category_count": len(categories),
        "excluded_crowd_annotation_count": excluded_crowd_annotation_count,
        "rejected_invalid_box_count": rejected_invalid_box_count,
        "rejected_other_annotation_count": rejected_other_count,
        "clipped_box_count": clipped_box_count,
        "duplicate_image_id_count": duplicate_image_id_count,
        "duplicate_annotation_id_count": duplicate_annotation_id_count,
        "missing_image_mapping_count": missing_mapping_count + missing_image_mapping_count,
        "missing_label_count": missing_label_count,
        "label_line_count": label_line_count,
        "generated_json_path": relative_path(output_json, root),
        "generated_json_sha256": generated_sha256,
        "validation_errors": validation_errors,
        "verification_status": "passed" if not validation_errors else "failed",
    }
    ensure_dir(verification_json.parent)
    verification_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if validation_errors:
        raise ValueError("Generated COCO subset annotation JSON failed validation; see verification report.")
    return report


def main() -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Create COCO annotations for the exact 2,500-image test subset.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--verification", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    report = create_coco_subset_annotations(args.manifest, args.output, args.verification)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
