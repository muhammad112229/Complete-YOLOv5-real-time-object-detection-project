"""Parse COCO annotations and create YOLO label files."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

from src.common import ensure_dir, project_root, setup_logging


LOGGER = logging.getLogger(__name__)


def load_coco_dataset(annotation_json: Path) -> dict[str, Any]:
    """Load COCO annotations using pycocotools when available."""
    try:
        coco_spec = importlib.util.find_spec("pycocotools.coco")
    except ModuleNotFoundError:
        coco_spec = None
    if coco_spec is not None:
        coco_module = importlib.import_module("pycocotools.coco")
        coco_api = coco_module.COCO(str(annotation_json))
        image_ids = coco_api.getImgIds()
        category_ids = coco_api.getCatIds()
        annotation_ids = coco_api.getAnnIds(imgIds=image_ids)
        return {
            "images": coco_api.loadImgs(image_ids),
            "annotations": coco_api.loadAnns(annotation_ids),
            "categories": coco_api.loadCats(category_ids),
        }
    LOGGER.warning("pycocotools is not installed; falling back to direct JSON parsing.")
    return json.loads(annotation_json.read_text(encoding="utf-8"))


def is_image_readable(image_path: Path) -> bool:
    """Return whether an image can be opened and verified when Pillow is installed."""
    try:
        pillow_spec = importlib.util.find_spec("PIL.Image")
    except ModuleNotFoundError:
        pillow_spec = None
    if pillow_spec is None:
        return True
    image_module = importlib.import_module("PIL.Image")
    try:
        with image_module.open(image_path) as image:
            image.verify()
    except Exception:
        return False
    return True


def coco_bbox_to_yolo(
    bbox: list[float] | tuple[float, float, float, float],
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float]:
    """Convert a COCO pixel bbox to normalized YOLO xywh format.

    COCO format is `[x_min, y_min, width, height]` in pixels. Coordinates are
    clipped to the image boundary before normalization.
    """
    if image_width <= 0 or image_height <= 0:
        raise ValueError("Image dimensions must be positive.")
    if len(bbox) != 4:
        raise ValueError("COCO bbox must contain four values.")

    x_min, y_min, width, height = (float(value) for value in bbox)
    if width <= 0 or height <= 0:
        raise ValueError("COCO bbox width and height must be positive.")

    x1 = max(0.0, x_min)
    y1 = max(0.0, y_min)
    x2 = min(float(image_width), x_min + width)
    y2 = min(float(image_height), y_min + height)
    if x2 <= x1 or y2 <= y1:
        raise ValueError("COCO bbox lies outside the image after clipping.")

    box_width = x2 - x1
    box_height = y2 - y1
    x_center = x1 + box_width / 2.0
    y_center = y1 + box_height / 2.0
    normalized = (
        x_center / image_width,
        y_center / image_height,
        box_width / image_width,
        box_height / image_height,
    )
    if not all(0.0 <= value <= 1.0 for value in normalized):
        raise ValueError(f"Normalized bbox is out of range: {normalized}")
    return normalized


def build_category_mapping(categories: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    """Build contiguous YOLO class IDs from COCO category IDs."""
    sorted_categories = sorted(categories, key=lambda item: int(item["id"]))
    mapping: dict[int, dict[str, Any]] = {}
    for class_index, category in enumerate(sorted_categories):
        coco_id = int(category["id"])
        mapping[coco_id] = {
            "class_id": class_index,
            "coco_category_id": coco_id,
            "name": str(category["name"]),
            "supercategory": str(category.get("supercategory", "")),
        }
    return mapping


def write_label_file(path: Path, labels: list[tuple[int, tuple[float, float, float, float]]]) -> None:
    """Write one YOLO label file."""
    ensure_dir(path.parent)
    lines = [
        f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}"
        for class_id, (x, y, w, h) in labels
    ]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def convert_coco_annotations(
    annotation_json: Path,
    image_root: Path,
    output_root: Path,
    split_name: str,
    check_images: bool = True,
    limit: int | None = None,
) -> dict[str, Any]:
    """Convert a COCO annotation JSON into YOLO labels and a manifest."""
    annotation_json = annotation_json.resolve()
    image_root = image_root.resolve()
    output_root = output_root.resolve()
    labels_dir = ensure_dir(output_root / "labels_source" / split_name)
    manifests_dir = ensure_dir(output_root / "manifests")

    coco = load_coco_dataset(annotation_json)
    category_mapping = build_category_mapping(coco.get("categories", []))
    annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in coco.get("annotations", []):
        annotations_by_image[int(annotation["image_id"])].append(annotation)

    records: list[dict[str, Any]] = []
    converted_annotations: list[dict[str, Any]] = []
    stats = {
        "split_name": split_name,
        "images_seen": 0,
        "images_written": 0,
        "annotations_seen": len(coco.get("annotations", [])),
        "annotations_written": 0,
        "invalid_boxes": 0,
        "missing_images": 0,
        "corrupt_images": 0,
        "unsupported_annotations": 0,
        "crowd_annotations_skipped": 0,
    }

    for image in coco.get("images", []):
        if limit is not None and stats["images_seen"] >= limit:
            break
        stats["images_seen"] += 1
        image_id = int(image["id"])
        width = int(image["width"])
        height = int(image["height"])
        file_name = str(image["file_name"])
        image_path = image_root / file_name
        if check_images and not image_path.exists():
            stats["missing_images"] += 1
            continue
        if check_images and not is_image_readable(image_path):
            stats["corrupt_images"] += 1
            continue

        yolo_labels: list[tuple[int, tuple[float, float, float, float]]] = []
        objects: list[dict[str, Any]] = []
        for annotation in annotations_by_image.get(image_id, []):
            if int(annotation.get("iscrowd", 0)) == 1:
                stats["crowd_annotations_skipped"] += 1
                continue
            if "bbox" not in annotation or "category_id" not in annotation:
                stats["unsupported_annotations"] += 1
                continue
            category_id = int(annotation["category_id"])
            if category_id not in category_mapping:
                stats["unsupported_annotations"] += 1
                continue
            try:
                yolo_box = coco_bbox_to_yolo(annotation["bbox"], width, height)
            except ValueError:
                stats["invalid_boxes"] += 1
                continue

            class_id = int(category_mapping[category_id]["class_id"])
            yolo_labels.append((class_id, yolo_box))
            converted = {
                "id": int(annotation.get("id", len(converted_annotations))),
                "image_id": image_id,
                "category_id": category_id,
                "class_id": class_id,
                "category_name": category_mapping[category_id]["name"],
                "bbox_coco": annotation["bbox"],
                "bbox_yolo": list(yolo_box),
                "area": annotation.get("area"),
            }
            converted_annotations.append(converted)
            objects.append(converted)

        label_path = labels_dir / f"{Path(file_name).stem}.txt"
        write_label_file(label_path, yolo_labels)
        records.append(
            {
                "image_id": image_id,
                "source_split": split_name,
                "file_name": file_name,
                "image_path": str(image_path),
                "label_path": str(label_path),
                "width": width,
                "height": height,
                "annotations": objects,
            }
        )
        stats["images_written"] += 1
        stats["annotations_written"] += len(yolo_labels)

    class_mapping_path = output_root / "class_mapping.json"
    class_mapping_path.write_text(
        json.dumps(category_mapping, indent=2),
        encoding="utf-8",
    )
    (output_root / "class_names.txt").write_text(
        "\n".join(item["name"] for _, item in sorted(category_mapping.items())) + "\n",
        encoding="utf-8",
    )

    manifest = {
        "source_annotation_json": str(annotation_json),
        "image_root": str(image_root),
        "split_name": split_name,
        "categories": list(category_mapping.values()),
        "images": records,
        "annotations": converted_annotations,
        "statistics": stats,
    }
    manifest_path = manifests_dir / f"{split_name}_processed.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (manifests_dir / f"{split_name}_statistics.json").write_text(
        json.dumps(stats, indent=2),
        encoding="utf-8",
    )
    LOGGER.info("Wrote %s", manifest_path)
    return manifest


def main() -> int:
    """CLI entrypoint for COCO annotation conversion."""
    root = project_root()
    parser = argparse.ArgumentParser(description="Convert COCO annotations to YOLO labels.")
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=root / "data" / "processed" / "coco2017_yolo")
    parser.add_argument("--split-name", required=True, help="Example: train2017 or val2017")
    parser.add_argument("--no-check-images", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="Optional small subset for smoke tests.")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)
    convert_coco_annotations(
        annotation_json=args.annotations,
        image_root=args.images,
        output_root=args.output_root,
        split_name=args.split_name,
        check_images=not args.no_check_images,
        limit=args.limit,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
