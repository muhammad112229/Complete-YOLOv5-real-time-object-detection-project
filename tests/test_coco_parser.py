"""Tests for COCO annotation conversion helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.parse_coco_annotations import (
    build_category_mapping,
    coco_bbox_to_yolo,
    convert_coco_annotations,
)


def test_coco_bbox_to_yolo_normalizes_values() -> None:
    """COCO boxes should convert to normalized YOLO xywh."""
    converted = coco_bbox_to_yolo([10, 20, 30, 40], image_width=100, image_height=200)
    assert converted == pytest.approx((0.25, 0.2, 0.3, 0.2))


def test_coco_bbox_to_yolo_rejects_invalid_box() -> None:
    """Invalid boxes should be rejected."""
    with pytest.raises(ValueError):
        coco_bbox_to_yolo([10, 20, 0, 40], image_width=100, image_height=200)


def test_build_category_mapping_uses_contiguous_ids() -> None:
    """COCO category IDs should map to contiguous YOLO class IDs."""
    mapping = build_category_mapping(
        [
            {"id": 3, "name": "car"},
            {"id": 1, "name": "person"},
        ]
    )
    assert mapping[1]["class_id"] == 0
    assert mapping[3]["class_id"] == 1


def test_convert_coco_annotations_writes_labels(tmp_path: Path) -> None:
    """A tiny COCO JSON should produce a processed manifest and label file."""
    annotation_json = tmp_path / "instances.json"
    image_root = tmp_path / "images"
    image_root.mkdir()
    annotation_json.write_text(
        json.dumps(
            {
                "images": [{"id": 1, "file_name": "0001.jpg", "width": 100, "height": 100}],
                "annotations": [
                    {
                        "id": 11,
                        "image_id": 1,
                        "category_id": 1,
                        "bbox": [10, 10, 20, 20],
                        "area": 400,
                        "iscrowd": 0,
                    }
                ],
                "categories": [{"id": 1, "name": "person", "supercategory": "person"}],
            }
        ),
        encoding="utf-8",
    )
    manifest = convert_coco_annotations(
        annotation_json=annotation_json,
        image_root=image_root,
        output_root=tmp_path / "processed",
        split_name="mini",
        check_images=False,
    )
    label_path = Path(manifest["images"][0]["label_path"])
    assert label_path.exists()
    assert label_path.read_text(encoding="utf-8").strip().startswith("0 ")

