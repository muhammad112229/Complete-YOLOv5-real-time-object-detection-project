"""Tests for COCO preparation helpers using tiny synthetic fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.prepare_coco_dataset import (
    clip_coco_bbox,
    deterministic_image_split,
    letterbox_transform,
    validate_dataset_yaml,
)


def test_clip_coco_bbox_clips_to_image_bounds() -> None:
    """Boxes that extend outside the image should be clipped before YOLO conversion."""
    clipped, yolo = clip_coco_bbox([-5, 10, 20, 20], 100, 100)
    assert clipped == pytest.approx([0, 10, 15, 20])
    assert all(0 <= value <= 1 for value in yolo)


def test_clip_coco_bbox_rejects_malformed_after_clip() -> None:
    """Boxes outside the image should be rejected after clipping."""
    with pytest.raises(ValueError):
        clip_coco_bbox([200, 200, 10, 10], 100, 100)


def test_prepare_split_reproducible_no_leakage() -> None:
    """Preparation split should be deterministic and image-level."""
    records = [
        {"internal_record_id": f"train2017_{index}", "usable_for_training": True}
        for index in range(20)
    ]
    first = deterministic_image_split(records, seed=42)
    second = deterministic_image_split(records, seed=42)
    assert [item["internal_record_id"] for item in first["train"]] == [
        item["internal_record_id"] for item in second["train"]
    ]
    all_ids = [item["internal_record_id"] for split in first.values() for item in split]
    assert len(all_ids) == len(set(all_ids))
    assert len(first["train"]) == 16
    assert len(first["val"]) == 2
    assert len(first["test"]) == 2


def test_letterbox_transform_preserves_aspect_ratio_and_padding() -> None:
    """Letterbox transform should preserve aspect ratio and add padding."""
    scale, pad_x, pad_y, boxes = letterbox_transform(800, 400, [(0, 0, 800, 400)], 640)
    assert scale == pytest.approx(0.8)
    assert pad_x == 0
    assert pad_y == 160
    assert boxes[0] == pytest.approx((0, 160, 640, 480))


def test_validate_dataset_yaml_accepts_80_names(tmp_path: Path) -> None:
    """Dataset YAML validation should require 80 classes and resolvable paths."""
    root = tmp_path / "dataset"
    for split in ("train", "val", "test"):
        (root / "images" / split).mkdir(parents=True)
    yaml_path = tmp_path / "coco_project.yaml"
    names = "\n".join(f"  {index}: class_{index}" for index in range(80))
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {root.as_posix()}",
                "train: images/train",
                "val: images/val",
                "test: images/test",
                "nc: 80",
                "names:",
                names,
                "",
            ]
        ),
        encoding="utf-8",
    )
    result = validate_dataset_yaml(yaml_path)
    assert result["valid"] is True
    assert result["name_count"] == 80
