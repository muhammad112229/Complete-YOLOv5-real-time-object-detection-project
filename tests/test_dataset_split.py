"""Tests for deterministic dataset splitting."""

from __future__ import annotations

from src.split_coco_dataset import assert_no_leakage, deterministic_split


def _record(index: int) -> dict[str, object]:
    return {
        "image_id": index,
        "source_split": "synthetic",
        "image_path": f"image_{index}.jpg",
        "label_path": f"image_{index}.txt",
    }


def test_deterministic_split_counts_and_seed() -> None:
    """Split counts and ordering should be reproducible."""
    records = [_record(index) for index in range(10)]
    first = deterministic_split(records, seed=42)
    second = deterministic_split(records, seed=42)
    assert [item["image_id"] for item in first["train"]] == [
        item["image_id"] for item in second["train"]
    ]
    assert len(first["train"]) == 8
    assert len(first["val"]) == 1
    assert len(first["test"]) == 1
    assert_no_leakage(first)

