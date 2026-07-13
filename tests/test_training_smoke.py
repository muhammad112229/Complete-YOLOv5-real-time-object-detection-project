"""Tests for YOLOv5s local training smoke helpers."""

from __future__ import annotations

from pathlib import Path

from src.training_smoke import select_diverse_records, validate_smoke_labels, write_dataset_yaml


def test_select_diverse_records_is_deterministic(tmp_path: Path) -> None:
    """Smoke subset selection should be deterministic and favor class diversity."""
    records = []
    for index in range(12):
        image = tmp_path / f"image_{index}.jpg"
        label = tmp_path / f"image_{index}.txt"
        image.write_bytes(b"placeholder")
        label.write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")
        records.append(
            {
                "internal_record_id": f"record_{index}",
                "valid_object_count": 1,
                "category_names_list": [f"class_{index % 6}"],
                "project_image_path_obj": image,
                "project_label_path_obj": label,
            }
        )

    first = select_diverse_records(records, 6, seed=42)
    second = select_diverse_records(records, 6, seed=42)

    assert [record["internal_record_id"] for record in first] == [
        record["internal_record_id"] for record in second
    ]
    assert len({record["category_names_list"][0] for record in first}) == 6


def test_validate_smoke_labels_rejects_bad_ranges(tmp_path: Path) -> None:
    """Label validation should reject malformed YOLO rows."""
    label = tmp_path / "bad.txt"
    label.write_text("80 0.5 0.5 0.2 0.2\n1 1.2 0.5 0.2 0.2\n", encoding="utf-8")

    result = validate_smoke_labels([label])

    assert result["valid"] is False
    assert len(result["errors"]) == 2


def test_write_dataset_yaml_contains_80_names(tmp_path: Path) -> None:
    """Smoke YAML should contain all 80 COCO class names."""
    names = {index: f"class_{index}" for index in range(80)}
    yaml_path = tmp_path / "smoke_dataset.yaml"

    write_dataset_yaml(yaml_path, tmp_path, names)

    text = yaml_path.read_text(encoding="utf-8")
    assert "nc: 80" in text
    assert "  79: class_79" in text
