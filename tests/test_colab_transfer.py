"""Tests for compact Colab transfer and reconstruction helpers."""

from __future__ import annotations

import csv
import json
import zipfile
from pathlib import Path

import pytest

from src.colab_transfer import (
    EXPECTED_REFERENCE,
    sha256_file,
    split_identity_hash,
    validate_bundle,
)
from src.common import load_yaml
from src.prepare_coco_colab import paths_for, reconstruct_dataset, write_dataset_yaml


def test_sha256_file_generation(tmp_path: Path) -> None:
    """SHA-256 generation should be deterministic."""
    sample = tmp_path / "sample.txt"
    sample.write_text("abc", encoding="utf-8")

    assert sha256_file(sample) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_bundle_validator_rejects_prohibited_paths(tmp_path: Path) -> None:
    """Bundles must not contain raw COCO media or other prohibited paths."""
    bundle = tmp_path / "bad.zip"
    with zipfile.ZipFile(bundle, "w") as archive:
        archive.writestr("data/raw/coco2017/train2017/000000000009.jpg", b"not really an image")

    result = validate_bundle(bundle)

    assert result.valid is False
    assert any("Prohibited path" in error or "Dataset media" in error for error in result.errors)


def test_actual_bundle_integrity_if_present() -> None:
    """The generated local bundle should open and satisfy exclusion rules."""
    bundle = Path("transfer/yolov5_colab_bundle.zip")
    if not bundle.exists():
        pytest.skip("bundle has not been generated")

    result = validate_bundle(bundle)

    assert result.valid is True
    assert result.file_count > 0


def test_manifest_identity_hash_uses_source_split_and_image_id(tmp_path: Path) -> None:
    """Split identity hashes should ignore machine-specific absolute paths."""
    manifest = tmp_path / "train_manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["source_split", "image_id"])
        writer.writeheader()
        writer.writerow({"source_split": "train2017", "image_id": "9"})
        writer.writerow({"source_split": "val2017", "image_id": "42"})

    result = split_identity_hash(manifest)

    assert result["count"] == 2
    assert len(result["sha256"]) == 64


def test_colab_yaml_template_has_80_classes() -> None:
    """Colab dataset YAML template should be Linux-compatible and complete."""
    data = load_yaml(Path("configs/coco_project_colab.yaml"))

    assert data["path"].startswith("/content/")
    assert data["nc"] == 80
    assert len(data["names"]) == 80


def test_training_guard_disabled_by_default() -> None:
    """The notebook must not allow accidental training."""
    notebook = json.loads(Path("notebooks/YOLOv5_COCO_Training_Colab.ipynb").read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", "")) for cell in notebook["cells"])

    assert "START_TRAINING = False" in source
    assert "Training is disabled. Complete and review integrity checks first." in source


def test_expected_reference_counts() -> None:
    """Reference values should match the locally verified split."""
    assert EXPECTED_REFERENCE["train_image_count"] == 98629
    assert EXPECTED_REFERENCE["val_image_count"] == 12328
    assert EXPECTED_REFERENCE["test_image_count"] == 12330
    assert EXPECTED_REFERENCE["accepted_annotation_count"] == 886282
    assert EXPECTED_REFERENCE["class_count"] == 80


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    """Write a tiny synthetic split manifest."""
    fieldnames = [
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
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_reconstruction_dry_run_with_synthetic_fixture(tmp_path: Path) -> None:
    """Reconstruction should work with tiny COCO-like fixtures and Linux paths."""
    workspace = tmp_path / "workspace"
    storage = tmp_path / "storage"
    manifests = workspace / "data" / "splits"
    paths = paths_for(workspace, storage, "runtime", manifests)
    for source_split in ("train2017", "val2017"):
        (paths.coco_root / source_split).mkdir(parents=True)
    (paths.coco_root / "annotations").mkdir(parents=True)
    (paths.coco_root / "train2017" / "000000000001.jpg").write_bytes(b"fake")
    (paths.coco_root / "val2017" / "000000000002.jpg").write_bytes(b"fake")

    categories = [{"id": 1, "name": "person", "supercategory": "person"}]
    train_json = {
        "images": [{"id": 1, "file_name": "000000000001.jpg", "width": 100, "height": 100}],
        "annotations": [{"id": 10, "image_id": 1, "category_id": 1, "bbox": [10, 20, 30, 40], "area": 1200, "iscrowd": 0}],
        "categories": categories,
    }
    val_json = {
        "images": [{"id": 2, "file_name": "000000000002.jpg", "width": 100, "height": 100}],
        "annotations": [{"id": 20, "image_id": 2, "category_id": 1, "bbox": [0, 0, 50, 50], "area": 2500, "iscrowd": 0}],
        "categories": categories,
    }
    (paths.coco_root / "annotations" / "instances_train2017.json").write_text(json.dumps(train_json), encoding="utf-8")
    (paths.coco_root / "annotations" / "instances_val2017.json").write_text(json.dumps(val_json), encoding="utf-8")

    rows = {
        "train": {
            "internal_record_id": "train2017_1",
            "source_split": "train2017",
            "image_id": "1",
            "source_image_path": r"C:\\data\\train2017\\000000000001.jpg",
            "project_image_path": r"C:\\project\\images\\train\\train2017_000000000001.jpg",
            "project_label_path": r"C:\\project\\labels\\train\\train2017_000000000001.txt",
            "width": "100",
            "height": "100",
            "valid_object_count": "1",
            "category_names": "[\"person\"]",
        },
        "val": {
            "internal_record_id": "val2017_2",
            "source_split": "val2017",
            "image_id": "2",
            "source_image_path": r"C:\\data\\val2017\\000000000002.jpg",
            "project_image_path": r"C:\\project\\images\\val\\val2017_000000000002.jpg",
            "project_label_path": r"C:\\project\\labels\\val\\val2017_000000000002.txt",
            "width": "100",
            "height": "100",
            "valid_object_count": "1",
            "category_names": "[\"person\"]",
        },
    }
    write_manifest(manifests / "train_manifest.csv", [rows["train"]])
    write_manifest(manifests / "val_manifest.csv", [rows["val"]])
    write_manifest(manifests / "test_manifest.csv", [rows["val"]])

    report = reconstruct_dataset(paths)

    assert report["splits"]["train"]["image_count"] == 1
    assert report["splits"]["val"]["image_count"] == 1
    assert (paths.processed_root / "labels" / "train" / "train2017_000000000001.txt").exists()
    assert (paths.processed_root / "coco_project.yaml").exists()


def test_write_dataset_yaml_uses_linux_paths(tmp_path: Path) -> None:
    """Generated Colab YAML should use forward-slash paths."""
    yaml_path = tmp_path / "dataset.yaml"
    write_dataset_yaml(yaml_path, Path("/content/datasets/coco2017/coco_yolo_exact_split"), {0: "person"})
    text = yaml_path.read_text(encoding="utf-8")

    assert "path: /content/datasets/coco2017/coco_yolo_exact_split" in text
    assert "\\" not in text
