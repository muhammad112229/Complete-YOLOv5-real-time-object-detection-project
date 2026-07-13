"""Tests for evaluation, analysis, checkpoint, and sample-summary helpers."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from src.analyze_training_results import analyze_training_results
from src.create_coco_subset_annotations import create_coco_subset_annotations
from src.evaluate_model import (
    EvaluationConfig,
    convert_yolov5_predictions_to_coco,
    load_evaluation_config,
    validate_test_evaluation_inputs,
)
from src.inspect_checkpoint import inspect_checkpoint
from src.recreate_test_subset import recreate_subset, sha256_file_bytes
from src.sample_predictions import confidence_statistics, sample_record


def _write_manifest(path: Path, image: Path, label: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
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
        writer.writerow(
            {
                "internal_record_id": "x",
                "source_split": "test",
                "image_id": "1",
                "source_image_path": str(image),
                "project_image_path": str(image),
                "project_label_path": str(label),
                "width": 10,
                "height": 10,
                "valid_object_count": 1,
                "category_names": "[]",
            }
        )


def test_evaluation_config_loading(tmp_path: Path) -> None:
    config_path = tmp_path / "evaluation.yaml"
    config_path.write_text(
        "\n".join(
            [
                "model_path: models/yolov5s_coco20k_best.pt",
                "dataset_yaml: data/processed/coco_yolo/coco_project.yaml",
                "test_manifest: data/splits/test_manifest.csv",
                "image_size: 320",
                "batch_size: 4",
                "confidence_threshold: 0.001",
                "iou_threshold: 0.6",
                "device: auto",
                "workers: 2",
                "output_directory: results/evaluation",
            ]
        ),
        encoding="utf-8",
    )
    config = load_evaluation_config(config_path)
    assert config.image_size == 320
    assert config.batch_size == 4
    assert config.device == "auto"


def test_missing_test_dataset_handling(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"not-a-real-checkpoint")
    image = tmp_path / "image.jpg"
    label = tmp_path / "image.txt"
    image.write_bytes(b"image")
    label.write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")
    manifest = tmp_path / "test_manifest.csv"
    _write_manifest(manifest, image, label)
    config = EvaluationConfig(
        model_path=checkpoint,
        dataset_yaml=tmp_path / "missing.yaml",
        test_manifest=manifest,
        output_directory=tmp_path / "eval",
    )
    with pytest.raises(FileNotFoundError):
        validate_test_evaluation_inputs(config)


def test_prevents_validation_test_substitution(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"not-a-real-checkpoint")
    dataset_yaml = tmp_path / "dataset.yaml"
    dataset_yaml.write_text("path: .\ntrain: images/train\nval: images/val\ntest: images/val\nnc: 1\n", encoding="utf-8")
    image = tmp_path / "image.jpg"
    label = tmp_path / "image.txt"
    image.write_bytes(b"image")
    label.write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")
    manifest = tmp_path / "test_manifest.csv"
    _write_manifest(manifest, image, label)
    config = EvaluationConfig(
        model_path=checkpoint,
        dataset_yaml=dataset_yaml,
        test_manifest=manifest,
        output_directory=tmp_path / "eval",
    )
    with pytest.raises(ValueError, match="test split must not be the same"):
        validate_test_evaluation_inputs(config)


def test_training_results_analysis(tmp_path: Path) -> None:
    results_csv = tmp_path / "results.csv"
    results_csv.write_text(
        "\n".join(
            [
                "epoch,train/box_loss,train/obj_loss,train/cls_loss,metrics/precision,metrics/recall,metrics/mAP_0.5,metrics/mAP_0.5:0.95,val/box_loss,val/obj_loss,val/cls_loss",
                "0,0.2,0.3,0.4,0.5,0.6,0.7,0.2,0.3,0.4,0.5",
                "1,0.1,0.2,0.3,0.6,0.7,0.8,0.4,0.2,0.3,0.4",
            ]
        ),
        encoding="utf-8",
    )
    opt_yaml = tmp_path / "opt.yaml"
    opt_yaml.write_text("epochs: 3\n", encoding="utf-8")
    result = analyze_training_results(results_csv, opt_yaml, tmp_path / "analysis", make_plots=False)
    assert result["summary"]["best_epoch"] == 1
    assert result["summary"]["total_completed_epochs"] == 2
    assert "2 epochs completed although 3 were configured" in result["summary"]["completed_epochs_note"]


def test_checkpoint_inspection(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    checkpoint_path = tmp_path / "checkpoint.pt"
    model = torch.nn.Linear(2, 1)
    model.nc = 1
    model.names = ["object"]
    model.stride = torch.tensor([32])
    torch.save({"model": model, "epoch": 2, "best_fitness": 0.5, "optimizer": None, "ema": None}, checkpoint_path)
    report = inspect_checkpoint(checkpoint_path)
    assert report["number_of_classes"] == 1
    assert report["class_names_count"] == 1
    assert report["model_parameter_count"] == 3
    assert report["checkpoint_epoch"] == 2


def test_sample_prediction_summary_structure(tmp_path: Path) -> None:
    detections = (
        {"class_id": 0, "class_name": "person", "confidence": 0.5, "x1": 1.0, "y1": 2.0, "x2": 3.0, "y2": 4.0},
        {"class_id": 0, "class_name": "person", "confidence": 0.75, "x1": 2.0, "y1": 3.0, "x2": 4.0, "y2": 5.0},
    )
    record = sample_record(
        source=tmp_path / "source.jpg",
        output=tmp_path / "output.jpg",
        width=640,
        height=480,
        detection_count=2,
        class_counts={"person": 2},
        confidence_stats=confidence_statistics(detections),
        inference_ms=12.5,
    )
    assert record["number_of_detections"] == 2
    assert record["confidence_statistics"]["max"] == 0.75
    assert record["class_counts"] == {"person": 2}


def test_deterministic_subset_recreation(tmp_path: Path) -> None:
    image_dir = tmp_path / "images" / "test"
    label_dir = tmp_path / "labels" / "test"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    paths = []
    for index in range(10):
        image = image_dir / f"image_{index}.jpg"
        label = label_dir / f"image_{index}.txt"
        image.write_bytes(b"image")
        label.write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")
        paths.append(str(image))
    source = tmp_path / "test_images.txt"
    source.write_text("\n".join(paths) + "\n", encoding="utf-8")
    first_output = tmp_path / "subset_a.txt"
    second_output = tmp_path / "subset_b.txt"
    report_a = recreate_subset(source, first_output, tmp_path / "report_a.json", seed=42, subset_size=5)
    report_b = recreate_subset(source, second_output, tmp_path / "report_b.json", seed=42, subset_size=5)
    assert first_output.read_text(encoding="utf-8") == second_output.read_text(encoding="utf-8")
    assert report_a["subset_line_count"] == 5
    assert report_a["subset_duplicate_count"] == 0
    assert report_a["subset_missing_image_count"] == 0
    assert report_a["subset_missing_label_count"] == 0
    assert report_a["subset_manifest_sha256"] == report_b["subset_manifest_sha256"]


def test_manifest_sha256_stability(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.txt"
    manifest.write_text("a\nb\n", encoding="utf-8")
    assert sha256_file_bytes(manifest) == sha256_file_bytes(manifest)


def _write_synthetic_coco_subset_source(tmp_path: Path, count: int = 2500) -> tuple[Path, Path]:
    image_dir = tmp_path / "processed" / "images" / "test"
    label_dir = tmp_path / "processed" / "labels" / "test"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    manifest_lines = []
    images = []
    annotations = []
    for image_id in range(1, count + 1):
        project_image = image_dir / f"train2017_{image_id:012d}.jpg"
        project_label = label_dir / f"train2017_{image_id:012d}.txt"
        project_image.write_bytes(b"image")
        project_label.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
        manifest_lines.append(str(project_image))
        images.append({"id": image_id, "file_name": f"{image_id:012d}.jpg", "width": 100, "height": 100})
        annotations.append(
            {
                "id": 100000 + image_id,
                "image_id": image_id,
                "category_id": 1,
                "bbox": [10, 10, 20, 20],
                "area": 400,
                "iscrowd": 0,
            }
        )
    categories = [{"id": category_id, "name": f"class_{category_id}", "supercategory": "object"} for category_id in range(1, 81)]
    source = tmp_path / "instances_train2017.json"
    source.write_text(
        json.dumps({"info": {}, "licenses": [], "images": images, "annotations": annotations, "categories": categories}),
        encoding="utf-8",
    )
    manifest = tmp_path / "test_subset_2500_seed42.txt"
    manifest.write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
    return manifest, source


def test_deterministic_coco_subset_annotation_generation(tmp_path: Path) -> None:
    manifest, source = _write_synthetic_coco_subset_source(tmp_path)
    first_output = tmp_path / "first.json"
    second_output = tmp_path / "second.json"
    first_report = create_coco_subset_annotations(
        manifest,
        first_output,
        tmp_path / "first_report.json",
        source_annotations={"train2017": source},
    )
    second_report = create_coco_subset_annotations(
        manifest,
        second_output,
        tmp_path / "second_report.json",
        source_annotations={"train2017": source},
    )
    assert first_report["verification_status"] == "passed"
    assert first_report["generated_json_sha256"] == second_report["generated_json_sha256"]
    assert first_report["selected_image_count"] == 2500
    assert first_report["selected_annotation_count"] == 2500
    assert first_report["category_count"] == 80


def test_coco_subset_annotation_integrity(tmp_path: Path) -> None:
    manifest, source = _write_synthetic_coco_subset_source(tmp_path)
    output = tmp_path / "subset.json"
    create_coco_subset_annotations(
        manifest,
        output,
        tmp_path / "report.json",
        source_annotations={"train2017": source},
    )
    data = json.loads(output.read_text(encoding="utf-8"))
    image_ids = [image["id"] for image in data["images"]]
    annotation_ids = [annotation["id"] for annotation in data["annotations"]]
    category_ids = {category["id"] for category in data["categories"]}
    assert len(image_ids) == len(set(image_ids)) == 2500
    assert len(annotation_ids) == len(set(annotation_ids)) == 2500
    assert len(category_ids) == 80
    assert all(annotation["image_id"] in set(image_ids) for annotation in data["annotations"])
    assert all(annotation["category_id"] in category_ids for annotation in data["annotations"])


def test_cocoeval_configuration_loading(tmp_path: Path) -> None:
    config_path = tmp_path / "evaluation.yaml"
    config_path.write_text(
        "\n".join(
            [
                "model_path: models/yolov5s_coco20k_best.pt",
                "dataset_yaml: data/processed/coco_yolo/coco_project.yaml",
                "test_manifest: data/splits/test_subset_2500_seed42.txt",
                "coco_annotation_json: data/processed/coco_yolo/annotations/instances_test_subset_2500_seed42.json",
                "output_directory: results/evaluation/test_subset_2500",
            ]
        ),
        encoding="utf-8",
    )
    config = load_evaluation_config(config_path)
    assert config.coco_annotation_json is not None
    assert config.coco_annotation_json.name == "instances_test_subset_2500_seed42.json"


def test_yolov5_predictions_convert_to_coco_ids(tmp_path: Path) -> None:
    annotation_json = tmp_path / "subset.json"
    annotation_json.write_text(
        json.dumps(
            {
                "images": [{"id": 7, "file_name": "000000000007.jpg", "width": 100, "height": 100}],
                "annotations": [],
                "categories": [{"id": 1, "name": "person"}, {"id": 3, "name": "car"}],
            }
        ),
        encoding="utf-8",
    )
    yolov5_predictions = tmp_path / "predictions.json"
    yolov5_predictions.write_text(
        json.dumps([{"image_id": "train2017_000000000007", "category_id": 1, "bbox": [1, 2, 3, 4], "score": 0.9}]),
        encoding="utf-8",
    )
    report = convert_yolov5_predictions_to_coco(yolov5_predictions, annotation_json, tmp_path / "coco_predictions.json")
    converted = json.loads((tmp_path / "coco_predictions.json").read_text(encoding="utf-8"))
    assert report["converted_prediction_count"] == 1
    assert converted[0]["image_id"] == 7
    assert converted[0]["category_id"] == 3
