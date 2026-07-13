"""Evaluation readiness checks and YOLOv5 test-set evaluation orchestration."""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.common import ensure_dir, project_root, require_file, require_python_package, setup_logging, yolov5_root
from src.inference_engine import resolve_device
from src.recreate_test_subset import label_path_for_image


DEFAULT_CONFIG = Path("configs") / "evaluation.yaml"
DEFAULT_MODEL = Path("models") / "yolov5s_coco20k_best.pt"
DEFAULT_DATASET_YAML = Path("data") / "processed" / "coco_yolo" / "coco_project.yaml"
DEFAULT_TEST_MANIFEST = Path("data") / "splits" / "test_manifest.csv"
DEFAULT_COCO_ANNOTATION_JSON = (
    Path("data")
    / "processed"
    / "coco_yolo"
    / "annotations"
    / "instances_test_subset_2500_seed42.json"
)
DEFAULT_OUTPUT_DIR = Path("results") / "evaluation"
COCOEVAL_METRIC_NAMES = [
    "AP@[0.50:0.95]",
    "AP@0.50",
    "AP@0.75",
    "AP small",
    "AP medium",
    "AP large",
    "AR maxDets=1",
    "AR maxDets=10",
    "AR maxDets=100",
    "AR small",
    "AR medium",
    "AR large",
]


@dataclass(frozen=True)
class EvaluationConfig:
    """Configuration for genuine YOLOv5 test-set evaluation."""

    model_path: Path
    dataset_yaml: Path
    test_manifest: Path
    coco_annotation_json: Path | None = None
    image_size: int = 640
    batch_size: int = 16
    confidence_threshold: float = 0.001
    iou_threshold: float = 0.6
    device: str = "auto"
    workers: int = 4
    output_directory: Path = DEFAULT_OUTPUT_DIR
    save_json: bool = True
    save_txt: bool = True
    save_confusion_matrix: bool = True
    save_plots: bool = True
    save_sample_predictions: bool = True
    maximum_sample_predictions: int = 20


def resolve_workspace_path(value: str | Path, root: Path | None = None) -> Path:
    """Resolve a workspace-relative path while preserving absolute Windows paths."""
    base = root or project_root()
    path = Path(value)
    return path if path.is_absolute() else base / path


def resolve_optional_workspace_path(value: str | Path | None, root: Path | None = None) -> Path | None:
    """Resolve an optional workspace path."""
    if value in {None, ""}:
        return None
    return resolve_workspace_path(value, root)


def relative_path(path: Path, root: Path | None = None) -> str:
    """Return a workspace-relative path when possible."""
    base = root or project_root()
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return str(path)


def parse_scalar(value: str) -> Any:
    """Parse the scalar subset of YAML used by project config files."""
    text = value.strip()
    if text.lower() in {"null", "none", "~"}:
        return None
    if text.lower() == "true":
        return True
    if text.lower() == "false":
        return False
    if text.startswith(("'", '"')) and text.endswith(("'", '"')):
        return text[1:-1]
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


def load_flat_yaml(path: Path) -> dict[str, Any]:
    """Load flat YAML config without requiring PyYAML for lightweight tests."""
    data: dict[str, Any] = {}
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if raw_line.startswith((" ", "\t")):
            continue
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("- "):
            continue
        if ":" not in line:
            raise ValueError(f"Invalid YAML line {line_number} in {path}: {raw_line}")
        key, value = line.split(":", 1)
        data[key.strip()] = parse_scalar(value)
    return data


def load_top_level_yaml(path: Path) -> dict[str, Any]:
    """Load top-level YAML keys; nested values are ignored except by PyYAML when installed."""
    try:
        import yaml  # type: ignore

        with path.open("r", encoding="utf-8") as file:
            data = yaml.safe_load(file) or {}
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    data: dict[str, Any] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if not raw_line or raw_line.startswith((" ", "\t", "#")):
            continue
        line = raw_line.split("#", 1)[0].strip()
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        if value.strip():
            data[key.strip()] = parse_scalar(value)
    return data


def load_evaluation_config(config_path: Path | None = None, **overrides: Any) -> EvaluationConfig:
    """Load evaluation config and apply CLI overrides."""
    root = project_root()
    path = resolve_workspace_path(config_path or DEFAULT_CONFIG, root)
    data = {
        "model_path": DEFAULT_MODEL,
        "dataset_yaml": DEFAULT_DATASET_YAML,
        "test_manifest": DEFAULT_TEST_MANIFEST,
        "coco_annotation_json": DEFAULT_COCO_ANNOTATION_JSON,
        "image_size": 640,
        "batch_size": 16,
        "confidence_threshold": 0.001,
        "iou_threshold": 0.6,
        "device": "auto",
        "workers": 4,
        "output_directory": DEFAULT_OUTPUT_DIR,
        "save_json": True,
        "save_txt": True,
        "save_confusion_matrix": True,
        "save_plots": True,
        "save_sample_predictions": True,
        "maximum_sample_predictions": 20,
    }
    if path.exists():
        data.update(load_flat_yaml(path))
    for key, value in overrides.items():
        if value is not None:
            data[key] = value

    return EvaluationConfig(
        model_path=resolve_workspace_path(data["model_path"], root),
        dataset_yaml=resolve_workspace_path(data["dataset_yaml"], root),
        test_manifest=resolve_workspace_path(data["test_manifest"], root),
        coco_annotation_json=resolve_optional_workspace_path(data.get("coco_annotation_json"), root),
        image_size=int(data["image_size"]),
        batch_size=int(data["batch_size"]),
        confidence_threshold=float(data["confidence_threshold"]),
        iou_threshold=float(data["iou_threshold"]),
        device=str(data["device"]),
        workers=int(data["workers"]),
        output_directory=resolve_workspace_path(data["output_directory"], root),
        save_json=bool(data["save_json"]),
        save_txt=bool(data["save_txt"]),
        save_confusion_matrix=bool(data["save_confusion_matrix"]),
        save_plots=bool(data["save_plots"]),
        save_sample_predictions=bool(data["save_sample_predictions"]),
        maximum_sample_predictions=int(data["maximum_sample_predictions"]),
    )


def sha256_file(path: Path) -> str | None:
    """Compute SHA256 for a file, returning None when the file is missing."""
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_manifest_availability(manifest_path: Path) -> dict[str, Any]:
    """Count usable image/label pairs referenced by a split manifest."""
    if not manifest_path.is_file():
        return {
            "available": False,
            "rows": 0,
            "image_exists": 0,
            "label_exists": 0,
            "usable_image_label_pairs": 0,
            "instances": 0,
            "missing_images_preview": [],
            "missing_labels_preview": [],
        }

    if manifest_path.suffix.lower() == ".txt":
        lines = [line.strip() for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        image_exists = 0
        label_exists = 0
        usable = 0
        instances = 0
        missing_images: list[str] = []
        missing_labels: list[str] = []
        for line in lines:
            image = Path(line)
            label = label_path_for_image(image)
            image_ok = image.is_file()
            label_ok = label.is_file()
            image_exists += int(image_ok)
            label_exists += int(label_ok)
            usable += int(image_ok and label_ok)
            if label_ok:
                instances += sum(1 for item in label.read_text(encoding="utf-8").splitlines() if item.strip())
            if not image_ok and len(missing_images) < 5:
                missing_images.append(str(image))
            if not label_ok and len(missing_labels) < 5:
                missing_labels.append(str(label))
        return {
            "available": True,
            "rows": len(lines),
            "image_exists": image_exists,
            "label_exists": label_exists,
            "usable_image_label_pairs": usable,
            "instances": instances,
            "missing_images_preview": missing_images,
            "missing_labels_preview": missing_labels,
        }

    rows = 0
    image_exists = 0
    label_exists = 0
    usable = 0
    instances = 0
    missing_images: list[str] = []
    missing_labels: list[str] = []
    with manifest_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            rows += 1
            image = Path(row.get("project_image_path", ""))
            label = Path(row.get("project_label_path", ""))
            image_ok = image.is_file()
            label_ok = label.is_file()
            image_exists += int(image_ok)
            label_exists += int(label_ok)
            usable += int(image_ok and label_ok)
            instances += int(row.get("valid_object_count") or 0)
            if not image_ok and len(missing_images) < 5:
                missing_images.append(str(image))
            if not label_ok and len(missing_labels) < 5:
                missing_labels.append(str(label))

    return {
        "available": True,
        "rows": rows,
        "image_exists": image_exists,
        "label_exists": label_exists,
        "usable_image_label_pairs": usable,
        "instances": instances,
        "missing_images_preview": missing_images,
        "missing_labels_preview": missing_labels,
    }


def find_exact_2500_test_manifest(root: Path | None = None) -> dict[str, Any]:
    """Search known metadata locations for an exact 2,500-image test subset manifest."""
    base = root or project_root()
    search_roots = [base / "data" / "splits", base / "artifacts", base / "transfer", base / "configs"]
    candidates: list[dict[str, Any]] = []
    for search_root in search_roots:
        if not search_root.exists():
            continue
        for path in search_root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in {".csv", ".txt", ".json", ".yaml", ".yml"}:
                continue
            lowered = path.name.lower()
            if not any(token in lowered for token in ["2500", "2k", "subset", "test", "manifest"]):
                continue
            row_count: int | None = None
            if path.suffix.lower() == ".csv":
                try:
                    with path.open("r", encoding="utf-8", newline="") as file:
                        row_count = sum(1 for _ in csv.DictReader(file))
                except Exception:
                    row_count = None
            elif path.suffix.lower() == ".txt":
                try:
                    row_count = sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
                except Exception:
                    row_count = None
            candidates.append({"path": relative_path(path, base), "row_count": row_count})
            if row_count == 2500 and "test" in lowered:
                return {"available": True, "path": relative_path(path, base), "candidate_files": candidates}
    return {"available": False, "path": None, "candidate_files": candidates}


def create_evaluation_readiness_report(config: EvaluationConfig | None = None) -> dict[str, Any]:
    """Create a structured report describing whether genuine evaluation can run."""
    root = project_root()
    cfg = config or load_evaluation_config()
    dataset = load_top_level_yaml(cfg.dataset_yaml) if cfg.dataset_yaml.is_file() else {}
    manifest = inspect_manifest_availability(cfg.test_manifest)
    exact_2500 = find_exact_2500_test_manifest(root)
    if cfg.test_manifest.suffix.lower() == ".txt" and manifest["rows"] == 2500 and "test" in cfg.test_manifest.name.lower():
        exact_2500 = {
            "available": True,
            "path": relative_path(cfg.test_manifest),
            "candidate_files": exact_2500.get("candidate_files", []),
        }
    checkpoint_exists = cfg.model_path.is_file()
    checkpoint_size = cfg.model_path.stat().st_size if checkpoint_exists else None
    dataset_train = dataset.get("train")
    dataset_val = dataset.get("val")
    dataset_test = dataset.get("test")
    class_count = int(dataset.get("nc")) if dataset.get("nc") is not None else None
    no_validation_substitution = bool(dataset_test and dataset_test != dataset_val and "val" not in cfg.test_manifest.name.lower())
    no_training_substitution = bool(dataset_test and dataset_test != dataset_train and "train" not in cfg.test_manifest.name.lower())
    coco_annotation_exists = bool(cfg.coco_annotation_json and cfg.coco_annotation_json.is_file())

    blocking_issues: list[str] = []
    if not checkpoint_exists:
        blocking_issues.append("Production checkpoint is missing.")
    if not cfg.dataset_yaml.is_file():
        blocking_issues.append("Dataset YAML is missing.")
    if not cfg.test_manifest.is_file():
        blocking_issues.append("Test manifest is missing.")
    if manifest["available"] and manifest["usable_image_label_pairs"] == 0:
        blocking_issues.append("Test manifest has no usable image/label pairs.")

    full_project_ready = not blocking_issues
    exact_2500_ready = full_project_ready and bool(exact_2500["available"])
    if full_project_ready and not exact_2500_ready:
        blocking_issues.append(
            "Exact 2,500-image faster-experiment test subset manifest was not found; "
            "full project test split is available but is not the requested exact subset."
        )

    if exact_2500_ready:
        status = "ready_for_exact_2500_test_evaluation"
    elif full_project_ready:
        status = "ready_for_full_project_test_evaluation_exact_2500_subset_missing"
    else:
        status = "blocked"

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint_path": relative_path(cfg.model_path, root),
        "checkpoint_exists": checkpoint_exists,
        "checkpoint_size": checkpoint_size,
        "checkpoint_hash": sha256_file(cfg.model_path),
        "dataset_yaml": relative_path(cfg.dataset_yaml, root),
        "dataset_test_yaml_availability": bool(cfg.dataset_yaml.is_file() and dataset.get("test")),
        "dataset_yaml_train_value": dataset_train,
        "dataset_yaml_val_value": dataset_val,
        "dataset_yaml_test_value": dataset_test,
        "coco_annotation_json": relative_path(cfg.coco_annotation_json, root) if cfg.coco_annotation_json else None,
        "coco_annotation_json_exists": coco_annotation_exists,
        "model_class_count": class_count,
        "test_manifest": relative_path(cfg.test_manifest, root),
        "test_manifest_availability": cfg.test_manifest.is_file(),
        "test_image_availability": manifest["image_exists"] > 0,
        "test_label_availability": manifest["label_exists"] > 0,
        "number_of_usable_test_images": manifest["usable_image_label_pairs"],
        "number_of_usable_test_labels": manifest["label_exists"],
        "number_of_test_instances": manifest["instances"],
        "no_validation_data_substitution": no_validation_substitution,
        "no_training_data_substitution": no_training_substitution,
        "full_project_test_set_ready": full_project_ready,
        "exact_2500_test_subset_manifest": exact_2500,
        "evaluation_readiness_status": status,
        "blocking_issues": blocking_issues,
    }


def write_evaluation_readiness_report(config: EvaluationConfig | None = None, output: Path | None = None) -> dict[str, Any]:
    """Write artifacts/evaluation_readiness.json."""
    root = project_root()
    report = create_evaluation_readiness_report(config)
    output_path = output or root / "artifacts" / "evaluation_readiness.json"
    ensure_dir(output_path.parent)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def validate_test_evaluation_inputs(config: EvaluationConfig, require_exact_2500: bool = False) -> dict[str, Any]:
    """Validate that evaluation targets a real test split and never val data."""
    require_file(config.model_path, "production checkpoint")
    require_file(config.dataset_yaml, "dataset YAML")
    require_file(config.test_manifest, "test manifest")
    dataset = load_top_level_yaml(config.dataset_yaml)
    if "test" not in dataset or not dataset.get("test"):
        raise ValueError(f"Dataset YAML must define a test split: {config.dataset_yaml}")
    if dataset.get("test") == dataset.get("val"):
        raise ValueError("Dataset YAML test split must not be the same path as validation split.")
    if "val" in config.test_manifest.name.lower() and "test" not in config.test_manifest.name.lower():
        raise ValueError(f"Refusing to use validation manifest as test manifest: {config.test_manifest}")
    manifest = inspect_manifest_availability(config.test_manifest)
    coco_annotation_json = config.coco_annotation_json if config.coco_annotation_json and config.coco_annotation_json.is_file() else None
    if manifest["usable_image_label_pairs"] == 0:
        raise ValueError(f"No usable test image/label pairs found in {config.test_manifest}")
    exact_2500 = find_exact_2500_test_manifest(project_root())
    if config.test_manifest.suffix.lower() == ".txt" and manifest["rows"] == 2500 and "test" in config.test_manifest.name.lower():
        exact_2500 = {
            "available": True,
            "path": relative_path(config.test_manifest),
            "candidate_files": exact_2500.get("candidate_files", []),
        }
    if require_exact_2500 and not exact_2500["available"]:
        raise ValueError("Exact 2,500-image faster-experiment test subset manifest is not available.")
    return {
        "dataset": dataset,
        "manifest": manifest,
        "exact_2500": exact_2500,
        "coco_annotation_json": coco_annotation_json,
    }


def _yaml_value(value: Any) -> str:
    text = str(value).replace("\\", "/")
    return f'"{text}"'


def prepare_evaluation_dataset_yaml(config: EvaluationConfig) -> Path:
    """Create a dataset YAML whose test split points at the configured test manifest."""
    dataset = load_top_level_yaml(config.dataset_yaml)
    if not dataset:
        raise ValueError(f"Could not load dataset YAML: {config.dataset_yaml}")
    output_yaml = config.output_directory / "evaluation_dataset.yaml"
    ensure_dir(output_yaml.parent)
    lines = [
        f"path: {_yaml_value(dataset.get('path', config.dataset_yaml.parent))}",
        f"train: {_yaml_value(dataset.get('train', 'images/train'))}",
        f"val: {_yaml_value(dataset.get('val', 'images/val'))}",
        f"test: {_yaml_value(config.test_manifest)}",
        f"nc: {int(dataset.get('nc', 80))}",
        "names:",
    ]
    names = dataset.get("names", {})
    if isinstance(names, dict):
        iterable = sorted(((int(key), value) for key, value in names.items()), key=lambda item: item[0])
    elif isinstance(names, list):
        iterable = list(enumerate(names))
    else:
        iterable = []
    for index, name in iterable:
        lines.append(f"  {index}: {name}")
    output_yaml.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_yaml


def build_yolov5_val_command(config: EvaluationConfig) -> list[str]:
    """Build a YOLOv5 v7.0 val.py command for test-set evaluation."""
    output_dir = config.output_directory
    evaluation_dataset_yaml = prepare_evaluation_dataset_yaml(config)
    command = [
        sys.executable,
        str(yolov5_root(project_root()) / "val.py"),
        "--weights",
        str(config.model_path),
        "--data",
        str(evaluation_dataset_yaml),
        "--task",
        "test",
        "--imgsz",
        str(config.image_size),
        "--batch-size",
        str(config.batch_size),
        "--conf-thres",
        str(config.confidence_threshold),
        "--iou-thres",
        str(config.iou_threshold),
        "--device",
        resolve_device(config.device),
        "--workers",
        str(config.workers),
        "--project",
        str(output_dir.parent),
        "--name",
        output_dir.name,
        "--exist-ok",
        "--verbose",
    ]
    if config.save_json:
        command.append("--save-json")
    if config.save_txt:
        command.extend(["--save-txt", "--save-conf"])
    if not config.save_plots:
        command.append("--verbose")
    return command


def expected_evaluation_artifacts(config: EvaluationConfig) -> dict[str, str]:
    """Return the standard artifact paths expected from a YOLOv5 evaluation run."""
    output_dir = config.output_directory
    return {
        "metrics_summary.json": str(output_dir / "metrics_summary.json"),
        "per_class_metrics.csv": str(output_dir / "per_class_metrics.csv"),
        "evaluation_metadata.json": str(output_dir / "evaluation_metadata.json"),
        "coco_eval": str(output_dir / "coco_eval"),
        "confusion_matrix.png": str(output_dir / "confusion_matrix.png"),
        "confusion_matrix_normalized.png": str(output_dir / "confusion_matrix_normalized.png"),
        "PR_curve.png": str(output_dir / "PR_curve.png"),
        "P_curve.png": str(output_dir / "P_curve.png"),
        "R_curve.png": str(output_dir / "R_curve.png"),
        "F1_curve.png": str(output_dir / "F1_curve.png"),
        "predictions.json": str(output_dir / "predictions.json"),
        "labels": str(output_dir / "labels"),
        "sample_predictions": str(output_dir / "sample_predictions"),
    }


def parse_yolov5_metrics(log_text: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Parse YOLOv5 verbose validation logs into aggregate and per-class metrics."""
    rows: list[dict[str, Any]] = []
    speed: dict[str, float] = {}
    for raw_line in log_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("Class "):
            continue
        speed_match = re.search(r"Speed:\s*([0-9.]+)ms pre-process,\s*([0-9.]+)ms inference,\s*([0-9.]+)ms NMS", line)
        if speed_match:
            speed = {
                "preprocess_time_ms_per_image": float(speed_match.group(1)),
                "inference_time_ms_per_image": float(speed_match.group(2)),
                "nms_time_ms_per_image": float(speed_match.group(3)),
            }
            continue
        parts = line.split()
        if len(parts) < 7:
            continue
        try:
            images = int(parts[-6])
            instances = int(parts[-5])
            precision = float(parts[-4])
            recall = float(parts[-3])
            map50 = float(parts[-2])
            map50_95 = float(parts[-1])
        except ValueError:
            continue
        class_name = " ".join(parts[:-6])
        if not class_name or class_name == "Class":
            continue
        rows.append(
            {
                "class_name": class_name,
                "images": images,
                "instances": instances,
                "precision": precision,
                "recall": recall,
                "mAP@0.5": map50,
                "mAP@0.5:0.95": map50_95,
            }
        )
    aggregate = next((row for row in rows if row["class_name"] == "all"), {})
    per_class = [row for row in rows if row["class_name"] != "all"]
    aggregate.update(speed)
    return aggregate, per_class


def clean_generated_evaluation_outputs(output_dir: Path) -> None:
    """Remove generated evaluation outputs that would become stale or append on rerun."""
    ensure_dir(output_dir)
    directories = [output_dir / "labels", output_dir / "sample_predictions", output_dir / "coco_eval"]
    files = [
        output_dir / "metrics_summary.json",
        output_dir / "per_class_metrics.csv",
        output_dir / "evaluation_metadata.json",
        output_dir / "test_results_summary.md",
        output_dir / "predictions.json",
        output_dir / "sample_prediction_summary.json",
        output_dir / "yolov5_val_stdout.log",
        output_dir / "yolov5_val_stderr.log",
        output_dir / "evaluate_cli_stdout.log",
        output_dir / "evaluate_cli_stderr.log",
        output_dir / "confusion_matrix.png",
        output_dir / "confusion_matrix_normalized.png",
        output_dir / "PR_curve.png",
        output_dir / "P_curve.png",
        output_dir / "R_curve.png",
        output_dir / "F1_curve.png",
    ]
    files.extend(output_dir.glob("*_predictions.json"))
    files.extend(output_dir.glob("val_batch*_labels.jpg"))
    files.extend(output_dir.glob("val_batch*_pred.jpg"))
    for directory in directories:
        if directory.exists():
            shutil.rmtree(directory)
    for path in files:
        if path.exists() and path.is_file():
            path.unlink()


def ensure_confusion_matrix_normalized_artifact(config: EvaluationConfig) -> str | None:
    """Create the expected normalized confusion matrix artifact for YOLOv5 v7.0."""
    source = config.output_directory / "confusion_matrix.png"
    target = config.output_directory / "confusion_matrix_normalized.png"
    if source.is_file() and not target.is_file():
        shutil.copy2(source, target)
        return "Copied from YOLOv5 v7.0 confusion_matrix.png, which is normalized by default."
    if target.is_file():
        return "YOLOv5 normalized confusion matrix artifact is available."
    return None


def prediction_image_id_to_coco_id(value: Any) -> int | None:
    """Convert a YOLOv5 prediction image_id to the original numeric COCO image ID."""
    if isinstance(value, int):
        return value
    text = str(value)
    if text.isdigit():
        return int(text)
    match = re.match(r"^(?:train2017|val2017)_(\d{12})$", text)
    if match:
        return int(match.group(1))
    return None


def convert_yolov5_predictions_to_coco(
    yolov5_predictions: Path,
    annotation_json: Path,
    output_predictions: Path,
) -> dict[str, Any]:
    """Convert YOLOv5 prediction JSON into original COCO image/category IDs."""
    annotation_data = json.loads(annotation_json.read_text(encoding="utf-8"))
    categories = sorted(annotation_data.get("categories", []), key=lambda item: int(item["id"]))
    class_to_category_id = {index: int(category["id"]) for index, category in enumerate(categories)}
    included_image_ids = {int(image["id"]) for image in annotation_data.get("images", [])}
    raw_predictions = json.loads(yolov5_predictions.read_text(encoding="utf-8"))
    converted: list[dict[str, Any]] = []
    dropped_unknown_image = 0
    dropped_unknown_category = 0
    dropped_invalid_bbox = 0

    for prediction in raw_predictions:
        image_id = prediction_image_id_to_coco_id(prediction.get("image_id"))
        if image_id not in included_image_ids:
            dropped_unknown_image += 1
            continue
        try:
            class_id = int(prediction.get("category_id"))
        except (TypeError, ValueError):
            dropped_unknown_category += 1
            continue
        category_id = class_to_category_id.get(class_id)
        if category_id is None:
            dropped_unknown_category += 1
            continue
        bbox = prediction.get("bbox", [])
        if not isinstance(bbox, list) or len(bbox) != 4 or float(bbox[2]) <= 0 or float(bbox[3]) <= 0:
            dropped_invalid_bbox += 1
            continue
        converted.append(
            {
                "image_id": image_id,
                "category_id": category_id,
                "bbox": [round(float(value), 3) for value in bbox],
                "score": round(float(prediction.get("score", 0.0)), 5),
            }
        )

    ensure_dir(output_predictions.parent)
    output_predictions.write_text(json.dumps(converted, indent=2) + "\n", encoding="utf-8")
    return {
        "source_prediction_path": relative_path(yolov5_predictions),
        "converted_prediction_path": relative_path(output_predictions),
        "source_prediction_count": len(raw_predictions),
        "converted_prediction_count": len(converted),
        "dropped_unknown_image_count": dropped_unknown_image,
        "dropped_unknown_category_count": dropped_unknown_category,
        "dropped_invalid_bbox_count": dropped_invalid_bbox,
    }


def run_coco_eval(config: EvaluationConfig, yolov5_predictions: Path | None) -> dict[str, Any]:
    """Run official pycocotools COCOeval as a separate post-processing step."""
    coco_dir = config.output_directory / "coco_eval"
    ensure_dir(coco_dir)
    stdout_path = coco_dir / "coco_eval_stdout.txt"
    summary_path = coco_dir / "coco_eval_summary.json"
    metadata_path = coco_dir / "coco_eval_metadata.json"
    converted_predictions_path = coco_dir / "coco_predictions.json"
    generated_at = datetime.now(timezone.utc).isoformat()

    metadata: dict[str, Any] = {
        "generated_at_utc": generated_at,
        "status": "not_run",
        "annotation_json": relative_path(config.coco_annotation_json) if config.coco_annotation_json else None,
        "yolov5_prediction_json": relative_path(yolov5_predictions) if yolov5_predictions else None,
        "converted_prediction_json": relative_path(converted_predictions_path),
        "summary_json": relative_path(summary_path),
        "stdout_log": relative_path(stdout_path),
    }

    if not config.coco_annotation_json or not config.coco_annotation_json.is_file():
        reason = "COCO annotation JSON is not configured or does not exist."
        metadata.update({"status": "failed", "failure_reason": reason})
        stdout_path.write_text(reason + "\n", encoding="utf-8")
        summary_path.write_text(json.dumps({"status": "failed", "failure_reason": reason}, indent=2), encoding="utf-8")
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return metadata
    if not yolov5_predictions or not yolov5_predictions.is_file():
        reason = "YOLOv5 prediction JSON was not generated."
        metadata.update({"status": "failed", "failure_reason": reason})
        stdout_path.write_text(reason + "\n", encoding="utf-8")
        summary_path.write_text(json.dumps({"status": "failed", "failure_reason": reason}, indent=2), encoding="utf-8")
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return metadata

    stdout_buffer = io.StringIO()
    try:
        conversion = convert_yolov5_predictions_to_coco(
            yolov5_predictions,
            config.coco_annotation_json,
            converted_predictions_path,
        )
        coco_module = require_python_package("pycocotools.coco", "pycocotools")
        cocoeval_module = require_python_package("pycocotools.cocoeval", "pycocotools")
        COCO = coco_module.COCO
        COCOeval = cocoeval_module.COCOeval
        with contextlib.redirect_stdout(stdout_buffer):
            coco_gt = COCO(str(config.coco_annotation_json))
            coco_dt = coco_gt.loadRes(str(converted_predictions_path))
            evaluator = COCOeval(coco_gt, coco_dt, "bbox")
            evaluator.params.imgIds = sorted(coco_gt.getImgIds())
            evaluator.evaluate()
            evaluator.accumulate()
            evaluator.summarize()
        metrics = {name: round(float(value), 6) for name, value in zip(COCOEVAL_METRIC_NAMES, evaluator.stats, strict=True)}
        summary = {
            "status": "completed",
            "metric_source": "pycocotools COCOeval bbox evaluation",
            "annotation_json": relative_path(config.coco_annotation_json),
            "prediction_json": relative_path(converted_predictions_path),
            **metrics,
        }
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        metadata.update(
            {
                "status": "completed",
                "conversion": conversion,
                "image_count": len(coco_gt.getImgIds()),
                "category_count": len(coco_gt.getCatIds()),
                "annotation_count": len(coco_gt.getAnnIds()),
                "metrics": metrics,
            }
        )
    except Exception as exc:
        metadata.update({"status": "failed", "failure_reason": str(exc)})
        summary_path.write_text(
            json.dumps({"status": "failed", "failure_reason": str(exc)}, indent=2),
            encoding="utf-8",
        )
    stdout_path.write_text(stdout_buffer.getvalue(), encoding="utf-8", errors="replace")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def load_json_if_available(path: Path) -> dict[str, Any] | None:
    """Load a JSON file if it exists."""
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def update_test_results_summary_with_coco(config: EvaluationConfig, coco_eval: dict[str, Any] | None) -> None:
    """Rewrite the Markdown summary with separate YOLOv5 and COCOeval sections."""
    metrics = load_json_if_available(config.output_directory / "metrics_summary.json") or {}
    per_class_rows: list[dict[str, Any]] = []
    per_class_path = config.output_directory / "per_class_metrics.csv"
    if per_class_path.is_file():
        with per_class_path.open("r", encoding="utf-8", newline="") as file:
            per_class_rows = list(csv.DictReader(file))
    strongest = sorted(per_class_rows, key=lambda row: float(row["mAP@0.5:0.95"]), reverse=True)[:5]
    weakest = sorted(per_class_rows, key=lambda row: float(row["mAP@0.5:0.95"]))[:5]
    coco_summary = load_json_if_available(config.output_directory / "coco_eval" / "coco_eval_summary.json") or {}
    annotation_report = load_json_if_available(project_root() / "artifacts" / "test_subset_coco_annotation_verification.json") or {}

    lines = [
        "# Test Subset 2500 Evaluation Results",
        "",
        "These are genuine labeled test-set metrics computed from `data/splits/test_subset_2500_seed42.txt`.",
        "YOLOv5 label-based metrics and official COCOeval metrics are reported separately.",
        "",
        "## YOLOv5 Label-Based Metrics",
        "",
        f"- Test images: {metrics.get('number_of_test_images')}",
        f"- Labeled instances: {metrics.get('number_of_labeled_instances')}",
        f"- Precision: {metrics.get('precision')}",
        f"- Recall: {metrics.get('recall')}",
        f"- mAP@0.5: {metrics.get('mAP@0.5')}",
        f"- mAP@0.5:0.95: {metrics.get('mAP@0.5:0.95')}",
        f"- Inference time: {metrics.get('inference_time_ms_per_image')} ms/image",
        f"- NMS time: {metrics.get('nms_time_ms_per_image')} ms/image",
        "",
        "## Strongest Classes by YOLOv5 mAP@0.5:0.95",
        "",
    ]
    lines.extend(f"- {row['class_name']}: {row['mAP@0.5:0.95']}" for row in strongest)
    lines.extend(["", "## Weakest Classes by YOLOv5 mAP@0.5:0.95", ""])
    lines.extend(f"- {row['class_name']}: {row['mAP@0.5:0.95']}" for row in weakest)
    lines.extend(["", "## Official COCOeval Metrics", ""])
    if coco_summary.get("status") == "completed":
        for metric_name in COCOEVAL_METRIC_NAMES:
            lines.append(f"- {metric_name}: {coco_summary.get(metric_name)}")
    else:
        lines.append(f"- Status: {coco_summary.get('status', 'not_run')}")
        lines.append(f"- Reason: {coco_summary.get('failure_reason', 'COCOeval did not complete')}")

    lines.extend(
        [
            "",
            "## Annotation JSON Verification",
            "",
            f"- Annotation JSON: {annotation_report.get('generated_json_path')}",
            f"- Verification status: {annotation_report.get('verification_status')}",
            f"- Selected images: {annotation_report.get('selected_image_count')}",
            f"- Selected annotations: {annotation_report.get('selected_annotation_count')}",
            f"- Categories: {annotation_report.get('category_count')}",
            f"- Excluded crowd annotations: {annotation_report.get('excluded_crowd_annotation_count')}",
            f"- Rejected invalid boxes: {annotation_report.get('rejected_invalid_box_count')}",
            f"- Annotation JSON SHA256: {annotation_report.get('generated_json_sha256')}",
            "",
            "## Metric-System Notes",
            "",
            "- YOLOv5 metrics are computed from YOLO label files inside `external/yolov5/val.py`.",
            "- COCOeval metrics are computed by pycocotools from COCO-format ground truth and converted prediction JSON.",
            "- Both systems use the same exact 2,500-image labeled test subset.",
            "- Small numerical differences are expected because COCOeval reports the official AP/AR protocol, including area ranges and max-detection settings.",
        ]
    )
    if coco_eval:
        lines.append(f"- COCOeval status: {coco_eval.get('status')}")
    (config.output_directory / "test_results_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_metrics_artifacts(config: EvaluationConfig, aggregate: dict[str, Any], per_class: list[dict[str, Any]]) -> None:
    """Write metrics summary, per-class CSV, and Markdown test summary."""
    metrics_summary = {
        "status": "completed",
        "metric_source": "YOLOv5 v7.0 genuine labeled test evaluation",
        "test_manifest": relative_path(config.test_manifest),
        "number_of_test_images": aggregate.get("images"),
        "number_of_labeled_instances": aggregate.get("instances"),
        "precision": aggregate.get("precision"),
        "recall": aggregate.get("recall"),
        "mAP@0.5": aggregate.get("mAP@0.5"),
        "mAP@0.5:0.95": aggregate.get("mAP@0.5:0.95"),
        "preprocess_time_ms_per_image": aggregate.get("preprocess_time_ms_per_image"),
        "inference_time_ms_per_image": aggregate.get("inference_time_ms_per_image"),
        "nms_time_ms_per_image": aggregate.get("nms_time_ms_per_image"),
    }
    (config.output_directory / "metrics_summary.json").write_text(json.dumps(metrics_summary, indent=2), encoding="utf-8")

    csv_path = config.output_directory / "per_class_metrics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        fieldnames = ["class_name", "images", "instances", "precision", "recall", "mAP@0.5", "mAP@0.5:0.95"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_class)

    strongest = sorted(per_class, key=lambda row: float(row["mAP@0.5:0.95"]), reverse=True)[:5]
    weakest = sorted(per_class, key=lambda row: float(row["mAP@0.5:0.95"]))[:5]
    lines = [
        "# Test Subset 2500 Evaluation Results",
        "",
        "These are genuine labeled test-set metrics computed from `data/splits/test_subset_2500_seed42.txt`.",
        "",
        f"- Test images: {metrics_summary['number_of_test_images']}",
        f"- Labeled instances: {metrics_summary['number_of_labeled_instances']}",
        f"- Precision: {metrics_summary['precision']}",
        f"- Recall: {metrics_summary['recall']}",
        f"- mAP@0.5: {metrics_summary['mAP@0.5']}",
        f"- mAP@0.5:0.95: {metrics_summary['mAP@0.5:0.95']}",
        f"- Inference time: {metrics_summary['inference_time_ms_per_image']} ms/image",
        f"- NMS time: {metrics_summary['nms_time_ms_per_image']} ms/image",
        "",
        "## Strongest Classes by mAP@0.5:0.95",
        "",
    ]
    lines.extend(f"- {row['class_name']}: {row['mAP@0.5:0.95']}" for row in strongest)
    lines.extend(["", "## Weakest Classes by mAP@0.5:0.95", ""])
    lines.extend(f"- {row['class_name']}: {row['mAP@0.5:0.95']}" for row in weakest)
    (config.output_directory / "test_results_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_evaluation_sample_predictions(config: EvaluationConfig) -> dict[str, Any]:
    """Generate annotated sample predictions inside the evaluation output directory."""
    from src.sample_predictions import generate_sample_predictions

    return generate_sample_predictions(
        max_images=config.maximum_sample_predictions,
        output_dir=config.output_directory / "sample_predictions",
        summary_path=config.output_directory / "sample_prediction_summary.json",
        device=config.device,
    )


def write_not_run_metadata(config: EvaluationConfig, reason: str) -> dict[str, Any]:
    """Write evaluation metadata when genuine evaluation is not run."""
    ensure_dir(config.output_directory)
    expected = expected_evaluation_artifacts(config)
    missing = {
        name: {"path": path, "generated": False, "reason": reason}
        for name, path in expected.items()
        if name != "evaluation_metadata.json"
    }
    notes = [
        "YOLOv5 label-based metrics are genuine only for the configured YOLO test split.",
        "Official COCOeval metrics are recorded separately under coco_eval when pycocotools completes.",
    ]
    if confusion_matrix_note:
        notes.append(confusion_matrix_note)

    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "not_run",
        "reason": reason,
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()},
        "artifacts": missing,
        "notes": [
            "No test metrics are reported because evaluation did not run.",
            "Validation metrics and unlabeled sample predictions are not test-set metrics.",
        ],
    }
    metadata_path = config.output_directory / "evaluation_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def run_evaluation(config: EvaluationConfig, require_exact_2500: bool = False) -> dict[str, Any]:
    """Run genuine YOLOv5 evaluation on the configured test set."""
    validation = validate_test_evaluation_inputs(config, require_exact_2500=require_exact_2500)
    ensure_dir(config.output_directory)
    clean_generated_evaluation_outputs(config.output_directory)
    command = build_yolov5_val_command(config)
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("YOLOv5_AUTOINSTALL", "False")
    completed = subprocess.run(
        command,
        cwd=project_root(),
        check=False,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        env=env,
    )
    stdout_path = config.output_directory / "yolov5_val_stdout.log"
    stderr_path = config.output_directory / "yolov5_val_stderr.log"
    stdout_path.write_text(completed.stdout or "", encoding="utf-8", errors="replace")
    stderr_path.write_text(completed.stderr or "", encoding="utf-8", errors="replace")
    if completed.returncode != 0:
        metadata = write_not_run_metadata(
            config,
            f"YOLOv5 val.py failed with exit code {completed.returncode}; see log files.",
        )
        metadata["command"] = command
        (config.output_directory / "evaluation_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        raise RuntimeError(f"YOLOv5 validation failed; see {stdout_path} and {stderr_path}")

    predictions = next(config.output_directory.glob("*_predictions.json"), None)
    if predictions and not (config.output_directory / "predictions.json").exists():
        shutil.copy2(predictions, config.output_directory / "predictions.json")
    yolov5_prediction_path = config.output_directory / "predictions.json"
    if not yolov5_prediction_path.is_file():
        yolov5_prediction_path = predictions if predictions and predictions.is_file() else None

    aggregate, per_class = parse_yolov5_metrics(completed.stdout + "\n" + completed.stderr)
    write_metrics_artifacts(config, aggregate, per_class)
    sample_summary = generate_evaluation_sample_predictions(config) if config.save_sample_predictions else None
    coco_eval = run_coco_eval(config, yolov5_prediction_path)
    update_test_results_summary_with_coco(config, coco_eval)
    confusion_matrix_note = ensure_confusion_matrix_normalized_artifact(config)

    artifact_status = {}
    for name, path_text in expected_evaluation_artifacts(config).items():
        path = Path(path_text)
        artifact_status[name] = {
            "path": relative_path(path),
            "generated": path.exists(),
            "reason": None if path.exists() else "YOLOv5 did not generate this artifact or post-processing could not create it.",
        }

    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "completed",
        "command": command,
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()},
        "manifest": validation["manifest"],
        "exact_2500_test_subset_manifest": validation["exact_2500"],
        "metrics_summary": aggregate,
        "per_class_count": len(per_class),
        "coco_eval": coco_eval,
        "sample_prediction_summary": sample_summary,
        "artifacts": artifact_status,
        "stdout_log": relative_path(stdout_path),
        "stderr_log": relative_path(stderr_path),
        "notes": notes,
    }
    (config.output_directory / "evaluation_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def main() -> int:
    """CLI entrypoint for evaluation readiness and optional test evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate production YOLOv5 checkpoint on a genuine test split.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--test-manifest", type=Path)
    parser.add_argument("--device")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--imgsz", type=int)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--readiness-only", action="store_true")
    parser.add_argument("--require-exact-2500", action="store_true")
    parser.add_argument("--write-not-run-metadata", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)
    config = load_evaluation_config(
        args.config,
        model_path=args.weights,
        dataset_yaml=args.data,
        test_manifest=args.test_manifest,
        device=args.device,
        batch_size=args.batch_size,
        image_size=args.imgsz,
        output_directory=args.output_dir,
        workers=args.workers,
    )
    readiness = write_evaluation_readiness_report(config)
    print(json.dumps(readiness, indent=2))
    if args.readiness_only:
        return 0
    if args.write_not_run_metadata:
        write_not_run_metadata(config, "Evaluation was intentionally not run by CLI request.")
        return 0
    run_evaluation(config, require_exact_2500=args.require_exact_2500)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
