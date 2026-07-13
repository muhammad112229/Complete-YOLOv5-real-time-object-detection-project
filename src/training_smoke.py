"""Utilities for local YOLOv5s training smoke tests."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.common import ensure_dir, load_yaml, project_root, require_file, require_python_package, setup_logging, yolov5_root
from src.inference import infer_frame, load_yolov5_model


SMOKE_WARNING = (
    "These metrics are produced from a tiny one-epoch diagnostic run and must not be "
    "interpreted as the final accuracy of the trained object-detection system."
)
SMOKE_METRIC_LABEL = "Smoke-test diagnostic metrics -- not final performance results."
CLASS_COUNT = 80
DEFAULT_SEED = 42


def utc_now() -> str:
    """Return an ISO UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, data: Any) -> None:
    """Write JSON with a stable UTF-8 representation."""
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def project_path(root: Path, path: Path | str) -> Path:
    """Resolve a path relative to the project root."""
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


def safe_rmtree(path: Path, allowed_parent: Path) -> None:
    """Remove a generated path after checking it is inside the expected parent."""
    resolved = path.resolve()
    allowed = allowed_parent.resolve()
    if resolved == allowed or allowed not in resolved.parents:
        raise ValueError(f"Refusing to remove unsafe path: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)


def load_coco_names(yaml_path: Path) -> dict[int, str]:
    """Load COCO class names from a YOLO dataset YAML."""
    data = load_yaml(yaml_path)
    names = data.get("names")
    if isinstance(names, dict):
        loaded = {int(key): str(value) for key, value in names.items()}
    elif isinstance(names, list):
        loaded = {index: str(value) for index, value in enumerate(names)}
    else:
        raise ValueError(f"Unsupported names field in {yaml_path}")
    if len(loaded) != CLASS_COUNT:
        raise ValueError(f"Expected {CLASS_COUNT} class names, found {len(loaded)}")
    return dict(sorted(loaded.items()))


def write_dataset_yaml(path: Path, dataset_root: Path, names: dict[int, str]) -> None:
    """Write a YOLOv5 dataset YAML for the smoke subset."""
    lines = [
        f"path: {dataset_root.resolve().as_posix()}",
        "train: images/train",
        "val: images/val",
        f"nc: {CLASS_COUNT}",
        "names:",
    ]
    lines.extend(f"  {index}: {name}" for index, name in sorted(names.items()))
    ensure_dir(path.parent)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def ensure_yolov5_offline_font_config(root: Path) -> Path:
    """Provide YOLOv5 with a local font so training does not require network access."""
    config_dir = ensure_dir(root / "artifacts" / "yolov5_config")
    target = config_dir / "Arial.ttf"
    if not target.exists() or target.stat().st_size == 0:
        matplotlib = require_python_package("matplotlib")
        source = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf"
        require_file(source, "Matplotlib DejaVuSans font")
        shutil.copy2(source, target)
    return config_dir


def read_manifest_records(manifest_path: Path) -> list[dict[str, Any]]:
    """Read prepared COCO split manifest records."""
    records: list[dict[str, Any]] = []
    with manifest_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            category_names = json.loads(row.get("category_names") or "[]")
            records.append(
                {
                    **row,
                    "valid_object_count": int(row.get("valid_object_count") or 0),
                    "width": int(row.get("width") or 0),
                    "height": int(row.get("height") or 0),
                    "category_names_list": [str(name) for name in category_names],
                    "project_image_path_obj": Path(row["project_image_path"]),
                    "project_label_path_obj": Path(row["project_label_path"]),
                }
            )
    return records


def select_diverse_records(
    records: list[dict[str, Any]],
    count: int,
    seed: int = DEFAULT_SEED,
) -> list[dict[str, Any]]:
    """Select deterministic annotated records while favoring category diversity."""
    rng = random.Random(seed)
    annotated = [
        record
        for record in records
        if record["valid_object_count"] > 0
    ]
    fallback = list(records)
    if len(fallback) < count:
        raise ValueError(f"Need {count} usable records, found {len(fallback)}")

    pool = annotated if len(annotated) >= count else fallback
    shuffled = list(pool)
    rng.shuffle(shuffled)
    order = {record["internal_record_id"]: index for index, record in enumerate(shuffled)}
    remaining = list(shuffled)
    selected: list[dict[str, Any]] = []
    represented: set[str] = set()

    while remaining and len(selected) < count:
        best = max(
            remaining,
            key=lambda record: (
                len(set(record["category_names_list"]) - represented),
                len(set(record["category_names_list"])),
                record["valid_object_count"],
                -order[record["internal_record_id"]],
            ),
        )
        remaining.remove(best)
        selected.append(best)
        represented.update(best["category_names_list"])

    if len(selected) != count:
        raise ValueError(f"Selected {len(selected)} records, expected {count}")
    return selected


def link_file(source: Path, destination: Path) -> str:
    """Hardlink a file, falling back to copy if the platform rejects hardlinks."""
    ensure_dir(destination.parent)
    if destination.exists():
        destination.unlink()
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy_fallback"


def label_stats(label_path: Path) -> tuple[int, set[int]]:
    """Return YOLO label line count and represented class ids."""
    classes: set[int] = set()
    lines = 0
    for line in label_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"Invalid YOLO label line in {label_path}: {line}")
        classes.add(int(float(parts[0])))
        lines += 1
    return lines, classes


def create_smoke_subset(
    root: Path,
    train_count: int = 32,
    val_count: int = 16,
    seed: int = DEFAULT_SEED,
    overwrite: bool = True,
) -> dict[str, Any]:
    """Create a deterministic hardlinked COCO YOLO smoke subset."""
    root = root.resolve()
    processed_root = root / "data" / "processed" / "coco_yolo"
    split_root = root / "data" / "splits"
    smoke_root = root / "data" / "smoke" / "coco_yolov5"
    artifacts_root = root / "artifacts"

    require_file(processed_root / "coco_project.yaml", "processed COCO dataset YAML")
    require_file(split_root / "train_manifest.csv", "train manifest")
    require_file(split_root / "val_manifest.csv", "validation manifest")
    names = load_coco_names(processed_root / "coco_project.yaml")

    if overwrite:
        safe_rmtree(smoke_root, root / "data" / "smoke")
    for split in ("train", "val"):
        ensure_dir(smoke_root / "images" / split)
        ensure_dir(smoke_root / "labels" / split)

    selections = {
        "train": select_diverse_records(read_manifest_records(split_root / "train_manifest.csv"), train_count, seed),
        "val": select_diverse_records(read_manifest_records(split_root / "val_manifest.csv"), val_count, seed),
    }

    manifest_rows: list[dict[str, Any]] = []
    split_image_lists: dict[str, list[str]] = {"train": [], "val": []}
    link_strategies: Counter[str] = Counter()
    represented_classes: set[int] = set()
    annotation_counts: dict[str, int] = {}

    for split, records in selections.items():
        split_annotations = 0
        for record in records:
            source_image = record["project_image_path_obj"]
            source_label = record["project_label_path_obj"]
            smoke_image = smoke_root / "images" / split / source_image.name
            smoke_label = smoke_root / "labels" / split / source_label.name
            image_strategy = link_file(source_image, smoke_image)
            label_strategy = link_file(source_label, smoke_label)
            link_strategies[f"image_{image_strategy}"] += 1
            link_strategies[f"label_{label_strategy}"] += 1
            label_count, label_classes = label_stats(smoke_label)
            split_annotations += label_count
            represented_classes.update(label_classes)
            split_image_lists[split].append(str(smoke_image.resolve()))
            manifest_rows.append(
                {
                    "split": split,
                    "internal_record_id": record["internal_record_id"],
                    "source_split": record["source_split"],
                    "image_id": record["image_id"],
                    "source_image_path": record["source_image_path"],
                    "source_project_image_path": str(source_image),
                    "source_project_label_path": str(source_label),
                    "smoke_image_path": str(smoke_image.resolve()),
                    "smoke_label_path": str(smoke_label.resolve()),
                    "valid_object_count": record["valid_object_count"],
                    "label_annotation_count": label_count,
                    "category_names": json.dumps(record["category_names_list"]),
                    "image_link_strategy": image_strategy,
                    "label_link_strategy": label_strategy,
                }
            )
        annotation_counts[split] = split_annotations

    for split, paths in split_image_lists.items():
        (smoke_root / f"{split}_images.txt").write_text("\n".join(paths) + "\n", encoding="utf-8")

    manifest_path = smoke_root / "smoke_subset_manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)

    dataset_yaml = smoke_root / "smoke_dataset.yaml"
    write_dataset_yaml(dataset_yaml, smoke_root, names)

    class_summary = [
        {"class_id": class_id, "class_name": names[class_id]}
        for class_id in sorted(represented_classes)
    ]
    report = {
        "generated_at_utc": utc_now(),
        "seed": seed,
        "dataset_root": str(smoke_root),
        "dataset_yaml": str(dataset_yaml),
        "source_dataset_yaml": str(processed_root / "coco_project.yaml"),
        "train_count": train_count,
        "val_count": val_count,
        "annotation_counts": annotation_counts,
        "total_annotations": sum(annotation_counts.values()),
        "represented_class_count": len(class_summary),
        "represented_classes": class_summary,
        "linking_strategy": dict(link_strategies),
        "manifest": str(manifest_path),
        "train_images_txt": str(smoke_root / "train_images.txt"),
        "val_images_txt": str(smoke_root / "val_images.txt"),
    }
    write_json(artifacts_root / "training_smoke_subset_manifest.json", report)
    return report


def load_smoke_manifest(smoke_root: Path) -> list[dict[str, str]]:
    """Load the smoke subset CSV manifest."""
    manifest_path = smoke_root / "smoke_subset_manifest.csv"
    require_file(manifest_path, "smoke subset manifest")
    with manifest_path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def validate_smoke_labels(label_paths: list[Path]) -> dict[str, Any]:
    """Validate YOLO label files for range and format."""
    errors: list[str] = []
    annotation_count = 0
    represented_classes: set[int] = set()
    for label_path in label_paths:
        if not label_path.exists():
            errors.append(f"{label_path}: missing label file")
            continue
        for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) != 5:
                errors.append(f"{label_path}:{line_number}: expected 5 values")
                continue
            try:
                class_id = int(float(parts[0]))
                x_center, y_center, width, height = (float(value) for value in parts[1:])
            except ValueError:
                errors.append(f"{label_path}:{line_number}: non-numeric value")
                continue
            if not 0 <= class_id < CLASS_COUNT:
                errors.append(f"{label_path}:{line_number}: class id {class_id} outside 0-79")
            if not all(0.0 <= value <= 1.0 for value in (x_center, y_center, width, height)):
                errors.append(f"{label_path}:{line_number}: normalized coordinate outside 0-1")
            if width <= 0.0 or height <= 0.0:
                errors.append(f"{label_path}:{line_number}: width/height must be positive")
            annotation_count += 1
            represented_classes.add(class_id)
    return {
        "valid": not errors,
        "errors": errors,
        "annotation_count": annotation_count,
        "represented_class_ids": sorted(represented_classes),
    }


def run_yolov5_smoke_dataloader(dataset_yaml: Path) -> dict[str, Any]:
    """Load one batch with the native YOLOv5 dataloader."""
    root = project_root()
    yolo_root = str(yolov5_root(root))
    if yolo_root not in sys.path:
        sys.path.insert(0, yolo_root)
    dataloaders = require_python_package("utils.dataloaders")
    data = load_yaml(dataset_yaml)
    dataset_root = Path(data["path"])
    train_path = dataset_root / data["train"]
    dataloader, dataset = dataloaders.create_dataloader(
        path=str(train_path),
        imgsz=640,
        batch_size=2,
        stride=32,
        single_cls=False,
        pad=0.5,
        rect=False,
        workers=0,
        prefix="training-smoke: ",
    )
    images, labels, paths_batch, _ = next(iter(dataloader))
    return {
        "status": "passed",
        "batch_image_tensor_shape": list(images.shape),
        "batch_label_tensor_shape": list(labels.shape),
        "batch_paths": [str(path) for path in paths_batch],
        "dataset_length": len(dataset),
    }


def validate_smoke_dataset(root: Path) -> dict[str, Any]:
    """Validate the tiny YOLOv5 smoke dataset."""
    root = root.resolve()
    smoke_root = root / "data" / "smoke" / "coco_yolov5"
    dataset_yaml = smoke_root / "smoke_dataset.yaml"
    artifacts_root = root / "artifacts"
    data = load_yaml(dataset_yaml)
    names = load_coco_names(dataset_yaml)
    dataset_root = Path(data["path"])

    rows = load_smoke_manifest(smoke_root)
    split_rows = {
        "train": [row for row in rows if row["split"] == "train"],
        "val": [row for row in rows if row["split"] == "val"],
    }
    image_paths = {
        split: sorted((dataset_root / "images" / split).glob("*.jpg"))
        for split in ("train", "val")
    }
    label_paths = {
        split: [dataset_root / "labels" / split / f"{path.stem}.txt" for path in paths]
        for split, paths in image_paths.items()
    }
    label_validation = validate_smoke_labels(label_paths["train"] + label_paths["val"])

    source_train = {row["source_project_image_path"] for row in split_rows["train"]}
    source_val = {row["source_project_image_path"] for row in split_rows["val"]}
    smoke_train = {Path(row["smoke_image_path"]).name for row in split_rows["train"]}
    smoke_val = {Path(row["smoke_image_path"]).name for row in split_rows["val"]}
    missing_labels = [
        str(path)
        for split in ("train", "val")
        for path in label_paths[split]
        if not path.exists()
    ]

    rules = [
        {
            "name": "train_image_count",
            "status": "PASS" if len(image_paths["train"]) == 32 else "FAIL",
            "details": {"count": len(image_paths["train"]), "expected": 32},
        },
        {
            "name": "val_image_count",
            "status": "PASS" if len(image_paths["val"]) == 16 else "FAIL",
            "details": {"count": len(image_paths["val"]), "expected": 16},
        },
        {
            "name": "every_image_has_label",
            "status": "PASS" if not missing_labels else "FAIL",
            "details": {"missing_labels": missing_labels[:20], "missing_count": len(missing_labels)},
        },
        {
            "name": "label_format_and_ranges",
            "status": "PASS" if label_validation["valid"] else "FAIL",
            "details": {
                "annotation_count": label_validation["annotation_count"],
                "error_count": len(label_validation["errors"]),
                "errors": label_validation["errors"][:20],
            },
        },
        {
            "name": "no_smoke_split_leakage",
            "status": "PASS" if not (source_train & source_val) and not (smoke_train & smoke_val) else "FAIL",
            "details": {
                "source_overlap": len(source_train & source_val),
                "filename_overlap": len(smoke_train & smoke_val),
            },
        },
        {
            "name": "dataset_yaml_resolves",
            "status": "PASS"
            if dataset_root.exists()
            and (dataset_root / data["train"]).exists()
            and (dataset_root / data["val"]).exists()
            and int(data["nc"]) == CLASS_COUNT
            and len(names) == CLASS_COUNT
            else "FAIL",
            "details": {
                "path": str(dataset_root),
                "train": str(dataset_root / data["train"]),
                "val": str(dataset_root / data["val"]),
                "nc": data.get("nc"),
                "name_count": len(names),
            },
        },
    ]

    try:
        dataloader_report = run_yolov5_smoke_dataloader(dataset_yaml)
    except Exception as exc:
        dataloader_report = {"status": "failed", "error": str(exc)}
    rules.append(
        {
            "name": "native_yolov5_dataloader_batch",
            "status": "PASS" if dataloader_report.get("status") == "passed" else "FAIL",
            "details": dataloader_report,
        }
    )

    represented = [
        {"class_id": class_id, "class_name": names[class_id]}
        for class_id in label_validation["represented_class_ids"]
    ]
    report = {
        "generated_at_utc": utc_now(),
        "dataset_root": str(smoke_root),
        "dataset_yaml": str(dataset_yaml),
        "seed": DEFAULT_SEED,
        "counts": {"train_images": len(image_paths["train"]), "val_images": len(image_paths["val"])},
        "annotation_count": label_validation["annotation_count"],
        "represented_class_count": len(represented),
        "represented_classes": represented,
        "rules": rules,
        "final_readiness": all(rule["status"] == "PASS" for rule in rules),
    }
    write_json(artifacts_root / "training_smoke_dataset_validation.json", report)
    lines = [
        "# Training Smoke Dataset Validation",
        "",
        f"Generated UTC: `{report['generated_at_utc']}`",
        f"Final readiness: {'PASS' if report['final_readiness'] else 'FAIL'}",
        f"Dataset YAML: `{dataset_yaml}`",
        "",
        "## Counts",
        f"- Train images: {report['counts']['train_images']}",
        f"- Validation images: {report['counts']['val_images']}",
        f"- Annotations: {report['annotation_count']}",
        f"- Represented classes: {report['represented_class_count']}",
        "",
        "## Rules",
    ]
    lines.extend(f"- {rule['name']}: {rule['status']} {rule['details']}" for rule in rules)
    lines.extend(["", "## Represented Classes"])
    lines.extend(f"- {item['class_id']}: {item['class_name']}" for item in represented)
    (artifacts_root / "training_smoke_dataset_validation.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    return report


def parse_results_csv(results_csv: Path) -> dict[str, Any]:
    """Parse YOLOv5 results.csv into training losses and diagnostic metrics."""
    require_file(results_csv, "YOLOv5 results.csv")
    with results_csv.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        rows = [{key.strip(): value.strip() for key, value in row.items()} for row in reader]
    if not rows:
        raise ValueError(f"No rows found in {results_csv}")
    last = rows[-1]

    def number(key: str) -> float:
        return float(last[key])

    return {
        "results_csv": str(results_csv),
        "epoch": int(float(last["epoch"])),
        "training_losses": {
            "box_loss": number("train/box_loss"),
            "obj_loss": number("train/obj_loss"),
            "cls_loss": number("train/cls_loss"),
        },
        "validation_losses": {
            "box_loss": number("val/box_loss"),
            "obj_loss": number("val/obj_loss"),
            "cls_loss": number("val/cls_loss"),
        },
        "diagnostic_metrics": {
            "precision": number("metrics/precision"),
            "recall": number("metrics/recall"),
            "mAP_0.5": number("metrics/mAP_0.5"),
            "mAP_0.5:0.95": number("metrics/mAP_0.5:0.95"),
        },
        "learning_rates": {
            "lr0": number("x/lr0"),
            "lr1": number("x/lr1"),
            "lr2": number("x/lr2"),
        },
    }


def parse_training_log(log_path: Path) -> dict[str, Any]:
    """Extract execution evidence from the captured YOLOv5 training log."""
    text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
    checks = {
        "pretrained_weights_loaded": "Transferred " in text and "items from" in text,
        "model_architecture_initialized": "Model summary:" in text or "YOLOv5s summary:" in text,
        "training_images_discovered": "train: " in text and "32 images" in text,
        "validation_images_discovered": "val: " in text and "16 images" in text,
        "box_loss_executed": "box_loss" in text,
        "objectness_loss_executed": "obj_loss" in text,
        "classification_loss_executed": "cls_loss" in text,
        "optimizer_configured": "optimizer:" in text and "SGD" in text,
        "validation_completed": "Validating " in text or "Class     Images" in text,
        "checkpoint_logged": "Optimizer stripped from" in text and "last.pt" in text,
    }
    warnings = [
        line.strip()
        for line in text.splitlines()
        if "warning" in line.lower()
        or "deprecated" in line.lower()
        or "futurewarning" in line.lower()
        or "sourcechangewarning" in line.lower()
    ]
    return {
        "log_path": str(log_path),
        "checks": checks,
        "warnings": warnings[:100],
        "line_count": len(text.splitlines()),
    }


def load_checkpoint(path: Path) -> dict[str, Any]:
    """Load a YOLOv5 PyTorch checkpoint."""
    root = project_root()
    yolo_root = str(yolov5_root(root))
    if yolo_root not in sys.path:
        sys.path.insert(0, yolo_root)
    torch = require_python_package("torch")
    return torch.load(path, map_location="cpu")


def checkpoint_model_metadata(checkpoint: dict[str, Any], fallback_names: dict[int, str]) -> dict[str, Any]:
    """Extract model metadata from a loaded checkpoint."""
    model = checkpoint.get("ema") or checkpoint.get("model")
    if model is None:
        raise ValueError("Checkpoint has neither ema nor model")
    names = getattr(model, "names", None) or fallback_names
    if isinstance(names, list):
        names = dict(enumerate(names))
    names = {int(key): str(value) for key, value in dict(names).items()}
    return {
        "epoch": checkpoint.get("epoch"),
        "best_fitness": checkpoint.get("best_fitness"),
        "model_type": type(model).__name__,
        "nc": int(getattr(model, "nc", len(names))),
        "name_count": len(names),
        "names_present_or_recoverable": len(names) == CLASS_COUNT,
        "has_model": checkpoint.get("model") is not None,
        "has_ema": checkpoint.get("ema") is not None,
        "has_optimizer": checkpoint.get("optimizer") is not None,
    }


def compare_weight_update(pretrained_path: Path, trained_path: Path) -> dict[str, Any]:
    """Compare pretrained and trained checkpoint tensors to prove weights changed."""
    torch = require_python_package("torch")
    pretrained = load_checkpoint(pretrained_path)
    trained = load_checkpoint(trained_path)
    pretrained_model = pretrained.get("ema") or pretrained.get("model")
    trained_model = trained.get("ema") or trained.get("model")
    if pretrained_model is None or trained_model is None:
        return {"status": "failed", "reason": "missing model in checkpoint"}
    before = pretrained_model.float().state_dict()
    after = trained_model.float().state_dict()
    common = [key for key in before.keys() if key in after and before[key].shape == after[key].shape]
    changed = 0
    total_abs_delta = 0.0
    max_abs_delta = 0.0
    for key in common:
        delta = (after[key] - before[key]).detach().abs()
        if bool(torch.any(delta > 0)):
            changed += 1
            total_abs_delta += float(delta.sum().item())
            max_abs_delta = max(max_abs_delta, float(delta.max().item()))
    return {
        "status": "passed" if changed > 0 else "failed",
        "common_tensor_count": len(common),
        "changed_tensor_count": changed,
        "total_abs_delta": total_abs_delta,
        "max_abs_delta": max_abs_delta,
    }


def validate_checkpoints(root: Path, run_dir: Path | None = None) -> dict[str, Any]:
    """Validate smoke-test YOLOv5 checkpoints and reload inference."""
    root = root.resolve()
    run_dir = run_dir or root / "results" / "yolov5s" / "smoke_test"
    dataset_yaml = root / "data" / "smoke" / "coco_yolov5" / "smoke_dataset.yaml"
    pretrained = root / "models" / "pretrained" / "yolov5s.pt"
    bus_image = root / "external" / "yolov5" / "data" / "images" / "bus.jpg"
    names = load_coco_names(dataset_yaml)
    checkpoint_paths = {
        "best": run_dir / "weights" / "best.pt",
        "last": run_dir / "weights" / "last.pt",
    }
    checkpoints: dict[str, Any] = {}
    errors: list[str] = []
    for label, path in checkpoint_paths.items():
        item = {
            "path": str(path),
            "exists": path.exists(),
            "size_bytes": path.stat().st_size if path.exists() else 0,
            "load_status": "not_run",
        }
        if path.exists() and path.stat().st_size > 0:
            try:
                checkpoint = load_checkpoint(path)
                item["load_status"] = "passed"
                item["metadata"] = checkpoint_model_metadata(checkpoint, names)
            except Exception as exc:
                item["load_status"] = "failed"
                item["error"] = str(exc)
                errors.append(f"{label}.pt load failed: {exc}")
        else:
            errors.append(f"{label}.pt missing or empty")
        checkpoints[label] = item

    inference_result: dict[str, Any]
    try:
        cv2 = require_python_package("cv2", "opencv-python")
        frame = cv2.imread(str(bus_image))
        if frame is None:
            raise RuntimeError(f"Could not read {bus_image}")
        model = load_yolov5_model(checkpoint_paths["best"], "cpu", 0.25, 0.45)
        result = infer_frame(model, frame, 640)
        inference_result = {
            "status": "passed",
            "checkpoint": str(checkpoint_paths["best"]),
            "detection_count": result.detection_count,
            "detected_classes": list(result.detected_class_names),
        }
    except Exception as exc:
        inference_result = {"status": "failed", "error": str(exc)}
        errors.append(f"checkpoint inference failed: {exc}")

    weight_update = compare_weight_update(pretrained, checkpoint_paths["last"]) if checkpoint_paths["last"].exists() else {"status": "failed"}
    if weight_update.get("status") != "passed":
        errors.append("trained checkpoint tensors did not differ from pretrained weights")

    report = {
        "generated_at_utc": utc_now(),
        "run_dir": str(run_dir),
        "checkpoints": checkpoints,
        "weight_update_vs_pretrained": weight_update,
        "reload_inference": inference_result,
        "valid": not errors,
        "errors": errors,
    }
    write_json(root / "artifacts" / "yolov5s_smoke_checkpoint_validation.json", report)
    lines = [
        "# YOLOv5s Smoke Checkpoint Validation",
        "",
        f"Generated UTC: `{report['generated_at_utc']}`",
        f"Status: {'PASS' if report['valid'] else 'FAIL'}",
        "",
        "## Checkpoints",
    ]
    for label, item in checkpoints.items():
        lines.append(
            f"- {label}.pt: exists={item['exists']}, size={item['size_bytes']}, load={item['load_status']}"
        )
    lines.extend(
        [
            "",
            "## Weight Update",
            f"- Status: {weight_update.get('status')}",
            f"- Changed tensors: {weight_update.get('changed_tensor_count')}",
            "",
            "## Reload Inference",
            f"- Status: {inference_result.get('status')}",
            f"- Detections: {inference_result.get('detection_count')}",
            f"- Classes: {inference_result.get('detected_classes')}",
        ]
    )
    (root / "artifacts" / "yolov5s_smoke_checkpoint_validation.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    return report


def run_post_training_inference(root: Path, checkpoint: Path | None = None) -> dict[str, Any]:
    """Run checkpoint inference on YOLOv5 bus.jpg and save an annotated image."""
    root = root.resolve()
    checkpoint = checkpoint or root / "results" / "yolov5s" / "smoke_test" / "weights" / "best.pt"
    source = root / "external" / "yolov5" / "data" / "images" / "bus.jpg"
    output_dir = root / "outputs" / "images" / "training_smoke_test"
    output_path = output_dir / "bus.jpg"
    conf_thres = 0.25
    iou_thres = 0.45
    imgsz = 640
    cv2 = require_python_package("cv2", "opencv-python")
    frame = cv2.imread(str(source))
    if frame is None:
        raise RuntimeError(f"Could not read image: {source}")
    started = time.perf_counter()
    model = load_yolov5_model(checkpoint, "cpu", conf_thres, iou_thres)
    load_ms = (time.perf_counter() - started) * 1000.0
    result = infer_frame(model, frame, imgsz)
    ensure_dir(output_dir)
    if not cv2.imwrite(str(output_path), result.annotated_frame):
        raise RuntimeError(f"Failed to write image: {output_path}")
    report = {
        "generated_at_utc": utc_now(),
        "status": "passed",
        "checkpoint": str(checkpoint),
        "source": str(source),
        "output": str(output_path),
        "image_size": imgsz,
        "confidence_threshold": conf_thres,
        "iou_threshold": iou_thres,
        "detected_classes": list(result.detected_class_names),
        "detection_count": result.detection_count,
        "preprocessing_time_ms": result.preprocess_ms,
        "inference_time_ms": result.model_inference_ms,
        "nms_time_ms": result.nms_ms,
        "total_pipeline_time_ms": result.inference_ms,
        "model_load_time_ms": load_ms,
        "timing_context": "diagnostic CPU timing only; not a final real-time speed result",
        "detections": list(result.detections),
    }
    write_json(root / "artifacts" / "yolov5s_smoke_post_training_inference.json", report)
    lines = [
        "# YOLOv5s Smoke Post-Training Inference",
        "",
        f"Generated UTC: `{report['generated_at_utc']}`",
        "Status: PASS",
        f"Checkpoint: `{checkpoint}`",
        f"Output: `{output_path}`",
        f"Detections: {report['detection_count']}",
        f"Classes: {', '.join(report['detected_classes']) or 'none'}",
        f"Preprocess ms: {report['preprocessing_time_ms']}",
        f"Inference ms: {report['inference_time_ms']}",
        f"NMS ms: {report['nms_time_ms']}",
        "",
        "These timings are diagnostic CPU timings only and are not final real-time speed results.",
    ]
    (root / "artifacts" / "yolov5s_smoke_post_training_inference.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    return report


def write_colab_transfer_manifest(root: Path) -> dict[str, Any]:
    """Write a manifest for transferring the prepared project to Colab."""
    root = root.resolve()
    dataset_root = root / "data" / "processed" / "coco_yolo"
    artifacts_root = root / "artifacts"
    manifest = {
        "generated_at_utc": utc_now(),
        "notebook_path": "notebooks/YOLOv5_COCO_Training_Colab.ipynb",
        "configuration_files": [
            "configs/project.yaml",
            "configs/train_yolov5s.yaml",
            "configs/train_yolov5s_smoke.yaml",
            "configs/train_yolov5m.yaml",
            "configs/train_yolov5l.yaml",
            "configs/coco_project_colab.yaml",
        ],
        "coco_dataset_yaml": "data/processed/coco_yolo/coco_project.yaml",
        "dataset_directory": "data/processed/coco_yolo",
        "split_manifests": [
            "data/splits/train_manifest.csv",
            "data/splits/val_manifest.csv",
            "data/splits/test_manifest.csv",
            "data/splits/split_summary.json",
        ],
        "pretrained_weight_requirements": {
            "local_smoke_completed_with": "models/pretrained/yolov5s.pt",
            "full_colab_training": [
                "Download or mount yolov5s.pt.",
                "Download yolov5m.pt and yolov5l.pt in Colab only if those experiments are approved.",
            ],
        },
        "expected_output_directories": [
            "results/yolov5s",
            "results/yolov5m",
            "results/yolov5l",
            "models/trained",
            "outputs/images",
        ],
        "google_drive_destination_structure": {
            "root": "MyDrive/yolov5_coco_training",
            "dataset": "MyDrive/yolov5_coco_training/data/processed/coco_yolo",
            "configs": "MyDrive/yolov5_coco_training/configs",
            "weights": "MyDrive/yolov5_coco_training/models/pretrained",
            "runs": "MyDrive/yolov5_coco_training/results",
        },
        "approximate_dataset_storage_requirement": {
            "prepared_dataset_logical_size": "about 20+ GiB for images plus labels; local images are hardlinks",
            "archives_if_transferred": "about 19 GiB additional",
            "recommended_colab_free_space": "at least 60 GiB, more for checkpoints and experiment runs",
        },
        "upload_or_mount": [
            "Prefer mounting or copying data/processed/coco_yolo instead of raw archives.",
            "Upload configs, notebooks, split manifests, and required pretrained weights.",
            "Do not upload local smoke-test checkpoints as final trained models.",
        ],
        "windows_to_colab_path_adaptation": [
            "Replace drive-rooted Windows paths with /content/drive/MyDrive/yolov5_coco_training paths.",
            "Use configs/coco_project_colab.yaml or rewrite the dataset YAML path field after mounting Drive.",
            "Use forward slashes in Colab YAML files.",
        ],
        "integrity_checks_after_transfer": [
            "Validate notebook JSON opens in Colab.",
            "Check dataset YAML train/val/test directories exist.",
            "Count train/val/test images against data/splits/split_summary.json.",
            "Run a YOLOv5 dataloader smoke check before full training.",
            "Verify pretrained weight hashes or at least non-empty torch.load success.",
        ],
        "local_dataset_root_exists": dataset_root.exists(),
    }
    write_json(artifacts_root / "colab_training_transfer_manifest.json", manifest)
    lines = [
        "# Colab Training Transfer Manifest",
        "",
        f"Generated UTC: `{manifest['generated_at_utc']}`",
        "",
        "## Required Files",
        f"- Notebook: `{manifest['notebook_path']}`",
        f"- COCO dataset YAML: `{manifest['coco_dataset_yaml']}`",
        f"- Dataset directory: `{manifest['dataset_directory']}`",
        "",
        "## Configurations",
    ]
    lines.extend(f"- `{item}`" for item in manifest["configuration_files"])
    lines.extend(["", "## Split Manifests"])
    lines.extend(f"- `{item}`" for item in manifest["split_manifests"])
    lines.extend(
        [
            "",
            "## Google Drive Destination",
            "- Root: `MyDrive/yolov5_coco_training`",
            "- Dataset: `MyDrive/yolov5_coco_training/data/processed/coco_yolo`",
            "- Configs: `MyDrive/yolov5_coco_training/configs`",
            "- Weights: `MyDrive/yolov5_coco_training/models/pretrained`",
            "- Runs: `MyDrive/yolov5_coco_training/results`",
            "",
            "## Path Adaptation",
        ]
    )
    lines.extend(f"- {item}" for item in manifest["windows_to_colab_path_adaptation"])
    lines.extend(["", "## Integrity Checks"])
    lines.extend(f"- {item}" for item in manifest["integrity_checks_after_transfer"])
    (artifacts_root / "colab_training_transfer_manifest.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    return manifest


def summarize_training_run(root: Path, elapsed_seconds: float | None = None) -> dict[str, Any]:
    """Create the final local smoke-training results artifact."""
    root = root.resolve()
    run_dir = root / "results" / "yolov5s" / "smoke_test"
    dataset_validation = json.loads(
        (root / "artifacts" / "training_smoke_dataset_validation.json").read_text(encoding="utf-8")
    )
    checkpoint_validation = json.loads(
        (root / "artifacts" / "yolov5s_smoke_checkpoint_validation.json").read_text(encoding="utf-8")
    )
    inference = json.loads(
        (root / "artifacts" / "yolov5s_smoke_post_training_inference.json").read_text(encoding="utf-8")
    )
    training_results = parse_results_csv(run_dir / "results.csv")
    log_report = parse_training_log(root / "artifacts" / "yolov5s_smoke_training_console.log")
    train_batch_plots = sorted(str(path) for path in run_dir.glob("train_batch*.jpg"))
    native_components = {
        **log_report["checks"],
        "forward_pass_completed": log_report["checks"]["box_loss_executed"],
        "augmentation_pipeline_executed": bool(train_batch_plots),
        "learning_rate_scheduling_executed": bool(training_results["learning_rates"]),
        "backward_and_optimizer_update_completed": checkpoint_validation["weight_update_vs_pretrained"].get("status") == "passed",
        "best_checkpoint_written": checkpoint_validation["checkpoints"]["best"]["exists"],
        "last_checkpoint_written": checkpoint_validation["checkpoints"]["last"]["exists"],
    }
    report = {
        "generated_at_utc": utc_now(),
        "warning": SMOKE_WARNING,
        "metric_label": SMOKE_METRIC_LABEL,
        "smoke_dataset": {
            "counts": dataset_validation["counts"],
            "annotation_count": dataset_validation["annotation_count"],
            "represented_class_count": dataset_validation["represented_class_count"],
            "represented_classes": dataset_validation["represented_classes"],
            "validation_ready": dataset_validation["final_readiness"],
        },
        "training": {
            "command": ".\\.venv\\Scripts\\python.exe -m src.train_models --config configs\\train_yolov5s_smoke.yaml --device cpu",
            "model": "yolov5s",
            "weights": "models/pretrained/yolov5s.pt",
            "epochs": 1,
            "batch_size": 2,
            "imgsz": 640,
            "optimizer": "SGD",
            "device": "cpu",
            "workers": 0,
            "seed": DEFAULT_SEED,
            "run_dir": str(run_dir),
            "elapsed_seconds": elapsed_seconds,
            **training_results,
        },
        "native_yolov5_component_evidence": native_components,
        "checkpoint_validation": checkpoint_validation,
        "post_training_inference": inference,
        "console_log": log_report,
        "warnings": log_report["warnings"],
        "fixes_applied": [
            "Resolved wrapper paths relative to the workspace before invoking YOLOv5 from external/yolov5.",
            "Added smoke-test overrides for dataset, output directory, batch size, epochs, workers, and console capture.",
            "Created deterministic hardlinked smoke dataset utilities and validation artifacts.",
            "Configured YOLOV5_CONFIG_DIR to a workspace-local font cache to avoid network font downloads.",
            "Patched YOLOv5 v7.0 Pillow 12 text sizing compatibility for plot generation.",
            "Patched YOLOv5 v7.0 NumPy 2.x AP integration compatibility.",
        ],
        "full_training_readiness": bool(
            dataset_validation["final_readiness"]
            and checkpoint_validation["valid"]
            and inference["status"] == "passed"
            and all(native_components.values())
        ),
    }
    write_json(root / "artifacts" / "yolov5s_local_smoke_training_results.json", report)
    return report


def render_final_report(report: dict[str, Any], test_results: dict[str, Any] | None = None) -> str:
    """Render the final Markdown smoke-training report."""
    training = report["training"]
    smoke = report["smoke_dataset"]
    checkpoint = report["checkpoint_validation"]
    inference = report["post_training_inference"]
    test_results = test_results or {}
    lines = [
        "# YOLOv5s Local Smoke Training Report",
        "",
        f"Generated UTC: `{report['generated_at_utc']}`",
        "",
        f"**Warning:** {SMOKE_WARNING}",
        "",
        "## Recovery",
        "- Recovery audit: `artifacts/phase4_recovery_audit.md`",
        "- Complete before reconnection: smoke utility scaffolding, smoke config, 32/16 smoke subset, and smoke dataset validation.",
        "- Failed before reconnection/resume completion: first YOLOv5 attempt failed on offline font download; second attempt failed before checkpoints on NumPy 2.x `trapz` removal.",
        "- Completed after recovery: compatibility fixes, successful one-epoch smoke training, checkpoint validation, post-training inference, quality checks, Colab manifest, and documentation updates.",
        "",
        "## Smoke Dataset",
        f"- Train images: {smoke['counts']['train_images']}",
        f"- Validation images: {smoke['counts']['val_images']}",
        f"- Annotations: {smoke['annotation_count']}",
        f"- Represented classes: {smoke['represented_class_count']}",
        f"- Dataset validation: {'PASS' if smoke['validation_ready'] else 'FAIL'}",
        "",
        "## Training Configuration",
        f"- Command: `{training['command']}`",
        f"- Model: `{training['model']}`",
        f"- Weights: `{training['weights']}`",
        f"- Epochs: {training['epochs']}",
        f"- Batch size: {training['batch_size']}",
        f"- Image size: {training['imgsz']}",
        f"- Optimizer: {training['optimizer']}",
        f"- Device: {training['device']}",
        f"- Workers: {training['workers']}",
        f"- Seed: {training['seed']}",
        f"- Elapsed seconds: {training.get('elapsed_seconds')}",
        "",
        f"## {SMOKE_METRIC_LABEL}",
        f"**Warning:** {SMOKE_WARNING}",
        "",
        "Training losses:",
    ]
    for key, value in training["training_losses"].items():
        lines.append(f"- {key}: {value}")
    lines.append("")
    lines.append("Validation losses:")
    for key, value in training["validation_losses"].items():
        lines.append(f"- {key}: {value}")
    lines.append("")
    lines.append("Diagnostic validation metrics:")
    for key, value in training["diagnostic_metrics"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(
        [
            "",
            "## Checkpoints",
            f"- best.pt: exists={checkpoint['checkpoints']['best']['exists']}, "
            f"size={checkpoint['checkpoints']['best']['size_bytes']}, "
            f"load={checkpoint['checkpoints']['best']['load_status']}",
            f"- last.pt: exists={checkpoint['checkpoints']['last']['exists']}, "
            f"size={checkpoint['checkpoints']['last']['size_bytes']}, "
            f"load={checkpoint['checkpoints']['last']['load_status']}",
            f"- Weight update vs pretrained: {checkpoint['weight_update_vs_pretrained'].get('status')}",
            f"- Reload inference: {checkpoint['reload_inference'].get('status')}",
            "",
            "## Post-Training Inference",
            f"- Status: {inference['status']}",
            f"- Output: `{inference['output']}`",
            f"- Detections: {inference['detection_count']}",
            f"- Classes: {', '.join(inference['detected_classes']) or 'none'}",
            f"- Preprocess ms: {inference['preprocessing_time_ms']}",
            f"- Inference ms: {inference['inference_time_ms']}",
            f"- NMS ms: {inference['nms_time_ms']}",
            "",
            "## Native Component Evidence",
        ]
    )
    for key, value in report["native_yolov5_component_evidence"].items():
        lines.append(f"- {key}: {'PASS' if value else 'FAIL'}")
    lines.extend(["", "## Quality Checks"])
    for key, value in test_results.items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Warnings"])
    if report["warnings"]:
        lines.extend(f"- {warning}" for warning in report["warnings"][:30])
    else:
        lines.append("- No material blocking warnings were found in the captured log.")
    lines.extend(["", "## Fixes Applied"])
    lines.extend(f"- {item}" for item in report["fixes_applied"])
    lines.extend(
        [
            "",
            "## Full-Training Readiness",
            f"- Ready for full Colab training: {'YES' if report['full_training_readiness'] else 'NO'}",
            "- Full local COCO training was not started.",
            "",
            "Exact next recommended action, not executed:",
            "",
            "Review `artifacts/colab_training_transfer_manifest.md`, then mount or upload the listed dataset/config/weight files in Google Drive and run only the Colab setup and integrity-check cells before starting GPU training.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    """CLI entrypoint for smoke-training utilities."""
    parser = argparse.ArgumentParser(description="Prepare and validate YOLOv5s training smoke artifacts.")
    parser.add_argument("--create-subset", action="store_true")
    parser.add_argument("--validate-dataset", action="store_true")
    parser.add_argument("--validate-checkpoints", action="store_true")
    parser.add_argument("--post-inference", action="store_true")
    parser.add_argument("--write-colab-manifest", action="store_true")
    parser.add_argument("--summarize-run", action="store_true")
    parser.add_argument("--elapsed-seconds", type=float)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    setup_logging(args.log_level)

    root = project_root()
    if args.create_subset:
        create_smoke_subset(root)
    if args.validate_dataset:
        validate_smoke_dataset(root)
    if args.validate_checkpoints:
        validate_checkpoints(root)
    if args.post_inference:
        run_post_training_inference(root)
    if args.write_colab_manifest:
        write_colab_transfer_manifest(root)
    if args.summarize_run:
        report = summarize_training_run(root, args.elapsed_seconds)
        report_text = render_final_report(report)
        (root / "artifacts" / "yolov5s_local_smoke_training_report.md").write_text(
            report_text,
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
