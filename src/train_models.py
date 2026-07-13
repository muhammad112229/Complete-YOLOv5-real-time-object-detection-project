"""Training wrapper for YOLOv5s, YOLOv5m, and YOLOv5l."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

from src.common import load_yaml, project_root, run_command, setup_logging, yolov5_root
from src.training_smoke import create_smoke_subset, ensure_yolov5_offline_font_config, validate_smoke_dataset


MODEL_WEIGHTS = {
    "yolov5s": "yolov5s.pt",
    "yolov5m": "yolov5m.pt",
    "yolov5l": "yolov5l.pt",
}


def resolve_weights(model: str, weights: Path | None, allow_auto_download: bool) -> str:
    """Resolve pretrained weights for YOLOv5 training."""
    if weights is not None and weights.exists():
        return str(weights)
    if allow_auto_download:
        return MODEL_WEIGHTS[model]
    expected = weights or Path("models") / "pretrained" / MODEL_WEIGHTS[model]
    raise FileNotFoundError(
        f"Missing pretrained weights: {expected}. Place weights there or use "
        "--allow-yolov5-auto-download to let YOLOv5 fetch official COCO weights."
    )


def resolve_project_path(root: Path, path: Path | None) -> Path | None:
    """Resolve a path relative to the project root."""
    if path is None:
        return None
    return path if path.is_absolute() else root / path


def clean_smoke_run_dir(root: Path, run_dir: Path) -> None:
    """Remove the generated smoke-test run directory before a fresh smoke run."""
    resolved = run_dir.resolve()
    allowed = (root / "results" / "yolov5s").resolve()
    if resolved == allowed or allowed not in resolved.parents:
        raise ValueError(f"Refusing to clean unexpected training output path: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)


def build_train_command(
    model: str,
    data_yaml: Path,
    weights: str,
    imgsz: int,
    epochs: int,
    batch_size: int,
    optimizer: str,
    device: str,
    seed: int,
    patience: int,
    project: Path,
    name: str,
    resume: bool = False,
    workers: int = 4,
    exist_ok: bool = False,
) -> list[str]:
    """Build a YOLOv5 train.py command."""
    command = [
        sys.executable,
        "train.py",
        "--imgsz",
        str(imgsz),
        "--batch-size",
        str(batch_size),
        "--epochs",
        str(epochs),
        "--data",
        str(data_yaml),
        "--weights",
        weights,
        "--optimizer",
        optimizer,
        "--device",
        device,
        "--seed",
        str(seed),
        "--patience",
        str(patience),
        "--project",
        str(project),
        "--name",
        name,
        "--workers",
        str(workers),
    ]
    if resume:
        command.append("--resume")
    if exist_ok:
        command.append("--exist-ok")
    return command


def main() -> int:
    """CLI entrypoint for model training."""
    root = project_root()
    parser = argparse.ArgumentParser(description="Run YOLOv5 training.")
    parser.add_argument("--config", type=Path, help="Optional training YAML config.")
    parser.add_argument("--model", choices=sorted(MODEL_WEIGHTS), default="yolov5s")
    parser.add_argument("--data", type=Path, default=root / "data" / "processed" / "coco2017_yolo" / "dataset.yaml")
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--optimizer", choices=["SGD", "Adam", "AdamW"], default="SGD")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--project", type=Path, default=root / "results")
    parser.add_argument("--name", default="coco2017_yolov5")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke-test", action="store_true", help="Force one epoch and tiny batch.")
    parser.add_argument("--allow-yolov5-auto-download", action="store_true")
    parser.add_argument("--exist-ok", action="store_true", help="Allow YOLOv5 to reuse the output directory.")
    parser.add_argument("--console-log", type=Path, help="Optional file for combined stdout/stderr.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    setup_logging(args.log_level)

    if args.config:
        args.config = resolve_project_path(root, args.config)
        config = load_yaml(args.config)
        args.model = config.get("model", args.model)
        args.weights = Path(config["weights"]) if config.get("weights") else args.weights
        args.data = Path(config.get("data", args.data))
        args.imgsz = int(config.get("imgsz", args.imgsz))
        args.epochs = int(config.get("epochs", args.epochs))
        args.batch_size = int(config.get("batch_size", args.batch_size))
        args.optimizer = str(config.get("optimizer", args.optimizer))
        args.device = str(config.get("device", args.device))
        args.seed = int(config.get("seed", args.seed))
        args.patience = int(config.get("patience", args.patience))
        args.project = Path(config.get("project", args.project))
        args.name = str(config.get("name", args.name))
        args.workers = int(config.get("workers", args.workers))
        args.resume = bool(config.get("resume", args.resume))
        args.exist_ok = bool(config.get("exist_ok", args.exist_ok))
        args.console_log = Path(config["console_log"]) if config.get("console_log") else args.console_log

    if args.smoke_test:
        create_smoke_subset(root)
        validation = validate_smoke_dataset(root)
        if not validation["final_readiness"]:
            raise RuntimeError("Smoke dataset validation failed; see artifacts/training_smoke_dataset_validation.json")
        args.data = root / "data" / "smoke" / "coco_yolov5" / "smoke_dataset.yaml"
        args.epochs = 1
        args.batch_size = 2
        args.optimizer = "SGD"
        args.device = "cpu"
        args.workers = 0
        args.seed = 42
        args.project = root / "results" / "yolov5s"
        args.name = "smoke_test"
        args.exist_ok = True
        args.console_log = args.console_log or root / "artifacts" / "yolov5s_smoke_training_console.log"

    args.data = resolve_project_path(root, args.data)
    args.weights = resolve_project_path(root, args.weights) or root / "models" / "pretrained" / MODEL_WEIGHTS[args.model]
    args.project = resolve_project_path(root, args.project)
    args.console_log = resolve_project_path(root, args.console_log)

    if not args.data.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {args.data}")
    weights = resolve_weights(args.model, args.weights, args.allow_yolov5_auto_download)
    os.environ["YOLOV5_CONFIG_DIR"] = str(ensure_yolov5_offline_font_config(root))
    if args.smoke_test and not args.resume and not args.dry_run:
        clean_smoke_run_dir(root, args.project / args.name)
    command = build_train_command(
        model=args.model,
        data_yaml=args.data,
        weights=weights,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch_size=args.batch_size,
        optimizer=args.optimizer,
        device=args.device,
        seed=args.seed,
        patience=args.patience,
        project=args.project,
        name=args.name,
        resume=args.resume,
        workers=args.workers,
        exist_ok=args.exist_ok,
    )
    return run_command(command, cwd=yolov5_root(root), dry_run=args.dry_run, log_path=args.console_log)


if __name__ == "__main__":
    raise SystemExit(main())
