"""Evaluation wrapper around YOLOv5 val.py."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.common import project_root, require_file, run_command, setup_logging, yolov5_root


def build_eval_command(
    data_yaml: Path,
    weights: Path,
    imgsz: int,
    batch_size: int,
    conf_thres: float,
    iou_thres: float,
    device: str,
    task: str,
    project: Path,
    name: str,
    save_json: bool,
) -> list[str]:
    """Build a YOLOv5 val.py command."""
    command = [
        sys.executable,
        "val.py",
        "--data",
        str(data_yaml),
        "--weights",
        str(weights),
        "--imgsz",
        str(imgsz),
        "--batch-size",
        str(batch_size),
        "--conf-thres",
        str(conf_thres),
        "--iou-thres",
        str(iou_thres),
        "--device",
        device,
        "--task",
        task,
        "--project",
        str(project),
        "--name",
        name,
        "--plots",
    ]
    if save_json:
        command.append("--save-json")
    return command


def main() -> int:
    """CLI entrypoint for model evaluation."""
    root = project_root()
    parser = argparse.ArgumentParser(description="Evaluate YOLOv5 weights.")
    parser.add_argument("--data", type=Path, default=root / "data" / "processed" / "coco2017_yolo" / "dataset.yaml")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--conf-thres", type=float, default=0.001)
    parser.add_argument("--iou-thres", type=float, default=0.6)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--task", choices=["train", "val", "test", "speed", "study"], default="val")
    parser.add_argument("--project", type=Path, default=root / "results" / "comparisons")
    parser.add_argument("--name", default="evaluation")
    parser.add_argument("--save-json", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)
    require_file(args.data, "dataset YAML")
    require_file(args.weights, "weights")
    command = build_eval_command(
        data_yaml=args.data,
        weights=args.weights,
        imgsz=args.imgsz,
        batch_size=args.batch_size,
        conf_thres=args.conf_thres,
        iou_thres=args.iou_thres,
        device=args.device,
        task=args.task,
        project=args.project,
        name=args.name,
        save_json=args.save_json,
    )
    return run_command(command, cwd=yolov5_root(root), dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())

