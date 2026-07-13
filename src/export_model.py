"""Export YOLOv5 checkpoints to TorchScript and ONNX formats."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.common import project_root, require_file, run_command, setup_logging, yolov5_root


def build_export_command(
    weights: Path,
    include: list[str],
    imgsz: int,
    device: str,
    half: bool,
    data_yaml: Path | None,
) -> list[str]:
    """Build a YOLOv5 export.py command."""
    command = [
        sys.executable,
        "export.py",
        "--weights",
        str(weights),
        "--imgsz",
        str(imgsz),
        "--include",
        *include,
        "--device",
        device,
    ]
    if half:
        command.append("--half")
    if data_yaml:
        command.extend(["--data", str(data_yaml)])
    return command


def main() -> int:
    """CLI entrypoint for model export."""
    root = project_root()
    parser = argparse.ArgumentParser(description="Export YOLOv5 weights.")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--include", nargs="+", default=["torchscript", "onnx"])
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--half", action="store_true", help="FP16 export where supported.")
    parser.add_argument("--data", type=Path, default=root / "data" / "processed" / "coco2017_yolo" / "dataset.yaml")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)
    require_file(args.weights, "weights")
    data_yaml = args.data if args.data.exists() else None
    command = build_export_command(args.weights, args.include, args.imgsz, args.device, args.half, data_yaml)
    return run_command(command, cwd=yolov5_root(root), dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())

