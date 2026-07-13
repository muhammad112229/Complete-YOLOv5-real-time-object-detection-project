"""Production video inference CLI."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common import setup_logging
from src.inference_engine import InferenceEngine, add_common_inference_args, config_from_args


def main() -> int:
    parser = argparse.ArgumentParser(description="Run production YOLOv5 video inference.")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--display", action="store_true")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--log-level", default="INFO")
    add_common_inference_args(parser)
    args = parser.parse_args()

    setup_logging(args.log_level)
    config = config_from_args(args)
    engine = InferenceEngine(config)
    output_root = args.output or config.output_directory / "videos"
    output_path = output_root if output_root.suffix else output_root / f"{args.source.stem}_detected.mp4"
    summary = engine.predict_video(args.source, output_path, display=args.display, max_frames=args.max_frames)
    print(f"Saved: {output_path}")
    print(f"Frames: {summary['frame_count']}")
    print(f"Detections: {summary['total_detections']}")
    print(f"Classes: {summary['class_counts'] or 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
