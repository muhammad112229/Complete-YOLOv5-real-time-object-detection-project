"""Production image inference CLI."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common import setup_logging
from src.inference_engine import InferenceEngine, add_common_inference_args, config_from_args, summarize_detections


def main() -> int:
    parser = argparse.ArgumentParser(description="Run production YOLOv5 image inference.")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--log-level", default="INFO")
    add_common_inference_args(parser)
    args = parser.parse_args()

    setup_logging(args.log_level)
    config = config_from_args(args)
    engine = InferenceEngine(config)
    result = engine.predict_image(args.source)
    output_root = args.output or config.output_directory / "images"
    output_path = output_root if output_root.suffix else output_root / args.source.name
    engine.save_annotated_image(result, output_path)
    print(f"Saved: {output_path}")
    print(f"Detections: {result.detection_count}")
    print(f"Classes: {summarize_detections(result.detections) or 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
