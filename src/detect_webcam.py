"""Run YOLOv5 detection on a webcam or live stream."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.common import project_root, setup_logging
from src.inference_engine import InferenceEngine, load_inference_config, parse_class_filter


LOGGER = logging.getLogger(__name__)


def detect_webcam(
    weights: Path | None = None,
    camera_index: int = 0,
    output: Path | None = None,
    conf_thres: float | None = None,
    iou_thres: float | None = None,
    device: str | None = None,
    imgsz: int | None = None,
    display: bool = True,
) -> Path | None:
    """Run live detection until Q is pressed."""
    config = load_inference_config(
        model_path=weights,
        confidence_threshold=conf_thres,
        iou_threshold=iou_thres,
        device=device,
        image_size=imgsz,
    )
    engine = InferenceEngine(config)
    output_path = None
    if output is not None:
        output_path = output if output.suffix else output / "webcam_detected.mp4"
    summary = engine.predict_webcam(camera_index, output_path=output_path, display=display)
    if output_path:
        LOGGER.info(
            "Saved %s (%d frames, %d detections)",
            output_path,
            summary["frame_count"],
            summary["total_detections"],
        )
    return output_path


def main() -> int:
    """CLI entrypoint for webcam detection."""
    root = project_root()
    parser = argparse.ArgumentParser(description="YOLOv5 webcam detection.")
    parser.add_argument("--config", type=Path, default=root / "configs" / "inference.yaml")
    parser.add_argument("--weights", "--model-path", dest="model_path", type=Path)
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--conf-thres", type=float)
    parser.add_argument("--iou-thres", type=float)
    parser.add_argument("--device")
    parser.add_argument("--imgsz", type=int)
    parser.add_argument("--class-filter", nargs="+")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)
    config = load_inference_config(
        args.config,
        model_path=args.model_path,
        confidence_threshold=args.conf_thres,
        iou_threshold=args.iou_thres,
        device=args.device,
        image_size=args.imgsz,
        class_filter=parse_class_filter(args.class_filter),
    )
    engine = InferenceEngine(config)
    output_root = args.output or config.output_directory / "webcam"
    output_path = None if args.no_save else (output_root if output_root.suffix else output_root / "webcam_detected.mp4")
    summary = engine.predict_webcam(args.camera_index, output_path=output_path, display=not args.no_display)
    if output_path:
        LOGGER.info("Saved %s", output_path)
        print(f"Saved: {output_path}")
    print(f"Frames: {summary['frame_count']}")
    print(f"Detections: {summary['total_detections']}")
    print(f"Classes: {summary['class_counts'] or 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
