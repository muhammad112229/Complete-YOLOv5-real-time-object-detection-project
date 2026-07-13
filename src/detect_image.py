"""Run YOLOv5 detection on a single image."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.common import project_root, setup_logging
from src.inference_engine import InferenceEngine, load_inference_config, parse_class_filter, summarize_detections


LOGGER = logging.getLogger(__name__)


def detect_image(
    source: Path,
    weights: Path | None = None,
    output: Path | None = None,
    conf_thres: float | None = None,
    iou_thres: float | None = None,
    device: str | None = None,
    imgsz: int | None = None,
) -> Path:
    """Run detection on one image and save the annotated result."""
    config = load_inference_config(
        model_path=weights,
        confidence_threshold=conf_thres,
        iou_threshold=iou_thres,
        device=device,
        image_size=imgsz,
    )
    engine = InferenceEngine(config)
    result = engine.predict_image(source)
    output_root = output or config.output_directory / "images"
    output_path = output_root if output_root.suffix else output_root / source.name
    engine.save_annotated_image(result, output_path)
    LOGGER.info(
        "Saved %s (%.2f ms, %.2f FPS, %d detections: %s)",
        output_path,
        result.inference_ms,
        result.fps,
        result.detection_count,
        ", ".join(result.detected_class_names) or "none",
    )
    return output_path


def main() -> int:
    """CLI entrypoint for image detection."""
    root = project_root()
    parser = argparse.ArgumentParser(description="YOLOv5 image detection.")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=root / "configs" / "inference.yaml")
    parser.add_argument("--weights", "--model-path", dest="model_path", type=Path)
    parser.add_argument("--output", type=Path)
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
    result = engine.predict_image(args.source)
    output_root = args.output or config.output_directory / "images"
    output_path = output_root if output_root.suffix else output_root / args.source.name
    engine.save_annotated_image(result, output_path)
    summary = summarize_detections(result.detections)
    LOGGER.info("Saved %s", output_path)
    print(f"Saved: {output_path}")
    print(f"Detections: {result.detection_count}")
    print(f"Classes: {summary or 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
