"""Backward-compatible imports for the reusable inference engine."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.inference_engine import (
    InferenceConfig,
    InferenceEngine,
    InferenceResult,
    annotate_frame,
    format_detections,
    infer_frame_with_model,
    load_inference_config,
    load_yolov5_model,
    open_video_writer,
    summarize_detections,
)


def infer_frame(model: Any, frame: Any, imgsz: int = 640) -> InferenceResult:
    """Run inference on one OpenCV BGR frame and return an annotated frame."""
    return infer_frame_with_model(model, frame, imgsz)


__all__ = [
    "InferenceConfig",
    "InferenceEngine",
    "InferenceResult",
    "annotate_frame",
    "format_detections",
    "infer_frame",
    "load_inference_config",
    "load_yolov5_model",
    "open_video_writer",
    "summarize_detections",
]
