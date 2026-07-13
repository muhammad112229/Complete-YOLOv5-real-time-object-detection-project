"""Benchmark YOLOv5 inference latency and FPS."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from src.common import file_size_mb, project_root, require_file, require_python_package, setup_logging
from src.inference import infer_frame, load_yolov5_model


def count_parameters(model: object) -> int:
    """Count model parameters when available."""
    parameter_owner = getattr(model, "model", model)
    if not hasattr(parameter_owner, "parameters"):
        return 0
    return int(sum(parameter.numel() for parameter in parameter_owner.parameters()))


def benchmark_images(
    weights: Path,
    image_paths: list[Path],
    output_json: Path,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    device: str = "cpu",
    imgsz: int = 640,
    warmup: int = 3,
) -> dict[str, object]:
    """Benchmark inference over image files and save measured results."""
    cv2 = require_python_package("cv2", "opencv-python")
    model = load_yolov5_model(weights, device, conf_thres, iou_thres)
    frames = []
    for path in image_paths:
        require_file(path, "benchmark image")
        frame = cv2.imread(str(path))
        if frame is None:
            raise RuntimeError(f"Could not read benchmark image: {path}")
        frames.append(frame)
    if not frames:
        raise ValueError("At least one benchmark image is required.")

    for index in range(warmup):
        infer_frame(model, frames[index % len(frames)], imgsz)

    frame_results = [infer_frame(model, frame, imgsz) for frame in frames]
    latencies = [result.inference_ms for result in frame_results]
    fps_values = [1000.0 / latency for latency in latencies if latency > 0]
    preprocess = [result.preprocess_ms for result in frame_results if result.preprocess_ms is not None]
    model_inference = [
        result.model_inference_ms for result in frame_results if result.model_inference_ms is not None
    ]
    nms = [result.nms_ms for result in frame_results if result.nms_ms is not None]
    result = {
        "weights": str(weights),
        "model_size_mb": file_size_mb(weights),
        "parameter_count": count_parameters(model),
        "device": device,
        "imgsz": imgsz,
        "num_images": len(frames),
        "preprocess_ms_mean": statistics.mean(preprocess) if preprocess else None,
        "model_inference_ms_mean": statistics.mean(model_inference) if model_inference else None,
        "nms_ms_mean": statistics.mean(nms) if nms else None,
        "latency_ms_mean": statistics.mean(latencies),
        "latency_ms_median": statistics.median(latencies),
        "latency_ms_min": min(latencies),
        "latency_ms_max": max(latencies),
        "fps_mean": statistics.mean(fps_values) if fps_values else 0.0,
        "note": "Measured locally for the supplied images only; not a fabricated benchmark.",
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> int:
    """CLI entrypoint for inference benchmarking."""
    root = project_root()
    parser = argparse.ArgumentParser(description="Benchmark YOLOv5 inference.")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--image", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, default=root / "results" / "comparisons" / "inference_benchmark.json")
    parser.add_argument("--conf-thres", type=float, default=0.25)
    parser.add_argument("--iou-thres", type=float, default=0.45)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)
    benchmark_images(
        args.weights,
        args.image,
        args.output,
        args.conf_thres,
        args.iou_thres,
        args.device,
        args.imgsz,
        args.warmup,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
