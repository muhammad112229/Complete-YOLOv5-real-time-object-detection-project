"""Generate annotated sample predictions with structured summaries."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from src.common import ensure_dir, project_root, require_python_package, setup_logging
from src.evaluate_model import relative_path, resolve_workspace_path
from src.inference_engine import InferenceEngine, load_inference_config, summarize_detections


DEFAULT_OUTPUT_DIR = Path("outputs") / "inference" / "samples"
DEFAULT_SUMMARY = Path("results") / "sample_prediction_summary.json"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def candidate_image_roots(root: Path | None = None) -> list[Path]:
    """Return ordered local image locations for sample prediction generation."""
    base = root or project_root()
    return [
        base / "external" / "yolov5" / "data" / "images",
        base / "outputs" / "images",
        base / "data" / "smoke" / "coco_yolov5" / "images" / "val",
        base / "data" / "smoke" / "coco_yolov5" / "images" / "train",
        base / "data" / "processed" / "coco_yolo" / "images" / "test",
    ]


def find_sample_images(max_images: int, roots: Iterable[Path] | None = None) -> list[Path]:
    """Find suitable local sample images without downloading data."""
    selected: list[Path] = []
    seen: set[Path] = set()
    for image_root in roots or candidate_image_roots():
        if not image_root.exists():
            continue
        for path in sorted(image_root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            resolved = path.resolve()
            if resolved in seen:
                continue
            selected.append(path)
            seen.add(resolved)
            if len(selected) >= max_images:
                return selected
    return selected


def confidence_statistics(detections: tuple[dict[str, float | int | str], ...]) -> dict[str, float | None]:
    """Compute confidence statistics for detections."""
    confidences = [float(detection["confidence"]) for detection in detections]
    if not confidences:
        return {"min": None, "max": None, "mean": None}
    return {
        "min": min(confidences),
        "max": max(confidences),
        "mean": sum(confidences) / len(confidences),
    }


def sample_record(
    source: Path,
    output: Path,
    width: int,
    height: int,
    detection_count: int,
    class_counts: dict[str, int],
    confidence_stats: dict[str, float | None],
    inference_ms: float,
) -> dict[str, Any]:
    """Build one JSON-serializable sample prediction record."""
    return {
        "source_path": relative_path(source),
        "output_path": relative_path(output),
        "image_dimensions": {"width": width, "height": height},
        "number_of_detections": detection_count,
        "class_counts": class_counts,
        "confidence_statistics": confidence_stats,
        "inference_time_ms": inference_ms,
    }


def generate_sample_predictions(
    max_images: int = 5,
    output_dir: Path | None = None,
    summary_path: Path | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    """Generate annotated predictions for local sample images."""
    root = project_root()
    output_dir = resolve_workspace_path(output_dir or DEFAULT_OUTPUT_DIR, root)
    summary_path = resolve_workspace_path(summary_path or DEFAULT_SUMMARY, root)
    ensure_dir(output_dir)
    ensure_dir(summary_path.parent)
    config = load_inference_config(
        model_path=root / "models" / "yolov5s_coco20k_best.pt",
        output_directory=output_dir.parent,
        device=device,
    )
    engine = InferenceEngine(config)
    cv2 = require_python_package("cv2", "opencv-python")
    images = find_sample_images(max_images)
    records: list[dict[str, Any]] = []
    for index, source in enumerate(images, start=1):
        frame = cv2.imread(str(source))
        if frame is None:
            continue
        result = engine.predict_image(source)
        output = output_dir / f"{index:02d}_{source.stem}_predicted{source.suffix.lower()}"
        engine.save_annotated_image(result, output)
        height, width = frame.shape[:2]
        records.append(
            sample_record(
                source=source,
                output=output,
                width=int(width),
                height=int(height),
                detection_count=result.detection_count,
                class_counts=summarize_detections(result.detections),
                confidence_stats=confidence_statistics(result.detections),
                inference_ms=result.inference_ms,
            )
        )

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_path": "models/yolov5s_coco20k_best.pt",
        "output_directory": relative_path(output_dir),
        "requested_max_images": max_images,
        "generated_count": len(records),
        "accuracy_claim": "none",
        "accuracy_note": "These are unlabeled visual inference examples and are not test-set accuracy metrics.",
        "samples": records,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    """CLI entrypoint for sample prediction generation."""
    parser = argparse.ArgumentParser(description="Generate annotated sample predictions.")
    parser.add_argument("--max-images", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--device")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)
    summary = generate_sample_predictions(args.max_images, args.output_dir, args.summary, args.device)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
