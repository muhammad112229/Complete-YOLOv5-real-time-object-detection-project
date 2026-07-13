"""Import smoke tests for project modules."""

from __future__ import annotations

import importlib


MODULES = [
    "src.environment",
    "src.download_coco",
    "src.prepare_coco_dataset",
    "src.parse_coco_annotations",
    "src.split_coco_dataset",
    "src.validate_dataset",
    "src.visualize_dataset",
    "src.train_models",
    "src.training_smoke",
    "src.evaluate_models",
    "src.evaluate_model",
    "src.analyze_training_results",
    "src.inspect_checkpoint",
    "src.sample_predictions",
    "src.recreate_test_subset",
    "src.create_coco_subset_annotations",
    "src.calculate_coco_metrics",
    "src.detect_image",
    "src.detect_video",
    "src.detect_webcam",
    "src.inference_engine",
    "src.media_utils",
    "src.result_store",
    "src.video_compatibility",
    "src.webcam_stream",
    "src.benchmark_inference",
    "src.prune_model",
    "src.quantize_model",
    "src.export_model",
    "src.robustness_tests",
]


def test_project_modules_import() -> None:
    """All project modules should import without heavyweight runtime dependencies."""
    for module in MODULES:
        importlib.import_module(module)
