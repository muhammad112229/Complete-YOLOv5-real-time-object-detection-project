"""Inspect a YOLOv5 checkpoint without modifying it."""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.common import ensure_dir, project_root, require_file, require_python_package, setup_logging, yolov5_root
from src.evaluate_model import relative_path, resolve_workspace_path, sha256_file


DEFAULT_CHECKPOINT = Path("models") / "yolov5s_coco20k_best.pt"
DEFAULT_OUTPUT = Path("artifacts") / "checkpoint_inspection.json"


def safe_torch_load(path: Path) -> Any:
    """Load a checkpoint on CPU with YOLOv5 v7.0-compatible torch.load options."""
    yolo_root = yolov5_root(project_root())
    if str(yolo_root) not in sys.path:
        sys.path.insert(0, str(yolo_root))
    torch = require_python_package("torch")
    kwargs: dict[str, Any] = {"map_location": "cpu"}
    try:
        signature = inspect.signature(torch.load)
        if "weights_only" in signature.parameters:
            kwargs["weights_only"] = False
    except (TypeError, ValueError):
        pass
    return torch.load(path, **kwargs)


def tensor_to_list(value: Any) -> Any:
    """Convert tensor-like metadata to JSON-serializable values."""
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [tensor_to_list(item) for item in value]
    return value


def count_parameters(model: Any) -> tuple[int | None, int | None]:
    """Return total and trainable parameter counts when possible."""
    if model is None or not hasattr(model, "parameters"):
        return None, None
    total = 0
    trainable = 0
    for parameter in model.parameters():
        count = int(parameter.numel())
        total += count
        if getattr(parameter, "requires_grad", False):
            trainable += count
    return total, trainable


def inspect_checkpoint(checkpoint_path: Path, output_path: Path | None = None) -> dict[str, Any]:
    """Inspect a checkpoint and optionally write JSON output."""
    checkpoint_path = require_file(checkpoint_path, "checkpoint")
    checkpoint = safe_torch_load(checkpoint_path)
    is_mapping = isinstance(checkpoint, dict)
    model = checkpoint.get("model") if is_mapping else checkpoint
    ema = checkpoint.get("ema") if is_mapping else None
    optimizer = checkpoint.get("optimizer") if is_mapping else None
    names = getattr(model, "names", None)
    if names is None and ema is not None:
        names = getattr(ema, "names", None)
    class_names_count = len(names) if hasattr(names, "__len__") else None
    number_of_classes = getattr(model, "nc", None)
    if number_of_classes is None and ema is not None:
        number_of_classes = getattr(ema, "nc", None)
    total_parameters, trainable_parameters = count_parameters(model)

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint_path": relative_path(checkpoint_path),
        "sha256": sha256_file(checkpoint_path),
        "file_size": checkpoint_path.stat().st_size,
        "checkpoint_keys": sorted(str(key) for key in checkpoint.keys()) if is_mapping else [],
        "model_type": f"{type(model).__module__}.{type(model).__name__}" if model is not None else None,
        "number_of_classes": int(number_of_classes) if number_of_classes is not None else None,
        "class_names_count": class_names_count,
        "model_parameter_count": total_parameters,
        "trainable_parameter_count": trainable_parameters,
        "stride": tensor_to_list(getattr(model, "stride", None)),
        "checkpoint_epoch": checkpoint.get("epoch") if is_mapping else None,
        "best_fitness": tensor_to_list(checkpoint.get("best_fitness")) if is_mapping else None,
        "optimizer_state_presence": optimizer is not None,
        "ema_presence": ema is not None,
    }
    if output_path is not None:
        ensure_dir(output_path.parent)
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> int:
    """CLI entrypoint for checkpoint inspection."""
    parser = argparse.ArgumentParser(description="Inspect a YOLOv5 checkpoint.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)
    report = inspect_checkpoint(resolve_workspace_path(args.checkpoint), resolve_workspace_path(args.output))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
