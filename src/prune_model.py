"""Apply unstructured pruning to a YOLOv5 checkpoint for experimentation."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from src.common import file_size_mb, project_root, require_file, require_python_package, setup_logging, yolov5_root


LOGGER = logging.getLogger(__name__)


def prune_checkpoint(weights: Path, output: Path, amount: float) -> dict[str, object]:
    """Prune Conv2d weights and save a new checkpoint.

    The pruned model must be evaluated and usually fine-tuned before use.
    """
    if not 0.0 < amount < 1.0:
        raise ValueError("Pruning amount must be between 0 and 1.")
    sys.path.insert(0, str(yolov5_root(project_root())))
    torch = require_python_package("torch")
    prune = require_python_package("torch.nn.utils.prune")
    require_file(weights, "weights")
    checkpoint = torch.load(weights, map_location="cpu")
    model = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    pruned_layers = 0
    for module in model.modules():
        if module.__class__.__name__ == "Conv2d":
            prune.l1_unstructured(module, name="weight", amount=amount)
            prune.remove(module, "weight")
            pruned_layers += 1
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output)
    report = {
        "input": str(weights),
        "output": str(output),
        "amount": amount,
        "pruned_conv_layers": pruned_layers,
        "input_size_mb": file_size_mb(weights),
        "output_size_mb": file_size_mb(output),
        "status": "created; accuracy and speed must be evaluated separately",
    }
    return report


def main() -> int:
    """CLI entrypoint for pruning."""
    parser = argparse.ArgumentParser(description="Prune a YOLOv5 checkpoint.")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--amount", type=float, default=0.2)
    parser.add_argument("--report", type=Path, default=Path("results/comparisons/pruning_report.json"))
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)
    report = prune_checkpoint(args.weights, args.output, args.amount)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    LOGGER.info("Pruned checkpoint written to %s", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
