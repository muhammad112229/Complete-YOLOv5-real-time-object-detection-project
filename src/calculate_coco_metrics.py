"""Calculate COCO-style metrics with pycocotools."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.common import require_python_package


METRIC_NAMES = [
    "mAP@0.5:0.95",
    "mAP@0.5",
    "mAP@0.75",
    "mAP@0.5:0.95_small",
    "mAP@0.5:0.95_medium",
    "mAP@0.5:0.95_large",
    "AR@1",
    "AR@10",
    "AR@100",
    "AR@100_small",
    "AR@100_medium",
    "AR@100_large",
]


def evaluate_coco(ground_truth_json: Path, predictions_json: Path) -> dict[str, Any]:
    """Run COCOeval and return overall plus class-wise AP metrics."""
    coco_module = require_python_package("pycocotools.coco", "pycocotools")
    cocoeval_module = require_python_package("pycocotools.cocoeval", "pycocotools")
    COCO = coco_module.COCO
    COCOeval = cocoeval_module.COCOeval

    coco_gt = COCO(str(ground_truth_json))
    coco_dt = coco_gt.loadRes(str(predictions_json))
    evaluator = COCOeval(coco_gt, coco_dt, "bbox")
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()

    metrics = {name: float(value) for name, value in zip(METRIC_NAMES, evaluator.stats, strict=True)}
    class_wise_ap: dict[str, float] = {}
    for category_id in coco_gt.getCatIds():
        category = coco_gt.loadCats([category_id])[0]
        class_eval = COCOeval(coco_gt, coco_dt, "bbox")
        class_eval.params.catIds = [category_id]
        class_eval.evaluate()
        class_eval.accumulate()
        class_wise_ap[str(category["name"])] = float(class_eval.stats[0]) if len(class_eval.stats) else -1.0

    return {
        "ground_truth": str(ground_truth_json),
        "predictions": str(predictions_json),
        "metrics": metrics,
        "AP50": metrics["mAP@0.5"],
        "AP75": metrics["mAP@0.75"],
        "class_wise_AP": class_wise_ap,
    }


def main() -> int:
    """CLI entrypoint for COCO metrics."""
    parser = argparse.ArgumentParser(description="Calculate COCO metrics from prediction JSON.")
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/comparisons/coco_metrics.json"))
    args = parser.parse_args()

    results = evaluate_coco(args.ground_truth, args.predictions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

