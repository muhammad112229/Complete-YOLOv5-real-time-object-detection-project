"""Validate a processed YOLO-format COCO dataset."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from src.common import project_root, setup_logging


LOGGER = logging.getLogger(__name__)


def validate_label_file(path: Path) -> list[str]:
    """Validate one YOLO label file and return error messages."""
    errors: list[str] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        parts = line.split()
        if len(parts) != 5:
            errors.append(f"{path}:{line_number} expected 5 fields")
            continue
        try:
            class_id = int(parts[0])
            values = [float(value) for value in parts[1:]]
        except ValueError:
            errors.append(f"{path}:{line_number} contains non-numeric values")
            continue
        if class_id < 0:
            errors.append(f"{path}:{line_number} class_id is negative")
        if not all(0.0 <= value <= 1.0 for value in values):
            errors.append(f"{path}:{line_number} normalized bbox out of range")
        if values[2] <= 0 or values[3] <= 0:
            errors.append(f"{path}:{line_number} width/height must be positive")
    return errors


def validate_project_dataset(dataset_root: Path, split_root: Path) -> dict[str, object]:
    """Validate labels, image/label pairing, and split leakage."""
    errors: list[str] = []
    split_images: dict[str, set[str]] = {}
    for split_name in ["train", "val", "test"]:
        manifest_path = split_root / f"{split_name}.jsonl"
        split_images[split_name] = set()
        if not manifest_path.exists():
            errors.append(f"Missing split manifest: {manifest_path}")
            continue
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            image_path = Path(record["project_image_path"])
            label_path = Path(record["project_label_path"])
            split_images[split_name].add(f"{record['source_split']}:{record['image_id']}")
            if not image_path.exists():
                errors.append(f"Missing image: {image_path}")
            if not label_path.exists():
                errors.append(f"Missing label: {label_path}")
            else:
                errors.extend(validate_label_file(label_path))

    for left in ["train", "val", "test"]:
        for right in ["train", "val", "test"]:
            if left >= right:
                continue
            overlap = split_images[left].intersection(split_images[right])
            if overlap:
                errors.append(f"Leakage between {left} and {right}: {len(overlap)} images")

    summary = {
        "dataset_root": str(dataset_root),
        "split_root": str(split_root),
        "errors": errors,
        "valid": not errors,
    }
    return summary


def main() -> int:
    """CLI entrypoint for dataset validation."""
    root = project_root()
    parser = argparse.ArgumentParser(description="Validate processed YOLO dataset.")
    parser.add_argument("--dataset-root", type=Path, default=root / "data" / "processed" / "coco2017_yolo")
    parser.add_argument("--split-root", type=Path, default=root / "data" / "splits")
    parser.add_argument("--output", type=Path, default=root / "artifacts" / "dataset_validation.json")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)
    summary = validate_project_dataset(args.dataset_root, args.split_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if summary["valid"]:
        LOGGER.info("Dataset validation passed.")
    else:
        LOGGER.error("Dataset validation failed with %d errors.", len(summary["errors"]))
    return 0 if summary["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

