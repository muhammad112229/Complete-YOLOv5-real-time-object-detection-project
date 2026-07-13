"""Visualize YOLO labels on source images."""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

from src.common import ensure_dir, project_root, require_python_package, setup_logging


LOGGER = logging.getLogger(__name__)


def draw_yolo_boxes(image_path: Path, label_path: Path, output_path: Path, class_names: list[str]) -> None:
    """Draw normalized YOLO labels on an image using Pillow."""
    Image = require_python_package("PIL.Image", "Pillow")
    ImageDraw = require_python_package("PIL.ImageDraw", "Pillow")

    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size
    for line in label_path.read_text(encoding="utf-8").splitlines():
        class_id_text, x_text, y_text, w_text, h_text = line.split()
        class_id = int(class_id_text)
        x_center, y_center, box_width, box_height = map(float, [x_text, y_text, w_text, h_text])
        x1 = (x_center - box_width / 2) * width
        y1 = (y_center - box_height / 2) * height
        x2 = (x_center + box_width / 2) * width
        y2 = (y_center + box_height / 2) * height
        label = class_names[class_id] if class_id < len(class_names) else str(class_id)
        draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=3)
        draw.text((x1, max(0, y1 - 12)), label, fill=(255, 255, 0))
    ensure_dir(output_path.parent)
    image.save(output_path)


def visualize_samples(
    manifest_jsonl: Path,
    output_dir: Path,
    class_names_path: Path,
    count: int = 16,
    seed: int = 42,
) -> list[Path]:
    """Render a deterministic sample of dataset images with annotations."""
    records = [json.loads(line) for line in manifest_jsonl.read_text(encoding="utf-8").splitlines()]
    class_names = class_names_path.read_text(encoding="utf-8").splitlines()
    random.Random(seed).shuffle(records)
    outputs: list[Path] = []
    for record in records[:count]:
        image_path = Path(record["project_image_path"])
        label_path = Path(record["project_label_path"])
        output_path = output_dir / f"{manifest_jsonl.stem}_{image_path.stem}.jpg"
        draw_yolo_boxes(image_path, label_path, output_path, class_names)
        outputs.append(output_path)
    return outputs


def main() -> int:
    """CLI entrypoint for dataset visualization."""
    root = project_root()
    parser = argparse.ArgumentParser(description="Visualize processed YOLO labels.")
    parser.add_argument("--manifest", type=Path, default=root / "data" / "splits" / "train.jsonl")
    parser.add_argument("--class-names", type=Path, default=root / "data" / "processed" / "coco2017_yolo" / "class_names.txt")
    parser.add_argument("--output-dir", type=Path, default=root / "outputs" / "images" / "dataset_samples")
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)
    outputs = visualize_samples(args.manifest, args.output_dir, args.class_names, args.count, args.seed)
    LOGGER.info("Wrote %d visualizations to %s", len(outputs), args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
