"""Generate reproducible robustness-test image variants."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from src.common import ensure_dir, require_python_package


ROBUSTNESS_CASES = [
    "low_lighting",
    "bright_lighting",
    "occlusion",
    "small_objects",
    "scale_variation",
    "motion_blur",
    "complex_backgrounds",
    "multiple_objects",
]


def create_variant(image_path: Path, output_path: Path, case: str, seed: int = 42) -> None:
    """Create one deterministic robustness-test image variant."""
    Image = require_python_package("PIL.Image", "Pillow")
    ImageEnhance = require_python_package("PIL.ImageEnhance", "Pillow")
    ImageFilter = require_python_package("PIL.ImageFilter", "Pillow")
    ImageDraw = require_python_package("PIL.ImageDraw", "Pillow")

    rng = random.Random(seed)
    image = Image.open(image_path).convert("RGB")
    if case == "low_lighting":
        image = ImageEnhance.Brightness(image).enhance(0.35)
    elif case == "bright_lighting":
        image = ImageEnhance.Brightness(image).enhance(1.8)
    elif case == "occlusion":
        draw = ImageDraw.Draw(image)
        width, height = image.size
        box = [
            rng.randint(0, width // 2),
            rng.randint(0, height // 2),
            rng.randint(width // 2, width),
            rng.randint(height // 2, height),
        ]
        draw.rectangle(box, fill=(32, 32, 32))
    elif case == "motion_blur":
        image = image.filter(ImageFilter.GaussianBlur(radius=2.0))
    elif case == "scale_variation":
        width, height = image.size
        image = image.resize((max(1, width // 2), max(1, height // 2))).resize((width, height))
    elif case == "small_objects":
        width, height = image.size
        canvas = Image.new("RGB", (width, height), (114, 114, 114))
        small = image.resize((max(1, width // 3), max(1, height // 3)))
        canvas.paste(small, (width // 3, height // 3))
        image = canvas
    elif case in {"complex_backgrounds", "multiple_objects"}:
        image = ImageEnhance.Contrast(image).enhance(1.6)
    else:
        raise ValueError(f"Unsupported robustness case: {case}")
    ensure_dir(output_path.parent)
    image.save(output_path)


def generate_robustness_suite(image_paths: list[Path], output_dir: Path, seed: int = 42) -> dict[str, object]:
    """Create robustness image variants and a manifest."""
    manifest: list[dict[str, str]] = []
    for image_path in image_paths:
        for case in ROBUSTNESS_CASES:
            output_path = output_dir / case / image_path.name
            create_variant(image_path, output_path, case, seed)
            manifest.append({"case": case, "source": str(image_path), "output": str(output_path)})
    report = {
        "status": "variants_generated_only; run evaluation before claiming robustness results",
        "cases": ROBUSTNESS_CASES,
        "items": manifest,
    }
    ensure_dir(output_dir)
    (output_dir / "robustness_manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> int:
    """CLI entrypoint for robustness suite generation."""
    parser = argparse.ArgumentParser(description="Generate robustness-test image variants.")
    parser.add_argument("--image", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/robustness/generated"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    generate_robustness_suite(args.image, args.output_dir, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
