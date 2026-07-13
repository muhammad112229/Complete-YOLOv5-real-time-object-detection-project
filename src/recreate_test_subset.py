"""Recreate the deterministic 2,500-image COCO test subset manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.common import ensure_dir, project_root, setup_logging


DEFAULT_SOURCE_MANIFEST = Path("data") / "splits" / "test_images.txt"
DEFAULT_OUTPUT_MANIFEST = Path("data") / "splits" / "test_subset_2500_seed42.txt"
DEFAULT_REPORT = Path("artifacts") / "test_subset_2500_verification.json"
DEFAULT_SEED = 42
DEFAULT_SUBSET_SIZE = 2500


def resolve_workspace_path(value: str | Path, root: Path | None = None) -> Path:
    """Resolve a workspace-relative path."""
    base = root or project_root()
    path = Path(value)
    return path if path.is_absolute() else base / path


def relative_path(path: Path, root: Path | None = None) -> str:
    """Return a workspace-relative path when possible."""
    base = root or project_root()
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return str(path)


def sha256_file_bytes(path: Path) -> str:
    """Compute SHA256 from the exact on-disk bytes."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest_lines(path: Path) -> list[str]:
    """Load non-empty manifest lines in their stored on-disk order."""
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def label_path_for_image(image_path: str | Path) -> Path:
    """Resolve the YOLO label path corresponding to an image path."""
    path = Path(image_path)
    parts = list(path.parts)
    if "images" in parts:
        index = parts.index("images")
        parts[index] = "labels"
        return Path(*parts).with_suffix(".txt")
    return path.parent.parent / "labels" / path.parent.name / f"{path.stem}.txt"


def count_duplicate_paths(paths: list[str]) -> int:
    """Return the number of repeated path entries beyond the first occurrence."""
    counts = Counter(paths)
    return sum(count - 1 for count in counts.values() if count > 1)


def inspect_path_availability(paths: list[str]) -> dict[str, Any]:
    """Count image and label availability for manifest paths."""
    missing_images: list[str] = []
    missing_labels: list[str] = []
    image_missing_count = 0
    label_missing_count = 0
    instance_count = 0
    for item in paths:
        image_path = Path(item)
        label_path = label_path_for_image(image_path)
        if not image_path.is_file():
            image_missing_count += 1
            if len(missing_images) < 10:
                missing_images.append(item)
        if not label_path.is_file():
            label_missing_count += 1
            if len(missing_labels) < 10:
                missing_labels.append(str(label_path))
        else:
            instance_count += sum(1 for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip())
    return {
        "missing_image_count": image_missing_count,
        "missing_label_count": label_missing_count,
        "missing_images_preview": missing_images,
        "missing_labels_preview": missing_labels,
        "labeled_instance_count": instance_count,
    }


def recreate_subset(
    source_manifest: Path,
    output_manifest: Path,
    report_path: Path,
    seed: int = DEFAULT_SEED,
    subset_size: int = DEFAULT_SUBSET_SIZE,
) -> dict[str, Any]:
    """Recreate and verify the deterministic subset manifest."""
    source_manifest = resolve_workspace_path(source_manifest)
    output_manifest = resolve_workspace_path(output_manifest)
    report_path = resolve_workspace_path(report_path)
    if not source_manifest.is_file():
        raise FileNotFoundError(f"Missing full test manifest: {source_manifest}")
    full_paths = load_manifest_lines(source_manifest)
    if len(full_paths) < subset_size:
        raise ValueError(f"Cannot sample {subset_size} paths from only {len(full_paths)} entries.")

    subset_paths = random.Random(seed).sample(full_paths, subset_size)
    ensure_dir(output_manifest.parent)
    output_manifest.write_text("\n".join(subset_paths) + "\n", encoding="utf-8")

    source_availability = inspect_path_availability(full_paths)
    subset_availability = inspect_path_availability(subset_paths)
    duplicate_count = count_duplicate_paths(full_paths)
    subset_duplicate_count = count_duplicate_paths(subset_paths)
    verification_passed = (
        len(full_paths) == 12330
        and len(subset_paths) == subset_size
        and duplicate_count == 0
        and subset_duplicate_count == 0
        and source_availability["missing_image_count"] == 0
        and source_availability["missing_label_count"] == 0
        and subset_availability["missing_image_count"] == 0
        and subset_availability["missing_label_count"] == 0
    )

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_full_test_manifest_path": relative_path(source_manifest),
        "source_line_count": len(full_paths),
        "subset_manifest_path": relative_path(output_manifest),
        "subset_line_count": len(subset_paths),
        "random_seed": seed,
        "sampling_algorithm_description": (
            "Read full_test_paths from data/splits/test_images.txt in stored disk order, then run "
            "random.Random(42).sample(full_test_paths, 2500) without sorting before sampling."
        ),
        "source_manifest_sha256": sha256_file_bytes(source_manifest),
        "subset_manifest_sha256": sha256_file_bytes(output_manifest),
        "duplicate_count": duplicate_count,
        "subset_duplicate_count": subset_duplicate_count,
        "missing_image_count": source_availability["missing_image_count"],
        "missing_label_count": source_availability["missing_label_count"],
        "subset_missing_image_count": subset_availability["missing_image_count"],
        "subset_missing_label_count": subset_availability["missing_label_count"],
        "subset_labeled_instance_count": subset_availability["labeled_instance_count"],
        "first_10_sampled_entries": subset_paths[:10],
        "verification_status": "passed" if verification_passed else "failed",
        "source_availability": source_availability,
        "subset_availability": subset_availability,
    }
    ensure_dir(report_path.parent)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> int:
    """CLI entrypoint for deterministic subset recreation."""
    parser = argparse.ArgumentParser(description="Recreate deterministic 2,500-image test subset manifest.")
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST)
    parser.add_argument("--output-manifest", type=Path, default=DEFAULT_OUTPUT_MANIFEST)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--subset-size", type=int, default=DEFAULT_SUBSET_SIZE)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)
    report = recreate_subset(args.source_manifest, args.output_manifest, args.report, args.seed, args.subset_size)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
