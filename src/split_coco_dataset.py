"""Create deterministic train/validation/test splits for converted COCO data."""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import shutil
from pathlib import Path
from typing import Any, Literal

from src.common import ensure_dir, project_root, setup_logging


LOGGER = logging.getLogger(__name__)
LinkMode = Literal["none", "hardlink", "symlink", "copy"]


def deterministic_split(
    records: list[dict[str, Any]],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> dict[str, list[dict[str, Any]]]:
    """Split image records deterministically at image level."""
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-9:
        raise ValueError(f"Split ratios must sum to 1.0, got {total_ratio}")
    unique_keys = {(record["source_split"], int(record["image_id"])) for record in records}
    if len(unique_keys) != len(records):
        raise ValueError("Duplicate image records detected before splitting.")

    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    n_total = len(shuffled)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train : n_train + n_val],
        "test": shuffled[n_train + n_val :],
    }


def assert_no_leakage(splits: dict[str, list[dict[str, Any]]]) -> None:
    """Raise if any image appears in more than one split."""
    seen: dict[tuple[str, int], str] = {}
    for split_name, records in splits.items():
        for record in records:
            key = (str(record["source_split"]), int(record["image_id"]))
            if key in seen:
                raise ValueError(f"Image leakage: {key} in {seen[key]} and {split_name}")
            seen[key] = split_name


def link_or_copy_image(source: Path, destination: Path, mode: LinkMode) -> None:
    """Link or copy one image without resizing."""
    ensure_dir(destination.parent)
    if destination.exists():
        return
    if mode == "none":
        return
    if mode == "hardlink":
        os.link(source, destination)
    elif mode == "symlink":
        destination.symlink_to(source)
    elif mode == "copy":
        shutil.copy2(source, destination)
    else:
        raise ValueError(f"Unsupported link mode: {mode}")


def write_dataset_yaml(output_root: Path, class_names: list[str]) -> Path:
    """Write a YOLOv5 dataset YAML file."""
    dataset_yaml = output_root / "dataset.yaml"
    names = "\n".join(f"  {index}: {name}" for index, name in enumerate(class_names))
    dataset_yaml.write_text(
        "\n".join(
            [
                f"path: {output_root.as_posix()}",
                "train: images/train",
                "val: images/val",
                "test: images/test",
                f"nc: {len(class_names)}",
                "names:",
                names,
                "",
            ]
        ),
        encoding="utf-8",
    )
    return dataset_yaml


def load_processed_manifests(paths: list[Path]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    """Load processed annotation manifests produced by parse_coco_annotations."""
    records: list[dict[str, Any]] = []
    categories: list[dict[str, Any]] | None = None
    class_names: list[str] | None = None
    for path in paths:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        records.extend(manifest["images"])
        manifest_categories = sorted(manifest["categories"], key=lambda item: item["class_id"])
        names = [category["name"] for category in manifest_categories]
        if categories is None:
            categories = manifest_categories
        elif categories != manifest_categories:
            raise ValueError(f"Category mapping mismatch in {path}")
        if class_names is None:
            class_names = names
        elif class_names != names:
            raise ValueError(f"Category mapping mismatch in {path}")
    return records, categories or [], class_names or []


def write_project_coco_json(
    split_name: str,
    records: list[dict[str, Any]],
    categories: list[dict[str, Any]],
    output_root: Path,
) -> Path:
    """Write a COCO-style ground-truth JSON for one project split."""
    coco_categories = [
        {
            "id": int(category["coco_category_id"]),
            "name": str(category["name"]),
            "supercategory": str(category.get("supercategory", "")),
        }
        for category in categories
    ]
    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    for record in records:
        images.append(
            {
                "id": int(record["image_id"]),
                "file_name": str(record["file_name"]),
                "width": int(record["width"]),
                "height": int(record["height"]),
            }
        )
        for annotation in record.get("annotations", []):
            annotations.append(
                {
                    "id": int(annotation["id"]),
                    "image_id": int(annotation["image_id"]),
                    "category_id": int(annotation["category_id"]),
                    "bbox": annotation["bbox_coco"],
                    "area": annotation.get("area"),
                    "iscrowd": 0,
                }
            )
    output_path = output_root / f"instances_{split_name}_project.json"
    output_path.write_text(
        json.dumps(
            {
                "images": images,
                "annotations": annotations,
                "categories": coco_categories,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return output_path


def create_project_split(
    manifest_paths: list[Path],
    output_root: Path,
    split_root: Path,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    link_mode: LinkMode = "hardlink",
) -> dict[str, Any]:
    """Create deterministic split manifests and YOLOv5 dataset layout."""
    records, categories, class_names = load_processed_manifests(manifest_paths)
    splits = deterministic_split(records, train_ratio, val_ratio, test_ratio, seed)
    assert_no_leakage(splits)
    ensure_dir(output_root)
    ensure_dir(split_root)

    split_summary: dict[str, Any] = {
        "seed": seed,
        "ratios": {"train": train_ratio, "val": val_ratio, "test": test_ratio},
        "counts": {},
        "link_mode": link_mode,
        "output_root": str(output_root),
    }
    for split_name, split_records in splits.items():
        split_summary["counts"][split_name] = len(split_records)
        manifest_jsonl = split_root / f"{split_name}.jsonl"
        image_list = output_root / f"{split_name}.txt"
        image_lines: list[str] = []
        with manifest_jsonl.open("w", encoding="utf-8") as manifest_file:
            for record in split_records:
                source_image = Path(record["image_path"])
                source_label = Path(record["label_path"])
                destination_image = output_root / "images" / split_name / source_image.name
                destination_label = output_root / "labels" / split_name / f"{source_image.stem}.txt"
                link_or_copy_image(source_image, destination_image, link_mode)
                ensure_dir(destination_label.parent)
                if source_label.exists() and not destination_label.exists():
                    shutil.copy2(source_label, destination_label)
                enriched = {
                    **record,
                    "project_split": split_name,
                    "project_image_path": str(destination_image),
                    "project_label_path": str(destination_label),
                }
                manifest_file.write(json.dumps(enriched) + "\n")
                image_lines.append(str(destination_image))
        image_list.write_text("\n".join(image_lines) + ("\n" if image_lines else ""), encoding="utf-8")
        split_summary[f"{split_name}_coco_json"] = str(
            write_project_coco_json(split_name, split_records, categories, output_root)
        )

    dataset_yaml = write_dataset_yaml(output_root, class_names)
    split_summary["dataset_yaml"] = str(dataset_yaml)
    (split_root / "split_summary.json").write_text(json.dumps(split_summary, indent=2), encoding="utf-8")
    LOGGER.info("Wrote split summary to %s", split_root / "split_summary.json")
    return split_summary


def main() -> int:
    """CLI entrypoint for project split generation."""
    root = project_root()
    parser = argparse.ArgumentParser(description="Create deterministic COCO project splits.")
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--output-root", type=Path, default=root / "data" / "processed" / "coco2017_yolo")
    parser.add_argument("--split-root", type=Path, default=root / "data" / "splits")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--link-mode", choices=["none", "hardlink", "symlink", "copy"], default="hardlink")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)
    create_project_split(
        manifest_paths=args.manifest,
        output_root=args.output_root,
        split_root=args.split_root,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        link_mode=args.link_mode,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
