"""Analyze genuine YOLOv5 training results from results.csv."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.common import ensure_dir, project_root, require_file, require_python_package, setup_logging
from src.evaluate_model import load_flat_yaml, relative_path, resolve_workspace_path


DEFAULT_RESULTS_CSV = Path("artifacts") / "trained_coco20k_yolov5s" / "results.csv"
DEFAULT_OPT_YAML = Path("artifacts") / "trained_coco20k_yolov5s" / "opt.yaml"
DEFAULT_OUTPUT_DIR = Path("results") / "training_analysis"


METRICS = {
    "training_box_loss": ("train/box_loss", "min"),
    "training_objectness_loss": ("train/obj_loss", "min"),
    "training_classification_loss": ("train/cls_loss", "min"),
    "validation_box_loss": ("val/box_loss", "min"),
    "validation_objectness_loss": ("val/obj_loss", "min"),
    "validation_classification_loss": ("val/cls_loss", "min"),
    "precision": ("metrics/precision", "max"),
    "recall": ("metrics/recall", "max"),
    "mAP@0.5": ("metrics/mAP_0.5", "max"),
    "mAP@0.5:0.95": ("metrics/mAP_0.5:0.95", "max"),
}


def read_results_csv(path: Path) -> list[dict[str, float]]:
    """Read YOLOv5 results.csv with stripped column names."""
    require_file(path, "training results.csv")
    rows: list[dict[str, float]] = []
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            raise ValueError(f"Missing CSV header in {path}")
        reader.fieldnames = [field.strip() for field in reader.fieldnames]
        for raw in reader:
            row: dict[str, float] = {}
            for key, value in raw.items():
                if key is None:
                    continue
                row[key.strip()] = float(str(value).strip())
            rows.append(row)
    if not rows:
        raise ValueError(f"No training rows found in {path}")
    return sorted(rows, key=lambda item: int(item["epoch"]))


def write_training_metrics_csv(rows: list[dict[str, float]], output: Path) -> None:
    """Write normalized training metrics CSV."""
    ensure_dir(output.parent)
    fieldnames = list(rows[0].keys())
    with output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_training(rows: list[dict[str, float]], configured_epochs: int | None) -> dict[str, Any]:
    """Summarize final/best metrics and completed epoch count."""
    final_row = rows[-1]
    summary: dict[str, Any] = {
        "best_epoch_metric": "metrics/mAP_0.5:0.95",
        "best_epoch": int(max(rows, key=lambda row: row["metrics/mAP_0.5:0.95"])["epoch"]),
        "final_epoch": int(final_row["epoch"]),
        "total_completed_epochs": len(rows),
        "configured_epochs": configured_epochs,
        "completed_epochs_note": (
            f"{len(rows)} epochs completed although {configured_epochs} were configured."
            if configured_epochs is not None and configured_epochs != len(rows)
            else f"{len(rows)} epochs completed."
        ),
        "early_stopping_claim": "not_claimed",
        "early_stopping_note": "The artifacts do not prove early stopping caused termination.",
        "metrics": {},
    }
    for metric_name, (column, direction) in METRICS.items():
        selector = min if direction == "min" else max
        best_row = selector(rows, key=lambda row, col=column: row[col])
        summary["metrics"][metric_name] = {
            "csv_column": column,
            "direction": direction,
            "final": final_row[column],
            "final_epoch": int(final_row["epoch"]),
            "best": best_row[column],
            "best_epoch": int(best_row["epoch"]),
        }
    return summary


def create_plots(rows: list[dict[str, float]], output_dir: Path) -> None:
    """Create high-resolution training plots from genuine results.csv values."""
    matplotlib = require_python_package("matplotlib")
    matplotlib.use("Agg")
    pyplot = require_python_package("matplotlib.pyplot", "matplotlib")

    epochs = [int(row["epoch"]) for row in rows]
    plot_specs = [
        (
            "losses.png",
            "YOLOv5 Training and Validation Losses",
            [
                ("train/box_loss", "Train box loss"),
                ("train/obj_loss", "Train objectness loss"),
                ("train/cls_loss", "Train classification loss"),
                ("val/box_loss", "Validation box loss"),
                ("val/obj_loss", "Validation objectness loss"),
                ("val/cls_loss", "Validation classification loss"),
            ],
            "Loss",
        ),
        (
            "precision_recall.png",
            "YOLOv5 Precision and Recall",
            [
                ("metrics/precision", "Precision"),
                ("metrics/recall", "Recall"),
            ],
            "Score",
        ),
        (
            "map_metrics.png",
            "YOLOv5 mAP Metrics",
            [
                ("metrics/mAP_0.5", "mAP@0.5"),
                ("metrics/mAP_0.5:0.95", "mAP@0.5:0.95"),
            ],
            "mAP",
        ),
    ]
    ensure_dir(output_dir)
    for filename, title, columns, ylabel in plot_specs:
        pyplot.figure(figsize=(10, 6), dpi=150)
        for column, label in columns:
            pyplot.plot(epochs, [row[column] for row in rows], marker="o", linewidth=2, label=label)
        pyplot.title(title)
        pyplot.xlabel("Epoch")
        pyplot.ylabel(ylabel)
        pyplot.grid(True, alpha=0.3)
        pyplot.legend()
        pyplot.tight_layout()
        pyplot.savefig(output_dir / filename, dpi=300)
        pyplot.close()


def analyze_training_results(
    results_csv: Path,
    opt_yaml: Path | None,
    output_dir: Path,
    make_plots: bool = True,
) -> dict[str, Any]:
    """Generate training analysis artifacts from genuine YOLOv5 outputs."""
    rows = read_results_csv(results_csv)
    configured_epochs = None
    training_config: dict[str, Any] = {}
    if opt_yaml and opt_yaml.is_file():
        training_config = load_flat_yaml(opt_yaml)
        value = training_config.get("epochs")
        configured_epochs = int(value) if value is not None else None

    ensure_dir(output_dir)
    write_training_metrics_csv(rows, output_dir / "training_metrics.csv")
    summary = summarize_training(rows, configured_epochs)
    (output_dir / "best_epoch_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_results_csv": relative_path(results_csv),
        "source_opt_yaml": relative_path(opt_yaml) if opt_yaml else None,
        "output_directory": relative_path(output_dir),
        "configured_epochs": configured_epochs,
        "total_completed_epochs": len(rows),
        "completed_vs_configured": summary["completed_epochs_note"],
        "early_stopping_claim": "not_claimed",
        "training_configuration": training_config,
        "artifacts": {
            "training_metrics_csv": relative_path(output_dir / "training_metrics.csv"),
            "best_epoch_summary_json": relative_path(output_dir / "best_epoch_summary.json"),
            "losses_plot": relative_path(output_dir / "losses.png"),
            "precision_recall_plot": relative_path(output_dir / "precision_recall.png"),
            "map_metrics_plot": relative_path(output_dir / "map_metrics.png"),
        },
    }
    if make_plots:
        create_plots(rows, output_dir)
    (output_dir / "training_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {"summary": summary, "metadata": metadata}


def main() -> int:
    """CLI entrypoint for training results analysis."""
    parser = argparse.ArgumentParser(description="Analyze genuine YOLOv5 training results.csv.")
    parser.add_argument("--results-csv", type=Path, default=DEFAULT_RESULTS_CSV)
    parser.add_argument("--opt-yaml", type=Path, default=DEFAULT_OPT_YAML)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)
    result = analyze_training_results(
        resolve_workspace_path(args.results_csv),
        resolve_workspace_path(args.opt_yaml),
        resolve_workspace_path(args.output_dir),
        make_plots=not args.no_plots,
    )
    print(json.dumps(result["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
