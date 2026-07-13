# Large Artifact Tracking Policy

This repository keeps compact metrics, summaries, plots, metadata, source code, configuration, tests, reports, deterministic manifests, and the production checkpoint in normal Git.

The raw prediction dumps below are generated evaluation artifacts. They are preserved locally after cleanup but are not tracked in Git because they are large, reproducible, and not needed for normal review of the project results.

| File | Size | Git status | Reason |
|---|---:|---|---|
| `results/evaluation/test_subset_2500/coco_eval/coco_predictions.json` | 79.818 MB | Ignored / not tracked | Raw COCOeval prediction dump; compact COCOeval summaries remain tracked. |
| `results/evaluation/test_subset_2500/predictions.json` | 58.041 MB | Ignored / not tracked | Raw YOLOv5 prediction dump; compact YOLOv5 metrics and reports remain tracked. |
| `results/evaluation/test_subset_2500/yolov5s_coco20k_best_predictions.json` | 58.041 MB | Ignored / not tracked | Duplicate model-specific raw prediction dump; reproducible from the same evaluation command. |
| `results/evaluation/test_subset_2500/labels/` | generated label files | Ignored / not tracked | Generated prediction labels; not required when compact summaries and plots are tracked. |

## Tracked Compact Evaluation Evidence

- `results/evaluation/test_subset_2500/metrics_summary.json`
- `results/evaluation/test_subset_2500/per_class_metrics.csv`
- `results/evaluation/test_subset_2500/test_results_summary.md`
- `results/evaluation/test_subset_2500/confusion_matrix.png`
- `results/evaluation/test_subset_2500/confusion_matrix_normalized.png`
- `results/evaluation/test_subset_2500/PR_curve.png`
- `results/evaluation/test_subset_2500/P_curve.png`
- `results/evaluation/test_subset_2500/R_curve.png`
- `results/evaluation/test_subset_2500/F1_curve.png`
- `results/evaluation/test_subset_2500/coco_eval/coco_eval_summary.json`
- `results/evaluation/test_subset_2500/coco_eval/coco_eval_metadata.json`
- `results/evaluation/test_subset_2500/coco_eval/coco_eval_stdout.txt`
- `results/final_project_metrics.json`

## Regeneration

The raw prediction dumps can be regenerated from the production checkpoint and the exact deterministic test subset:

```powershell
.\.venv\Scripts\python.exe deployment\evaluate_model.py --config configs\evaluation.yaml --require-exact-2500
```

This uses:

- checkpoint: `models/yolov5s_coco20k_best.pt`
- evaluation config: `configs/evaluation.yaml`
- test manifest: `data/splits/test_subset_2500_seed42.txt`
- COCO annotation JSON: `data/processed/coco_yolo/annotations/instances_test_subset_2500_seed42.json`

No retraining is required to regenerate these files.
