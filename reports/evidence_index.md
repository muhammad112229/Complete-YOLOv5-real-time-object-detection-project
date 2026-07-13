# Evidence Index

This index links the main verification evidence for GitHub and internship review.

| Evidence | File |
|---|---|
| Trained checkpoint verification | [`artifacts/trained_model_verification.json`](../artifacts/trained_model_verification.json) |
| Checkpoint inspection | [`artifacts/checkpoint_inspection.json`](../artifacts/checkpoint_inspection.json) |
| Training metrics summary | [`results/training_metrics_summary.json`](../results/training_metrics_summary.json) |
| Training analysis best epoch | [`results/training_analysis/best_epoch_summary.json`](../results/training_analysis/best_epoch_summary.json) |
| Exact 2,500-image test subset verification | [`artifacts/test_subset_2500_verification.json`](../artifacts/test_subset_2500_verification.json) |
| COCO subset annotation verification | [`artifacts/test_subset_coco_annotation_verification.json`](../artifacts/test_subset_coco_annotation_verification.json) |
| Genuine YOLOv5 test metrics | [`results/evaluation/test_subset_2500/metrics_summary.json`](../results/evaluation/test_subset_2500/metrics_summary.json) |
| Official COCOeval metrics | [`results/evaluation/test_subset_2500/coco_eval/coco_eval_summary.json`](../results/evaluation/test_subset_2500/coco_eval/coco_eval_summary.json) |
| Test result summary | [`results/evaluation/test_subset_2500/test_results_summary.md`](../results/evaluation/test_subset_2500/test_results_summary.md) |
| Image inference evidence metadata | [`outputs/flask/metadata/7fab10546d2047d8bab5b889391ed1ba.json`](../outputs/flask/metadata/7fab10546d2047d8bab5b889391ed1ba.json) |
| Video inference evidence metadata | [`outputs/flask/metadata/2de3cad407914270a34508e48cf148cf.json`](../outputs/flask/metadata/2de3cad407914270a34508e48cf148cf.json) |
| Browser video compatibility verification | [`artifacts/video_browser_compatibility_verification.json`](../artifacts/video_browser_compatibility_verification.json) |
| Flask API and image smoke evidence | [`outputs/flask/metadata/7fab10546d2047d8bab5b889391ed1ba.json`](../outputs/flask/metadata/7fab10546d2047d8bab5b889391ed1ba.json) |
| Screenshot checklist | [`reports/screenshots/README.md`](screenshots/README.md) |
| Final project metrics | [`results/final_project_metrics.json`](../results/final_project_metrics.json) |
| Final test report | [`artifacts/final_test_report.json`](../artifacts/final_test_report.json) |

Physical verification:

- Image detection: verified through Flask.
- Video detection: verified through Flask.
- Webcam detection: verified through Flask.
- Physical webcam observation: 2.4 FPS, 324.1 ms latency, 1 detected object.
