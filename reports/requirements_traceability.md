# Requirements Traceability

Status values:

- **Implemented**: Script, config, or documentation exists.
- **Prepared**: Workflow exists, but requires dataset, weights, GPU, or hardware execution.
- **Pending execution**: Must be run after approval or after trained artifacts exist.

| Requirement | Implementation file | Command / artifact | Status |
|---|---|---|---|
| Environment audit | `src/environment.py`, `verify_setup.py` | `artifacts/environment_audit.json`, `artifacts/environment_audit.md` | Implemented |
| Use COCO 2017 | `src/download_coco.py`, `src/prepare_coco_dataset.py` | `python -m src.download_coco --confirm-large-download`; `artifacts/coco_preprocessing_report.md` | Implemented |
| Do not use PASCAL VOC | `configs/project.yaml`, README | N/A | Implemented |
| Official YOLOv5 v7.0 | `external/yolov5` | `git -C external/yolov5 describe --tags --exact-match` | Implemented |
| Pretrained YOLOv5s/m/l weights | `configs/train_yolov5*.yaml`, `src/train_models.py` | `models/pretrained/*.pt` or YOLOv5 auto-download option | Prepared |
| Data preprocessing | `src/prepare_coco_dataset.py`, `src/parse_coco_annotations.py` | `data/interim/*`, `data/processed/coco_yolo/*`, `artifacts/coco_dataset_statistics.*` | Implemented |
| Preserve COCO IDs and metadata | `src/prepare_coco_dataset.py` | `data/interim/coco_combined_annotation_manifest.json` | Implemented |
| YOLO bbox conversion | `src/prepare_coco_dataset.py`, `src/parse_coco_annotations.py` | label files under `data/processed/coco_yolo/labels` | Implemented |
| Invalid/missing/corrupt detection | `src/prepare_coco_dataset.py`, `src/validate_dataset.py` | `artifacts/coco_extraction_validation.json`, `artifacts/coco_dataset_validation.json` | Implemented |
| 80/10/10 deterministic split seed 42 | `src/prepare_coco_dataset.py` | `data/splits/split_summary.json` | Implemented |
| No split leakage | `src/prepare_coco_dataset.py` | `artifacts/coco_dataset_validation.md` | Implemented |
| No full physical resize/duplication | `src/prepare_coco_dataset.py` | hardlink image strategy in `data/processed/coco_yolo/images` | Implemented |
| Letterbox 640x640 | YOLOv5 native loader, `src/prepare_coco_dataset.py` | `outputs/images/dataset_preprocessing/` | Implemented |
| Dataset visualization | `src/prepare_coco_dataset.py`, `src/visualize_dataset.py` | `outputs/images/dataset_validation/` | Implemented |
| YOLOv5s/m/l training wrappers | `src/train_models.py`, `configs/train_yolov5*.yaml` | `python -m src.train_models --config ...`; smoke report `artifacts/yolov5s_local_smoke_training_report.md` | Implemented for YOLOv5s smoke; prepared for full YOLOv5s/m/l training |
| SGD and AdamW | `src/train_models.py` | `--optimizer SGD` / `--optimizer AdamW` | Prepared |
| Native YOLOv5 losses | `external/yolov5/train.py` | `results/yolov5s/smoke_test/results.csv` | Implemented for YOLOv5s smoke |
| Early stopping, checkpoints, resume | `src/train_models.py` | YOLOv5 `--patience`, `--resume`, `results/yolov5s/smoke_test/weights/{best,last}.pt` | Implemented for YOLOv5s smoke; prepared for full training |
| Colab GPU training notebook | `notebooks/YOLOv5_COCO_Training_Colab.ipynb` | Google Colab | Implemented |
| Compact Colab transfer bundle | `src/colab_transfer.py`, `src/prepare_coco_colab.py` | `transfer/yolov5_colab_bundle.zip`, `artifacts/phase5a_colab_transfer_report.md` | Implemented |
| Exact Colab split reconstruction | `src/prepare_coco_colab.py`, `data/splits/*` | Official COCO download in Colab plus manifest reconstruction | Implemented |
| Colab training guard | `notebooks/YOLOv5_COCO_Training_Colab.ipynb` | `START_TRAINING = False`, guarded training cell | Implemented |
| Evaluation metrics | `src/evaluate_models.py`, `src/calculate_coco_metrics.py` | YOLOv5 validation outputs and COCOeval JSON | Prepared |
| Curves/confusion matrix/prediction samples | `src/evaluate_models.py` | YOLOv5 `--plots` outputs | Prepared |
| Image detection | `src/detect_image.py`, `src/training_smoke.py` | `outputs/images/training_smoke_test/bus.jpg` | Implemented for smoke checkpoint; prepared for final trained weights |
| Video detection | `src/detect_video.py` | `outputs/videos` | Prepared |
| Webcam detection | `src/detect_webcam.py` | `outputs/webcam` | Prepared |
| Robustness framework | `src/robustness_tests.py` | `results/robustness/generated` | Prepared |
| Pruning | `src/prune_model.py` | `models/optimized`, pruning report | Prepared |
| Quantization | `src/quantize_model.py` | ONNX INT8 outputs and report | Prepared |
| TorchScript/ONNX/FP16 export | `src/export_model.py` | YOLOv5 export artifacts | Prepared |
| Jetson deployment docs | `deployment/NVIDIA_JETSON.md` | edge commands and benchmark template | Implemented |
| Raspberry Pi deployment docs | `deployment/RASPBERRY_PI.md` | edge commands and benchmark template | Implemented |
| Report template | `reports/report_template.md` | final report draft | Implemented |
| README | `README.md` | project instructions | Implemented |
| Tests | `tests/` | `python -m pytest` | Implemented |
| Colab transfer manifest | `src/colab_transfer.py`, `artifacts/colab_training_transfer_manifest.*` | Compact bundle plus official COCO reconstruction strategy | Implemented |
