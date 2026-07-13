# COCO Preprocessing Report

Generated UTC: `2026-07-05T08:03:37.668464+00:00`

## 1. Download Status

- `train2017.zip`: downloaded, verified, retained at `data/raw/archives/train2017.zip`
- `val2017.zip`: downloaded, verified, retained at `data/raw/archives/val2017.zip`
- `annotations_trainval2017.zip`: downloaded, verified, retained at `data/raw/archives/annotations_trainval2017.zip`

Official URLs used:

- `http://images.cocodataset.org/zips/train2017.zip`
- `http://images.cocodataset.org/zips/val2017.zip`
- `http://images.cocodataset.org/annotations/annotations_trainval2017.zip`

## 2. Archive and Extraction Verification

- Archive manifest: `artifacts/coco_download_manifest.json`
- Extraction validation: PASS
- `train2017` extracted images: 118,287
- `val2017` extracted images: 5,000
- Missing referenced images: 0
- Corrupt/unreadable images: 0
- Instance JSON files load with `pycocotools`: yes

`train2017.zip` was assembled from resumable segments and verified by successful `bsdtar` extraction plus marker; `val2017.zip` and annotations were verified with ZIP validation.

## 3. Source Counts

- Total source images: 123,287
- Total source annotations: 896,782
- Usable images: 123,287
- Accepted YOLO annotations: 886,282
- Rejected invalid annotations: 2
- Crowd annotations excluded from YOLO labels: 10,498

Crowd annotations are preserved in `data/interim/coco_combined_annotation_manifest.json` and excluded from YOLO training labels for YOLOv5 object-detection compatibility.

## 4. Split Counts

Deterministic image-level split, seed 42:

- Train: 98,629 images, 711,796 annotations, 79.9995%
- Validation: 12,328 images, 86,566 annotations, 9.9994%
- Test: 12,330 images, 87,920 annotations, 10.0011%

No image, project path, label path, or source-split/image-id key leakage was found.

## 5. COCO Mapping and YOLO Conversion

- COCO category mapping: PASS, 80 categories
- YOLO class indices: contiguous 0-79
- Dataset YAML: `data/processed/coco_yolo/coco_project.yaml`
- Colab template: `configs/coco_project_colab.yaml`
- COCO boxes were clipped safely to image bounds and converted to normalized YOLO `class x_center y_center width height`.
- Conversion reversibility validation: PASS, 0 failures

## 6. Storage Strategy

- Selected strategy: hardlink
- Processed images are linked under `data/processed/coco_yolo/images/{train,val,test}`.
- The full image dataset was not intentionally duplicated and was not permanently resized.
- YOLO label files are stored under `data/processed/coco_yolo/labels/{train,val,test}`.

Approximate storage:

- Drive free after phase: 341.43 GiB
- Approximate physical space used this phase: 39.69 GiB
- Archives: 19,460.06 MiB
- Annotation JSON files: 795.76 MiB
- YOLO labels: 40.21 MiB
- Interim manifests: 495.45 MiB
- Split manifests: 69.51 MiB

## 7. Letterbox Resizing

YOLOv5 applies native letterbox resizing during training and inference. It preserves aspect ratio, adds padding, and produces a 640 x 640 model input without distorting or duplicating the full COCO image set.

Examples saved under:

- `outputs/images/dataset_preprocessing/`

## 8. Statistics and Visuals

Statistics:

- `artifacts/coco_dataset_statistics.json`
- `artifacts/coco_dataset_statistics.csv`
- `artifacts/coco_dataset_statistics.md`

Charts:

- `results/dataset_analysis/class_wise_object_distribution.png`
- `results/dataset_analysis/split_image_counts.png`
- `results/dataset_analysis/split_annotation_counts.png`
- `results/dataset_analysis/objects_per_image_distribution.png`
- `results/dataset_analysis/bbox_area_distribution.png`
- `results/dataset_analysis/split_wise_class_distribution.png`

Visual validation samples:

- Train: 5 samples
- Validation: 3 samples
- Test: 3 samples
- Root: `outputs/images/dataset_validation/`

## 9. Validation Result

Dataset validation: PASS

Validation artifacts:

- `artifacts/coco_dataset_validation.json`
- `artifacts/coco_dataset_validation.md`

All validation rules passed, including label format/range checks, YAML path checks, split leakage checks, source traceability, and COCO-to-YOLO reversibility.

## 10. Tests and Loading Smoke

- Syntax compile: passed
- Unit tests: `12 passed in 17.14s`
- `verify_setup.py`: passed
- Notebook JSON validation: passed
- YOLOv5 dataset-loading smoke test: passed

Smoke-test artifact:

- `artifacts/coco_dataset_loading_smoke.json`

The YOLOv5 dataloader discovered a batch with tensor shape `[2, 3, 640, 640]` using the generated dataset layout.

## 11. Warnings and Exclusions

- No model training was run.
- No `yolov5m.pt` or `yolov5l.pt` download was performed.
- No mAP/AP/FPS/training metrics were generated.
- The initial large archive download required resumable segmented retries due network/tool execution limits.
- Two annotations were rejected because their boxes had zero or negative area after clipping.
- 10,498 `iscrowd` annotations were excluded from YOLO label files.

## 12. Training Readiness

Ready for YOLOv5 training: YES

Phase 4 local YOLOv5s smoke training has now been completed on a tiny 32/16
diagnostic subset. The full COCO dataset was not used for local training.

Smoke-training report:

- `artifacts/yolov5s_local_smoke_training_report.md`

Exact next recommended action, not executed:

```powershell
.\.venv\Scripts\python.exe -m json.tool notebooks\YOLOv5_COCO_Training_Colab.ipynb
```
