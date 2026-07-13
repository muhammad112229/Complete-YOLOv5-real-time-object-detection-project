# Data Directory

This project uses COCO 2017 as the primary dataset.

Expected raw files after approved download:

- `data/raw/archives/train2017.zip`
- `data/raw/archives/val2017.zip`
- `data/raw/archives/annotations_trainval2017.zip`
- `data/raw/coco2017/train2017/`
- `data/raw/coco2017/val2017/`
- `data/raw/coco2017/annotations/instances_train2017.json`
- `data/raw/coco2017/annotations/instances_val2017.json`

The preprocessing pipeline converts COCO boxes from `[x, y, width, height]`
pixels into normalized YOLO labels:

```text
class_id x_center y_center width height
```

The full dataset is not duplicated or physically resized. YOLOv5 applies
letterbox resizing to 640 x 640 during training and inference. Processed image
paths under `data/processed/coco_yolo/images/` use hard links when supported, so
the YOLOv5 layout works without duplicating the full dataset bytes.

Current prepared dataset:

- Source images: 123,287
- Accepted YOLO annotations: 886,282
- Train/validation/test split: 98,629 / 12,328 / 12,330 images
- Dataset YAML: `data/processed/coco_yolo/coco_project.yaml`
- Validation report: `artifacts/coco_dataset_validation.md`
