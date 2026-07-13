# Test Subset 2500 Evaluation Results

These are genuine labeled test-set metrics computed from `data/splits/test_subset_2500_seed42.txt`.
YOLOv5 label-based metrics and official COCOeval metrics are reported separately.

## YOLOv5 Label-Based Metrics

- Test images: 2500
- Labeled instances: 17751
- Precision: 0.68
- Recall: 0.553
- mAP@0.5: 0.61
- mAP@0.5:0.95: 0.38
- Inference time: 363.1 ms/image
- NMS time: 15.1 ms/image

## Strongest Classes by YOLOv5 mAP@0.5:0.95

- giraffe: 0.666
- zebra: 0.639
- fire hydrant: 0.606
- cat: 0.599
- bear: 0.597

## Weakest Classes by YOLOv5 mAP@0.5:0.95

- hair drier: 0.0381
- handbag: 0.097
- book: 0.11
- knife: 0.137
- backpack: 0.15

## Official COCOeval Metrics

- AP@[0.50:0.95]: 0.378696
- AP@0.50: 0.607692
- AP@0.75: 0.404071
- AP small: 0.147706
- AP medium: 0.376504
- AP large: 0.523755
- AR maxDets=1: 0.302949
- AR maxDets=10: 0.525675
- AR maxDets=100: 0.573531
- AR small: 0.313309
- AR medium: 0.592047
- AR large: 0.700708

## Annotation JSON Verification

- Annotation JSON: data/processed/coco_yolo/annotations/instances_test_subset_2500_seed42.json
- Verification status: passed
- Selected images: 2500
- Selected annotations: 17751
- Categories: 80
- Excluded crowd annotations: 195
- Rejected invalid boxes: 0
- Annotation JSON SHA256: ea3378d56591681f1eb8f82de6c2632ccab6b40d633a245a528a298af5689cd6

## Metric-System Notes

- YOLOv5 metrics are computed from YOLO label files inside `external/yolov5/val.py`.
- COCOeval metrics are computed by pycocotools from COCO-format ground truth and converted prediction JSON.
- Both systems use the same exact 2,500-image labeled test subset.
- Small numerical differences are expected because COCOeval reports the official AP/AR protocol, including area ranges and max-detection settings.
- COCOeval status: completed
