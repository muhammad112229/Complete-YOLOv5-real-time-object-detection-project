# Real-Time Object Detection using YOLOv5

## Title Page

Project title, author, institution, supervisor, date, and version placeholder.

## Abstract

Placeholder for a concise summary of objectives, dataset, methods, measured results, and limitations.

## Introduction

Placeholder for the project background and motivation.

## Objectives

- Build a reproducible YOLOv5-based real-time object detection pipeline.
- Use COCO 2017 as the primary dataset.
- Compare YOLOv5s, YOLOv5m, and YOLOv5l after actual training/evaluation.
- Prepare optimization and edge-deployment workflows.

## Dataset Description

COCO 2017 train and validation images with `instances_train2017.json` and
`instances_val2017.json` annotations. Full download and processing are pending
explicit approval.

## Preprocessing

COCO bounding boxes are converted from pixel `[x, y, width, height]` format to
normalized YOLO `class_id x_center y_center width height`. The project split is
deterministic at image level with seed 42.

## YOLOv5 Architecture

Placeholder for YOLOv5 backbone, neck, and detection head description.

## Model Variants

- YOLOv5s
- YOLOv5m
- YOLOv5l

## Training Setup

Placeholder for actual hardware, epochs, batch sizes, optimizer, learning-rate
schedule, augmentation settings, and early stopping configuration.

## Optimizer and Loss

YOLOv5 native CIoU box loss, objectness loss, and classification loss are used.
SGD and AdamW experiments are prepared.

## Augmentation

Placeholder for YOLOv5 v7.0 augmentation configuration and any experiment-specific changes.

## Evaluation Metrics

Precision, recall, F1-score, mAP@0.5, mAP@0.5:0.95, AP50, AP75, class-wise AP,
preprocessing time, inference time, NMS time, FPS, model size, and parameter count.

## Experimental Results

No metrics have been generated yet. Add measured results only after training and evaluation.

## Speed Analysis

Placeholder for measured preprocessing, inference, NMS, and end-to-end FPS.

## Robustness Testing

Placeholder for low lighting, bright lighting, occlusion, multiple objects, small
objects, scale variation, motion blur, and complex background measurements.

## Pruning

Placeholder for pruning method, sparsity, fine-tuning, evaluation, and before/after comparison.

## Quantization

Placeholder for TorchScript/ONNX export, dynamic/static INT8 quantization,
calibration data, evaluation, and before/after comparison.

## Edge Deployment

Placeholder for NVIDIA Jetson and Raspberry Pi deployment preparation, commands,
benchmarking procedure, and hardware limitations.

## Limitations

Placeholder for dataset, compute, latency, hardware, and deployment limitations.

## Conclusion

Placeholder for final conclusions after experiments.

## References

Placeholder for YOLOv5, COCO, PyTorch, OpenCV, pycocotools, and deployment references.

