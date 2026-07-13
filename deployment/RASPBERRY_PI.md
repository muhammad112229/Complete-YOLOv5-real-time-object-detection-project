# Raspberry Pi Deployment Preparation

This document prepares deployment; it does not claim physical Raspberry Pi testing.

## Recommended Formats

- ONNX for ONNX Runtime CPU inference
- TorchScript only for very small smoke tests
- INT8 ONNX after calibration and accuracy validation

## Packages

Use a 64-bit Raspberry Pi OS image when possible. Install:

```bash
python -m pip install numpy pillow opencv-python onnxruntime
```

PyTorch availability depends on the OS and Python version. If PyTorch is not
available or is too slow, prefer ONNX Runtime with an exported/quantized model.

## Export Commands

Run on the development machine:

```powershell
python -m src.export_model --weights models/yolov5s_coco20k_best.pt --include onnx --imgsz 640 --device cpu
python -m src.quantize_model --input models/yolov5s_coco20k_best.onnx --output models/optimized/yolov5s_coco20k_best_int8.onnx --method dynamic-onnx
```

For static INT8 calibration:

```powershell
python -m src.quantize_model --input models/yolov5s_coco20k_best.onnx --output models/optimized/yolov5s_coco20k_best_static_int8.onnx --method static-onnx --calibration-dir data/processed/coco2017_yolo/images/val
```

## Inference Commands

If using PyTorch weights:

```bash
python deployment/infer_image.py --source sample.jpg --device cpu
```

For ONNX Runtime, add a dedicated ONNX inference runner before deployment if
PyTorch is not installed on the device.

## Camera Setup

Enable the camera interface and test capture:

```bash
libcamera-hello
```

USB cameras can be checked with:

```bash
v4l2-ctl --list-devices
```

## Benchmark Procedure

1. Record Raspberry Pi model, RAM, OS, Python version, and cooling state.
2. Warm up for at least 30 frames.
3. Measure latency and FPS on the target input resolution.
4. Save results in `deployment/edge_benchmark_template.csv`.
5. Do not report mAP changes unless evaluated with COCOeval.

## Known Limitations

- Full YOLOv5l is usually too heavy for Raspberry Pi CPU inference.
- Camera FPS may be limited by sensor mode and OpenCV backend.
- INT8 quantization can reduce accuracy; evaluate before reporting.
