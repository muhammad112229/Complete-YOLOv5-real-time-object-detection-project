# NVIDIA Jetson Deployment Preparation

This document prepares deployment; it does not claim physical Jetson testing.

## Recommended Formats

- PyTorch `.pt` for development checks
- TorchScript for PyTorch inference
- ONNX for ONNX Runtime
- TensorRT engine if built on the target Jetson
- FP16 where supported by the Jetson GPU

## Packages

Install JetPack first, then use Python 3.8/3.10 depending on the JetPack release.
Install compatible builds of PyTorch, TorchVision, OpenCV, NumPy, PyYAML, and ONNX Runtime.

## Export Commands

From the project root on the development machine:

```powershell
python -m src.export_model --weights models/yolov5s_coco20k_best.pt --include torchscript onnx --imgsz 640 --device cpu
```

For FP16 export on supported CUDA hardware:

```powershell
python -m src.export_model --weights models/yolov5s_coco20k_best.pt --include onnx --imgsz 640 --device 0 --half
```

## Inference Commands

```bash
python deployment/infer_image.py --source sample.jpg --device 0
python deployment/infer_video.py --source sample.mp4 --device 0
python deployment/infer_webcam.py --camera-index 0 --device 0
```

## Camera Setup

Verify the CSI or USB camera first:

```bash
v4l2-ctl --list-devices
```

Use the correct camera index or a GStreamer pipeline if required by the camera.

## Benchmark Procedure

1. Record JetPack version, power mode, clock mode, model format, and precision.
2. Warm up the model for at least 30 frames.
3. Measure preprocessing, inference, NMS, and end-to-end FPS.
4. Save values in `deployment/edge_benchmark_template.csv`.
5. Run COCO validation if accuracy is being reported.

## Known Limitations

- TensorRT engines are hardware and software-version specific.
- FP16 speedups depend on GPU capability and export path.
- mAP/AP50/AP75 must be recalculated after pruning or quantization.
