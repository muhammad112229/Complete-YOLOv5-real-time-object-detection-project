# Dependency Installation and Inference Verification Report

Generated for:

`G:\intership projects\Real-Time Object Detection using YOLOv5\code`

## 1. Python and Virtual Environment Status

- Python executable: `G:\intership projects\Real-Time Object Detection using YOLOv5\code\.venv\Scripts\python.exe`
- Python version: `3.11.9`
- Virtual environment path: `G:\intership projects\Real-Time Object Detection using YOLOv5\code\.venv`
- pip after upgrade: `26.1.2`
- setuptools after compatibility pin: `80.10.2`
- wheel after upgrade: `0.47.0`

Commands used:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pip freeze > artifacts\installed_packages.txt
```

## 2. PyTorch Installation Type and Version

- Installation type: CPU
- torch: `2.5.1+cpu`
- torchvision: `0.20.1+cpu`
- CUDA runtime in PyTorch: `None`
- `torch.cuda.is_available()`: `False`
- Selected local device: `cpu`
- CUDA tensor check: not run because CUDA is unavailable in the installed CPU build

Reason for CPU build:

The detected NVIDIA GeForce 840M is listed by NVIDIA as a legacy compute-capability 5.0 GPU. PyTorch's current install guidance says to choose a CUDA build only when a CUDA-capable system and suitable CUDA version are available, and to verify CUDA with `torch.cuda.is_available()`. A current, safe Windows/Python 3.11 CUDA PyTorch wheel combination for this specific legacy GPU was not verified, so the project uses official CPU wheels for local VS Code inference and smoke tests.

References:

- PyTorch install and CUDA verification guidance: https://pytorch.org/get-started/locally/
- NVIDIA legacy GPU compute capability table: https://developer.nvidia.com/cuda/gpus/legacy

## 3. GPU Compatibility Conclusion

GPU audit:

- `nvidia-smi`: NVIDIA GeForce 840M
- Driver: `581.57`
- Driver-reported CUDA version: `13.0`
- VRAM: `2048 MiB`
- `nvcc`: unavailable
- PowerShell GPU info: NVIDIA GeForce 840M, driver `32.0.15.8157`, adapter RAM `2147483648`

Conclusion:

- The GPU is detected by the NVIDIA driver.
- Local PyTorch CUDA is not enabled because the installed PyTorch build is CPU-only.
- GPU support was not claimed.
- Local development and inference use CPU.
- Full YOLOv5 training remains planned for Google Colab GPU.
- This does not block the next dataset-preparation phase.

## 4. OpenCV and pycocotools Status

- OpenCV: `5.0.0`
- pycocotools: available, package version `2.0.11`
- NumPy: `2.4.6`
- pandas: `3.0.3`
- Matplotlib: `3.11.0`
- PyYAML: `6.0.3`
- Pillow: `12.3.0`
- tqdm: `4.68.3`
- ONNX: `1.22.0`
- ONNX Runtime: `1.27.0`
- pytest: `9.1.1`

Full environment package freeze:

- `artifacts/installed_packages.txt`

## 5. Dependency Pins

Compatibility pins added to `requirements.txt`:

- `torch==2.5.1+cpu`
- `torchvision==0.20.1+cpu`
- `--extra-index-url https://download.pytorch.org/whl/cpu`
- `setuptools<81`

Pin rationale:

- CPU PyTorch avoids installing an unverified CUDA build on a legacy GeForce 840M.
- PyTorch 2.5.1 stays below the PyTorch 2.6 `torch.load` default behavior change that can affect legacy YOLOv5 checkpoint loading.
- YOLOv5 v7.0 imports `pkg_resources`; newer setuptools removed or no longer provides it in this environment, so `setuptools<81` is required.

## 6. Errors Encountered and Fixes

Issue:

```text
ModuleNotFoundError: No module named 'pkg_resources'
```

Cause:

YOLOv5 v7.0 imports `pkg_resources` in `external/yolov5/utils/general.py`, but the newest setuptools installed during the packaging-tool upgrade did not provide it.

Fix:

Pinned and installed `setuptools<81`, resulting in `setuptools==80.10.2`.

Remaining warning:

YOLOv5 emits a deprecation warning for `pkg_resources`. This is expected for YOLOv5 v7.0 and is documented by the pin.

## 7. YOLOv5s Weight Status

- Downloaded only: `models/pretrained/yolov5s.pt`
- Size: `14,808,437` bytes (`14.12 MB`)
- `yolov5m.pt`: not downloaded
- `yolov5l.pt`: not downloaded
- `.gitignore` excludes `models/pretrained/*`, `models/trained/*`, `models/optimized/*`, and `*.pt`

Download source:

```text
https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt
```

## 8. Direct YOLOv5 Smoke-Test Result

Command:

```powershell
.\.venv\Scripts\python.exe external\yolov5\detect.py --weights models\pretrained\yolov5s.pt --source external\yolov5\data\images\bus.jpg --imgsz 640 --conf-thres 0.25 --device cpu --project outputs\images --name smoke_test --exist-ok
```

Result:

- Status: passed
- Output image: `outputs/images/smoke_test/bus.jpg`
- Preprocess time: `0.0 ms`
- Inference time: `382.3 ms`
- NMS time: `4.0 ms`
- Detection count: `5`
- Detected classes: `bus`, `person`
- Raw YOLOv5 summary: `4 persons, 1 bus`

The timing values above are from the actual run recorded in `artifacts/inference_smoke_test.json`. This is a pipeline smoke test, not a performance benchmark.

## 9. Custom Pipeline Smoke-Test Result

Command:

```powershell
.\.venv\Scripts\python.exe -m src.detect_image --weights models\pretrained\yolov5s.pt --source external\yolov5\data\images\bus.jpg --output outputs\images\custom_smoke_test --conf-thres 0.25 --iou-thres 0.45 --device cpu --imgsz 640
```

Result:

- Status: passed
- Output image: `outputs/images/custom_smoke_test/bus.jpg`
- Total inference time: `421.66 ms`
- FPS: `2.37`
- Detection count: `5`
- Detected classes: `bus`, `person`

Smoke-test artifacts:

- `artifacts/inference_smoke_test.json`
- `artifacts/inference_smoke_test.md`

## 10. Quality Checks

All commands used `.venv\Scripts\python.exe`.

| Check | Result |
|---|---|
| `python -m compileall -q src tests verify_setup.py` | Passed |
| `python -m pytest` | `7 passed in 11.46s` |
| `python verify_setup.py` | Passed |
| YOLOv5 import check | Passed |
| Direct YOLOv5 smoke test | Passed |
| Custom inference smoke test | Passed |
| Notebook JSON validation | Passed |
| `python -m pip check` | No broken requirements |

Non-Python commands used:

- `nvidia-smi`: required to query GPU, driver, CUDA version, and VRAM from the NVIDIA driver.
- `git -C external/yolov5 ...`: required to verify the cloned YOLOv5 tag and commit.
- `Get-CimInstance Win32_VideoController`: required as a Windows GPU audit fallback.
- `nvcc --version`: required to check whether a local CUDA toolkit is installed.

## 11. Files Created or Modified

Created:

- `artifacts/installed_packages.txt`
- `artifacts/dependency_installation_report.md`
- `artifacts/inference_smoke_test.json`
- `artifacts/inference_smoke_test.md`
- `models/pretrained/yolov5s.pt`
- `outputs/images/smoke_test/bus.jpg`
- `outputs/images/custom_smoke_test/bus.jpg`

Modified:

- `requirements.txt`
- `verify_setup.py`
- `src/inference.py`
- `src/detect_image.py`
- `README.md`
- `artifacts/environment_audit.json`
- `artifacts/environment_audit.md`
- `artifacts/initial_setup_report.md`

## 12. Warnings and Blockers

- CUDA is unavailable to local PyTorch; CPU is the selected local device.
- The NVIDIA GeForce 840M has only 2 GB VRAM and remains unsuitable for full COCO training.
- YOLOv5 emits a harmless Git warning because the workspace path contains spaces.
- YOLOv5 emits a `pkg_resources` deprecation warning; the `setuptools<81` pin keeps YOLOv5 v7.0 functional.
- COCO 2017 has not been downloaded.
- No full training was run.
- No webcam test was run.
- No YOLOv5m or YOLOv5l weights were downloaded.
- No metrics, FPS comparisons, or deployment results were fabricated.

## 13. Ready for Full COCO Download

Yes. The local runtime environment and image inference smoke path are verified. The project is ready for the next phase: COCO download and dataset preparation, after explicit approval for the large download.

Exact next command, not executed:

```powershell
.\.venv\Scripts\python.exe -m src.download_coco --confirm-large-download
```

## 14. COCO Preparation Follow-Up

The COCO 2017 dataset has now been downloaded, processed, and validated.

- Dataset YAML: `data/processed/coco_yolo/coco_project.yaml`
- Dataset validation: PASS
- YOLOv5 dataloader smoke test: PASS
- Primary report: `artifacts/coco_preprocessing_report.md`

## 15. Local Training Smoke-Test Follow-Up

The local YOLOv5s training wrapper has now been verified with a tiny one-epoch CPU smoke test.

- Command: `.\.venv\Scripts\python.exe -m src.train_models --config configs\train_yolov5s_smoke.yaml --device cpu`
- Smoke dataset: `data/smoke/coco_yolov5`
- Run directory: `results/yolov5s/smoke_test`
- Checkpoints: `weights/best.pt` and `weights/last.pt`, both loadable
- Post-training inference output: `outputs/images/training_smoke_test/bus.jpg`
- Final quality checks: compileall passed, pytest `15 passed in 9.08s`, `verify_setup.py` passed, notebook JSON validation passed
- Primary report: `artifacts/yolov5s_local_smoke_training_report.md`

Warnings remain expected for YOLOv5 v7.0 with current dependencies:

- `pkg_resources` deprecation warning from YOLOv5 v7.0.
- PyTorch `torch.load(weights_only=False)` safety warning.
- PyTorch AMP deprecation warnings on CPU paths.
- Harmless YOLOv5 Git status warnings caused by the workspace path containing spaces.
