# Real-Time Object Detection using YOLOv5

Author: Muhammad Hamzala

## 1. Project Title

Real-Time Object Detection using YOLOv5

## 2. Overview

This project implements a reproducible YOLOv5 object detection workflow using COCO 2017 data, a fine-tuned YOLOv5s production checkpoint, command-line inference tools, genuine labeled test evaluation, official COCOeval metrics, and a Flask visualization application for image, video, and webcam detection.

The local Windows environment uses CPU PyTorch. The application is usable locally, but CPU video and webcam processing should be interpreted as functional inference, not high-FPS real-time performance.

## 3. Key Features

- YOLOv5 v7.0 pinned under `external/yolov5`
- Production checkpoint: `models/yolov5s_coco20k_best.pt`
- Image upload detection with annotated output
- Uploaded video frame-by-frame detection with browser-compatible MP4 output
- Webcam MJPEG detection stream
- Confidence threshold, IoU threshold, and optional class filter controls
- Downloadable processed outputs
- JSON result metadata and API endpoints
- Genuine 2,500-image labeled test evaluation
- Official pycocotools COCOeval on the same test subset
- Windows-compatible paths through `pathlib`

## 4. Project Architecture

```text
COCO 2017 data
  -> preprocessing and YOLO labels
  -> deterministic train/val/test split
  -> 20,000-image training subset experiment
  -> YOLOv5s production checkpoint
  -> CLI inference and Flask application
  -> YOLOv5 labeled test metrics and COCOeval metrics
```

Core modules:

- `src/inference_engine.py`: shared YOLOv5 inference engine
- `app.py`: Flask web application factory and routes
- `src/webcam_stream.py`: webcam MJPEG stream wrapper
- `src/media_utils.py`: upload and media validation helpers
- `src/result_store.py`: JSON result metadata store
- `src/video_compatibility.py`: browser MP4 compatibility handling

## 5. Dataset and Preprocessing

Prepared COCO split:

- Train: 98,629 images
- Validation: 12,328 images
- Test: 12,330 images
- Classes: 80 COCO classes
- Dataset YAML: `data/processed/coco_yolo/coco_project.yaml`

Exact faster test subset:

- Manifest: `data/splits/test_subset_2500_seed42.txt`
- Size: 2,500 images
- Sampling: `random.Random(42).sample(full_test_paths, 2500)`
- Verification: `artifacts/test_subset_2500_verification.json`

Generated COCO annotation JSON for the test subset:

- `data/processed/coco_yolo/annotations/instances_test_subset_2500_seed42.json`
- Verification: `artifacts/test_subset_coco_annotation_verification.json`

## 6. Training Configuration

The production model was fine-tuned on a deterministic 20,000-image COCO subset and validated on 2,500 images.

- Model: YOLOv5s
- YOLOv5 version: v7.0
- Configured epochs: 6
- Completed epochs in artifacts: 4
- Image size: 640
- Training artifacts: `artifacts/trained_coco20k_yolov5s/`
- Production checkpoint: `models/yolov5s_coco20k_best.pt`
- Checkpoint SHA256: `8e75b741661acbd882cad802ab2d177cb636a0327af74dca944fbd3db5e10ee6`

Best validation metrics from training artifacts:

- Best epoch: 0
- Precision: 0.70704
- Recall: 0.55338
- mAP@0.5: 0.61301
- mAP@0.5:0.95: 0.38618

These are validation metrics from the training run, not final test metrics.

## 7. Genuine Evaluation Results

Genuine YOLOv5 labeled test metrics were computed on the exact deterministic 2,500-image test subset.

- Images: 2,500
- Labeled instances: 17,751
- Precision: 0.680
- Recall: 0.553
- mAP@0.5: 0.610
- mAP@0.5:0.95: 0.380
- Preprocess: 4.4 ms/image
- Inference: 363.1 ms/image on CPU
- NMS: 15.1 ms/image

Artifacts:

- `results/evaluation/test_subset_2500/metrics_summary.json`
- `results/evaluation/test_subset_2500/per_class_metrics.csv`
- `results/evaluation/test_subset_2500/test_results_summary.md`

## 8. Official COCOeval Results

Official pycocotools COCOeval was run on the same exact 2,500-image labeled test subset.

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

Artifacts:

- `results/evaluation/test_subset_2500/coco_eval/coco_eval_summary.json`
- `results/evaluation/test_subset_2500/coco_eval/coco_predictions.json`
- `results/evaluation/test_subset_2500/coco_eval/coco_eval_metadata.json`

## 9. Flask Application

Application config: `configs/app.yaml`

Start the Flask app:

```powershell
.\.venv\Scripts\python.exe app.py
```

Default URL:

```text
http://127.0.0.1:5000
```

The app loads the production model once at startup when run as `python app.py`. Debug mode and the Flask reloader are disabled by default.

## 10. Image Inference

Flask workflow:

1. Open `/`.
2. Select an image.
3. Set confidence, IoU, and optional class filter.
4. Submit and download the annotated output.

CLI workflow:

```powershell
.\.venv\Scripts\python.exe deployment\infer_image.py --source external\yolov5\data\images\bus.jpg
```

Evidence:

- `outputs/flask/metadata/7fab10546d2047d8bab5b889391ed1ba.json`
- `outputs/flask/images/7fab10546d2047d8bab5b889391ed1ba_bus_annotated.jpg`

## 11. Video Inference

Flask video uploads are processed frame by frame. The source video FPS is preserved when valid, while processing FPS is reported separately and reflects local CPU throughput.

Browser playback compatibility:

- Original annotated Flask MP4 codec: FMP4
- Browser-compatible output codec: h264
- Compatibility method: OpenCV `avc1` transcode
- Verification: `artifacts/video_browser_compatibility_verification.json`

Evidence:

- `outputs/flask/metadata/2de3cad407914270a34508e48cf148cf.json`
- `outputs/flask/videos/2de3cad407914270a34508e48cf148cf_WhatsApp_Video_2026-07-13_at_8_49_06_PM_annotated_browser.mp4`

## 12. Webcam Inference

Webcam routes:

- `/webcam`
- `/video_feed`
- `/webcam/stop`

Physical webcam testing was completed through Flask. Observed physical webcam performance:

- FPS: 2.4
- Latency: 324.1 ms
- Objects: 1

These values are manually observed physical-test values and are stored in `results/final_project_metrics.json`.

## 13. CLI Usage

Image:

```powershell
.\.venv\Scripts\python.exe deployment\infer_image.py --source path\to\image.jpg
```

Video:

```powershell
.\.venv\Scripts\python.exe deployment\infer_video.py --source path\to\video.mp4
```

Webcam:

```powershell
.\.venv\Scripts\python.exe deployment\infer_webcam.py --camera-index 0
```

Evaluation readiness:

```powershell
.\.venv\Scripts\python.exe deployment\evaluate_model.py --readiness-only --config configs\evaluation.yaml
```

Training analysis:

```powershell
.\.venv\Scripts\python.exe deployment\analyze_training_results.py
```

## 14. API Endpoints

- `GET /api/health`
- `GET /api/model`
- `GET /api/results/<result_id>`
- `GET /outputs/<path:filename>`

## 15. Installation

Use Python 3.11 on Windows.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python verify_setup.py
```

## 16. Running Locally

Run the Flask application:

```powershell
.\.venv\Scripts\python.exe app.py
```

Run the setup verifier:

```powershell
.\.venv\Scripts\python.exe verify_setup.py
```

Run an inference smoke test:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_inference_smoke.py -q
```

## 17. Testing

Focused tests:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_flask_app.py tests\test_video_compatibility.py tests\test_inference_smoke.py tests\test_evaluation_pipeline.py tests\test_imports.py -q
```

Full suite:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Final test report:

- `artifacts/final_test_report.json`

## 18. Output Structure

- Uploaded files: `uploads/`
- Flask image outputs: `outputs/flask/images/`
- Flask video outputs: `outputs/flask/videos/`
- Flask metadata: `outputs/flask/metadata/`
- CLI inference outputs: `outputs/inference/`
- Training analysis: `results/training_analysis/`
- Final evaluation: `results/evaluation/test_subset_2500/`
- Final metrics: `results/final_project_metrics.json`
- Final summary: `reports/final_project_summary.md`

Screenshots:

- Checklist: `reports/screenshots/README.md`
- Expected screenshot files are documented there and should be copied manually. No screenshots are fabricated by the repository audit.

## 19. Deployment Notes

- CPU inference is supported locally.
- GPU acceleration requires a compatible CUDA PyTorch environment.
- Full training should be performed on Colab or a modern CUDA GPU.
- The local Flask server is suitable for development and demonstration. Use a production WSGI server for deployment.
- Large data and generated outputs should be excluded from Git or stored with Git LFS according to repository policy.

## 20. Limitations

- CPU video and webcam inference are slow.
- Physical webcam values are manually observed and not produced by automated tests.
- The repository contains large datasets and generated artifacts locally; GitHub submission should follow the size report recommendations.
- Training completed 4 epochs although 6 were configured; no unsupported early-stopping claim is made.

## 21. Reproducibility

- Seed: 42
- YOLOv5 tag: v7.0
- Image size: 640
- Production checkpoint SHA256: `8e75b741661acbd882cad802ab2d177cb636a0327af74dca944fbd3db5e10ee6`
- Test subset manifest SHA256: `3e73eb09d1ef3eb6d41051e11efd4570e59c9808cf56d06a4da426780debe605`
- COCO subset annotation SHA256: `ea3378d56591681f1eb8f82de6c2632ccab6b40d633a245a528a298af5689cd6`

## 22. Repository Structure

```text
app.py          Flask application entrypoint
configs/        Project, inference, evaluation, and app YAML configs
data/           Local COCO data, split manifests, and generated labels
deployment/     CLI wrappers and edge deployment notes
external/       Pinned YOLOv5 v7.0 source
models/         Production and pretrained model checkpoints
notebooks/      Colab training notebook
outputs/        Flask and CLI inference outputs
reports/        Final summaries, screenshot checklist, evidence index
results/        Training analysis, evaluation, and final metrics
src/            Reusable project modules
static/         Flask CSS and JavaScript
templates/      Flask HTML templates
tests/          Pytest test suite
transfer/       Colab transfer bundle and documentation
```

## 23. Author

Muhammad Hamzala
