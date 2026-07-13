# Screenshot Evidence Checklist

The application workflows were physically verified through Flask, including image detection, video detection, and webcam detection. Actual screenshot image files are not fabricated by this repository audit. Copy the verified screenshots into this directory using the exact filenames below.

| Expected file | Page or route | Expected content |
|---|---|---|
| `01_home_dashboard.png` | `/` | Flask dashboard with image and video upload panels and model metric cards |
| `02_image_detection_result.png` | `/detect/image` result | Image detection result page with metrics and original/annotated comparison |
| `03_image_annotated_output.jpg` | `/outputs/images/...` | Downloaded or served annotated image output |
| `04_video_detection_result_top.png` | `/detect/video` result | Video result page top section with processed video player |
| `05_video_detection_result_summary.png` | `/detect/video` result | Video summary, class counts, latency, and processing FPS |
| `06_webcam_live_detection.png` | `/webcam` | Live webcam MJPEG stream with detection overlay |
| `07_model_information.png` | `/#model-information` | Model metadata, verified test metrics, and COCOeval values |
| `08_test_evaluation_metrics.png` | `results/evaluation/test_subset_2500/test_results_summary.md` or rendered equivalent | Genuine YOLOv5 labeled test metrics |
| `09_cocoeval_metrics.png` | `results/evaluation/test_subset_2500/coco_eval/coco_eval_summary.json` or rendered equivalent | Official COCOeval AP/AR metrics |
| `10_training_analysis.png` | `results/training_analysis/` or README training section | Training analysis plots or best-epoch summary |

Manual verification notes:

- Image detection was physically verified through Flask.
- Video detection was physically verified through Flask.
- Webcam detection was physically verified through Flask.
- Observed physical webcam performance: 2.4 FPS, 324.1 ms latency, 1 detected object.
- Browser video compatibility was repaired and verified with `artifacts/video_browser_compatibility_verification.json`.
