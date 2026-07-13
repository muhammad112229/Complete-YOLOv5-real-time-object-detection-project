# Inference Smoke Test

Generated UTC: `2026-07-04T11:15:31.994802+00:00`
Device: `cpu`
Weights: `G:\intership projects\Real-Time Object Detection using YOLOv5\code\models\pretrained\yolov5s.pt` (14.12 MB)
Source image: `G:\intership projects\Real-Time Object Detection using YOLOv5\code\external\yolov5\data\images\bus.jpg`

## Direct YOLOv5
- Status: passed
- Output image: `G:\intership projects\Real-Time Object Detection using YOLOv5\code\outputs\images\smoke_test\bus.jpg`
- Preprocess: 0.0 ms
- Inference: 382.3 ms
- NMS: 4.0 ms
- Detection count: 5
- Classes: bus, person
- Raw summary: 4 persons, 1 bus

## Custom Pipeline
- Status: passed
- Output image: `G:\intership projects\Real-Time Object Detection using YOLOv5\code\outputs\images\custom_smoke_test\bus.jpg`
- Total inference: 421.66 ms
- FPS: 2.37
- Detection count: 5
- Classes: bus, person

## Notes
- This is a pipeline smoke test, not a performance benchmark.
- No COCO dataset download or training was run.
- YOLOv5 emits a harmless Git path warning because the workspace path contains spaces.
