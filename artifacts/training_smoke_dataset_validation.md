# Training Smoke Dataset Validation

Generated UTC: `2026-07-06T08:12:52.657278+00:00`
Final readiness: PASS
Dataset YAML: `G:\intership projects\Real-Time Object Detection using YOLOv5\code\data\smoke\coco_yolov5\smoke_dataset.yaml`

## Counts
- Train images: 32
- Validation images: 16
- Annotations: 1257
- Represented classes: 80

## Rules
- train_image_count: PASS {'count': 32, 'expected': 32}
- val_image_count: PASS {'count': 16, 'expected': 16}
- every_image_has_label: PASS {'missing_labels': [], 'missing_count': 0}
- label_format_and_ranges: PASS {'annotation_count': 1257, 'error_count': 0, 'errors': []}
- no_smoke_split_leakage: PASS {'source_overlap': 0, 'filename_overlap': 0}
- dataset_yaml_resolves: PASS {'path': 'G:\\intership projects\\Real-Time Object Detection using YOLOv5\\code\\data\\smoke\\coco_yolov5', 'train': 'G:\\intership projects\\Real-Time Object Detection using YOLOv5\\code\\data\\smoke\\coco_yolov5\\images\\train', 'val': 'G:\\intership projects\\Real-Time Object Detection using YOLOv5\\code\\data\\smoke\\coco_yolov5\\images\\val', 'nc': 80, 'name_count': 80}
- native_yolov5_dataloader_batch: PASS {'status': 'passed', 'batch_image_tensor_shape': [2, 3, 640, 640], 'batch_label_tensor_shape': [48, 6], 'batch_paths': ['G:\\intership projects\\Real-Time Object Detection using YOLOv5\\code\\data\\smoke\\coco_yolov5\\images\\train\\train2017_000000010138.jpg', 'G:\\intership projects\\Real-Time Object Detection using YOLOv5\\code\\data\\smoke\\coco_yolov5\\images\\train\\train2017_000000024081.jpg'], 'dataset_length': 32}

## Represented Classes
- 0: person
- 1: bicycle
- 2: car
- 3: motorcycle
- 4: airplane
- 5: bus
- 6: train
- 7: truck
- 8: boat
- 9: traffic light
- 10: fire hydrant
- 11: stop sign
- 12: parking meter
- 13: bench
- 14: bird
- 15: cat
- 16: dog
- 17: horse
- 18: sheep
- 19: cow
- 20: elephant
- 21: bear
- 22: zebra
- 23: giraffe
- 24: backpack
- 25: umbrella
- 26: handbag
- 27: tie
- 28: suitcase
- 29: frisbee
- 30: skis
- 31: snowboard
- 32: sports ball
- 33: kite
- 34: baseball bat
- 35: baseball glove
- 36: skateboard
- 37: surfboard
- 38: tennis racket
- 39: bottle
- 40: wine glass
- 41: cup
- 42: fork
- 43: knife
- 44: spoon
- 45: bowl
- 46: banana
- 47: apple
- 48: sandwich
- 49: orange
- 50: broccoli
- 51: carrot
- 52: hot dog
- 53: pizza
- 54: donut
- 55: cake
- 56: chair
- 57: couch
- 58: potted plant
- 59: bed
- 60: dining table
- 61: toilet
- 62: tv
- 63: laptop
- 64: mouse
- 65: remote
- 66: keyboard
- 67: cell phone
- 68: microwave
- 69: oven
- 70: toaster
- 71: sink
- 72: refrigerator
- 73: book
- 74: clock
- 75: vase
- 76: scissors
- 77: teddy bear
- 78: hair drier
- 79: toothbrush
