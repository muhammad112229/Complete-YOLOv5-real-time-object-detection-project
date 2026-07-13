# COCO Dataset Validation

Generated UTC: `2026-07-05T08:01:54.259881+00:00`
Final readiness: PASS

## Rules
- no_image_source_key_leakage: PASS {'overlaps': 0}
- no_project_image_path_leakage: PASS {'overlaps': 0}
- no_label_path_leakage: PASS {'overlaps': 0}
- manifest_paths_exist: PASS {'errors': []}
- label_format_and_ranges: PASS {'error_count': 0, 'examples': []}
- dataset_yaml_paths_resolve: PASS {'path': 'G:\\intership projects\\Real-Time Object Detection using YOLOv5\\code\\data\\processed\\coco_yolo\\coco_project.yaml', 'dataset_root': 'G:\\intership projects\\Real-Time Object Detection using YOLOv5\\code\\data\\processed\\coco_yolo', 'name_count': 80, 'nc': 80, 'paths_ok': {'train': True, 'val': True, 'test': True}, 'valid': True}
- all_usable_images_open: PASS {'method': 'raw extraction validation opened every source image; processed images are hardlinks', 'corrupt_images': {'train2017': 0, 'val2017': 0}, 'missing_images': {'train2017': 0, 'val2017': 0}}
- coco_to_yolo_reversible: PASS {'failures': 0}
- split_ratios_close_to_80_10_10: PASS {'train': 0.7999951333068369, 'val': 0.09999432219130971, 'test': 0.1000105445018534}
- source_traceability_present: PASS {'manifest_rows': 123287, 'split_total': 123287}
- no_placeholder_paths: PASS {'files': []}
- all_labels_belong_to_single_split: PASS {}

## Excluded Records
- Non-accepted annotations: 10500
- iscrowd_excluded_for_yolov5_compatibility: 10498
- invalid_bbox:bbox has zero or negative area after clipping: 2
