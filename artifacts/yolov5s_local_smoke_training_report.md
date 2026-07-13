# YOLOv5s Local Smoke Training Report

Generated UTC: `2026-07-06T08:38:00.804516+00:00`

**Warning:** These metrics are produced from a tiny one-epoch diagnostic run and must not be interpreted as the final accuracy of the trained object-detection system.

## Recovery
- Recovery audit: `artifacts/phase4_recovery_audit.md`
- Complete before reconnection: smoke utility scaffolding, smoke config, 32/16 smoke subset, and smoke dataset validation.
- Failed before reconnection/resume completion: first YOLOv5 attempt failed on offline font download; second attempt failed before checkpoints on NumPy 2.x `trapz` removal.
- Completed after recovery: compatibility fixes, successful one-epoch smoke training, checkpoint validation, post-training inference, quality checks, Colab manifest, and documentation updates.

## Smoke Dataset
- Train images: 32
- Validation images: 16
- Annotations: 1257
- Represented classes: 80
- Dataset validation: PASS

## Training Configuration
- Command: `.\.venv\Scripts\python.exe -m src.train_models --config configs\train_yolov5s_smoke.yaml --device cpu`
- Model: `yolov5s`
- Weights: `models/pretrained/yolov5s.pt`
- Epochs: 1
- Batch size: 2
- Image size: 640
- Optimizer: SGD
- Device: cpu
- Workers: 0
- Seed: 42
- Elapsed seconds: 111.859

## Smoke-test diagnostic metrics -- not final performance results.
**Warning:** These metrics are produced from a tiny one-epoch diagnostic run and must not be interpreted as the final accuracy of the trained object-detection system.

Training losses:
- box_loss: 0.066526
- obj_loss: 0.28967
- cls_loss: 0.059732

Validation losses:
- box_loss: 0.043424
- obj_loss: 0.13474
- cls_loss: 0.023367

Diagnostic validation metrics:
- precision: 0.72668
- recall: 0.45811
- mAP_0.5: 0.57765
- mAP_0.5:0.95: 0.375

## Checkpoints
- best.pt: exists=True, size=14869928, load=passed
- last.pt: exists=True, size=14869928, load=passed
- Weight update vs pretrained: passed
- Reload inference: passed

## Post-Training Inference
- Status: passed
- Output: `G:\intership projects\Real-Time Object Detection using YOLOv5\code\outputs\images\training_smoke_test\bus.jpg`
- Detections: 5
- Classes: bus, person
- Preprocess ms: 7.99560546875
- Inference ms: 354.4504642486572
- NMS ms: 0.0

## Native Component Evidence
- pretrained_weights_loaded: PASS
- model_architecture_initialized: PASS
- training_images_discovered: PASS
- validation_images_discovered: PASS
- box_loss_executed: PASS
- objectness_loss_executed: PASS
- classification_loss_executed: PASS
- optimizer_configured: PASS
- validation_completed: PASS
- checkpoint_logged: PASS
- forward_pass_completed: PASS
- augmentation_pipeline_executed: PASS
- learning_rate_scheduling_executed: PASS
- backward_and_optimizer_update_completed: PASS
- best_checkpoint_written: PASS
- last_checkpoint_written: PASS

## Quality Checks
- compileall: passed
- pytest: 15 passed in 9.08s
- verify_setup.py: passed
- notebook_json_validation: passed
- smoke_dataset_validation: passed
- checkpoint_reload_validation: passed
- post_training_inference: passed

## Warnings
- G:\intership projects\Real-Time Object Detection using YOLOv5\code\external\yolov5\utils\general.py:34: UserWarning: pkg_resources is deprecated as an API. See https://setuptools.pypa.io/en/latest/pkg_resources.html. The pkg_resources package is slated for removal as early as 2025-11-30. Refrain from using this package or pin to Setuptools<81.
- G:\intership projects\Real-Time Object Detection using YOLOv5\code\external\yolov5\train.py:123: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
- G:\intership projects\Real-Time Object Detection using YOLOv5\code\external\yolov5\train.py:252: FutureWarning: `torch.cuda.amp.GradScaler(args...)` is deprecated. Please use `torch.amp.GradScaler('cuda', args...)` instead.
- 0%|          | 0/16 00:00G:\intership projects\Real-Time Object Detection using YOLOv5\code\external\yolov5\train.py:307: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
- 0/0         0G    0.06802     0.3185    0.03798        103        640:   6%|▋         | 1/16 00:10G:\intership projects\Real-Time Object Detection using YOLOv5\code\external\yolov5\train.py:307: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
- 0/0         0G     0.0686     0.3736    0.04276        167        640:  12%|█▎        | 2/16 00:13G:\intership projects\Real-Time Object Detection using YOLOv5\code\external\yolov5\train.py:307: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
- 0/0         0G    0.06732     0.3234    0.04442         78        640:  19%|█▉        | 3/16 00:17G:\intership projects\Real-Time Object Detection using YOLOv5\code\external\yolov5\train.py:307: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
- 0/0         0G     0.0677     0.3057    0.05199         60        640:  25%|██▌       | 4/16 00:20G:\intership projects\Real-Time Object Detection using YOLOv5\code\external\yolov5\train.py:307: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
- 0/0         0G    0.06738      0.296    0.05433        117        640:  31%|███▏      | 5/16 00:23G:\intership projects\Real-Time Object Detection using YOLOv5\code\external\yolov5\train.py:307: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
- 0/0         0G    0.06655     0.2797    0.05437        106        640:  38%|███▊      | 6/16 00:25G:\intership projects\Real-Time Object Detection using YOLOv5\code\external\yolov5\train.py:307: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
- 0/0         0G     0.0664      0.277    0.05367        106        640:  44%|████▍     | 7/16 00:28G:\intership projects\Real-Time Object Detection using YOLOv5\code\external\yolov5\train.py:307: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
- 0/0         0G    0.06661     0.2828    0.05419        137        640:  50%|█████     | 8/16 00:31G:\intership projects\Real-Time Object Detection using YOLOv5\code\external\yolov5\train.py:307: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
- 0/0         0G    0.06731     0.2896    0.05538        146        640:  56%|█████▋    | 9/16 00:34G:\intership projects\Real-Time Object Detection using YOLOv5\code\external\yolov5\train.py:307: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
- 0/0         0G    0.06663     0.2769    0.05518         48        640:  62%|██████▎   | 10/16 00:37G:\intership projects\Real-Time Object Detection using YOLOv5\code\external\yolov5\train.py:307: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
- 0/0         0G    0.06642     0.2825    0.05875         65        640:  69%|██████▉   | 11/16 00:39G:\intership projects\Real-Time Object Detection using YOLOv5\code\external\yolov5\train.py:307: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
- 0/0         0G    0.06663     0.2822    0.05942         99        640:  75%|███████▌  | 12/16 00:42G:\intership projects\Real-Time Object Detection using YOLOv5\code\external\yolov5\train.py:307: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
- 0/0         0G    0.06667     0.2896    0.05864        132        640:  81%|████████▏ | 13/16 00:45G:\intership projects\Real-Time Object Detection using YOLOv5\code\external\yolov5\train.py:307: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
- 0/0         0G    0.06648     0.2812     0.0598         53        640:  88%|████████▊ | 14/16 00:47G:\intership projects\Real-Time Object Detection using YOLOv5\code\external\yolov5\train.py:307: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
- 0/0         0G    0.06658     0.2827    0.05938        159        640:  94%|█████████▍| 15/16 00:50G:\intership projects\Real-Time Object Detection using YOLOv5\code\external\yolov5\train.py:307: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
- G:\intership projects\Real-Time Object Detection using YOLOv5\code\external\yolov5\utils\general.py:1004: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
- G:\intership projects\Real-Time Object Detection using YOLOv5\code\external\yolov5\models\experimental.py:79: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.

## Fixes Applied
- Resolved wrapper paths relative to the workspace before invoking YOLOv5 from external/yolov5.
- Added smoke-test overrides for dataset, output directory, batch size, epochs, workers, and console capture.
- Created deterministic hardlinked smoke dataset utilities and validation artifacts.
- Configured YOLOV5_CONFIG_DIR to a workspace-local font cache to avoid network font downloads.
- Patched YOLOv5 v7.0 Pillow 12 text sizing compatibility for plot generation.
- Patched YOLOv5 v7.0 NumPy 2.x AP integration compatibility.

## Full-Training Readiness
- Ready for full Colab training: YES
- Full local COCO training was not started.

Exact next recommended action, not executed:

Review `artifacts/colab_training_transfer_manifest.md`, then mount or upload the listed dataset/config/weight files in Google Drive and run only the Colab setup and integrity-check cells before starting GPU training.

## Final Response Summary

Recovery audit completed. The workspace root is not a Git repository, no active Python training process remained, and two failed diagnostic attempts were preserved instead of overwritten: one failed on YOLOv5 font download and one failed on NumPy 2.x `np.trapz` removal.

Found complete before resuming: smoke utility scaffolding, `configs/train_yolov5s_smoke.yaml`, the 32/16 hardlinked smoke subset, and smoke dataset validation. Completed after resuming: offline font handling, Pillow 12 and NumPy 2.x YOLOv5 compatibility fixes, successful one-epoch YOLOv5s CPU smoke training, checkpoint validation, post-training inference, quality checks, Colab transfer manifest, and documentation updates.

Smoke subset: 32 train images, 16 validation images, 1,257 labels, all 80 COCO classes represented, no leakage, YOLOv5 dataloader batch passed. Training command: `.\.venv\Scripts\python.exe -m src.train_models --config configs\train_yolov5s_smoke.yaml --device cpu`. Configuration: YOLOv5s, `models/pretrained/yolov5s.pt`, 1 epoch, batch size 2, image size 640, SGD, CPU, workers 0, seed 42, output `results/yolov5s/smoke_test`.

Training completed in 111.859 seconds. Forward pass, box/objectness/classification loss, backward pass, SGD optimizer update, LR scheduling, validation, and checkpoint writing all executed. Diagnostic metrics only, not final performance: precision 0.72668, recall 0.45811, mAP50 0.57765, mAP50-95 0.375; train losses 0.066526 box, 0.28967 obj, 0.059732 cls; validation losses 0.043424 box, 0.13474 obj, 0.023367 cls.

Checkpoints: `best.pt` and `last.pt` both exist, both are 14,869,928 bytes, both load successfully, class count is 80, and weight-update validation passed. Post-training inference on `external/yolov5/data/images/bus.jpg` passed with 5 detections (`bus`, `person`) and output saved to `outputs/images/training_smoke_test/bus.jpg`.

Quality checks: compileall passed, pytest `15 passed in 9.08s`, `verify_setup.py` passed, notebook JSON validation passed. Warnings documented: YOLOv5 `pkg_resources` deprecation, PyTorch `torch.load(weights_only=False)` safety warning, PyTorch AMP deprecation warnings, and harmless YOLOv5 Git warnings from spaces in the workspace path.

Files created or modified include `src/training_smoke.py`, `src/train_models.py`, `src/common.py`, `external/yolov5/utils/metrics.py`, `external/yolov5/utils/plots.py`, `configs/train_yolov5s_smoke.yaml`, `tests/test_training_smoke.py`, `tests/test_imports.py`, `data/smoke/coco_yolov5/`, `results/yolov5s/smoke_test/`, `outputs/images/training_smoke_test/`, `artifacts/phase4_recovery_audit.md`, `artifacts/yolov5s_local_smoke_training_report.md`, `artifacts/yolov5s_local_smoke_training_results.json`, `artifacts/colab_training_transfer_manifest.*`, `README.md`, `reports/requirements_traceability.md`, `artifacts/initial_setup_report.md`, `artifacts/dependency_installation_report.md`, and `artifacts/coco_preprocessing_report.md`.

Ready for full Google Colab training: YES. Exact next recommended action, not executed: review `artifacts/colab_training_transfer_manifest.md`, then mount or upload the listed dataset/config/weight files in Google Drive and run only the Colab setup and integrity-check cells before starting GPU training.
