# Colab Training Transfer Manifest

Generated UTC: `2026-07-06T09:33:38.3849653Z`

## Selected Strategy

Use a compact transfer bundle. Do not upload the full local COCO image dataset,
processed image links, local COCO ZIP archives, smoke-test checkpoints, or cache
directories.

Colab reconstructs the verified dataset by:

1. Extracting `transfer/yolov5_colab_bundle.zip`.
2. Downloading official COCO 2017 archives directly in Colab.
3. Reusing valid downloaded archives after reconnects.
4. Rebuilding the exact train, validation, and test image-level split from the
   transferred manifests.
5. Regenerating YOLO labels from COCO annotations with the same crowd exclusion,
   invalid-box rejection, and 0-79 contiguous class mapping.
6. Comparing counts and split identity hashes against
   `artifacts/colab_reference_checksums.json`.

## Required Uploads

- `transfer/yolov5_colab_bundle.zip`
- `transfer/yolov5_colab_bundle.sha256`
- `notebooks/YOLOv5_COCO_Training_Colab.ipynb` for convenient opening in Colab

Recommended Google Drive destination:

- `MyDrive/YOLOv5_COCO_Project/bundles/`

## Bundle Contents

The bundle contains portable source and metadata only:

- Colab notebook
- `src/prepare_coco_colab.py`
- `src/colab_transfer.py`
- supporting source modules
- Colab and local configuration YAML files
- train, validation, and test manifests
- split summary
- class mapping and class names
- dataset statistics and validation metadata
- reference checksum artifacts
- README and dependency files

## Explicit Exclusions

- `.venv`
- `data/raw`
- `data/processed/coco_yolo/images`
- `data/processed/coco_yolo/labels`
- `data/smoke`
- `external/yolov5/.git`
- `models/pretrained/*.pt`
- `models/trained`
- `models/optimized`
- `results/yolov5s/smoke_test`
- output videos, webcam captures, caches, and Python bytecode

## Reference Counts

- Train images: 98,629
- Validation images: 12,328
- Test images: 12,330
- Accepted annotations: 886,282
- Excluded crowd annotations: 10,498
- Rejected invalid boxes: 2
- Classes: 80
- Split seed: 42

## Colab Storage Modes

- Runtime mode: dataset cache and reconstructed YOLO dataset under
  `/content/datasets/coco2017`; fastest setup, lost when the Colab runtime is
  reset.
- Drive mode: dataset cache and reconstructed YOLO dataset under Google Drive;
  slower I/O, persists across Colab disconnects.

Training outputs are configured to use:

- `MyDrive/YOLOv5_COCO_Project/runs/`
- `MyDrive/YOLOv5_COCO_Project/weights/`
- `MyDrive/YOLOv5_COCO_Project/logs/`

## Integrity Checks After Transfer

- Verify bundle SHA-256 before extraction.
- Validate ZIP entries for corruption, absolute paths, and path traversal.
- Check official archive sizes and ZIP readability.
- Confirm 118,287 `train2017` images and 5,000 `val2017` images after extraction.
- Reconstruct exactly 98,629 train, 12,328 validation, and 12,330 test images.
- Compare split identity hashes with local references.
- Confirm 886,282 accepted YOLO annotations, 10,498 crowd exclusions, and 2
  rejected invalid boxes.
- Validate Colab dataset YAML uses Linux paths, `nc: 80`, and exactly 80 names.
- Run a one-batch dataloader smoke test before training.

## Training Guard

The notebook defaults to:

```python
START_TRAINING = False
```

Full training cells are present only after the explicit integrity stop and raise
a `RuntimeError` until `START_TRAINING` is intentionally changed after reviewing
all PASS results.
