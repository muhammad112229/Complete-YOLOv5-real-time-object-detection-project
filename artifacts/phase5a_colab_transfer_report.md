# Phase 5A Colab Transfer Report

## 1. Existing Colab Notebook Audit

The previous Colab notebook was not sufficient for this phase because it exposed
direct training flow without a complete setup-first integrity gate. It has been
replaced with a 17-section notebook that verifies GPU runtime, mounts Drive,
checks the compact bundle, clones YOLOv5 v7.0, installs compatible dependencies,
downloads official COCO in Colab, reconstructs the exact split, runs integrity
checks, performs a one-batch dataloader smoke test, shows visual samples, and
then stops before training.

## 2. Selected Transfer Strategy

The selected strategy is a compact transfer bundle only. The full local COCO
dataset, local processed hardlink image tree, local COCO ZIP archives, smoke
checkpoints, and caches are excluded. Colab downloads official COCO 2017
archives directly, rebuilds labels, and reconstructs the exact local split from
the transferred manifests.

## 3. Bundle Path

`transfer/yolov5_colab_bundle.zip`

## 4. Bundle Size

4,678,978 bytes

## 5. Bundle Checksum

SHA-256:

```text
e56ae769ddb1854c7edb864574dd0554e091d57f90e72319cf9a40873e3cfa04
```

## 6. Number Of Bundled Files

37

## 7. Major Included Files

- `notebooks/YOLOv5_COCO_Training_Colab.ipynb`
- `src/prepare_coco_colab.py`
- `src/colab_transfer.py`
- supporting source modules
- `configs/coco_project_colab.yaml`
- Colab YOLOv5s/m/l training configs
- train, validation, and test manifests
- split image lists and `data/splits/split_summary.json`
- class mapping and class names
- dataset statistics and validation JSON
- reference checksum artifacts
- README and dependency files

## 8. Major Excluded Files

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
- output videos, webcam captures, caches, pycache, and pytest cache

## 9. COCO Reconstruction Method

`src.prepare_coco_colab` downloads `train2017.zip`, `val2017.zip`, and
`annotations_trainval2017.zip` inside Colab when missing or invalid. It validates
archive sizes and ZIP readability, extracts COCO source images, reads the
transferred manifests, rebuilds YOLO labels from official COCO annotations,
skips the same `iscrowd` annotations, rejects the same invalid boxes, preserves
the 0-79 class mapping, and writes a Linux-compatible dataset YAML.

Expected reconstructed values:

- Train images: 98,629
- Validation images: 12,328
- Test images: 12,330
- Accepted annotations: 886,282
- Excluded crowd annotations: 10,498
- Rejected invalid boxes: 2
- Classes: 80
- Seed: 42

## 10. Runtime Versus Drive Storage Options

Runtime mode stores data under `/content/datasets/coco2017`. It is faster but is
lost when runtime storage resets.

Drive mode stores data under
`/content/drive/MyDrive/YOLOv5_COCO_Project/datasets/coco2017`. It persists
across disconnects but Drive I/O is slower.

Training outputs are configured for Google Drive under
`/content/drive/MyDrive/YOLOv5_COCO_Project/runs/`.

## 11. Integrity-Check Implementation

Implemented checks include bundle SHA-256 verification, safe ZIP extraction,
official archive size and ZIP validation, source image counts, manifest-based
split counts, split identity hash comparison, annotation count comparison,
dataset YAML validation, class count validation, one-batch YOLOv5 dataloader
smoke test, and deterministic visual sample display.

Reference hashes are stored in:

- `artifacts/colab_reference_checksums.json`
- `artifacts/colab_reference_checksums.md`

## 12. Training-Guard Status

Training remains disabled by default:

```python
START_TRAINING = False
```

The full training cell raises:

```python
RuntimeError("Training is disabled. Complete and review integrity checks first.")
```

Full YOLOv5 training was not started.

## 13. Test Result

Local validation completed successfully:

- `.\.venv\Scripts\python.exe -m compileall -q src tests verify_setup.py` passed
- Notebook JSON validation passed
- `.\.venv\Scripts\python.exe -m src.colab_transfer --validate-bundle transfer\yolov5_colab_bundle.zip` passed
- JSON artifact parsing passed
- `.\.venv\Scripts\python.exe -m pytest` passed: 24 tests

## 14. Files Created Or Modified

- `src/prepare_coco_colab.py`
- `src/colab_transfer.py`
- `notebooks/YOLOv5_COCO_Training_Colab.ipynb`
- `configs/coco_project_colab.yaml`
- `configs/train_yolov5s_colab.yaml`
- `configs/train_yolov5m_colab.yaml`
- `configs/train_yolov5l_colab.yaml`
- `tests/test_colab_transfer.py`
- `artifacts/colab_reference_checksums.json`
- `artifacts/colab_reference_checksums.md`
- `artifacts/colab_training_transfer_manifest.json`
- `artifacts/colab_training_transfer_manifest.md`
- `artifacts/colab_bundle_report.json`
- `artifacts/phase5a_colab_transfer_report.md`
- `artifacts/phase5a_colab_transfer_results.json`
- `transfer/yolov5_colab_bundle.zip`
- `transfer/yolov5_colab_bundle.sha256`
- `transfer/yolov5_colab_bundle_contents.txt`
- `transfer/COLAB_TRANSFER_GUIDE.md`
- `transfer/google_drive_upload_checklist.md`
- `README.md`
- `reports/requirements_traceability.md`

## 15. Warnings

- The notebook has not been executed in Colab, so no Colab GPU result is claimed.
- COCO was not downloaded again locally.
- Full training was not started.
- Runtime mode requires enough temporary Colab disk and is not persistent.
- Drive mode persists the dataset but can be slower and needs substantial Drive quota.
- Plan for at least 60 GiB free space for COCO archives, extracted source data,
  reconstructed dataset, labels, and future run outputs.

## 16. Ready To Upload

Yes. The compact bundle is ready to upload to Google Drive with its checksum
file and the notebook.

## 17. Exact Manual Upload And Colab Steps

1. In Google Drive, create `MyDrive/YOLOv5_COCO_Project/bundles/`.
2. Upload `transfer/yolov5_colab_bundle.zip`.
3. Upload `transfer/yolov5_colab_bundle.sha256`.
4. Upload `notebooks/YOLOv5_COCO_Training_Colab.ipynb`.
5. Open the notebook in Google Colab.
6. Select `Runtime > Change runtime type > GPU`.
7. Run notebook Sections 1 through 16 in order.
8. Keep `START_TRAINING = False`.
9. Confirm all PASS values, especially split counts, class count, annotation
   counts, and split identity hashes.
10. Stop at the explicit integrity message before Section 17.

The upload checklist is available at
`transfer/google_drive_upload_checklist.md`, and the full transfer guide is at
`transfer/COLAB_TRANSFER_GUIDE.md`.

## 18. Full Training Status

Full training was not started. No YOLOv5s, YOLOv5m, or YOLOv5l full experiment
was launched in this phase.
