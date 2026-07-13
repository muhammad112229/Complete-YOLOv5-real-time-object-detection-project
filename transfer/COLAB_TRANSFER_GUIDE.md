# Colab Transfer Guide

This guide transfers only the compact project bundle. Do not upload the full
local COCO dataset, processed hardlink tree, local COCO ZIP archives, or local
smoke-test checkpoints.

## Files To Upload

Upload these files to Google Drive:

- `transfer/yolov5_colab_bundle.zip`
- `transfer/yolov5_colab_bundle.sha256`
- `notebooks/YOLOv5_COCO_Training_Colab.ipynb`

Recommended destination:

```text
MyDrive/YOLOv5_COCO_Project/bundles/
```

Final bundle values:

- Size: 4,678,978 bytes
- SHA-256: `e56ae769ddb1854c7edb864574dd0554e091d57f90e72319cf9a40873e3cfa04`
- Bundled files: 37

## Open The Notebook

1. Open Google Drive.
2. Go to `MyDrive/YOLOv5_COCO_Project/bundles/`.
3. Open `YOLOv5_COCO_Training_Colab.ipynb` with Google Colab.
4. Select `Runtime > Change runtime type`.
5. Choose a GPU runtime.

The notebook intentionally stops if CUDA is unavailable. Do not continue on CPU.

## Drive Mount And Folders

Run the notebook sections in order. The Drive section mounts Google Drive and
creates:

- `bundles/`
- `datasets/`
- `weights/`
- `runs/`
- `evaluations/`
- `exports/`
- `logs/`

under:

```text
/content/drive/MyDrive/YOLOv5_COCO_Project
```

## Dataset Storage Mode

The editable configuration cell contains:

```python
STORAGE_MODE = "runtime"
```

Use `runtime` for the first setup attempt:

- Dataset storage root: `/content/datasets`
- Faster I/O.
- Dataset is lost when runtime storage is reset.

Use `drive` when persistence matters more than speed:

- Dataset storage root: `/content/drive/MyDrive/YOLOv5_COCO_Project/datasets`
- Survives Colab disconnects and runtime restarts.
- Drive I/O is usually slower.

## Cells To Run First

Run Sections 1 through 16 in order:

1. Project and phase explanation
2. GPU runtime verification
3. Google Drive mount
4. Configuration
5. Bundle extraction and checksum verification
6. YOLOv5 v7.0 clone
7. Dependency installation
8. Official COCO download
9. Extraction and source validation
10. Exact split reconstruction
11. Integrity comparison
12. Dataset YAML validation
13. One-batch dataloader smoke test
14. Visual sample check
15. Training configuration preview
16. Explicit integrity stop

The setup phase stops at:

```text
Colab setup and dataset integrity verification completed. Review all PASS results before enabling full training.
```

Do not run Section 17 until every integrity result is reviewed.

## PASS Values To Confirm

Confirm these values in the notebook output:

- `train2017` source images: 118,287
- `val2017` source images: 5,000
- Reconstructed train images: 98,629
- Reconstructed validation images: 12,328
- Reconstructed test images: 12,330
- Accepted annotations: 886,282
- Excluded crowd annotations: 10,498
- Rejected invalid boxes: 2
- Classes: 80
- Split seed: 42
- Dataset YAML `nc`: 80
- Dataset YAML names: 80

The split identity hashes must match
`artifacts/colab_reference_checksums.json`.

## Resume After Disconnect

After a disconnect:

1. Reopen the notebook.
2. Mount Drive again.
3. Keep the same `STORAGE_MODE`.
4. Re-run setup cells from configuration onward.

The download step reuses valid archives. The extraction and reconstruction
steps reuse valid files where possible and can be safely re-run.

## Disk Requirements

Runtime mode needs enough Colab runtime disk for:

- COCO archives: about 19.0 GiB train ZIP, 0.8 GiB val ZIP, 0.25 GiB annotations ZIP
- Extracted source images: about 25 GiB
- Reconstructed YOLO split links or copied files, depending on link support
- Labels, logs, and temporary files

Use a runtime with at least 60 GiB free disk. More space is safer if links are
not available and files must be copied.

Drive mode needs enough Google Drive quota for the archive cache, extracted
source images, reconstructed dataset, training runs, and checkpoints. Plan for
60 GiB or more before full training.

## Future Training Outputs

When training is intentionally enabled later, YOLOv5 run outputs are configured
under:

```text
/content/drive/MyDrive/YOLOv5_COCO_Project/runs/yolov5s/
```

The future `best.pt` and `last.pt` checkpoints will be under that run
directory, typically:

```text
/content/drive/MyDrive/YOLOv5_COCO_Project/runs/yolov5s/<run_name>/weights/
```

## Avoid Duplicate COCO Downloads

- Keep `STORAGE_MODE` consistent across resumes.
- Do not delete `archives/` under the chosen dataset root.
- The notebook verifies existing ZIP files before downloading.
- If a partial download exists, the downloader attempts to resume it.

## Training Guard

The notebook default is:

```python
START_TRAINING = False
```

Section 17 raises:

```python
RuntimeError("Training is disabled. Complete and review integrity checks first.")
```

Leave training disabled until the setup sections produce PASS results.
