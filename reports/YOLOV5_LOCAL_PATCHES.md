# YOLOv5 Local Patch Audit

The project uses YOLOv5 v7.0 at commit `915bbf294bb74c859f0b41f1c23bc395014ea679`.

`external/yolov5` is a nested Git repository recorded by the parent repository as a gitlink. The parent repository has no `.gitmodules` file, so the nested checkout should remain pinned and clean rather than carrying uncommitted local edits.

## Audited Local Modifications

| File | Classification | Reason |
|---|---|---|
| `external/yolov5/utils/metrics.py` | Required compatibility patch | The local environment uses NumPy where `np.trapz` is not available. YOLOv5 v7.0 calls `np.trapz` during AP calculation. |
| `external/yolov5/utils/plots.py` | Required compatibility patch | The local environment uses Pillow >= 10, where `ImageFont.getsize()` is removed. YOLOv5 v7.0 calls `font.getsize()` during plot/label rendering. |

These changes were not generated files and were not accidental edits. Historical smoke logs under `artifacts/` show failures from both compatibility issues.

## Resolution

The nested YOLOv5 checkout was restored to the verified upstream v7.0 commit so the repository can be clean and reproducible.

The compatibility behavior is preserved in project-owned code:

- `src/yolov5_runtime_compat.py`
- `sitecustomize.py`
- `src/common.py` adds the repository root to `PYTHONPATH` for project-run YOLOv5 subprocesses.
- `src/inference_engine.py` applies the shim before local YOLOv5 model loading.

The original source-level diff is preserved as:

- `patches/yolov5/yolov5_v7_numpy_pillow_compat.patch`

## Reapplying The Source Patch

Source-level patching is not required for the cleaned repository because the runtime shim covers project workflows. If needed for an isolated YOLOv5 command, the patch can be applied with:

```powershell
git -C external/yolov5 apply ../../patches/yolov5/yolov5_v7_numpy_pillow_compat.patch
```

To return the nested checkout to upstream v7.0 after applying it:

```powershell
git -C external/yolov5 restore -- utils/metrics.py utils/plots.py
```

No training metrics, evaluation metrics, checkpoint files, dataset files, or annotation files are changed by this resolution.
