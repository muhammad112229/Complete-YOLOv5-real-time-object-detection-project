# Google Drive Upload Checklist

Use this checklist manually. Do not automate Google login or store credentials.

## Destination

Create this Drive folder:

```text
MyDrive/YOLOv5_COCO_Project/bundles/
```

## Upload These Files

| Local file | Drive destination | Expected size |
|---|---|---:|
| `transfer/yolov5_colab_bundle.zip` | `MyDrive/YOLOv5_COCO_Project/bundles/yolov5_colab_bundle.zip` | 4,678,978 bytes |
| `transfer/yolov5_colab_bundle.sha256` | `MyDrive/YOLOv5_COCO_Project/bundles/yolov5_colab_bundle.sha256` | 91 bytes |
| `notebooks/YOLOv5_COCO_Training_Colab.ipynb` | `MyDrive/YOLOv5_COCO_Project/bundles/YOLOv5_COCO_Training_Colab.ipynb` | 16,253 bytes |

Expected bundle SHA-256:

```text
e56ae769ddb1854c7edb864574dd0554e091d57f90e72319cf9a40873e3cfa04
```

## Do Not Upload

- Full local COCO images
- `data/raw`
- `data/processed/coco_yolo/images`
- `data/processed/coco_yolo/labels`
- local COCO ZIP archives
- `.venv`
- `external/yolov5/.git`
- local smoke-test checkpoints
- caches or pycache folders

## Post-Upload Check In Colab

After mounting Drive, the notebook verifies the bundle checksum. You can also
run this manually in a Colab cell:

```python
import hashlib
from pathlib import Path

path = Path("/content/drive/MyDrive/YOLOv5_COCO_Project/bundles/yolov5_colab_bundle.zip")
digest = hashlib.sha256(path.read_bytes()).hexdigest()
print(digest)
assert digest == "e56ae769ddb1854c7edb864574dd0554e091d57f90e72319cf9a40873e3cfa04"
```

Only continue after the checksum matches.
