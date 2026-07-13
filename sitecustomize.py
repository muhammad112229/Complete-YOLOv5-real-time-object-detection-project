"""Project-local Python startup hooks.

This module is loaded automatically when the repository root is on PYTHONPATH.
It keeps the pinned YOLOv5 v7.0 checkout usable with newer local dependencies.
"""

from __future__ import annotations


try:
    from src.yolov5_runtime_compat import apply_yolov5_runtime_compatibility

    apply_yolov5_runtime_compatibility()
except Exception:
    # Startup hooks must never block unrelated Python commands.
    pass
