"""Runtime compatibility shims for the pinned YOLOv5 v7.0 checkout."""

from __future__ import annotations


def _patch_numpy_trapz() -> bool:
    """Restore np.trapz for YOLOv5 v7.0 when NumPy only exposes trapezoid."""
    try:
        import numpy as np
    except Exception:
        return False
    if hasattr(np, "trapz") or not hasattr(np, "trapezoid"):
        return False
    np.trapz = np.trapezoid  # type: ignore[attr-defined]
    return True


def _patch_pillow_getsize() -> bool:
    """Restore ImageFont.getsize for YOLOv5 v7.0 plotting on Pillow >= 10."""
    try:
        from PIL import ImageFont
    except Exception:
        return False

    patched = False

    def getsize(self: object, text: str, *args: object, **kwargs: object) -> tuple[int, int]:
        left, top, right, bottom = self.getbbox(text, *args, **kwargs)  # type: ignore[attr-defined]
        return int(right - left), int(bottom - top)

    for class_name in ("ImageFont", "FreeTypeFont", "TransposedFont"):
        font_class = getattr(ImageFont, class_name, None)
        if font_class is None or hasattr(font_class, "getsize") or not hasattr(font_class, "getbbox"):
            continue
        setattr(font_class, "getsize", getsize)
        patched = True
    return patched


def apply_yolov5_runtime_compatibility() -> dict[str, bool]:
    """Apply compatibility shims needed by the unmodified YOLOv5 v7.0 source tree."""
    return {
        "numpy_trapz": _patch_numpy_trapz(),
        "pillow_getsize": _patch_pillow_getsize(),
    }
