"""Media validation and path helpers for the Flask application."""

from __future__ import annotations

import mimetypes
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.common import ensure_dir, require_python_package


@dataclass(frozen=True)
class VideoMetadata:
    """Basic video container metadata."""

    width: int
    height: int
    fps: float
    frame_count: int
    duration_seconds: float


def extension(path_or_name: str | Path) -> str:
    """Return the lowercase extension without a leading dot."""
    return Path(str(path_or_name)).suffix.lower().lstrip(".")


def is_allowed_extension(filename: str, allowed_extensions: set[str]) -> bool:
    """Check whether a filename has one of the allowed extensions."""
    return bool(filename and extension(filename) in {item.lower().lstrip(".") for item in allowed_extensions})


def validate_file_size(path: Path, maximum_size_mb: int | float) -> None:
    """Raise when a file exceeds the configured maximum size."""
    max_bytes = int(float(maximum_size_mb) * 1024 * 1024)
    if path.stat().st_size > max_bytes:
        raise ValueError(f"File exceeds maximum upload size of {maximum_size_mb} MB.")


def validate_image_file(path: Path) -> tuple[int, int]:
    """Validate that a path is readable as an image and return width, height."""
    cv2 = require_python_package("cv2", "opencv-python")
    image = cv2.imread(str(path))
    if image is None or not hasattr(image, "shape") or len(image.shape) < 2:
        raise ValueError(f"Uploaded file is not a readable image: {path.name}")
    height, width = image.shape[:2]
    if width <= 0 or height <= 0:
        raise ValueError(f"Image has invalid dimensions: {path.name}")
    return int(width), int(height)


def extract_video_metadata(path: Path) -> VideoMetadata:
    """Open a video with OpenCV and return basic metadata."""
    cv2 = require_python_package("cv2", "opencv-python")
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError(f"Uploaded file is not a readable video: {path.name}")
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if width <= 0 or height <= 0:
            raise ValueError(f"Video has invalid dimensions: {path.name}")
        safe_fps = fps if fps > 0 else 30.0
        duration = float(frame_count / safe_fps) if frame_count > 0 else 0.0
        return VideoMetadata(width=width, height=height, fps=fps, frame_count=frame_count, duration_seconds=duration)
    finally:
        capture.release()


def validate_video_file(path: Path) -> VideoMetadata:
    """Validate that a path is readable as a video and return metadata."""
    return extract_video_metadata(path)


def safe_output_path(directory: Path, result_id: str, original_filename: str, suffix: str, extension_override: str | None = None) -> Path:
    """Build a deterministic output path under a target directory."""
    ensure_dir(directory)
    original = Path(original_filename)
    ext = extension_override or original.suffix.lower().lstrip(".") or "dat"
    stem = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in original.stem)[:80] or "upload"
    return directory / f"{result_id}_{stem}_{suffix}.{ext}"


def public_output_name(output_directory: Path, path: Path) -> str:
    """Return a path relative to the Flask output root for /outputs serving."""
    return path.resolve().relative_to(output_directory.resolve()).as_posix()


def guess_media_type(path: Path) -> str:
    """Return a best-effort media type."""
    return mimetypes.guess_type(str(path))[0] or "application/octet-stream"


def coerce_allowed_extensions(config: dict[str, Any], media_type: str) -> set[str]:
    """Read an extension set from app config."""
    values = config.get("allowed_extensions", {}).get(media_type, [])
    return {str(item).lower().lstrip(".") for item in values}
