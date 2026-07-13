"""Tests for browser video compatibility helpers."""

from __future__ import annotations

from pathlib import Path

from src.common import require_python_package
from src.video_compatibility import ensure_browser_compatible_mp4, inspect_video


def write_mp4v_video(path: Path) -> Path:
    """Write a tiny MP4V video for compatibility testing."""
    cv2 = require_python_package("cv2", "opencv-python")
    numpy = require_python_package("numpy")
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 8.0, (64, 64))
    assert writer.isOpened()
    try:
        for index in range(4):
            frame = numpy.full((64, 64, 3), index * 50, dtype=numpy.uint8)
            writer.write(frame)
    finally:
        writer.release()
    return path


def test_video_inspection_and_browser_compatibility(tmp_path: Path) -> None:
    source = write_mp4v_video(tmp_path / "source.mp4")
    source_info = inspect_video(source)
    assert source_info.opened
    assert source_info.first_frame_read
    result = ensure_browser_compatible_mp4(source, tmp_path / "browser.mp4")
    assert result["opencv_readability"] is True
    assert Path(result["final_video"]).is_file()
    assert result["final_inspection"]["frame_count"] == source_info.frame_count
    assert result["verification_result"] in {"passed", "warning"}
