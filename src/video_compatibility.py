"""Browser-playback compatibility helpers for generated MP4 videos."""

from __future__ import annotations

import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.common import ensure_dir, require_python_package


BROWSER_COMPATIBLE_CODECS = {"h264", "avc1", "avc3"}
PREFERRED_OPENCV_CODECS = ("avc1", "H264", "X264", "mp4v")


@dataclass(frozen=True)
class VideoInspection:
    """OpenCV video metadata and readability status."""

    path: str
    exists: bool
    file_size_bytes: int
    opened: bool
    first_frame_read: bool
    codec: str | None
    fourcc_int: int | None
    fps: float
    frame_count: int
    width: int
    height: int


def fourcc_to_string(value: int) -> str:
    """Convert a CAP_PROP_FOURCC integer to a readable codec tag."""
    if not value:
        return ""
    return "".join(chr((int(value) >> (8 * index)) & 255) for index in range(4)).strip("\x00")


def inspect_video(path: Path) -> VideoInspection:
    """Inspect a video file with OpenCV."""
    cv2 = require_python_package("cv2", "opencv-python")
    if not path.is_file():
        return VideoInspection(
            path=str(path),
            exists=False,
            file_size_bytes=0,
            opened=False,
            first_frame_read=False,
            codec=None,
            fourcc_int=None,
            fps=0.0,
            frame_count=0,
            width=0,
            height=0,
        )
    capture = cv2.VideoCapture(str(path))
    try:
        opened = bool(capture.isOpened())
        fourcc_int = int(capture.get(cv2.CAP_PROP_FOURCC) or 0)
        ok, _frame = capture.read() if opened else (False, None)
        return VideoInspection(
            path=str(path),
            exists=True,
            file_size_bytes=path.stat().st_size,
            opened=opened,
            first_frame_read=bool(ok),
            codec=fourcc_to_string(fourcc_int) or None,
            fourcc_int=fourcc_int or None,
            fps=float(capture.get(cv2.CAP_PROP_FPS) or 0.0),
            frame_count=int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
            width=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
            height=int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
        )
    finally:
        capture.release()


def codec_is_browser_compatible(codec: str | None) -> bool:
    """Return whether a codec tag is a conservative Chrome-compatible MP4 signal."""
    if not codec:
        return False
    return codec.strip().lower() in BROWSER_COMPATIBLE_CODECS


def ffmpeg_available() -> bool:
    """Return whether FFmpeg is available on PATH."""
    return shutil.which("ffmpeg") is not None


def ffprobe_available() -> bool:
    """Return whether ffprobe is available on PATH."""
    return shutil.which("ffprobe") is not None


def open_writer(path: Path, codec: str, fps: float, width: int, height: int) -> Any:
    """Open an OpenCV VideoWriter for one codec."""
    cv2 = require_python_package("cv2", "opencv-python")
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), fps, (width, height))
    if not writer.isOpened():
        writer.release()
        raise RuntimeError(f"OpenCV VideoWriter could not open codec {codec} for {path}")
    return writer


def transcode_with_opencv(source: Path, output: Path, codec: str) -> dict[str, Any]:
    """Transcode a video by decoding and rewriting frames with OpenCV."""
    cv2 = require_python_package("cv2", "opencv-python")
    source_info = inspect_video(source)
    if not source_info.opened or source_info.width <= 0 or source_info.height <= 0:
        raise RuntimeError(f"Source video is not readable: {source}")
    fps = source_info.fps if source_info.fps > 0 else 30.0
    ensure_dir(output.parent)
    capture = cv2.VideoCapture(str(source))
    writer = open_writer(output, codec, fps, source_info.width, source_info.height)
    frames_written = 0
    start = time.perf_counter()
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            writer.write(frame)
            frames_written += 1
    finally:
        capture.release()
        writer.release()
    output_info = inspect_video(output)
    if not output_info.opened or not output_info.first_frame_read or frames_written == 0:
        raise RuntimeError(f"Transcoded output is not readable: {output}")
    return {
        "codec_requested": codec,
        "frames_written": frames_written,
        "duration_seconds": time.perf_counter() - start,
        "output_inspection": asdict(output_info),
    }


def transcode_with_ffmpeg(source: Path, output: Path) -> dict[str, Any]:
    """Transcode to H.264 MP4 with FFmpeg when available."""
    if not ffmpeg_available():
        raise RuntimeError("FFmpeg is not available on PATH.")
    ensure_dir(output.parent)
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(source),
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-an",
        str(output),
    ]
    start = time.perf_counter()
    completed = subprocess.run(command, check=False, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if completed.returncode != 0:
        raise RuntimeError(f"FFmpeg transcode failed: {completed.stderr[-1000:]}")
    return {
        "command": command,
        "duration_seconds": time.perf_counter() - start,
        "stdout": completed.stdout[-2000:],
        "stderr": completed.stderr[-2000:],
        "output_inspection": asdict(inspect_video(output)),
    }


def ensure_browser_compatible_mp4(source: Path, final_output: Path | None = None) -> dict[str, Any]:
    """Return a verified browser-compatible MP4 when local tools can produce one."""
    source = Path(source)
    final_output = final_output or source.with_name(f"{source.stem}_browser.mp4")
    source_info = inspect_video(source)
    result: dict[str, Any] = {
        "source_video": str(source),
        "source_inspection": asdict(source_info),
        "source_codec": source_info.codec,
        "ffmpeg_available": ffmpeg_available(),
        "ffprobe_available": ffprobe_available(),
        "final_video": str(source),
        "final_inspection": asdict(source_info),
        "final_codec": source_info.codec,
        "opencv_readability": bool(source_info.opened and source_info.first_frame_read),
        "browser_compatible": False,
        "compatibility_method": "source_not_browser_compatible",
        "transcoded": False,
        "attempts": [],
        "verification_result": "failed",
    }
    if source_info.opened and source_info.first_frame_read and source.suffix.lower() == ".mp4" and codec_is_browser_compatible(source_info.codec):
        result.update(
            {
                "browser_compatible": True,
                "compatibility_method": "source_mp4_codec_verified",
                "verification_result": "passed",
            }
        )
        return result

    h264_output = final_output
    for codec in PREFERRED_OPENCV_CODECS:
        candidate = h264_output if codec != "mp4v" else source.with_name(f"{source.stem}_mp4v_browser_fallback.mp4")
        if candidate.exists():
            candidate.unlink()
        try:
            attempt = transcode_with_opencv(source, candidate, codec)
            final_info = inspect_video(candidate)
            compatible = candidate.suffix.lower() == ".mp4" and codec_is_browser_compatible(final_info.codec)
            attempt.update(
                {
                    "candidate": str(candidate),
                    "final_codec": final_info.codec,
                    "browser_compatible": compatible,
                }
            )
            result["attempts"].append(attempt)
            if compatible:
                result.update(
                    {
                        "final_video": str(candidate),
                        "final_inspection": asdict(final_info),
                        "final_codec": final_info.codec,
                        "opencv_readability": bool(final_info.opened and final_info.first_frame_read),
                        "browser_compatible": True,
                        "compatibility_method": f"opencv_transcode_{codec}",
                        "transcoded": True,
                        "verification_result": "passed",
                    }
                )
                return result
        except Exception as exc:
            result["attempts"].append({"codec_requested": codec, "candidate": str(candidate), "error": str(exc)})

    if ffmpeg_available():
        ffmpeg_output = final_output
        if ffmpeg_output.exists():
            ffmpeg_output.unlink()
        try:
            attempt = transcode_with_ffmpeg(source, ffmpeg_output)
            final_info = inspect_video(ffmpeg_output)
            compatible = codec_is_browser_compatible(final_info.codec)
            result["attempts"].append({"method": "ffmpeg", "candidate": str(ffmpeg_output), **attempt})
            if compatible:
                result.update(
                    {
                        "final_video": str(ffmpeg_output),
                        "final_inspection": asdict(final_info),
                        "final_codec": final_info.codec,
                        "opencv_readability": bool(final_info.opened and final_info.first_frame_read),
                        "browser_compatible": True,
                        "compatibility_method": "ffmpeg_libx264",
                        "transcoded": True,
                        "verification_result": "passed",
                    }
                )
                return result
        except Exception as exc:
            result["attempts"].append({"method": "ffmpeg", "candidate": str(ffmpeg_output), "error": str(exc)})

    fallback_info = inspect_video(source)
    result.update(
        {
            "final_video": str(source),
            "final_inspection": asdict(fallback_info),
            "final_codec": fallback_info.codec,
            "opencv_readability": bool(fallback_info.opened and fallback_info.first_frame_read),
            "compatibility_method": "fallback_original_not_browser_verified",
            "verification_result": "warning",
        }
    )
    return result
