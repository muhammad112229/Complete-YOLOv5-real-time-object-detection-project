"""Thread-safe webcam MJPEG stream support for the Flask application."""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any

from src.common import require_python_package


class WebcamStream:
    """Lazy OpenCV camera wrapper that reuses a shared inference engine."""

    def __init__(
        self,
        engine: Any,
        camera_index: int = 0,
        image_size: int = 640,
        jpeg_quality: int = 85,
        target_fps: int = 15,
        model_lock: threading.Lock | None = None,
    ) -> None:
        self.engine = engine
        self.camera_index = int(camera_index)
        self.image_size = int(image_size)
        self.jpeg_quality = int(jpeg_quality)
        self.target_fps = max(1, int(target_fps))
        self.model_lock = model_lock or threading.Lock()
        self._capture: Any | None = None
        self._lock = threading.RLock()
        self._running = False
        self._fps_window: deque[float] = deque(maxlen=20)
        self._last_frame_time: float | None = None
        self._status: dict[str, Any] = {
            "running": False,
            "camera_index": self.camera_index,
            "instantaneous_fps": 0.0,
            "moving_average_fps": 0.0,
            "inference_latency_ms": 0.0,
            "detection_count": 0,
            "last_error": None,
        }

    @property
    def status(self) -> dict[str, Any]:
        """Return a copy of current stream status."""
        with self._lock:
            return dict(self._status)

    def _open_capture(self) -> Any:
        cv2 = require_python_package("cv2", "opencv-python")
        capture = cv2.VideoCapture(self.camera_index)
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"Could not open camera index {self.camera_index}")
        return capture

    def start(self) -> None:
        """Start camera capture lazily."""
        with self._lock:
            if self._capture is None:
                self._capture = self._open_capture()
            self._running = True
            self._status["running"] = True
            self._status["last_error"] = None

    def stop(self) -> None:
        """Stop and release camera capture."""
        with self._lock:
            self._running = False
            self._status["running"] = False
            capture = self._capture
            self._capture = None
        if capture is not None:
            capture.release()

    def _reconnect(self) -> None:
        with self._lock:
            capture = self._capture
            self._capture = None
        if capture is not None:
            capture.release()
        time.sleep(0.1)
        with self._lock:
            if self._running:
                self._capture = self._open_capture()

    def read_annotated_jpeg(self) -> bytes:
        """Read, infer, annotate, and encode one JPEG frame."""
        cv2 = require_python_package("cv2", "opencv-python")
        self.start()
        with self._lock:
            capture = self._capture
            running = self._running
        if not running or capture is None:
            raise RuntimeError("Webcam stream is not running.")

        ok, frame = capture.read()
        if not ok or frame is None:
            with self._lock:
                self._status["last_error"] = "Camera frame capture failed; reconnecting."
            self._reconnect()
            raise RuntimeError("Camera frame capture failed.")

        with self.model_lock:
            result = self.engine.predict_frame(frame, source=f"webcam:{self.camera_index}")

        now = time.perf_counter()
        if self._last_frame_time is not None:
            interval = now - self._last_frame_time
            instantaneous_fps = 1.0 / interval if interval > 0 else 0.0
            self._fps_window.append(instantaneous_fps)
        else:
            instantaneous_fps = 0.0
        self._last_frame_time = now
        moving_average_fps = sum(self._fps_window) / len(self._fps_window) if self._fps_window else instantaneous_fps

        annotated = result.annotated_frame
        cv2.putText(
            annotated,
            f"FPS {moving_average_fps:.1f} | {result.inference_ms:.1f} ms | {result.detection_count} objects",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        ok, encoded = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
        if not ok:
            raise RuntimeError("Could not encode webcam frame as JPEG.")

        with self._lock:
            self._status.update(
                {
                    "running": True,
                    "instantaneous_fps": round(float(instantaneous_fps), 3),
                    "moving_average_fps": round(float(moving_average_fps), 3),
                    "inference_latency_ms": round(float(result.inference_ms), 3),
                    "detection_count": int(result.detection_count),
                    "last_error": None,
                }
            )

        target_interval = 1.0 / self.target_fps
        elapsed = time.perf_counter() - now
        if elapsed < target_interval:
            time.sleep(target_interval - elapsed)
        return encoded.tobytes()

    def frames(self) -> Any:
        """Yield multipart MJPEG frames until stopped."""
        while self.status["running"]:
            try:
                frame = self.read_annotated_jpeg()
            except RuntimeError:
                time.sleep(0.2)
                continue
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"

    def __enter__(self) -> "WebcamStream":
        self.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.stop()
