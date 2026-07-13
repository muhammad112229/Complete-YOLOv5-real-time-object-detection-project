"""Tests for the Flask visualization application."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import pytest

from app import create_app
from src.common import require_python_package
from src.inference_engine import InferenceResult
from src.media_utils import extract_video_metadata
from src.webcam_stream import WebcamStream


class FakeEngine:
    """Small inference engine double for Flask tests."""

    def __init__(self) -> None:
        self.resolved_device = "cpu"
        self.calls = 0

    def predict_frame(self, frame: Any, source: str | None = None) -> InferenceResult:
        self.calls += 1
        cv2 = require_python_package("cv2", "opencv-python")
        annotated = frame.copy()
        cv2.rectangle(annotated, (5, 5), (40, 40), (0, 255, 0), 2)
        return InferenceResult(
            annotated_frame=annotated,
            inference_ms=7.5,
            fps=100.0,
            detection_count=1,
            detected_class_names=("person",),
            detections=(
                {
                    "class_id": 0,
                    "class_name": "person",
                    "confidence": 0.9,
                    "x1": 5.0,
                    "y1": 5.0,
                    "x2": 40.0,
                    "y2": 40.0,
                },
            ),
            source=source,
        )


@pytest.fixture()
def app_config(tmp_path: Path) -> Path:
    model_path = tmp_path / "model.pt"
    model_path.write_bytes(b"fake")
    config = tmp_path / "app.yaml"
    config.write_text(
        "\n".join(
            [
                "app:",
                '  host: "127.0.0.1"',
                "  port: 5000",
                "  debug: false",
                "  threaded: true",
                "model:",
                f'  path: "{model_path.as_posix()}"',
                '  device: "auto"',
                "  image_size: 640",
                "inference:",
                "  confidence_threshold: 0.25",
                "  iou_threshold: 0.45",
                "  maximum_upload_size_mb: 5",
                "webcam:",
                "  default_camera_index: 0",
                "  jpeg_quality: 85",
                "  target_fps: 15",
                "paths:",
                f'  upload_directory: "{(tmp_path / "uploads").as_posix()}"',
                f'  output_directory: "{(tmp_path / "outputs").as_posix()}"',
                f'  image_output_directory: "{(tmp_path / "outputs" / "images").as_posix()}"',
                f'  video_output_directory: "{(tmp_path / "outputs" / "videos").as_posix()}"',
                f'  metadata_directory: "{(tmp_path / "outputs" / "metadata").as_posix()}"',
                "allowed_extensions:",
                "  images:",
                "    - jpg",
                "    - jpeg",
                "    - png",
                "  videos:",
                "    - mp4",
                "    - avi",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return config


@pytest.fixture()
def flask_app(app_config: Path) -> Any:
    return create_app(app_config, engine=FakeEngine(), load_model=False)


@pytest.fixture()
def client(flask_app: Any) -> Any:
    return flask_app.test_client()


def image_bytes() -> bytes:
    cv2 = require_python_package("cv2", "opencv-python")
    numpy = require_python_package("numpy")
    image = numpy.zeros((80, 100, 3), dtype=numpy.uint8)
    cv2.rectangle(image, (20, 20), (60, 60), (255, 255, 255), -1)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    return encoded.tobytes()


def write_test_video(path: Path) -> Path:
    cv2 = require_python_package("cv2", "opencv-python")
    numpy = require_python_package("numpy")
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (64, 48))
    assert writer.isOpened()
    try:
        for index in range(3):
            frame = numpy.full((48, 64, 3), index * 50, dtype=numpy.uint8)
            writer.write(frame)
    finally:
        writer.release()
    return path


def test_app_startup_and_health_endpoint(client: Any) -> None:
    response = client.get("/")
    assert response.status_code == 200
    health = client.get("/api/health")
    assert health.status_code == 200
    data = health.get_json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True
    assert data["active_device"] == "cpu"


def test_model_endpoint(client: Any) -> None:
    response = client.get("/api/model")
    assert response.status_code == 200
    data = response.get_json()
    assert data["model_path"] == "models/yolov5s_coco20k_best.pt"
    assert "official_cocoeval_ap_50" in data


def test_missing_upload_handling(client: Any) -> None:
    response = client.post("/detect/image", data={}, content_type="multipart/form-data")
    assert response.status_code == 400


def test_unsupported_extension_rejection(client: Any) -> None:
    response = client.post(
        "/detect/image",
        data={"file": (io.BytesIO(b"not image"), "bad.txt")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 400


def test_image_detection_request_creates_metadata_and_result_api(flask_app: Any, client: Any) -> None:
    response = client.post(
        "/detect/image",
        data={"file": (io.BytesIO(image_bytes()), "sample.jpg")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 200
    metadata_files = list(flask_app.config["METADATA_DIR"].glob("*.json"))
    assert len(metadata_files) == 1
    result_id = metadata_files[0].stem
    api_response = client.get(f"/api/results/{result_id}")
    assert api_response.status_code == 200
    metadata = api_response.get_json()
    assert metadata["task_type"] == "image"
    assert metadata["detection_count"] == 1
    assert Path(flask_app.config["IMAGE_OUTPUT_DIR"]).joinpath(Path(metadata["output_path"]).name).exists()
    assert client.get(metadata["output_url"]).status_code == 200


def test_invalid_result_id_rejected(client: Any) -> None:
    response = client.get("/api/results/bad!id")
    assert response.status_code == 400


def test_output_serving_security(client: Any) -> None:
    response = client.get("/outputs/%2e%2e/app.py")
    assert response.status_code == 404


def test_video_metadata_extraction_and_detection(flask_app: Any, client: Any, tmp_path: Path) -> None:
    video = write_test_video(tmp_path / "input.mp4")
    metadata = extract_video_metadata(video)
    assert metadata.width == 64
    assert metadata.height == 48
    with video.open("rb") as file:
        response = client.post(
            "/detect/video",
            data={"file": (file, "input.mp4")},
            content_type="multipart/form-data",
        )
    assert response.status_code == 200
    result_files = list(flask_app.config["METADATA_DIR"].glob("*.json"))
    assert len(result_files) == 1
    result_id = result_files[0].stem
    result = client.get(f"/api/results/{result_id}").get_json()
    assert result["task_type"] == "video"
    assert result["processed_frames"] == 3
    assert result["total_detections"] == 3
    output_path = Path.cwd() / result["output_path"]
    output_metadata = extract_video_metadata(output_path)
    assert output_metadata.width == 64
    assert output_metadata.height == 48


def test_webcam_component_lifecycle_without_physical_camera(monkeypatch: pytest.MonkeyPatch) -> None:
    cv2 = require_python_package("cv2", "opencv-python")
    numpy = require_python_package("numpy")
    released = {"value": False}

    class FakeCapture:
        def isOpened(self) -> bool:
            return True

        def read(self) -> tuple[bool, Any]:
            return True, numpy.zeros((48, 64, 3), dtype=numpy.uint8)

        def release(self) -> None:
            released["value"] = True

    monkeypatch.setattr(cv2, "VideoCapture", lambda _index: FakeCapture())
    stream = WebcamStream(FakeEngine(), camera_index=0, target_fps=30)
    frame = stream.read_annotated_jpeg()
    assert frame.startswith(b"\xff\xd8")
    assert stream.status["running"] is True
    stream.stop()
    assert released["value"] is True
    assert stream.status["running"] is False


def test_shared_model_loader_called_once(monkeypatch: pytest.MonkeyPatch, app_config: Path) -> None:
    import app as app_module

    calls = {"count": 0}

    def fake_loader(_config: dict[str, Any]) -> FakeEngine:
        calls["count"] += 1
        return FakeEngine()

    monkeypatch.setattr(app_module, "create_inference_engine", fake_loader)
    flask_app = app_module.create_app(app_config, load_model=True)
    assert calls["count"] == 1
    flask_app.test_client().get("/api/health")
    flask_app.test_client().get("/api/model")
    assert calls["count"] == 1
