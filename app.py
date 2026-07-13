"""Flask web application for real-time YOLOv5 object detection."""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any

from flask import (
    Flask,
    Response,
    abort,
    jsonify,
    redirect,
    render_template,
    request,
    send_file,
    url_for,
)
from werkzeug.datastructures import FileStorage
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename

from src.common import ensure_dir, project_root, require_python_package
from src.inference_engine import (
    InferenceConfig,
    InferenceEngine,
    infer_frame_with_model,
    load_inference_config,
    parse_class_filter,
    summarize_detections,
)
from src.media_utils import (
    coerce_allowed_extensions,
    public_output_name,
    safe_output_path,
    validate_file_size,
    validate_image_file,
    validate_video_file,
)
from src.result_store import ResultStore, utc_timestamp
from src.video_compatibility import ensure_browser_compatible_mp4, inspect_video
from src.webcam_stream import WebcamStream


DEFAULT_APP_CONFIG = Path("configs") / "app.yaml"
LOGGER = logging.getLogger(__name__)


def relative_path(path: Path) -> str:
    """Return a workspace-relative path when possible."""
    root = project_root().resolve()
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError:
        return str(path)


def load_yaml(path: Path) -> dict[str, Any]:
    """Load YAML configuration."""
    yaml = require_python_package("yaml", "PyYAML")
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    return data


def resolve_config_paths(config: dict[str, Any]) -> dict[str, Any]:
    """Resolve app path config to absolute workspace paths."""
    root = project_root()
    resolved = json.loads(json.dumps(config))
    paths = resolved.setdefault("paths", {})
    for key in [
        "upload_directory",
        "output_directory",
        "image_output_directory",
        "video_output_directory",
        "metadata_directory",
    ]:
        if key in paths:
            candidate = Path(paths[key])
            paths[key] = candidate if candidate.is_absolute() else root / candidate
    model = resolved.setdefault("model", {})
    if "path" in model:
        candidate = Path(model["path"])
        model["path"] = candidate if candidate.is_absolute() else root / candidate
    return resolved


def load_app_config(config_path: Path | None = None) -> dict[str, Any]:
    """Load and normalize Flask app configuration."""
    root = project_root()
    path = config_path or root / DEFAULT_APP_CONFIG
    path = path if path.is_absolute() else root / path
    config = load_yaml(path)
    return resolve_config_paths(config)


def class_counts(detections: tuple[dict[str, Any], ...]) -> dict[str, int]:
    """Count detections by class name."""
    return summarize_detections(detections)


def confidence_statistics(detections: tuple[dict[str, Any], ...]) -> dict[str, float | None]:
    """Return min/max/mean confidence statistics."""
    values = [float(item["confidence"]) for item in detections]
    if not values:
        return {"min": None, "max": None, "mean": None}
    return {
        "min": min(values),
        "max": max(values),
        "mean": sum(values) / len(values),
    }


def parse_thresholds() -> tuple[float, float]:
    """Parse confidence and IoU thresholds from form data."""
    app_config = request.app_config  # type: ignore[attr-defined]
    defaults = app_config["inference"]
    confidence = float(request.form.get("confidence_threshold") or defaults["confidence_threshold"])
    iou = float(request.form.get("iou_threshold") or defaults["iou_threshold"])
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("Confidence threshold must be between 0 and 1.")
    if not 0.0 <= iou <= 1.0:
        raise ValueError("IoU threshold must be between 0 and 1.")
    return confidence, iou


def parse_optional_class_filter() -> tuple[int | str, ...] | None:
    """Parse optional class filter from form values."""
    values = request.form.getlist("class_filter")
    if len(values) == 1 and "," in values[0]:
        values = [item.strip() for item in values[0].split(",")]
    values = [item for item in values if item.strip()]
    return parse_class_filter(values) if values else None


def get_upload_file() -> FileStorage:
    """Return the uploaded file or raise a user-facing error."""
    upload = request.files.get("file") or request.files.get("image") or request.files.get("video")
    if upload is None or not upload.filename:
        raise ValueError("No upload file was provided.")
    return upload


def save_upload(upload: FileStorage, upload_dir: Path, result_id: str, allowed_extensions: set[str]) -> Path:
    """Validate and save an upload safely."""
    filename = secure_filename(upload.filename or "")
    if not filename:
        raise ValueError("Upload filename is empty.")
    suffix = Path(filename).suffix.lower().lstrip(".")
    if suffix not in allowed_extensions:
        raise ValueError(f"Unsupported file extension: .{suffix}")
    ensure_dir(upload_dir)
    target = upload_dir / f"{result_id}_{filename}"
    upload.save(target)
    return target


def apply_model_thresholds(model: Any, confidence: float, iou: float) -> tuple[Any, Any]:
    """Set model thresholds and return old values."""
    old_conf = getattr(model, "conf", None)
    old_iou = getattr(model, "iou", None)
    if old_conf is not None:
        model.conf = float(confidence)
    if old_iou is not None:
        model.iou = float(iou)
    return old_conf, old_iou


def restore_model_thresholds(model: Any, old_conf: Any, old_iou: Any) -> None:
    """Restore model thresholds after request-scoped inference."""
    if old_conf is not None:
        model.conf = old_conf
    if old_iou is not None:
        model.iou = old_iou


def run_frame_inference(
    app: Flask,
    frame: Any,
    confidence: float,
    iou: float,
    class_filter: tuple[int | str, ...] | None,
    source: str | None = None,
) -> Any:
    """Run inference with request-scoped filters using the shared model."""
    engine = app.config["INFERENCE_ENGINE"]
    if engine is None:
        raise RuntimeError("Inference engine is not loaded.")
    image_size = int(app.config["APP_CONFIG"]["model"]["image_size"])
    with app.config["MODEL_LOCK"]:
        model = getattr(engine, "model", None)
        if model is not None:
            old_conf, old_iou = apply_model_thresholds(model, confidence, iou)
            try:
                return infer_frame_with_model(model, frame, image_size=image_size, class_filter=class_filter, source=source)
            finally:
                restore_model_thresholds(model, old_conf, old_iou)
        return engine.predict_frame(frame, source=source)


def output_url_for(app: Flask, path: Path) -> str:
    """Build a URL for an output or upload path."""
    output_root = app.config["OUTPUT_DIR"]
    upload_root = app.config["UPLOAD_DIR"]
    try:
        relative = path.resolve().relative_to(output_root.resolve()).as_posix()
        return url_for("serve_output", filename=relative)
    except ValueError:
        pass
    try:
        relative = path.resolve().relative_to(upload_root.resolve()).as_posix()
        return url_for("serve_output", filename=f"uploads/{relative}")
    except ValueError:
        return ""


def load_json_if_exists(path: Path) -> dict[str, Any]:
    """Load a JSON object when present."""
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def build_model_metadata(engine: Any | None = None) -> dict[str, Any]:
    """Read safe model metadata from existing verified artifacts."""
    root = project_root()
    checkpoint = load_json_if_exists(root / "artifacts" / "checkpoint_inspection.json")
    trained = load_json_if_exists(root / "artifacts" / "trained_model_verification.json")
    subset = load_json_if_exists(root / "artifacts" / "test_subset_2500_verification.json")
    yolo_metrics = load_json_if_exists(root / "results" / "evaluation" / "test_subset_2500" / "metrics_summary.json")
    coco_metrics = load_json_if_exists(root / "results" / "evaluation" / "test_subset_2500" / "coco_eval" / "coco_eval_summary.json")
    names = getattr(getattr(engine, "model", None), "names", None)
    class_names = list(names.values()) if isinstance(names, dict) else list(names or [])
    return {
        "model_name": "YOLOv5s COCO20K production checkpoint",
        "model_path": checkpoint.get("checkpoint_path", "models/yolov5s_coco20k_best.pt"),
        "model_type": checkpoint.get("model_type"),
        "number_of_classes": checkpoint.get("number_of_classes") or trained.get("model_class_count"),
        "parameter_count": checkpoint.get("model_parameter_count"),
        "checkpoint_sha256": checkpoint.get("sha256"),
        "training_subset_size": 20000,
        "validation_subset_size": 2500,
        "test_subset_size": subset.get("subset_line_count"),
        "completed_epochs": 4,
        "genuine_test_precision": yolo_metrics.get("precision"),
        "genuine_test_recall": yolo_metrics.get("recall"),
        "genuine_test_map_50": yolo_metrics.get("mAP@0.5"),
        "genuine_test_map_50_95": yolo_metrics.get("mAP@0.5:0.95"),
        "official_cocoeval_ap_50_95": coco_metrics.get("AP@[0.50:0.95]"),
        "official_cocoeval_ap_50": coco_metrics.get("AP@0.50"),
        "class_names": class_names,
    }


def create_inference_engine(app_config: dict[str, Any]) -> InferenceEngine:
    """Create the shared production inference engine."""
    inference_config = load_inference_config(
        model_path=app_config["model"]["path"],
        confidence_threshold=app_config["inference"]["confidence_threshold"],
        iou_threshold=app_config["inference"]["iou_threshold"],
        image_size=app_config["model"]["image_size"],
        device=app_config["model"]["device"],
        output_directory=app_config["paths"]["output_directory"],
    )
    return InferenceEngine(inference_config)


def process_image_upload(app: Flask) -> dict[str, Any]:
    """Handle image upload, inference, annotated output, and metadata."""
    cv2 = require_python_package("cv2", "opencv-python")
    app_config = app.config["APP_CONFIG"]
    store: ResultStore = app.config["RESULT_STORE"]
    result_id = store.new_result_id()
    upload = get_upload_file()
    confidence, iou = parse_thresholds()
    class_filter = parse_optional_class_filter()
    upload_path = save_upload(
        upload,
        app.config["UPLOAD_DIR"] / "images",
        result_id,
        coerce_allowed_extensions(app_config, "images"),
    )
    validate_file_size(upload_path, app_config["inference"]["maximum_upload_size_mb"])
    width, height = validate_image_file(upload_path)
    frame = cv2.imread(str(upload_path))
    if frame is None:
        raise ValueError("Uploaded image could not be read.")

    start = time.perf_counter()
    result = run_frame_inference(app, frame, confidence, iou, class_filter, source=str(upload_path))
    output_path = safe_output_path(app.config["IMAGE_OUTPUT_DIR"], result_id, upload.filename or "image.jpg", "annotated", "jpg")
    if not cv2.imwrite(str(output_path), result.annotated_frame):
        raise RuntimeError(f"Failed to write annotated image: {output_path}")
    total_ms = (time.perf_counter() - start) * 1000.0

    detections = tuple(dict(item) for item in result.detections)
    metadata = {
        "result_id": result_id,
        "task_type": "image",
        "original_filename": upload.filename,
        "source_path": relative_path(upload_path),
        "output_path": relative_path(output_path),
        "source_url": output_url_for(app, upload_path),
        "output_url": output_url_for(app, output_path),
        "download_url": output_url_for(app, output_path),
        "image_width": width,
        "image_height": height,
        "detection_count": int(result.detection_count),
        "class_counts": class_counts(result.detections),
        "confidence_statistics": confidence_statistics(result.detections),
        "inference_time_ms": float(result.inference_ms),
        "total_processing_time_ms": total_ms,
        "confidence_threshold": confidence,
        "iou_threshold": iou,
        "class_filter": list(class_filter) if class_filter else None,
        "detections": detections,
        "timestamp": utc_timestamp(),
    }
    store.write(result_id, metadata)
    return metadata


def process_video_upload(app: Flask) -> dict[str, Any]:
    """Handle video upload, frame inference, annotated output, and metadata."""
    cv2 = require_python_package("cv2", "opencv-python")
    app_config = app.config["APP_CONFIG"]
    store: ResultStore = app.config["RESULT_STORE"]
    result_id = store.new_result_id()
    upload = get_upload_file()
    confidence, iou = parse_thresholds()
    class_filter = parse_optional_class_filter()
    upload_path = save_upload(
        upload,
        app.config["UPLOAD_DIR"] / "videos",
        result_id,
        coerce_allowed_extensions(app_config, "videos"),
    )
    validate_file_size(upload_path, app_config["inference"]["maximum_upload_size_mb"])
    source_meta = validate_video_file(upload_path)
    source_inspection = inspect_video(upload_path)
    output_fps = source_meta.fps if source_meta.fps > 0 else 30.0
    output_path = safe_output_path(app.config["VIDEO_OUTPUT_DIR"], result_id, upload.filename or "video.mp4", "annotated", "mp4")

    capture = cv2.VideoCapture(str(upload_path))
    writer = None
    processed_frames = 0
    total_detections = 0
    aggregate_counts: dict[str, int] = {}
    inference_times: list[float] = []
    start = time.perf_counter()
    try:
        if not capture.isOpened():
            raise ValueError("Uploaded video could not be opened.")
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            float(output_fps),
            (source_meta.width, source_meta.height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Could not open output video writer: {output_path}")
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            result = run_frame_inference(app, frame, confidence, iou, class_filter, source=f"{upload_path}#{processed_frames}")
            writer.write(result.annotated_frame)
            processed_frames += 1
            total_detections += int(result.detection_count)
            inference_times.append(float(result.inference_ms))
            for name, count in class_counts(result.detections).items():
                aggregate_counts[name] = aggregate_counts.get(name, 0) + count
    finally:
        capture.release()
        if writer is not None:
            writer.release()

    elapsed_seconds = time.perf_counter() - start
    generated_inspection = inspect_video(output_path)
    compatibility = ensure_browser_compatible_mp4(
        output_path,
        output_path.with_name(f"{output_path.stem}_browser.mp4"),
    )
    final_output_path = Path(compatibility["final_video"])

    metadata = {
        "result_id": result_id,
        "task_type": "video",
        "original_filename": upload.filename,
        "source_path": relative_path(upload_path),
        "output_path": relative_path(output_path),
        "final_output_path": relative_path(final_output_path),
        "output_url": output_url_for(app, final_output_path),
        "download_url": output_url_for(app, final_output_path),
        "width": source_meta.width,
        "height": source_meta.height,
        "source_fps": source_meta.fps,
        "output_fps": output_fps,
        "frame_count": source_meta.frame_count,
        "processed_frames": processed_frames,
        "duration_seconds": source_meta.duration_seconds,
        "total_detections": total_detections,
        "class_counts": dict(sorted(aggregate_counts.items())),
        "average_detections_per_frame": total_detections / processed_frames if processed_frames else 0.0,
        "average_inference_time_ms": sum(inference_times) / len(inference_times) if inference_times else 0.0,
        "average_processing_fps": processed_frames / elapsed_seconds if elapsed_seconds > 0 else 0.0,
        "total_processing_time_seconds": elapsed_seconds,
        "source_codec": source_inspection.codec,
        "generated_codec": generated_inspection.codec,
        "final_codec": compatibility.get("final_codec"),
        "browser_compatible": bool(compatibility.get("browser_compatible")),
        "compatibility_method": compatibility.get("compatibility_method"),
        "transcoded": bool(compatibility.get("transcoded")),
        "video_compatibility": compatibility,
        "confidence_threshold": confidence,
        "iou_threshold": iou,
        "class_filter": list(class_filter) if class_filter else None,
        "timestamp": utc_timestamp(),
    }
    store.write(result_id, metadata)
    return metadata


def register_routes(app: Flask) -> None:
    """Register Flask routes."""

    @app.before_request
    def attach_config_to_request() -> None:
        request.app_config = app.config["APP_CONFIG"]  # type: ignore[attr-defined]

    @app.route("/")
    def index() -> str:
        return render_template("index.html", model=app.config["MODEL_METADATA"])

    @app.route("/detect/image", methods=["POST"])
    def detect_image() -> str:
        try:
            metadata = process_image_upload(app)
            return render_template("image_result.html", result=metadata)
        except Exception as exc:
            LOGGER.exception("Image detection failed")
            return render_template("error.html", message=str(exc)), 400

    @app.route("/detect/video", methods=["POST"])
    def detect_video() -> str:
        try:
            metadata = process_video_upload(app)
            return render_template("video_result.html", result=metadata)
        except Exception as exc:
            LOGGER.exception("Video detection failed")
            return render_template("error.html", message=str(exc)), 400

    @app.route("/webcam")
    def webcam() -> str:
        return render_template("webcam.html", status=app.config["WEBCAM_STREAM"].status)

    @app.route("/video_feed")
    def video_feed() -> Response:
        stream: WebcamStream = app.config["WEBCAM_STREAM"]
        stream.start()
        return Response(stream.frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.route("/webcam/stop", methods=["POST"])
    def webcam_stop() -> Any:
        app.config["WEBCAM_STREAM"].stop()
        if request.accept_mimetypes.best == "application/json":
            return jsonify({"status": "stopped", "webcam": app.config["WEBCAM_STREAM"].status})
        return redirect(url_for("webcam"))

    @app.route("/outputs/<path:filename>")
    def serve_output(filename: str) -> Any:
        roots: list[tuple[str, Path]] = [("", app.config["OUTPUT_DIR"]), ("uploads", app.config["UPLOAD_DIR"])]
        normalized = Path(filename)
        for prefix, root in roots:
            if prefix:
                if not normalized.parts or normalized.parts[0] != prefix:
                    continue
                relative = Path(*normalized.parts[1:])
            else:
                if normalized.parts and normalized.parts[0] == "uploads":
                    continue
                relative = normalized
            target = (root / relative).resolve()
            try:
                target.relative_to(root.resolve())
            except ValueError:
                abort(404)
            if target.is_file():
                return send_file(target)
        abort(404)

    @app.route("/api/health")
    def api_health() -> Any:
        model_path = Path(app.config["APP_CONFIG"]["model"]["path"])
        engine = app.config["INFERENCE_ENGINE"]
        return jsonify(
            {
                "status": "ok",
                "model_loaded": engine is not None,
                "active_device": getattr(engine, "resolved_device", None),
                "checkpoint_exists": model_path.is_file(),
                "webcam": app.config["WEBCAM_STREAM"].status,
            }
        )

    @app.route("/api/model")
    def api_model() -> Any:
        return jsonify(app.config["MODEL_METADATA"])

    @app.route("/api/results/<result_id>")
    def api_result(result_id: str) -> Any:
        try:
            metadata = app.config["RESULT_STORE"].read(result_id)
        except ValueError:
            return jsonify({"error": "invalid result_id"}), 400
        if metadata is None:
            return jsonify({"error": "result not found"}), 404
        return jsonify(metadata)

    @app.errorhandler(RequestEntityTooLarge)
    def upload_too_large(_error: RequestEntityTooLarge) -> Any:
        return render_template("error.html", message="Uploaded file exceeds the configured maximum size."), 413


def create_app(config_path: Path | None = None, engine: Any | None = None, load_model: bool = True) -> Flask:
    """Create and configure the Flask application."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    app_config = load_app_config(config_path)
    flask_app = Flask(__name__, template_folder="templates", static_folder="static")
    maximum_upload_size = int(float(app_config["inference"]["maximum_upload_size_mb"]) * 1024 * 1024)
    flask_app.config["MAX_CONTENT_LENGTH"] = maximum_upload_size
    flask_app.config["APP_CONFIG"] = app_config
    flask_app.config["UPLOAD_DIR"] = Path(app_config["paths"]["upload_directory"])
    flask_app.config["OUTPUT_DIR"] = Path(app_config["paths"]["output_directory"])
    flask_app.config["IMAGE_OUTPUT_DIR"] = Path(app_config["paths"]["image_output_directory"])
    flask_app.config["VIDEO_OUTPUT_DIR"] = Path(app_config["paths"]["video_output_directory"])
    flask_app.config["METADATA_DIR"] = Path(app_config["paths"]["metadata_directory"])
    for directory in [
        flask_app.config["UPLOAD_DIR"],
        flask_app.config["OUTPUT_DIR"],
        flask_app.config["IMAGE_OUTPUT_DIR"],
        flask_app.config["VIDEO_OUTPUT_DIR"],
        flask_app.config["METADATA_DIR"],
    ]:
        ensure_dir(directory)

    shared_engine = engine if engine is not None else (create_inference_engine(app_config) if load_model else None)
    flask_app.config["INFERENCE_ENGINE"] = shared_engine
    flask_app.config["MODEL_LOCK"] = threading.Lock()
    flask_app.config["RESULT_STORE"] = ResultStore(flask_app.config["METADATA_DIR"])
    flask_app.config["MODEL_METADATA"] = build_model_metadata(shared_engine)
    flask_app.config["WEBCAM_STREAM"] = WebcamStream(
        shared_engine,
        camera_index=app_config["webcam"]["default_camera_index"],
        image_size=app_config["model"]["image_size"],
        jpeg_quality=app_config["webcam"]["jpeg_quality"],
        target_fps=app_config["webcam"]["target_fps"],
        model_lock=flask_app.config["MODEL_LOCK"],
    )
    register_routes(flask_app)
    return flask_app


app = create_app(load_model=os.environ.get("YOLOV5_FLASK_LOAD_MODEL_ON_IMPORT") == "1")


if __name__ == "__main__":
    runtime_app = create_app(load_model=True)
    cfg = runtime_app.config["APP_CONFIG"]["app"]
    runtime_app.run(
        host=str(cfg["host"]),
        port=int(cfg["port"]),
        debug=bool(cfg["debug"]),
        threaded=bool(cfg["threaded"]),
        use_reloader=False,
    )
