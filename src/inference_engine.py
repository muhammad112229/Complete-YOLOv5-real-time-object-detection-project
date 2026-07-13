"""Reusable YOLOv5 inference engine with project-level defaults."""

from __future__ import annotations

import argparse
import contextlib
import inspect
import importlib
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from src.common import project_root, require_file, require_python_package, yolov5_root
from src.yolov5_runtime_compat import apply_yolov5_runtime_compatibility


Detection = dict[str, float | int | str]
ClassFilter = tuple[int | str, ...] | None

DEFAULT_CONFIG_PATH = Path("configs") / "inference.yaml"
DEFAULT_MODEL_PATH = Path("models") / "yolov5s_coco20k_best.pt"
DEFAULT_OUTPUT_DIRECTORY = Path("outputs") / "inference"

_MODEL_CACHE: dict[tuple[str, str, float, float], Any] = {}


def _parse_yaml_scalar(value: str) -> Any:
    text = value.strip()
    if text.lower() in {"null", "none", "~"}:
        return None
    if text.lower() == "true":
        return True
    if text.lower() == "false":
        return False
    if text.startswith(("'", '"')) and text.endswith(("'", '"')):
        return text[1:-1]
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


def _load_flat_yaml(path: Path) -> dict[str, Any]:
    """Load the flat inference YAML without requiring PyYAML."""
    data: dict[str, Any] = {}
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if ":" not in line:
            raise ValueError(f"Invalid inference config line {line_number}: {raw_line}")
        key, value = line.split(":", 1)
        data[key.strip()] = _parse_yaml_scalar(value)
    return data


@dataclass(frozen=True)
class InferenceConfig:
    """Runtime configuration for YOLOv5 inference."""

    model_path: Path
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45
    image_size: int = 640
    device: str = "auto"
    class_filter: ClassFilter = None
    output_directory: Path = DEFAULT_OUTPUT_DIRECTORY


@dataclass(frozen=True)
class InferenceResult:
    """Per-frame inference result with annotated frame and structured detections."""

    annotated_frame: Any
    inference_ms: float
    fps: float
    preprocess_ms: float | None = None
    model_inference_ms: float | None = None
    nms_ms: float | None = None
    detection_count: int = 0
    detected_class_names: tuple[str, ...] = ()
    detections: tuple[Detection, ...] = ()
    source: str | None = None


def _resolve_path(path: str | Path, root: Path | None = None) -> Path:
    base = root or project_root()
    candidate = Path(path)
    return candidate if candidate.is_absolute() else base / candidate


def _coerce_class_filter(value: Any) -> ClassFilter:
    if value is None:
        return None
    if isinstance(value, (str, int)):
        return (value,)
    if isinstance(value, Iterable):
        items = tuple(value)
        if not all(isinstance(item, (str, int)) for item in items):
            raise ValueError("class_filter entries must be class IDs or class names.")
        return items
    raise ValueError("class_filter must be null, a class ID/name, or a list of class IDs/names.")


def load_inference_config(config_path: Path | None = None, **overrides: Any) -> InferenceConfig:
    """Load central inference YAML and apply optional overrides."""
    root = project_root()
    path = _resolve_path(config_path or DEFAULT_CONFIG_PATH, root)
    data: dict[str, Any] = {}
    if path.exists():
        try:
            yaml = require_python_package("yaml", "PyYAML")
            with path.open("r", encoding="utf-8") as file:
                loaded = yaml.safe_load(file) or {}
        except RuntimeError:
            loaded = _load_flat_yaml(path)
        if not isinstance(loaded, dict):
            raise ValueError(f"Expected YAML mapping in inference config: {path}")
        data = loaded

    data = {
        "model_path": DEFAULT_MODEL_PATH,
        "confidence_threshold": 0.25,
        "iou_threshold": 0.45,
        "image_size": 640,
        "device": "auto",
        "class_filter": None,
        "output_directory": DEFAULT_OUTPUT_DIRECTORY,
        **data,
    }
    for key, value in overrides.items():
        if value is not None:
            data[key] = value

    return InferenceConfig(
        model_path=_resolve_path(data["model_path"], root),
        confidence_threshold=float(data["confidence_threshold"]),
        iou_threshold=float(data["iou_threshold"]),
        image_size=int(data["image_size"]),
        device=str(data["device"]),
        class_filter=_coerce_class_filter(data.get("class_filter")),
        output_directory=_resolve_path(data["output_directory"], root),
    )


def parse_class_filter(values: list[str] | None) -> ClassFilter:
    """Parse CLI class filters as IDs where possible, otherwise names."""
    if not values:
        return None
    parsed: list[int | str] = []
    for value in values:
        parsed.append(int(value) if value.strip().isdigit() else value.strip())
    return tuple(parsed)


def resolve_device(device: str = "auto") -> str:
    """Resolve auto/cuda/cpu device settings for YOLOv5 v7.0."""
    normalized = str(device).strip().lower()
    if normalized in {"", "auto"}:
        torch = require_python_package("torch")
        return "0" if torch.cuda.is_available() else "cpu"
    if normalized in {"cuda", "gpu"}:
        torch = require_python_package("torch")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")
        return "0"
    return str(device)


def validate_yolov5_v7(root: Path | None = None) -> dict[str, str | bool | None]:
    """Check that the local YOLOv5 repository is present and preferably pinned to v7.0."""
    yolo_root = yolov5_root(root or project_root())
    hubconf = yolo_root / "hubconf.py"
    if not hubconf.is_file():
        raise FileNotFoundError(f"Missing YOLOv5 hubconf.py: {hubconf}")

    tag: str | None = None
    commit: str | None = None
    if (yolo_root / ".git").exists():
        try:
            tag_result = subprocess.run(
                ["git", "-C", str(yolo_root), "describe", "--tags", "--exact-match"],
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
            tag = tag_result.stdout.strip() or None
            commit_result = subprocess.run(
                ["git", "-C", str(yolo_root), "rev-parse", "HEAD"],
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
            commit = commit_result.stdout.strip() or None
        except (OSError, subprocess.SubprocessError):
            tag = None
            commit = None
    return {"path": str(yolo_root), "tag": tag, "commit": commit, "is_v7": tag == "v7.0"}


def patch_yolov5_git_describe(root: Path | None = None) -> None:
    """Patch YOLOv5's unquoted git command so workspace paths with spaces are quiet."""
    yolo_root = yolov5_root(root or project_root())
    if str(yolo_root) not in sys.path:
        sys.path.insert(0, str(yolo_root))

    def safe_git_describe(path: Path = yolo_root) -> str:
        try:
            target = Path(path)
            if not (target / ".git").is_dir():
                return ""
            result = subprocess.run(
                ["git", "-C", str(target), "describe", "--tags", "--long", "--always"],
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
            return result.stdout.strip() if result.returncode == 0 else ""
        except (OSError, subprocess.SubprocessError):
            return ""

    general = importlib.import_module("utils.general")
    general.git_describe = safe_git_describe
    torch_utils = sys.modules.get("utils.torch_utils")
    if torch_utils is not None:
        torch_utils.git_describe = safe_git_describe


@contextlib.contextmanager
def _torch_load_yolov5_v7_compat(torch: Any) -> Any:
    """Keep YOLOv5 v7.0 checkpoint loading compatible with newer torch.load defaults."""
    original_load = torch.load
    try:
        signature = inspect.signature(original_load)
    except (TypeError, ValueError):
        signature = None
    if signature is None or "weights_only" not in signature.parameters:
        yield
        return

    def patched_load(*args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("weights_only", False)
        return original_load(*args, **kwargs)

    torch.load = patched_load
    try:
        yield
    finally:
        torch.load = original_load


def load_yolov5_model(
    weights: Path,
    device: str = "auto",
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
) -> Any:
    """Load and cache a local YOLOv5 v7.0 checkpoint without network access."""
    weights = require_file(Path(weights), "model weights").resolve()
    yolo_info = validate_yolov5_v7(project_root())
    if yolo_info["tag"] is not None and not yolo_info["is_v7"]:
        raise RuntimeError(f"Expected YOLOv5 v7.0, found tag {yolo_info['tag']} at {yolo_info['path']}")
    torch = require_python_package("torch")
    os.environ.setdefault("YOLOv5_AUTOINSTALL", "False")
    apply_yolov5_runtime_compatibility()
    patch_yolov5_git_describe(project_root())
    resolved_device = resolve_device(device)
    cache_key = (str(weights), resolved_device, float(conf_thres), float(iou_thres))
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    with _torch_load_yolov5_v7_compat(torch):
        model = torch.hub.load(
            str(yolov5_root(project_root())),
            "custom",
            path=str(weights),
            source="local",
            device=resolved_device,
            _verbose=False,
        )
    model.conf = float(conf_thres)
    model.iou = float(iou_thres)
    _MODEL_CACHE[cache_key] = model
    return model


def _class_name(class_id: int, class_names: dict[int, str] | list[str] | None) -> str:
    if isinstance(class_names, dict):
        return str(class_names.get(class_id, class_id))
    if isinstance(class_names, list) and 0 <= class_id < len(class_names):
        return str(class_names[class_id])
    return str(class_id)


def _allowed_class_ids(class_filter: ClassFilter, class_names: dict[int, str] | list[str] | None) -> set[int] | None:
    if class_filter is None:
        return None
    if isinstance(class_names, dict):
        lookup = {str(name).lower(): int(class_id) for class_id, name in class_names.items()}
    elif isinstance(class_names, list):
        lookup = {str(name).lower(): index for index, name in enumerate(class_names)}
    else:
        lookup = {}

    allowed: set[int] = set()
    unknown: list[str] = []
    for item in class_filter:
        if isinstance(item, int):
            allowed.add(item)
            continue
        text = item.strip()
        if text.isdigit():
            allowed.add(int(text))
        elif text.lower() in lookup:
            allowed.add(lookup[text.lower()])
        else:
            unknown.append(text)
    if unknown:
        raise ValueError(f"Unknown class_filter entries for this model: {', '.join(unknown)}")
    return allowed


def filter_detections(detections: Any, allowed_class_ids: set[int] | None) -> Any:
    """Filter raw YOLO detections by class ID while preserving tensor structure."""
    if allowed_class_ids is None:
        return detections
    keep_indices = [index for index, detection in enumerate(detections) if int(detection.tolist()[5]) in allowed_class_ids]
    if len(keep_indices) == len(detections):
        return detections
    if not keep_indices:
        return detections[:0]
    return detections[keep_indices]


def format_detections(
    detections: Any,
    class_names: dict[int, str] | list[str] | None = None,
) -> tuple[Detection, ...]:
    """Convert YOLO tensor detections into serializable dictionaries."""
    formatted: list[Detection] = []
    for detection in detections:
        x1, y1, x2, y2, confidence, class_id = detection.tolist()
        class_index = int(class_id)
        formatted.append(
            {
                "class_id": class_index,
                "class_name": _class_name(class_index, class_names),
                "confidence": float(confidence),
                "x1": float(x1),
                "y1": float(y1),
                "x2": float(x2),
                "y2": float(y2),
            }
        )
    return tuple(formatted)


def annotate_frame(frame: Any, detections: Any, class_names: dict[int, str] | list[str] | None = None) -> Any:
    """Draw YOLO detections onto an OpenCV BGR frame."""
    cv2 = require_python_package("cv2", "opencv-python")
    annotated = frame.copy()
    for detection in detections:
        x1, y1, x2, y2, confidence, class_id = detection.tolist()
        class_index = int(class_id)
        label = f"{_class_name(class_index, class_names)} {float(confidence):.2f}"
        cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(
            annotated,
            label,
            (int(x1), max(20, int(y1) - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
    return annotated


def infer_frame_with_model(
    model: Any,
    frame: Any,
    image_size: int = 640,
    class_filter: ClassFilter = None,
    source: str | None = None,
) -> InferenceResult:
    """Run inference on one OpenCV BGR frame array."""
    cv2 = require_python_package("cv2", "opencv-python")
    if frame is None or not hasattr(frame, "shape"):
        raise ValueError("frame must be a valid image array.")
    if len(frame.shape) != 3 or frame.shape[2] != 3:
        raise ValueError(f"frame must have shape HxWx3, got {getattr(frame, 'shape', None)}")

    start = time.perf_counter()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(rgb, size=int(image_size))
    inference_ms = (time.perf_counter() - start) * 1000.0
    timings = list(getattr(results, "t", []))
    class_names = getattr(model, "names", None)
    raw_detections = results.xyxy[0].detach().cpu()
    allowed = _allowed_class_ids(class_filter, class_names)
    raw_detections = filter_detections(raw_detections, allowed)
    annotated = annotate_frame(frame, raw_detections, class_names)
    formatted = format_detections(raw_detections, class_names)
    detected_class_names = tuple(sorted({str(item["class_name"]) for item in formatted}))
    fps = 1000.0 / inference_ms if inference_ms > 0 else 0.0
    return InferenceResult(
        annotated_frame=annotated,
        inference_ms=inference_ms,
        fps=fps,
        preprocess_ms=float(timings[0]) if len(timings) > 0 else None,
        model_inference_ms=float(timings[1]) if len(timings) > 1 else None,
        nms_ms=float(timings[2]) if len(timings) > 2 else None,
        detection_count=len(formatted),
        detected_class_names=detected_class_names,
        detections=formatted,
        source=source,
    )


def open_video_writer(output_path: Path, fps: float, frame_width: int, frame_height: int) -> Any:
    """Open an OpenCV video writer or raise a clear error."""
    cv2 = require_python_package("cv2", "opencv-python")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (frame_width, frame_height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {output_path}")
    return writer


def summarize_detections(detections: Iterable[Detection]) -> dict[str, int]:
    """Count detections by class name."""
    summary: dict[str, int] = {}
    for detection in detections:
        name = str(detection["class_name"])
        summary[name] = summary.get(name, 0) + 1
    return dict(sorted(summary.items()))


class InferenceEngine:
    """Reusable inference engine for images, videos, webcams, and frame arrays."""

    def __init__(self, config: InferenceConfig | None = None) -> None:
        self.config = config or load_inference_config()
        self.resolved_device = resolve_device(self.config.device)
        self.model = load_yolov5_model(
            self.config.model_path,
            self.resolved_device,
            self.config.confidence_threshold,
            self.config.iou_threshold,
        )

    def predict_frame(self, frame: Any, source: str | None = None) -> InferenceResult:
        """Run inference on a BGR frame array."""
        return infer_frame_with_model(
            self.model,
            frame,
            self.config.image_size,
            self.config.class_filter,
            source,
        )

    def predict_image(self, source: Path) -> InferenceResult:
        """Run inference on one image path."""
        cv2 = require_python_package("cv2", "opencv-python")
        source = require_file(Path(source), "source image")
        frame = cv2.imread(str(source))
        if frame is None:
            raise RuntimeError(f"Could not read image: {source}")
        return self.predict_frame(frame, str(source))

    def save_annotated_image(self, result: InferenceResult, output_path: Path) -> Path:
        """Save an annotated image result."""
        cv2 = require_python_package("cv2", "opencv-python")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output_path), result.annotated_frame):
            raise RuntimeError(f"Failed to write image: {output_path}")
        return output_path

    def predict_video(
        self,
        source: Path,
        output_path: Path,
        display: bool = False,
        max_frames: int | None = None,
    ) -> dict[str, Any]:
        """Run inference on a video path and save an annotated video."""
        cv2 = require_python_package("cv2", "opencv-python")
        source = require_file(Path(source), "source video")
        capture = cv2.VideoCapture(str(source))
        if not capture.isOpened():
            raise RuntimeError(f"Could not open video: {source}")

        fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = open_video_writer(output_path, fps, width, height)
        frames: list[dict[str, Any]] = []
        class_counts: dict[str, int] = {}
        frame_index = 0
        try:
            while True:
                if max_frames is not None and frame_index >= max_frames:
                    break
                ok, frame = capture.read()
                if not ok:
                    break
                result = self.predict_frame(frame, f"{source}#{frame_index}")
                cv2.putText(
                    result.annotated_frame,
                    f"FPS {result.fps:.1f} | {result.inference_ms:.1f} ms",
                    (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                writer.write(result.annotated_frame)
                frame_summary = summarize_detections(result.detections)
                for name, count in frame_summary.items():
                    class_counts[name] = class_counts.get(name, 0) + count
                frames.append(
                    {
                        "frame_index": frame_index,
                        "detection_count": result.detection_count,
                        "detections": list(result.detections),
                    }
                )
                frame_index += 1
                if display:
                    cv2.imshow("YOLOv5 video detection", result.annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
        finally:
            capture.release()
            writer.release()
            if display:
                cv2.destroyAllWindows()

        return {
            "source": str(source),
            "output_path": str(output_path),
            "frame_count": frame_index,
            "total_detections": sum(item["detection_count"] for item in frames),
            "class_counts": dict(sorted(class_counts.items())),
            "frames": frames,
        }

    def predict_webcam(
        self,
        camera_index: int,
        output_path: Path | None = None,
        display: bool = True,
        max_frames: int | None = None,
    ) -> dict[str, Any]:
        """Run inference on a webcam index and optionally save annotated video."""
        cv2 = require_python_package("cv2", "opencv-python")
        capture = cv2.VideoCapture(int(camera_index))
        if not capture.isOpened():
            raise RuntimeError(f"Could not open camera index {camera_index}")

        writer = None
        frames: list[dict[str, Any]] = []
        class_counts: dict[str, int] = {}
        frame_index = 0
        try:
            while True:
                if max_frames is not None and frame_index >= max_frames:
                    break
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError("Camera frame capture failed.")
                result = self.predict_frame(frame, f"webcam:{camera_index}#{frame_index}")
                cv2.putText(
                    result.annotated_frame,
                    f"FPS {result.fps:.1f} | {result.inference_ms:.1f} ms",
                    (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                if output_path is not None and writer is None:
                    writer = open_video_writer(output_path, 30.0, frame.shape[1], frame.shape[0])
                if writer is not None:
                    writer.write(result.annotated_frame)
                frame_summary = summarize_detections(result.detections)
                for name, count in frame_summary.items():
                    class_counts[name] = class_counts.get(name, 0) + count
                frames.append(
                    {
                        "frame_index": frame_index,
                        "detection_count": result.detection_count,
                        "detections": list(result.detections),
                    }
                )
                frame_index += 1
                if display:
                    cv2.imshow("YOLOv5 webcam detection - press Q to quit", result.annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
        finally:
            capture.release()
            if writer is not None:
                writer.release()
            if display:
                cv2.destroyAllWindows()

        return {
            "camera_index": camera_index,
            "output_path": str(output_path) if output_path else None,
            "frame_count": frame_index,
            "total_detections": sum(item["detection_count"] for item in frames),
            "class_counts": dict(sorted(class_counts.items())),
            "frames": frames,
        }


def add_common_inference_args(parser: argparse.ArgumentParser) -> None:
    """Add central-config override arguments to deployment CLIs."""
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--model-path", "--weights", dest="model_path", type=Path)
    parser.add_argument("--conf-thres", "--confidence-threshold", dest="confidence_threshold", type=float)
    parser.add_argument("--iou-thres", "--iou-threshold", dest="iou_threshold", type=float)
    parser.add_argument("--imgsz", "--image-size", dest="image_size", type=int)
    parser.add_argument("--device")
    parser.add_argument("--class-filter", nargs="+")
    parser.add_argument("--output-directory", type=Path)


def config_from_args(args: argparse.Namespace) -> InferenceConfig:
    """Build an inference config from argparse values."""
    return load_inference_config(
        args.config,
        model_path=args.model_path,
        confidence_threshold=args.confidence_threshold,
        iou_threshold=args.iou_threshold,
        image_size=args.image_size,
        device=args.device,
        class_filter=parse_class_filter(args.class_filter),
        output_directory=args.output_directory,
    )
