"""Quantize exported YOLOv5 ONNX models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from src.common import file_size_mb, require_file, require_python_package


def _letterbox(image: Any, new_shape: int) -> Any:
    """Resize with YOLO-style letterbox padding for calibration."""
    cv2 = require_python_package("cv2", "opencv-python")
    import numpy as np

    shape = image.shape[:2]
    scale = min(new_shape / shape[0], new_shape / shape[1])
    resized = (int(round(shape[1] * scale)), int(round(shape[0] * scale)))
    pad_w = new_shape - resized[0]
    pad_h = new_shape - resized[1]
    image = cv2.resize(image, resized, interpolation=cv2.INTER_LINEAR)
    top, bottom = pad_h // 2, pad_h - pad_h // 2
    left, right = pad_w // 2, pad_w - pad_w // 2
    return cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))


class ImageCalibrationDataReader:
    """ONNX Runtime calibration reader for image folders."""

    def __init__(self, image_paths: Iterable[Path], input_name: str, imgsz: int) -> None:
        self.image_paths = list(image_paths)
        self.input_name = input_name
        self.imgsz = imgsz
        self.index = 0

    def get_next(self) -> dict[str, Any] | None:
        """Return the next calibration batch."""
        cv2 = require_python_package("cv2", "opencv-python")
        import numpy as np

        if self.index >= len(self.image_paths):
            return None
        path = self.image_paths[self.index]
        self.index += 1
        image = cv2.imread(str(path))
        if image is None:
            raise RuntimeError(f"Could not read calibration image: {path}")
        image = _letterbox(image, self.imgsz)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = image.transpose(2, 0, 1).astype(np.float32) / 255.0
        tensor = np.expand_dims(tensor, axis=0)
        return {self.input_name: tensor}


def quantize_dynamic_onnx(input_model: Path, output_model: Path) -> dict[str, object]:
    """Apply dynamic INT8 quantization to an ONNX model."""
    quantization = require_python_package("onnxruntime.quantization", "onnxruntime")
    require_file(input_model, "ONNX model")
    output_model.parent.mkdir(parents=True, exist_ok=True)
    quantization.quantize_dynamic(
        model_input=str(input_model),
        model_output=str(output_model),
        weight_type=quantization.QuantType.QInt8,
    )
    return {
        "method": "dynamic_onnx_int8",
        "input_model": str(input_model),
        "output_model": str(output_model),
        "input_size_mb": file_size_mb(input_model),
        "output_size_mb": file_size_mb(output_model),
        "status": "created; evaluate mAP/AP50/AP75 before reporting accuracy",
    }


def quantize_static_onnx(
    input_model: Path,
    output_model: Path,
    calibration_dir: Path,
    imgsz: int,
) -> dict[str, object]:
    """Apply static INT8 quantization with image calibration data."""
    ort = require_python_package("onnxruntime", "onnxruntime")
    quantization = require_python_package("onnxruntime.quantization", "onnxruntime")
    require_file(input_model, "ONNX model")
    image_paths = sorted(
        path
        for pattern in ("*.jpg", "*.jpeg", "*.png")
        for path in calibration_dir.glob(pattern)
    )
    if not image_paths:
        raise ValueError(f"No calibration images found in {calibration_dir}")
    session = ort.InferenceSession(str(input_model), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    reader = ImageCalibrationDataReader(image_paths, input_name, imgsz)
    output_model.parent.mkdir(parents=True, exist_ok=True)
    quantization.quantize_static(
        model_input=str(input_model),
        model_output=str(output_model),
        calibration_data_reader=reader,
        quant_format=quantization.QuantFormat.QDQ,
        activation_type=quantization.QuantType.QUInt8,
        weight_type=quantization.QuantType.QInt8,
    )
    return {
        "method": "static_onnx_int8",
        "input_model": str(input_model),
        "output_model": str(output_model),
        "calibration_images": len(image_paths),
        "input_size_mb": file_size_mb(input_model),
        "output_size_mb": file_size_mb(output_model),
        "status": "created; evaluate mAP/AP50/AP75 before reporting accuracy",
    }


def main() -> int:
    """CLI entrypoint for quantization."""
    parser = argparse.ArgumentParser(description="Quantize a YOLOv5 ONNX model.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method", choices=["dynamic-onnx", "static-onnx"], default="dynamic-onnx")
    parser.add_argument("--calibration-dir", type=Path)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--report", type=Path, default=Path("results/comparisons/quantization_report.json"))
    args = parser.parse_args()

    if args.method == "dynamic-onnx":
        report = quantize_dynamic_onnx(args.input, args.output)
    else:
        if args.calibration_dir is None:
            raise ValueError("--calibration-dir is required for static ONNX quantization")
        report = quantize_static_onnx(args.input, args.output, args.calibration_dir, args.imgsz)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

