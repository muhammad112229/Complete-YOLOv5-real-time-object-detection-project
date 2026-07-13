"""Verify that the local YOLOv5 project scaffold is usable.

This script performs lightweight checks only. It does not download COCO,
download model weights, or start training.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from src.environment import audit_environment, write_audit


ROOT = Path(__file__).resolve().parent
YOLOV5_ROOT = ROOT / "external" / "yolov5"
REQUIRED_PATHS = [
    ROOT / "configs" / "project.yaml",
    ROOT / "configs" / "coco_split.yaml",
    ROOT / "src" / "download_coco.py",
    ROOT / "src" / "train_models.py",
    ROOT / "src" / "detect_webcam.py",
    ROOT / "notebooks" / "YOLOv5_COCO_Training_Colab.ipynb",
    YOLOV5_ROOT / "train.py",
    YOLOV5_ROOT / "val.py",
    YOLOV5_ROOT / "detect.py",
]
OPTIONAL_RUNTIME_PACKAGES = [
    "torch",
    "torchvision",
    "cv2",
    "pycocotools",
    "yaml",
    "onnx",
    "onnxruntime",
    "psutil",
]


def runtime_details() -> dict[str, object]:
    """Return runtime package and device details for verification output."""
    details: dict[str, object] = {
        "selected_local_device": "cpu",
        "cuda_tensor_check": "not run; CUDA unavailable",
    }
    try:
        import torch

        details["torch_version"] = torch.__version__
        details["torch_cuda_runtime"] = torch.version.cuda
        details["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            tensor = torch.ones((2, 2), device="cuda") + 1
            details["cuda_tensor_check"] = float(tensor.sum().item())
            details["selected_local_device"] = "cuda:0"
            details["cuda_device_name"] = torch.cuda.get_device_name(0)
    except Exception as exc:
        details["torch_error"] = str(exc)

    package_imports = {
        "torchvision_version": "torchvision",
        "opencv_version": "cv2",
        "numpy_version": "numpy",
        "pandas_version": "pandas",
        "matplotlib_version": "matplotlib",
        "pyyaml_version": "yaml",
        "pillow_version": "PIL",
        "tqdm_version": "tqdm",
        "onnx_version": "onnx",
        "onnxruntime_version": "onnxruntime",
    }
    for output_key, import_name in package_imports.items():
        try:
            module = __import__(import_name)
            details[output_key] = getattr(module, "__version__", "installed")
        except Exception as exc:
            details[output_key] = f"unavailable: {exc}"

    try:
        __import__("pycocotools")
        details["pycocotools_available"] = True
    except Exception:
        details["pycocotools_available"] = False
    return details


def run_git(args: list[str]) -> str:
    """Run a Git command and return stdout."""
    completed = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return completed.stdout.strip()


def verify_yolov5_tag() -> tuple[bool, str]:
    """Return whether the cloned YOLOv5 repository is pinned to v7.0."""
    if not (YOLOV5_ROOT / ".git").exists():
        return False, "external/yolov5 is missing or is not a Git repository"
    try:
        tag = subprocess.run(
            ["git", "-C", str(YOLOV5_ROOT), "describe", "--tags", "--exact-match"],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout.strip()
        commit = subprocess.run(
            ["git", "-C", str(YOLOV5_ROOT), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout.strip()
    except subprocess.CalledProcessError as exc:
        return False, exc.stderr.strip() or str(exc)
    return tag == "v7.0", f"{tag} ({commit})"


def main() -> int:
    """Run setup verification."""
    parser = argparse.ArgumentParser(description="Verify project setup.")
    parser.add_argument("--strict", action="store_true", help="Fail on missing optional runtime packages.")
    args = parser.parse_args()

    audit = audit_environment(ROOT)
    write_audit(audit, ROOT / "artifacts")

    errors: list[str] = []
    warnings: list[str] = []

    if tuple(audit.python_version_info[:2]) != (3, 11):
        errors.append(f"Expected Python 3.11, found {audit.python_version}.")

    for path in REQUIRED_PATHS:
        if not path.exists():
            errors.append(f"Missing required path: {path.relative_to(ROOT)}")

    ok, detail = verify_yolov5_tag()
    if not ok:
        errors.append(f"YOLOv5 tag verification failed: {detail}")

    for package in OPTIONAL_RUNTIME_PACKAGES:
        if audit.packages.get(package) is None:
            warnings.append(f"Runtime package not installed: {package}")

    report = {
        "python": audit.python_version,
        "python_executable": audit.python_executable,
        "yolov5": detail,
        "nvidia_smi": audit.nvidia_smi.get("stdout") or audit.nvidia_smi.get("stderr") or audit.nvidia_smi.get("error"),
        **runtime_details(),
        "warnings": warnings,
        "errors": errors,
    }
    print(json.dumps(report, indent=2))

    if warnings and args.strict:
        errors.extend(warnings)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
