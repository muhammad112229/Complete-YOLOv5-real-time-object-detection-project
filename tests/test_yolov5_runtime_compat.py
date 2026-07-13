"""Tests for project-owned YOLOv5 runtime compatibility shims."""

from __future__ import annotations

import os

from src.common import ensure_project_pythonpath, project_root, require_python_package
from src.yolov5_runtime_compat import apply_yolov5_runtime_compatibility


def test_yolov5_runtime_compatibility_restores_expected_apis() -> None:
    result = apply_yolov5_runtime_compatibility()
    numpy = require_python_package("numpy")
    image_font = require_python_package("PIL.ImageFont", "Pillow")

    font = image_font.load_default()
    assert hasattr(numpy, "trapz")
    assert hasattr(font, "getsize")
    assert set(result) == {"numpy_trapz", "pillow_getsize"}


def test_project_pythonpath_includes_repository_root() -> None:
    env: dict[str, str] = {}
    ensure_project_pythonpath(env)
    paths = env["PYTHONPATH"].split(os.pathsep)
    assert str(project_root().resolve()) in paths
