"""Shared helpers for project scripts."""

from __future__ import annotations

import importlib
import logging
import os
import subprocess
from pathlib import Path
from typing import Any


def project_root() -> Path:
    """Return the repository root directory."""
    return Path(__file__).resolve().parents[1]


def yolov5_root(root: Path | None = None) -> Path:
    """Return the pinned YOLOv5 repository path."""
    base = root or project_root()
    return base / "external" / "yolov5"


def setup_logging(level: str = "INFO") -> None:
    """Configure process-wide logging."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def ensure_dir(path: Path) -> Path:
    """Create a directory if it does not exist and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def require_file(path: Path, label: str = "file") -> Path:
    """Return a path if it exists, otherwise raise a clear error."""
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")
    return path


def require_python_package(import_name: str, package_name: str | None = None) -> Any:
    """Import a dependency or raise an installation-focused error."""
    try:
        return importlib.import_module(import_name)
    except ImportError as exc:
        name = package_name or import_name
        raise RuntimeError(
            f"Missing dependency '{name}'. Install project dependencies with "
            "'python -m pip install -r requirements.txt'."
        ) from exc


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file using PyYAML."""
    yaml = require_python_package("yaml", "PyYAML")
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    return data


def ensure_project_pythonpath(env: dict[str, str], root: Path | None = None) -> dict[str, str]:
    """Ensure project startup hooks are visible to Python subprocesses."""
    project = str((root or project_root()).resolve())
    existing = env.get("PYTHONPATH")
    parts = [part for part in (existing or "").split(os.pathsep) if part]
    if project not in parts:
        parts.insert(0, project)
    env["PYTHONPATH"] = os.pathsep.join(parts)
    return env


def run_command(
    command: list[str],
    cwd: Path | None = None,
    dry_run: bool = False,
    log_path: Path | None = None,
) -> int:
    """Run a subprocess command with logging."""
    logging.getLogger(__name__).info("Command: %s", " ".join(command))
    if dry_run:
        return 0
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    ensure_project_pythonpath(env)
    if log_path:
        ensure_dir(log_path.parent)
        with log_path.open("w", encoding="utf-8", errors="replace") as log_file:
            completed = subprocess.run(
                command,
                cwd=cwd,
                check=False,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                env=env,
            )
    else:
        completed = subprocess.run(command, cwd=cwd, check=False, env=env)
    return int(completed.returncode)


def file_size_mb(path: Path) -> float:
    """Return a file size in megabytes."""
    return round(path.stat().st_size / 1024**2, 3)
