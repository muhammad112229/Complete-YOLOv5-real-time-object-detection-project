"""Environment auditing utilities."""

from __future__ import annotations

import importlib.util
import json
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PACKAGE_IMPORTS = [
    "torch",
    "torchvision",
    "cv2",
    "numpy",
    "pandas",
    "matplotlib",
    "yaml",
    "pycocotools",
    "onnx",
    "onnxruntime",
    "psutil",
    "PIL",
    "pytest",
]


@dataclass(frozen=True)
class EnvironmentAudit:
    """Serializable environment audit result."""

    generated_at_utc: str
    workspace: str
    os: dict[str, str]
    python_executable: str
    python_version: str
    python_version_info: list[int]
    pip: dict[str, Any]
    git: dict[str, Any]
    nvidia_smi: dict[str, Any]
    disk: dict[str, float]
    packages: dict[str, str | None]
    pytorch: dict[str, Any]
    opencv: dict[str, Any]


def run_command_capture(command: list[str], timeout: int = 20) -> dict[str, Any]:
    """Run a command and capture stdout/stderr without raising."""
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )
    except Exception as exc:
        return {"error": str(exc)}
    return {
        "returncode": int(completed.returncode),
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def package_version(import_name: str) -> str | None:
    """Return a package version if importable."""
    if importlib.util.find_spec(import_name) is None:
        return None
    try:
        module = __import__(import_name)
    except Exception as exc:
        return f"import-error: {exc}"
    return str(getattr(module, "__version__", "installed"))


def audit_environment(root: Path) -> EnvironmentAudit:
    """Collect system, tooling, package, CUDA, and disk information."""
    root = root.resolve()
    usage = shutil.disk_usage(root)
    packages = {name: package_version(name) for name in PACKAGE_IMPORTS}

    try:
        import torch

        pytorch: dict[str, Any] = {
            "version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_version": getattr(torch.version, "cuda", None),
            "device_count": int(torch.cuda.device_count()),
            "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        }
    except Exception as exc:
        pytorch = {"available": False, "error": str(exc)}

    try:
        import cv2

        opencv: dict[str, Any] = {"version": cv2.__version__}
    except Exception as exc:
        opencv = {"available": False, "error": str(exc)}

    return EnvironmentAudit(
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        workspace=str(root),
        os={
            "platform": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
        },
        python_executable=sys.executable,
        python_version=sys.version.split()[0],
        python_version_info=list(sys.version_info[:3]),
        pip=run_command_capture([sys.executable, "-m", "pip", "--version"]),
        git=run_command_capture(["git", "--version"]),
        nvidia_smi=run_command_capture(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"]
        ),
        disk={
            "total_gb": round(usage.total / 1024**3, 2),
            "used_gb": round(usage.used / 1024**3, 2),
            "free_gb": round(usage.free / 1024**3, 2),
        },
        packages=packages,
        pytorch=pytorch,
        opencv=opencv,
    )


def write_audit(audit: EnvironmentAudit, artifacts_dir: Path) -> None:
    """Write JSON and Markdown environment audit artifacts."""
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    audit_dict = asdict(audit)
    (artifacts_dir / "environment_audit.json").write_text(
        json.dumps(audit_dict, indent=2),
        encoding="utf-8",
    )

    nvidia = audit.nvidia_smi.get("stdout") or audit.nvidia_smi.get("stderr") or audit.nvidia_smi.get("error")
    lines = [
        "# Environment Audit",
        "",
        f"Generated UTC: `{audit.generated_at_utc}`",
        f"Workspace: `{audit.workspace}`",
        "",
        "## System",
        f"- OS: {audit.os['platform']}",
        f"- Machine: {audit.os['machine']}",
        f"- Disk free: {audit.disk['free_gb']} GB / {audit.disk['total_gb']} GB",
        "",
        "## Tooling",
        f"- Python: {audit.python_version} (`{audit.python_executable}`)",
        f"- pip: {audit.pip.get('stdout') or audit.pip.get('stderr')}",
        f"- Git: {audit.git.get('stdout') or audit.git.get('stderr')}",
        f"- NVIDIA SMI: {nvidia or 'not available'}",
        "",
        "## Python Packages",
    ]
    for name, version in audit.packages.items():
        lines.append(f"- {name}: {version if version is not None else 'not installed'}")

    lines.extend(
        [
            "",
            "## PyTorch CUDA",
            f"- PyTorch: {audit.pytorch.get('version', 'not installed')}",
            f"- CUDA available: {audit.pytorch.get('cuda_available', False)}",
            f"- CUDA version: {audit.pytorch.get('cuda_version', 'not available')}",
            f"- CUDA devices: {', '.join(audit.pytorch.get('devices', [])) or 'none'}",
            "",
            "## OpenCV",
            f"- OpenCV: {audit.opencv.get('version', 'not installed')}",
        ]
    )
    (artifacts_dir / "environment_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    """Write audit artifacts for the current environment."""
    root = Path(__file__).resolve().parents[1]
    audit = audit_environment(root)
    write_audit(audit, root / "artifacts")
    print(json.dumps(asdict(audit), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

