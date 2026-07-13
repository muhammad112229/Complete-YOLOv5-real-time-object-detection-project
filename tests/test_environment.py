"""Tests for environment audit helpers."""

from __future__ import annotations

from pathlib import Path

from src.environment import audit_environment, write_audit


def test_environment_audit_writes_artifacts(tmp_path: Path) -> None:
    """Environment audit should produce JSON and Markdown files."""
    audit = audit_environment(Path.cwd())
    write_audit(audit, tmp_path)
    assert (tmp_path / "environment_audit.json").exists()
    assert (tmp_path / "environment_audit.md").exists()
    assert audit.python_version_info[0] >= 3

