"""JSON metadata storage for Flask detection results."""

from __future__ import annotations

import json
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.common import ensure_dir, project_root


RESULT_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]{12,64}$")


def utc_timestamp() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


class ResultStore:
    """Store and retrieve result metadata as JSON files."""

    def __init__(self, metadata_directory: Path) -> None:
        self.metadata_directory = metadata_directory
        ensure_dir(self.metadata_directory)

    def new_result_id(self) -> str:
        """Generate a collision-resistant result ID."""
        while True:
            result_id = uuid.uuid4().hex
            if not self.path_for(result_id).exists():
                return result_id

    def validate_result_id(self, result_id: str) -> str:
        """Validate a result ID and prevent traversal via path-like input."""
        if not RESULT_ID_PATTERN.fullmatch(result_id or ""):
            raise ValueError("Invalid result ID.")
        return result_id

    def path_for(self, result_id: str) -> Path:
        """Return the metadata path for a result ID."""
        valid_id = self.validate_result_id(result_id)
        path = self.metadata_directory / f"{valid_id}.json"
        root = self.metadata_directory.resolve()
        resolved = path.resolve()
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise ValueError("Result metadata path escapes metadata directory.") from exc
        return path

    def write(self, result_id: str, metadata: dict[str, Any]) -> Path:
        """Atomically write metadata for a result ID."""
        path = self.path_for(result_id)
        ensure_dir(path.parent)
        payload = {**metadata, "result_id": result_id}
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(temporary, path)
        return path

    def read(self, result_id: str) -> dict[str, Any] | None:
        """Read metadata for a result ID, returning None when absent."""
        path = self.path_for(result_id)
        if not path.is_file():
            return None
        return json.loads(path.read_text(encoding="utf-8"))


def default_result_store() -> ResultStore:
    """Return the default app result store."""
    return ResultStore(project_root() / "outputs" / "flask" / "metadata")
