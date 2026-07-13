"""Download and extract COCO 2017 object-detection archives.

The downloader is intentionally conservative: it performs a disk/network
preflight, downloads with resume support, verifies ZIP integrity, keeps archives,
and extracts into the project-standard raw layout.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import time
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.common import ensure_dir, project_root, setup_logging


LOGGER = logging.getLogger(__name__)

COCO_2017_ARCHIVES: dict[str, dict[str, Any]] = {
    "train2017": {
        "url": "http://images.cocodataset.org/zips/train2017.zip",
        "filename": "train2017.zip",
        "expected_size_bytes": 19_336_861_798,
        "expected_top_level": "train2017",
    },
    "val2017": {
        "url": "http://images.cocodataset.org/zips/val2017.zip",
        "filename": "val2017.zip",
        "expected_size_bytes": 815_585_330,
        "expected_top_level": "val2017",
    },
    "annotations_trainval2017": {
        "url": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
        "filename": "annotations_trainval2017.zip",
        "expected_size_bytes": 252_907_541,
        "expected_top_level": "annotations",
    },
}

MIN_FREE_SPACE_BYTES = 75 * 1024**3
CHUNK_SIZE = 8 * 1024 * 1024
SEGMENT_SIZE = 128 * 1024 * 1024
SEGMENTED_DOWNLOAD_THRESHOLD = 100 * 1024 * 1024
SEGMENTED_WORKERS = 16


@dataclass(frozen=True)
class EndpointCheck:
    """Official COCO endpoint metadata."""

    name: str
    url: str
    reachable: bool
    status: int | None
    content_length: int | None
    accept_ranges: str | None
    final_url: str | None
    error: str | None = None


class PartialDownload(RuntimeError):
    """Raised when a download stops cleanly with resumable partial data."""


def bytes_to_gb(size: int | float) -> float:
    """Convert bytes to GiB with two decimal places."""
    return round(float(size) / 1024**3, 2)


def check_endpoint(name: str, url: str, timeout: int = 30) -> EndpointCheck:
    """Check a COCO endpoint with an HTTP HEAD request."""
    request = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - official COCO URL
            content_length = response.headers.get("Content-Length")
            return EndpointCheck(
                name=name,
                url=url,
                reachable=200 <= response.status < 400,
                status=response.status,
                content_length=int(content_length) if content_length else None,
                accept_ranges=response.headers.get("Accept-Ranges"),
                final_url=response.geturl(),
            )
    except Exception as exc:
        return EndpointCheck(
            name=name,
            url=url,
            reachable=False,
            status=None,
            content_length=None,
            accept_ranges=None,
            final_url=None,
            error=str(exc),
        )


def pre_download_check(root: Path) -> dict[str, Any]:
    """Check disk space and official endpoint reachability."""
    root = root.resolve()
    artifacts_dir = ensure_dir(root / "artifacts")
    usage = shutil.disk_usage(root)
    archive_bytes = sum(int(item["expected_size_bytes"]) for item in COCO_2017_ARCHIVES.values())
    estimated_extracted_bytes = 24 * 1024**3
    estimated_processed_bytes = 2 * 1024**3
    estimated_required_bytes = archive_bytes + estimated_extracted_bytes + estimated_processed_bytes
    safe_required_bytes = max(MIN_FREE_SPACE_BYTES, estimated_required_bytes)
    endpoints = [
        check_endpoint(name, str(metadata["url"]))
        for name, metadata in COCO_2017_ARCHIVES.items()
    ]
    errors: list[str] = []
    if usage.free < safe_required_bytes:
        errors.append(
            f"Insufficient free disk space: {bytes_to_gb(usage.free)} GiB available, "
            f"{bytes_to_gb(safe_required_bytes)} GiB required with safety margin."
        )
    for endpoint in endpoints:
        if not endpoint.reachable:
            errors.append(f"Official COCO endpoint unreachable: {endpoint.name} ({endpoint.error})")
        expected = COCO_2017_ARCHIVES[endpoint.name]["expected_size_bytes"]
        if endpoint.content_length is not None and int(endpoint.content_length) != int(expected):
            errors.append(
                f"Unexpected Content-Length for {endpoint.name}: "
                f"{endpoint.content_length} != {expected}"
            )

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "workspace": str(root),
        "python_executable_requirement": ".venv\\Scripts\\python.exe",
        "disk": {
            "total_bytes": usage.total,
            "used_bytes": usage.used,
            "free_bytes": usage.free,
            "free_gib": bytes_to_gb(usage.free),
        },
        "estimates": {
            "archive_bytes": archive_bytes,
            "archive_gib": bytes_to_gb(archive_bytes),
            "estimated_extracted_bytes": estimated_extracted_bytes,
            "estimated_processed_bytes": estimated_processed_bytes,
            "safe_required_bytes": safe_required_bytes,
            "safe_required_gib": bytes_to_gb(safe_required_bytes),
        },
        "endpoints": [asdict(endpoint) for endpoint in endpoints],
        "errors": errors,
        "ready": not errors,
    }
    (artifacts_dir / "coco_pre_download_check.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    lines = [
        "# COCO Pre-Download Check",
        "",
        f"Generated UTC: `{report['generated_at_utc']}`",
        f"Workspace: `{root}`",
        f"Free disk: {report['disk']['free_gib']} GiB",
        f"Safe required disk: {report['estimates']['safe_required_gib']} GiB",
        f"Ready: {'PASS' if report['ready'] else 'FAIL'}",
        "",
        "## Official Endpoints",
    ]
    for endpoint in endpoints:
        lines.append(
            f"- {endpoint.name}: {'reachable' if endpoint.reachable else 'unreachable'}, "
            f"status={endpoint.status}, size={endpoint.content_length}, "
            f"accept_ranges={endpoint.accept_ranges}"
        )
    if errors:
        lines.extend(["", "## Errors"])
        lines.extend(f"- {error}" for error in errors)
    (artifacts_dir / "coco_pre_download_check.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    return report


def verify_zip(path: Path) -> None:
    """Validate that a ZIP archive is readable."""
    with zipfile.ZipFile(path) as archive:
        bad_file = archive.testzip()
    if bad_file:
        raise ValueError(f"Corrupt ZIP entry in {path}: {bad_file}")


def archive_is_verified(path: Path, expected_size: int) -> bool:
    """Return whether an archive exists, has the expected size, and passes ZIP validation."""
    if not path.exists() or path.stat().st_size != expected_size:
        return False
    marker = path.with_suffix(path.suffix + ".verified")
    if marker.exists():
        try:
            marker_data = json.loads(marker.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            marker_data = {}
        if marker_data.get("size_bytes") == expected_size and marker_data.get("zip_integrity") == "passed":
            return True
    try:
        verify_zip(path)
    except Exception:
        return False
    write_archive_verified_marker(path, expected_size, "zipfile.testzip")
    return True


def write_archive_verified_marker(path: Path, expected_size: int, method: str) -> None:
    """Write a local marker that an archive was verified."""
    marker = path.with_suffix(path.suffix + ".verified")
    marker.write_text(
        json.dumps(
            {
                "path": str(path),
                "size_bytes": expected_size,
                "zip_integrity": "passed",
                "method": method,
                "verified_at_utc": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def download_with_resume(
    url: str,
    destination: Path,
    expected_size: int,
    retries: int = 5,
    timeout: int = 60,
    max_seconds: int | None = None,
) -> dict[str, Any]:
    """Download a file with retry and resume support."""
    ensure_dir(destination.parent)
    if expected_size >= SEGMENTED_DOWNLOAD_THRESHOLD:
        return download_segmented(
            url,
            destination,
            expected_size,
            retries=retries,
            timeout=timeout,
            max_seconds=max_seconds,
        )
    part_path = destination.with_suffix(destination.suffix + ".part")
    if destination.exists() and destination.stat().st_size == expected_size:
        LOGGER.info("Archive already has expected size: %s", destination)
        return {"status": "skipped_existing", "path": str(destination), "size_bytes": expected_size}
    if destination.exists() and destination.stat().st_size < expected_size:
        LOGGER.warning("Moving partial final file to %s", part_path)
        destination.replace(part_path)
    if destination.exists() and destination.stat().st_size > expected_size:
        raise RuntimeError(f"Archive is larger than expected: {destination}")

    for attempt in range(1, retries + 1):
        existing = part_path.stat().st_size if part_path.exists() else 0
        headers = {}
        mode = "ab" if existing else "wb"
        if existing:
            headers["Range"] = f"bytes={existing}-"
            LOGGER.info("Resuming %s from byte %d", destination.name, existing)
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
                if existing and response.status != 206:
                    LOGGER.warning("Server did not honor resume; restarting %s", destination.name)
                    existing = 0
                    mode = "wb"
                downloaded = existing
                next_log_at = max(downloaded + 512 * 1024**2, int(expected_size * 0.01))
                started = time.perf_counter()
                with part_path.open(mode + "") as file:
                    while True:
                        chunk = response.read(CHUNK_SIZE)
                        if not chunk:
                            break
                        file.write(chunk)
                        downloaded += len(chunk)
                        if downloaded >= next_log_at or downloaded == expected_size:
                            elapsed = max(time.perf_counter() - started, 0.001)
                            rate = (downloaded - existing) / 1024**2 / elapsed
                            LOGGER.info(
                                "%s %.2f%% (%s/%s GiB, %.2f MiB/s)",
                                destination.name,
                                downloaded / expected_size * 100,
                                bytes_to_gb(downloaded),
                                bytes_to_gb(expected_size),
                                rate,
                            )
                            next_log_at = downloaded + 512 * 1024**2
            actual_size = part_path.stat().st_size
            if actual_size != expected_size:
                raise RuntimeError(
                    f"Incomplete download for {destination.name}: {actual_size} != {expected_size}"
                )
            part_path.replace(destination)
            return {"status": "downloaded", "path": str(destination), "size_bytes": expected_size}
        except (urllib.error.URLError, TimeoutError, RuntimeError) as exc:
            LOGGER.warning("Attempt %d/%d failed for %s: %s", attempt, retries, destination.name, exc)
            if attempt == retries:
                raise RuntimeError(
                    f"Failed to download {url}. Partial file kept at {part_path}. "
                    "Re-run the same command to resume."
                ) from exc
            time.sleep(min(2**attempt, 30))
    raise RuntimeError(f"Download loop failed unexpectedly for {url}")


def _segment_path(segment_dir: Path, index: int) -> Path:
    """Return the path for a segment file."""
    return segment_dir / f"segment_{index:05d}.part"


def _download_segment(
    url: str,
    segment_path: Path,
    index: int,
    start: int,
    end: int,
    timeout: int,
    retries: int,
    deadline: float | None = None,
) -> dict[str, Any]:
    """Download one byte-range segment with resume support."""
    expected_length = end - start + 1
    ensure_dir(segment_path.parent)
    if segment_path.exists() and segment_path.stat().st_size == expected_length:
        return {"index": index, "status": "skipped", "bytes": expected_length}
    if segment_path.exists() and segment_path.stat().st_size > expected_length:
        with segment_path.open("r+b") as file:
            file.truncate(expected_length)
        return {"index": index, "status": "truncated_complete", "bytes": expected_length}
    for attempt in range(1, retries + 1):
        if deadline is not None and time.perf_counter() >= deadline:
            return {"index": index, "status": "partial_time_budget", "bytes": segment_path.stat().st_size if segment_path.exists() else 0}
        existing = segment_path.stat().st_size if segment_path.exists() else 0
        range_start = start + existing
        headers = {
            "Range": f"bytes={range_start}-{end}",
            "User-Agent": "Mozilla/5.0 COCO downloader",
        }
        request = urllib.request.Request(url, headers=headers)
        mode = "ab" if existing else "wb"
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
                if response.status != 206:
                    raise RuntimeError(f"Range request not honored; status={response.status}")
                downloaded = existing
                with segment_path.open(mode) as file:
                    while True:
                        if deadline is not None and time.perf_counter() >= deadline:
                            return {"index": index, "status": "partial_time_budget", "bytes": downloaded}
                        chunk = response.read(CHUNK_SIZE)
                        if not chunk:
                            break
                        remaining = expected_length - downloaded
                        if remaining <= 0:
                            break
                        if len(chunk) > remaining:
                            chunk = chunk[:remaining]
                        file.write(chunk)
                        downloaded += len(chunk)
            actual = segment_path.stat().st_size
            if actual != expected_length:
                if deadline is not None and time.perf_counter() >= deadline:
                    return {"index": index, "status": "partial_time_budget", "bytes": actual}
                raise RuntimeError(f"segment incomplete: {actual} != {expected_length}")
            return {"index": index, "status": "downloaded", "bytes": actual}
        except Exception as exc:
            if attempt == retries:
                raise RuntimeError(f"segment {index} failed after {retries} attempts: {exc}") from exc
            time.sleep(min(2**attempt, 20))
    raise RuntimeError(f"segment {index} failed unexpectedly")


def _combine_segments(segment_dir: Path, destination: Path, expected_size: int, segment_count: int) -> None:
    """Combine complete segment files into the final archive atomically."""
    assemble_path = destination.with_suffix(destination.suffix + ".assembling")
    if assemble_path.exists():
        assemble_path.unlink()
    with assemble_path.open("wb") as output:
        for index in range(segment_count):
            segment_path = _segment_path(segment_dir, index)
            with segment_path.open("rb") as segment_file:
                shutil.copyfileobj(segment_file, output, length=CHUNK_SIZE)
    actual = assemble_path.stat().st_size
    if actual != expected_size:
        raise RuntimeError(f"Assembled archive has wrong size: {actual} != {expected_size}")
    assemble_path.replace(destination)


def download_segmented(
    url: str,
    destination: Path,
    expected_size: int,
    retries: int = 5,
    timeout: int = 60,
    workers: int = SEGMENTED_WORKERS,
    max_seconds: int | None = None,
) -> dict[str, Any]:
    """Download a large archive as resumable byte-range segments."""
    ensure_dir(destination.parent)
    if destination.exists() and destination.stat().st_size == expected_size:
        try:
            verify_zip(destination)
            return {"status": "skipped_existing", "path": str(destination), "size_bytes": expected_size}
        except Exception as exc:
            LOGGER.warning("Existing archive failed ZIP verification and will be reassembled: %s", exc)
            destination.unlink()
    if destination.exists() and destination.stat().st_size != expected_size:
        LOGGER.warning("Removing incomplete final archive before segmented resume: %s", destination)
        destination.unlink()
    segment_dir = destination.with_name(destination.name + ".segments")
    ensure_dir(segment_dir)
    segment_count = (expected_size + SEGMENT_SIZE - 1) // SEGMENT_SIZE
    ranges = [
        (index, index * SEGMENT_SIZE, min(expected_size - 1, (index + 1) * SEGMENT_SIZE - 1))
        for index in range(segment_count)
    ]
    legacy_part = destination.with_suffix(destination.suffix + ".part")
    if legacy_part.exists():
        migrate_legacy_partial_to_segments(legacy_part, segment_dir, ranges)
    LOGGER.info(
        "Segmented download for %s: %d segments, %d workers",
        destination.name,
        segment_count,
        workers,
    )
    deadline = time.perf_counter() + max_seconds if max_seconds else None
    while True:
        pending = []
        completed_bytes = 0
        for index, start, end in ranges:
            segment_path = _segment_path(segment_dir, index)
            expected_length = end - start + 1
            if segment_path.exists() and segment_path.stat().st_size > expected_length:
                with segment_path.open("r+b") as file:
                    file.truncate(expected_length)
            if segment_path.exists() and segment_path.stat().st_size == expected_length:
                completed_bytes += expected_length
            else:
                pending.append((index, start, end))
                completed_bytes += segment_path.stat().st_size if segment_path.exists() else 0
        LOGGER.info(
            "%s progress: %.2f%% (%s/%s GiB), pending segments=%d",
            destination.name,
            completed_bytes / expected_size * 100,
            bytes_to_gb(completed_bytes),
            bytes_to_gb(expected_size),
            len(pending),
        )
        if not pending:
            break
        if deadline is not None and time.perf_counter() >= deadline:
            raise PartialDownload(f"Time budget reached for {destination.name}; rerun to resume.")
        batch = pending[:workers]
        partial_seen = False
        with ThreadPoolExecutor(max_workers=min(workers, len(batch))) as executor:
            futures = [
                executor.submit(
                    _download_segment,
                    url,
                    _segment_path(segment_dir, index),
                    index,
                    start,
                    end,
                    timeout,
                    retries,
                    deadline,
                )
                for index, start, end in batch
            ]
            for future in as_completed(futures):
                result = future.result()
                if str(result["status"]).startswith("partial"):
                    partial_seen = True
        if partial_seen:
            raise PartialDownload(f"Time budget reached for {destination.name}; rerun to resume.")

    for index, start, end in ranges:
        segment_path = _segment_path(segment_dir, index)
        expected_length = end - start + 1
        if not segment_path.exists() or segment_path.stat().st_size != expected_length:
            raise RuntimeError(
                f"Segmented download incomplete for {destination.name}; rerun the same command to resume."
            )
    _combine_segments(segment_dir, destination, expected_size, segment_count)
    verify_zip(destination)
    shutil.rmtree(segment_dir)
    return {"status": "downloaded_segmented", "path": str(destination), "size_bytes": expected_size}


def migrate_legacy_partial_to_segments(
    legacy_part: Path,
    segment_dir: Path,
    ranges: list[tuple[int, int, int]],
) -> None:
    """Reuse bytes from an old sequential partial file in segmented downloads."""
    legacy_size = legacy_part.stat().st_size
    if legacy_size == 0:
        legacy_part.unlink()
        return
    LOGGER.info("Migrating legacy partial %s (%s GiB) into segments", legacy_part, bytes_to_gb(legacy_size))
    with legacy_part.open("rb") as source:
        for index, start, end in ranges:
            if start >= legacy_size:
                break
            expected_length = end - start + 1
            available = min(expected_length, legacy_size - start)
            if available <= 0:
                break
            segment_path = _segment_path(segment_dir, index)
            if segment_path.exists() and segment_path.stat().st_size >= available:
                continue
            source.seek(start)
            with segment_path.open("wb") as destination:
                remaining = available
                while remaining > 0:
                    chunk = source.read(min(CHUNK_SIZE, remaining))
                    if not chunk:
                        break
                    destination.write(chunk)
                    remaining -= len(chunk)


def extract_zip(archive_path: Path, output_dir: Path) -> dict[str, Any]:
    """Extract a ZIP archive into output_dir."""
    ensure_dir(output_dir)
    LOGGER.info("Extracting %s -> %s", archive_path, output_dir)
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(output_dir)
        names = archive.namelist()
    return {"archive": str(archive_path), "output_dir": str(output_dir), "members": len(names)}


def count_files(path: Path, pattern: str) -> int:
    """Count files matching a pattern under a directory."""
    if not path.exists():
        return 0
    return sum(1 for item in path.rglob(pattern) if item.is_file())


def download_coco2017(
    raw_dir: Path,
    confirm_large_download: bool,
    extract: bool = True,
    dry_run: bool = False,
    max_download_seconds: int | None = None,
) -> list[Path]:
    """Download and optionally extract COCO 2017 archives."""
    if not confirm_large_download:
        raise RuntimeError(
            "COCO 2017 download is large. Re-run with --confirm-large-download after approval."
        )
    root = project_root()
    precheck = pre_download_check(root)
    if not precheck["ready"]:
        raise RuntimeError("Pre-download check failed; see artifacts/coco_pre_download_check.md")

    archives_dir = ensure_dir(raw_dir / "archives")
    coco_root = ensure_dir(raw_dir / "coco2017")
    artifacts_dir = ensure_dir(root / "artifacts")
    manifest: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "archives_dir": str(archives_dir),
        "extraction_root": str(coco_root),
        "archives": {},
        "extractions": {},
        "errors": [],
    }
    downloaded: list[Path] = []
    for name, metadata in COCO_2017_ARCHIVES.items():
        archive_path = archives_dir / str(metadata["filename"])
        expected_size = int(metadata["expected_size_bytes"])
        try:
            if dry_run:
                LOGGER.info("Dry run: would download %s", metadata["url"])
                status = {"status": "dry_run", "path": str(archive_path), "size_bytes": 0}
            elif archive_is_verified(archive_path, expected_size):
                status = {"status": "skipped_verified", "path": str(archive_path), "size_bytes": archive_path.stat().st_size}
            else:
                status = download_with_resume(
                    str(metadata["url"]),
                    archive_path,
                    expected_size,
                    max_seconds=max_download_seconds,
                )
                verify_zip(archive_path)
                write_archive_verified_marker(archive_path, expected_size, "zipfile.testzip")
            manifest["archives"][name] = {
                **status,
                "url": metadata["url"],
                "expected_size_bytes": expected_size,
                "actual_size_bytes": archive_path.stat().st_size if archive_path.exists() else 0,
                "zip_integrity": "passed" if archive_path.exists() and not dry_run else "not_run",
            }
            if archive_path.exists():
                downloaded.append(archive_path)
            if extract and archive_path.exists() and not dry_run:
                top_level = coco_root / str(metadata["expected_top_level"])
                already_extracted = (
                    top_level.exists()
                    and (
                        count_files(top_level, "*.jpg") > 0
                        or (top_level / "instances_train2017.json").exists()
                    )
                )
                if already_extracted:
                    manifest["extractions"][name] = {
                        "status": "skipped_existing",
                        "target": str(top_level),
                    }
                else:
                    manifest["extractions"][name] = {
                        "status": "extracted",
                        **extract_zip(archive_path, coco_root),
                    }
        except PartialDownload as exc:
            segment_dir = archive_path.with_name(archive_path.name + ".segments")
            segment_bytes = 0
            if segment_dir.exists():
                segment_bytes = sum(path.stat().st_size for path in segment_dir.glob("segment_*.part"))
            manifest["archives"][name] = {
                "status": "partial",
                "url": metadata["url"],
                "path": str(archive_path),
                "segment_dir": str(segment_dir),
                "expected_size_bytes": expected_size,
                "partial_size_bytes": segment_bytes,
                "partial_percent": round(segment_bytes / expected_size * 100, 4),
                "resume_command": ".\\.venv\\Scripts\\python.exe -m src.download_coco --confirm-large-download --max-download-seconds 780",
                "error": str(exc),
            }
            (artifacts_dir / "coco_download_manifest.json").write_text(
                json.dumps(manifest, indent=2),
                encoding="utf-8",
            )
            raise RuntimeError(
                f"Partial download saved for {name}. "
                "Re-run: .\\.venv\\Scripts\\python.exe -m src.download_coco "
                "--confirm-large-download --max-download-seconds 780"
            ) from exc
        except Exception as exc:
            manifest["errors"].append({"archive": name, "error": str(exc)})
            (artifacts_dir / "coco_download_manifest.json").write_text(
                json.dumps(manifest, indent=2),
                encoding="utf-8",
            )
            raise

    (artifacts_dir / "coco_download_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    return downloaded


def main() -> int:
    """CLI entrypoint for COCO download."""
    parser = argparse.ArgumentParser(description="Download COCO 2017 archives.")
    parser.add_argument("--raw-dir", type=Path, default=project_root() / "data" / "raw")
    parser.add_argument("--confirm-large-download", action="store_true")
    parser.add_argument("--no-extract", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--max-download-seconds",
        type=int,
        default=None,
        help="Stop cleanly after this many seconds with resumable partial files.",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)
    archives = download_coco2017(
        raw_dir=args.raw_dir,
        confirm_large_download=args.confirm_large_download,
        extract=not args.no_extract,
        dry_run=args.dry_run,
        max_download_seconds=args.max_download_seconds,
    )
    LOGGER.info("Prepared %d archives.", len(archives))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
