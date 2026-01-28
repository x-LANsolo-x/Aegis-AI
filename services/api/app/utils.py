from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO


_SAFE_CHARS_RE = re.compile(r"[^A-Za-z0-9._-]+")


@dataclass(frozen=True)
class StreamSaveResult:
    sha256: str
    saved_path: Path
    size_bytes: int


def stream_save_and_sha256(
    *,
    fileobj: BinaryIO,
    destination_path: Path,
    chunk_size: int = 1024 * 1024,
) -> StreamSaveResult:
    """Stream a file-like object to disk while computing SHA-256.

    - Reads in chunks (default 1MB)
    - Updates hashlib.sha256()
    - Writes chunks to destination_path
    - Returns sha256 hex digest, saved path, and total size

    Notes:
    - Caller is responsible for ensuring destination directory exists.
    - This function overwrites destination_path if it already exists.
    """

    h = hashlib.sha256()
    size = 0

    destination_path = Path(destination_path)

    with destination_path.open("wb") as f:
        while True:
            chunk = fileobj.read(chunk_size)
            if not chunk:
                break
            size += len(chunk)
            h.update(chunk)
            f.write(chunk)

    return StreamSaveResult(sha256=h.hexdigest(), saved_path=destination_path, size_bytes=size)


def sanitize_filename(filename: str | None) -> str:
    """Sanitize untrusted filenames for safe local storage.

    Rules:
    - Strip any directory traversal using os.path.basename.
    - Replace spaces/unsafe chars with '_'.
    - Ensure result is never empty (fallback to 'file.bin').

    This does not attempt to guarantee uniqueness; callers should prefix with
    UUIDs or hashes if needed.
    """

    if not filename:
        return "file.bin"

    # Remove any path components (../ etc.)
    name = os.path.basename(filename)

    # Normalize whitespace/unsafe chars
    name = name.strip()
    name = _SAFE_CHARS_RE.sub("_", name)

    # Collapse multiple underscores
    name = re.sub(r"_+", "_", name)

    # Remove underscores directly before a dot (e.g., "name_.wav" -> "name.wav")
    name = re.sub(r"_+\.", ".", name)

    # Avoid names that become empty or only dots/underscores
    name = name.strip("._ ")

    if not name:
        return "file.bin"

    return name
