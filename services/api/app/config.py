from __future__ import annotations

import os
from pathlib import Path
from typing import Set

def get_upload_dir() -> Path:
    """Get upload directory from env (default ./uploads).

    Returns a Path without creating it.
    """

    return Path(os.environ.get("UPLOAD_DIR", "./uploads"))


# Back-compat constant-like alias (avoid using this in new code).
UPLOAD_DIR: Path = get_upload_dir()

# Maximum allowed upload size (bytes). Default: 20 MB.
MAX_UPLOAD_BYTES: int = int(os.environ.get("MAX_UPLOAD_BYTES", 20 * 1024 * 1024))

# Allowed file extensions (lowercase, including leading dot).
ALLOWED_EXTENSIONS: Set[str] = {
    ".wav",
    ".mp3",
    ".m4a",
    ".ogg",
    ".opus",
    ".mp4",
    ".mov",
    ".png",
    ".jpg",
    ".jpeg",
}


def ensure_upload_dir_exists() -> Path:
    """Create the upload directory if it doesn't exist and return it."""
    upload_dir = get_upload_dir()
    upload_dir.mkdir(parents=True, exist_ok=True)
    return upload_dir


def is_allowed_extension(filename: str) -> bool:
    ext = Path(filename).suffix.lower()
    return ext in ALLOWED_EXTENSIONS
