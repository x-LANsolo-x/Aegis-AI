from __future__ import annotations

from pathlib import Path


def media_type_from_filename(filename: str) -> str:
    """Infer media type from file extension.

    This is intentionally simple for now; expand as the product grows.
    """

    ext = Path(filename).suffix.lower()
    if ext in {".wav", ".mp3", ".m4a", ".ogg", ".opus", ".flac"}:
        return "audio"
    if ext in {".mp4", ".mov", ".avi", ".mkv", ".webm"}:
        return "video"
    if ext in {".png", ".jpg", ".jpeg"}:
        return "image"
    return "unknown"


# Video file extensions
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".ogg", ".opus", ".flac"}


def is_video_file(filename: str) -> bool:
    """Check if filename is a video file."""
    return Path(filename).suffix.lower() in VIDEO_EXTENSIONS


def is_audio_file(filename: str) -> bool:
    """Check if filename is an audio file."""
    return Path(filename).suffix.lower() in AUDIO_EXTENSIONS
