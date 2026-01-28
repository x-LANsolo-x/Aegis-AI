"""ASVspoof dataset downloader (free, self-hostable).

This script downloads ASVspoof dataset archives from official mirrors and
extracts them into a versioned directory:

  ml/datasets/asvspoof_<version>/

Notes:
- ASVspoof archives are large. Plan disk space accordingly.
- Mirrors/filenames occasionally change; if a URL breaks, update MIRRORS below.
- Checksum verification is performed when a SHA-256 is available in MIRRORS.

Usage:
  py -m ml.datasets.download_asvspoof --version 2019 --output_dir ml/datasets

"""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional
from urllib.error import URLError
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class Mirror:
    url: str
    filename: str
    sha256: Optional[str] = None


# Official mirrors (primarily University of Edinburgh DataShare).
# These links can change; keep them updated as needed.
#
# IMPORTANT: These are intentionally the *primary landing* download URLs used
# by the official hosts. If your environment requires a different mirror,
# add it here.
MIRRORS: dict[str, list[Mirror]] = {
    # ASVspoof 2019 (Logical Access / Physical Access)
    # Landing page: https://datashare.ed.ac.uk/handle/10283/3336
    # Files are large; some environments may block direct download.
    "2019": [
        Mirror(
            url="https://datashare.ed.ac.uk/bitstream/handle/10283/3336/LA.zip",
            filename="LA.zip",
            sha256=None,
        ),
        Mirror(
            url="https://datashare.ed.ac.uk/bitstream/handle/10283/3336/PA.zip",
            filename="PA.zip",
            sha256=None,
        ),
    ],
    # ASVspoof 2021 (DF)
    # Landing page: https://datashare.ed.ac.uk/handle/10283/3756 (may vary)
    "2021": [
        Mirror(
            url="https://datashare.ed.ac.uk/bitstream/handle/10283/3756/ASVspoof2021_DF.zip",
            filename="ASVspoof2021_DF.zip",
            sha256=None,
        ),
    ],
}


def _iter_chunks(stream, chunk_size: int = 1024 * 1024) -> Iterable[bytes]:
    while True:
        b = stream.read(chunk_size)
        if not b:
            break
        yield b


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in _iter_chunks(f):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path) -> None:
    """Download URL to dest (no external deps)."""

    dest.parent.mkdir(parents=True, exist_ok=True)

    req = Request(url, headers={"User-Agent": "AegisAI-ASVspoof-Downloader/1.0"})
    with urlopen(req) as resp:
        total = resp.headers.get("Content-Length")
        total_bytes = int(total) if total and total.isdigit() else None

        tmp = dest.with_suffix(dest.suffix + ".part")
        downloaded = 0

        with tmp.open("wb") as f:
            for chunk in _iter_chunks(resp):
                f.write(chunk)
                downloaded += len(chunk)
                if total_bytes:
                    pct = (downloaded / total_bytes) * 100
                    sys.stdout.write(f"\rDownloading {dest.name}: {pct:6.2f}%")
                    sys.stdout.flush()

        if total_bytes:
            sys.stdout.write("\n")

        tmp.replace(dest)


def extract_archive(archive_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    if zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path, "r") as z:
            z.extractall(out_dir)
        return

    # handle .tar, .tar.gz, .tgz
    if tarfile.is_tarfile(archive_path):
        with tarfile.open(archive_path, "r:*") as t:
            t.extractall(out_dir)
        return

    raise ValueError(f"Unsupported archive format: {archive_path}")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Download and extract ASVspoof datasets")
    parser.add_argument("--output_dir", required=True, help="Base output directory (e.g., ml/datasets)")
    parser.add_argument("--version", required=True, choices=sorted(MIRRORS.keys()), help="ASVspoof version")
    parser.add_argument(
        "--keep_archives",
        action="store_true",
        help="Keep downloaded archives after extraction (default: delete)",
    )
    parser.add_argument(
        "--background",
        action="store_true",
        help="Run download in background (detached subprocess); progress logged to <output_dir>/download.log",
    )

    args = parser.parse_args(argv)

    base_out = Path(args.output_dir)
    version = str(args.version)

    target_dir = base_out / f"asvspoof_{version}"
    target_dir.mkdir(parents=True, exist_ok=True)

    # If --background, relaunch in a detached subprocess with stdout/stderr redirected.
    if args.background:
        log_file = target_dir / "download.log"
        print(f"Launching download in background; logs: {log_file}")

        # Build command without --background to avoid recursion
        cmd = [
            sys.executable,
            "-m",
            "ml.datasets.download_asvspoof",
            "--output_dir",
            str(args.output_dir),
            "--version",
            version,
        ]
        if args.keep_archives:
            cmd.append("--keep_archives")

        with log_file.open("w", encoding="utf-8") as log:
            subprocess.Popen(
                cmd,
                stdout=log,
                stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0,
                start_new_session=(sys.platform != "win32"),
            )

        print(f"Background process started. Monitor: tail -f {log_file}")
        return 0

    mirrors = MIRRORS[version]
    for m in mirrors:
        archive_path = target_dir / m.filename

        if archive_path.exists():
            print(f"Found existing archive: {archive_path}")
        else:
            print(f"Downloading: {m.url}")
            try:
                download(m.url, archive_path)
            except URLError as e:
                raise SystemExit(f"Download failed for {m.url}: {e}")

        if m.sha256:
            print(f"Verifying SHA-256 for {archive_path.name}...")
            digest = sha256_file(archive_path)
            if digest.lower() != m.sha256.lower():
                raise SystemExit(
                    f"Checksum mismatch for {archive_path.name}: expected {m.sha256}, got {digest}"
                )
            print("Checksum OK")
        else:
            print(f"No checksum available for {archive_path.name}; skipping verification")

        print(f"Extracting {archive_path.name} -> {target_dir}")
        extract_archive(archive_path, target_dir)

        if not args.keep_archives:
            try:
                archive_path.unlink()
            except Exception:
                pass

    print(f"Done. Dataset extracted to: {target_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
