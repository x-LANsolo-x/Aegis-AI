"""Build a manifest (JSONL) for ASVspoof datasets.

Scans the extracted ASVspoof directory, reads protocol files (train/dev/eval),
extracts audio metadata (duration, sample_rate), and outputs:

  ml/datasets/manifests/asvspoof_<version>.jsonl

Each line is a JSON record:
{
  "path": "ml/datasets/asvspoof_2019/LA/ASVspoof2019_LA_train/flac/T_1000137.flac",
  "label": "bonafide",
  "duration_sec": 3.21,
  "sample_rate": 16000,
  "split": "train"
}

Usage:
  py -m ml.datasets.build_manifest --version 2019 --base_dir ml/datasets

"""

from __future__ import annotations

import argparse
import json
import sys
import wave
from pathlib import Path
from typing import Iterator, Optional


def get_wav_metadata(path: Path) -> tuple[float, int]:
    """Return (duration_sec, sample_rate) using stdlib wave module.

    Raises if file is not a valid WAV or cannot be read.
    """

    with wave.open(str(path), "rb") as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
        duration = frames / float(rate) if rate else 0.0
        return duration, rate


def parse_protocol_2019_la(
    protocol_path: Path, split_name: str, base_audio_dir: Path
) -> Iterator[dict]:
    """Parse ASVspoof 2019 LA protocol file.

    Format (space-separated):
      SPEAKER_ID AUDIO_FILE_ID - SYSTEM_ID LABEL

    Example:
      LA_0079 LA_T_1138215 - - bonafide
      LA_0079 LA_T_1138216 - A07 spoof

    We only care about AUDIO_FILE_ID and LABEL.
    """

    if not protocol_path.exists():
        return

    with protocol_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            if len(parts) < 5:
                continue

            file_id = parts[1]
            label = parts[4].lower()

            # ASVspoof 2019 LA uses .flac
            audio_path = base_audio_dir / f"{file_id}.flac"

            if not audio_path.exists():
                # Try .wav fallback
                audio_path = base_audio_dir / f"{file_id}.wav"
                if not audio_path.exists():
                    sys.stderr.write(f"Warning: missing audio for {file_id}\n")
                    continue

            try:
                duration, sr = get_wav_metadata(audio_path)
            except Exception as e:
                sys.stderr.write(f"Warning: cannot read {audio_path}: {e}\n")
                continue

            yield {
                "path": str(audio_path),
                "label": label,
                "duration_sec": round(duration, 2),
                "sample_rate": sr,
                "split": split_name,
            }


def build_manifest_2019(base_dir: Path, out_path: Path) -> int:
    """Build manifest for ASVspoof 2019 LA."""

    asvspoof_dir = base_dir / "asvspoof_2019"
    la_dir = asvspoof_dir / "LA"

    if not la_dir.exists():
        raise SystemExit(f"ASVspoof 2019 LA directory not found: {la_dir}")

    # Protocol files
    protocol_dir = la_dir / "ASVspoof2019_LA_cm_protocols"
    splits = [
        ("train", protocol_dir / "ASVspoof2019.LA.cm.train.trn.txt", la_dir / "ASVspoof2019_LA_train" / "flac"),
        ("dev", protocol_dir / "ASVspoof2019.LA.cm.dev.trl.txt", la_dir / "ASVspoof2019_LA_dev" / "flac"),
        ("eval", protocol_dir / "ASVspoof2019.LA.cm.eval.trl.txt", la_dir / "ASVspoof2019_LA_eval" / "flac"),
    ]

    count = 0
    with out_path.open("w", encoding="utf-8") as out:
        for split_name, protocol_file, audio_dir in splits:
            print(f"Processing {split_name}...")
            for rec in parse_protocol_2019_la(protocol_file, split_name, audio_dir):
                out.write(json.dumps(rec) + "\n")
                count += 1

    print(f"Wrote {count} records to {out_path}")
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build ASVspoof manifest (JSONL)")
    parser.add_argument("--version", required=True, choices=["2019", "2021"], help="ASVspoof version")
    parser.add_argument("--base_dir", required=True, help="Base directory (e.g., ml/datasets)")

    args = parser.parse_args(argv)

    base = Path(args.base_dir)
    manifests_dir = base / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)

    out_path = manifests_dir / f"asvspoof_{args.version}.jsonl"

    if args.version == "2019":
        return build_manifest_2019(base, out_path)
    elif args.version == "2021":
        raise NotImplementedError("ASVspoof 2021 manifest builder not implemented yet")

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
