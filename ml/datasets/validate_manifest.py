"""Validate an ASVspoof manifest (JSONL).

Checks:
- File exists
- Audio readable
- Duration within reasonable range (0.5s – 15s by default)
- Label distribution sanity (bonafide vs spoof >= 1%)

Prints summary + error count.

Usage:
  py -m ml.datasets.validate_manifest --manifest ml/datasets/manifests/asvspoof_2019.jsonl

"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

from ml.datasets.loader import load_audio, read_manifest


def validate_manifest(
    manifest_path: Path,
    *,
    min_duration: float = 0.5,
    max_duration: float = 15.0,
    min_label_fraction: float = 0.01,
) -> int:
    """Validate manifest and return error count."""

    if not manifest_path.exists():
        sys.stderr.write(f"Error: manifest not found: {manifest_path}\n")
        return 1

    errors = 0
    total = 0
    missing_files = 0
    unreadable_files = 0
    duration_violations = 0

    label_counts = Counter()
    split_counts = Counter()

    print(f"Validating manifest: {manifest_path}")
    print(f"Duration range: [{min_duration}, {max_duration}] sec")
    print("-" * 80)

    for sample in read_manifest(manifest_path):
        total += 1
        label_counts[sample.label] += 1
        split_counts[sample.split] += 1

        # Check file exists
        if not Path(sample.path).exists():
            missing_files += 1
            errors += 1
            if errors <= 10:  # limit spam
                sys.stderr.write(f"Missing: {sample.path}\n")
            continue

        # Check audio readable
        try:
            waveform, sr = load_audio(sample.path)
        except Exception as e:
            unreadable_files += 1
            errors += 1
            if errors <= 10:
                sys.stderr.write(f"Unreadable: {sample.path} ({e})\n")
            continue

        # Check duration range
        if not (min_duration <= sample.duration_sec <= max_duration):
            duration_violations += 1
            errors += 1
            if errors <= 10:
                sys.stderr.write(
                    f"Duration violation: {sample.path} ({sample.duration_sec:.2f}s)\n"
                )

    print("-" * 80)
    print(f"Total records: {total}")
    print(f"Missing files: {missing_files}")
    print(f"Unreadable files: {unreadable_files}")
    print(f"Duration violations: {duration_violations}")
    print(f"Total errors: {errors}")
    print()

    # Label distribution
    print("Label distribution:")
    for label, count in sorted(label_counts.items()):
        pct = (count / total * 100) if total else 0
        print(f"  {label:12s}: {count:8d} ({pct:6.2f}%)")

        if pct < min_label_fraction * 100:
            sys.stderr.write(
                f"Warning: label '{label}' is <{min_label_fraction*100:.1f}% of dataset\n"
            )
    print()

    # Split distribution
    print("Split distribution:")
    for split, count in sorted(split_counts.items()):
        pct = (count / total * 100) if total else 0
        print(f"  {split:12s}: {count:8d} ({pct:6.2f}%)")
    print()

    if errors == 0:
        print("Manifest is valid.")
        return 0
    else:
        print(f"Manifest has {errors} errors.")
        return 1


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Validate ASVspoof manifest")
    parser.add_argument("--manifest", required=True, help="Path to manifest JSONL")
    parser.add_argument("--min_duration", type=float, default=0.5, help="Min duration (sec)")
    parser.add_argument("--max_duration", type=float, default=15.0, help="Max duration (sec)")
    parser.add_argument(
        "--min_label_fraction",
        type=float,
        default=0.01,
        help="Min fraction per label (warn if below)",
    )

    args = parser.parse_args(argv)

    return validate_manifest(
        Path(args.manifest),
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        min_label_fraction=args.min_label_fraction,
    )


if __name__ == "__main__":
    raise SystemExit(main())
