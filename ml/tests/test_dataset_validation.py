import json
from pathlib import Path

import pytest

from ml.datasets.validate_manifest import validate_manifest


@pytest.fixture
def fixtures_dir():
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def tmp_manifest(tmp_path, fixtures_dir):
    """Helper to create a temp manifest JSONL."""

    def _make(records):
        manifest = tmp_path / "test_manifest.jsonl"
        with manifest.open("w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")
        return manifest

    return _make


def test_no_missing_files(tmp_manifest, fixtures_dir):
    manifest = tmp_manifest(
        [
            {
                "path": str(fixtures_dir / "valid_normal.wav"),
                "label": "bonafide",
                "duration_sec": 3.0,
                "sample_rate": 16000,
                "split": "train",
            },
        ]
    )

    exit_code = validate_manifest(manifest)
    assert exit_code == 0


def test_sample_rate_check(tmp_manifest, fixtures_dir):
    # This test just ensures files with stated sample_rate are readable.
    # We don't enforce matching here; just readability.
    manifest = tmp_manifest(
        [
            {
                "path": str(fixtures_dir / "valid_normal.wav"),
                "label": "bonafide",
                "duration_sec": 3.0,
                "sample_rate": 16000,
                "split": "train",
            },
        ]
    )

    exit_code = validate_manifest(manifest)
    assert exit_code == 0


def test_duration_range_violations(tmp_manifest, fixtures_dir):
    # valid_toolong.wav is 16s > max default 15s
    manifest = tmp_manifest(
        [
            {
                "path": str(fixtures_dir / "valid_toolong.wav"),
                "label": "bonafide",
                "duration_sec": 16.0,
                "sample_rate": 16000,
                "split": "train",
            },
        ]
    )

    exit_code = validate_manifest(manifest, max_duration=15.0)
    assert exit_code == 1  # should have errors


def test_label_distribution_warning(tmp_manifest, fixtures_dir, capsys):
    # Only one label; should warn if <1% bonafide (not applicable here, but test logic)
    manifest = tmp_manifest(
        [
            {
                "path": str(fixtures_dir / "valid_normal.wav"),
                "label": "bonafide",
                "duration_sec": 3.0,
                "sample_rate": 16000,
                "split": "train",
            },
        ]
        * 100
    )

    exit_code = validate_manifest(manifest, min_label_fraction=0.01)
    # All bonafide => 100% => no warning
    assert exit_code == 0


def test_corrupt_file_unreadable(tmp_manifest, fixtures_dir):
    manifest = tmp_manifest(
        [
            {
                "path": str(fixtures_dir / "corrupt.wav"),
                "label": "spoof",
                "duration_sec": 1.0,
                "sample_rate": 16000,
                "split": "train",
            },
        ]
    )

    exit_code = validate_manifest(manifest)
    assert exit_code == 1  # should have errors for unreadable file
