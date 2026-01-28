import hashlib
from io import BytesIO
from pathlib import Path

from services.api.app.utils import stream_save_and_sha256


def test_stream_save_and_sha256_roundtrip(tmp_path: Path):
    payload = b"hello" * 100_000  # 500kB
    expected = hashlib.sha256(payload).hexdigest()

    dest = tmp_path / "out.bin"

    res = stream_save_and_sha256(fileobj=BytesIO(payload), destination_path=dest, chunk_size=1024 * 64)

    assert res.sha256 == expected
    assert res.saved_path == dest
    assert res.size_bytes == len(payload)
    assert dest.exists()
    assert dest.read_bytes() == payload


def test_stream_save_and_sha256_empty_file(tmp_path: Path):
    payload = b""
    expected = hashlib.sha256(payload).hexdigest()

    dest = tmp_path / "empty.bin"
    res = stream_save_and_sha256(fileobj=BytesIO(payload), destination_path=dest)

    assert res.sha256 == expected
    assert res.size_bytes == 0
    assert dest.exists()
    assert dest.read_bytes() == b""
