"""Generate tiny synthetic WAV fixtures for ML dataset tests."""

import struct
import wave
from pathlib import Path


def generate_wav(path: Path, *, duration_sec: float, sample_rate: int = 16000) -> None:
    """Generate a tiny WAV file with silence (stdlib wave module)."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)  # mono
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        frames = int(duration_sec * sample_rate)
        # Write silence (all zeros)
        data = b"\x00\x00" * frames
        wf.writeframes(data)


def generate_corrupt_wav(path: Path) -> None:
    """Generate an invalid WAV file (truncated header)."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(b"RIFF")
        f.write(struct.pack("<I", 100))  # claim 100 bytes but don't write them


if __name__ == "__main__":
    fixtures_dir = Path(__file__).parent / "fixtures"

    # Valid files
    generate_wav(fixtures_dir / "valid_short.wav", duration_sec=0.6, sample_rate=16000)
    generate_wav(fixtures_dir / "valid_normal.wav", duration_sec=3.0, sample_rate=16000)
    generate_wav(fixtures_dir / "valid_long.wav", duration_sec=14.0, sample_rate=16000)
    generate_wav(fixtures_dir / "valid_toolong.wav", duration_sec=16.0, sample_rate=16000)

    # Corrupt
    generate_corrupt_wav(fixtures_dir / "corrupt.wav")

    print(f"Generated fixtures in {fixtures_dir}")
