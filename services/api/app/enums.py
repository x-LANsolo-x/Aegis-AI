from __future__ import annotations

from enum import Enum


class Verdict(str, Enum):
    PENDING = "PENDING"
    AUTHENTIC = "AUTHENTIC"
    SUSPICIOUS = "SUSPICIOUS"
    DEEPFAKE = "DEEPFAKE"
    FAILED = "FAILED"
