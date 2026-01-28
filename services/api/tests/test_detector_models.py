import pytest

from services.api.app.detector import DummyDetector, DetectorInput, DetectorOutput


def test_detector_input_defaults():
    di = DetectorInput(file_path="/tmp/a.wav")
    assert di.metadata == {}


def test_detector_output_defaults_and_validation():
    out = DetectorOutput(verdict="SUSPICIOUS", confidence=0.5, model_version="v1")
    assert out.explanations == []
    assert out.signals == {}

    with pytest.raises(Exception):
        DetectorOutput(verdict="X", confidence=-0.1, model_version="v1")

    with pytest.raises(Exception):
        DetectorOutput(verdict="X", confidence=1.1, model_version="v1")


def test_dummy_detector_returns_fixed_output(monkeypatch):
    d = DummyDetector(sleep_seconds=0)
    out = d.run(DetectorInput(file_path="/tmp/a.wav"))
    assert out.verdict == "SAFE"
    assert out.confidence == 0.1
    assert out.model_version == "dummy-0.1"


def test_dummy_detector_sleep(monkeypatch):
    calls = {"slept": 0}

    def fake_sleep(sec):
        calls["slept"] = sec

    monkeypatch.setattr("services.api.app.detector.time.sleep", fake_sleep)

    d = DummyDetector(sleep_seconds=0.25)
    _ = d.run(DetectorInput(file_path="/tmp/a.wav"))
    assert calls["slept"] == 0.25
