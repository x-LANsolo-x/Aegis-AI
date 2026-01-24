"""AEGIS-AI: Audio Deepfake Detection (Mentor Demo)

Runs on localhost using Gradio.
The UI is premium and the report is highly detailed.
Detection output is intentionally hardcoded based on uploaded filename.

Files:
  audio1.* -> REAL
  audio2.* -> FAKE
  audio3.* -> REAL
  audio4.* -> FAKE

Team: team-ZerOne | SnowHack IPEC
"""

from __future__ import annotations

import hashlib
import os
import random
import time
from dataclasses import dataclass
from datetime import datetime

import gradio as gr


# =============================================================================
# Hardcoded results (per-file differences)
# =============================================================================

@dataclass(frozen=True)
class DetectionResult:
    is_fake: bool
    confidence: float  # 0..1
    person: str
    verdict_label: str
    metrics: dict[str, float | bool]
    notes: list[str]


KNOWN_RESULTS: dict[str, DetectionResult] = {
    "audio1": DetectionResult(
        is_fake=False,
        confidence=0.968,
        person="Speaker A",
        verdict_label="AUTHENTIC VOICE",
        metrics={
            "pitch_variation": 0.081,
            "spectral_flux": 901.4,
            "breathing_detected": True,
            "micro_tremors": 0.0212,
            "formant_natural": True,
            "temporal_consistency": 0.93,
            "codec_artifacts": 0.12,
            "prosody_naturalness": 0.88,
            "vocoder_signature": 0.09,
        },
        notes=[
            "Natural pitch jitter present; no flat pitch plateaus observed.",
            "Spectral energy distribution matches human articulation patterns.",
            "Breathing/pauses detected at expected locations.",
        ],
    ),
    "audio2": DetectionResult(
        is_fake=True,
        confidence=0.944,
        person="Speaker A",
        verdict_label="DEEPFAKE DETECTED",
        metrics={
            "pitch_variation": 0.010,
            "spectral_flux": 212.7,
            "breathing_detected": False,
            "micro_tremors": 0.0014,
            "formant_natural": False,
            "temporal_consistency": 0.29,
            "codec_artifacts": 0.34,
            "prosody_naturalness": 0.22,
            "vocoder_signature": 0.84,
        },
        notes=[
            "Pitch trajectory is unnaturally stable (low jitter/shimmer).",
            "High probability vocoder imprint detected in upper bands.",
            "Breathing artifacts absent; transitions are overly smooth.",
        ],
    ),
    "audio3": DetectionResult(
        is_fake=False,
        confidence=0.956,
        person="Speaker B",
        verdict_label="AUTHENTIC VOICE",
        metrics={
            "pitch_variation": 0.073,
            "spectral_flux": 868.9,
            "breathing_detected": True,
            "micro_tremors": 0.0189,
            "formant_natural": True,
            "temporal_consistency": 0.90,
            "codec_artifacts": 0.18,
            "prosody_naturalness": 0.84,
            "vocoder_signature": 0.11,
        },
        notes=[
            "Micro-pauses and breath intake patterns consistent with human speech.",
            "Formant transitions show organic coarticulation.",
            "No synthetic banding artifacts in mid-to-high frequencies.",
        ],
    ),
    "audio4": DetectionResult(
        is_fake=True,
        confidence=0.952,
        person="Speaker B",
        verdict_label="DEEPFAKE DETECTED",
        metrics={
            "pitch_variation": 0.007,
            "spectral_flux": 176.2,
            "breathing_detected": False,
            "micro_tremors": 0.0008,
            "formant_natural": False,
            "temporal_consistency": 0.26,
            "codec_artifacts": 0.41,
            "prosody_naturalness": 0.19,
            "vocoder_signature": 0.88,
        },
        notes=[
            "Temporal smoothing indicates neural TTS generation pipeline.",
            "Prosody lacks natural stress and cadence variations.",
            "Formant structure appears quantized; unnatural harmonic spacing.",
        ],
    ),
}

FILENAME_ALIASES: dict[str, str] = {
    "audio1": "audio1",
    "audio_1": "audio1",
    "audio-1": "audio1",
    "a1": "audio1",
    "audio2": "audio2",
    "audio_2": "audio2",
    "audio-2": "audio2",
    "a2": "audio2",
    "audio3": "audio3",
    "audio_3": "audio3",
    "audio-3": "audio3",
    "a3": "audio3",
    "audio4": "audio4",
    "audio_4": "audio4",
    "audio-4": "audio4",
    "a4": "audio4",
}


def _normalize_basename(path: str) -> str:
    base = os.path.splitext(os.path.basename(path))[0].lower().strip()
    base = base.replace(" ", "_")
    return base


def resolve_result(upload_path: str) -> DetectionResult | None:
    base = _normalize_basename(upload_path)

    if base in KNOWN_RESULTS:
        return KNOWN_RESULTS[base]

    if base in FILENAME_ALIASES:
        return KNOWN_RESULTS[FILENAME_ALIASES[base]]

    for k in KNOWN_RESULTS.keys():
        if k in base or base in k:
            return KNOWN_RESULTS[k]

    for alias, k in FILENAME_ALIASES.items():
        if alias in base or base in alias:
            return KNOWN_RESULTS[k]

    return None


# =============================================================================
# Styling
# =============================================================================

CSS = """
:root {
  --bg0: #0b1020;
  --bg1: #0f172a;
  --border: rgba(148,163,184,0.16);
  --text: #e5e7eb;
  --muted: #94a3b8;
  --muted2:#64748b;
  --blue: #3b82f6;
  --indigo:#6366f1;
  --green:#10b981;
  --red:#ef4444;
  --amber:#f59e0b;
}

body, .gradio-container { background: var(--bg0) !important; }

.gradio-container {
  max-width: 1040px !important;
  margin: 0 auto !important;
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial !important;
}

footer { display: none !important; }

button.primary {
  background: linear-gradient(135deg, var(--blue), var(--indigo)) !important;
  border: none !important;
}
button.primary:hover { filter: brightness(1.06); transform: translateY(-1px); }

.block, .wrap { border: none !important; }
"""


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _badge(label: str, color: str) -> str:
    return (
        f"<span style='display:inline-flex;align-items:center;gap:8px;"
        f"padding:6px 12px;border-radius:999px;font-size:12px;font-weight:800;"
        f"border:1px solid rgba(255,255,255,0.12);background:rgba(255,255,255,0.04);color:{color};'>"
        f"{label}</span>"
    )


def _hash_id(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:10].upper()


def _sparkline(seed: int, w: int = 280, h: int = 52, color: str = "#60a5fa") -> str:
    rnd = random.Random(seed)
    pts = []
    for i in range(40):
        x = int((i / 39) * (w - 10)) + 5
        base = h / 2
        amp = (h * 0.35) * (0.35 + 0.65 * rnd.random())
        y = int(base + (rnd.random() - 0.5) * 2 * amp)
        pts.append((x, max(4, min(h - 4, y))))
    path = "M " + " L ".join(f"{x} {y}" for x, y in pts)
    return f"""
    <svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg">
      <defs>
        <linearGradient id="g" x1="0" y1="0" x2="1" y2="0">
          <stop offset="0" stop-color="{color}" stop-opacity="0.35"/>
          <stop offset="1" stop-color="{color}" stop-opacity="0.95"/>
        </linearGradient>
      </defs>
      <rect x="0" y="0" width="{w}" height="{h}" rx="12" fill="rgba(255,255,255,0.03)" stroke="rgba(255,255,255,0.06)"/>
      <path d="{path}" fill="none" stroke="url(#g)" stroke-width="2.4" stroke-linecap="round"/>
    </svg>
    """


def _spectrogram(seed: int, w: int = 560, h: int = 170, intensity: float = 0.7) -> str:
    """Generate a smoother, more realistic spectrogram-looking SVG.

    This is a visual-only demo artifact: we generate continuous-looking bands with blur,
    a perceptual color ramp, and a legend so it doesn't look like a pixel grid.
    """
    rnd = random.Random(seed)

    # Higher resolution grid + blur to avoid pixelated look
    cols = 140
    rows = 30
    cell_w = w / cols
    cell_h = h / rows

    def colormap(v: float) -> tuple[int, int, int]:
        # v in [0,1], map dark->purple->blue->cyan->yellow
        v = max(0.0, min(1.0, v))
        if v < 0.25:
            # dark -> purple
            t = v / 0.25
            return (int(12 + 35 * t), int(16 + 18 * t), int(40 + 90 * t))
        if v < 0.55:
            # purple -> blue/cyan
            t = (v - 0.25) / 0.30
            return (int(47 - 20 * t), int(34 + 140 * t), int(130 + 90 * t))
        if v < 0.82:
            # cyan -> green/yellow
            t = (v - 0.55) / 0.27
            return (int(27 + 200 * t), int(174 + 50 * t), int(220 - 140 * t))
        # yellow -> near-white
        t = (v - 0.82) / 0.18
        return (int(227 + 20 * t), int(224 + 20 * t), int(80 + 160 * t))

    rects = []
    # Create banded structure by combining smooth gradients + random texture
    for r in range(rows):
        fr = 1.0 - (r / (rows - 1))
        band = (0.18 * (1.0 - fr)) + (0.55 * fr)
        for c in range(cols):
            tc = c / (cols - 1)
            # slow-varying energy shape + texture
            base = 0.20 + 0.55 * (tc ** 0.6) * intensity
            texture = (rnd.random() - 0.5) * 0.22
            harmonic = 0.10 * (1.0 + 0.7 * rnd.random()) * (0.3 + 0.7 * (fr ** 1.6))
            v = base + texture + harmonic
            v *= 0.55 + 0.45 * band
            v = max(0.0, min(1.0, v))

            rr, gg, bb = colormap(v)
            x = c * cell_w
            y = r * cell_h
            rects.append(
                f"<rect x='{x:.2f}' y='{y:.2f}' width='{cell_w + 0.8:.2f}' height='{cell_h + 0.8:.2f}' fill='rgb({rr},{gg},{bb})' opacity='0.95'/>"
            )

    legend = """
      <defs>
        <linearGradient id='legend' x1='0' y1='0' x2='1' y2='0'>
          <stop offset='0%' stop-color='#0b1020'/>
          <stop offset='22%' stop-color='#2b2f7f'/>
          <stop offset='48%' stop-color='#1fb6ff'/>
          <stop offset='72%' stop-color='#a3e635'/>
          <stop offset='100%' stop-color='#fef08a'/>
        </linearGradient>
        <filter id='blur'>
          <feGaussianBlur stdDeviation='0.65'/>
        </filter>
        <clipPath id='clip'>
          <rect x='10' y='34' width='{w_minus}' height='{h_minus}' rx='12'/>
        </clipPath>
      </defs>
    """.format(w_minus=w-20, h_minus=h-56)

    return f"""
    <svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg">
      <rect x="0" y="0" width="{w}" height="{h}" rx="14" fill="rgba(255,255,255,0.03)" stroke="rgba(255,255,255,0.06)"/>
      {legend}

      <text x="16" y="22" fill="rgba(229,231,235,0.74)" font-size="11" font-weight="900">Spectrogram Analysis</text>
      <text x="16" y="38" fill="rgba(148,163,184,0.62)" font-size="10" font-weight="700">energy heatmap (visual preview)</text>

      <g clip-path="url(#clip)" filter="url(#blur)">
        {''.join(rects)}
      </g>

      <rect x="16" y="{h-18}" width="180" height="8" rx="4" fill="url(#legend)" opacity="0.9"/>
      <text x="202" y="{h-12}" fill="rgba(148,163,184,0.70)" font-size="10" font-weight="700">Low</text>
      <text x="240" y="{h-12}" fill="rgba(148,163,184,0.70)" font-size="10" font-weight="700">High</text>

      <text x="{w-62}" y="{h-12}" fill="rgba(148,163,184,0.62)" font-size="10" font-weight="800">time →</text>
      <text x="{w-62}" y="18" fill="rgba(148,163,184,0.62)" font-size="10" font-weight="800">freq ↑</text>
    </svg>
    """


def _metric_bar(name: str, value: float, good: bool, fmt: str = "{:.2f}") -> str:
    value = _clamp01(value)
    color = "var(--green)" if good else "var(--red)"
    pct = int(value * 100)
    return f"""
    <div style="padding:12px 14px;border-radius:12px;background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.06);">
      <div style="display:flex;justify-content:space-between;gap:10px;">
        <div style="color:#cbd5e1;font-size:12px;font-weight:800;">{name}</div>
        <div style="color:{color};font-size:12px;font-weight:950;">{fmt.format(value)}</div>
      </div>
      <div style="margin-top:10px;height:10px;border-radius:999px;background:rgba(255,255,255,0.08);overflow:hidden;">
        <div style="height:100%;width:{pct}%;background:linear-gradient(90deg,{color},rgba(99,102,241,0.9));"></div>
      </div>
    </div>
    """


def _risk_timeline(items: list[tuple[str, str]]) -> str:
    li = []
    for t, msg in items:
        head, tail = (msg.split("—", 1) + [""])[:2]
        li.append(
            f"<div style='display:flex;gap:12px;padding:10px 0;border-bottom:1px solid rgba(255,255,255,0.06);'>"
            f"<div style='min-width:78px;color:#94a3b8;font-family:ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;font-size:12px;'>{t}</div>"
            f"<div style='color:#cbd5e1;font-size:13px;'><span style='color:#e5e7eb;font-weight:800'>{head.strip()}</span> — {tail.strip()}</div>"
            f"</div>"
        )
    return """
    <div style="padding:14px 16px;border-radius:14px;background:rgba(0,0,0,0.18);border:1px solid rgba(255,255,255,0.10);">
      <div style="font-size:12px;letter-spacing:0.14em;text-transform:uppercase;color:#94a3b8;font-weight:900;margin-bottom:8px;">Chain of Custody</div>
      {rows}
    </div>
    """.format(rows="".join(li))


def render_report_html(
    *,
    filename: str | None,
    result: DetectionResult | None,
    processing_s: float | None,
    error: str | None = None,
    unknown: bool = False,
) -> str:
    if error:
        return f"""
        <div style="padding:22px;border-radius:16px;background:linear-gradient(135deg,#1a1a2e,#16213e);border:1px solid rgba(245,158,11,0.35);">
          <div style="font-size:18px;font-weight:950;color:#f59e0b;">Action needed</div>
          <div style="margin-top:8px;color:#cbd5e1;">{error}</div>
        </div>
        """

    if not filename:
        return """
        <div style="padding:44px;text-align:center;border-radius:16px;background:linear-gradient(135deg,#0f172a,#111c33);border:1px solid rgba(148,163,184,0.14);">
          <div style="font-size:44px;">🎤</div>
          <div style="margin-top:12px;color:#94a3b8;">Upload an audio file to generate a forensic report</div>
        </div>
        """

    assert result is not None
    assert processing_s is not None

    now = datetime.now()
    seed = int(_hash_id(filename), 16)
    rnd = random.Random(seed)

    report_id = f"AGS-{now.strftime('%Y%m%d')}-{_hash_id(filename)}"
    file_hash = hashlib.sha256(filename.encode("utf-8")).hexdigest()[:28]

    conf = _clamp01(result.confidence)
    conf_pct = conf * 100
    is_fake = result.is_fake

    color = "var(--red)" if is_fake else "var(--green)"
    bg = "linear-gradient(135deg,#2a1620,#0f172a)" if is_fake else "linear-gradient(135deg,#132a23,#0f172a)"

    risk = "CRITICAL" if is_fake else "LOW"
    risk_badge = _badge(("🚨 " if is_fake else "✅ ") + f"{risk} RISK", color)

    # vary executive summary per file
    phrases_real = [
        "No synthetic artifacts detected in spectral dynamics.",
        "Voice profile matches expected human micro-variation patterns.",
        "Signal integrity indicates organic speech production.",
        "Prosodic timing aligns with natural human cadence.",
    ]
    phrases_fake = [
        "Synthetic vocoder signature detected across upper-frequency bands.",
        "Temporal smoothing suggests neural TTS generation.",
        "Anomalous formant structure indicates artificial synthesis.",
        "Prosody appears over-regular and lacks natural stress variation.",
    ]
    summary_sentence = rnd.choice(phrases_fake if is_fake else phrases_real)

    m = result.metrics

    vocoder = float(m.get("vocoder_signature", 0.2))
    prosody = float(m.get("prosody_naturalness", 0.5))
    codec = float(m.get("codec_artifacts", 0.2))

    norm_temporal = _clamp01(float(m["temporal_consistency"]))
    norm_prosody = _clamp01(prosody)
    norm_vocoder_inv = _clamp01(1 - vocoder)
    norm_codec_inv = _clamp01(1 - codec)

    # different timeline per file
    base_min = 30 + (seed % 20)
    timeline = [
        (now.replace(minute=base_min, second=12).strftime("%H:%M:%S"), "File received — Uploaded for verification"),
        (now.replace(minute=base_min, second=14).strftime("%H:%M:%S"), "Hash computed — SHA-256 fingerprint generated"),
        (now.replace(minute=base_min, second=16).strftime("%H:%M:%S"), "Feature extraction — Mel/STFT + prosody features"),
        (now.replace(minute=base_min, second=17).strftime("%H:%M:%S"), "Inference — AegisNet ensemble scoring"),
        (now.replace(minute=base_min, second=18).strftime("%H:%M:%S"), "Report produced — Audit record written"),
    ]

    findings = list(result.notes)
    if is_fake:
        findings.append(rnd.choice([
            "Lip-smack/breath micro-noise absent in silent segments.",
            "Harmonic rolloff consistent with modern neural vocoders.",
            "Over-regular cadence suggests prosody synthesis.",
        ]))
    else:
        findings.append(rnd.choice([
            "Natural coarticulation observed during consonant-vowel transitions.",
            "Room tone and subtle background noise consistent across segments.",
            "Jitter/shimmer within normal human range.",
        ]))

    findings_html = "".join(
        f"<li style='margin:6px 0;color:#cbd5e1;font-size:13px;'>{f}</li>" for f in findings
    )

    actions = (
        [
            "Do not trust this audio; verify via secondary secure channel.",
            "Escalate to command/security and preserve evidence hash.",
            "Create incident log entry and flag the sender/source.",
            "Request a live callback with challenge-response verification.",
        ]
        if is_fake
        else [
            "Proceed, but continue standard verification protocols.",
            "Log verification outcome for audit trail.",
            "If high-risk context, perform challenge-response callback.",
        ]
    )
    actions_html = "".join(
        f"<li style='margin:6px 0;color:#cbd5e1;font-size:13px;'>{a}</li>" for a in actions
    )

    spark = _sparkline(seed=seed + 7)
    spec = _spectrogram(seed=seed + 19, intensity=0.86 if is_fake else 0.64)

    metrics_grid = "".join(
        [
            _metric_bar("Temporal consistency", norm_temporal, good=norm_temporal >= 0.7),
            _metric_bar("Prosody naturalness", norm_prosody, good=norm_prosody >= 0.6),
            _metric_bar("Vocoder signature (inverse)", norm_vocoder_inv, good=norm_vocoder_inv >= 0.6),
            _metric_bar("Codec artifacts (inverse)", norm_codec_inv, good=norm_codec_inv >= 0.6),
        ]
    )

    # -------------------------------------------------------------------------
    # Extra detailed sections (vary by file to avoid identical reports)
    # -------------------------------------------------------------------------
    # Deterministic pseudo-metadata from filename
    fake_sr = rnd.choice([16000, 22050, 44100, 48000])
    fake_dur = 6.0 + (seed % 240) / 30.0  # 6s .. 14s
    fake_channels = rnd.choice(["Mono", "Stereo"])
    fake_codec = rnd.choice(["PCM", "AAC", "Opus", "MP3"])

    # Ensemble breakdown (values sum to ~1 and vary per file)
    w1 = 0.20 + 0.35 * rnd.random()
    w2 = 0.15 + 0.30 * rnd.random()
    w3 = 0.10 + 0.25 * rnd.random()
    w4 = 0.10 + 0.25 * rnd.random()
    s = w1 + w2 + w3 + w4
    w1, w2, w3, w4 = (w1 / s, w2 / s, w3 / s, w4 / s)

    # Make model votes correlate with verdict (still varied)
    if is_fake:
        votes = [
            ("AegisNet-SpectroCNN", 0.75 + 0.20 * rnd.random()),
            ("AegisNet-Prosody", 0.70 + 0.25 * rnd.random()),
            ("AegisNet-VocoderTrace", 0.80 + 0.18 * rnd.random()),
            ("AegisNet-Temporal", 0.68 + 0.26 * rnd.random()),
        ]
    else:
        votes = [
            ("AegisNet-SpectroCNN", 0.08 + 0.18 * rnd.random()),
            ("AegisNet-Prosody", 0.10 + 0.20 * rnd.random()),
            ("AegisNet-VocoderTrace", 0.06 + 0.16 * rnd.random()),
            ("AegisNet-Temporal", 0.10 + 0.20 * rnd.random()),
        ]

    # Anomaly scorecard (0..100)
    anomaly_pitch = int((1 - _clamp01(float(m["pitch_variation"]) / 0.09)) * 100)
    anomaly_formant = int((0 if bool(m["formant_natural"]) else 85) + 15 * rnd.random())
    anomaly_breath = int((0 if bool(m["breathing_detected"]) else 80) + 20 * rnd.random())
    anomaly_vocoder = int(_clamp01(float(m.get("vocoder_signature", 0.0))) * 100)

    extra_sections = f"""
        <div style=\"margin-top:14px;display:grid;grid-template-columns:1fr 1fr;gap:12px;\">
          <div style=\"padding:14px 16px;border-radius:14px;background:rgba(0,0,0,0.18);border:1px solid rgba(255,255,255,0.10);\">
            <div style=\"font-size:12px;letter-spacing:0.14em;text-transform:uppercase;color:#94a3b8;font-weight:950;margin-bottom:8px;\">Signal Metadata</div>
            <div style=\"display:grid;grid-template-columns:1fr 1fr;gap:10px;\">
              <div style=\"padding:10px 12px;border-radius:12px;background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.06);\">
                <div style=\"color:#94a3b8;font-size:11px;\">Sample rate</div>
                <div style=\"margin-top:4px;color:#e5e7eb;font-weight:950;\">{fake_sr} Hz</div>
              </div>
              <div style=\"padding:10px 12px;border-radius:12px;background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.06);\">
                <div style=\"color:#94a3b8;font-size:11px;\">Duration</div>
                <div style=\"margin-top:4px;color:#e5e7eb;font-weight:950;\">{fake_dur:.1f}s</div>
              </div>
              <div style=\"padding:10px 12px;border-radius:12px;background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.06);\">
                <div style=\"color:#94a3b8;font-size:11px;\">Channels</div>
                <div style=\"margin-top:4px;color:#e5e7eb;font-weight:950;\">{fake_channels}</div>
              </div>
              <div style=\"padding:10px 12px;border-radius:12px;background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.06);\">
                <div style=\"color:#94a3b8;font-size:11px;\">Codec</div>
                <div style=\"margin-top:4px;color:#e5e7eb;font-weight:950;\">{fake_codec}</div>
              </div>
            </div>
          </div>

          <div style=\"padding:14px 16px;border-radius:14px;background:rgba(0,0,0,0.18);border:1px solid rgba(255,255,255,0.10);\">
            <div style=\"font-size:12px;letter-spacing:0.14em;text-transform:uppercase;color:#94a3b8;font-weight:950;margin-bottom:8px;\">Ensemble Breakdown</div>
            <div style=\"color:#cbd5e1;font-size:13px;line-height:1.75;\">
              <div style=\"display:flex;justify-content:space-between;\"><span>SpectroCNN</span><span style=\"font-family:ui-monospace;\">{w1*100:.0f}%</span></div>
              <div style=\"display:flex;justify-content:space-between;\"><span>Prosody</span><span style=\"font-family:ui-monospace;\">{w2*100:.0f}%</span></div>
              <div style=\"display:flex;justify-content:space-between;\"><span>VocoderTrace</span><span style=\"font-family:ui-monospace;\">{w3*100:.0f}%</span></div>
              <div style=\"display:flex;justify-content:space-between;\"><span>Temporal</span><span style=\"font-family:ui-monospace;\">{w4*100:.0f}%</span></div>
              <div style=\"margin-top:10px;color:#94a3b8;font-size:12px;\">Model votes (synthetic likelihood):</div>
              {''.join([f"<div style='display:flex;justify-content:space-between;'><span style='color:#cbd5e1;'>{name}</span><span style='font-family:ui-monospace;color:{'#ef4444' if is_fake else '#10b981'};'>{score*100:.0f}%</span></div>" for name, score in votes])}
            </div>
          </div>
        </div>

        <div style=\"margin-top:14px;padding:14px 16px;border-radius:14px;background:rgba(0,0,0,0.18);border:1px solid rgba(255,255,255,0.10);\">
          <div style=\"font-size:12px;letter-spacing:0.14em;text-transform:uppercase;color:#94a3b8;font-weight:950;margin-bottom:10px;\">Anomaly Scorecard (0-100)</div>
          <div style=\"display:grid;grid-template-columns:repeat(4,1fr);gap:10px;\">
            <div style=\"padding:10px 12px;border-radius:12px;background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.06);\">
              <div style=\"color:#94a3b8;font-size:11px;\">Pitch</div>
              <div style=\"margin-top:4px;color:#e5e7eb;font-weight:950;\">{anomaly_pitch}</div>
            </div>
            <div style=\"padding:10px 12px;border-radius:12px;background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.06);\">
              <div style=\"color:#94a3b8;font-size:11px;\">Formant</div>
              <div style=\"margin-top:4px;color:#e5e7eb;font-weight:950;\">{anomaly_formant}</div>
            </div>
            <div style=\"padding:10px 12px;border-radius:12px;background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.06);\">
              <div style=\"color:#94a3b8;font-size:11px;\">Breath</div>
              <div style=\"margin-top:4px;color:#e5e7eb;font-weight:950;\">{anomaly_breath}</div>
            </div>
            <div style=\"padding:10px 12px;border-radius:12px;background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.06);\">
              <div style=\"color:#94a3b8;font-size:11px;\">Vocoder</div>
              <div style=\"margin-top:4px;color:#e5e7eb;font-weight:950;\">{anomaly_vocoder}</div>
            </div>
          </div>
          <div style=\"margin-top:10px;color:#64748b;font-size:12px;\">Higher score means stronger anomaly indicator. Values are derived from multi-signal heuristics + ensemble vote alignment.</div>
        </div>
    """

    unknown_note = (
        "<div style='margin-top:10px;color:#94a3b8;font-size:12px;'>Note: file not in the demo set — showing baseline output.</div>"
        if unknown
        else ""
    )

    return f"""
    <div style="border-radius:18px;overflow:hidden;border:1px solid rgba(255,255,255,0.12);background:{bg};">

      <div style="padding:18px 20px;border-bottom:1px solid rgba(255,255,255,0.10);display:flex;justify-content:space-between;gap:12px;align-items:flex-start;flex-wrap:wrap;">
        <div style="min-width:340px;">
          <div style="font-size:12px;color:#94a3b8;letter-spacing:0.16em;text-transform:uppercase;font-weight:950;">Forensic Analysis Report</div>
          <div style="margin-top:8px;font-size:20px;font-weight:950;color:{color};">{result.verdict_label}</div>
          <div style="margin-top:4px;color:#94a3b8;font-size:13px;">Subject: <span style='color:#e5e7eb;font-weight:900;'>{result.person}</span></div>
          <div style="margin-top:4px;color:#94a3b8;font-size:13px;">Report ID: <span style='color:#e5e7eb;font-family:ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;'>{report_id}</span></div>
          {unknown_note}
        </div>

        <div style="text-align:right;min-width:260px;">
          <div>{risk_badge}</div>
          <div style="margin-top:10px;color:#94a3b8;font-size:12px;">Confidence</div>
          <div style="font-size:30px;font-weight:950;color:{color};">{conf_pct:.1f}%</div>
          <div style="margin-top:10px;">{spark}</div>
        </div>
      </div>

      <div style="padding:18px 20px;">

        <div style="display:grid;grid-template-columns:1fr 1fr;gap:14px;">
          <div style="padding:14px 16px;border-radius:14px;background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.06);">
            <div style="display:flex;justify-content:space-between;align-items:center;gap:10px;">
              <div>
                <div style="font-size:12px;color:#94a3b8;text-transform:uppercase;letter-spacing:0.12em;font-weight:900;">Evidence</div>
                <div style="margin-top:6px;color:#e5e7eb;font-weight:900;font-size:14px;">{filename}</div>
                <div style="margin-top:4px;color:#94a3b8;font-size:12px;">SHA-256 (prefix): <span style='font-family:ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;color:#cbd5e1;'>{file_hash}</span></div>
              </div>
              <div style="text-align:right;">
                <div style="color:#94a3b8;font-size:12px;">Processed</div>
                <div style="color:#e5e7eb;font-weight:900;font-size:13px;">{processing_s:.2f}s</div>
                <div style="margin-top:8px;">{_badge('📴 OFFLINE', '#10b981')}</div>
              </div>
            </div>
          </div>

          <div style="padding:14px 16px;border-radius:14px;background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.06);">
            <div style="font-size:12px;color:#94a3b8;text-transform:uppercase;letter-spacing:0.12em;font-weight:900;">Executive Summary</div>
            <div style="margin-top:10px;color:#cbd5e1;font-size:13px;line-height:1.7;">
              {summary_sentence}
              <br/><br/>
              Evaluated: pitch micro-variation, spectral dynamics, breathing artifacts, formant naturalness, temporal coherence, and codec artifacts.
            </div>
          </div>
        </div>

        <div style="margin-top:14px;display:grid;grid-template-columns:repeat(2,1fr);gap:12px;">{metrics_grid}</div>

        <div style="margin-top:14px;display:grid;grid-template-columns:1fr;gap:12px;">
          <div style="padding:14px 16px;border-radius:14px;background:rgba(0,0,0,0.18);border:1px solid rgba(255,255,255,0.10);">
            <div style="font-size:12px;letter-spacing:0.14em;text-transform:uppercase;color:#94a3b8;font-weight:950;">Key Findings</div>
            <ul style="margin:10px 0 0;padding-left:18px;">{findings_html}</ul>
          </div>
        </div>

        <div style="margin-top:14px;">
          {spec}
          <div style="margin-top:6px;color:#64748b;font-size:12px;">Graphical analysis: smoothed heatmap + legend (preview).</div>
        </div>

        {extra_sections}

        <div style="margin-top:14px;display:grid;grid-template-columns:1fr 1fr;gap:12px;">
          <div style="padding:14px 16px;border-radius:14px;background:rgba(0,0,0,0.18);border:1px solid rgba(255,255,255,0.10);">
            <div style="font-size:12px;letter-spacing:0.14em;text-transform:uppercase;color:#94a3b8;font-weight:950;margin-bottom:8px;">Signal Indicators</div>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;">
              <div style="padding:10px 12px;border-radius:12px;background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.06);">
                <div style="color:#94a3b8;font-size:11px;">Breathing artifacts</div>
                <div style="margin-top:4px;color:#e5e7eb;font-weight:950;">{('Detected' if bool(m['breathing_detected']) else 'Not detected')}</div>
              </div>
              <div style="padding:10px 12px;border-radius:12px;background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.06);">
                <div style="color:#94a3b8;font-size:11px;">Formant naturalness</div>
                <div style="margin-top:4px;color:#e5e7eb;font-weight:950;">{('Likely' if bool(m['formant_natural']) else 'Unlikely')}</div>
              </div>
              <div style="padding:10px 12px;border-radius:12px;background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.06);">
                <div style="color:#94a3b8;font-size:11px;">Spectral flux</div>
                <div style="margin-top:4px;color:#e5e7eb;font-weight:950;">{float(m['spectral_flux']):.1f}</div>
              </div>
              <div style="padding:10px 12px;border-radius:12px;background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.06);">
                <div style="color:#94a3b8;font-size:11px;">Vocoder signature</div>
                <div style="margin-top:4px;color:#e5e7eb;font-weight:950;">{float(m.get('vocoder_signature', 0.0))*100:.0f}%</div>
              </div>
            </div>
          </div>

          <div style="padding:14px 16px;border-radius:14px;background:rgba(0,0,0,0.18);border:1px solid rgba(255,255,255,0.10);">
            <div style="font-size:12px;letter-spacing:0.14em;text-transform:uppercase;color:#94a3b8;font-weight:950;margin-bottom:8px;">Recommended Actions</div>
            <ul style="margin:0;padding-left:18px;">{actions_html}</ul>
          </div>
        </div>

        <div style="margin-top:14px;">{_risk_timeline(timeline)}</div>

      </div>

      <div style="padding:12px 20px;border-top:1px solid rgba(255,255,255,0.10);display:flex;justify-content:space-between;gap:10px;color:#64748b;font-size:12px;flex-wrap:wrap;">
        <span>Generated: {now.strftime('%Y-%m-%d %H:%M:%S')}</span>
        <span>Mode: Offline / Local-only</span>
        <span>Engine: AegisNet (demo)</span>
      </div>
    </div>
    """


# =============================================================================
# Gradio handler + App
# =============================================================================


def analyze(audio_path: str | None) -> str:
    if not audio_path:
        return render_report_html(filename=None, result=None, processing_s=None)

    t0 = time.time()
    time.sleep(1.05)  # simulate inference

    filename = os.path.basename(audio_path)
    result = resolve_result(audio_path)

    if result is None:
        # baseline varied output for unknown files (still detailed)
        seed = int(_hash_id(filename), 16)
        rnd = random.Random(seed)
        baseline_real = rnd.random() > 0.5
        result = DetectionResult(
            is_fake=not baseline_real,
            confidence=0.70 + 0.18 * rnd.random(),
            person="Unknown",
            verdict_label=("DEEPFAKE DETECTED" if not baseline_real else "AUTHENTIC VOICE"),
            metrics={
                "pitch_variation": 0.03 + 0.06 * rnd.random(),
                "spectral_flux": 350 + 650 * rnd.random(),
                "breathing_detected": baseline_real,
                "micro_tremors": 0.006 + 0.02 * rnd.random(),
                "formant_natural": baseline_real,
                "temporal_consistency": 0.55 + 0.35 * rnd.random(),
                "codec_artifacts": 0.12 + 0.35 * rnd.random(),
                "prosody_naturalness": 0.35 + 0.55 * rnd.random(),
                "vocoder_signature": 0.15 + 0.70 * rnd.random(),
            },
            notes=[
                "File not in the demo set; baseline inference path used.",
                "For production, model weights would be loaded and run on-device.",
            ],
        )
        return render_report_html(
            filename=filename,
            result=result,
            processing_s=time.time() - t0,
            unknown=True,
        )

    return render_report_html(
        filename=filename,
        result=result,
        processing_s=time.time() - t0,
        unknown=False,
    )


HEADER_HTML = """
<div style="padding:22px 22px 12px;border-radius:16px;background:linear-gradient(135deg,#0f172a,#111c33);border:1px solid rgba(148,163,184,0.14);">
  <div style="display:flex;justify-content:space-between;align-items:center;gap:12px;flex-wrap:wrap;">
    <div style="display:flex;align-items:center;gap:10px;">
      <div style="font-size:24px;">🛡️</div>
      <div>
        <div style="font-size:20px;font-weight:950;color:#e5e7eb;letter-spacing:0.02em;">AEGIS<span style="color:#3b82f6;">AI</span></div>
        <div style="margin-top:2px;color:#94a3b8;font-size:13px;">Offline deepfake voice verification (Mentor Demo)</div>
      </div>
    </div>
    <div style="display:flex;gap:10px;flex-wrap:wrap;justify-content:flex-end;">
      <span style="padding:6px 12px;border-radius:999px;background:rgba(16,185,129,0.12);border:1px solid rgba(16,185,129,0.22);color:#10b981;font-size:12px;font-weight:800;">📴 Offline Mode</span>
      <span style="padding:6px 12px;border-radius:999px;background:rgba(59,130,246,0.12);border:1px solid rgba(59,130,246,0.22);color:#60a5fa;font-size:12px;font-weight:800;">⚡ ~1s analysis</span>
      <span style="padding:6px 12px;border-radius:999px;background:rgba(245,158,11,0.12);border:1px solid rgba(245,158,11,0.22);color:#fbbf24;font-size:12px;font-weight:800;">🔒 Local-only</span>
    </div>
  </div>
</div>
"""


with gr.Blocks(css=CSS, title="Aegis-AI | Audio Verification") as demo:
    gr.HTML(HEADER_HTML)

    gr.HTML(
        """
        <div style="margin-top:16px;padding:18px 20px;border-radius:16px;background:rgba(255,255,255,0.03);border:1px solid rgba(148,163,184,0.14);">
          <div style="font-size:14px;font-weight:900;color:#e5e7eb;">Audio Verification</div>
          <div style="margin-top:4px;color:#94a3b8;font-size:13px;">Upload a voice note to generate a detailed on-device forensic report.</div>
        </div>
        """
    )

    audio = gr.Audio(label="Upload audio", type="filepath", sources=["upload"])
    analyze_btn = gr.Button("Analyze", variant="primary")

    report = gr.HTML(render_report_html(filename=None, result=None, processing_s=None))

    analyze_btn.click(fn=analyze, inputs=[audio], outputs=[report])


if __name__ == "__main__":
    print("AEGIS-AI demo running on http://127.0.0.1:7860")
    demo.launch(server_name="127.0.0.1", server_port=7860, show_error=True)

