# Aegis-AI

Aegis-AI is a privacy-first, **offline-capable**, agentic deepfake threat detection concept built for high-stakes environments (defense, law enforcement, emergency response, enterprise incident response). The goal is to provide **real-time authenticity verification** of incoming media (audio / video / images) and deliver **actionable forensic reporting**—even when connectivity is unavailable.

This repository contains two finalized hackathon deliverables:

1. **UI Prototype (Product Vision)** — a polished multi-page UI showcasing how the final product will look and feel.
2. **Localhost Model Demo (Mentoring Demo)** — a localhost web demo that accepts an uploaded audio file and generates a detailed forensic-style report.

---

## What’s Implemented

### 1) UI Prototype (Audio / Video / Image)
Location: `aegis-ui-demo/`

A fully navigable, professional UI prototype showing the core product workflow:

- **Mode Tabs:** Audio / Video / Image dashboards
- **Dashboards:** upload/record entry points + results preview area
- **New Analysis** flow
- **Batch Upload** flow
- **Generate Report** flow
- **Analysis History:** searchable list of previous detections
- **Reports Hub:** report cards with view/download actions
- **Detailed Result View:** rich breakdown layout
- **Full Forensic Report Page:** printable report-style view with:
  - file metadata
  - detection breakdown
  - visual placeholders (spectrogram/waveform)
  - chain-of-custody concept
  - recommended security actions

> This UI is designed to look like a real, production-ready security product.

### 2) Localhost Model Demo (Audio)
Location: `model-demo/`

A localhost demo that:

- Runs completely on-device (localhost)
- Accepts audio uploads
- Produces a **high-detail forensic report** including:
  - confidence verdict
  - multi-signal indicators
  - anomaly scorecards
  - ensemble-style breakdown (module contributions)
  - graphical report elements (waveform sparkline + spectrogram heatmap preview)
  - chain-of-custody timeline and recommended actions

This demo is intended for a fast mentoring/pitch environment: it is optimized for **clarity, repeatability, and strong presentation**.

---

## Quick Start

### Requirements
- Windows / macOS / Linux
- Python 3.x

### Run the Localhost Model Demo

```bash
py -m pip install gradio
py model-demo/aegis_demo.py
```

Then open:

- http://127.0.0.1:7860

### Open the UI Prototype

Open any of these directly in a browser:

- `aegis-ui-demo/index.html` (Audio dashboard)
- `aegis-ui-demo/video-dashboard.html` (Video dashboard)
- `aegis-ui-demo/image-dashboard.html` (Image dashboard)

---

## Product Architecture (Planned)

Aegis-AI is designed as an **offline-first edge product**:

- **On-device inference** for deepfake detection (audio/video/image)
- **Agentic explanation layer** that converts signals into actionable guidance
- **Forensic reporting + chain-of-custody** to support investigations
- **Privacy-first design**: no raw media leaves the device by default

### RunAnywhere SDK (Planned Integration)
We plan to integrate the **RunAnywhere SDK** to enable:

- efficient on-device model execution
- cross-platform deployment (mobile + edge)
- optimized inference performance for real-time workflows

---

## Roadmap (Next Development Phase)

Planned additions and upgrades:

- Replace demo inference pipeline with full ML models (audio/video/image)
- On-device optimization (quantization, ONNX/TFLite execution)
- Real-time recording ingestion (mic/camera)
- Model update strategy + federated learning option
- Improved forensic export (PDF generation, signing, verification)
- Role-based access workflows and audit log hardening
- Admin policy controls (thresholds, escalation rules, incident templates)

---

## License

To be added.

---

### Team
Built by **team-ZerOne** for SnowHack IPEC.
