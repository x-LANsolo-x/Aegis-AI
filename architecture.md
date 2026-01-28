# AegisAI-Edge — Architecture (100% Free to Implement & Deploy)

> Goal: a defense-grade, **offline-first** deepfake verification product that is hard to beat because it combines (1) fast on-device inference, (2) explainable, actionable decisioning, (3) tamper-evident audit/chain-of-custody, and (4) continuous improvement via **self-hosted federated learning** — all using **free/open-source** software and commodity hardware.

---

## 0) Architecture Principles

1. **Offline-first by default**
   - Every critical capability (scan → verdict → explanation → evidence log) works with **zero connectivity**.

2. **Zero-trust media handling**
   - Treat every incoming media artifact as hostile until verified.

3. **Privacy-first**
   - No raw media leaves the device unless an explicit policy allows it (default: never).

4. **Defense-grade auditability**
   - Every scan produces a tamper-evident, signed record (chain-of-custody).

5. **Model agility**
   - Modular detection pipeline (audio/video/image) with pluggable models and a stable interface.

6. **Resilience against attacker adaptation**
   - Multi-signal fusion + adversarial checks + uncertainty handling (don’t overclaim).

7. **100% free stack**
   - Use open-source runtimes + self-hosting (no paid SaaS required).

---

## 1) System Overview

AegisAI-Edge consists of:

### A. Edge App (Primary product)
Runs on phones / rugged devices / body-cam companions.

- On-device inference (TFLite / ONNX Runtime)
- Preprocessing (audio/video decoding, feature extraction)
- Explainability + recommended actions (policy-driven)
- Secure local storage (encrypted) for scan records
- Optional sync when online (federated learning + policy updates)

### B. Self-Hosted Control Plane (Optional but recommended)
Runs in an on-prem environment (HQ / secure network) and provides:

- Federated learning coordinator (Flower)
- Model registry + signing
- Policy distribution
- Audit aggregation (optional)
- Admin console (optional)

> You can deploy the control plane on a single machine with Docker Compose and keep everything air-gapped.

---

## 2) High-Level Logical Architecture

```text
+-------------------------+              (optional sync)               +------------------------------+
|        Edge App         |  <------------------------------------->  |     Self-Hosted Control      |
| (Android/iOS/Desktop)   |                                           |          Plane               |
|                         |                                           | (On-Prem / Air-gapped)       |
|  +-------------------+  |                                           |  +-------------------------+ |
|  | Inference Engine  |  |                                           |  | Federated Coordinator    | |
|  | (TFLite/ONNX)     |  |                                           |  | (Flower Server)          | |
|  +-------------------+  |                                           |  +-------------------------+ |
|  +-------------------+  |                                           |  +-------------------------+ |
|  | Explainability    |  |                                           |  | Model Registry + Signing | |
|  | + Action Planner  |  |                                           |  | (MinIO + metadata DB)    | |
|  +-------------------+  |                                           |  +-------------------------+ |
|  +-------------------+  |                                           |  +-------------------------+ |
|  | Evidence Vault    |  |                                           |  | Policy Service            | |
|  | (encrypted store) |  |                                           |  | (FastAPI)                 | |
|  +-------------------+  |                                           |  +-------------------------+ |
|  +-------------------+  |                                           |  +-------------------------+ |
|  | Secure Identity   |  |                                           |  | Audit Aggregator          | |
|  | (device/user keys)|  |                                           |  | (optional)                | |
|  +-------------------+  |                                           |  +-------------------------+ |
+-------------------------+                                           +------------------------------+
```

---

## 3) Edge App Architecture (Unbeatable Core)

### 3.1 Modules

#### 3.1.1 Media Ingestion Layer
- Accept inputs from:
  - local files
  - share sheet / intent (mobile)
  - direct capture (mic/camera)
  - body-cam companion import (future)

**Open-source building blocks**
- FFmpeg (decoding/transcoding; mobile uses platform media APIs where possible)
- WebRTC / platform recorder APIs (optional)

#### 3.1.2 Preprocessing & Feature Extraction
- Audio:
  - resample (16 kHz)
  - normalize loudness
  - compute mel-spectrogram, CQCC/LFCC (optional), prosody features
- Video:
  - sample frames (e.g., 5–10 fps)
  - detect faces + landmarks
  - stabilize/crop faces

**Open-source building blocks**
- librosa (server/dev), or custom DSP on device
- MediaPipe (face detection + landmarks)

#### 3.1.3 Inference Engine (Pluggable)
A stable interface that supports multiple models and runtimes.

**Runtime choices (free)**
- TensorFlow Lite (Android/iOS) for int8 quantized models
- ONNX Runtime Mobile (Android/iOS)

**Model families (suggested)**
- Audio deepfake:
  - lightweight CNN on log-mel
  - transformer encoder on features
  - optional Whisper-lite embeddings for robustness
- Video deepfake:
  - frame-level CNN + temporal aggregator (LSTM/TCN/Transformer)
  - face-landmark dynamics anomaly module

#### 3.1.4 Multi-Signal Fusion (Key differentiator)
Instead of a single score, compute multiple signals and fuse:

- Model score(s)
- Compression artifacts indicators
- Liveness / naturalness cues
- Landmark motion consistency
- Uncertainty estimates

Fusion strategies (free)
- weighted logistic regression / calibrated stacking
- rule-based guardrails (for obvious attacks)
- conformal prediction thresholds (optional)

#### 3.1.5 Explainability + Action Planner (Agentic Layer)
Produces:
- verdict (AUTHENTIC / SUSPICIOUS / DEEPFAKE)
- confidence + calibration note
- top contributing reasons
- recommended actions (policy-driven)

Implementation approach
- deterministic explanation templates mapped to signals
- policy rules decide actions based on:
  - verdict
  - confidence
  - mission context
  - user role

> This avoids dependence on paid LLM APIs. If you later add an on-device small LLM, it remains optional.

#### 3.1.6 Evidence Vault + Chain-of-Custody
Every analysis produces a **Verification Record**:
- record id
- timestamp
- media hash (SHA-256)
- verdict + confidence
- signals summary
- explanation text
- user/device identity metadata (minimal)
- tamper-evident signature

**Tamper evidence design (free)**
- store records in an append-only local log
- each record contains:
  - `prev_record_hash`
  - `record_hash`
  - signature by device key

This forms a local hash chain.

Local storage
- SQLite (encrypted)
  - Android: SQLCipher (open-source) or EncryptedFile + Room
  - Desktop: SQLCipher

Key storage
- Android Keystore / iOS Keychain / Secure Enclave

#### 3.1.7 Adversarial & Abuse Detection
Detect common abuse patterns:
- extreme compression / re-encoding loops
- replay attacks (speakerphone artifacts)
- truncated audio
- out-of-distribution embeddings

Behavior
- when uncertain: return **SUSPICIOUS** and force secondary verification.

---

## 4) Control Plane Architecture (Free + Self-Hosted)

### 4.1 Components

#### 4.1.1 Federated Learning Coordinator
- Flower Server (Python)
- Performs aggregation (FedAvg + robust variants)

Robust FL (to resist poisoning)
- Trimmed mean / median aggregation
- Krum / Multi-Krum (if needed)
- update norm clipping

#### 4.1.2 Model Registry + Signing
- Store model artifacts in MinIO (S3-compatible, open-source)
- Store metadata in PostgreSQL (open-source)
- Sign models with a private key (kept in control plane)

Device behavior
- only accepts model updates signed by HQ key

#### 4.1.3 Policy Service
- FastAPI service to deliver:
  - thresholds
  - escalation rules
  - UI/role policies
  - allowed sharing/export rules

Policies are versioned and signed.

#### 4.1.4 Audit Aggregation (Optional)
- Devices may upload only verification records (no raw media)
- Stored in PostgreSQL
- Export to PDF/JSON (free libraries)

#### 4.1.5 Admin UI (Optional)
- A simple web console (React/Vue/Svelte or server-rendered)
- Manage:
  - users/roles
  - policies
  - device enrollment
  - model versions

Identity (free)
- Keycloak (open-source) OR simple JWT auth initially

---

## 5) Deployment Topologies (No Paid Services)

### 5.1 "Pure Offline" Mode (Minimum viable deployment)
- No server required.
- Edge app ships with a baseline model.
- All analysis + logs are local.

### 5.2 "Occasionally Connected" Mode (Recommended)
- Edge app operates offline.
- When internet is available (HQ Wi-Fi / secure uplink):
  - uploads federated updates
  - downloads signed model + policy updates
  - uploads audit records (optional)

### 5.3 Fully Air-Gapped HQ
- Control plane deployed inside secure network.
- Devices sync only when physically docked or connected to HQ network.

---

## 6) Data Flows

### 6.1 Scan Flow (Offline)
1. Media selected/ingested
2. Preprocessing + feature extraction
3. Inference (audio/video)
4. Multi-signal fusion
5. Explainability + recommended actions
6. Evidence Vault writes:
   - media hash
   - record hash chain update
   - signature

### 6.2 Federated Learning Flow (When Online)
1. Device trains locally on permitted data
2. Device computes model delta
3. Device sends delta to Flower server (TLS)
4. Server aggregates + runs robust checks
5. Server produces a new model version
6. Model signed and published
7. Devices download and verify signature

---

## 7) Security Architecture

### 7.1 Threat Model (Minimum)
- Adversary can:
  - craft deepfakes to evade detection
  - attempt replay attacks
  - compromise network links
  - attempt to poison federated learning
  - attempt to tamper with logs

### 7.2 Controls
- **Secure boot + device attestation** (where available)
- **Encrypted local storage**
- **Key isolation** via Keystore/Keychain
- **Signed model updates**
- **Hash-chained records** for tamper evidence
- **RBAC** + optional MFA
- **Robust federated aggregation** + anomaly detection

---

## 8) Model Strategy (Making It "Unbeatable")

The key is not just one model — it’s a *system*.

### 8.1 Multi-Model Ensemble (Edge-feasible)
- Small model for fast screening
- Medium model for confirmatory pass (only when needed)
- Specialized detectors:
  - vocoder artifact detector
  - face landmark dynamics detector

### 8.2 Calibration + Uncertainty
- Temperature scaling / isotonic regression
- Output includes uncertainty notes
- Auto-switch to SUSPICIOUS when uncertain

### 8.3 Compression Robustness
- Train with heavy augmentation:
  - Opus/AAC re-encodes
  - WhatsApp-like resampling
  - background noise + channel distortion

### 8.4 Anti-Poisoning for Federated Learning
- robust aggregation
- client update validation
- per-device reputation scoring (optional)

---

## 9) Technology Choices (All Free)

### Edge
- **Android**: Kotlin + TFLite / ONNX Runtime Mobile + MediaPipe
- **iOS**: Swift + TFLite / ORT + Vision/MediaPipe
- **Desktop (optional)**: Python/C++ wrapper for ORT

### Control Plane
- FastAPI (Python)
- Flower (Python)
- PostgreSQL
- MinIO
- Nginx (reverse proxy)
- Docker + Docker Compose
- (Optional) Keycloak

### Observability (optional, free)
- Prometheus + Grafana
- Loki for logs

---

## 10) Repository Structure Recommendation

```text
/ (root)
  /apps
    /edge-mobile       # Android/iOS app
    /edge-desktop      # optional
    /admin-console     # optional
  /services
    /control-plane-api # FastAPI
    /federated-server  # Flower
  /ml
    /training
    /models
    /export            # tflite/onnx pipelines
    /evaluation
  /docs
    architecture.md
  /infra
    docker-compose.yml
    k8s/ (optional)
```

---

## 11) Implementation Phases

### Phase 0 — Hardening the Demo into a Real Core
- Define stable `VerificationRecord` schema
- Implement encrypted Evidence Vault + hash chain
- Build deterministic XAI + action policy engine

### Phase 1 — Audio MVP (Edge)
- TFLite/ONNX audio model
- compression-robust preprocessing
- <2s latency target

### Phase 2 — Video MVP (Edge)
- face tracking + landmark module (MediaPipe)
- temporal consistency scoring

### Phase 3 — Federated Learning + Model Signing
- Flower server + robust aggregation
- signed model distribution

### Phase 4 — Multi-Modal Fusion + Escalation
- fuse audio+video
- emergency escalation policies

---

## 12) Acceptance Criteria (Architecture-Level)

1. **Offline scan** completes end-to-end with no network.
2. Evidence Vault produces hash-chained, signed records.
3. Model updates are rejected if unsigned/invalid.
4. Federated learning server can run locally with Docker Compose.
5. Latency budget met on mid-range devices for typical clips.

---

## 13) Appendix: Minimal Docker Compose (Control Plane)

This is a reference target (not committed yet in repo):
- `fastapi` policy + model registry API
- `postgres`
- `minio`
- `flower-server`
- `nginx`

---

If you want, I can also:
1) generate the initial `/infra/docker-compose.yml` for the control plane,
2) define the exact `VerificationRecord` JSON schema,
3) propose the MVP model export pipeline (PyTorch → ONNX → ORT / TFLite).
