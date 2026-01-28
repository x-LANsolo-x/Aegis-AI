# Backend + AI/ML Development Guide (Intensive Testing After Every Step)

This guide is tailored to your current priority:
- Build **backend** + **AI/ML** first
- Delay frontend/web UI until later
- Use a **100% free** tech stack (open source + self-host)
- Apply **intensive testing after every step** so the system is production-grade from the start

---

## A) Target End State (Backend + ML Only)

### What you will have

1. A **backend service** (FastAPI) that:
   - accepts audio/video uploads
   - runs inference using a local ML runtime (PyTorch/ONNX/TFLite later)
   - stores analysis results + audit logs in a DB
   - generates forensic “reports” (JSON first; PDF later)
   - is structured to support RBAC + MFA later

2. An **ML pipeline** that:
   - trains an audio deepfake classifier (start audio-first)
   - evaluates with reproducible metrics
   - exports model artifacts (PyTorch + ONNX)
   - supports explainability outputs (signal-based + calibration)
   - supports regression testing against “golden” samples

---

# Part 1 — Backend Development Guide (FastAPI)

## Step B1: Create backend skeleton + health checks

**Goal:** A running service with predictable structure.

### Implement

- Create folders:
  - `services/api/`
  - `services/api/app/`
  - `services/api/tests/`

- Add FastAPI app with:
  - `GET /health` → returns `{ "status": "ok" }`
  - `GET /version` → returns a git hash or static version string

### Tests (must pass before moving on)

- Unit test:
  - call `/health`, assert 200 and JSON payload
- Run:
  - `pytest -q`
- Add “contract test” using `httpx` TestClient to ensure response schema stays stable.

**Intensive testing additions**
- Negative test: call unknown endpoint → 404
- Load test smoke: 100 sequential `/health` calls to ensure no leaks/crashes

---

## Step B2: Define core data model (AnalysisRecord) + DB layer

**Goal:** You can persist results reliably.

### Implement

Choose free DB stack:
- Dev: SQLite
- Prod: PostgreSQL (free)

Create tables (via SQLAlchemy or SQLModel):

- `analysis_records`
  - `id` (UUID)
  - `created_at`
  - `media_type` (audio/video)
  - `filename`
  - `sha256`
  - `duration_s` (nullable)
  - `verdict` (AUTHENTIC/SUSPICIOUS/DEEPFAKE)
  - `confidence` (0–1)
  - `explanations` (JSON array)
  - `signals` (JSON dict)
  - `model_version`
  - `processing_ms`

- `audit_events` (optional now, but recommended)
  - `id`, `timestamp`, `event_type`, `analysis_id`, `details`

### Tests

- Migration test: create schema cleanly on empty DB.
- Repository tests:
  - insert analysis record, fetch it, verify fields.

**Intensive testing**
- DB concurrency test: 20 inserts in parallel (threadpool) and ensure no corruption.
- Constraint tests:
  - confidence must be within [0,1]
  - verdict must be a valid enum

---

## Step B3: Implement upload endpoint (no ML yet) + hashing

**Goal:** Accept file, compute hash, store metadata, return record id.

### Implement

- `POST /v1/analysis/upload`
  - multipart file upload
  - stream to disk (or object store later)
  - compute SHA-256 while streaming
  - create `AnalysisRecord` with verdict = `SUSPICIOUS` and confidence = `0.0` initially
  - return `{analysis_id, sha256}`

### Tests

- Upload test with a fixture file:
  - assert `analysis_id` is valid UUID
  - assert `sha256` equals expected
  - assert DB row exists

- Negative tests:
  - missing file field
  - unsupported extension
  - huge file limit check (set small limit in tests)

- Security tests:
  - filename traversal attempt `../../evil.wav` must be sanitized

**Intensive testing**
- Fuzz test filenames and content-type headers.
- Repeat same upload twice → identical sha, new `analysis_id`.

---

## Step B4: Introduce “Detector Interface” abstraction (ML plug-in)

**Goal:** Backend depends on a stable interface, not on any specific model.

### Implement

Define:
- `DetectorInput` (file path + metadata)
- `DetectorOutput`:
  - verdict
  - confidence
  - explanations (list[str])
  - signals (dict)
  - model_version

Implement a stub `DummyDetector` initially.

Endpoint:
- `POST /v1/analysis/{id}/run`
  - loads uploaded file path from DB
  - calls detector
  - updates DB record with detector output

### Tests

- Use dependency injection to swap detector with dummy in tests.
- Test that `/run` updates the record correctly.
- Test id not found → 404.

**Intensive testing**
- Detector timeout simulation:
  - dummy sleeps; API must enforce timeout and set a failure state.

---

## Step B5: Add report retrieval endpoints

**Goal:** Backend provides analysis results cleanly.

### Implement

- `GET /v1/analysis/{id}` → returns AnalysisRecord JSON
- `GET /v1/analysis?limit=&offset=` → list
- `GET /v1/analysis/{id}/report` → “forensic report” JSON (structured)

Report JSON should include:
- evidence metadata
- verdict/confidence
- key findings (explanations)
- recommended actions (policy-driven later)
- chain-of-custody stub fields (created_at, device_id nullable)

### Tests

- Contract tests on JSON schema for report
- Pagination tests
- Snapshot tests: stable report shape

**Intensive testing**
- Backward compatibility: version your report schema (`report_version` field).

---

## Step B6: Add auth later (backend-ready, but not blocking)

**Goal:** Don’t block ML progress, but structure code so auth can be added.

### Implement (minimal now)

- Add request-scoped `user` concept via header `X-User` in dev mode.
- Later replace with:
  - Keycloak (free) or JWT auth.

### Tests

- Ensure auth middleware doesn’t break existing endpoints.
- Role check stubs (field/analyst/admin).

---

# Part 2 — AI/ML Development Guide (Audio-first, production mindset)

## Step M1: Set up ML repo structure + reproducibility

**Goal:** Deterministic training and evaluation.

### Implement

Create folders:
- `ml/`
  - `datasets/` (download scripts, not data)
  - `training/`
  - `models/`
  - `evaluation/`
  - `export/`
  - `tests/`

Pin environment:
- `requirements.txt` or `pyproject.toml`
- fix random seeds
- log config
- dataset versions

### Tests

- `pytest` unit test ensures:
  - seeds set
  - training script runs one mini-batch without crashing (smoke test)

**Intensive testing**
- CI-style run locally: training script with `--max_steps 2` must always pass.

---

## Step M2: Dataset acquisition + validation (no training yet)

**Goal:** Avoid “garbage in”, ensure dataset integrity.

### Implement

Use free public datasets:
- ASVspoof (audio spoof/deepfake)

Write:
- dataset downloader script
- dataset manifest builder:
  - records path, label, duration, sample rate, split

### Tests

Validate:
- no missing files
- sample rates are readable
- duration within expected range
- label distribution sanity check

**Intensive testing**
- Corrupt file test: dataset loader must skip or fail clearly.

---

## Step M3: Baseline feature pipeline (log-mel + augmentations)

**Goal:** A robust preprocessing pipeline that simulates real-world compression.

### Implement

Preprocessing:
- resample to 16k
- normalize loudness
- compute log-mel

Augmentations (for robustness):
- AAC/Opus re-encode simulation (approximate)
- random bitrate reduction
- background noise mix
- phone-call bandpass simulation
- clipping & saturation mild

### Tests

- Determinism test:
  - same input → same output when augmentations off
- Stability test:
  - output shape stable
  - no NaNs/Infs

**Intensive testing**
- 100 random audio clips transformed in a loop, assert no crashes.

---

## Step M4: Train baseline model (small CNN)

**Goal:** First real detector you can deploy behind the API.

### Implement

- small CNN on log-mel
- output probability fake
- train with:
  - train/val split
  - early stopping
  - metrics: ROC-AUC, EER, F1, confusion matrix

### Tests

- Training smoke test: `--max_steps 5` finishes
- Evaluation test: metrics computation runs

**Intensive testing**
- Regression metric test:
  - store expected metric range for a mini dataset
  - alert if it drops beyond tolerance

---

## Step M5: Calibration + threshold policy

**Goal:** Confidence that means something.

### Implement

- Calibrate probabilities (temperature scaling on validation set)
- Define thresholds:
  - `DEEPFAKE` if `p_fake >= T_high`
  - `AUTHENTIC` if `p_fake <= T_low`
  - else `SUSPICIOUS`

### Tests

- Calibration test: ECE improves vs raw
- Threshold test: ensure SUSPICIOUS band works

**Intensive testing**
- Extreme compression should increase uncertainty.

---

## Step M6: Explainability outputs (deterministic, no paid LLM)

**Goal:** Human-readable reasons tied to measurable signals.

### Implement

Compute signals:
- pitch variance proxy
- spectral flux
- energy variance (breathing proxy)
- embedding distance / OOD score (optional)

Map to explanations via templates:
- “Pitch is unusually consistent”
- “Spectral changes too smooth”
- “Energy variation unusually uniform”
- “Compression artifacts reduce certainty”

### Tests

- Deterministic mapping test: same signals → same explanations
- Snapshot tests of explanation text

**Intensive testing**
- Ensure explanations never contradict verdict (rule tests)

---

## Step M7: Export model for backend inference

**Goal:** Ship a model artifact.

### Implement

- Export to ONNX
- Validate ONNX inference matches PyTorch within tolerance

### Tests

- Numeric parity test:
  - random batch → compare outputs
- Performance smoke test:
  - run inference 100 times, measure avg latency

**Intensive testing**
- Fail export if dynamic shapes are incompatible with planned runtime.

---

# Part 3 — Integrate ML into Backend (Real Inference)

## Step I1: Add ONNX Runtime inference into Detector implementation

**Goal:** Replace DummyDetector with a real model.

### Implement

- `OnnxAudioDetector`:
  - loads model on startup
  - runs preprocessing
  - returns DetectorOutput with verdict/confidence/explanations/signals

### Tests

- End-to-end API test:
  - upload → run → fetch report
- Golden sample tests:
  - include a few known samples with expected verdict ranges

**Intensive testing**
- Concurrency: 10 parallel `/run` requests.
- Memory stability: run 200 inferences, ensure no growth.

---

## Step I2: Add model versioning + artifact management

**Goal:** You can update models without breaking API.

### Implement

- models stored locally initially: `models/audio/latest.onnx`
- `model_version` stored in DB
- API exposes `GET /v1/models`

### Tests

- Load failure test: missing model file → service fails fast with clear error
- Hot reload test (optional): swap model, restart, confirm version changes

---

# Decision Needed (to proceed cleanly)

Do you want to start with:
- **Audio-only first** (recommended)
- **Audio + Video from day one** (slower but more complete)

If you pick audio-only, the next concrete artifacts to produce are:
1) exact folder layout for `services/` + `ml/`
2) API spec (endpoints + request/response)
3) test matrix (unit/integration/load/regression)
4) milestone checklist
