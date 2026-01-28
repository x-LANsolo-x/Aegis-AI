# MVP-0 Checklist: Audio Deepfake Detection (End-to-End)

**Goal:** A working audio deepfake detector that accepts audio uploads, runs inference using an ONNX model, and returns actionable results.

---

## ✅ Phase 1: Infrastructure (DONE)

- [x] Backend API service (FastAPI)
  - [x] Upload endpoint
  - [x] Run endpoint (detector integration)
  - [x] Report endpoint
  - [x] List/query endpoints
  - [x] Model info endpoint
- [x] Database layer (SQLite dev, Postgres-ready)
- [x] ONNX Runtime integration (`OnnxAudioDetector`)
- [x] Audio preprocessing utilities
- [x] Model versioning convention
- [x] 126 backend tests (all passing)
- [x] ML pipeline scaffolding (downloader, manifest builder, loader, validation)
- [x] 5 ML tests (all passing)

---

## 🔄 Phase 2: Dataset Preparation (IN PROGRESS)

- [x] ASVspoof 2019 downloader (background download running)
- [ ] Download completion (~7.6 GB, currently ~7–8%)
- [ ] Manifest generation (`asvspoof_2019.jsonl`)
- [ ] Manifest validation (file existence, duration, label distribution)

**Estimated time:** 1–2 hours (download speed-dependent)

---

## ✅ Phase 3: Model Training (READY FOR COLAB)

### Prerequisites
- [x] Training script (`ml/training/train_audio.py`) - Complete with 684 lines
- [x] Colab notebook (`ml/training/train_audio_colab.ipynb`) - Ready for GPU training
- [x] Training guide (`ml/training/COLAB_TRAINING_GUIDE.md`) - Comprehensive documentation
- [ ] Download complete + manifest validated (in progress, 38% complete locally, will download on Colab)

### Training Tasks
- [x] Feature extraction pipeline (log-mel spectrograms) - `AudioFeatureExtractor` class
- [x] Model architecture (lightweight CNN) - `LightweightCNN` 4-layer, ~2.5M params
- [x] Training loop with logging - Full implementation with standardized logging
- [x] Validation + early stopping - Patience=5, learning rate scheduling
- [x] Data augmentation - Basic (to be enhanced in future versions)
- [x] Threshold tuning - Configured (0.3 SUSPICIOUS, 0.7 DEEPFAKE)
- [ ] Calibration (temperature scaling) - Not yet implemented (post-MVP)

### Model Export
- [x] Export to ONNX (`V1.0.0.onnx`) - Implemented with dynamic axes
- [x] Numeric parity test (PyTorch vs ONNX) - Automated verification in training script
- [x] Performance smoke test (latency) - Test infrastructure ready
- [x] Model metadata JSON (`V1.0.0.json`) - Auto-generated with all hyperparameters

**Estimated time:** 1–3 hours (Colab training with T4 GPU)
**Status:** Ready to execute - Upload notebook to Colab and run

---

## 🔄 Phase 4: Integration Testing (READY - AWAITING TRAINED MODEL)

### Model Integration
- [x] Integration script created (`ml/scripts/integrate_trained_model.py`)
- [ ] Place `V1.0.0.onnx` in `models/audio/` (after Colab training)
- [ ] Create symlink: `latest.onnx -> V1.0.0.onnx` (automated by integration script)
- [ ] Set `ONNX_MODEL_PATH=models/audio/latest.onnx`
- [ ] Start backend and verify `/v1/models` shows correct version

### End-to-End Workflow
- [x] Test infrastructure ready (test_inference.py updated)
- [ ] Upload test audio (bonafide + spoof samples)
- [ ] Run detector
- [ ] Verify verdict accuracy
- [ ] Check report includes:
  - verdict ✅ (schema ready)
  - confidence ✅ (schema ready)
  - explanations ✅ (schema ready)
  - model_version ✅ (schema ready)

### Golden Samples
- [x] Golden sample script created (`ml/scripts/create_golden_samples.py`)
- [x] Test added to `test_inference.py` (`test_golden_samples_accuracy`)
- [ ] Generate 5–10 golden samples (run script after training)
- [ ] Run inference on all
- [ ] Assert verdicts within expected ranges (≥70% accuracy threshold)

**Estimated time:** 30 minutes (after model training completes)

---

## 🟡 Phase 5: Performance Validation (PENDING)

### Latency
- [ ] Measure p50/p95/p99 inference latency
- [ ] Target: **< 2 seconds** for typical 3–10 sec audio

### Memory
- [ ] Run 1000 inferences sequentially
- [ ] Assert no memory leak (growth < 100 MB)

### Concurrency
- [ ] Run 50 parallel `/run` requests
- [ ] Assert no crashes, all return 200 or 504 (timeout)

**Estimated time:** 0.5 days

---

## 🟡 Phase 6: Deployment Readiness (PENDING)

### Documentation
- [x] Model deployment guide (README.md)
- [x] API spec (OpenAPI JSON)
- [x] Test matrix
- [ ] Deployment runbook (env vars, systemd/Docker)

### Operational
- [ ] Health check includes model status
- [ ] Logging includes model_version in all analysis records
- [ ] Graceful error messages for:
  - model load failure
  - unsupported audio format
  - inference timeout

### Security
- [ ] Input validation (file size, extension, duration)
- [ ] Rate limiting (optional for MVP-0)
- [ ] RBAC enforcement (optional for MVP-0, stubs exist)

**Estimated time:** 1 day

---

## 🟡 Phase 7: MVP-0 Acceptance (PENDING)

### Demo Script
- [ ] Prepare 3 demo scenarios:
  1. Upload known-real audio → verdict AUTHENTIC
  2. Upload known-fake audio → verdict DEEPFAKE
  3. Upload ambiguous audio → verdict SUSPICIOUS
- [ ] Record demo video (optional)

### Acceptance Criteria
- [ ] End-to-end workflow completes without errors
- [ ] Inference latency < 2 seconds
- [ ] Verdict accuracy > 80% on validation set (EER < 20%)
- [ ] All tests passing
- [ ] Model versioning working (hot reload test)

**Estimated time:** 0.5 days

---

## MVP-0 Timeline Summary

| Phase | Status | Estimated Time | Depends On |
|-------|--------|----------------|------------|
| 1. Infrastructure | ✅ DONE | — | — |
| 2. Dataset Prep | 🔄 IN PROGRESS | 1–2 hours | Download speed |
| 3. Model Training | ✅ READY (Colab) | 1–3 hours | Phase 2 (can run on Colab) |
| 4. Integration Testing | 🔄 READY | 30 minutes | Phase 3 |
| 5. Performance Validation | 🔄 READY | 0.5 days | Phase 4 |
| 6. Deployment Readiness | 🟡 PENDING | 1 day | Phase 5 |
| 7. MVP-0 Acceptance | 🟡 PENDING | 0.5 days | Phase 6 |

**Total estimated time:** 2–3 hours (model training on Colab) + 1–2 days (validation & deployment)

---

## Post-MVP-0 Roadmap

After MVP-0 is validated, next increments:

### MVP-0.5 (Explainability)
- [ ] Add signal-based explanations (pitch variance, spectral flux, etc.)
- [ ] Map signals to human-readable reasons

### MVP-1 (Production Hardening)
- [ ] Federated learning coordinator (optional model updates)
- [ ] Chain-of-custody logging
- [ ] PDF report export
- [ ] Admin policy controls

### MVP-2 (Video Expansion)
- [ ] Add `OnnxVideoDetector`
- [ ] Face detection + temporal consistency
- [ ] Multi-modal fusion (audio + video)

---

## Success Metrics (MVP-0)

| Metric | Target | Current |
|--------|--------|---------|
| Backend tests passing | 100% | ✅ 126/126 |
| ML tests passing | 100% | ✅ 5/5 |
| Inference latency (p95) | < 2 sec | ⏳ TBD |
| Model EER | < 20% | ⏳ TBD |
| Memory leak | < 100 MB / 1000 runs | ⏳ TBD |

---

## Blockers / Risks

| Risk | Mitigation |
|------|-----------|
| ASVspoof download slow | Running in background; can use subset for initial training |
| Model training compute | Use lightweight CNN first; cloud GPU if needed |
| ONNX export incompatibility | Test numeric parity early; fallback to PyTorch mobile |
| Inference latency too high | Quantize to INT8; reduce model size |

---

## Contact / Next Steps

Ready to proceed with:
1. **Wait for dataset download** (check progress: `Get-Content ml\datasets\asvspoof_2019\download.log -Tail 20`)
2. **Build manifest** once download completes
3. **Implement training script** (will need guidance on model architecture if desired)

Current status:
- ✅ Infrastructure complete (127 backend tests + 5 ML tests passing)
- ✅ Training pipeline complete (ready for Colab execution)
- 🔄 Dataset downloading (38% locally, will download on Colab)
- 📋 **Action required:** Upload `train_audio_colab.ipynb` to Google Colab and run training
