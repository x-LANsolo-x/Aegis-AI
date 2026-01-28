# AegisAI-Edge Test Matrix (Audio-Only)

Test coverage summary for backend + ML pipeline.

---

## Backend Tests (`services/api/tests`)

### Test Categories

#### **1. Unit Tests**
| Test File | Focus | Count |
|-----------|-------|-------|
| `test_api.py` | Health + version contract | 3 |
| `test_db_models.py` | DB schema roundtrip + defaults | 2 |
| `test_repository.py` | Repository CRUD | 1 |
| `test_repository_list_count.py` | Pagination + count | 1 |
| `test_schemas.py` | Pydantic response models | 2 |
| `test_detector_models.py` | DetectorInput/Output validation | 2 |
| `test_upload_config.py` | Extension validation + dir creation | 2 |
| `test_utils_sanitize_filename.py` | Filename sanitization | 5 |
| `test_utils_stream_sha256.py` | Streaming hash + save | 2 |
| `test_reporting_builder.py` | Report builder output | 1 |
| `test_auth.py` | User model + roles + middleware | 4 |
| `test_models_endpoint.py` | GET /v1/models | 2 |

**Subtotal:** ~27 unit tests

#### **2. Integration Tests**
| Test File | Focus | Count |
|-----------|-------|-------|
| `test_upload_endpoint.py` | Upload workflow + validation | 4 |
| `test_upload.py` | Upload + DB verification + repeat uploads | 6 |
| `test_get_analysis_endpoints.py` | GET /analysis + list pagination | 5 |
| `test_report_endpoint.py` | GET /analysis/{id}/report | 2 |
| `test_reports.py` | Report contract + snapshot | 3 |
| `test_run_endpoint.py` | POST /run + detector integration | 3 |
| `test_detector_run.py` | Detector run with timeout/FAILED | 3 |
| `test_startup_db_init.py` | Startup DB initialization | 1 |

**Subtotal:** ~27 integration tests

#### **3. Regression Tests**
| Test File | Focus | Count |
|-----------|-------|-------|
| `test_upload_fuzz.py` | Fuzz tests (no 500s) | 65 parametrized |
| `test_reports.py` | Snapshot comparison (fixture-based) | 1 |

**Subtotal:** ~66 regression tests

#### **4. Load / Intensive Tests**
| Test File | Focus | Count |
|-----------|-------|-------|
| `test_inference.py` | Concurrency (10 parallel /run) | 1 |
| `test_inference.py` | Memory stability (200 sequential) | 1 |

**Subtotal:** 2 intensive tests

#### **5. Operational / Deployment Tests**
| Test File | Focus | Count |
|-----------|-------|-------|
| `test_model_loading.py` | Startup validation + hot reload | 4 |

**Subtotal:** 4 deployment tests

---

### Backend Test Summary

| Category | Tests | Status |
|----------|-------|--------|
| **Unit** | 27 | ✅ passing |
| **Integration** | 27 | ✅ passing |
| **Regression** | 66 | ✅ passing |
| **Load/Intensive** | 2 | ✅ passing |
| **Deployment** | 4 | ✅ passing |
| **Total Backend** | **126** | ✅ **all passing** |

---

## ML Tests (`ml/tests`)

### Test Categories

#### **1. Dataset Validation Tests**
| Test File | Focus | Count |
|-----------|-------|-------|
| `test_dataset_validation.py` | No missing files | 1 |
| `test_dataset_validation.py` | Sample rate check | 1 |
| `test_dataset_validation.py` | Duration range violations | 1 |
| `test_dataset_validation.py` | Label distribution warnings | 1 |
| `test_dataset_validation.py` | Corrupt file handling | 1 |

**Subtotal:** 5 tests

---

### ML Test Summary

| Category | Tests | Status |
|----------|-------|--------|
| **Dataset validation** | 5 | ✅ passing |
| **Total ML** | **5** | ✅ **all passing** |

---

## Combined Test Summary

| Domain | Tests | Status |
|--------|-------|--------|
| **Backend (services/api)** | 126 | ✅ passing |
| **ML (ml/)** | 5 | ✅ passing |
| **TOTAL** | **131** | ✅ **all passing** |

---

## Test Coverage by Feature

### ✅ Fully Tested
- Upload (multipart, hashing, sanitization, validation)
- Analysis record CRUD
- Report generation (JSON + snapshot)
- Detector interface (Dummy + ONNX)
- Detector run (timeout, failure, success)
- Model loading (startup validation, hot reload)
- RBAC stubs (middleware, roles)
- Auth (user model, get_current_user)
- Fuzz testing (no 500s)
- Concurrency + memory stability

### 🔄 Pending (requires real model)
- ONNX inference with real model (placeholder test exists, skipped)
- Audio preprocessing with real audio clips
- Verdict accuracy validation

---

## Running Tests

### Backend
```bash
# All backend tests
py -m pytest services/api/tests

# Specific category
py -m pytest services/api/tests/test_upload*.py

# With coverage (requires pytest-cov)
py -m pytest services/api/tests --cov=services.api.app --cov-report=html
```

### ML
```bash
# All ML tests
py -m pytest ml/tests

# Dataset validation only
py -m pytest ml/tests/test_dataset_validation.py
```

### All
```bash
# Everything
py -m pytest services/api/tests ml/tests
```

---

## Test Quality Metrics

### Isolation
- ✅ Each test uses isolated tmp directories + databases
- ✅ Module reloading ensures clean env vars per test
- ✅ No shared state between tests

### Determinism
- ✅ Fixed seeds for any randomness
- ✅ Snapshot tests for contract stability
- ✅ Golden samples with known hashes

### Performance
- Backend tests: **~7–8 seconds** (126 tests)
- ML tests: **~7 seconds** (5 tests)
- **Total runtime: <20 seconds**

---

## Next Test Additions (Post-MVP-0)

1. **Real ONNX model tests** (pending trained model)
2. **API rate limiting** tests
3. **Pagination edge cases** (offset > total)
4. **Chain-of-custody** verification tests
5. **Federated learning** integration tests (when implemented)
