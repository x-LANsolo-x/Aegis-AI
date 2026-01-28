# AegisAI-Edge Folder Layout (Audio-Only)

Current workspace structure for audio deepfake detection product.

```
/
├── README.md                          # Product overview + deployment guide
├── architecture.md                    # System architecture (100% free stack)
├── backend_ml_dev_guide.md           # Backend + ML development guide
├── DEV_CHANGELOG.md                  # Development change log
├── FOLDER_LAYOUT.md                  # This file
├── .gitignore
│
├── legacy-demo/                       # Archived hackathon prototype
│   ├── aegis-ui-demo/                # Static UI prototype
│   ├── model-demo/                   # Early Gradio demos
│   └── ...
│
├── services/                          # Backend API service
│   ├── __init__.py
│   └── api/
│       ├── __init__.py
│       ├── requirements.txt          # FastAPI, onnxruntime, soundfile, scipy, etc.
│       ├── app/
│       │   ├── __init__.py
│       │   ├── main.py               # FastAPI app + endpoints
│       │   ├── models.py             # SQLModel DB schema (AnalysisRecord)
│       │   ├── database.py           # DB engine + session
│       │   ├── db.py                 # Session helpers
│       │   ├── repository.py         # CRUD operations
│       │   ├── schemas.py            # Pydantic response models
│       │   ├── detector.py           # Detector protocol + OnnxAudioDetector + DummyDetector
│       │   ├── audio_preprocess.py   # Audio loading + resampling + normalization
│       │   ├── reporting.py          # Report builder (AnalysisReportOut)
│       │   ├── config.py             # Upload dir + max size + allowed extensions
│       │   ├── media.py              # Media type inference from extension
│       │   ├── utils.py              # Filename sanitization + SHA-256 streaming
│       │   ├── auth.py               # User model + roles + RBAC stubs
│       │   └── enums.py              # Verdict enum (PENDING/AUTHENTIC/SUSPICIOUS/DEEPFAKE/FAILED)
│       │
│       └── tests/
│           ├── __init__.py
│           ├── test_api.py           # Health + version contract tests
│           ├── test_auth.py          # Auth middleware + role checks
│           ├── test_db_models.py     # DB schema + roundtrip
│           ├── test_repository.py    # Repository CRUD
│           ├── test_repository_list_count.py
│           ├── test_schemas.py       # Pydantic response models
│           ├── test_detector_models.py  # DetectorInput/Output validation
│           ├── test_upload_config.py # Upload dir + extension validation
│           ├── test_utils_sanitize_filename.py
│           ├── test_utils_stream_sha256.py
│           ├── test_upload.py        # Upload endpoint + DB verification
│           ├── test_upload_fuzz.py   # Fuzz tests (no 500s)
│           ├── test_upload_endpoint.py
│           ├── test_get_analysis_endpoints.py  # GET /v1/analysis/* tests
│           ├── test_report_endpoint.py
│           ├── test_reports.py       # Report contract + snapshot
│           ├── test_run_endpoint.py  # POST /run tests
│           ├── test_detector_run.py  # Detector run with timeout/failure
│           ├── test_startup_db_init.py
│           ├── test_get_current_user.py
│           ├── test_inference.py     # End-to-end + golden samples + intensive (concurrency/memory)
│           ├── test_models_endpoint.py  # GET /v1/models
│           ├── test_model_loading.py # Startup validation + hot reload
│           ├── test_reporting_builder.py
│           └── fixtures/
│               └── report_snapshot.json
│
├── ml/                                # ML pipeline (audio-only)
│   ├── __init__.py
│   ├── datasets/
│   │   ├── __init__.py
│   │   ├── download_asvspoof.py      # CLI downloader (2019/2021)
│   │   ├── build_manifest.py         # Manifest builder (JSONL output)
│   │   ├── validate_manifest.py      # Validation script
│   │   ├── loader.py                 # AudioSample + load_audio + read_manifest
│   │   ├── asvspoof_2019/            # Downloaded dataset (excluded from git)
│   │   │   └── download.log
│   │   └── manifests/
│   │       └── asvspoof_2019.jsonl   # (generated after download completes)
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   └── logging_config.py         # configure_logging + log_run_header
│   │
│   └── tests/
│       ├── __init__.py
│       ├── generate_fixtures.py      # Generate tiny synthetic WAVs
│       ├── test_dataset_validation.py # Manifest validation tests
│       └── fixtures/
│           ├── valid_short.wav
│           ├── valid_normal.wav
│           ├── valid_long.wav
│           ├── valid_toolong.wav
│           └── corrupt.wav
│
├── models/                            # Model artifacts (excluded from git)
│   └── audio/
│       ├── README.md                 # Versioning convention
│       ├── .gitkeep
│       ├── latest.onnx               # (symlink or copy to current model)
│       ├── V1.0.0.onnx               # Versioned snapshots
│       └── V1.0.0.json               # Model metadata (optional)
│
└── audio1.ogg, audio2.ogg, ...       # Test audio samples (legacy; can be moved)
```

## Key Directories

### `services/api/`
Backend API service (FastAPI):
- Production-ready endpoints for upload/run/report/list
- ONNX Runtime integration
- SQLModel DB layer (SQLite dev, Postgres prod)
- RBAC stubs
- **121+ tests**

### `ml/`
ML pipeline (audio-only):
- Dataset downloading + manifest building
- Loader + validation utilities
- Logging config for training runs
- **5 tests**

### `models/audio/`
Model artifact storage:
- `latest.onnx` → production model
- `V{major}.{minor}.{patch}.onnx` → versioned snapshots

### `legacy-demo/`
Archived hackathon prototype (not used in production).

## Entry Points

### Backend API
```bash
# Dev mode (DummyDetector)
uvicorn services.api.app.main:app --reload --port 8000

# Production mode (ONNX model)
export ONNX_MODEL_PATH=models/audio/latest.onnx
uvicorn services.api.app.main:app --port 8000
```

### ML Pipeline
```bash
# Download dataset
py -m ml.datasets.download_asvspoof --version 2019 --output_dir ml/datasets --background

# Build manifest
py -m ml.datasets.build_manifest --version 2019 --base_dir ml/datasets

# Validate manifest
py -m ml.datasets.validate_manifest --manifest ml/datasets/manifests/asvspoof_2019.jsonl
```

### Tests
```bash
# Backend tests
py -m pytest services/api/tests

# ML tests
py -m pytest ml/tests
```

## Git Exclusions

The following are excluded from version control:
- `models/**/*.onnx` (large model files)
- `ml/datasets/asvspoof_*/` (large datasets)
- `*.db` (SQLite dev databases)
- Standard Python excludes (`__pycache__`, `.pytest_cache`, etc.)
