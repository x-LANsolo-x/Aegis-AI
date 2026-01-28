# Model Storage Convention

This directory stores ONNX model artifacts for audio deepfake detection.

## Naming Convention

- `latest.onnx` — symlink or copy of the current production model
- `V{MAJOR}.{MINOR}.{PATCH}.onnx` — versioned models (semantic versioning)

Example:
```
models/audio/
  latest.onnx -> V1.0.0.onnx
  V1.0.0.onnx
  V0.9.0.onnx
```

## Model Metadata

Each model should have an accompanying metadata file:
- `V1.0.0.json` containing:
  - training date
  - dataset version
  - performance metrics (EER, ROC-AUC, etc.)
  - input/output specs
  - preprocessing requirements

## Deployment

The backend API reads the model path from the `ONNX_MODEL_PATH` environment variable:
```bash
export ONNX_MODEL_PATH=models/audio/latest.onnx
```

## Version Bumps

- **Patch** (x.y.Z): bug fixes, minor improvements (no retraining)
- **Minor** (x.Y.0): retrained on new data, architecture unchanged
- **Major** (X.0.0): new architecture or breaking input/output changes

## .gitignore

Large model files should be excluded from git and stored via:
- DVC (Data Version Control)
- Git LFS
- MinIO / S3 artifact registry
