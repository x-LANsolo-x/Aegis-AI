# Colab Training Guide

This guide walks you through training the Aegis-AI audio deepfake detection model on Google Colab.

## Why Colab?

- **Free GPU access** (T4, better performance than CPU)
- **No local system load** (training runs in the cloud)
- **Faster downloads** (Colab has better bandwidth than most home connections)
- **Pre-installed dependencies** (PyTorch, torchaudio already available)

## Prerequisites

1. Google account (for Colab access)
2. Colab extension installed in VS Code (optional, for editing notebooks)

## Step-by-Step Instructions

### 1. Upload Notebook to Colab

**Option A: Direct Upload**
1. Go to [Google Colab](https://colab.research.google.com/)
2. Click "File" → "Upload notebook"
3. Upload `ml/training/train_audio_colab.ipynb`

**Option B: From GitHub**
1. Push your code to GitHub
2. In Colab: "File" → "Open notebook" → "GitHub" tab
3. Enter your repo URL

### 2. Enable GPU Runtime

1. In Colab: "Runtime" → "Change runtime type"
2. Hardware accelerator: **GPU** (T4 recommended)
3. Click "Save"

### 3. Upload Required Files

The notebook will prompt you to upload these files:
- `ml/training/train_audio.py`
- `ml/training/logging_config.py`
- `ml/datasets/loader.py`
- `ml/datasets/__init__.py` (create empty file if needed)

### 4. Run Training

**Quick Test (5-10 minutes):**
```python
# Cell 7: Quick test with subset
# Uses 500 train samples, 100 val samples, 2 epochs
```

**Full Training (1-3 hours):**
```python
# Cell 8: Full training
# Uses entire dataset, 20 epochs with early stopping
```

### 5. Download Trained Model

After training completes:
1. Run Cell 10 to download files
2. You'll get:
   - `V1.0.0.onnx` (ONNX model)
   - `V1.0.0.json` (metadata)
   - `V1.0.0_best.pt` (PyTorch checkpoint)

### 6. Integrate Model Locally

Once downloaded, integrate into your local API:

```bash
# Copy model files
cp ~/Downloads/V1.0.0.onnx models/audio/
cp ~/Downloads/V1.0.0.json models/audio/

# Or use the integration script
python ml/scripts/integrate_trained_model.py --model ~/Downloads/V1.0.0.onnx

# The script will:
# - Copy model to models/audio/
# - Create latest.onnx symlink
# - Verify model loads correctly
# - Show next steps
```

## Alternative: Manual Training Script Upload

If you prefer not to use the notebook, you can upload the Python script directly:

```python
# In Colab cell:
from google.colab import files
uploaded = files.upload()  # Upload train_audio.py

# Then run directly:
!python train_audio.py \
    --manifest ml/datasets/manifests/asvspoof_2019.jsonl \
    --output-dir models/audio \
    --model-version V1.0.0 \
    --epochs 20 \
    --batch-size 32 \
    --device cuda
```

## Training Configuration

### Recommended Settings

**Quick Test (fast iteration):**
- `--epochs 2`
- `--batch-size 16`
- `--max-train-samples 500`
- `--max-val-samples 100`
- Time: ~5-10 minutes

**Production Training (best performance):**
- `--epochs 20`
- `--batch-size 32`
- No sample limits
- Time: ~1-3 hours

### Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--epochs` | 20 | Maximum training epochs |
| `--batch-size` | 32 | Batch size (reduce if OOM) |
| `--lr` | 1e-3 | Learning rate |
| `--seed` | 42 | Random seed (reproducibility) |
| `--device` | cuda | Device (cuda/cpu) |

### Model Architecture

- **Input:** Log-mel spectrogram (64 mel bins, ~1001 time frames for 10s audio)
- **Architecture:** 4-layer CNN with batch norm and max pooling
- **Parameters:** ~2.5M (lightweight)
- **Output:** 2 classes (bonafide, spoof)

### Expected Performance

With full training on ASVspoof 2019 LA:
- **Training time:** 1-3 hours on T4 GPU
- **Validation accuracy:** 85-95% (depends on hyperparameters)
- **EER (Equal Error Rate):** 10-20% (lower is better)
- **Inference latency:** < 100ms per 10s audio

## Troubleshooting

### "Out of Memory" Error
- Reduce batch size: `--batch-size 16` or `--batch-size 8`
- Reduce max audio duration (edit `AudioFeatureExtractor.max_duration_sec`)

### Dataset Download Slow
- Colab has good bandwidth, but ASVspoof is ~7.6 GB
- Download takes 10-30 minutes depending on server load
- Alternative: Upload your own pre-downloaded dataset

### Model Not Loading
- Check ONNX Runtime is installed: `!pip install onnxruntime`
- Verify model was exported successfully (check logs)
- Try re-running the export cell

### Training Too Slow
- Verify GPU is enabled (Cell 1 output should show GPU type)
- If on CPU, switch to GPU runtime
- Consider using fewer samples for quick iteration

## What's Next?

After training and downloading the model:

1. **Integrate model** (see Step 6 above)
2. **Test inference** locally:
   ```bash
   export ONNX_MODEL_PATH=models/audio/latest.onnx
   cd services/api
   uvicorn app.main:app --reload
   ```
3. **Run inference tests**:
   ```bash
   pytest services/api/tests/test_inference.py -v
   ```
4. **Create golden samples** for validation:
   ```bash
   python ml/scripts/create_golden_samples.py \
       --manifest ml/datasets/manifests/asvspoof_2019.jsonl \
       --output-dir services/api/tests/fixtures/golden_samples
   ```

## Tips

- **Save checkpoints frequently:** The notebook automatically saves the best model
- **Monitor training:** Watch the validation loss; training should stop early if no improvement
- **Version your models:** Use semantic versioning (V1.0.0, V1.1.0, etc.)
- **Test before deploying:** Always verify the ONNX export matches PyTorch output

## Resources

- [ASVspoof 2019 Challenge](https://www.asvspoof.org/)
- [ONNX Runtime Documentation](https://onnxruntime.ai/)
- [Google Colab FAQ](https://research.google.com/colaboratory/faq.html)
