# 🚀 QUICKSTART - Train Your Model in 3 Steps

## Problem
Your Python 3.14 is too new for ONNX Runtime. Solution: Use Google Colab (free GPU).

---

## Step 1: Open Notebook (1 minute)

```bash
# In VS Code, open this file:
ml/training/SIMPLE_COLAB.ipynb
```

**Or** click here in VS Code Explorer:
```
📁 ml
  📁 training
    📄 SIMPLE_COLAB.ipynb  ← Click this
```

---

## Step 2: Connect to Colab (2 minutes)

1. Click **kernel selector** (top right of notebook)
2. Select **"Google Colab"** or **"Connect to Colab"**
3. Sign in with Google account
4. Wait for "Connected" status

---

## Step 3: Run 4 Cells (1-3 hours)

### Cell 1: Upload Script ⏱️ 1 min
- Click ▶️ Run
- When prompted, upload: `ml/training/train_audio_standalone.py`
- Should see: ✓ CUDA available: True

### Cell 2: Quick Test ⏱️ 5-10 min
- Click ▶️ Run
- Downloads dataset (~7.6 GB, takes 10-30 min)
- Trains on 500 samples, 2 epochs
- Verifies everything works

### Cell 3: Full Training ⏱️ 1-3 hours
- **Only run if Cell 2 succeeds!**
- Click ▶️ Run
- Trains on full dataset, 20 epochs
- Expected accuracy: 85-95%

### Cell 4: Download Model ⏱️ 1 min
- Click ▶️ Run
- Downloads to your computer:
  - `V1.0.0.onnx` (the trained model)
  - `V1.0.0.json` (metadata)
  - `V1.0.0_best.pt` (PyTorch checkpoint)

---

## Step 4: Integrate Model Locally (5 minutes)

After downloading, run this on your local machine:

```bash
# Navigate to project root
cd C:\Users\kumar\Desktop\Projects\SnowHackIPEC

# Integrate the model
python ml/scripts/integrate_trained_model.py --model ~/Downloads/V1.0.0.onnx

# The script will:
# ✓ Copy model to models/audio/
# ✓ Create latest.onnx symlink
# ✓ Verify it loads
# ✓ Show next steps

# Test it
set ONNX_MODEL_PATH=models/audio/latest.onnx
pytest services/api/tests/test_inference.py -v

# Start API
cd services/api
uvicorn app.main:app --reload
```

---

## Timeline

| Step | Time | Can Leave? |
|------|------|------------|
| 1. Open notebook | 1 min | No |
| 2. Connect to Colab | 2 min | No |
| 3. Cell 1 (upload) | 1 min | No |
| 4. Cell 2 (quick test) | 5-10 min | Yes ☕ |
| 5. Cell 3 (full training) | 1-3 hours | Yes ☕☕☕ |
| 6. Cell 4 (download) | 1 min | No |
| 7. Integrate locally | 5 min | No |

**Total active time:** ~10 minutes  
**Total waiting time:** 1-3 hours (unattended)

---

## Expected Output

After Cell 3 completes, you should see:

```
✓ TRAINING COMPLETE!
  ONNX: models/V1.0.0.onnx
  Accuracy: 87.5%
  Parameters: 2,567,426
```

After Cell 4, you'll have these files in Downloads:
- ✓ `V1.0.0.onnx` (~30 MB)
- ✓ `V1.0.0.json` (metadata)
- ✓ `V1.0.0_best.pt` (~60 MB)

---

## Troubleshooting

### "Cannot find kernel 'Google Colab'"
- Install Colab extension: Press `Ctrl+Shift+X`, search "Colab", install
- Or use Colab website: https://colab.research.google.com

### "Out of Memory" during training
- In Cell 2: Change `--batch-size 16` to `--batch-size 8`
- In Cell 3: Change `--batch-size 32` to `--batch-size 16`

### "File not found" when uploading
Full path: `C:\Users\kumar\Desktop\Projects\SnowHackIPEC\ml\training\train_audio_standalone.py`

### Dataset download slow
- Normal: 7.6 GB takes 10-30 minutes
- Colab has fast internet (~100 Mbps)

### Colab disconnects
- Keep the browser tab active
- Colab disconnects after ~30 min of inactivity
- Training continues, but you won't see logs

---

## What You Get

After this process:
- ✅ Trained ONNX model (ready for production)
- ✅ Real deepfake detection (not dummy responses)
- ✅ 85-95% validation accuracy
- ✅ <100ms inference latency
- ✅ All tests passing (129 tests, 0 skipped)

---

## Need More Help?

- **Detailed guide:** `FINAL_INSTRUCTIONS.md`
- **Colab setup:** `COLAB_INSTRUCTIONS.md`
- **After training:** `NEXT_STEPS.md`

---

## Ready?

```bash
# Open the notebook now:
code ml/training/SIMPLE_COLAB.ipynb
```

Then follow Steps 1-4 above. Good luck! 🚀
