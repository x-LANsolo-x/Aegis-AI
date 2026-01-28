# 🚀 Running Training in VS Code with Colab Extension

## Why We Need Colab

Your system has **Python 3.14** which is too new for ONNX Runtime (not yet supported). Google Colab provides:
- ✅ Python 3.10 (stable, all packages available)
- ✅ Free GPU (T4, much faster than CPU)
- ✅ Pre-installed PyTorch, torchaudio, ONNX Runtime
- ✅ High-bandwidth dataset download

## Step-by-Step: Using VS Code Colab Extension

### 1. Open the Notebook

In VS Code:
```
File → Open File → ml/training/train_audio_colab.ipynb
```

Or from terminal:
```bash
code ml/training/train_audio_colab.ipynb
```

### 2. Select Colab Kernel

When the notebook opens:
1. Click the **kernel selector** in the top right (should show "Select Kernel")
2. Choose **"Google Colab"** or **"Connect to Colab"**
3. You'll be prompted to sign in with your Google account
4. Authorize VS Code to access Colab

### 3. Enable GPU

In the notebook, add this cell at the top (or it might already be there):
```python
# Check runtime type
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("⚠ Running on CPU - switch to GPU runtime for faster training")
```

If not on GPU:
- In Colab web interface: Runtime → Change runtime type → GPU (T4)
- Or in VS Code, the Colab extension should have a runtime selector

### 4. Run the Notebook Cells

Execute cells **in order** (Shift+Enter or click ▶️ button):

**Cell 1: Install Dependencies**
```python
!pip install -q onnx onnxruntime soundfile
```

**Cell 2-5: Setup and File Upload**
- Follow prompts to upload Python files
- Or skip and use the inline versions I'll provide below

**Cell 6-7: Dataset Download & Manifest**
- This will download ASVspoof 2019 (~7.6 GB)
- Takes 10-30 minutes depending on connection
- Much faster than your local 38% download!

**Cell 8: Quick Test Training** (5-10 minutes)
- Run this first to verify everything works
- Uses 500 training samples, 2 epochs

**Cell 9: Full Training** (1-3 hours)
- Once test passes, run full training
- Uses entire dataset, 20 epochs with early stopping

**Cell 10: Download Model**
- Downloads V1.0.0.onnx, V1.0.0.json to your local machine

---

## Alternative: Simplified Colab Notebook

If you want a simpler experience, I can create a **single-cell** notebook that does everything. Here's what it would look like:

### Single-Cell Training Script

Create a new notebook cell with this complete script:

```python
# === COMPLETE TRAINING SCRIPT (ALL-IN-ONE) ===

# 1. Install dependencies
!pip install -q onnx onnxruntime soundfile

# 2. Download dataset
!mkdir -p datasets
!wget -O datasets/LA.zip https://datashare.ed.ac.uk/bitstream/handle/10283/3336/LA.zip
!unzip -q datasets/LA.zip -d datasets/

# 3. Paste the entire train_audio.py script here (or upload it)
# (I'll create a self-contained version if you prefer)

# 4. Run training
!python train_audio.py \
    --manifest datasets/manifest.jsonl \
    --output-dir models/ \
    --model-version V1.0.0 \
    --epochs 20 \
    --batch-size 32 \
    --device cuda

# 5. Download results
from google.colab import files
files.download('models/V1.0.0.onnx')
files.download('models/V1.0.0.json')
```

---

## What I Recommend

**Option A: Use the Existing Notebook (Recommended)**
1. Open `ml/training/train_audio_colab.ipynb` in VS Code
2. Connect to Colab kernel
3. Run cells in order
4. Download trained model

**Option B: Simple All-in-One Cell**
1. I create a single massive cell with everything
2. You paste it into a new Colab notebook
3. Click Run once
4. Come back in 1-3 hours

**Option C: Use Google Colab Website Directly**
1. Go to https://colab.research.google.com
2. Upload `train_audio_colab.ipynb`
3. Run cells
4. Download model

Which option would you prefer? I can help you with any of them!

---

## Important Notes

⚠️ **Python 3.14 Issue**: Your local Python 3.14 is too new for onnxruntime. This is why Colab is the best option (uses Python 3.10).

✅ **After Training**: Once you download the trained model, the integration script will work fine locally:
```bash
python ml/scripts/integrate_trained_model.py --model ~/Downloads/V1.0.0.onnx
```

✅ **Local Testing**: After model integration, all 129 tests will run locally (including the 2 currently skipped).

---

## Need Help?

Let me know:
1. Which option you prefer (A, B, or C)
2. If you're seeing the Colab kernel option in VS Code
3. Any errors you encounter

I'll guide you through each step! 🚀
