# 🔧 Colab PyTorch Error Fix

## Problem
You're seeing: `SystemError: @+() method: bad call flags`

This is a PyTorch compatibility issue in the current Colab environment.

## Solution

Replace Cell 1 in the notebook with this updated version:

```python
# Upload the standalone training script
from google.colab import files
from pathlib import Path
import shutil
import os

print("Please upload: train_audio_standalone.py")
print("Location: ml/training/train_audio_standalone.py")
uploaded = files.upload()

# Move to current directory
for filename in uploaded:
    # Handle the "(1)" suffix Colab adds
    actual_name = 'train_audio_standalone.py'
    shutil.move(filename, actual_name)
    print(f"✓ Uploaded: {actual_name}")

# Create __init__.py files (cross-platform way)
os.makedirs('ml', exist_ok=True)
os.makedirs('ml/training', exist_ok=True)
os.makedirs('ml/datasets', exist_ok=True)
Path('ml/__init__.py').touch()
Path('ml/training/__init__.py').touch()
Path('ml/datasets/__init__.py').touch()
print("✓ Created directory structure")

# Restart runtime to fix PyTorch (run this after Cell 1 completes)
print("\n⚠️ After this cell completes:")
print("   1. Click 'Runtime' → 'Restart runtime'")
print("   2. Then run Cell 2 (Quick Test)")
```

## Steps to Fix

### Option A: Edit Cell 1 in Colab

1. In Colab, **click on Cell 1**
2. **Delete the current code**
3. **Paste the fixed code above**
4. **Run Cell 1**
5. After it completes: **Runtime → Restart runtime**
6. **Then run Cell 2**

### Option B: Simpler - Skip GPU Check

Or just remove the GPU check from Cell 1:

1. In Cell 1, **delete these lines:**
   ```python
   # Verify GPU
   import torch
   print(f"\n✓ CUDA available: {torch.cuda.is_available()}")
   if torch.cuda.is_available():
       print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
   ```

2. **Run Cell 1** (will upload file only, no GPU check)
3. **Go directly to Cell 2** (training script will auto-detect GPU)

---

## ✅ Recommended: Use Option B (Simpler)

The GPU check isn't critical - the training script will automatically use GPU if available.

**Modified Cell 1 (Simpler Version):**
```python
# Upload the standalone training script
from google.colab import files
from pathlib import Path
import shutil
import os

print("Please upload: train_audio_standalone.py")
print("Location: ml/training/train_audio_standalone.py")
uploaded = files.upload()

# Move to current directory and rename properly
for filename in uploaded:
    actual_name = 'train_audio_standalone.py'
    if filename != actual_name:
        shutil.move(filename, actual_name)
    print(f"✓ Uploaded: {actual_name}")

# Create __init__.py files
os.makedirs('ml', exist_ok=True)
os.makedirs('ml/training', exist_ok=True)
os.makedirs('ml/datasets', exist_ok=True)
Path('ml/__init__.py').touch()
Path('ml/training/__init__.py').touch()
Path('ml/datasets/__init__.py').touch()
print("✓ Created directory structure")

print("\n✅ Ready to train!")
print("📍 Next: Run Cell 2 (Quick Test)")
```

---

## What to Do Now

1. **In Colab**, click on **Cell 1**
2. **Clear all the code** (select all and delete)
3. **Paste the "Simpler Version" code** from above
4. **Run Cell 1** again
5. **Proceed to Cell 2** (Quick Test)

The training script will check for GPU automatically when it runs!

---

## Why This Happens

Colab sometimes has PyTorch environment issues. The training script has its own GPU detection that works properly:

```python
# In train_audio_standalone.py (line ~600)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Device: {device}")
```

So you'll see the GPU status when training starts!

---

**Ready?** Update Cell 1 with the simpler code and run it again! 🚀
