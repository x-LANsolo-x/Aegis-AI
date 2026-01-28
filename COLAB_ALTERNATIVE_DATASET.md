# 🔧 Alternative Dataset Solutions

The ASVspoof server is blocking automated downloads (403 Forbidden).

## 🎯 BEST OPTION: Upload Your Local Partial Dataset

You already have 38% of the dataset downloaded locally!

### Steps:

1. **On your local machine, compress what you have:**
```powershell
# In your project directory
cd ml/datasets/asvspoof_2019
Compress-Archive -Path LA -DestinationPath LA_partial.zip
```

2. **Upload to Google Drive:**
   - Upload `LA_partial.zip` to your Google Drive
   - Right-click → Get link → Copy link ID

3. **In Colab, download from your Drive:**
```python
from google.colab import drive
drive.mount('/content/drive')

# Copy from your Drive
!cp /content/drive/MyDrive/LA_partial.zip datasets/
!unzip -q datasets/LA_partial.zip -d datasets/

# Continue with training on partial dataset
!python train_audio_standalone.py \
    --dataset-dir datasets \
    --model-version V0.1.0-test \
    --epochs 2 \
    --batch-size 16 \
    --quick-test \
    --device cuda
```

---

## 🎯 OPTION 2: Use Alternative Dataset (Faster)

Use a smaller, publicly available dataset for testing:

### FakeAVCeleb (Smaller, ~500MB)

```python
# Download FakeAVCeleb audio subset
!wget https://github.com/DASH-Lab/FakeAVCeleb/releases/download/v1.0/FakeAVCeleb_v1.2_audio.zip
!unzip -q FakeAVCeleb_v1.2_audio.zip -d datasets/

# Build simple manifest
# (I can provide a script for this if you choose this option)
```

---

## 🎯 OPTION 3: Manual Download from Browser

1. **Open in your browser:**
   https://datashare.ed.ac.uk/handle/10283/3336

2. **Click on "LA.zip"** - Should start download

3. **Wait for download** (~7.6 GB)

4. **Upload to Colab:**
```python
from google.colab import files

# This will open file upload dialog
print("Select LA.zip from your Downloads folder")
uploaded = files.upload()

# Extract
import zipfile
with zipfile.ZipFile('LA.zip', 'r') as zip_ref:
    zip_ref.extractall('datasets/')
    
print("✓ Dataset ready!")
```

---

## 🎯 OPTION 4: Use Kaggle Mirror (If Available)

```python
!pip install -q kaggle

# Upload your kaggle.json credentials
from google.colab import files
files.upload()  # Select kaggle.json

!mkdir -p ~/.kaggle
!mv kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

# Download from Kaggle (if someone uploaded it there)
!kaggle datasets download -d asvspoof2019/la-dataset
```

---

## 🎯 OPTION 5: Train on Synthetic Data (Quick Test)

For just testing the pipeline works:

```python
# Create synthetic audio data for testing
import numpy as np
import soundfile as sf
import os
import json
from pathlib import Path

print("Creating synthetic dataset for testing...")

os.makedirs('datasets/LA/ASVspoof2019_LA_train/flac', exist_ok=True)
os.makedirs('datasets/LA/ASVspoof2019_LA_dev/flac', exist_ok=True)
os.makedirs('datasets/LA/ASVspoof2019_LA_cm_protocols', exist_ok=True)

# Create 100 fake audio files (bonafide + spoof)
for i in range(100):
    # Generate random audio (16kHz, 4 seconds)
    audio = np.random.randn(16000 * 4).astype(np.float32) * 0.1
    
    # Save as FLAC
    split = 'train' if i < 80 else 'dev'
    file_id = f'LA_{split[0].upper()}_{i:07d}'
    path = f'datasets/LA/ASVspoof2019_LA_{split}/flac/{file_id}.flac'
    
    sf.write(path, audio, 16000)

# Create protocol files
train_protocol = 'datasets/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt'
dev_protocol = 'datasets/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt'

with open(train_protocol, 'w') as f:
    for i in range(80):
        file_id = f'LA_T_{i:07d}'
        label = 'bonafide' if i % 2 == 0 else 'spoof'
        f.write(f'- {file_id} - - {label}\n')

with open(dev_protocol, 'w') as f:
    for i in range(80, 100):
        file_id = f'LA_D_{i:07d}'
        label = 'bonafide' if i % 2 == 0 else 'spoof'
        f.write(f'- {file_id} - - {label}\n')

print("✓ Created 100 synthetic audio samples")
print("⚠️ This is for TESTING ONLY - won't produce a real model")
print("✅ But it will verify the entire pipeline works!")

# Now train on synthetic data
!python train_audio_standalone.py \
    --dataset-dir datasets \
    --model-version V0.1.0-synthetic \
    --epochs 2 \
    --batch-size 16 \
    --device cuda
```

---

## 📊 Which Option Should You Choose?

| Option | Time | Real Model? | Complexity |
|--------|------|-------------|------------|
| 1. Upload local partial | 10-20 min | Partial | Easy |
| 2. Alternative dataset | 5-10 min | Yes | Medium |
| 3. Manual download | 30-60 min | Yes | Easy |
| 4. Kaggle mirror | 10-20 min | Yes | Medium |
| 5. Synthetic data | 2 min | No (test only) | Easy |

---

## 💡 My Recommendation

**Option 5 first (Synthetic Data)** - Takes 2 minutes, proves everything works

**Then Option 1 (Upload your 38% local)** - You already have data downloaded!

---

## 🚀 Quick Start: Option 5 (Synthetic)

Copy this into Cell 2 right now:

```python
import numpy as np
import soundfile as sf
import os

print("Creating synthetic test dataset...")

os.makedirs('datasets/LA/ASVspoof2019_LA_train/flac', exist_ok=True)
os.makedirs('datasets/LA/ASVspoof2019_LA_dev/flac', exist_ok=True)
os.makedirs('datasets/LA/ASVspoof2019_LA_cm_protocols', exist_ok=True)

# Create 100 fake audio files
for i in range(100):
    audio = np.random.randn(16000 * 4).astype(np.float32) * 0.1
    split = 'train' if i < 80 else 'dev'
    file_id = f'LA_{split[0].upper()}_{i:07d}'
    path = f'datasets/LA/ASVspoof2019_LA_{split}/flac/{file_id}.flac'
    sf.write(path, audio, 16000)

# Create protocol files
with open('datasets/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt', 'w') as f:
    for i in range(80):
        label = 'bonafide' if i % 2 == 0 else 'spoof'
        f.write(f'- LA_T_{i:07d} - - {label}\n')

with open('datasets/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt', 'w') as f:
    for i in range(80, 100):
        label = 'bonafide' if i % 2 == 0 else 'spoof'
        f.write(f'- LA_D_{i:07d} - - {label}\n')

print("✓ Created 100 synthetic samples")
print("⚠️ Testing pipeline only - not a real model")

# Train
!python train_audio_standalone.py \
    --dataset-dir datasets \
    --model-version V0.1.0-test \
    --epochs 2 \
    --batch-size 16 \
    --device cuda
```

This will complete in ~5 minutes and prove everything works! 🚀
