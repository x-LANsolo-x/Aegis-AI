# 🔧 Colab Dataset Download Fix

## Problem
The ASVspoof dataset URL is returning 403 Forbidden error.

## Solution

Replace Cell 2 in Colab with this code that uses `wget` with proper headers:

```python
# Download ASVspoof 2019 LA dataset using wget with headers
import os

# Create directories
os.makedirs('datasets', exist_ok=True)

# Download using wget with user-agent (bypasses 403)
print("Downloading ASVspoof 2019 LA dataset (~7.6 GB)...")
print("This will take 10-30 minutes depending on connection speed...")

!wget --user-agent="Mozilla/5.0" \
  -O datasets/LA.zip \
  "https://datashare.ed.ac.uk/bitstream/handle/10283/3336/LA.zip"

# Extract
import zipfile
print("\nExtracting dataset...")
with zipfile.ZipFile('datasets/LA.zip', 'r') as zip_ref:
    zip_ref.extractall('datasets/')
print("✓ Dataset extracted")

# Build manifest
print("\nBuilding manifest...")
!python train_audio_standalone.py \
    --dataset-dir datasets \
    --model-version V0.1.0-test \
    --epochs 0 \
    > /dev/null 2>&1 || true

# Quick verification
import json
from pathlib import Path

manifest_path = Path("datasets/manifest.jsonl")
if manifest_path.exists():
    count = sum(1 for _ in open(manifest_path))
    print(f"✓ Manifest created with {count:,} samples")
    
    # Show sample counts
    from collections import Counter
    splits = Counter()
    with open(manifest_path) as f:
        for line in f:
            rec = json.loads(line)
            splits[rec['split']] += 1
    print(f"  Splits: {dict(splits)}")
else:
    print("⚠ Manifest not found - will be created during training")

print("\n✅ Dataset ready! Now run the training command below:")
```

Then run this in a new cell for training:

```python
# Quick test training (500 samples, 2 epochs)
!python train_audio_standalone.py \
    --dataset-dir datasets \
    --model-version V0.1.0-test \
    --epochs 2 \
    --batch-size 16 \
    --quick-test \
    --device cuda
```

## Alternative: Manual Download

If wget still fails, you can manually download:

1. **Download from browser:**
   - Go to: https://datashare.ed.ac.uk/handle/10283/3336
   - Click "LA.zip" 
   - Wait for download (~7.6 GB)

2. **Upload to Colab:**
   ```python
   from google.colab import files
   uploaded = files.upload()  # Select LA.zip from your computer
   
   # Extract
   import zipfile
   with zipfile.ZipFile('LA.zip', 'r') as zip_ref:
       zip_ref.extractall('datasets/')
   ```

3. **Then run training**

## Even Faster: Use Subset for Testing

If you just want to test the pipeline quickly, use a tiny subset:

```python
# Create a tiny fake dataset for testing
import os
import json
from pathlib import Path

os.makedirs('datasets/tiny', exist_ok=True)

# Create fake manifest with just a few samples
manifest = []
for i in range(10):
    manifest.append({
        "path": f"fake_{i}.flac",
        "label": "bonafide" if i % 2 == 0 else "spoof",
        "duration_sec": 4.0,
        "sample_rate": 16000,
        "split": "train" if i < 8 else "dev"
    })

with open('datasets/tiny/manifest.jsonl', 'w') as f:
    for rec in manifest:
        f.write(json.dumps(rec) + '\n')

print("✓ Created tiny test dataset")
print("⚠ This won't train a real model, just tests the pipeline")
```
