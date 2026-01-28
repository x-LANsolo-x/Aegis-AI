# Video Deepfake Dataset Download Guide

This guide explains how to download and prepare datasets for training video deepfake detection models.

---

## FaceForensics++ Dataset (Recommended)

**Overview:**
- 1,000 original videos from YouTube
- 4,000 manipulated videos (4 manipulation methods)
- High quality and widely used benchmark

### Download Instructions

#### 1. Request Access

FaceForensics++ requires academic access:

1. Go to: https://github.com/ondyari/FaceForensics
2. Fill out the request form
3. Wait for approval email (usually 1-3 days)

#### 2. Download Script

Once approved, you'll receive download credentials. Use the official download script:

```bash
# Clone the FaceForensics repository
git clone https://github.com/ondyari/FaceForensics.git
cd FaceForensics

# Download videos (c23 compression - balanced quality/size)
python download-FaceForensics.py \
    -d all \
    -c c23 \
    -t videos \
    --server EU2

# This will download:
# - Original videos (~100 GB)
# - Deepfakes manipulations (~100 GB)
# - Face2Face manipulations (~100 GB)
# - FaceSwap manipulations (~100 GB)
# - NeuralTextures manipulations (~100 GB)
# Total: ~500 GB
```

**Compression Options:**
- `raw`: Original quality (~1.5 TB)
- `c23`: High quality, H.264 (CRF 23) - **Recommended** (~500 GB)
- `c40`: Lower quality, H.264 (CRF 40) (~100 GB)

#### 3. Extract and Organize

After download, organize the structure:

```
data/FaceForensics++/
├── original_sequences/
│   └── youtube/
│       └── c23/
│           └── videos/
│               ├── 000.mp4
│               ├── 001.mp4
│               └── ...
└── manipulated_sequences/
    ├── Deepfakes/
    │   └── c23/
    │       └── videos/
    ├── Face2Face/
    ├── FaceSwap/
    └── NeuralTextures/
```

#### 4. Create Split Files

Create train/val/test split JSON files:

```python
# create_splits.py
import json
from pathlib import Path

# Official splits from FaceForensics++
train_videos = list(range(0, 720))
val_videos = list(range(720, 840))
test_videos = list(range(840, 1000))

# Convert to string IDs
train_ids = [f"{i:03d}" for i in train_videos]
val_ids = [f"{i:03d}" for i in val_videos]
test_ids = [f"{i:03d}" for i in test_videos]

# Save
data_dir = Path("data/FaceForensics++")
with open(data_dir / "train.json", "w") as f:
    json.dump(train_ids, f)

with open(data_dir / "val.json", "w") as f:
    json.dump(val_ids, f)

with open(data_dir / "test.json", "w") as f:
    json.dump(test_ids, f)

print("✓ Split files created")
```

---

## Celeb-DF Dataset (Alternative)

**Overview:**
- 590 real celebrity videos
- 5,639 high-quality deepfakes
- More challenging than FaceForensics++

### Download

```bash
# Visit: https://github.com/yuezunli/celeb-deepfakeforensics
# Download links provided in README

# Structure:
data/Celeb-DF/
├── Celeb-real/
│   ├── id0_0000.mp4
│   └── ...
├── Celeb-synthesis/
│   ├── id0_id1_0000.mp4
│   └── ...
└── YouTube-real/
    ├── 00001.mp4
    └── ...
```

---

## Pre-Extracted Frames (Fast Alternative)

If you have limited storage or want faster training, use pre-extracted frames:

### Structure

```
data/frames/
├── real/
│   ├── video000_frame00.jpg
│   ├── video000_frame01.jpg
│   └── ...
└── fake/
    ├── video001_frame00.jpg
    ├── video001_frame01.jpg
    └── ...
```

### Extract Frames Script

```python
# extract_frames.py
import cv2
from pathlib import Path
from tqdm import tqdm

def extract_frames(video_path, output_dir, max_frames=10):
    """Extract frames from video."""
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        return
    
    # Sample frames uniformly
    frame_indices = list(range(0, total_frames, total_frames // max_frames))[:max_frames]
    
    for i, frame_idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if ret:
            output_path = output_dir / f"{video_path.stem}_frame{i:02d}.jpg"
            cv2.imwrite(str(output_path), frame)
    
    cap.release()

# Usage
video_dir = Path("data/FaceForensics++/original_sequences/youtube/c23/videos")
output_dir = Path("data/frames/real")
output_dir.mkdir(parents=True, exist_ok=True)

for video_path in tqdm(list(video_dir.glob("*.mp4"))):
    extract_frames(video_path, output_dir, max_frames=10)
```

---

## Synthetic Test Dataset (For Quick Testing)

Create a small synthetic dataset for testing:

```python
# create_synthetic_dataset.py
import numpy as np
import cv2
from pathlib import Path

output_dir = Path("data/synthetic")
(output_dir / "real").mkdir(parents=True, exist_ok=True)
(output_dir / "fake").mkdir(parents=True, exist_ok=True)

# Create 100 real and 100 fake images
for i in range(100):
    # Real images (random smooth patterns)
    real_img = np.random.randint(100, 200, (224, 224, 3), dtype=np.uint8)
    cv2.imwrite(str(output_dir / "real" / f"img_{i:03d}.jpg"), real_img)
    
    # Fake images (random noisy patterns)
    fake_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    cv2.imwrite(str(output_dir / "fake" / f"img_{i:03d}.jpg"), fake_img)

print("✓ Created 200 synthetic samples")
```

---

## Dataset Statistics

### FaceForensics++

| Split | Real | Fake | Total |
|-------|------|------|-------|
| Train | 720 | 2,880 | 3,600 |
| Val | 120 | 480 | 600 |
| Test | 160 | 640 | 800 |

**Manipulation Methods:**
- Deepfakes (face swap with autoencoders)
- Face2Face (facial reenactment)
- FaceSwap (classic face swap)
- NeuralTextures (facial expression transfer)

### Celeb-DF

| Split | Real | Fake | Total |
|-------|------|------|-------|
| Train | 408 | 3,947 | 4,355 |
| Val | 59 | 564 | 623 |
| Test | 123 | 1,128 | 1,251 |

---

## Verification

After downloading, verify the dataset:

```python
# verify_dataset.py
from pathlib import Path

data_dir = Path("data/FaceForensics++")

# Check original videos
real_dir = data_dir / "original_sequences/youtube/c23/videos"
num_real = len(list(real_dir.glob("*.mp4")))
print(f"Real videos: {num_real} (expected: 1000)")

# Check manipulated videos
for method in ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]:
    fake_dir = data_dir / "manipulated_sequences" / method / "c23/videos"
    num_fake = len(list(fake_dir.glob("*.mp4")))
    print(f"{method}: {num_fake} (expected: 1000)")
```

---

## Troubleshooting

### Download Fails
- Try different server: `--server EU` or `--server US`
- Use smaller compression: `-c c40`
- Download in parts: `-d Deepfakes` (one method at a time)

### Out of Disk Space
- Use c40 compression (~100 GB)
- Download only Deepfakes method
- Use pre-extracted frames

### Slow Training
- Extract frames to disk (avoid video decoding during training)
- Use smaller image size (224x224 instead of 299x299)
- Reduce frames_per_video parameter

---

## Next Steps

After downloading the dataset:

1. **Quick test:**
   ```bash
   python ml/training/train_video.py \
       --data-dir data/synthetic \
       --dataset-type frames \
       --model lightweight \
       --epochs 5 \
       --batch-size 16
   ```

2. **Full training:**
   ```bash
   python ml/training/train_video.py \
       --data-dir data/FaceForensics++ \
       --dataset-type faceforensics \
       --model xception \
       --epochs 50 \
       --batch-size 32
   ```

---

## References

- FaceForensics++: https://github.com/ondyari/FaceForensics
- Celeb-DF: https://github.com/yuezunli/celeb-deepfakeforensics
- Paper: https://arxiv.org/abs/1901.08971
