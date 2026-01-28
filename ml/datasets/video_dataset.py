"""
Video dataset loaders for deepfake detection.

Supports:
- FaceForensics++ (primary dataset)
- Celeb-DF (high-quality deepfakes)
- Custom video datasets
"""

import json
import random
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import logging

import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
from torchvision import transforms

logger = logging.getLogger(__name__)


# ============================================================================
# Data Augmentation
# ============================================================================

def get_training_transforms(image_size: int = 299):
    """Get training data augmentation transforms."""
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def get_validation_transforms(image_size: int = 299):
    """Get validation transforms (no augmentation)."""
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


# ============================================================================
# FaceForensics++ Dataset
# ============================================================================

class FaceForensicsDataset(Dataset):
    """
    FaceForensics++ dataset loader.
    
    Dataset structure:
        FaceForensics++/
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
    """
    
    def __init__(
        self,
        root_dir: Path,
        split: str = "train",
        compression: str = "c23",
        manipulation_methods: List[str] = None,
        frames_per_video: int = 10,
        image_size: int = 299,
        augment: bool = True
    ):
        """
        Args:
            root_dir: Path to FaceForensics++ root directory
            split: "train", "val", or "test"
            compression: "raw", "c23", or "c40"
            manipulation_methods: List of methods or None for all
            frames_per_video: Number of frames to sample per video
            image_size: Target image size
            augment: Whether to apply data augmentation
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.compression = compression
        self.frames_per_video = frames_per_video
        self.image_size = image_size
        
        if manipulation_methods is None:
            self.manipulation_methods = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]
        else:
            self.manipulation_methods = manipulation_methods
        
        # Transforms
        if augment and split == "train":
            self.transform = get_training_transforms(image_size)
        else:
            self.transform = get_validation_transforms(image_size)
        
        # Load video paths
        self.samples = self._load_samples()
        
        logger.info(f"Loaded {len(self.samples)} videos for split={split}")
    
    def _load_samples(self) -> List[Tuple[Path, int]]:
        """Load video paths and labels."""
        samples = []
        
        # Load split information
        splits_file = self.root_dir / f"{self.split}.json"
        if splits_file.exists():
            with open(splits_file) as f:
                split_videos = set(json.load(f))
        else:
            # If no split file, use all videos
            split_videos = None
        
        # Real videos (label = 0)
        real_dir = self.root_dir / "original_sequences" / "youtube" / self.compression / "videos"
        if real_dir.exists():
            for video_path in real_dir.glob("*.mp4"):
                if split_videos is None or video_path.stem in split_videos:
                    samples.append((video_path, 0))
        
        # Fake videos (label = 1)
        for method in self.manipulation_methods:
            fake_dir = self.root_dir / "manipulated_sequences" / method / self.compression / "videos"
            if fake_dir.exists():
                for video_path in fake_dir.glob("*.mp4"):
                    if split_videos is None or video_path.stem in split_videos:
                        samples.append((video_path, 1))
        
        # Shuffle
        random.shuffle(samples)
        
        return samples
    
    def _extract_frames(self, video_path: Path) -> List[np.ndarray]:
        """Extract frames from video."""
        frames = []
        
        try:
            cap = cv2.VideoCapture(str(video_path))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if total_frames == 0:
                logger.warning(f"Video has 0 frames: {video_path}")
                cap.release()
                return frames
            
            # Sample frames uniformly
            frame_indices = np.linspace(0, total_frames - 1, self.frames_per_video, dtype=int)
            
            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if ret:
                    # Convert BGR to RGB
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame)
            
            cap.release()
            
        except Exception as e:
            logger.error(f"Error extracting frames from {video_path}: {e}")
        
        return frames
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        video_path, label = self.samples[idx]
        
        # Extract frames
        frames = self._extract_frames(video_path)
        
        if not frames:
            # Return black image if extraction failed
            dummy_frame = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
            return self.transform(dummy_frame), label
        
        # Pick one random frame
        frame = random.choice(frames)
        
        # Apply transforms
        frame_tensor = self.transform(frame)
        
        return frame_tensor, label


# ============================================================================
# Simple Video Frames Dataset (from extracted frames)
# ============================================================================

class VideoFramesDataset(Dataset):
    """
    Dataset loader for pre-extracted video frames.
    
    Expected structure:
        frames/
        ├── real/
        │   ├── video1_frame0.jpg
        │   ├── video1_frame1.jpg
        │   └── ...
        └── fake/
            ├── video2_frame0.jpg
            ├── video2_frame1.jpg
            └── ...
    """
    
    def __init__(
        self,
        frames_dir: Path,
        split: str = "train",
        image_size: int = 299,
        augment: bool = True
    ):
        """
        Args:
            frames_dir: Path to frames directory
            split: "train" or "val"
            image_size: Target image size
            augment: Whether to augment
        """
        self.frames_dir = Path(frames_dir)
        self.split = split
        self.image_size = image_size
        
        # Transforms
        if augment and split == "train":
            self.transform = get_training_transforms(image_size)
        else:
            self.transform = get_validation_transforms(image_size)
        
        # Load frame paths
        self.samples = self._load_samples()
        
        logger.info(f"Loaded {len(self.samples)} frames for split={split}")
    
    def _load_samples(self) -> List[Tuple[Path, int]]:
        """Load frame paths and labels."""
        samples = []
        
        # Real frames (label = 0)
        real_dir = self.frames_dir / "real"
        if real_dir.exists():
            for img_path in real_dir.glob("*.jpg"):
                samples.append((img_path, 0))
            for img_path in real_dir.glob("*.png"):
                samples.append((img_path, 0))
        
        # Fake frames (label = 1)
        fake_dir = self.frames_dir / "fake"
        if fake_dir.exists():
            for img_path in fake_dir.glob("*.jpg"):
                samples.append((img_path, 1))
            for img_path in fake_dir.glob("*.png"):
                samples.append((img_path, 1))
        
        # Shuffle
        random.shuffle(samples)
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        
        # Load image
        image = cv2.imread(str(img_path))
        if image is None:
            # Return black image if loading failed
            image = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        image_tensor = self.transform(image)
        
        return image_tensor, label


# ============================================================================
# Dataset Factory
# ============================================================================

def create_video_dataset(
    dataset_type: str,
    data_dir: Path,
    split: str = "train",
    image_size: int = 299,
    augment: bool = True,
    **kwargs
) -> Dataset:
    """
    Create a video dataset.
    
    Args:
        dataset_type: "faceforensics" or "frames"
        data_dir: Path to dataset
        split: "train", "val", or "test"
        image_size: Target image size
        augment: Whether to augment
        **kwargs: Additional dataset-specific arguments
    
    Returns:
        PyTorch Dataset
    """
    
    if dataset_type == "faceforensics":
        return FaceForensicsDataset(
            root_dir=data_dir,
            split=split,
            image_size=image_size,
            augment=augment,
            **kwargs
        )
    elif dataset_type == "frames":
        return VideoFramesDataset(
            frames_dir=data_dir,
            split=split,
            image_size=image_size,
            augment=augment
        )
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")


# ============================================================================
# Utility Functions
# ============================================================================

def get_dataset_stats(dataset: Dataset) -> Dict:
    """Get dataset statistics."""
    labels = [label for _, label in dataset.samples]
    
    num_real = labels.count(0)
    num_fake = labels.count(1)
    
    return {
        "total_samples": len(labels),
        "real_samples": num_real,
        "fake_samples": num_fake,
        "real_ratio": num_real / len(labels) if labels else 0,
        "fake_ratio": num_fake / len(labels) if labels else 0,
    }


if __name__ == "__main__":
    # Test dataset loading
    print("Testing VideoFramesDataset (dummy)...")
    
    # Create dummy dataset structure
    import tempfile
    import os
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create structure
        (tmpdir / "real").mkdir()
        (tmpdir / "fake").mkdir()
        
        # Create dummy images
        for i in range(5):
            dummy_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            cv2.imwrite(str(tmpdir / "real" / f"frame_{i}.jpg"), dummy_img)
            cv2.imwrite(str(tmpdir / "fake" / f"frame_{i}.jpg"), dummy_img)
        
        # Load dataset
        dataset = create_video_dataset(
            dataset_type="frames",
            data_dir=tmpdir,
            split="train",
            image_size=299
        )
        
        print(f"  Loaded {len(dataset)} samples")
        print(f"  Stats: {get_dataset_stats(dataset)}")
        
        # Get one sample
        img, label = dataset[0]
        print(f"  Sample shape: {img.shape}, label: {label}")
        
        print("\n✓ Dataset loading works!")
