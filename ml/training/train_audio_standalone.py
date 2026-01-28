#!/usr/bin/env python3
"""
STANDALONE TRAINING SCRIPT - ALL-IN-ONE
Run this directly in Google Colab or any Python 3.10+ environment.

This script contains everything needed:
- Feature extraction
- Model architecture
- Training loop
- ONNX export
- No external file dependencies

Usage in Colab:
1. Upload this file
2. Run: !python train_audio_standalone.py --download-dataset
3. Wait 1-3 hours
4. Download: V1.0.0.onnx and V1.0.0.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torchaudio.transforms as T

# ============================================================================
# Logging Setup
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


# ============================================================================
# Dataset Downloader
# ============================================================================

def download_asvspoof_dataset(output_dir: Path):
    """Download ASVspoof 2019 LA dataset."""
    logger.info("Downloading ASVspoof 2019 LA dataset (~7.6 GB)...")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    zip_path = output_dir / "LA.zip"
    
    # Download
    import urllib.request
    url = "https://datashare.ed.ac.uk/bitstream/handle/10283/3336/LA.zip"
    
    def progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = min(100, downloaded * 100 / total_size)
        sys.stdout.write(f"\r  Progress: {percent:.1f}%")
        sys.stdout.flush()
    
    urllib.request.urlretrieve(url, zip_path, progress)
    print()
    
    # Extract
    logger.info("Extracting dataset...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(output_dir)
    
    logger.info("✓ Dataset downloaded and extracted")


# ============================================================================
# Manifest Builder
# ============================================================================

@dataclass
class AudioSample:
    path: str
    label: str
    duration_sec: float
    sample_rate: int
    split: str


def build_manifest(dataset_dir: Path, output_path: Path):
    """Build JSONL manifest from ASVspoof protocol files."""
    logger.info("Building manifest...")
    
    dataset_dir = dataset_dir / "LA"
    protocol_dir = dataset_dir / "ASVspoof2019_LA_cm_protocols"
    
    splits = {
        "train": protocol_dir / "ASVspoof2019.LA.cm.train.trn.txt",
        "dev": protocol_dir / "ASVspoof2019.LA.cm.dev.trl.txt",
        "eval": protocol_dir / "ASVspoof2019.LA.cm.eval.trl.txt",
    }
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as out:
        for split_name, protocol_file in splits.items():
            if not protocol_file.exists():
                logger.warning(f"Protocol file not found: {protocol_file}")
                continue
            
            with open(protocol_file) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    
                    file_id = parts[1]
                    label = parts[4]  # bonafide or spoof
                    
                    # Find audio file
                    audio_path = None
                    for try_split in ["train", "dev", "eval"]:
                        try_path = dataset_dir / f"ASVspoof2019_LA_{try_split}" / "flac" / f"{file_id}.flac"
                        if try_path.exists():
                            audio_path = try_path
                            break
                    
                    if not audio_path:
                        continue
                    
                    record = {
                        "path": str(audio_path),
                        "label": label,
                        "duration_sec": 4.0,
                        "sample_rate": 16000,
                        "split": split_name,
                    }
                    
                    out.write(json.dumps(record) + "\n")
    
    logger.info(f"✓ Manifest saved to {output_path}")


def read_manifest(manifest_path: Path) -> Iterator[AudioSample]:
    """Read manifest JSONL."""
    with open(manifest_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            yield AudioSample(
                path=rec["path"],
                label=rec["label"],
                duration_sec=rec["duration_sec"],
                sample_rate=rec["sample_rate"],
                split=rec["split"],
            )


# ============================================================================
# Feature Extraction
# ============================================================================

class AudioFeatureExtractor:
    """Extracts log-mel spectrogram features."""
    
    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 512,
        hop_length: int = 160,
        n_mels: int = 64,
        max_duration_sec: float = 10.0,
    ):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.max_duration_sec = max_duration_sec
        self.max_samples = int(sample_rate * max_duration_sec)
        
        self.mel_spec = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            normalized=True,
        )
    
    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.ndim > 1:
            waveform = waveform.mean(dim=0)
        
        if waveform.shape[0] < self.max_samples:
            waveform = F.pad(waveform, (0, self.max_samples - waveform.shape[0]))
        else:
            waveform = waveform[:self.max_samples]
        
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        
        mel_spec = self.mel_spec(waveform)
        log_mel = torch.log(mel_spec + 1e-8)
        
        return log_mel.squeeze(0)


# ============================================================================
# Dataset
# ============================================================================

class ASVspoofDataset(Dataset):
    """PyTorch dataset for ASVspoof."""
    
    def __init__(
        self,
        manifest_path: Path,
        split: str,
        feature_extractor: AudioFeatureExtractor,
        max_samples: int | None = None,
    ):
        self.split = split
        self.feature_extractor = feature_extractor
        
        self.samples = []
        for sample in read_manifest(manifest_path):
            if sample.split == split:
                self.samples.append(sample)
                if max_samples and len(self.samples) >= max_samples:
                    break
        
        logger.info(f"Loaded {len(self.samples)} samples for split={split}")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        sample = self.samples[idx]
        
        try:
            waveform, sr = torchaudio.load(sample.path)
            
            if sr != self.feature_extractor.sample_rate:
                resampler = T.Resample(sr, self.feature_extractor.sample_rate)
                waveform = resampler(waveform)
            
            features = self.feature_extractor(waveform)
            label = 0 if sample.label == "bonafide" else 1
            
            return features, label
            
        except Exception as e:
            logger.warning(f"Failed to load {sample.path}: {e}")
            dummy_features = torch.zeros(
                self.feature_extractor.n_mels,
                int(self.feature_extractor.max_samples / self.feature_extractor.hop_length) + 1
            )
            return dummy_features, 0


# ============================================================================
# Model
# ============================================================================

class LightweightCNN(nn.Module):
    """Lightweight CNN for audio deepfake detection."""
    
    def __init__(self, n_mels: int = 64, num_classes: int = 2, dropout: float = 0.3):
        super().__init__()
        
        self.conv1 = nn.Conv1d(n_mels, 128, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(128)
        self.pool1 = nn.MaxPool1d(2)
        
        self.conv2 = nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(256)
        self.pool2 = nn.MaxPool1d(2)
        
        self.conv3 = nn.Conv1d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(512)
        self.pool3 = nn.MaxPool1d(2)
        
        self.conv4 = nn.Conv1d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm1d(512)
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(512, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        
        x = F.relu(self.bn4(self.conv4(x)))
        
        x = self.global_pool(x).squeeze(-1)
        x = self.dropout(x)
        x = self.fc(x)
        
        return x


# ============================================================================
# Training
# ============================================================================

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    
    for batch_idx, (features, labels) in enumerate(dataloader):
        features = features.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        logits = model(features)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if (batch_idx + 1) % 100 == 0:
            logger.info(f"  Batch {batch_idx + 1}/{len(dataloader)}, Loss: {loss.item():.4f}")
    
    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for features, labels in dataloader:
            features = features.to(device)
            labels = labels.to(device)
            
            logits = model(features)
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    return total_loss / len(dataloader), correct / total if total > 0 else 0.0


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--download-dataset", action="store_true", help="Download ASVspoof dataset")
    parser.add_argument("--dataset-dir", type=Path, default=Path("datasets"), help="Dataset directory")
    parser.add_argument("--output-dir", type=Path, default=Path("models"), help="Output directory")
    parser.add_argument("--model-version", type=str, default="V1.0.0", help="Model version")
    parser.add_argument("--epochs", type=int, default=20, help="Epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--quick-test", action="store_true", help="Quick test with 500 samples")
    
    args = parser.parse_args()
    
    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    
    # Download dataset if requested
    if args.download_dataset:
        download_asvspoof_dataset(args.dataset_dir)
    
    # Build manifest
    manifest_path = args.dataset_dir / "manifest.jsonl"
    if not manifest_path.exists():
        build_manifest(args.dataset_dir, manifest_path)
    
    # Setup
    args.output_dir.mkdir(parents=True, exist_ok=True)
    extractor = AudioFeatureExtractor()
    time_frames = int(extractor.max_samples / extractor.hop_length) + 1
    
    # Datasets
    max_train = 500 if args.quick_test else None
    max_val = 100 if args.quick_test else None
    
    train_dataset = ASVspoofDataset(manifest_path, "train", extractor, max_train)
    val_dataset = ASVspoofDataset(manifest_path, "dev", extractor, max_val)
    
    if len(train_dataset) == 0:
        logger.error("No training samples!")
        return 1
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    # Model
    model = LightweightCNN().to(device)
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {num_params:,}")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    
    # Training loop
    best_val_loss = float("inf")
    patience_counter = 0
    
    for epoch in range(1, args.epochs + 1):
        logger.info(f"Epoch {epoch}/{args.epochs}")
        
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        logger.info(f"  Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
            }, args.output_dir / f"{args.model_version}_best.pt")
            logger.info("  ✓ Saved best model")
        else:
            patience_counter += 1
            if patience_counter >= 5:
                logger.info("Early stopping")
                break
    
    # Load best and export ONNX
    checkpoint = torch.load(args.output_dir / f"{args.model_version}_best.pt")
    model.load_state_dict(checkpoint["model_state_dict"])
    
    onnx_path = args.output_dir / f"{args.model_version}.onnx"
    model.eval()
    dummy_input = torch.randn(1, 64, time_frames, device=device)
    
    torch.onnx.export(
        model, dummy_input, onnx_path,
        export_params=True, opset_version=14,
        input_names=["audio_features"], output_names=["logits"],
        dynamic_axes={"audio_features": {0: "batch_size"}, "logits": {0: "batch_size"}},
    )
    
    logger.info(f"✓ ONNX model saved: {onnx_path}")
    
    # Save metadata
    metadata = {
        "model_version": args.model_version,
        "val_accuracy": float(checkpoint["val_acc"]),
        "num_parameters": num_params,
    }
    
    with open(args.output_dir / f"{args.model_version}.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    logger.info("=" * 70)
    logger.info("✓ TRAINING COMPLETE!")
    logger.info(f"  ONNX: {onnx_path}")
    logger.info(f"  Accuracy: {checkpoint['val_acc']:.1%}")
    logger.info("=" * 70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
