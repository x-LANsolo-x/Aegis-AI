#!/usr/bin/env python3
"""Training script for audio deepfake detection.

This script:
1. Loads ASVspoof manifest
2. Extracts log-mel spectrogram features
3. Trains a lightweight CNN
4. Exports to ONNX format

Usage:
    python ml/training/train_audio.py --manifest path/to/manifest.jsonl --output-dir models/audio/
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torchaudio.transforms as T

from ml.datasets.loader import read_manifest, AudioSample
from ml.training.logging_config import configure_logging, log_run_header


logger = logging.getLogger("ml.train_audio")


# ============================================================================
# Feature Extraction
# ============================================================================

class AudioFeatureExtractor:
    """Extracts log-mel spectrogram features from audio."""
    
    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 512,
        hop_length: int = 160,  # 10ms at 16kHz
        n_mels: int = 64,
        max_duration_sec: float = 10.0,
    ):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.max_duration_sec = max_duration_sec
        self.max_samples = int(sample_rate * max_duration_sec)
        
        # MelSpectrogram transform
        self.mel_spec = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            normalized=True,
        )
    
    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        """Extract log-mel spectrogram from waveform.
        
        Args:
            waveform: (channels, samples) or (samples,)
        
        Returns:
            log_mel: (n_mels, time_frames)
        """
        # Ensure mono
        if waveform.ndim > 1:
            waveform = waveform.mean(dim=0)
        
        # Pad or crop to max_duration
        if waveform.shape[0] < self.max_samples:
            waveform = F.pad(waveform, (0, self.max_samples - waveform.shape[0]))
        else:
            waveform = waveform[:self.max_samples]
        
        # Add channel dimension for MelSpectrogram
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        
        # Compute mel spectrogram
        mel_spec = self.mel_spec(waveform)  # (1, n_mels, time)
        
        # Log scale (add small epsilon to avoid log(0))
        log_mel = torch.log(mel_spec + 1e-8)
        
        # Remove batch dimension
        log_mel = log_mel.squeeze(0)  # (n_mels, time)
        
        return log_mel


# ============================================================================
# Dataset
# ============================================================================

class ASVspoofDataset(Dataset):
    """PyTorch dataset for ASVspoof audio samples."""
    
    def __init__(
        self,
        manifest_path: Path,
        split: str,
        feature_extractor: AudioFeatureExtractor,
        max_samples: int | None = None,
    ):
        self.split = split
        self.feature_extractor = feature_extractor
        
        # Load samples from manifest
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
        
        # Load audio
        try:
            waveform, sr = torchaudio.load(sample.path)
            
            # Resample if needed
            if sr != self.feature_extractor.sample_rate:
                resampler = T.Resample(sr, self.feature_extractor.sample_rate)
                waveform = resampler(waveform)
            
            # Extract features
            features = self.feature_extractor(waveform)
            
            # Label: bonafide=0, spoof=1
            label = 0 if sample.label == "bonafide" else 1
            
            return features, label
            
        except Exception as e:
            logger.warning(f"Failed to load {sample.path}: {e}")
            # Return zeros with correct shape
            dummy_features = torch.zeros(
                self.feature_extractor.n_mels,
                int(self.feature_extractor.max_samples / self.feature_extractor.hop_length) + 1
            )
            return dummy_features, 0


# ============================================================================
# Model Architecture
# ============================================================================

class LightweightCNN(nn.Module):
    """Lightweight CNN for audio deepfake detection.
    
    Architecture inspired by M5/M11 networks for audio classification.
    Uses 1D convolutions over time frames of mel-spectrograms.
    """
    
    def __init__(
        self,
        n_mels: int = 64,
        num_classes: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        
        self.n_mels = n_mels
        self.num_classes = num_classes
        
        # Input: (batch, n_mels, time_frames)
        # Treat n_mels as channels, time_frames as sequence length
        
        # Conv blocks (over time dimension)
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
        
        # Global average pooling over time
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Classifier
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(512, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: (batch, n_mels, time_frames)
        
        Returns:
            logits: (batch, num_classes)
        """
        # Conv block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        
        # Conv block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        
        # Conv block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool3(x)
        
        # Conv block 4
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        
        # Global pooling
        x = self.global_pool(x)  # (batch, 512, 1)
        x = x.squeeze(-1)  # (batch, 512)
        
        # Classifier
        x = self.dropout(x)
        x = self.fc(x)  # (batch, num_classes)
        
        return x


# ============================================================================
# Training Loop
# ============================================================================

def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Train for one epoch."""
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


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Validate model."""
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
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total if total > 0 else 0.0
    
    return avg_loss, accuracy


# ============================================================================
# ONNX Export
# ============================================================================

def export_to_onnx(
    model: nn.Module,
    output_path: Path,
    n_mels: int,
    time_frames: int,
    device: torch.device,
) -> None:
    """Export trained model to ONNX format."""
    logger.info(f"Exporting model to ONNX: {output_path}")
    
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, n_mels, time_frames, device=device)
    
    # Export
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["audio_features"],
        output_names=["logits"],
        dynamic_axes={
            "audio_features": {0: "batch_size", 2: "time_frames"},
            "logits": {0: "batch_size"},
        },
    )
    
    logger.info(f"ONNX model saved to {output_path}")


def verify_onnx_export(
    pytorch_model: nn.Module,
    onnx_path: Path,
    n_mels: int,
    time_frames: int,
    device: torch.device,
) -> bool:
    """Verify ONNX export matches PyTorch output."""
    logger.info("Verifying ONNX export...")
    
    try:
        import onnxruntime as ort
        
        # Create test input
        test_input = torch.randn(1, n_mels, time_frames, device=device)
        
        # PyTorch inference
        pytorch_model.eval()
        with torch.no_grad():
            pytorch_output = pytorch_model(test_input).cpu().numpy()
        
        # ONNX inference
        ort_session = ort.InferenceSession(str(onnx_path))
        onnx_input = test_input.cpu().numpy()
        onnx_output = ort_session.run(None, {"audio_features": onnx_input})[0]
        
        # Compare
        max_diff = np.abs(pytorch_output - onnx_output).max()
        logger.info(f"Max difference between PyTorch and ONNX: {max_diff:.6f}")
        
        if max_diff < 1e-4:
            logger.info("✓ ONNX export verification passed")
            return True
        else:
            logger.warning(f"✗ ONNX export verification failed: max_diff={max_diff}")
            return False
            
    except Exception as e:
        logger.error(f"ONNX verification failed: {e}")
        return False


# ============================================================================
# Main Training Function
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train audio deepfake detector")
    parser.add_argument("--manifest", type=Path, required=True, help="Path to manifest.jsonl")
    parser.add_argument("--output-dir", type=Path, default=Path("models/audio"), help="Output directory")
    parser.add_argument("--model-version", type=str, default="V1.0.0", help="Model version (e.g., V1.0.0)")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max-train-samples", type=int, default=None, help="Max training samples (for testing)")
    parser.add_argument("--max-val-samples", type=int, default=None, help="Max validation samples (for testing)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
    
    args = parser.parse_args()
    
    # Setup logging
    configure_logging()
    log_run_header(
        seed=args.seed,
        dataset_version=args.manifest.name,
        run_name=f"train_audio_{args.model_version}",
    )
    
    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Device
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Feature extractor
    feature_extractor = AudioFeatureExtractor(
        sample_rate=16000,
        n_fft=512,
        hop_length=160,
        n_mels=64,
        max_duration_sec=10.0,
    )
    
    # Calculate time frames for fixed-length audio
    time_frames = int(feature_extractor.max_samples / feature_extractor.hop_length) + 1
    logger.info(f"Feature shape: ({feature_extractor.n_mels}, {time_frames})")
    
    # Datasets
    logger.info("Loading datasets...")
    train_dataset = ASVspoofDataset(
        args.manifest,
        split="train",
        feature_extractor=feature_extractor,
        max_samples=args.max_train_samples,
    )
    
    val_dataset = ASVspoofDataset(
        args.manifest,
        split="dev",
        feature_extractor=feature_extractor,
        max_samples=args.max_val_samples,
    )
    
    if len(train_dataset) == 0:
        logger.error("No training samples found!")
        sys.exit(1)
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True if device.type == "cuda" else False,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True if device.type == "cuda" else False,
    )
    
    # Model
    logger.info("Creating model...")
    model = LightweightCNN(
        n_mels=feature_extractor.n_mels,
        num_classes=2,
        dropout=0.3,
    )
    model = model.to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {num_params:,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, verbose=True
    )
    
    # Training loop
    logger.info("Starting training...")
    best_val_loss = float("inf")
    patience = 5
    patience_counter = 0
    
    for epoch in range(1, args.epochs + 1):
        logger.info(f"Epoch {epoch}/{args.epochs}")
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        logger.info(f"  Train Loss: {train_loss:.4f}")
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        logger.info(f"  Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}")
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            # Save best model
            checkpoint_path = args.output_dir / f"{args.model_version}_best.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
            }, checkpoint_path)
            logger.info(f"  ✓ Saved best model to {checkpoint_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered after {epoch} epochs")
                break
    
    # Load best model for export
    logger.info("Loading best model for export...")
    checkpoint = torch.load(args.output_dir / f"{args.model_version}_best.pt")
    model.load_state_dict(checkpoint["model_state_dict"])
    logger.info(f"Best model: epoch={checkpoint['epoch']}, val_loss={checkpoint['val_loss']:.4f}, val_acc={checkpoint['val_acc']:.4f}")
    
    # Export to ONNX
    onnx_path = args.output_dir / f"{args.model_version}.onnx"
    export_to_onnx(model, onnx_path, feature_extractor.n_mels, time_frames, device)
    
    # Verify export
    verify_onnx_export(model, onnx_path, feature_extractor.n_mels, time_frames, device)
    
    # Save model metadata
    metadata = {
        "model_version": args.model_version,
        "model_type": "LightweightCNN",
        "num_parameters": num_params,
        "sample_rate": feature_extractor.sample_rate,
        "n_mels": feature_extractor.n_mels,
        "n_fft": feature_extractor.n_fft,
        "hop_length": feature_extractor.hop_length,
        "max_duration_sec": feature_extractor.max_duration_sec,
        "num_classes": 2,
        "best_val_loss": float(checkpoint["val_loss"]),
        "best_val_accuracy": float(checkpoint["val_acc"]),
        "training_epochs": checkpoint["epoch"],
        "manifest": str(args.manifest),
    }
    
    metadata_path = args.output_dir / f"{args.model_version}.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved model metadata to {metadata_path}")
    
    logger.info("=" * 88)
    logger.info("Training complete!")
    logger.info(f"ONNX model: {onnx_path}")
    logger.info(f"Metadata: {metadata_path}")
    logger.info(f"Best validation accuracy: {checkpoint['val_acc']:.4f}")
    logger.info("=" * 88)


if __name__ == "__main__":
    main()
