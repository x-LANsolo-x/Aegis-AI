#!/usr/bin/env python3
"""
Training script for video deepfake detection.

Usage:
    python ml/training/train_video.py --data-dir path/to/faceforensics --model xception --epochs 50
"""

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ml.training.models.video_detector import create_video_detector, count_parameters
from ml.datasets.video_dataset import create_video_dataset, get_dataset_stats
from ml.training.logging_config import configure_logging, log_run_header

logger = logging.getLogger("ml.train_video")


# ============================================================================
# Training Functions
# ============================================================================

def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    for batch_idx, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Collect predictions
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        if (batch_idx + 1) % 50 == 0:
            logger.info(f"  Epoch {epoch}, Batch {batch_idx + 1}/{len(dataloader)}, Loss: {loss.item():.4f}")
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    
    return avg_loss, accuracy


def validate(model, dataloader, criterion, device):
    """Validate model."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            
            # Get predictions and probabilities
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of fake
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    
    # AUC-ROC
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except:
        auc = 0.0
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    return avg_loss, accuracy, auc, cm


# ============================================================================
# ONNX Export
# ============================================================================

def export_to_onnx(model, output_path, image_size, device):
    """Export model to ONNX format."""
    logger.info(f"Exporting model to ONNX: {output_path}")
    
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, image_size, image_size, device=device)
    
    # Export
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={
            "image": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
    )
    
    logger.info(f"✓ ONNX model saved to {output_path}")


def verify_onnx_export(pytorch_model, onnx_path, image_size, device):
    """Verify ONNX export matches PyTorch output."""
    logger.info("Verifying ONNX export...")
    
    try:
        import onnxruntime as ort
        
        # Create test input
        test_input = torch.randn(1, 3, image_size, image_size, device=device)
        
        # PyTorch inference
        pytorch_model.eval()
        with torch.no_grad():
            pytorch_output = pytorch_model(test_input).cpu().numpy()
        
        # ONNX inference
        ort_session = ort.InferenceSession(str(onnx_path))
        onnx_input = test_input.cpu().numpy()
        onnx_output = ort_session.run(None, {"image": onnx_input})[0]
        
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
    parser = argparse.ArgumentParser(description="Train video deepfake detector")
    
    # Data
    parser.add_argument("--data-dir", type=Path, required=True, help="Path to dataset")
    parser.add_argument("--dataset-type", type=str, default="frames", 
                       choices=["faceforensics", "frames"],
                       help="Dataset type")
    
    # Model
    parser.add_argument("--model", type=str, default="xception",
                       choices=["xception", "lightweight"],
                       help="Model architecture")
    parser.add_argument("--image-size", type=int, default=299, help="Image size")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout rate")
    
    # Training
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="Weight decay")
    
    # Other
    parser.add_argument("--output-dir", type=Path, default=Path("models/video"), help="Output directory")
    parser.add_argument("--model-version", type=str, default="V1.0.0", help="Model version")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    
    args = parser.parse_args()
    
    # Setup logging
    configure_logging()
    log_run_header(
        seed=args.seed,
        dataset_version=args.dataset_type,
        run_name=f"train_video_{args.model}_{args.model_version}",
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
    
    # Load datasets
    logger.info("Loading datasets...")
    train_dataset = create_video_dataset(
        dataset_type=args.dataset_type,
        data_dir=args.data_dir,
        split="train",
        image_size=args.image_size,
        augment=True
    )
    
    val_dataset = create_video_dataset(
        dataset_type=args.dataset_type,
        data_dir=args.data_dir,
        split="val",
        image_size=args.image_size,
        augment=False
    )
    
    if len(train_dataset) == 0:
        logger.error("No training samples found!")
        return 1
    
    logger.info(f"Train dataset: {get_dataset_stats(train_dataset)}")
    logger.info(f"Val dataset: {get_dataset_stats(val_dataset)}")
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if device.type == "cuda" else False,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if device.type == "cuda" else False,
    )
    
    # Create model
    logger.info(f"Creating model: {args.model}")
    model = create_video_detector(
        architecture=args.model,
        num_classes=2,
        dropout=args.dropout
    )
    model = model.to(device)
    
    num_params = count_parameters(model)
    logger.info(f"Model parameters: {num_params:,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=True
    )
    
    # Training loop
    logger.info("Starting training...")
    best_val_loss = float("inf")
    best_val_acc = 0.0
    patience = 10
    patience_counter = 0
    
    training_history = []
    
    for epoch in range(1, args.epochs + 1):
        logger.info(f"Epoch {epoch}/{args.epochs}")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        logger.info(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        
        # Validate
        val_loss, val_acc, val_auc, cm = validate(model, val_loader, criterion, device)
        logger.info(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val AUC: {val_auc:.4f}")
        logger.info(f"  Confusion Matrix:\n{cm}")
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Record history
        training_history.append({
            "epoch": epoch,
            "train_loss": float(train_loss),
            "train_acc": float(train_acc),
            "val_loss": float(val_loss),
            "val_acc": float(val_acc),
            "val_auc": float(val_auc),
        })
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            patience_counter = 0
            
            # Save best model
            checkpoint_path = args.output_dir / f"{args.model_version}_best.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_auc": val_auc,
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
    logger.info(f"Best model: epoch={checkpoint['epoch']}, val_loss={checkpoint['val_loss']:.4f}, "
                f"val_acc={checkpoint['val_acc']:.4f}, val_auc={checkpoint['val_auc']:.4f}")
    
    # Export to ONNX
    onnx_path = args.output_dir / f"{args.model_version}.onnx"
    export_to_onnx(model, onnx_path, args.image_size, device)
    
    # Verify export (if onnxruntime available)
    try:
        verify_onnx_export(model, onnx_path, args.image_size, device)
    except:
        logger.warning("Could not verify ONNX export (onnxruntime not available)")
    
    # Save model metadata
    metadata = {
        "model_version": args.model_version,
        "model_type": args.model,
        "num_parameters": num_params,
        "image_size": args.image_size,
        "num_classes": 2,
        "best_val_loss": float(checkpoint["val_loss"]),
        "best_val_accuracy": float(checkpoint["val_acc"]),
        "best_val_auc": float(checkpoint["val_auc"]),
        "training_epochs": checkpoint["epoch"],
        "dataset_type": args.dataset_type,
        "training_history": training_history,
        "timestamp": datetime.utcnow().isoformat(),
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
    logger.info(f"Best validation AUC: {checkpoint['val_auc']:.4f}")
    logger.info("=" * 88)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
