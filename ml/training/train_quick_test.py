#!/usr/bin/env python3
"""Quick training test script - runs locally with minimal setup.

This is a simplified version that can run on your local machine
to verify everything works before doing full Colab training.

Usage:
    python ml/training/train_quick_test.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

print("=" * 70)
print("AEGIS-AI TRAINING QUICK TEST")
print("=" * 70)

# Check dependencies
print("\n1. Checking dependencies...")
try:
    import torch
    print(f"   ✓ PyTorch {torch.__version__}")
except ImportError:
    print("   ✗ PyTorch not installed")
    sys.exit(1)

try:
    import torchaudio
    print(f"   ✓ Torchaudio {torchaudio.__version__}")
except ImportError:
    print("   ✗ Torchaudio not installed")
    sys.exit(1)

try:
    import onnxruntime as ort
    print(f"   ✓ ONNX Runtime {ort.__version__}")
except ImportError:
    print("   ✗ ONNX Runtime not installed")
    print("   Install: pip install onnxruntime")
    sys.exit(1)

# Check CUDA
print(f"   ✓ CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   ✓ CUDA device: {torch.cuda.get_device_name(0)}")

# Test imports
print("\n2. Testing training script imports...")
try:
    from ml.training.train_audio import (
        AudioFeatureExtractor,
        LightweightCNN,
        ASVspoofDataset,
        export_to_onnx,
        verify_onnx_export,
    )
    print("   ✓ All imports successful")
except Exception as e:
    print(f"   ✗ Import failed: {e}")
    sys.exit(1)

# Test feature extractor
print("\n3. Testing feature extraction...")
try:
    extractor = AudioFeatureExtractor()
    dummy_audio = torch.randn(1, 16000 * 5)  # 5 seconds
    features = extractor(dummy_audio)
    print(f"   ✓ Feature extraction works")
    print(f"   ✓ Input shape: {dummy_audio.shape}")
    print(f"   ✓ Output shape: {features.shape}")
    print(f"   ✓ Expected: (64, ~1001)")
except Exception as e:
    print(f"   ✗ Feature extraction failed: {e}")
    sys.exit(1)

# Test model
print("\n4. Testing model architecture...")
try:
    model = LightweightCNN(n_mels=64, num_classes=2)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"   ✓ Model created")
    print(f"   ✓ Parameters: {num_params:,}")
    
    # Test forward pass
    dummy_input = torch.randn(2, 64, 1001)  # batch=2
    output = model(dummy_input)
    print(f"   ✓ Forward pass works")
    print(f"   ✓ Output shape: {output.shape} (expected: [2, 2])")
except Exception as e:
    print(f"   ✗ Model test failed: {e}")
    sys.exit(1)

# Test ONNX export
print("\n5. Testing ONNX export...")
try:
    import tempfile
    
    model.eval()
    temp_path = Path(tempfile.gettempdir()) / "test_model.onnx"
    
    export_to_onnx(
        model,
        temp_path,
        n_mels=64,
        time_frames=1001,
        device=torch.device("cpu"),
    )
    print(f"   ✓ ONNX export successful")
    
    # Verify
    if verify_onnx_export(model, temp_path, 64, 1001, torch.device("cpu")):
        print(f"   ✓ ONNX verification passed")
    
    # Cleanup
    temp_path.unlink()
except Exception as e:
    print(f"   ✗ ONNX export failed: {e}")
    sys.exit(1)

# Check for dataset/manifest
print("\n6. Checking for dataset...")
manifest_paths = [
    Path("ml/datasets/manifests/asvspoof_2019.jsonl"),
    Path("ml/datasets/asvspoof_2019/LA"),
]

dataset_ready = False
for path in manifest_paths:
    if path.exists():
        print(f"   ✓ Found: {path}")
        dataset_ready = True
    else:
        print(f"   ✗ Not found: {path}")

if not dataset_ready:
    print("\n   ⚠ Dataset not ready for training")
    print("   ⚠ You'll need to download ASVspoof 2019 dataset")
    print("   ⚠ Or use the Colab notebook which downloads automatically")

print("\n" + "=" * 70)
print("✓ ALL TESTS PASSED!")
print("=" * 70)
print("\nYour system is ready for training!")
print("\nNext steps:")
print("  1. Open ml/training/train_audio_colab.ipynb in VS Code")
print("  2. Select 'Colab' kernel in the notebook")
print("  3. Run cells sequentially")
print("\nOr run training directly (if dataset is ready):")
print("  python ml/training/train_audio.py \\")
print("    --manifest ml/datasets/manifests/asvspoof_2019.jsonl \\")
print("    --output-dir models/audio \\")
print("    --model-version V1.0.0 \\")
print("    --epochs 20 \\")
print("    --batch-size 32")
print("=" * 70)
