#!/usr/bin/env python3
"""Script to integrate a trained ONNX model into the API.

This script:
1. Copies the ONNX model to models/audio/
2. Creates a symlink to latest.onnx
3. Verifies the model can be loaded
4. Tests basic inference

Usage:
    python ml/scripts/integrate_trained_model.py --model path/to/V1.0.0.onnx
"""

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np


def create_symlink(target: Path, link_name: Path) -> None:
    """Create symlink (cross-platform)."""
    if link_name.exists():
        link_name.unlink()
    
    try:
        # Try symbolic link (Unix/Linux/Mac)
        link_name.symlink_to(target.name)
        print(f"✓ Created symlink: {link_name} -> {target.name}")
    except (OSError, NotImplementedError):
        # Fallback for Windows without admin rights: just copy
        shutil.copy(target, link_name)
        print(f"✓ Copied model to: {link_name}")


def verify_onnx_model(model_path: Path) -> bool:
    """Verify ONNX model can be loaded and run."""
    try:
        import onnxruntime as ort
        
        print(f"Loading ONNX model: {model_path}")
        session = ort.InferenceSession(str(model_path))
        
        print("Model inputs:")
        for inp in session.get_inputs():
            print(f"  {inp.name}: {inp.shape} ({inp.type})")
        
        print("Model outputs:")
        for out in session.get_outputs():
            print(f"  {out.name}: {out.shape} ({out.type})")
        
        # Test inference with dummy input
        input_name = session.get_inputs()[0].name
        dummy_input = np.random.randn(1, 64, 1001).astype(np.float32)
        
        outputs = session.run(None, {input_name: dummy_input})
        print(f"\nTest inference output shape: {outputs[0].shape}")
        print(f"Test inference output range: [{outputs[0].min():.3f}, {outputs[0].max():.3f}]")
        
        print("✓ ONNX model verified")
        return True
        
    except Exception as e:
        print(f"✗ ONNX model verification failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Integrate trained ONNX model")
    parser.add_argument("--model", type=Path, required=True, help="Path to ONNX model file")
    parser.add_argument("--metadata", type=Path, help="Path to metadata JSON file")
    parser.add_argument("--output-dir", type=Path, default=Path("models/audio"), help="Output directory")
    
    args = parser.parse_args()
    
    if not args.model.exists():
        print(f"Error: Model file not found: {args.model}")
        return 1
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy model
    model_name = args.model.name
    dest_path = args.output_dir / model_name
    
    print(f"Copying {args.model} -> {dest_path}")
    shutil.copy(args.model, dest_path)
    print("✓ Model copied")
    
    # Copy metadata if provided
    if args.metadata and args.metadata.exists():
        metadata_dest = args.output_dir / args.metadata.name
        shutil.copy(args.metadata, metadata_dest)
        print(f"✓ Metadata copied to {metadata_dest}")
    
    # Create symlink to latest.onnx
    latest_path = args.output_dir / "latest.onnx"
    create_symlink(dest_path, latest_path)
    
    # Verify model
    print("\nVerifying model...")
    if not verify_onnx_model(dest_path):
        return 1
    
    print("\n" + "=" * 70)
    print("✓ Model integration complete!")
    print("=" * 70)
    print("\nNext steps:")
    print(f"1. Set environment variable:")
    print(f"   export ONNX_MODEL_PATH={dest_path}")
    print(f"   # Or: export ONNX_MODEL_PATH={latest_path}")
    print("\n2. Start the API:")
    print("   cd services/api")
    print("   uvicorn app.main:app --reload")
    print("\n3. Test the model endpoint:")
    print("   curl http://localhost:8000/v1/models")
    print("\n4. Upload and analyze audio:")
    print("   curl -X POST http://localhost:8000/v1/analysis/upload -F 'file=@test.wav'")
    print("   curl -X POST http://localhost:8000/v1/analysis/{analysis_id}/run")
    
    return 0


if __name__ == "__main__":
    exit(main())
