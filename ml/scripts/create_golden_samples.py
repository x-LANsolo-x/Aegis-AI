#!/usr/bin/env python3
"""Create golden test samples for inference testing.

This script extracts a few representative samples from the ASVspoof dataset
to use as golden samples for testing the trained model.

Usage:
    python ml/scripts/create_golden_samples.py --manifest path/to/manifest.jsonl --output-dir services/api/tests/fixtures/golden_samples
"""

import argparse
import json
import shutil
from pathlib import Path
from typing import List, Dict


def select_golden_samples(manifest_path: Path, num_bonafide: int = 3, num_spoof: int = 3) -> List[Dict]:
    """Select representative samples from manifest."""
    bonafide_samples = []
    spoof_samples = []
    
    with open(manifest_path) as f:
        for line in f:
            rec = json.loads(line)
            
            # Only use dev or eval split for testing
            if rec['split'] not in ['dev', 'eval']:
                continue
            
            if rec['label'] == 'bonafide' and len(bonafide_samples) < num_bonafide:
                bonafide_samples.append(rec)
            elif rec['label'] != 'bonafide' and len(spoof_samples) < num_spoof:
                spoof_samples.append(rec)
            
            if len(bonafide_samples) >= num_bonafide and len(spoof_samples) >= num_spoof:
                break
    
    return bonafide_samples + spoof_samples


def main():
    parser = argparse.ArgumentParser(description="Create golden test samples")
    parser.add_argument("--manifest", type=Path, required=True, help="Path to manifest.jsonl")
    parser.add_argument("--output-dir", type=Path, default=Path("services/api/tests/fixtures/golden_samples"), help="Output directory")
    parser.add_argument("--num-bonafide", type=int, default=3, help="Number of bonafide samples")
    parser.add_argument("--num-spoof", type=int, default=3, help="Number of spoof samples")
    
    args = parser.parse_args()
    
    if not args.manifest.exists():
        print(f"Error: Manifest not found: {args.manifest}")
        return 1
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Select samples
    print("Selecting golden samples...")
    samples = select_golden_samples(args.manifest, args.num_bonafide, args.num_spoof)
    
    if not samples:
        print("Error: No samples found in manifest")
        return 1
    
    print(f"Selected {len(samples)} samples:")
    
    # Copy samples and create metadata
    golden_metadata = []
    
    for i, sample in enumerate(samples):
        src_path = Path(sample['path'])
        
        if not src_path.exists():
            print(f"Warning: Sample not found: {src_path}")
            continue
        
        # Create a simple filename
        label_prefix = "bonafide" if sample['label'] == 'bonafide' else "spoof"
        dest_filename = f"{label_prefix}_{i:02d}{src_path.suffix}"
        dest_path = args.output_dir / dest_filename
        
        # Copy file
        shutil.copy(src_path, dest_path)
        print(f"  ✓ {dest_filename} ({sample['label']})")
        
        # Add to metadata
        golden_metadata.append({
            "filename": dest_filename,
            "label": sample['label'],
            "expected_verdict": "AUTHENTIC" if sample['label'] == 'bonafide' else "DEEPFAKE",
            "split": sample['split'],
            "duration_sec": sample['duration_sec'],
        })
    
    # Save metadata
    metadata_path = args.output_dir / "golden_samples.json"
    with open(metadata_path, 'w') as f:
        json.dump(golden_metadata, f, indent=2)
    
    print(f"\n✓ Created {len(golden_metadata)} golden samples")
    print(f"✓ Metadata saved to {metadata_path}")
    
    return 0


if __name__ == "__main__":
    exit(main())
