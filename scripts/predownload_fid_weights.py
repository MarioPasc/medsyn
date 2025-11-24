#!/usr/bin/env python3
"""
Pre-download InceptionV3 weights for FID computation on HPC clusters without internet.

This script downloads the InceptionV3 weights used by:
1. torchmetrics.image.fid.FrechetInceptionDistance
2. torch-fidelity

The weights are saved to a configurable directory and can be loaded offline
by setting the TORCH_HOME environment variable or the fid.weights_path config.

Usage:
    python scripts/predownload_fid_weights.py --cache_dir /path/to/cache

    # Then on HPC without internet, either:
    # 1. Set environment variable:
    export TORCH_HOME=/path/to/cache

    # 2. Or configure in picasso_cfg.yaml:
    fid:
      weights_path: /path/to/cache/hub/checkpoints/weights-inception-2015-12-05-6726825d.pth

Author: MedSyn Project
"""

import argparse
import os
import sys
from pathlib import Path
import urllib.request
import shutil


# URLs for the inception weights
TORCH_FIDELITY_URL = (
    "https://github.com/toshas/torch-fidelity/releases/download/v0.2.0/"
    "weights-inception-2015-12-05-6726825d.pth"
)
TORCH_FIDELITY_FILENAME = "weights-inception-2015-12-05-6726825d.pth"

# torchmetrics uses torchvision's InceptionV3 which is downloaded via torch hub
TORCHVISION_INCEPTION_URL = (
    "https://download.pytorch.org/models/inception_v3_google-0cc3c7bd.pth"
)
TORCHVISION_INCEPTION_FILENAME = "inception_v3_google-0cc3c7bd.pth"


def check_internet_connectivity() -> bool:
    """Check if we have internet access."""
    try:
        urllib.request.urlopen('https://github.com', timeout=5)
        return True
    except Exception:
        return False


def download_file(url: str, dest_path: Path, description: str) -> bool:
    """Download a file with progress indicator."""
    try:
        print(f"  -> Downloading {description}...")
        print(f"     URL: {url}")
        print(f"     Destination: {dest_path}")

        dest_path.parent.mkdir(parents=True, exist_ok=True)

        # Download with progress
        def progress_hook(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                percent = min(100, downloaded * 100 // total_size)
                mb_downloaded = downloaded / (1024 * 1024)
                mb_total = total_size / (1024 * 1024)
                print(f"\r     Progress: {percent}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)", end="")

        urllib.request.urlretrieve(url, dest_path, progress_hook)
        print()  # Newline after progress
        print(f"  OK {description} downloaded")
        return True

    except Exception as e:
        print(f"\n  FAIL Failed to download {description}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Pre-download FID (InceptionV3) weights for offline HPC execution"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        required=True,
        help="Directory to store downloaded weights (will create torch/hub/checkpoints structure)"
    )
    parser.add_argument(
        "--skip-torch-fidelity",
        action="store_true",
        help="Skip downloading torch-fidelity weights"
    )
    parser.add_argument(
        "--skip-torchvision",
        action="store_true",
        help="Skip downloading torchvision InceptionV3 weights"
    )
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir).resolve()

    print("=" * 80)
    print("Pre-downloading FID Weights for Offline Execution")
    print("=" * 80)
    print(f"Cache directory: {cache_dir}")
    print()

    # Check internet connectivity
    print("[1/3] Checking internet connectivity...")
    if not check_internet_connectivity():
        print("  FAIL No internet connection available!")
        print("  Please run this script on a machine with internet access.")
        sys.exit(1)
    print("  OK Internet connection available")
    print()

    # Create directory structure matching PyTorch's expected layout
    # PyTorch looks for weights in $TORCH_HOME/hub/checkpoints/
    checkpoints_dir = cache_dir / "hub" / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    # Also create structure for torchvision models
    torchvision_dir = cache_dir / "hub" / "checkpoints"  # Same location

    success_count = 0
    total_count = 0

    # Download torch-fidelity weights (used by torch_fidelity.calculate_metrics)
    if not args.skip_torch_fidelity:
        print("[2/3] Downloading torch-fidelity InceptionV3 weights (~100 MB)...")
        total_count += 1
        dest = checkpoints_dir / TORCH_FIDELITY_FILENAME
        if dest.exists():
            print(f"  OK Already exists: {dest}")
            success_count += 1
        elif download_file(TORCH_FIDELITY_URL, dest, "torch-fidelity InceptionV3"):
            success_count += 1
        print()
    else:
        print("[2/3] Skipping torch-fidelity weights (--skip-torch-fidelity)")
        print()

    # Download torchvision InceptionV3 weights (used by torchmetrics FID)
    if not args.skip_torchvision:
        print("[3/3] Downloading torchvision InceptionV3 weights (~100 MB)...")
        total_count += 1
        dest = torchvision_dir / TORCHVISION_INCEPTION_FILENAME
        if dest.exists():
            print(f"  OK Already exists: {dest}")
            success_count += 1
        elif download_file(TORCHVISION_INCEPTION_URL, dest, "torchvision InceptionV3"):
            success_count += 1
        print()
    else:
        print("[3/3] Skipping torchvision weights (--skip-torchvision)")
        print()

    # Summary
    print("=" * 80)
    if success_count == total_count:
        print("OK All weights downloaded successfully!")
    else:
        print(f"WARN Downloaded {success_count}/{total_count} weight files")
    print("=" * 80)
    print()

    # List downloaded files
    print("Downloaded files:")
    for f in checkpoints_dir.glob("*.pth"):
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  - {f.name} ({size_mb:.1f} MB)")
    print()

    # Usage instructions
    print("Next Steps for HPC Offline Execution:")
    print()
    print("1. Copy cache directory to HPC cluster:")
    print(f"   rsync -avP {cache_dir}/ cluster:/path/to/torch_cache/")
    print()
    print("2. On HPC, set TORCH_HOME before running training:")
    print("   export TORCH_HOME=/path/to/torch_cache")
    print()
    print("3. Or add to your SLURM script:")
    print("   #SBATCH --export=TORCH_HOME=/path/to/torch_cache")
    print()
    print("4. Alternatively, configure in picasso_cfg.yaml:")
    print("   fid:")
    print(f"     weights_path: /path/to/torch_cache/hub/checkpoints/{TORCH_FIDELITY_FILENAME}")
    print()
    print("=" * 80)

    # Verify the files are valid PyTorch checkpoints
    print()
    print("Verifying downloaded weights...")
    try:
        import torch
        for f in checkpoints_dir.glob("*.pth"):
            try:
                # Just load to verify it's a valid checkpoint
                state_dict = torch.load(f, map_location="cpu", weights_only=True)
                print(f"  OK {f.name} is valid (contains {len(state_dict)} keys)")
            except Exception as e:
                print(f"  WARN {f.name} verification failed: {e}")
    except ImportError:
        print("  WARN Could not import torch for verification")

    print()
    print("Done!")


if __name__ == "__main__":
    main()
