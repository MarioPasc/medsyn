#!/usr/bin/env python3
"""
Pre-download DistDiff models for offline execution on HPC clusters.

This script downloads all required models to a specified directory,
which can then be copied to the cluster for offline use.

Usage:
    python scripts/predownload_distdiff_models.py --cache_dir /path/to/model/cache

Models downloaded:
    1. Stable Diffusion v1-4 (~4-5 GB)
       - UNet, VAE, Text Encoder, Tokenizer, Scheduler
    2. OpenCLIP ViT-B/32 (~350 MB)
    3. Transformers components

Author: MedSyn Project
"""

import argparse
import os
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(
        description="Pre-download DistDiff models for offline execution"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        required=True,
        help="Directory to store downloaded models (e.g., /scratch/models/distdiff)"
    )
    parser.add_argument(
        "--sd_model",
        type=str,
        default="CompVis/stable-diffusion-v1-4",
        help="Stable Diffusion model to download"
    )
    parser.add_argument(
        "--clip_model",
        type=str,
        default="ViT-B-32",
        help="OpenCLIP model name"
    )
    parser.add_argument(
        "--clip_pretrained",
        type=str,
        default="laion2b_s34b_b79k",
        help="OpenCLIP pretrained dataset"
    )
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("🔽 Pre-downloading DistDiff Models for Offline Execution")
    print("="*80)
    print(f"Cache directory: {cache_dir}")
    print(f"Stable Diffusion: {args.sd_model}")
    print(f"OpenCLIP: {args.clip_model} ({args.clip_pretrained})")
    print()

    # Check internet connectivity
    print("[1/3] Checking internet connectivity...")
    try:
        import urllib.request
        urllib.request.urlopen('https://huggingface.co', timeout=5)
        print("  ✓ Internet connection OK")
    except Exception as e:
        print(f"  ✗ No internet connection: {e}")
        print("  Please run this script with internet access!")
        sys.exit(1)
    print()

    # Download Stable Diffusion components
    print("[2/3] Downloading Stable Diffusion v1-4 (~4-5 GB, may take 10-30 min)...")
    print("  This includes: UNet, VAE, Text Encoder, Tokenizer, Scheduler")
    try:
        from diffusers import (
            StableDiffusionPipeline,
            DDIMScheduler,
            UNet2DConditionModel,
            AutoencoderKL
        )
        from transformers import AutoTokenizer, CLIPTextModel

        # Download complete pipeline (downloads all components)
        print(f"  → Downloading {args.sd_model}...")
        pipeline = StableDiffusionPipeline.from_pretrained(
            args.sd_model,
            cache_dir=str(cache_dir),
            torch_dtype=None,  # Use float32 for download
        )
        print("  ✓ Stable Diffusion pipeline downloaded")

        # Download individual components with cache_dir
        print("  → Downloading scheduler...")
        scheduler = DDIMScheduler.from_pretrained(
            args.sd_model,
            subfolder="scheduler",
            cache_dir=str(cache_dir)
        )
        print("  ✓ Scheduler downloaded")

        print("  → Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            args.sd_model,
            subfolder="tokenizer",
            cache_dir=str(cache_dir),
            use_fast=False
        )
        print("  ✓ Tokenizer downloaded")

        print("  → Downloading text encoder...")
        text_encoder = CLIPTextModel.from_pretrained(
            args.sd_model,
            subfolder="text_encoder",
            cache_dir=str(cache_dir)
        )
        print("  ✓ Text encoder downloaded")

        print("  → Downloading VAE...")
        vae = AutoencoderKL.from_pretrained(
            args.sd_model,
            subfolder="vae",
            cache_dir=str(cache_dir)
        )
        print("  ✓ VAE downloaded")

        print("  → Downloading UNet...")
        unet = UNet2DConditionModel.from_pretrained(
            args.sd_model,
            subfolder="unet",
            cache_dir=str(cache_dir)
        )
        print("  ✓ UNet downloaded")

    except Exception as e:
        print(f"  ✗ Failed to download Stable Diffusion: {e}")
        sys.exit(1)
    print()

    # Download OpenCLIP
    print("[3/3] Downloading OpenCLIP ViT-B/32 (~350 MB)...")
    try:
        import open_clip
        import torch

        print(f"  → Downloading {args.clip_model} pretrained on {args.clip_pretrained}...")

        # Download model
        model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
            args.clip_model,
            pretrained=args.clip_pretrained,
            cache_dir=str(cache_dir)
        )
        print("  ✓ OpenCLIP model downloaded")

        # Get tokenizer (no download needed, built-in)
        tokenizer = open_clip.get_tokenizer(args.clip_model)
        print("  ✓ OpenCLIP tokenizer ready")

    except Exception as e:
        print(f"  ✗ Failed to download OpenCLIP: {e}")
        sys.exit(1)
    print()

    # Summary
    print("="*80)
    print("✅ All models downloaded successfully!")
    print("="*80)
    print()
    print("Downloaded models:")
    print(f"  1. Stable Diffusion v1-4:  {cache_dir}/")
    print(f"  2. OpenCLIP ViT-B/32:      {cache_dir}/")
    print()

    # Calculate total size
    import subprocess
    try:
        result = subprocess.run(
            ['du', '-sh', str(cache_dir)],
            capture_output=True,
            text=True
        )
        total_size = result.stdout.split()[0]
        print(f"Total cache size: {total_size}")
    except:
        print("Total cache size: ~4-5 GB")
    print()

    # Next steps
    print("📋 Next Steps for HPC Offline Execution:")
    print()
    print("1. Copy cache directory to HPC cluster:")
    print(f"   rsync -avP {cache_dir}/ cluster:/path/to/models/")
    print()
    print("2. Update your config (config/distdiff_pathmnist.yaml):")
    print("   Add under 'expansion:':")
    print("   cache_dir: /path/to/models/")
    print()
    print("3. Or pass --cache_dir when running:")
    print("   python generate_data.py --cache_dir /path/to/models/ ...")
    print()
    print("4. The SLURM scripts will automatically use the cache_dir from config")
    print()
    print("="*80)
    print()
    print("⚠️  IMPORTANT: Make sure the cache directory is accessible from compute nodes!")
    print("   Recommended locations:")
    print("   - User scratch: /scratch/$USER/models/distdiff/")
    print("   - Shared scratch: /shared/scratch/models/distdiff/")
    print("   - NOT home directory (usually too slow/small)")
    print()

if __name__ == "__main__":
    main()
