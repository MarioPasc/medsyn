#!/usr/bin/env python3
"""
Validate ccDDPM checkpoint after training.

Usage:
    python validate_checkpoint.py /path/to/best.pt
    python validate_checkpoint.py /media/mpascual/PortableSSD/medsyn/PathMNIST_ccDDPM/ckpts/best.pt
"""
import sys
import torch
from pathlib import Path

def validate_checkpoint(ckpt_path: str) -> bool:
    """
    Validate a ccDDPM checkpoint for common training failures.

    Returns:
        True if checkpoint is valid, False otherwise
    """
    print(f"Validating checkpoint: {ckpt_path}")
    print("=" * 80)

    try:
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    except Exception as e:
        print(f"❌ Failed to load checkpoint: {e}")
        return False

    all_checks_passed = True

    # Check 1: Basic structure
    print("\n📋 Basic Structure Checks:")
    required_keys = ["model", "opt", "epoch", "val_loss", "ema", "cfg"]
    for key in required_keys:
        if key in ckpt:
            print(f"  ✓ '{key}' present")
        else:
            print(f"  ❌ '{key}' missing")
            all_checks_passed = False

    # Check 2: EMA weights
    print("\n🔄 EMA Weights Check:")
    if "ema" in ckpt:
        ema_dict = ckpt["ema"]
        if ema_dict is None:
            print(f"  ❌ EMA is None (EMA was disabled during training)")
            all_checks_passed = False
        elif len(ema_dict) == 0:
            print(f"  ❌ CRITICAL: EMA dict is EMPTY! Training bug not fixed.")
            all_checks_passed = False
        else:
            print(f"  ✓ EMA dict has {len(ema_dict)} parameters")

            # Verify against model
            model_params = len(ckpt["model"])
            if len(ema_dict) == model_params:
                print(f"  ✓ EMA param count matches model ({model_params})")
            else:
                print(f"  ⚠️  EMA has {len(ema_dict)} params, model has {model_params}")
                all_checks_passed = False
    else:
        print(f"  ❌ No 'ema' key in checkpoint")
        all_checks_passed = False

    # Check 3: Diagnostics (if available)
    print("\n🔍 Training Diagnostics Check:")
    if "diagnostics" in ckpt:
        diag = ckpt["diagnostics"]

        # Input-output correlation
        if "input_output_correlation" in diag:
            corr = diag["input_output_correlation"]
            print(f"  Input-Output Correlation: {corr:.4f}", end="")
            if corr > 0.7:
                print(f" ❌ CRITICAL: Model is echoing input!")
                all_checks_passed = False
            elif corr > 0.5:
                print(f" ⚠️  High correlation, may not be learning properly")
            else:
                print(f" ✓")

        # Prediction std
        if "prediction_std" in diag:
            std = diag["prediction_std"]
            print(f"  Prediction Std: {std:.4f}", end="")
            if std < 0.5 or std > 1.5:
                print(f" ⚠️  Unusual (should be ~0.8-1.2)")
            else:
                print(f" ✓")

        # Reconstruction PSNR
        if "reconstruction_psnr_t500" in diag:
            psnr = diag["reconstruction_psnr_t500"]
            print(f"  Reconstruction PSNR@t500: {psnr:.2f} dB", end="")
            if psnr < 15.0:
                print(f" ⚠️  Low quality")
            elif psnr < 20.0:
                print(f" ⚠️  Marginal (consider training longer)")
            else:
                print(f" ✓")

        # Reconstruction MSE
        if "reconstruction_mse_t500" in diag:
            mse = diag["reconstruction_mse_t500"]
            print(f"  Reconstruction MSE@t500: {mse:.4f}")
    else:
        print(f"  ⚠️  No diagnostics in checkpoint (trained with old code?)")

    # Check 4: Training config
    print("\n⚙️  Training Configuration:")
    if "cfg" in ckpt:
        cfg = ckpt["cfg"]

        # Gradient clipping
        if "grad_clip_norm" in cfg:
            grad_clip = cfg["grad_clip_norm"]
            print(f"  grad_clip_norm: {grad_clip}", end="")
            if grad_clip == 1.0:
                print(f" ⚠️  May be too aggressive (try 10.0)")
            else:
                print(f" ✓")

        # Min-SNR
        if "use_min_snr" in cfg:
            use_min_snr = cfg.get("use_min_snr", False)
            print(f"  use_min_snr: {use_min_snr}", end="")
            if not use_min_snr:
                print(f" ℹ️  Disabled (consider enabling for stability)")
            else:
                print(f" ✓")

        # Class embed dim
        if "class_embed_dim" in cfg:
            embed_dim = cfg["class_embed_dim"]
            print(f"  class_embed_dim: {embed_dim}", end="")
            if embed_dim == 32:
                print(f" ✓")
            else:
                print(f" ℹ️  (ensure generator config matches)")

        # EMA
        if "ema_use" in cfg:
            ema_use = cfg.get("ema_use", False)
            print(f"  ema_use: {ema_use} ✓" if ema_use else f"  ema_use: {ema_use}")

    # Check 5: Training metrics
    print("\n📊 Training Metrics:")
    if "val_metrics" in ckpt:
        vm = ckpt["val_metrics"]
        print(f"  Val Loss: {vm.get('loss', 'N/A')}")
        print(f"  Val PSNR: {vm.get('psnr', 'N/A'):.2f} dB" if 'psnr' in vm else "  Val PSNR: N/A")
        print(f"  Val SSIM: {vm.get('ssim', 'N/A'):.4f}" if 'ssim' in vm else "  Val SSIM: N/A")

    print(f"\n  Epoch: {ckpt.get('epoch', 'Unknown')}")
    print(f"  Val Loss: {ckpt.get('val_loss', 'Unknown')}")

    # Final verdict
    print("\n" + "=" * 80)
    if all_checks_passed:
        print("✅ CHECKPOINT VALIDATED SUCCESSFULLY")
        print("   Model appears to be properly trained and ready for generation.")
    else:
        print("❌ CHECKPOINT VALIDATION FAILED")
        print("   Model may have training issues. Review warnings above.")
        print("   Consider retraining with fixed configuration.")
    print("=" * 80)

    return all_checks_passed


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python validate_checkpoint.py <path/to/checkpoint.pt>")
        sys.exit(1)

    ckpt_path = sys.argv[1]
    if not Path(ckpt_path).exists():
        print(f"Error: Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    success = validate_checkpoint(ckpt_path)
    sys.exit(0 if success else 1)
