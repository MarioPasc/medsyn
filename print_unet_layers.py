#!/usr/bin/env python3
"""
Print the layer structure of the CCDDPM model to understand available layer names.
"""
import torch
from pathlib import Path
from medsyn.models.ccDDPM.model import CCDDPM
from medsyn.models.ccDDPM.config import CCDDPMInit, load_cfg

def print_model_structure(model, prefix="", max_depth=3, current_depth=0):
    """Recursively print model structure."""
    if current_depth >= max_depth:
        return

    for name, module in model.named_children():
        layer_path = f"{prefix}.{name}" if prefix else name
        print(f"{'  ' * current_depth}{layer_path}: {module.__class__.__name__}")

        # Print children recursively
        print_model_structure(module, layer_path, max_depth, current_depth + 1)

def main():
    # Load the actual configuration from the YAML file
    config_path = Path("experiments/AdamW_Batch64_lowt_enhancement_gamma8/config_AdamW_Batch64_lowt_enhancement_gamma8.yaml")

    print(f"Loading configuration from: {config_path}")
    proj_cfg = load_cfg(config_path)

    # Extract UNet config
    unet_cfg = proj_cfg.ccddpm.unet
    train_cfg = proj_cfg.ccddpm.train

    # Create CCDDPMInit from the loaded config
    cfg = CCDDPMInit(
        in_channels=train_cfg.in_channels,
        num_classes=train_cfg.num_classes,
        class_embed_dim=train_cfg.class_embed_dim,
        model_channels=unet_cfg.model_channels,
        channel_mult=unet_cfg.channel_mult,
        layers_per_block=unet_cfg.layers_per_block,
        down_block_types=unet_cfg.down_block_types,
        up_block_types=unet_cfg.up_block_types,
        add_attention=unet_cfg.add_attention,
        attention_head_dim=unet_cfg.attention_head_dim,
        norm_num_groups=unet_cfg.norm_num_groups,
        dropout=unet_cfg.dropout,
    )

    model = CCDDPM(cfg)

    print(f"Loaded model with configuration:")
    print(f"  - in_channels: {cfg.in_channels}")
    print(f"  - num_classes: {cfg.num_classes}")
    print(f"  - class_embed_dim: {cfg.class_embed_dim}")
    print(f"  - model_channels: {cfg.model_channels}")
    print(f"  - channel_mult: {cfg.channel_mult}")
    print(f"  - down_block_types: {cfg.down_block_types}")
    print(f"  - up_block_types: {cfg.up_block_types}")
    print()

    print("=" * 80)
    print("CCDDPM Model Layer Structure")
    print("=" * 80)
    print()

    print("Top-level attributes:")
    for name in dir(model):
        if not name.startswith('_'):
            attr = getattr(model, name)
            if isinstance(attr, torch.nn.Module):
                print(f"  - {name}: {attr.__class__.__name__}")
    print()

    print("=" * 80)
    print("Full Layer Hierarchy (depth=3):")
    print("=" * 80)
    print_model_structure(model, max_depth=3)
    print()

    print("=" * 80)
    print("Available UNet blocks:")
    print("=" * 80)
    if hasattr(model, 'unet'):
        unet = model.unet
        print(f"\nUNet has the following block attributes:")
        for attr_name in ['down_blocks', 'mid_block', 'up_blocks']:
            if hasattr(unet, attr_name):
                attr = getattr(unet, attr_name)
                if isinstance(attr, (list, torch.nn.ModuleList)):
                    print(f"  - unet.{attr_name}: ModuleList with {len(attr)} blocks")
                    for i, block in enumerate(attr):
                        print(f"      [{i}] {block.__class__.__name__}")
                else:
                    print(f"  - unet.{attr_name}: {attr.__class__.__name__}")
            else:
                print(f"  - unet.{attr_name}: NOT FOUND")

    print()
    print("=" * 80)
    print("Correct layer names to use in embeddings.py:")
    print("=" * 80)
    print("  Instead of 'down_blocks.0', use 'unet.down_blocks.0'")
    print("  Instead of 'mid_block', use 'unet.mid_block'")
    print("  Instead of 'up_blocks.0', use 'unet.up_blocks.0'")
    print()

if __name__ == "__main__":
    main()
