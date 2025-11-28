#!/usr/bin/env python3
"""
Diagnostic script to inspect layer outputs from ccDDPM model.
This helps identify which layers return tuples vs tensors.
Similar to print_unet_layers.py but focuses on output types.
"""

import sys
import torch
import torch.nn as nn
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from medsyn.models.ccDDPM.config import CCDDPMInit
from medsyn.models.ccDDPM.model import CCDDPM

def inspect_layer_outputs(model: nn.Module, x_noisy: torch.Tensor, timesteps: torch.Tensor, class_labels: torch.Tensor):
    """
    Register hooks on all layers to inspect their output types.
    Identifies which layers return tuples (potential issue) vs tensors.
    """
    outputs = {}
    handles = []

    def make_hook(name):
        def hook(module, input, output):
            output_type = type(output).__name__
            if isinstance(output, tuple):
                shapes = [tuple(o.shape) if isinstance(o, torch.Tensor) else type(o).__name__ for o in output]
                outputs[name] = {
                    'type': 'tuple',
                    'length': len(output),
                    'element_types': [type(o).__name__ for o in output],
                    'shapes': shapes
                }
            elif isinstance(output, torch.Tensor):
                outputs[name] = {
                    'type': 'Tensor',
                    'shape': tuple(output.shape)
                }
            else:
                outputs[name] = {
                    'type': output_type,
                    'value': str(output)
                }
        return hook

    # Register hooks on all named modules
    for name, module in model.named_modules():
        if name:  # Skip the root module
            handle = module.register_forward_hook(make_hook(name))
            handles.append(handle)

    # Forward pass
    try:
        with torch.no_grad():
            _ = model(x_noisy, timesteps, class_labels)
    except Exception as e:
        print(f"Error during forward pass: {e}")
        import traceback
        traceback.print_exc()

    # Remove hooks
    for handle in handles:
        handle.remove()

    return outputs


def main():
    print("=" * 80)
    print("ccDDPM Layer Output Type Diagnostic")
    print("=" * 80)

    # Create a simple model configuration
    print("\nCreating model with default configuration...")

    # Simple config for PathMNIST-like setup
    model_cfg = CCDDPMInit(
        in_channels=3,
        class_embed_dim=16,
        num_classes=9,
        model_channels=64,  # Smaller for faster testing
        channel_mult=(1, 2, 2),  # 3 levels
        layers_per_block=2,
        down_block_types=("DownBlock2D", "DownBlock2D", "AttnDownBlock2D"),
        up_block_types=("AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
    )

    # Build model
    print(f"Building ccDDPM model...")
    model = CCDDPM(model_cfg)
    model.eval()

    # Create dummy inputs
    batch_size = 2
    image_size = 28
    in_channels = model_cfg.in_channels
    num_classes = model_cfg.num_classes

    print(f"\nModel configuration:")
    print(f"  Image size: {image_size}")
    print(f"  In channels: {in_channels}")
    print(f"  Num classes: {num_classes}")
    print(f"  Class embed dim: {model_cfg.class_embed_dim}")

    x_noisy = torch.randn(batch_size, in_channels, image_size, image_size)
    timesteps = torch.tensor([100, 200], dtype=torch.long)
    class_labels = torch.tensor([0, 1], dtype=torch.long)

    print(f"\nInput shapes:")
    print(f"  x_noisy: {x_noisy.shape}")
    print(f"  timesteps: {timesteps.shape}")
    print(f"  class_labels: {class_labels.shape}")

    # Inspect outputs
    print("\n" + "=" * 80)
    print("Inspecting layer outputs...")
    print("=" * 80)

    outputs = inspect_layer_outputs(model, x_noisy, timesteps, class_labels)

    # Categorize outputs
    tuple_outputs = []
    tensor_outputs = []
    other_outputs = []

    for name, info in sorted(outputs.items()):
        if info['type'] == 'tuple':
            tuple_outputs.append((name, info))
        elif info['type'] == 'Tensor':
            tensor_outputs.append((name, info))
        else:
            other_outputs.append((name, info))

    # Display results
    print(f"\n{'='*80}")
    print(f"SUMMARY:")
    print(f"{'='*80}")
    print(f"Total layers inspected: {len(outputs)}")
    print(f"  Tensor outputs: {len(tensor_outputs)}")
    print(f"  Tuple outputs: {len(tuple_outputs)} ⚠️")
    print(f"  Other outputs: {len(other_outputs)}")

    if tuple_outputs:
        print(f"\n{'='*80}")
        print(f"⚠️  LAYERS RETURNING TUPLES (potential issues for hooks):")
        print(f"{'='*80}")
        for name, info in tuple_outputs:
            print(f"\n{name}:")
            print(f"  Length: {info['length']}")
            print(f"  Element types: {info['element_types']}")
            if 'shapes' in info:
                for i, shape in enumerate(info['shapes']):
                    print(f"  Element[{i}] shape: {shape}")

    # Show a sample of tensor outputs
    print(f"\n{'='*80}")
    print(f"SAMPLE TENSOR OUTPUTS (first 10):")
    print(f"{'='*80}")
    for name, info in tensor_outputs[:10]:
        print(f"  {name}: {info['shape']}")

    # Check specific layers that embeddings.py uses
    print(f"\n{'='*80}")
    print(f"CHECKING SPECIFIC LAYERS USED IN embeddings.py:")
    print(f"{'='*80}")

    test_layers = [
        "class_embed",
        "class_embed.emb",
        "unet.down_blocks.0",
        "unet.down_blocks.1",
        "unet.up_blocks.0",
        "unet.mid_block"
    ]

    for layer_name in test_layers:
        if layer_name in outputs:
            info = outputs[layer_name]
            status = "✓" if info['type'] == 'Tensor' else "⚠️"
            print(f"{status} {layer_name}: {info['type']}", end="")
            if info['type'] == 'Tensor':
                print(f" {info['shape']}")
            elif info['type'] == 'tuple':
                print(f" (length {info['length']})")
            else:
                print()
        else:
            print(f"✗ {layer_name}: NOT FOUND")

    print("\n" + "=" * 80)
    print("RECOMMENDATIONS:")
    print("=" * 80)
    print("1. For layers returning tuples, hooks should extract output[0]")
    print("2. Update embeddings.py hook_fn to handle tuple outputs (DONE ✓)")
    print("3. Test layer extraction with actual layer names from config")
    print("=" * 80)


if __name__ == "__main__":
    main()
