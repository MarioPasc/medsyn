#!/usr/bin/env python3
"""
Diagnostic script to check class_embed usage in ccDDPM model.
This tests whether class_embed() is being called correctly.
"""

import sys
import torch
import torch.nn as nn

# Simulate the ClassEmbedder from the model
class ClassEmbedder(nn.Module):
    def __init__(self, num_classes: int, emb_channels: int):
        super().__init__()
        self.emb = nn.Embedding(num_classes, emb_channels)
        nn.init.normal_(self.emb.weight, std=0.02)
        self.emb_channels = emb_channels

    def forward(self, labels, shape_hw: tuple, device):
        """
        NOTE: forward() requires 3 arguments!
        - labels: tensor of class indices
        - shape_hw: tuple (H, W)
        - device: torch device
        """
        if labels is None:
            return torch.zeros((1, self.emb_channels, *shape_hw), device=device)
        mask_uncond = (labels == -1)
        labels_clamped = labels.clamp_min(0)
        v = self.emb(labels_clamped)  # [B, C_emb]
        v[mask_uncond] = 0
        v = v[..., None, None]
        return v.expand(-1, -1, shape_hw[0], shape_hw[1])


print("=" * 80)
print("Testing ClassEmbedder usage")
print("=" * 80)

# Create a ClassEmbedder instance
num_classes = 10
emb_dim = 128
embedder = ClassEmbedder(num_classes, emb_dim)

label = torch.tensor([5])
device = torch.device('cpu')

print("\n1. Testing CORRECT usage - forward() with 3 args:")
try:
    result = embedder(label, (28, 28), device)
    print(f"   ✓ SUCCESS: shape = {result.shape}")
except Exception as e:
    print(f"   ✗ FAILED: {e}")

print("\n2. Testing INCORRECT usage - calling with 1 arg (as in embeddings.py line 309):")
try:
    result = embedder(label)  # This is what embeddings.py does!
    print(f"   ✓ SUCCESS: shape = {result.shape}")
except TypeError as e:
    print(f"   ✗ FAILED: {e}")
    print("   This is the BUG in embeddings.py!")

print("\n3. Testing direct embedding access - emb(label):")
try:
    result = embedder.emb(label)
    print(f"   ✓ SUCCESS: shape = {result.shape}")
    print(f"   This returns a 2D tensor [batch, emb_dim]")
except Exception as e:
    print(f"   ✗ FAILED: {e}")

print("\n" + "=" * 80)
print("DIAGNOSIS:")
print("=" * 80)
print("The embeddings.py code has TWO options:")
print("1. Call model.class_embed.emb(label) to get [B, D] embedding")
print("2. Call model.class_embed(label, (H, W), device) to get [B, D, H, W] embedding")
print("\nCurrently it's calling model.class_embed(label) which is WRONG!")
print("=" * 80)
