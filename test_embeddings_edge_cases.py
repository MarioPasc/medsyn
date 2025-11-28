#!/usr/bin/env python3
"""
Comprehensive test for edge cases in embeddings.py
Tests shape handling, empty values, and boundary conditions.
"""

import torch
import torch.nn as nn
import numpy as np

print("=" * 80)
print("Testing Tensor Shape Edge Cases")
print("=" * 80)

# Test 1: Different activation dimensions
print("\n1. Testing activation shape handling:")
test_shapes = [
    (1,),           # 1D - edge case
    (4, 128),       # 2D - already flat
    (4, 128, 8),    # 3D - sequence
    (4, 128, 8, 8), # 4D - spatial
    (4, 128, 8, 8, 3), # 5D - edge case
]

for shape in test_shapes:
    act = torch.randn(*shape)
    dims = act.dim()
    print(f"   Shape {shape} -> dim()={dims}")

    # Simulate the pooling logic from embeddings.py:214-231
    try:
        if dims == 4:
            feat = act.mean(dim=[2, 3])
            print(f"      ✓ 4D pooling: {shape} -> {feat.shape}")
        elif dims == 3:
            feat = act.mean(dim=1)
            print(f"      ✓ 3D pooling: {shape} -> {feat.shape}")
        elif dims == 2:
            feat = act
            print(f"      ✓ 2D no pooling: {shape} -> {feat.shape}")
        else:
            feat = act.flatten(1) if dims > 1 else act.unsqueeze(0)
            print(f"      ! Edge case: {shape} -> {feat.shape}")
    except Exception as e:
        print(f"      ✗ ERROR: {e}")

# Test 2: Empty tensors and edge cases
print("\n2. Testing empty/edge case tensors:")

test_cases = [
    ("Empty batch", torch.randn(0, 128, 8, 8)),
    ("Single sample", torch.randn(1, 128, 8, 8)),
    ("Zero channels", torch.randn(4, 0, 8, 8)),
    ("1x1 spatial", torch.randn(4, 128, 1, 1)),
]

for name, tensor in test_cases:
    print(f"   {name}: shape={tensor.shape}")
    try:
        if tensor.dim() == 4:
            pooled = tensor.mean(dim=[2, 3])
            print(f"      ✓ Pooled: {pooled.shape}")
        else:
            print(f"      - Skipped (not 4D)")
    except Exception as e:
        print(f"      ✗ ERROR: {e}")

# Test 3: Index out of bounds
print("\n3. Testing list indexing edge cases:")

def safe_middle_index(lst):
    """Get middle element safely"""
    if not lst:
        return None
    return lst[len(lst) // 2]

def safe_first_index(lst):
    """Get first element safely"""
    if not lst:
        return None
    return lst[0]

test_lists = [
    ("Empty list", []),
    ("Single item", [100]),
    ("Two items", [100, 200]),
    ("Multiple items", [100, 200, 300, 400, 500]),
]

for name, lst in test_lists:
    mid = safe_middle_index(lst)
    first = safe_first_index(lst)
    print(f"   {name}: middle={mid}, first={first}")

# Test 4: Numpy stacking with different shapes
print("\n4. Testing numpy stacking consistency:")

test_stack_cases = [
    ("Same shapes", [np.array([1, 2, 3]), np.array([4, 5, 6])]),
    ("Different shapes", [np.array([1, 2, 3]), np.array([4, 5])]),
    ("Empty list", []),
    ("Single array", [np.array([1, 2, 3])]),
]

for name, arrays in test_stack_cases:
    print(f"   {name}:")
    try:
        if arrays:
            stacked = np.stack(arrays)
            print(f"      ✓ Stacked: shape={stacked.shape}")
        else:
            print(f"      ! Empty: cannot stack")
    except ValueError as e:
        print(f"      ✗ ERROR: {e}")

# Test 5: Device consistency
print("\n5. Testing device handling:")
device = torch.device('cpu')
x0 = torch.randn(1, 3, 28, 28, device=device)

# Check that tensors created with correct device
t1 = torch.tensor([5], device=device)
t2 = torch.randn_like(x0)  # Should inherit device
t3 = torch.zeros(1, 128, device=device)

print(f"   x0 device: {x0.device}")
print(f"   tensor([5], device=device): {t1.device}")
print(f"   randn_like(x0): {t2.device}")
print(f"   zeros(..., device=device): {t3.device}")

all_same = (x0.device == t1.device == t2.device == t3.device)
print(f"   ✓ All on same device: {all_same}")

print("\n" + "=" * 80)
print("EDGE CASE RECOMMENDATIONS:")
print("=" * 80)
print("1. Add validation for empty layer_names and timesteps lists")
print("2. Add check for empty records before np.stack()")
print("3. Handle 1D and 5D+ activations explicitly")
print("4. Add shape validation for feature vectors before stacking")
print("=" * 80)
