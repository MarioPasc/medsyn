#!/usr/bin/env python3
"""
Demonstration script for per-class weight temperature scaling.

Shows how different temperature values affect the weight distribution
for imbalanced datasets.
"""
import numpy as np


def compute_class_weights_from_counts(
    counts,
    temperature: float = 1.0,
    normalize: bool = True,
):
    """
    Compute per-class loss weights from class sample counts with temperature scaling.
    """
    counts = np.asarray(counts)
    
    if np.any(counts < 0):
        raise ValueError(f"counts must be non-negative, got: {counts}")

    if np.all(counts == 0):
        raise ValueError("All class counts are zero, cannot compute weights")

    # Compute frequency
    total = float(counts.sum())
    freq = counts.astype("float64") / total

    # Compute inverse frequency
    inv_freq = 1.0 / np.maximum(freq, 1e-8)

    # Apply temperature scaling
    if temperature != 1.0:
        weights = np.power(inv_freq, temperature)
    else:
        weights = inv_freq

    # Normalize to mean=1.0
    if normalize:
        weights = weights / weights.mean()

    return weights


def demo_temperature_effect():
    """Demonstrate how temperature affects class weight distribution."""
    
    # Example 1: Moderate imbalance (10:1 ratio)
    print("=" * 80)
    print("EXAMPLE 1: Moderate Imbalance")
    print("=" * 80)
    counts_moderate = np.array([900, 90, 10])
    print(f"Class counts: {counts_moderate}")
    print(f"Imbalance ratios: 90:9:1\n")
    
    for temp in [0.5, 1.0, 1.5, 2.0]:
        weights = compute_class_weights_from_counts(counts_moderate, temperature=temp)
        print(f"Temperature = {temp:.1f}")
        print(f"  Weights: [{weights[0]:.3f}, {weights[1]:.3f}, {weights[2]:.3f}]")
        print(f"  Weight ratio (class 2 / class 0): {weights[2] / weights[0]:.2f}x")
        print()
    
    # Example 2: Severe imbalance (100:1 ratio)
    print("=" * 80)
    print("EXAMPLE 2: Severe Imbalance")
    print("=" * 80)
    counts_severe = np.array([9500, 450, 50])
    print(f"Class counts: {counts_severe}")
    print(f"Imbalance ratios: 190:9:1\n")
    
    for temp in [0.5, 1.0, 1.5, 2.0]:
        weights = compute_class_weights_from_counts(counts_severe, temperature=temp)
        print(f"Temperature = {temp:.1f}")
        print(f"  Weights: [{weights[0]:.3f}, {weights[1]:.3f}, {weights[2]:.3f}]")
        print(f"  Weight ratio (class 2 / class 0): {weights[2] / weights[0]:.2f}x")
        print()
    
    # Example 3: PathMNIST-like distribution (9 classes)
    print("=" * 80)
    print("EXAMPLE 3: PathMNIST-like Distribution (9 classes)")
    print("=" * 80)
    counts_pathmnist = np.array([2000, 1800, 1500, 1200, 800, 600, 400, 200, 100])
    print(f"Class counts: {counts_pathmnist}")
    print(f"Most common class: {counts_pathmnist[0]}, Rarest class: {counts_pathmnist[-1]}")
    print(f"Imbalance ratio: {counts_pathmnist[0] / counts_pathmnist[-1]:.1f}:1\n")
    
    for temp in [0.5, 1.0, 1.5, 2.0]:
        weights = compute_class_weights_from_counts(counts_pathmnist, temperature=temp)
        print(f"Temperature = {temp:.1f}")
        weights_str = ", ".join([f"{w:.2f}" for w in weights])
        print(f"  Weights: [{weights_str}]")
        print(f"  Min weight: {weights.min():.3f}, Max weight: {weights.max():.3f}")
        print(f"  Weight range: {weights.max() / weights.min():.2f}x")
        print()
    
    print("=" * 80)
    print("INTERPRETATION:")
    print("=" * 80)
    print("• Temperature < 1.0: More uniform weights (less aggressive reweighting)")
    print("  - Use when: Small imbalance or when you want gradual rebalancing")
    print("  - Effect: Minority classes get modest boost\n")
    
    print("• Temperature = 1.0: Standard inverse frequency weighting")
    print("  - Use when: Default balanced training with typical imbalance")
    print("  - Effect: Linear inverse relationship with class frequency\n")
    
    print("• Temperature > 1.0: More extreme weights (aggressive reweighting)")
    print("  - Use when: Severe imbalance or minority classes very important")
    print("  - Effect: Minority classes get exponential boost")
    print("  - Warning: Very high temperatures (>2.0) may destabilize training\n")
    
    print("• Recommended ranges:")
    print("  - Mild imbalance (2-5x):   temp ∈ [0.8, 1.2]")
    print("  - Moderate imbalance (5-20x): temp ∈ [1.0, 1.5]")
    print("  - Severe imbalance (>20x):    temp ∈ [1.2, 2.0]")
    print("=" * 80)


if __name__ == "__main__":
    demo_temperature_effect()
