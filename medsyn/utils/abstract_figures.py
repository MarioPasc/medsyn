#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Abstract figures generation for paper/presentation materials.

This script generates:
  (i) Forward diffusion process visualization: takes a random image from a 
      specified class and simulates the forward Markov process for 5 steps,
      saving images separately.
  (ii) Per-class balance barplot: a clean visualization showing class imbalance
       with representative images per class, highlighting the 3 minority classes.

Usage:
    python -m medsyn.utils.abstract_figures --config config/medsyn_cfg.yaml
    python -m medsyn.utils.abstract_figures --config config/medsyn_cfg.yaml --output-dir docs/paper-related/abstract
    python -m medsyn.utils.abstract_figures --config config/medsyn_cfg.yaml --class-id 4 --seed 42
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import torch
import yaml
from PIL import Image

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

# Default PathMNIST class names
PATHMNIST_CLASS_NAMES: Dict[int, str] = {
    0: "Adipose",
    1: "Background",
    2: "Debris",
    3: "Lymphocytes",
    4: "Mucus",
    5: "Smooth Muscle",
    6: "Normal Colon Mucosa",
    7: "Cancer-Associated Stroma",
    8: "Colorectal Adenocarcinoma Epithelium",
}


def load_config(config_path: Path) -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_npz_path(cfg: dict) -> Path:
    """
    Extract NPZ path from configuration.
    
    Priority:
      1. data.postprocess_npz.npz_path (if enabled)
      2. Fallback to download_dir/pathmnist.npz
    """
    data_cfg = cfg.get("data", {})
    
    # Check postprocess_npz first
    postprocess = data_cfg.get("postprocess_npz", {})
    if postprocess.get("enabled", False):
        npz_path = postprocess.get("npz_path")
        if npz_path and Path(npz_path).exists():
            return Path(npz_path)
    
    # Fallback to download_dir
    download_dir = data_cfg.get("download_dir", "./data_raw")
    flag = data_cfg.get("flag", "pathmnist")
    return Path(download_dir) / f"{flag}.npz"


def load_dataset(npz_path: Path) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """
    Load dataset from NPZ file.
    
    Returns:
        Tuple of (all_images, all_labels, split_data_dict)
    """
    if not npz_path.exists():
        raise FileNotFoundError(f"NPZ file not found: {npz_path}")
    
    logger.info(f"Loading dataset from {npz_path}")
    data = np.load(str(npz_path))
    
    # Collect all splits
    split_data = {}
    all_images = []
    all_labels = []
    
    for split in ["train", "val", "test"]:
        images_key = f"{split}_images"
        labels_key = f"{split}_labels"
        
        if images_key in data and labels_key in data:
            imgs = data[images_key]
            lbls = data[labels_key].reshape(-1)
            split_data[split] = {"images": imgs, "labels": lbls}
            all_images.append(imgs)
            all_labels.append(lbls)
            logger.info(f"  {split}: {len(imgs)} images")
    
    all_images = np.concatenate(all_images, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    return all_images, all_labels, split_data


def get_random_image_from_class(
    images: np.ndarray,
    labels: np.ndarray,
    class_id: int,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Get a random image from a specific class.
    
    Args:
        images: All images [N, H, W, C]
        labels: All labels [N]
        class_id: Target class ID
        seed: Random seed for reproducibility
    
    Returns:
        Single image [H, W, C] uint8
    """
    if seed is not None:
        np.random.seed(seed)
    
    class_mask = labels == class_id
    class_indices = np.where(class_mask)[0]
    
    if len(class_indices) == 0:
        raise ValueError(f"No images found for class {class_id}")
    
    selected_idx = np.random.choice(class_indices)
    return images[selected_idx]


def get_representative_image_per_class(
    images: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
    seed: Optional[int] = None
) -> Dict[int, np.ndarray]:
    """
    Get one representative image per class.
    
    Returns:
        Dictionary mapping class_id -> image [H, W, C]
    """
    if seed is not None:
        np.random.seed(seed)
    
    class_images = {}
    for class_id in range(num_classes):
        class_mask = labels == class_id
        class_indices = np.where(class_mask)[0]
        
        if len(class_indices) > 0:
            selected_idx = np.random.choice(class_indices)
            class_images[class_id] = images[selected_idx]
    
    return class_images


def sample_proportional_images(
    images: np.ndarray,
    labels: np.ndarray,
    total_samples: int = 20,
    seed: Optional[int] = None
) -> List[Tuple[np.ndarray, int]]:
    """
    Sample images proportionally to class distribution.
    
    The number of images sampled per class is proportional to the
    count of that class in the dataset.
    
    Args:
        images: All images [N, H, W, C]
        labels: All labels [N]
        total_samples: Total number of images to sample
        seed: Random seed for reproducibility
    
    Returns:
        List of (image, class_id) tuples
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Count samples per class
    unique_classes, counts = np.unique(labels, return_counts=True)
    total_count = counts.sum()
    
    # Calculate proportional samples per class
    proportions = counts / total_count
    samples_per_class = np.round(proportions * total_samples).astype(int)
    
    # Adjust to ensure we get exactly total_samples
    diff = total_samples - samples_per_class.sum()
    if diff != 0:
        # Add/remove from the largest class
        largest_class_idx = np.argmax(counts)
        samples_per_class[largest_class_idx] += diff
    
    # Sample images
    sampled = []
    for cls_idx, cls in enumerate(unique_classes):
        n_samples = samples_per_class[cls_idx]
        if n_samples <= 0:
            continue
        
        class_mask = labels == cls
        class_indices = np.where(class_mask)[0]
        
        # Sample with replacement if needed
        replace = n_samples > len(class_indices)
        selected_indices = np.random.choice(class_indices, size=n_samples, replace=replace)
        
        for idx in selected_indices:
            sampled.append((images[idx], int(cls)))
    
    # Shuffle the final list
    np.random.shuffle(sampled)
    
    return sampled


def save_proportional_samples(
    images: np.ndarray,
    labels: np.ndarray,
    output_dir: Path,
    total_samples: int = 20,
    seed: Optional[int] = None
) -> List[Path]:
    """
    Sample and save images proportionally to class distribution.
    
    Args:
        images: All images [N, H, W, C]
        labels: All labels [N]
        output_dir: Output directory for samples
        total_samples: Total number of images to sample
        seed: Random seed for reproducibility
    
    Returns:
        List of saved file paths
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get proportional samples
    sampled = sample_proportional_images(images, labels, total_samples, seed)
    
    # Log distribution
    class_counts = {}
    for _, cls in sampled:
        class_counts[cls] = class_counts.get(cls, 0) + 1
    logger.info(f"Sampled {len(sampled)} images with distribution: {dict(sorted(class_counts.items()))}")
    
    # Save each image
    saved_paths = []
    for i, (img, cls) in enumerate(sampled):
        filename = f"sample_{i:02d}_class{cls}.svg"
        filepath = output_dir / filename
        
        # Save as SVG using matplotlib
        fig_img, ax_img = plt.subplots(figsize=(2, 2))
        ax_img.imshow(img)
        ax_img.axis('off')
        fig_img.savefig(filepath, format='svg', bbox_inches='tight', pad_inches=0,
                        facecolor='white', edgecolor='none')
        plt.close(fig_img)
        saved_paths.append(filepath)
    
    return saved_paths


def simulate_forward_diffusion(
    image: np.ndarray,
    num_steps: int = 5,
    num_train_timesteps: int = 1000,
    beta_start: float = 1e-4,
    beta_end: float = 2e-2,
    beta_schedule: str = "squaredcos_cap_v2"
) -> List[np.ndarray]:
    """
    Simulate the forward Markov process of diffusion.
    
    The forward process progressively adds noise to the image following:
        x_t = sqrt(α̅_t) * x_0 + sqrt(1 - α̅_t) * ε
    
    Args:
        image: Original image [H, W, C] uint8
        num_steps: Number of diffusion steps to visualize
        num_train_timesteps: Total number of timesteps in the scheduler
        beta_start: Starting beta value
        beta_end: Ending beta value
        beta_schedule: Beta schedule type
    
    Returns:
        List of images at each timestep, starting with original
    """
    # Convert to tensor and normalize to [-1, 1]
    x0 = torch.from_numpy(image).float() / 255.0
    x0 = x0 * 2.0 - 1.0  # [0,1] -> [-1,1]
    x0 = x0.permute(2, 0, 1).unsqueeze(0)  # [1, C, H, W]
    
    # Compute betas and alphas
    if beta_schedule == "squaredcos_cap_v2":
        # Cosine schedule from Nichol & Dhariwal (2021)
        def alpha_bar(t):
            return np.cos((t + 0.008) / 1.008 * np.pi / 2) ** 2
        
        betas = []
        for i in range(num_train_timesteps):
            t1 = i / num_train_timesteps
            t2 = (i + 1) / num_train_timesteps
            beta = min(1 - alpha_bar(t2) / alpha_bar(t1), 0.999)
            betas.append(beta)
        betas = torch.tensor(betas, dtype=torch.float32)
    else:
        # Linear schedule
        betas = torch.linspace(beta_start, beta_end, num_train_timesteps)
    
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    
    # Sample timesteps evenly across the diffusion process
    timesteps = np.linspace(0, num_train_timesteps - 1, num_steps + 1, dtype=int)[1:]
    
    # Generate fixed noise
    noise = torch.randn_like(x0)
    
    # Generate noised images
    results = []
    
    # Add original image first
    original = ((x0.squeeze(0).permute(1, 2, 0) + 1) / 2 * 255).numpy().astype(np.uint8)
    results.append(original)
    
    for t in timesteps:
        sqrt_alpha_prod = alphas_cumprod[t].sqrt()
        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[t]).sqrt()
        
        # Forward diffusion: x_t = sqrt(α̅_t) * x_0 + sqrt(1 - α̅_t) * ε
        x_t = sqrt_alpha_prod * x0 + sqrt_one_minus_alpha_prod * noise
        
        # Convert back to image
        x_t_img = x_t.squeeze(0).permute(1, 2, 0)  # [H, W, C]
        x_t_img = (x_t_img + 1) / 2  # [-1,1] -> [0,1]
        x_t_img = torch.clamp(x_t_img, 0, 1)
        x_t_img = (x_t_img * 255).numpy().astype(np.uint8)
        
        results.append(x_t_img)
    
    return results


def save_forward_diffusion_images(
    images: List[np.ndarray],
    output_dir: Path,
    class_id: int,
    prefix: str = "forward_diffusion"
) -> List[Path]:
    """
    Save forward diffusion images separately.
    
    Args:
        images: List of images at each timestep
        output_dir: Output directory
        class_id: Class ID for naming
        prefix: Filename prefix
    
    Returns:
        List of saved file paths
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = []
    
    for i, img in enumerate(images):
        if i == 0:
            filename = f"{prefix}_class{class_id}_t0_original.svg"
        else:
            filename = f"{prefix}_class{class_id}_step{i}.svg"
        
        filepath = output_dir / filename
        # Save as SVG using matplotlib for vector format
        fig_img, ax_img = plt.subplots(figsize=(2, 2))
        ax_img.imshow(img)
        ax_img.axis('off')
        fig_img.savefig(filepath, format='svg', bbox_inches='tight', pad_inches=0,
                        facecolor='white', edgecolor='none')
        plt.close(fig_img)
        saved_paths.append(filepath)
        logger.info(f"  Saved: {filepath.name}")
    
    return saved_paths


def create_class_balance_barplot(
    labels: np.ndarray,
    class_images: Dict[int, np.ndarray],
    output_path: Path,
    class_names: Optional[Dict[int, str]] = None,
    num_minority_classes: int = 3
) -> None:
    """
    Create a clean barplot showing per-class distribution with representative images.
    
    Features:
    - No x-axis labels, replaced by representative images
    - Last 3 classes (by count) highlighted in orange
    - No y-axis counts, no grid - minimal clean design
    
    Args:
        labels: All labels [N]
        class_images: Dictionary mapping class_id -> representative image
        output_path: Path to save the figure
        class_names: Optional class name mapping
        num_minority_classes: Number of minority classes to highlight (default: 3)
    """
    if class_names is None:
        class_names = PATHMNIST_CLASS_NAMES
    
    # Count samples per class
    unique_classes, counts = np.unique(labels, return_counts=True)
    num_classes = len(unique_classes)
    
    # Sort classes by count (descending)
    sorted_indices = np.argsort(counts)[::-1]
    sorted_classes = unique_classes[sorted_indices]
    sorted_counts = counts[sorted_indices]
    
    # Identify minority classes (last N by count)
    minority_set = set(sorted_classes[-num_minority_classes:])
    
    # Assign colors: blue for majority, orange for minority
    colors = []
    for cls in sorted_classes:
        if cls in minority_set:
            colors.append('#E69F00')  # Orange for minority
        else:
            colors.append('#0072B2')  # Blue for majority
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create bars
    x_positions = np.arange(num_classes)
    bars = ax.bar(x_positions, sorted_counts, color=colors, width=0.7, edgecolor='none')
    
    # Set y-axis limits to zoom in on the imbalance
    # Start from min(counts) - 100 to emphasize differences
    y_min = max(0, sorted_counts.min() - 100)
    y_max = sorted_counts.max() * 1.05  # 5% padding at top
    ax.set_ylim(y_min, y_max)
    
    # Clean axis styling - keep left and bottom spines
    ax.set_xticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    
    # Y-axis ticks without labels
    ax.tick_params(axis='y', which='both', left=True, labelleft=False)
    
    # Add representative images below bars
    # Calculate image size based on figure dimensions
    img_size = 0.065  # Fraction of figure width per image
    
    for i, cls in enumerate(sorted_classes):
        if cls in class_images:
            img = class_images[cls]
            
            # Create inset axes for the image
            # Position: centered below each bar
            x_center = x_positions[i] / (num_classes - 1) if num_classes > 1 else 0.5
            # Adjust for axes position within figure
            x_fig = 0.189 + x_center * 0.65 - img_size / 2  # Shifted left
            y_fig = 0.07  # Move up
            
            img_ax = fig.add_axes((x_fig, y_fig, img_size, img_size))
            img_ax.imshow(img)
            img_ax.axis('off')
            
            # Add colored border matching bar color
            for spine in img_ax.spines.values():
                spine.set_visible(True)
                spine.set_color(colors[i])
                spine.set_linewidth(2)
    
    # Adjust layout
    plt.subplots_adjust(bottom=0.15)
    
    # Save figure as SVG
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Ensure output has .svg extension
    output_path = output_path.with_suffix('.svg')
    plt.savefig(output_path, format='svg', bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    
    logger.info(f"Saved class balance barplot: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate abstract figures for paper/presentation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--config", "-c",
        type=Path,
        default=Path("config/medsyn_cfg.yaml"),
        help="Path to medsyn configuration YAML"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=None,
        help="Output directory (default: docs/paper-related/abstract)"
    )
    parser.add_argument(
        "--class-id",
        type=int,
        default=8,
        help="Class ID to use for forward diffusion visualization"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--num-diffusion-steps",
        type=int,
        default=5,
        help="Number of forward diffusion steps to visualize"
    )
    parser.add_argument(
        "--skip-diffusion",
        action="store_true",
        help="Skip forward diffusion figure generation"
    )
    parser.add_argument(
        "--skip-barplot",
        action="store_true",
        help="Skip class balance barplot generation"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=20,
        help="Number of proportional samples to generate"
    )
    parser.add_argument(
        "--skip-samples",
        action="store_true",
        help="Skip proportional samples generation"
    )
    
    args = parser.parse_args()
    
    # Set default output directory
    if args.output_dir is None:
        args.output_dir = Path("docs/paper-related/abstract")
    
    # Load configuration
    logger.info(f"Loading configuration from {args.config}")
    cfg = load_config(args.config)
    
    # Get NPZ path
    npz_path = get_npz_path(cfg)
    logger.info(f"Using NPZ path: {npz_path}")
    
    # Load dataset
    all_images, all_labels, split_data = load_dataset(npz_path)
    num_classes = len(np.unique(all_labels))
    logger.info(f"Total: {len(all_images)} images, {num_classes} classes")
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # (i) Forward Diffusion Visualization
    if not args.skip_diffusion:
        logger.info(f"\n--- Forward Diffusion Visualization (class {args.class_id}) ---")
        
        # Get a random image from the specified class
        sample_image = get_random_image_from_class(
            all_images, all_labels, args.class_id, seed=args.seed
        )
        logger.info(f"Selected image shape: {sample_image.shape}")
        
        # Get scheduler config from yaml
        sched_cfg = cfg.get("ccddpm", {}).get("sched", {})
        
        # Simulate forward diffusion
        diffusion_images = simulate_forward_diffusion(
            sample_image,
            num_steps=args.num_diffusion_steps,
            num_train_timesteps=sched_cfg.get("num_train_timesteps", 1000),
            beta_start=sched_cfg.get("beta_start", 1e-4),
            beta_end=sched_cfg.get("beta_end", 2e-2),
            beta_schedule=sched_cfg.get("beta_schedule", "squaredcos_cap_v2")
        )
        
        # Save images separately
        diffusion_dir = args.output_dir / "forward_diffusion"
        saved_paths = save_forward_diffusion_images(
            diffusion_images, diffusion_dir, args.class_id
        )
        logger.info(f"Saved {len(saved_paths)} forward diffusion images to {diffusion_dir}")
    
    # (ii) Class Balance Barplot
    if not args.skip_barplot:
        logger.info("\n--- Class Balance Barplot ---")
        
        # Get representative images per class
        class_images = get_representative_image_per_class(
            all_images, all_labels, num_classes, seed=args.seed
        )
        
        # Create barplot
        barplot_path = args.output_dir / "class_balance_barplot.svg"
        create_class_balance_barplot(
            all_labels,
            class_images,
            barplot_path,
            class_names=PATHMNIST_CLASS_NAMES,
            num_minority_classes=3
        )
    
    # (iii) Proportional Samples
    if not args.skip_samples:
        logger.info(f"\n--- Proportional Samples ({args.num_samples} images) ---")
        
        samples_dir = args.output_dir / "samples"
        saved_samples = save_proportional_samples(
            all_images,
            all_labels,
            samples_dir,
            total_samples=args.num_samples,
            seed=args.seed
        )
        logger.info(f"Saved {len(saved_samples)} sample images to {samples_dir}")
    
    logger.info(f"\n✓ All figures saved to {args.output_dir}")


if __name__ == "__main__":
    main()
