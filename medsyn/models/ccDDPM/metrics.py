# medsyn/models/ccDDPM/metrics.py
# Purpose: Generation and training metrics for synthesis and interpretability.
# Includes: FID via clean-fid (optional), LPIPS (optional), per-class loss export,
# PSNR, SSIM (global and per-class).
from __future__ import annotations
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import logging
import math
import tempfile
import shutil
import torch
import torch.nn.functional as F
import numpy as np

logger = logging.getLogger(__name__)

def compute_psnr(x_hat: torch.Tensor, x: torch.Tensor, max_val: float = 1.0) -> float:
    """
    Compute Peak Signal-to-Noise Ratio between reconstructed and original images.
    Assumes inputs are in [0, max_val] range.
    
    Args:
        x_hat: Reconstructed images [B, C, H, W]
        x: Original images [B, C, H, W]
        max_val: Maximum possible pixel value (1.0 for normalized, 255 for uint8)
    
    Returns:
        PSNR in dB
    """
    mse = F.mse_loss(x_hat, x, reduction="mean")
    if mse < 1e-10:
        return 100.0  # Perfect reconstruction
    psnr = 20 * torch.log10(torch.tensor(max_val, device=mse.device) / torch.sqrt(mse))
    return float(psnr.cpu())

def compute_ssim(x_hat: torch.Tensor, x: torch.Tensor, window_size: int = 11, max_val: float = 1.0) -> float:
    """
    Compute Structural Similarity Index Measure between images.
    Simplified implementation for batch processing.
    
    Args:
        x_hat: Reconstructed images [B, C, H, W]
        x: Original images [B, C, H, W]
        window_size: Size of the Gaussian window
        max_val: Maximum possible pixel value
    
    Returns:
        Mean SSIM across batch
    """
    try:
        from torchmetrics.functional import structural_similarity_index_measure
        ssim_val = structural_similarity_index_measure(x_hat, x, data_range=max_val)
        return float(ssim_val.cpu())
    except ImportError:
        # Fallback: simple correlation-based approximation
        logger.warning("torchmetrics not available, using simplified SSIM approximation")
        x_mean = x.mean(dim=(2, 3), keepdim=True)
        x_hat_mean = x_hat.mean(dim=(2, 3), keepdim=True)
        x_var = ((x - x_mean) ** 2).mean(dim=(2, 3), keepdim=True)
        x_hat_var = ((x_hat - x_hat_mean) ** 2).mean(dim=(2, 3), keepdim=True)
        covar = ((x - x_mean) * (x_hat - x_hat_mean)).mean(dim=(2, 3), keepdim=True)
        
        c1 = (0.01 * max_val) ** 2
        c2 = (0.03 * max_val) ** 2
        
        ssim = ((2 * x_mean * x_hat_mean + c1) * (2 * covar + c2)) / \
               ((x_mean ** 2 + x_hat_mean ** 2 + c1) * (x_var + x_hat_var + c2))
        return float(ssim.mean().cpu())

def compute_psnr_per_sample(
    x_hat: torch.Tensor,
    x: torch.Tensor,
    max_val: float = 1.0
) -> torch.Tensor:
    """
    Compute PSNR for each sample in the batch.

    Args:
        x_hat: Reconstructed images [B, C, H, W]
        x: Original images [B, C, H, W]
        max_val: Maximum possible pixel value

    Returns:
        Tensor of shape [B] with PSNR for each sample (in dB)
    """
    # Per-sample MSE: mean over C, H, W dimensions
    mse = ((x_hat - x) ** 2).mean(dim=(1, 2, 3))  # [B]
    # Avoid log(0) by clamping MSE
    mse = torch.clamp(mse, min=1e-10)
    psnr = 20 * torch.log10(torch.tensor(max_val, device=mse.device) / torch.sqrt(mse))
    return psnr


def compute_ssim_per_sample(
    x_hat: torch.Tensor,
    x: torch.Tensor,
    max_val: float = 1.0
) -> torch.Tensor:
    """
    Compute SSIM for each sample in the batch.

    Args:
        x_hat: Reconstructed images [B, C, H, W]
        x: Original images [B, C, H, W]
        max_val: Maximum possible pixel value

    Returns:
        Tensor of shape [B] with SSIM for each sample
    """
    try:
        from torchmetrics.functional import structural_similarity_index_measure
        # torchmetrics returns a single value by default, we need per-sample
        # Use a loop for now (could be optimized if needed)
        ssim_vals = []
        for i in range(x_hat.size(0)):
            ssim = structural_similarity_index_measure(
                x_hat[i:i+1], x[i:i+1], data_range=max_val
            )
            ssim_vals.append(ssim)
        return torch.stack(ssim_vals)
    except ImportError:
        # Fallback: simplified SSIM approximation
        logger.warning("torchmetrics not available, using simplified SSIM approximation")
        # Per-sample simplified SSIM
        B = x.size(0)
        x_flat = x.view(B, -1)
        x_hat_flat = x_hat.view(B, -1)

        x_mean = x_flat.mean(dim=1, keepdim=True)
        x_hat_mean = x_hat_flat.mean(dim=1, keepdim=True)
        x_var = ((x_flat - x_mean) ** 2).mean(dim=1, keepdim=True)
        x_hat_var = ((x_hat_flat - x_hat_mean) ** 2).mean(dim=1, keepdim=True)
        covar = ((x_flat - x_mean) * (x_hat_flat - x_hat_mean)).mean(dim=1, keepdim=True)

        c1 = (0.01 * max_val) ** 2
        c2 = (0.03 * max_val) ** 2

        ssim = ((2 * x_mean * x_hat_mean + c1) * (2 * covar + c2)) / \
               ((x_mean ** 2 + x_hat_mean ** 2 + c1) * (x_var + x_hat_var + c2))
        return ssim.squeeze(1)


def compute_per_class_metrics(
    x_hat: torch.Tensor,
    x: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    max_val: float = 1.0,
) -> Dict[str, float]:
    """
    Compute PSNR and SSIM metrics per class.

    Since we work at the tile level (each tile has one class label), we compute
    metrics per sample and then aggregate by class label.

    Args:
        x_hat: Reconstructed images [B, C, H, W]
        x: Original images [B, C, H, W]
        labels: Class labels for each sample [B]
        num_classes: Total number of classes
        max_val: Maximum possible pixel value

    Returns:
        Dictionary with keys like 'psnr_c0', 'ssim_c0', 'psnr_c1', 'ssim_c1', etc.
        Classes not present in the batch will have NaN values.
    """
    B = x_hat.size(0)
    if B == 0:
        return {f"psnr_c{k}": float("nan") for k in range(num_classes)} | \
               {f"ssim_c{k}": float("nan") for k in range(num_classes)}

    # Compute per-sample metrics
    psnr_per_sample = compute_psnr_per_sample(x_hat, x, max_val)
    ssim_per_sample = compute_ssim_per_sample(x_hat, x, max_val)

    labels = labels.to(x_hat.device)
    result: Dict[str, float] = {}

    for k in range(num_classes):
        mask = labels == k
        if mask.sum() > 0:
            result[f"psnr_c{k}"] = float(psnr_per_sample[mask].mean().cpu())
            result[f"ssim_c{k}"] = float(ssim_per_sample[mask].mean().cpu())
        else:
            result[f"psnr_c{k}"] = float("nan")
            result[f"ssim_c{k}"] = float("nan")

    return result


class PerClassMetricsAccumulator:
    """
    Accumulates per-class metrics (PSNR, SSIM, loss) across batches.

    Usage:
        acc = PerClassMetricsAccumulator(num_classes=9)
        for batch in dataloader:
            acc.update(x_hat, x, labels, batch_metrics)
        epoch_metrics = acc.compute()
        acc.reset()
    """

    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.reset()

    def reset(self) -> None:
        """Reset all accumulators."""
        # Per-class sums and counts
        self._psnr_sum = torch.zeros(self.num_classes, dtype=torch.float64)
        self._ssim_sum = torch.zeros(self.num_classes, dtype=torch.float64)
        self._loss_raw_sum = torch.zeros(self.num_classes, dtype=torch.float64)
        self._loss_weighted_sum = torch.zeros(self.num_classes, dtype=torch.float64)
        self._count = torch.zeros(self.num_classes, dtype=torch.long)

        # Global sums
        self._global_psnr_sum = 0.0
        self._global_ssim_sum = 0.0
        self._global_count = 0

    def update(
        self,
        x_hat: torch.Tensor,
        x: torch.Tensor,
        labels: torch.Tensor,
        max_val: float = 1.0,
        loss_raw_per_sample: Optional[torch.Tensor] = None,
        loss_weighted_per_sample: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """
        Update accumulators with a batch of samples.

        Args:
            x_hat: Reconstructed images [B, C, H, W]
            x: Original images [B, C, H, W]
            labels: Class labels [B]
            max_val: Maximum pixel value for PSNR/SSIM
            loss_raw_per_sample: Optional raw loss per sample [B]
            loss_weighted_per_sample: Optional weighted loss per sample [B]

        Returns:
            Batch metrics dictionary for immediate logging.
        """
        B = x_hat.size(0)
        if B == 0:
            return {}

        # Compute per-sample metrics
        with torch.no_grad():
            psnr_per_sample = compute_psnr_per_sample(x_hat, x, max_val)
            ssim_per_sample = compute_ssim_per_sample(x_hat, x, max_val)

        # Move to CPU for accumulation
        psnr_cpu = psnr_per_sample.detach().cpu().to(torch.float64)
        ssim_cpu = ssim_per_sample.detach().cpu().to(torch.float64)
        labels_cpu = labels.detach().cpu()

        # Update global accumulators
        self._global_psnr_sum += float(psnr_cpu.sum())
        self._global_ssim_sum += float(ssim_cpu.sum())
        self._global_count += B

        # Update per-class accumulators
        for k in range(self.num_classes):
            mask = labels_cpu == k
            if mask.sum() > 0:
                idxs = mask.nonzero(as_tuple=False).view(-1)
                self._psnr_sum[k] += psnr_cpu[idxs].sum()
                self._ssim_sum[k] += ssim_cpu[idxs].sum()
                self._count[k] += int(idxs.numel())

                if loss_raw_per_sample is not None:
                    loss_raw_cpu = loss_raw_per_sample.detach().cpu().to(torch.float64)
                    self._loss_raw_sum[k] += loss_raw_cpu[idxs].sum()
                if loss_weighted_per_sample is not None:
                    loss_wgt_cpu = loss_weighted_per_sample.detach().cpu().to(torch.float64)
                    self._loss_weighted_sum[k] += loss_wgt_cpu[idxs].sum()

        # Return batch metrics for immediate logging
        batch_metrics = {
            "psnr": float(psnr_cpu.mean()),
            "ssim": float(ssim_cpu.mean()),
        }
        return batch_metrics

    def compute(self) -> Dict[str, float]:
        """
        Compute final per-class and global metrics.

        Returns:
            Dictionary with global metrics (psnr, ssim) and per-class metrics
            (psnr_c0, ssim_c0, loss_raw_c0, loss_weighted_c0, etc.)
        """
        result: Dict[str, float] = {}

        # Global metrics
        if self._global_count > 0:
            result["psnr"] = self._global_psnr_sum / self._global_count
            result["ssim"] = self._global_ssim_sum / self._global_count
        else:
            result["psnr"] = float("nan")
            result["ssim"] = float("nan")

        # Per-class metrics
        for k in range(self.num_classes):
            n = int(self._count[k].item())
            if n > 0:
                result[f"psnr_c{k}"] = float(self._psnr_sum[k] / n)
                result[f"ssim_c{k}"] = float(self._ssim_sum[k] / n)
                if self._loss_raw_sum[k] > 0:
                    result[f"loss_raw_c{k}"] = float(self._loss_raw_sum[k] / n)
                if self._loss_weighted_sum[k] > 0:
                    result[f"loss_weighted_c{k}"] = float(self._loss_weighted_sum[k] / n)
            else:
                result[f"psnr_c{k}"] = float("nan")
                result[f"ssim_c{k}"] = float("nan")

        return result

    def get_class_counts(self) -> Dict[int, int]:
        """Return sample counts per class."""
        return {k: int(self._count[k].item()) for k in range(self.num_classes)}


def compute_class_weight_correlation(
    class_weights: torch.Tensor,
    per_class_metrics: Dict[str, float],
    metric_name: str = "psnr"
) -> float:
    """
    Compute Pearson correlation between class weights and a per-class metric.

    This helps analyze whether class-reweighting is benefiting under-represented
    classes (higher weights should correlate with improvements).

    Args:
        class_weights: Inverse frequency weights [num_classes]
        per_class_metrics: Dictionary with per-class metrics (e.g., psnr_c0, psnr_c1, ...)
        metric_name: Metric to correlate with weights ('psnr' or 'ssim')

    Returns:
        Pearson correlation coefficient, or NaN if insufficient data.
    """
    num_classes = len(class_weights)
    weights = class_weights.cpu().numpy()

    # Extract metric values
    metric_vals = []
    valid_weights = []
    for k in range(num_classes):
        key = f"{metric_name}_c{k}"
        if key in per_class_metrics and not math.isnan(per_class_metrics[key]):
            metric_vals.append(per_class_metrics[key])
            valid_weights.append(weights[k])

    if len(metric_vals) < 3:  # Need at least 3 points for meaningful correlation
        return float("nan")

    metric_arr = np.array(metric_vals)
    weight_arr = np.array(valid_weights)

    # Compute Pearson correlation
    if np.std(metric_arr) < 1e-10 or np.std(weight_arr) < 1e-10:
        return float("nan")

    corr = np.corrcoef(weight_arr, metric_arr)[0, 1]
    return float(corr) if np.isfinite(corr) else float("nan")


def try_fid(real_dir: str | Path, fake_dir: str | Path, device: str = "cuda") -> Optional[float]:
    """
    Compute FID with clean-fid if installed. Returns None if not available.
    """
    try:
        from cleanfid import fid # type: ignore
    except Exception as e:
        logger.warning("clean-fid not available: %s", e)
        return None
    score = fid.compute_fid(str(real_dir), str(fake_dir), device=device)
    return float(score)

def try_lpips(x: torch.Tensor, y: torch.Tensor, net: str = "alex", device: str = "cuda") -> Optional[torch.Tensor]:
    """
    Compute LPIPS if installed. x,y in [0,1]. Returns mean score or None.
    """
    try:
        import lpips  # type: ignore
    except Exception as e:
        logger.warning("lpips not available: %s", e)
        return None
    loss_fn = lpips.LPIPS(net=net).to(device)
    with torch.no_grad():
        s = loss_fn(x.to(device), y.to(device))
    return s.mean()

def export_per_class_loss(loss_tracker, out_csv: Path) -> None:
    """
    Dump per-class training loss means to CSV with columns [class, raw_loss, weighted_loss].

    Uses the loss_tracker's per_class_table() method which returns a pandas DataFrame
    with both raw and weighted per-class losses.
    """
    table = loss_tracker.per_class_table()
    table.to_csv(out_csv, index=False)


# =============================================================================
# FID COMPUTATION (GLOBAL AND PER-CLASS)
# =============================================================================

def compute_fid_from_tensors(
    real_images: torch.Tensor,
    fake_images: torch.Tensor,
    device: str = "cuda",
    batch_size: int = 64,
) -> Optional[float]:
    """
    Compute FID between two sets of images provided as tensors.

    Uses torch-fidelity or clean-fid under the hood. Images are temporarily
    saved to disk for computation.

    Args:
        real_images: Real images tensor [N, C, H, W] in [0, 1] range
        fake_images: Generated images tensor [M, C, H, W] in [0, 1] range
        device: Device for computation
        batch_size: Batch size for feature extraction

    Returns:
        FID score, or None if computation fails.
    """
    try:
        import torch_fidelity
        HAS_TORCH_FIDELITY = True
    except ImportError:
        HAS_TORCH_FIDELITY = False

    if not HAS_TORCH_FIDELITY:
        logger.warning("torch-fidelity not available for FID computation")
        return None

    # Create temporary directories
    real_dir = Path(tempfile.mkdtemp(prefix="fid_real_"))
    fake_dir = Path(tempfile.mkdtemp(prefix="fid_fake_"))

    try:
        # Save images to disk
        _save_images_to_dir(real_images, real_dir)
        _save_images_to_dir(fake_images, fake_dir)

        # Compute FID using torch-fidelity
        metrics = torch_fidelity.calculate_metrics(
            input1=str(real_dir),
            input2=str(fake_dir),
            cuda=(device != "cpu"),
            fid=True,
            batch_size=batch_size,
            verbose=False,
        )
        return float(metrics.get("frechet_inception_distance", float("nan")))

    except Exception as e:
        logger.warning(f"FID computation failed: {e}")
        return None
    finally:
        # Clean up temporary directories
        shutil.rmtree(real_dir, ignore_errors=True)
        shutil.rmtree(fake_dir, ignore_errors=True)


def compute_fid_per_class(
    real_images: torch.Tensor,
    fake_images: torch.Tensor,
    real_labels: torch.Tensor,
    fake_labels: torch.Tensor,
    num_classes: int,
    device: str = "cuda",
    min_samples: int = 50,
) -> Dict[str, float]:
    """
    Compute FID for each class separately.

    Args:
        real_images: Real images tensor [N, C, H, W]
        fake_images: Generated images tensor [M, C, H, W]
        real_labels: Class labels for real images [N]
        fake_labels: Class labels for fake images [M]
        num_classes: Total number of classes
        device: Device for computation
        min_samples: Minimum samples needed per class to compute FID

    Returns:
        Dictionary with 'fid_global' and 'fid_c0', 'fid_c1', etc.
        Classes with insufficient samples will have NaN values.
    """
    result: Dict[str, float] = {}

    # Global FID
    global_fid = compute_fid_from_tensors(real_images, fake_images, device)
    result["fid_global"] = global_fid if global_fid is not None else float("nan")

    # Per-class FID
    real_labels_cpu = real_labels.cpu()
    fake_labels_cpu = fake_labels.cpu()

    for k in range(num_classes):
        real_mask = real_labels_cpu == k
        fake_mask = fake_labels_cpu == k

        n_real = real_mask.sum().item()
        n_fake = fake_mask.sum().item()

        if n_real >= min_samples and n_fake >= min_samples:
            class_fid = compute_fid_from_tensors(
                real_images[real_mask],
                fake_images[fake_mask],
                device
            )
            result[f"fid_c{k}"] = class_fid if class_fid is not None else float("nan")
        else:
            result[f"fid_c{k}"] = float("nan")
            if n_real < min_samples or n_fake < min_samples:
                logger.debug(
                    f"Skipping FID for class {k}: "
                    f"n_real={n_real}, n_fake={n_fake}, min_samples={min_samples}"
                )

    return result


def _save_images_to_dir(images: torch.Tensor, out_dir: Path) -> None:
    """
    Save tensor images to a directory as PNG files.

    Args:
        images: Tensor [N, C, H, W] in [0, 1] range
        out_dir: Output directory path
    """
    from torchvision.utils import save_image
    out_dir.mkdir(parents=True, exist_ok=True)

    for i in range(images.size(0)):
        img = images[i]
        # Clamp to [0, 1] and save
        img = torch.clamp(img, 0, 1)
        save_image(img, out_dir / f"{i:06d}.png")


class FIDAccumulator:
    """
    Accumulates images for FID computation across batches.

    FID requires a large number of samples (~50k) for reliable estimates.
    This class accumulates generated and real images during evaluation
    and computes FID at the end.

    Usage:
        acc = FIDAccumulator(num_classes=9, max_samples=10000)
        for batch in dataloader:
            acc.add_real(real_images, labels)
            acc.add_fake(fake_images, labels)
        fid_metrics = acc.compute()
    """

    def __init__(
        self,
        num_classes: int,
        max_samples: int = 10000,
        device: str = "cpu",
    ):
        self.num_classes = num_classes
        self.max_samples = max_samples
        self.device = device
        self.reset()

    def reset(self) -> None:
        """Reset all accumulated images."""
        self._real_images: List[torch.Tensor] = []
        self._fake_images: List[torch.Tensor] = []
        self._real_labels: List[torch.Tensor] = []
        self._fake_labels: List[torch.Tensor] = []
        self._real_count = 0
        self._fake_count = 0

    def add_real(self, images: torch.Tensor, labels: torch.Tensor) -> None:
        """Add real images to accumulator."""
        if self._real_count >= self.max_samples:
            return
        n = min(images.size(0), self.max_samples - self._real_count)
        self._real_images.append(images[:n].cpu())
        self._real_labels.append(labels[:n].cpu())
        self._real_count += n

    def add_fake(self, images: torch.Tensor, labels: torch.Tensor) -> None:
        """Add generated/reconstructed images to accumulator."""
        if self._fake_count >= self.max_samples:
            return
        n = min(images.size(0), self.max_samples - self._fake_count)
        self._fake_images.append(images[:n].cpu())
        self._fake_labels.append(labels[:n].cpu())
        self._fake_count += n

    def compute(self, min_samples_per_class: int = 50) -> Dict[str, float]:
        """
        Compute global and per-class FID.

        Args:
            min_samples_per_class: Minimum samples needed per class

        Returns:
            Dictionary with 'fid_global' and 'fid_c0', 'fid_c1', etc.
        """
        if len(self._real_images) == 0 or len(self._fake_images) == 0:
            result = {"fid_global": float("nan")}
            result.update({f"fid_c{k}": float("nan") for k in range(self.num_classes)})
            return result

        real_images = torch.cat(self._real_images, dim=0)
        fake_images = torch.cat(self._fake_images, dim=0)
        real_labels = torch.cat(self._real_labels, dim=0)
        fake_labels = torch.cat(self._fake_labels, dim=0)

        return compute_fid_per_class(
            real_images=real_images,
            fake_images=fake_images,
            real_labels=real_labels,
            fake_labels=fake_labels,
            num_classes=self.num_classes,
            device=self.device,
            min_samples=min_samples_per_class,
        )

    @property
    def real_count(self) -> int:
        return self._real_count

    @property
    def fake_count(self) -> int:
        return self._fake_count
