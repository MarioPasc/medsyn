# medsyn/models/ccDDPM/engine/ddp_utils.py
# Purpose: Utility functions for PyTorch DistributedDataParallel (DDP) training.
# Notes:
# - Handles DDP initialization, rank checking, synchronization, and cleanup.
# - Ensures compatibility with both single-GPU and multi-GPU training.
from __future__ import annotations
import os
import torch
import torch.distributed as dist
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


def ddp_is_enabled(cfg) -> bool:
    """
    Check if distributed training is enabled based on config and available GPUs.

    Args:
        cfg: ProjectCfg with ccddpm.dist section

    Returns:
        True if DDP should be used, False otherwise
    """
    return cfg.ccddpm.dist.enabled and torch.cuda.device_count() > 1

def sync_metrics_dict(metrics_dict: Dict[str, float], device: torch.device, use_ddp: bool) -> Dict[str, float]:
    """
    Synchronize a dictionary of metrics across all DDP processes by averaging.

    Args:
        metrics_dict: Dictionary of metric name -> value
        device: Device tensors are on
        use_ddp: Whether DDP is enabled

    Returns:
        Dictionary with globally averaged metrics
    """
    if not use_ddp:
        return metrics_dict

    synced_metrics = {}
    for key, value in metrics_dict.items():
        if isinstance(value, (int, float)):
            # Convert to tensor and synchronize
            tensor = torch.tensor([float(value)], device=device, dtype=torch.float32)
            synced_tensor = all_reduce_mean(tensor)
            synced_metrics[key] = synced_tensor.item()
        else:
            # Non-numeric values (shouldn't happen for metrics, but just in case)
            synced_metrics[key] = value

    return synced_metrics

def ddp_init(cfg) -> Dict[str, Any]:
    """
    Initialize distributed training and set local device.

    This function initializes the process group for distributed training using
    the configuration specified in cfg.ccddpm.dist. It should be called early
    in the training script, before creating the model.

    Args:
        cfg: ProjectCfg with ccddpm.dist section

    Returns:
        Dictionary with:
            - rank: Global rank of this process (0 to world_size-1)
            - world_size: Total number of processes
            - local_rank: Local rank on this node (GPU ID)
            - device: torch.device for this process
    """
    # Get local_rank BEFORE initializing process group (needed for device_id)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # Set device for this process BEFORE init_process_group
    # This ensures proper GPU affinity from the start
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    if not dist.is_initialized():
        # CRITICAL: Pass device_id to init_process_group for proper GPU affinity
        # This prevents the warning: "No device id is provided via init_process_group"
        # and ensures optimal NCCL performance with proper process-to-GPU binding
        try:
            # PyTorch 2.0+ accepts device_id parameter for better GPU affinity
            dist.init_process_group(
                backend=cfg.ccddpm.dist.backend,
                init_method=cfg.ccddpm.dist.init_method,
                device_id=device  # ✅ FIX: Pass CUDA device for optimal affinity
            )
        except TypeError:
            # Fallback for older PyTorch versions that don't support device_id
            # The torch.cuda.set_device() call above still ensures correct GPU binding
            dist.init_process_group(
                backend=cfg.ccddpm.dist.backend,
                init_method=cfg.ccddpm.dist.init_method,
            )
        logger.info("Initialized DDP process group with backend=%s, device=%s",
                   cfg.ccddpm.dist.backend, device)

    # Get rank information from distributed backend
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    logger.info("DDP initialized: rank=%d/%d, local_rank=%d, device=%s",
                rank, world_size, local_rank, device)

    return {
        "rank": rank,
        "world_size": world_size,
        "local_rank": local_rank,
        "device": device,
    }


def is_main_process() -> bool:
    """
    Check if this is the main process (rank 0).

    Use this to control logging, checkpoint saving, and other operations
    that should only happen once across all processes.

    Returns:
        True if this is rank 0 or if distributed training is not initialized
    """
    return (not dist.is_available()) or (not dist.is_initialized()) or (dist.get_rank() == 0)


def get_rank() -> int:
    """
    Get the rank of the current process.

    Returns:
        Rank of current process, or 0 if not using distributed training
    """
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def get_world_size() -> int:
    """
    Get the total number of processes.

    Returns:
        Total number of processes, or 1 if not using distributed training
    """
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1


def barrier():
    """
    Synchronize all processes.

    This function blocks until all processes in the group have called it.
    Useful for ensuring all processes reach a certain point before continuing.

    Note: The device is set during ddp_init() and persists throughout the process,
    ensuring barrier() uses the correct GPU context.
    """
    if dist.is_available() and dist.is_initialized():
        # Ensure we're on the correct device (defensive programming)
        # This should already be set from ddp_init(), but we verify here
        if torch.cuda.is_available():
            current_device = torch.cuda.current_device()
            # barrier() will use the current device context set by torch.cuda.set_device()
            dist.barrier()
        else:
            dist.barrier()


def cleanup():
    """
    Clean up distributed training resources.

    This function should be called at the end of training to properly
    destroy the process group.
    """
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
        logger.info("DDP process group destroyed")


@torch.no_grad()
def all_reduce_mean(x: torch.Tensor) -> torch.Tensor:
    """
    Compute the mean of a tensor across all processes.

    This is typically used to average loss or metric values from all GPUs
    to get a global statistic.

    Args:
        x: Tensor to average (typically a scalar)

    Returns:
        Averaged tensor
    """
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
        x = x / dist.get_world_size()
    return x


def broadcast_bool(flag: bool) -> bool:
    """
    Broadcast a boolean value from rank 0 to all processes.

    This is useful for synchronizing early stopping decisions or other
    control flow that should be consistent across all processes.

    In legacy (non-DDP) mode, simply returns the input flag unchanged.

    Args:
        flag: Boolean value on rank 0 (ignored on other ranks)

    Returns:
        The broadcasted boolean value
    """
    if not (dist.is_available() and dist.is_initialized()):
        # Legacy: No DDP, just return the flag
        return flag

    # DDP: Broadcast from rank 0 to all processes
    device = torch.device("cuda", torch.cuda.current_device())
    t = torch.tensor([1 if flag else 0], device=device, dtype=torch.int)
    dist.broadcast(t, src=0)
    return bool(t.item())


def strip_module_prefix(state_dict: dict) -> dict:
    """
    Remove 'module.' prefix from state dict keys.

    When saving checkpoints from DDP models, the state dict keys are prefixed
    with 'module.' because the model is wrapped in DistributedDataParallel.
    This function removes that prefix to make the checkpoint compatible with
    single-GPU inference.

    Args:
        state_dict: Model state dictionary

    Returns:
        State dictionary with 'module.' prefix removed
    """
    return {
        k.replace("module.", "", 1) if k.startswith("module.") else k: v
        for k, v in state_dict.items()
    }


def get_state_dict_for_save(model: torch.nn.Module) -> dict:
    """
    Get state dict from model, handling DDP wrapper.

    This function automatically unwraps DDP models and removes the 'module.'
    prefix to ensure checkpoints are compatible with single-GPU inference.

    Args:
        model: Model (possibly wrapped in DDP)

    Returns:
        State dictionary without 'module.' prefix
    """
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()

    # Strip 'module.' prefix for robustness
    return strip_module_prefix(state_dict)


def setup_for_distributed(is_master: bool):
    """
    Disable printing on non-master processes.

    This can be used to reduce log clutter in distributed training by only
    allowing rank 0 to print.

    Args:
        is_master: True if this is the main process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print
