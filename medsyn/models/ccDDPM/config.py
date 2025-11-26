# medsyn/models/ccDDPM/config.py
# Purpose: Typed config loader for ccDDPM training/inference.
# Notes:
# - Reads the global YAML, expects `data.index_json` and a `ccddpm` section.
# - Provides defaults if keys are missing.
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any, Mapping, Sequence
import yaml
import os
import logging

logger = logging.getLogger(__name__)

@dataclass
class OptimizerCfg:
    """
    Unified optimizer configuration with type selection.

    Supported optimizer types:
    - adamw: AdamW (default) - Adam with decoupled weight decay
    - adam: Adam - standard Adam optimizer
    - sgd: SGD - stochastic gradient descent with optional momentum
    - rmsprop: RMSprop - adaptive learning rate method
    """
    type: str = "adamw"  # Optimizer type: adamw, adam, sgd, rmsprop
    lr: float = 2e-4     # Learning rate
    wd: float = 0.0      # Weight decay (L2 regularization)
    # Adam/AdamW specific
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    amsgrad: bool = False  # Whether to use AMSGrad variant
    # SGD specific
    momentum: float = 0.9
    nesterov: bool = False
    # RMSprop specific
    alpha: float = 0.99  # Smoothing constant
    centered: bool = False  # Whether to compute centered RMSprop


# Legacy alias for backward compatibility
OptimCfg = OptimizerCfg

@dataclass
class SchedCfg:
    num_train_timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02
    beta_schedule: str = "linear"  # ["linear","squaredcos_cap_v2",...]
    prediction_type: str = "epsilon"  # DDPM default


@dataclass
class EarlyStoppingCfg:
    """Configuration for early stopping with EMA smoothing."""
    patience: int = 5  # Number of epochs without improvement before stopping
    metric: str = "psnr_ssim_composite"  # Metric for early stopping: psnr_ssim_composite, psnr, ssim, loss
    psnr_weight: float = 1.0  # Weight for PSNR in composite score
    ssim_weight: float = 1.0  # Weight for SSIM in composite score
    ema_alpha: float = 0.7  # EMA smoothing factor (0 = no smoothing, higher = more smoothing)
    use_ema_for_stopping: bool = True  # Whether to use EMA score for stopping decisions
    min_delta: float = 0.001  # Minimum improvement to count as "better"


@dataclass
class LRSchedulerCfg:
    """Configuration for learning rate scheduling."""
    type: str = "constant"  # Scheduler type: onecycle, cosine, cosine_warmup, linear_warmup_cosine_annealing_lr, step, constant
    # OneCycle parameters
    max_lr: Optional[float] = None  # Peak LR (default: 2x base lr from optimizer)
    pct_start: float = 0.3  # Fraction of training spent ramping up LR
    div_factor: float = 25.0  # Initial LR = max_lr / div_factor
    final_div_factor: float = 1000.0  # Final LR = max_lr / (div_factor * final_div_factor)
    # Cosine parameters
    T_max: Optional[int] = None  # Number of iterations for cosine (None = total_epochs)
    eta_min: float = 1e-7  # Minimum learning rate
    warmup_epochs: int = 5  # Warmup epochs (for cosine_warmup)
    # LinearWarmupCosineAnnealingLR parameters
    warmup_start_lr: Optional[float] = None  # Absolute LR at first warmup step (None = use warmup_start_factor)
    warmup_start_factor: float = 0.01  # Relative factor: start LR = base_lr * warmup_start_factor (used when warmup_start_lr is None)
    # Step parameters
    step_size: int = 30  # Decay LR every step_size epochs
    gamma: float = 0.1  # Multiplicative factor of LR decay


@dataclass
class UNetCfg:
    """
    Configuration for UNet2DModel architecture.

    These parameters directly map to diffusers.UNet2DModel constructor arguments.
    See: https://huggingface.co/docs/diffusers/api/models/unet2d

    Note: `in_channels` and `out_channels` are computed from TrainCfg.in_channels
    and TrainCfg.class_embed_dim, so they are not exposed here.
    """
    # Core architecture
    model_channels: int = 128  # Base channel count (used with channel_mult)
    channel_mult: tuple[int, ...] = (1, 2, 2, 4)  # Channel multipliers per level
    layers_per_block: int = 2  # Number of ResNet blocks per level (maps to num_res_blocks)

    # Block types - must match length of channel_mult
    down_block_types: tuple[str, ...] = ("DownBlock2D", "DownBlock2D", "DownBlock2D", "AttnDownBlock2D")
    up_block_types: tuple[str, ...] = ("AttnUpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D")

    # Attention configuration
    add_attention: bool = True
    attention_head_dim: Optional[int] = 8  # None = use default

    # Normalization
    norm_num_groups: int = 32  # Groups for GroupNorm
    attn_norm_num_groups: Optional[int] = None  # Groups for attention norm (None = use norm_num_groups)
    norm_eps: float = 1e-5  # Epsilon for normalization

    # Dropout and regularization
    dropout: float = 0.0

    # Time embedding
    time_embedding_type: str = "positional"  # "positional" or "fourier"
    freq_shift: int = 0
    flip_sin_to_cos: bool = True
    resnet_time_scale_shift: str = "default"  # "default", "scale_shift", or "ada_group"

    # Sampling configuration
    center_input_sample: bool = False
    mid_block_scale_factor: float = 1.0

    # Downsampling/Upsampling
    downsample_padding: int = 1
    downsample_type: str = "conv"  # "conv" or "resnet"
    upsample_type: str = "conv"  # "conv" or "resnet"

    # Activation
    act_fn: str = "silu"  # "silu", "mish", "gelu", etc.

    # Class embedding (handled externally via ClassEmbedder, but exposed for completeness)
    class_embed_type: Optional[str] = None  # None, "timestep", "identity"
    num_class_embeds: Optional[int] = None
    num_train_timesteps: Optional[int] = None  # For timestep class embedding

    def validate(self) -> None:
        """Validate UNet configuration for consistency."""
        errors = []

        # Check channel_mult and block_types length consistency
        n_levels = len(self.channel_mult)
        if len(self.down_block_types) != n_levels:
            errors.append(
                f"len(down_block_types)={len(self.down_block_types)} must equal "
                f"len(channel_mult)={n_levels}"
            )
        if len(self.up_block_types) != n_levels:
            errors.append(
                f"len(up_block_types)={len(self.up_block_types)} must equal "
                f"len(channel_mult)={n_levels}"
            )

        # Validate block type names
        valid_down_blocks = {"DownBlock2D", "AttnDownBlock2D", "CrossAttnDownBlock2D", "DownEncoderBlock2D"}
        valid_up_blocks = {"UpBlock2D", "AttnUpBlock2D", "CrossAttnUpBlock2D", "UpDecoderBlock2D"}

        for i, block_type in enumerate(self.down_block_types):
            if block_type not in valid_down_blocks:
                errors.append(
                    f"down_block_types[{i}]='{block_type}' is not valid. "
                    f"Must be one of: {valid_down_blocks}"
                )

        for i, block_type in enumerate(self.up_block_types):
            if block_type not in valid_up_blocks:
                errors.append(
                    f"up_block_types[{i}]='{block_type}' is not valid. "
                    f"Must be one of: {valid_up_blocks}"
                )

        # Validate norm_num_groups divides model_channels
        if self.model_channels % self.norm_num_groups != 0:
            errors.append(
                f"model_channels={self.model_channels} must be divisible by "
                f"norm_num_groups={self.norm_num_groups}"
            )

        # Validate all block_out_channels are divisible by norm_num_groups
        for i, mult in enumerate(self.channel_mult):
            block_channels = self.model_channels * mult
            if block_channels % self.norm_num_groups != 0:
                errors.append(
                    f"block_out_channels[{i}]={block_channels} (model_channels * channel_mult[{i}]) "
                    f"must be divisible by norm_num_groups={self.norm_num_groups}"
                )

        # Validate time_embedding_type
        valid_time_embed = {"positional", "fourier", "learned"}
        if self.time_embedding_type not in valid_time_embed:
            errors.append(
                f"time_embedding_type='{self.time_embedding_type}' is not valid. "
                f"Must be one of: {valid_time_embed}"
            )

        # Validate resnet_time_scale_shift
        valid_scale_shift = {"default", "scale_shift", "ada_group"}
        if self.resnet_time_scale_shift not in valid_scale_shift:
            errors.append(
                f"resnet_time_scale_shift='{self.resnet_time_scale_shift}' is not valid. "
                f"Must be one of: {valid_scale_shift}"
            )

        # Validate act_fn
        valid_act_fns = {"silu", "mish", "gelu", "relu", "swish"}
        if self.act_fn not in valid_act_fns:
            errors.append(
                f"act_fn='{self.act_fn}' is not valid. Must be one of: {valid_act_fns}"
            )

        # Validate downsample/upsample types
        valid_sample_types = {"conv", "resnet"}
        if self.downsample_type not in valid_sample_types:
            errors.append(
                f"downsample_type='{self.downsample_type}' is not valid. "
                f"Must be one of: {valid_sample_types}"
            )
        if self.upsample_type not in valid_sample_types:
            errors.append(
                f"upsample_type='{self.upsample_type}' is not valid. "
                f"Must be one of: {valid_sample_types}"
            )

        if errors:
            raise ValueError("UNet configuration validation failed:\n  - " + "\n  - ".join(errors))

    def get_block_out_channels(self) -> tuple[int, ...]:
        """Compute block_out_channels from model_channels and channel_mult."""
        return tuple(self.model_channels * m for m in self.channel_mult)

@dataclass
class CCDDPMInit:
    """
    Configuration for CCDDPM model initialization.

    This dataclass holds all parameters needed to construct the UNet2DModel backbone.
    Parameters directly map to diffusers.UNet2DModel constructor arguments.

    Note: `in_channels` for UNet is computed as `in_channels + class_embed_dim` since
    class embeddings are concatenated channel-wise to the input.
    """
    # Input/Output configuration
    in_channels: int = 3  # Number of input image channels (e.g., 3 for RGB)
    class_embed_dim: int = 16  # Dimension of class embedding (concatenated to input)
    num_classes: int = 9  # Number of classes for conditional generation

    # Core UNet architecture
    model_channels: int = 128  # Base channel count (multiplied by channel_mult)
    channel_mult: tuple[int, ...] = (1, 2, 2, 4)  # Channel multipliers per level
    layers_per_block: int = 2  # Number of ResNet blocks per level

    # Block types - must match length of channel_mult
    down_block_types: tuple[str, ...] = ("DownBlock2D", "DownBlock2D", "DownBlock2D", "AttnDownBlock2D")
    up_block_types: tuple[str, ...] = ("AttnUpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D")

    # Attention configuration
    add_attention: bool = True
    attention_head_dim: Optional[int] = 8  # None = use default

    # Normalization
    norm_num_groups: int = 32  # Groups for GroupNorm
    attn_norm_num_groups: Optional[int] = None  # Groups for attention norm
    norm_eps: float = 1e-5  # Epsilon for normalization

    # Dropout
    dropout: float = 0.0

    # Time embedding
    time_embedding_type: str = "positional"  # "positional" or "fourier"
    freq_shift: int = 0
    flip_sin_to_cos: bool = True
    resnet_time_scale_shift: str = "default"  # "default", "scale_shift", "ada_group"

    # Sampling configuration
    center_input_sample: bool = False
    mid_block_scale_factor: float = 1.0

    # Downsampling/Upsampling
    downsample_padding: int = 1
    downsample_type: str = "conv"  # "conv" or "resnet"
    upsample_type: str = "conv"  # "conv" or "resnet"

    # Activation
    act_fn: str = "silu"  # "silu", "mish", "gelu", etc.

    # Class embedding (handled externally, but available for UNet's internal use)
    class_embed_type: Optional[str] = None
    num_class_embeds: Optional[int] = None
    num_train_timesteps: Optional[int] = None

    def get_block_out_channels(self) -> tuple[int, ...]:
        """Compute block_out_channels from model_channels and channel_mult."""
        return tuple(self.model_channels * m for m in self.channel_mult)

@dataclass
class TrainCfg:
    image_size: int = 128
    in_channels: int = 3
    class_embed_dim: int = 16
    num_classes: int = 9
    batch_size: int = 64
    epochs: int = 50
    num_workers: int = 8
    seed: int = 1337
    mixed_precision: bool = True
    grad_clip_norm: Optional[float] = 1.0
    ema_use: bool = True
    ema_decay: float = 0.999
    guidance_p_uncond: float = 0.1  # classifier-free label drop prob
    use_min_snr: bool = False  # Min-SNR loss weighting (Hang et al. 2023)
    min_snr_gamma: float = 5.0  # SNR clamp value for Min-SNR weighting (typical 2-5)
    log_every: int = 100
    ckpt_every_epochs: int = 1
    snapshot_class_embedding_every: int = 0  # Save class embedding snapshots every X epochs (0 = disabled)
    patience: int = 15  # Early stopping: number of epochs without improvement (legacy, use early_stopping.patience)
    use_tqdm: bool = True  # Enable/disable tqdm progress bars (useful for HPC logs)
    skip_nonfinite_grads: bool = False  # If True, skip batches with non-finite gradients; if False, raise error
    per_class_loss_weighting: bool = False  # Enable per-class loss weighting based on class frequencies
    per_class_weight_temperature: float = 1.0  # Temperature for class weight distribution (>1: more extreme, <1: more uniform)
    output_dir: Path = Path("./outputs/ccddpm")
    # New: Early stopping configuration (replaces simple patience)
    early_stopping: Optional[EarlyStoppingCfg] = None
    # New: Learning rate scheduler configuration
    lr_scheduler: Optional[LRSchedulerCfg] = None

@dataclass
class InferenceCfg:
    guidance_scale: float = 0.0  # 0 = unconditional; >0 for cfg
    num_inference_steps: int = 1000
    save_grid: bool = True
    out_dir: Path = Path("./samples/ccddpm")

@dataclass
class DataloaderCfg:
    type: str = "json"  # "json" or "npz"
    npz_path: Optional[Path] = None  # Path to NPZ file (required if type="npz")

@dataclass
class DataCfg:
    index_json: Path
    processed_root: Optional[Path] = None  # optional; paths already absolute in JSON
    split: str = "train"
    normalize: bool = True  # scale to [-1,1]

@dataclass
class DistCfg:
    """Configuration for distributed training (DDP)."""
    enabled: bool = False  # Enable DDP multi-GPU training
    backend: str = "nccl"  # Backend for distributed training (nccl for GPU, gloo for CPU)
    init_method: str = "env://"  # torchrun uses env://
    num_gpus: int | str = "auto"  # "auto" uses WORLD_SIZE from torchrun
    find_unused_parameters: bool = False  # DDP flag for dynamic graphs
    broadcast_buffers: bool = True  # Sync buffers (e.g., BatchNorm stats)
    grad_accum_steps: int = 1  # Gradient accumulation steps (optional)
    seed_offset: int = 0  # Seed offset per rank for decorrelation

@dataclass
class CCDDPmCfg:
    train: TrainCfg = field(default_factory=TrainCfg)
    optimizer: OptimizerCfg = field(default_factory=OptimizerCfg)  # Renamed from optim
    sched: SchedCfg = field(default_factory=SchedCfg)
    infer: InferenceCfg = field(default_factory=InferenceCfg)
    dataloader: DataloaderCfg = field(default_factory=DataloaderCfg)
    unet: UNetCfg = field(default_factory=UNetCfg)  # UNet architecture config
    data: DataCfg | None = None  # set after reading YAML
    augmentation: Any = None  # AugmentationConfig, set after reading YAML
    dist: DistCfg = field(default_factory=DistCfg)  # Distributed training config

    @property
    def optim(self) -> OptimizerCfg:
        """Legacy alias for backward compatibility."""
        return self.optimizer

@dataclass
class ProjectCfg:
    data_index_json: Path
    ccddpm: CCDDPmCfg
    fid: Optional[Dict[str, Any]] = None  # FID computation config (optional)

def _as_path(p: Optional[str]) -> Optional[Path]:
    return None if p is None else Path(p).expanduser().resolve()

def _expand_env(obj: Any) -> Any:
    """
    Recursively expand ${ENV} and $ENV in strings within mappings/lists.
    Ensures backward compatibility with absolute paths.
    """
    if isinstance(obj, str):
        return os.path.expandvars(obj)
    if isinstance(obj, Mapping):
        return {k: _expand_env(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        expanded = [_expand_env(v) for v in obj]
        return type(obj)(expanded) if isinstance(obj, tuple) else expanded
    return obj

def load_cfg(yaml_path: str | Path, split: str = "train") -> ProjectCfg:
    """
    Load YAML config and build a ProjectCfg with ccDDPM fields.
    Supports environment variable expansion (${VAR} or $VAR) in YAML values.
    Expects:
      data.save_png.index_json: path to JSON index built by medsyn/cli/data.py (for JSON dataloader)
      data.postprocess_npz.npz_path: path to custom NPZ file (for NPZ dataloader)
      ccddpm: optional section overriding defaults.
    """
    with open(yaml_path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    
    # Expand environment variables in all string values
    raw = _expand_env(raw)

    # data section
    data_dict = raw.get("data", {})
    
    # Get index_json from save_png section (for JSON dataloader compatibility)
    save_png = data_dict.get("save_png", {})
    idx = save_png.get("index_json")
    if idx:
        index_json_path = _as_path(idx)
    else:
        # Fallback for backward compatibility
        idx = data_dict.get("index_json")
        if idx:
            index_json_path = _as_path(idx)
        else:
            logger.warning("No index_json found in YAML. JSON dataloader may not work.")
            index_json_path = None
    
    data_cfg = DataCfg(
        index_json=index_json_path if index_json_path else Path("./dummy.json"),
        processed_root=_as_path(save_png.get("processed_dir")) if save_png.get("processed_dir") else None,
        split=split
    )

    # ccddpm section with deep defaults
    cc = raw.get("ccddpm", {}) or {}

    # Parse TrainCfg with nested early_stopping and lr_scheduler configs
    train_dict = cc.get("train", {})
    train_base = {**TrainCfg().__dict__, **{k: train_dict.get(k, v) for k, v in TrainCfg().__dict__.items() if k not in ('early_stopping', 'lr_scheduler')}}

    # Parse early_stopping nested config
    early_stopping_dict = train_dict.get("early_stopping")
    if early_stopping_dict and isinstance(early_stopping_dict, dict):
        early_stopping_cfg = EarlyStoppingCfg(**{**EarlyStoppingCfg().__dict__, **early_stopping_dict})
    else:
        early_stopping_cfg = None
    train_base["early_stopping"] = early_stopping_cfg

    # Parse lr_scheduler nested config
    lr_scheduler_dict = train_dict.get("lr_scheduler")
    if lr_scheduler_dict and isinstance(lr_scheduler_dict, dict):
        lr_scheduler_cfg = LRSchedulerCfg(**{**LRSchedulerCfg().__dict__, **lr_scheduler_dict})
    else:
        lr_scheduler_cfg = None
    train_base["lr_scheduler"] = lr_scheduler_cfg

    train = TrainCfg(**train_base)

    # Parse optimizer config (new unified section, with fallback to legacy "optim")
    optimizer_dict = cc.get("optimizer", cc.get("optim", {}))
    if optimizer_dict:
        # Convert betas list to tuple if needed
        if 'betas' in optimizer_dict and isinstance(optimizer_dict['betas'], list):
            optimizer_dict['betas'] = tuple(optimizer_dict['betas'])
    optimizer_cfg = OptimizerCfg(**{**OptimizerCfg().__dict__, **optimizer_dict})
    sched = SchedCfg(**{**SchedCfg().__dict__, **cc.get("sched", {})})
    infer = InferenceCfg(**{**InferenceCfg().__dict__, **cc.get("infer", {})})
    dist = DistCfg(**{**DistCfg().__dict__, **cc.get("dist", {})})
    
    # dataloader configuration - read NPZ path from data.postprocess_npz section
    dl_cfg_dict = cc.get("dataloader", {})
    dataloader_type = dl_cfg_dict.get("type", "json")
    
    # Get NPZ path from data.postprocess_npz if type is npz
    npz_path = None
    if dataloader_type.lower() == "npz":
        postprocess_npz = data_dict.get("postprocess_npz", {})
        npz_path_str = postprocess_npz.get("npz_path")
        if npz_path_str:
            npz_path = _as_path(npz_path_str)
        else:
            raise ValueError("NPZ dataloader selected but data.postprocess_npz.npz_path not found in YAML")
    
    dataloader_cfg = DataloaderCfg(
        type=dataloader_type,
        npz_path=npz_path
    )

    # Parse augmentation configuration
    aug_dict = cc.get('augmentation', {})
    augmentation_cfg = None
    if aug_dict:
        try:
            from medsyn.models.ccDDPM.augmentation import AugmentationConfig
            augmentation_cfg = AugmentationConfig.from_dict(aug_dict)
            logger.info("Augmentation enabled: %s, probability: %.2f, transforms: %d",
                       augmentation_cfg.enabled, augmentation_cfg.probability, len(augmentation_cfg.transforms))
        except ImportError:
            logger.warning("Augmentation config found but augmentation module not available. Skipping augmentation.")
            augmentation_cfg = None

    # Parse UNet architecture configuration (denoising_unet section)
    unet_dict = cc.get('denoising_unet', {})
    if unet_dict:
        # Convert lists to tuples for tuple fields
        if 'channel_mult' in unet_dict and isinstance(unet_dict['channel_mult'], list):
            unet_dict['channel_mult'] = tuple(unet_dict['channel_mult'])
        if 'down_block_types' in unet_dict and isinstance(unet_dict['down_block_types'], list):
            unet_dict['down_block_types'] = tuple(unet_dict['down_block_types'])
        if 'up_block_types' in unet_dict and isinstance(unet_dict['up_block_types'], list):
            unet_dict['up_block_types'] = tuple(unet_dict['up_block_types'])

        # Build UNetCfg with defaults merged with YAML values
        unet_defaults = UNetCfg().__dict__
        unet_merged = {**unet_defaults, **unet_dict}
        unet_cfg = UNetCfg(**unet_merged)

        # Validate UNet configuration
        try:
            unet_cfg.validate()
            logger.info("UNet config loaded and validated: model_channels=%d, levels=%d, layers_per_block=%d",
                       unet_cfg.model_channels, len(unet_cfg.channel_mult), unet_cfg.layers_per_block)
            logger.info("  block_out_channels=%s", unet_cfg.get_block_out_channels())
            logger.info("  down_block_types=%s", unet_cfg.down_block_types)
            logger.info("  up_block_types=%s", unet_cfg.up_block_types)
        except ValueError as e:
            logger.error("UNet configuration validation failed: %s", e)
            raise
    else:
        # Use defaults with a warning
        unet_cfg = UNetCfg()
        logger.warning("No 'denoising_unet' section found in YAML. Using default UNet architecture.")
        logger.warning("  Consider adding 'denoising_unet' section for explicit control over UNet architecture.")

    cc_cfg = CCDDPmCfg(
        train=train, optimizer=optimizer_cfg, sched=sched, infer=infer,
        dataloader=dataloader_cfg, unet=unet_cfg, data=data_cfg,
        augmentation=augmentation_cfg, dist=dist
    )

    # Load FID configuration (optional, for FID computation during validation)
    fid_config = raw.get("fid")
    if fid_config:
        # Expand environment variables in weights_path if present
        if "weights_path" in fid_config and fid_config["weights_path"]:
            fid_config["weights_path"] = str(_as_path(fid_config["weights_path"]))
        logger.info("FID config loaded: enabled=%s, weights_path=%s",
                   fid_config.get("enabled", False),
                   fid_config.get("weights_path", "not specified"))

    proj = ProjectCfg(
        data_index_json=index_json_path if index_json_path else Path("./dummy.json"),
        ccddpm=cc_cfg,
        fid=fid_config
    )
    logger.info("Loaded ccDDPM config. image_size=%d, classes=%d, timesteps=%d, dataloader=%s",
                cc_cfg.train.image_size, cc_cfg.train.num_classes, cc_cfg.sched.num_train_timesteps, dataloader_type)
    if npz_path:
        logger.info("NPZ dataloader path: %s", npz_path)
    return proj
