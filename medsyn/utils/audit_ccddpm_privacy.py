# audit_ccddpm_privacy.py
# -*- coding: utf-8 -*-
"""
Audit de privacidad para ccDDPM (memorization + membership inference) en PathMNIST.

Dependencias mínimas:
  - torch, torchvision, diffusers, numpy, scipy, scikit-learn, pillow, tqdm
  - lpips (opcional, recomendado)
  - pyyaml
Entradas:
  - YAML de configuración del proyecto (con ruta NPZ y ccddpm.*)
  - Checkpoint .pt con state_dict del modelo
Salidas:
  - NPZ con métricas crudas y agregadas
  - CSV con pares sospechosos y puntajes
  - JSON con resumen de auditoría
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List
import os, json, math, random, logging, argparse, time, yaml
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from torchvision.utils import save_image
from torchmetrics.functional import structural_similarity_index_measure as ssim_torch

try:
    import lpips  # type: ignore
    _HAS_LPIPS = True
except Exception:
    _HAS_LPIPS = False

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
# Importar las implementaciones del proyecto (asumimos paquete instalable)
from medsyn.models.ccDDPM.model import CCDDPM, CCDDPMInit  # noqa: E402
from medsyn.models.ccDDPM.config import load_cfg          # noqa: E402

# ---------------------------------------------------------------------
# Configuración y utilidades
# ---------------------------------------------------------------------

logger = logging.getLogger("ccddpm.audit")
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")

@dataclass
class GenCfg:
    use_existing_synth: bool = True      # use existing synthetic images from dataset
    existing_synth_split: str = "train"  # which split to load synthetic images from
    per_class: int = 64                  # muestras a generar por clase
    guidance_scale: float = 2.0          # CFG para inferencia
    num_inference_steps: int = 250       # pasos de muestreo (<= timesteps entrenados)
    seed: int = 1337
    save_grid: bool = True

@dataclass
class NNACfg:
    k: int = 5                           # vecinos por buscar
    lpips_threshold: float = 0.12        # umbral LPIPS para "casi-copia"
    ssim_threshold: float = 0.90         # umbral SSIM para "casi-copia"
    use_clip: bool = False               # usar CLIP si está disponible
    batch_size: int = 256

@dataclass
class MIACfg:
    n_timesteps_per_sample: int = 32     # promedios de pérdida por muestra
    auc_points: int = 200                # discretización para ROC

@dataclass
class IOPaths:
    yaml: Path
    checkpoint: Path
    outdir: Path

# ---------------------------------------------------------------------
# Configuration loading from YAML
# ---------------------------------------------------------------------

def load_audit_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load audit configuration from YAML file.
    Returns a dictionary with all configuration parameters.
    """
    if config_path is None or not config_path.exists():
        logger.warning("No config file provided or file not found. Using default values.")
        return {}

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    logger.info(f"Loaded configuration from {config_path}")
    return config or {}

def merge_config_with_args(config: Dict[str, Any], args: argparse.Namespace) -> Tuple[IOPaths, GenCfg, NNACfg, MIACfg]:
    """
    Merge YAML config with command-line arguments (args override config).
    """
    # IO Paths (command-line args take precedence)
    io_cfg = config.get('io', {})
    io = IOPaths(
        yaml=args.yaml if hasattr(args, 'yaml') and args.yaml else Path(io_cfg.get('yaml', '')),
        checkpoint=args.checkpoint if hasattr(args, 'checkpoint') and args.checkpoint else Path(io_cfg.get('checkpoint', '')),
        outdir=args.outdir if hasattr(args, 'outdir') and args.outdir else Path(io_cfg.get('outdir', 'outputs'))
    )

    # Generation config
    gen_cfg = config.get('generation', {})
    gen = GenCfg(
        use_existing_synth=gen_cfg.get('use_existing_synth', True),
        existing_synth_split=gen_cfg.get('existing_synth_split', 'train'),
        per_class=args.per_class if hasattr(args, 'per_class') and args.per_class != 64 else gen_cfg.get('per_class', 64),
        guidance_scale=args.guidance_scale if hasattr(args, 'guidance_scale') and args.guidance_scale != 2.0 else gen_cfg.get('guidance_scale', 2.0),
        num_inference_steps=args.steps if hasattr(args, 'steps') and args.steps != 250 else gen_cfg.get('num_inference_steps', 250),
        seed=gen_cfg.get('seed', 1337),
        save_grid=gen_cfg.get('save_grid', True)
    )

    # Nearest neighbor config
    nn_cfg = config.get('nearest_neighbor', {})
    nna = NNACfg(
        k=args.knn_k if hasattr(args, 'knn_k') and args.knn_k != 5 else nn_cfg.get('k', 5),
        lpips_threshold=args.lpips_th if hasattr(args, 'lpips_th') and args.lpips_th != 0.12 else nn_cfg.get('lpips_threshold', 0.12),
        ssim_threshold=args.ssim_th if hasattr(args, 'ssim_th') and args.ssim_th != 0.90 else nn_cfg.get('ssim_threshold', 0.90),
        use_clip=args.use_clip if hasattr(args, 'use_clip') else nn_cfg.get('use_clip', False),
        batch_size=nn_cfg.get('batch_size', 256)
    )

    # Membership inference config
    mia_cfg = config.get('membership_inference', {})
    mia = MIACfg(
        n_timesteps_per_sample=args.mia_n_t if hasattr(args, 'mia_n_t') and args.mia_n_t != 32 else mia_cfg.get('n_timesteps_per_sample', 32),
        auc_points=mia_cfg.get('auc_points', 200)
    )

    return io, gen, nna, mia

def setup_logging(config: Dict[str, Any]) -> None:
    """
    Setup logging configuration from config file.
    """
    log_cfg = config.get('logging', {})
    level_str = log_cfg.get('level', 'INFO')
    level = getattr(logging, level_str.upper(), logging.INFO)

    # Configure root logger
    log_format = "[%(asctime)s] [%(levelname)s] %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    handlers = [logging.StreamHandler()]

    # Add file handler if specified
    log_file = log_cfg.get('log_file')
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path))
        logger.info(f"Logging to file: {log_path}")

    logging.basicConfig(
        level=level,
        format=log_format,
        datefmt=date_format,
        handlers=handlers,
        force=True
    )

# ---------------------------------------------------------------------
# Carga de datos NPZ
# ---------------------------------------------------------------------

def load_npz_splits(npz_path: Path) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Carga splits train/val/test desde un NPZ con claves estandarizadas.
    Devuelve diccionario {split: {'images':..., 'labels':..., 'is_synth':...}}.
    Only loads splits that exist in the NPZ file.
    """
    data = np.load(npz_path)
    out: Dict[str, Dict[str, np.ndarray]] = {}
    for split in ("train", "val", "test"):
        images_key = f"{split}_images"
        if images_key in data:
            out[split] = {
                "images": data[images_key],
                "labels": data[f"{split}_labels"],
                "is_synth": data.get(f"{split}_is_synth", np.zeros(len(data[images_key]), dtype=bool)),
            }
        else:
            logger.warning(f"Split '{split}' not found in NPZ file, skipping...")

    if len(out) == 0:
        raise ValueError(f"No valid splits found in NPZ file: {npz_path}")

    return out

# ---------------------------------------------------------------------
# Reconstrucción del modelo y scheduler
# ---------------------------------------------------------------------

def build_model_and_scheduler(project_yaml: Path, device: torch.device) -> Tuple[CCDDPM, DDPMScheduler, Dict[str, Any]]:
    """
    Lee el YAML del proyecto y construye el CCDDPM y su DDPMScheduler coherente con entrenamiento.
    Retorna (modelo_sin_pesos, scheduler, cfg_dict_serializable)
    """
    cfg = load_cfg(project_yaml, split="train")  # usa los defaults y overrides del YAML
    tcfg, scfg = cfg.ccddpm.train, cfg.ccddpm.sched

    mcfg = CCDDPMInit(in_channels=tcfg.in_channels,
                      class_embed_dim=tcfg.class_embed_dim,
                      num_classes=tcfg.num_classes)
    model = CCDDPM(mcfg).to(device)
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=scfg.num_train_timesteps,
        beta_start=scfg.beta_start,
        beta_end=scfg.beta_end,
        beta_schedule=scfg.beta_schedule,
        prediction_type=scfg.prediction_type,
        clip_sample=True,
        clip_sample_range=1.0,
        thresholding=False,
    )
    meta = {
        "image_size": tcfg.image_size,
        "num_classes": tcfg.num_classes,
        "num_train_timesteps": scfg.num_train_timesteps,
        "beta_schedule": scfg.beta_schedule,
        "prediction_type": scfg.prediction_type,
    }
    return model, noise_scheduler, meta

def load_checkpoint_(model: nn.Module, ckpt_path: Path) -> Dict[str, Any]:
    """
    Carga un checkpoint en el modelo. Devuelve metadatos si existen.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"], strict=False)
    elif "model" in ckpt:
        model.load_state_dict(ckpt["model"], strict=False)
    else:
        # Asumir que es un state_dict plano
        model.load_state_dict(ckpt, strict=False)
    return {k: v for k, v in ckpt.items() if k not in ("state_dict", "model")}

# ---------------------------------------------------------------------
# Muestreo DDPM con CFG
# ---------------------------------------------------------------------

@torch.no_grad()
def ddpm_sample_ccddpm(model: CCDDPM,
                       scheduler: DDPMScheduler,
                       num_classes: int,
                       per_class: int,
                       guidance_scale: float,
                       num_inference_steps: int,
                       image_size: int,
                       device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Genera muestras condicionadas por clase con CFG combinando predicciones cond/uncond.
    Devuelve (imgs[-1,1], labels).
    """
    model.eval()
    scheduler.set_timesteps(num_inference_steps=num_inference_steps, device=device)

    B = per_class * num_classes
    labels = torch.arange(num_classes, device=device).repeat_interleave(per_class)
    # ruido inicial
    x = torch.randn(B, 3, image_size, image_size, device=device)

    for t in tqdm(scheduler.timesteps, desc="DDPM sampling", leave=False):
        # Predicción condicional
        eps_c = model(x, t.repeat(B), labels)
        if guidance_scale > 0:
            # Predicción incondicional usando label = -1 (embedding cero)
            eps_u = model(x, t.repeat(B), torch.full_like(labels, -1))
            eps = eps_u + guidance_scale * (eps_c - eps_u)
        else:
            eps = eps_c
        x = scheduler.step(model_output=eps, timestep=t, sample=x).prev_sample  # x_{t-1}
    # x está en el dominio de salida del modelo; asumimos training con normalización a [-1,1]
    x = x.clamp(-1, 1)
    return x, labels

# ---------------------------------------------------------------------
# Embeddings y distancias
# ---------------------------------------------------------------------

class EmbeddingModel(nn.Module):
    """
    Adaptador simple para extraer embeddings globales de imágenes 64x64.
    """
    def __init__(self, use_clip: bool = False):
        super().__init__()
        self.use_clip = use_clip
        if use_clip:
            try:
                import open_clip  # type: ignore
                self.model, _, self.preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
                self.model.eval()
                self.dim = self.model.visual.output_dim
                self._clip = True
            except Exception:
                self._clip = False
        if not use_clip or not getattr(self, "_clip", False):
            m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
            # quitar la última fc y usar avgpool -> 2048-D
            modules = list(m.children())[:-1]
            self.backbone = nn.Sequential(*modules).eval()
            self.dim = 2048
            self._clip = False
        # normalización
        self.tf = transforms.Compose([
            transforms.Resize((64, 64), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ])

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [-1,1] en CHW; convertimos a [0,1] y normalizamos.
        """
        x = (x + 1.0) * 0.5
        x = self.tf(x)
        if self._clip:
            # open_clip espera [0,1] y su propio preprocess; asumimos tf compatible
            feats = self.model.encode_image(x)
        else:
            feats = self.backbone(x).flatten(1)
        feats = F.normalize(feats, dim=1)
        return feats

def batched_embed(model: EmbeddingModel, imgs: torch.Tensor, bsz: int, device: torch.device) -> torch.Tensor:
    """
    Extrae embeddings en lotes.
    """
    outs = []
    for i in range(0, imgs.size(0), bsz):
        batch = imgs[i:i+bsz].to(device)
        outs.append(model(batch).cpu())
    return torch.cat(outs, 0)

def compute_pair_metrics(x_synth: torch.Tensor, x_ref: torch.Tensor, device: torch.device) -> Tuple[float, float]:
    """
    Devuelve (lpips, ssim) para un par de imágenes en [-1,1].
    """
    if _HAS_LPIPS:
        loss_fn = lpips.LPIPS(net='vgg').to(device)
        a = x_synth.unsqueeze(0).to(device)
        b = x_ref.unsqueeze(0).to(device)
        lp = float(loss_fn(a, b).item())
    else:
        # aproximación con distancia L2 normalizada sobre pixeles
        lp = float(F.mse_loss(x_synth, x_ref).item()) ** 0.5
    # SSIM de torchvision espera [0,1]
    ssim = float(ssim_torch(((x_synth+1)/2).unsqueeze(0), ((x_ref+1)/2).unsqueeze(0)))
    return lp, ssim

# ---------------------------------------------------------------------
# Membership inference basado en pérdida de denoising
# ---------------------------------------------------------------------

@torch.no_grad()
def per_sample_denoising_loss(model: CCDDPM,
                              scheduler: DDPMScheduler,
                              x0: torch.Tensor,
                              y: torch.Tensor,
                              n_t: int,
                              device: torch.device) -> torch.Tensor:
    """
    Calcula la pérdida MSE esperada de denoising promedio en n_t muestras de t por muestra.
    """
    model.eval()
    B = x0.size(0)
    losses = []
    # asumimos normalización a [-1,1]
    for _ in range(n_t):
        t = torch.randint(0, scheduler.num_train_timesteps, (B,), device=device, dtype=torch.long)
        eps = torch.randn_like(x0)
        x_t = scheduler.add_noise(x0, eps, t)
        eps_hat = model(x_t, t, y)  # predice ruido
        losses.append(F.mse_loss(eps_hat, eps, reduction="none").flatten(1).mean(dim=1))
    return torch.stack(losses, 0).mean(0)  # [B]

def roc_auc(scores_in: np.ndarray, scores_out: np.ndarray, n_points: int = 200) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Curva ROC manual usando umbrales sobre el score (menor = más miembro).
    """
    y_true = np.concatenate([np.ones_like(scores_in), np.zeros_like(scores_out)])
    scores = np.concatenate([scores_in, scores_out])
    thresholds = np.linspace(scores.min(), scores.max(), n_points)
    tpr, fpr = [], []
    for tau in thresholds:
        pred = scores < tau
        tp = np.sum((pred == 1) & (y_true == 1))
        fp = np.sum((pred == 1) & (y_true == 0))
        fn = np.sum((pred == 0) & (y_true == 1))
        tn = np.sum((pred == 0) & (y_true == 0))
        tpr.append(tp / (tp + fn + 1e-9))
        fpr.append(fp / (fp + tn + 1e-9))
    # AUC por regla trapezoidal
    order = np.argsort(fpr)
    # Use trapezoid (NumPy 2.0+) or fallback to trapz (NumPy < 2.0)
    try:
        auc = np.trapezoid(np.array(tpr)[order], np.array(fpr)[order])
    except AttributeError:
        auc = np.trapz(np.array(tpr)[order], np.array(fpr)[order])
    return np.array(fpr)[order], np.array(tpr)[order], float(auc)

# ---------------------------------------------------------------------
# Flujo principal
# ---------------------------------------------------------------------

def main(io: IOPaths, gen: GenCfg, nna: NNACfg, mia: MIACfg) -> None:
    logger.info("="*80)
    logger.info("Starting ccDDPM Privacy Audit")
    logger.info("="*80)

    # Log configuration
    logger.info("Configuration:")
    logger.info(f"  IO Paths:")
    logger.info(f"    - YAML: {io.yaml}")
    logger.info(f"    - Checkpoint: {io.checkpoint}")
    logger.info(f"    - Output Directory: {io.outdir}")
    logger.info(f"  Generation:")
    logger.info(f"    - Use existing synth: {gen.use_existing_synth}")
    if gen.use_existing_synth:
        logger.info(f"    - Existing synth split: {gen.existing_synth_split}")
    else:
        logger.info(f"    - Per class: {gen.per_class}")
        logger.info(f"    - Guidance scale: {gen.guidance_scale}")
        logger.info(f"    - Inference steps: {gen.num_inference_steps}")
    logger.info(f"    - Seed: {gen.seed}")
    logger.info(f"  Nearest Neighbor:")
    logger.info(f"    - LPIPS threshold: {nna.lpips_threshold}")
    logger.info(f"    - SSIM threshold: {nna.ssim_threshold}")
    logger.info(f"    - Use CLIP: {nna.use_clip}")
    logger.info(f"    - Batch size: {nna.batch_size}")
    logger.info(f"  Membership Inference:")
    logger.info(f"    - Timesteps per sample: {mia.n_timesteps_per_sample}")
    logger.info(f"    - AUC points: {mia.auc_points}")

    # Dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Set random seeds
    torch.manual_seed(gen.seed); np.random.seed(gen.seed); random.seed(gen.seed)
    logger.info(f"Random seeds set to {gen.seed}")

    # Cargar config/modelo/scheduler
    logger.info("Loading model and scheduler from project YAML...")
    start_time = time.time()
    model, scheduler, meta = build_model_and_scheduler(io.yaml, device)
    logger.info(f"Model and scheduler built in {time.time() - start_time:.2f}s")

    logger.info("Loading checkpoint...")
    start_time = time.time()
    ckpt_meta = load_checkpoint_(model, io.checkpoint)
    logger.info(f"Checkpoint loaded in {time.time() - start_time:.2f}s")
    logger.info("Model metadata: %s", json.dumps(meta, indent=2))

    # Cargar NPZ
    logger.info("Loading dataset from NPZ...")
    cfg_loaded = load_cfg(io.yaml, split="train")
    npz_path = cfg_loaded.ccddpm.dataloader.npz_path
    if npz_path is None or not Path(npz_path).exists():
        logger.error(f"NPZ file not found at: {npz_path}")
        raise FileNotFoundError("No se encontró NPZ en YAML data.postprocess_npz.npz_path")

    logger.info(f"Loading NPZ from: {npz_path}")
    start_time = time.time()
    splits = load_npz_splits(Path(npz_path))
    logger.info(f"NPZ loaded in {time.time() - start_time:.2f}s")

    # Convertir a tensores normalizados [-1,1]
    def npimgs_to_tensor(arr: np.ndarray) -> torch.Tensor:
        # uint8 HWC -> CHW float32 en [-1,1]
        t = torch.from_numpy(arr).permute(0,3,1,2).float() / 255.0
        return t * 2.0 - 1.0

    # Separate REAL and SYNTHETIC samples
    logger.info("Separating REAL and SYNTHETIC samples...")
    synth_splits = {}  # Store synthetic samples separately
    for split_name in splits.keys():
        is_synth = splits[split_name]["is_synth"]
        real_mask = ~is_synth
        synth_mask = is_synth
        n_total = len(is_synth)
        n_synth = synth_mask.sum()
        n_real = real_mask.sum()

        if n_synth > 0:
            logger.info(f"  {split_name}: {n_real} real, {n_synth} synthetic samples")
            # Save synthetic samples separately
            synth_splits[split_name] = {
                "images": splits[split_name]["images"][synth_mask],
                "labels": splits[split_name]["labels"][synth_mask],
                "is_synth": splits[split_name]["is_synth"][synth_mask]
            }
            # Keep only real samples in splits
            splits[split_name]["images"] = splits[split_name]["images"][real_mask]
            splits[split_name]["labels"] = splits[split_name]["labels"][real_mask]
            splits[split_name]["is_synth"] = splits[split_name]["is_synth"][real_mask]
        else:
            logger.info(f"  {split_name}: All {n_total} samples are real (no synthetic samples)")
            synth_splits[split_name] = {"images": np.array([]), "labels": np.array([]), "is_synth": np.array([])}

    logger.info("Converting images to tensors...")
    x_train = npimgs_to_tensor(splits["train"]["images"])
    y_train = torch.from_numpy(splits["train"]["labels"].astype(np.int64))

    # Handle val split (always present for audit)
    x_val = npimgs_to_tensor(splits["val"]["images"]) if "val" in splits else None
    y_val = torch.from_numpy(splits["val"]["labels"].astype(np.int64)) if "val" in splits else None

    # Handle test split (optional - use val as fallback for MIA)
    if "test" in splits:
        x_test = npimgs_to_tensor(splits["test"]["images"])
        y_test = torch.from_numpy(splits["test"]["labels"].astype(np.int64))
        logger.info("Using 'test' split as non-members for membership inference")
    elif x_val is not None:
        x_test = x_val
        y_test = y_val
        logger.warning("No 'test' split found - using 'val' split as non-members for membership inference")
    else:
        raise ValueError("Dataset must have at least 'train' and either 'val' or 'test' splits")

    logger.info(f"Dataset loaded (REAL samples only):")
    logger.info(f"  - Train: {x_train.shape[0]} samples")
    if x_val is not None and "val" in splits:
        logger.info(f"  - Val: {x_val.shape[0]} samples")
    logger.info(f"  - Test/Non-members: {x_test.shape[0]} samples")

    outdir = io.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory created: {outdir}")

    # ---------------- A) Generación y k-NN/LPIPS/SSIM ----------------
    logger.info("="*80)
    logger.info("Phase 1: Synthetic Samples and Memorization Detection (k-NN/LPIPS/SSIM)")
    logger.info("="*80)

    # Check if we should use existing synthetic samples or generate new ones
    if gen.use_existing_synth:
        logger.info(f"Using EXISTING synthetic samples from '{gen.existing_synth_split}' split")
        start_time = time.time()

        # Validate that the requested split has synthetic samples
        if gen.existing_synth_split not in synth_splits:
            raise ValueError(f"Split '{gen.existing_synth_split}' not found in dataset")

        synth_data = synth_splits[gen.existing_synth_split]
        if len(synth_data["images"]) == 0:
            raise ValueError(
                f"No synthetic samples found in '{gen.existing_synth_split}' split. "
                f"Either generate samples first or set use_existing_synth=false to generate new ones."
            )

        # Convert to tensors
        imgs_s = npimgs_to_tensor(synth_data["images"])
        labels_s = torch.from_numpy(synth_data["labels"].astype(np.int64))

        logger.info(f"Loaded {imgs_s.shape[0]} existing synthetic samples in {time.time() - start_time:.2f}s")
        logger.info(f"  Shape: {imgs_s.shape}")
        logger.info(f"  Classes distribution: {np.bincount(synth_data['labels'])}")
    else:
        logger.info(f"GENERATING new synthetic samples ({gen.per_class} per class)...")
        start_time = time.time()
        imgs_s, labels_s = ddpm_sample_ccddpm(
            model=model, scheduler=scheduler,
            num_classes=meta["num_classes"], per_class=gen.per_class,
            guidance_scale=gen.guidance_scale, num_inference_steps=gen.num_inference_steps,
            image_size=meta["image_size"], device=device
        )
        logger.info(f"Generation completed in {time.time() - start_time:.2f}s")
        logger.info(f"Generated {imgs_s.shape[0]} samples with shape {imgs_s.shape}")

    if gen.save_grid:
        grid_path = outdir / "samples_grid.png"
        save_image((imgs_s+1)/2, grid_path, nrow=gen.per_class, padding=2)
        logger.info(f"Sample grid saved to: {grid_path}")
        logger.info(f"  - File size: {grid_path.stat().st_size / 1024:.2f} KB")

    # Embeddings
    logger.info("Extracting embeddings for similarity search...")
    emb_model = EmbeddingModel(use_clip=nna.use_clip).to(device)
    logger.info(f"Using embedding model: {'CLIP' if nna.use_clip and getattr(emb_model, '_clip', False) else 'ResNet50'}")
    logger.info(f"Embedding dimension: {emb_model.dim}")

    logger.info("  - Embedding synthetic samples...")
    start_time = time.time()
    emb_s = batched_embed(emb_model, imgs_s, nna.batch_size, device)
    logger.info(f"    Completed in {time.time() - start_time:.2f}s, shape: {emb_s.shape}")

    logger.info("  - Embedding training samples...")
    start_time = time.time()
    emb_tr = batched_embed(emb_model, x_train, nna.batch_size, device)
    logger.info(f"    Completed in {time.time() - start_time:.2f}s, shape: {emb_tr.shape}")

    logger.info("  - Embedding test samples...")
    start_time = time.time()
    emb_te = batched_embed(emb_model, x_test, nna.batch_size, device)
    logger.info(f"    Completed in {time.time() - start_time:.2f}s, shape: {emb_te.shape}")

    # Distancias coseno -> convertir a "distancia" (1 - cos_sim)
    def pairwise_cdist_cos(A: torch.Tensor, B: torch.Tensor, bsz: int = 1024) -> np.ndarray:
        out = []
        for i in tqdm(range(0, A.size(0), bsz), desc="kNN blocks", leave=False):
            a = A[i:i+bsz]
            # sim = a @ B^T
            sim = (a @ B.t()).clamp(-1,1).cpu().numpy()
            out.append(1.0 - sim)
        return np.concatenate(out, 0)

    logger.info("Computing nearest neighbors in embedding space...")
    start_time = time.time()
    D_tr = pairwise_cdist_cos(emb_s, emb_tr)   # [Ns, Ntrain]
    D_te = pairwise_cdist_cos(emb_s, emb_te)   # [Ns, Ntest]
    nn_tr_idx = np.argmin(D_tr, axis=1); nn_tr = D_tr[np.arange(D_tr.shape[0]), nn_tr_idx]
    nn_te_idx = np.argmin(D_te, axis=1); nn_te = D_te[np.arange(D_te.shape[0]), nn_te_idx]
    logger.info(f"Nearest neighbor search completed in {time.time() - start_time:.2f}s")
    logger.info(f"  - Mean NN distance to train: {nn_tr.mean():.4f} ± {nn_tr.std():.4f}")
    logger.info(f"  - Mean NN distance to test: {nn_te.mean():.4f} ± {nn_te.std():.4f}")

    # LPIPS/SSIM para pares NN de train y test
    logger.info("Computing LPIPS and SSIM for nearest neighbor pairs...")
    if not _HAS_LPIPS:
        logger.warning("LPIPS not available. Using L2 distance approximation instead.")

    start_time = time.time()
    lpips_tr, ssim_tr, lpips_te, ssim_te = [], [], [], []
    for i in tqdm(range(imgs_s.size(0)), desc="Pair metrics", leave=False):
        lp_tr, ss_tr = compute_pair_metrics(imgs_s[i], x_train[nn_tr_idx[i]], device)
        lp_te, ss_te = compute_pair_metrics(imgs_s[i], x_test[nn_te_idx[i]], device)
        lpips_tr.append(lp_tr); ssim_tr.append(ss_tr)
        lpips_te.append(lp_te); ssim_te.append(ss_te)
    lpips_tr = np.array(lpips_tr); ssim_tr = np.array(ssim_tr)
    lpips_te = np.array(lpips_te); ssim_te = np.array(ssim_te)
    logger.info(f"Pair metrics computed in {time.time() - start_time:.2f}s")
    logger.info(f"  - LPIPS train: {lpips_tr.mean():.4f} ± {lpips_tr.std():.4f}")
    logger.info(f"  - SSIM train: {ssim_tr.mean():.4f} ± {ssim_tr.std():.4f}")
    logger.info(f"  - LPIPS test: {lpips_te.mean():.4f} ± {lpips_te.std():.4f}")
    logger.info(f"  - SSIM test: {ssim_te.mean():.4f} ± {ssim_te.std():.4f}")

    suspicious = (lpips_tr <= nna.lpips_threshold) & (ssim_tr >= nna.ssim_threshold) & (nn_tr < nn_te)
    logger.info(f"Identified {suspicious.sum()} suspicious samples ({100.0 * suspicious.mean():.2f}%)")

    # Guardar CSV de sospechosos
    logger.info("Saving memorization suspects to CSV...")
    csv_path = outdir / "memorization_suspects.csv"
    import csv
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["idx_synth","class","nn_train_idx","nn_train_d","lpips_train","ssim_train",
                    "nn_test_idx","nn_test_d","lpips_test","ssim_test","flag_suspect"])
        for i in range(imgs_s.size(0)):
            w.writerow([i, int(labels_s[i].item()),
                        int(nn_tr_idx[i]), float(nn_tr[i]), float(lpips_tr[i]), float(ssim_tr[i]),
                        int(nn_te_idx[i]), float(nn_te[i]), float(lpips_te[i]), float(ssim_te[i]),
                        bool(suspicious[i])])
    logger.info(f"CSV saved to: {csv_path}")
    logger.info(f"  - File size: {csv_path.stat().st_size / 1024:.2f} KB")
    logger.info(f"  - Rows: {imgs_s.size(0)} + 1 header")

    # ---------------- B) Membership inference (Yeom adaptado) ----------------
    logger.info("="*80)
    logger.info("Phase 2: Membership Inference Attack")
    logger.info("="*80)
    # Tomamos un subconjunto balanceado si es enorme para acelerar
    logger.info("Selecting subsets for membership inference...")
    def select_subset(x: torch.Tensor, y: torch.Tensor, max_n: int = 5000) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.size(0) <= max_n:
            return x, y
        idx = torch.randperm(x.size(0))[:max_n]
        return x[idx], y[idx]

    x_tr_sub, y_tr_sub = select_subset(x_train, y_train)
    x_te_sub, y_te_sub = select_subset(x_test, y_test)
    logger.info(f"  - Train subset: {x_tr_sub.shape[0]} samples")
    logger.info(f"  - Test subset: {x_te_sub.shape[0]} samples")

    # Calcular pérdidas promedio por muestra
    logger.info("Computing per-sample denoising losses...")
    def batched_losses(x: torch.Tensor, y: torch.Tensor, n_t: int) -> np.ndarray:
        vals = []
        for i in tqdm(range(0, x.size(0), 128), desc="Per-sample loss", leave=False):
            xb = x[i:i+128].to(device); yb = y[i:i+128].to(device)
            vals.append(per_sample_denoising_loss(model, scheduler, xb, yb, n_t, device).cpu().numpy())
        return np.concatenate(vals, 0)

    logger.info("  - Computing losses for training samples (members)...")
    start_time = time.time()
    loss_in  = batched_losses(x_tr_sub, y_tr_sub, mia.n_timesteps_per_sample)   # miembros
    logger.info(f"    Completed in {time.time() - start_time:.2f}s")
    logger.info(f"    Mean loss: {loss_in.mean():.6f} ± {loss_in.std():.6f}")

    logger.info("  - Computing losses for test samples (non-members)...")
    start_time = time.time()
    loss_out = batched_losses(x_te_sub, y_te_sub, mia.n_timesteps_per_sample)   # no-miembros
    logger.info(f"    Completed in {time.time() - start_time:.2f}s")
    logger.info(f"    Mean loss: {loss_out.mean():.6f} ± {loss_out.std():.6f}")

    logger.info("Computing ROC curve and AUC...")
    start_time = time.time()
    fpr, tpr, auc = roc_auc(loss_in, loss_out, mia.auc_points)
    logger.info(f"ROC curve computed in {time.time() - start_time:.2f}s")
    logger.info(f"  - AUC: {auc:.4f}")

    roc_csv_path = outdir / "mia_roc.csv"
    with open(roc_csv_path, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["fpr","tpr"]); w.writerows(zip(fpr.tolist(), tpr.tolist()))
    logger.info(f"ROC curve saved to: {roc_csv_path}")
    logger.info(f"  - File size: {roc_csv_path.stat().st_size / 1024:.2f} KB")
    logger.info(f"  - Rows: {len(fpr)} + 1 header")

    # ---------------- Guardado agregado ----------------
    logger.info("="*80)
    logger.info("Saving aggregated results...")
    logger.info("="*80)

    npz_path = outdir / "audit_ccddpm_privacy.npz"
    logger.info(f"Saving all metrics to NPZ: {npz_path}")
    start_time = time.time()
    np.savez_compressed(
        npz_path,
        synth_labels=labels_s.cpu().numpy(),
        nn_train_dist=nn_tr, nn_train_idx=nn_tr_idx,
        nn_test_dist=nn_te, nn_test_idx=nn_te_idx,
        lpips_train=lpips_tr, ssim_train=ssim_tr,
        lpips_test=lpips_te, ssim_test=ssim_te,
        suspicious=suspicious.astype(np.bool_),
        mia_loss_in=loss_in, mia_loss_out=loss_out,
        mia_fpr=fpr, mia_tpr=tpr, mia_auc=np.array([auc], dtype=np.float32),
        meta=json.dumps({"model": meta, "ckpt_meta_keys": list(ckpt_meta.keys())}),
    )
    logger.info(f"NPZ saved in {time.time() - start_time:.2f}s")
    logger.info(f"  - File path: {npz_path}")
    logger.info(f"  - File size: {npz_path.stat().st_size / (1024*1024):.2f} MB")
    logger.info(f"  - Arrays saved: synth_labels, nn_train_dist, nn_train_idx, nn_test_dist, nn_test_idx,")
    logger.info(f"                  lpips_train, ssim_train, lpips_test, ssim_test, suspicious,")
    logger.info(f"                  mia_loss_in, mia_loss_out, mia_fpr, mia_tpr, mia_auc, meta")

    summary = {
        "memorization": {
            "num_suspicious": int(suspicious.sum()),
            "rate": float(np.mean(suspicious)),
            "lpips_threshold": nna.lpips_threshold,
            "ssim_threshold": nna.ssim_threshold
        },
        "membership_inference": {
            "auc": auc,
            "n_in": int(loss_in.size), "n_out": int(loss_out.size),
            "n_timesteps_per_sample": mia.n_timesteps_per_sample
        }
    }

    json_path = outdir / "audit_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary saved to: {json_path}")
    logger.info(f"  - File size: {json_path.stat().st_size / 1024:.2f} KB")

    logger.info("="*80)
    logger.info("AUDIT SUMMARY")
    logger.info("="*80)
    logger.info(json.dumps(summary, indent=2))
    logger.info("="*80)
    logger.info("Audit completed successfully!")
    logger.info(f"All results saved to: {outdir}")
    logger.info("="*80)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Auditoría de privacidad ccDDPM (memorization + membership inference)")
    p.add_argument("--config", type=Path, default=None,
                   help="Path to audit configuration YAML file (configs/audit_ccddpm_privacy.yaml)")
    p.add_argument("--yaml", type=Path, default=None, help="Ruta al YAML del proyecto")
    p.add_argument("--checkpoint", type=Path, default=None, help="Ruta al checkpoint .pt")
    p.add_argument("--outdir", type=Path, default=None, help="Directorio de salida")
    p.add_argument("--per-class", type=int, default=64)
    p.add_argument("--guidance-scale", type=float, default=2.0)
    p.add_argument("--steps", type=int, default=250)
    p.add_argument("--knn-k", type=int, default=5)
    p.add_argument("--lpips-th", type=float, default=0.12)
    p.add_argument("--ssim-th", type=float, default=0.90)
    p.add_argument("--use-clip", action="store_true")
    p.add_argument("--mia-n-t", type=int, default=32)
    return p.parse_args()

def cli_main():
    """Entry point for the ccddpm-audit console command."""
    args = parse_args()

    # Load configuration from YAML if provided
    config = load_audit_config(args.config)

    # Setup logging first
    setup_logging(config)

    # Merge config with command-line arguments
    io, gen, nna, mia = merge_config_with_args(config, args)

    # Validate required arguments
    if not io.yaml or not io.yaml.exists():
        logger.error("Project YAML file is required and must exist!")
        logger.error("Provide via --yaml argument or in config file under io.yaml")
        exit(1)
    if not io.checkpoint or not io.checkpoint.exists():
        logger.error("Checkpoint file is required and must exist!")
        logger.error("Provide via --checkpoint argument or in config file under io.checkpoint")
        exit(1)
    if not io.outdir:
        logger.error("Output directory is required!")
        logger.error("Provide via --outdir argument or in config file under io.outdir")
        exit(1)

    # Run the audit
    main(io, gen, nna, mia)

if __name__ == "__main__":
    cli_main()
