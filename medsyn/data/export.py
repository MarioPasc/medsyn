# data/export.py
from __future__ import annotations
from typing import Dict, Any
from pathlib import Path
import logging
import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_pil_image

logger = logging.getLogger(__name__)


def _save_tensor_as_png(x: torch.Tensor, path: Path) -> None:
    """
    Guarda un tensor CxHxW en PNG RGB/Grayscale sin alterar valores [0,1].
    """
    img = to_pil_image(x.cpu().detach())  # maneja 1 o 3 canales
    img.save(path, format="PNG", compress_level=6)


def export_split_to_pngs_and_index(dataset: Dataset, out_dir: Path) -> Dict[int, Dict[str, Any]]:
    """
    Exporta un Dataset (o Subset) de MedMNIST a archivos PNG y construye un índice:
      { idx: {"image": "/abs/path.png", "label": int, "is_synth": false} }

    Parámetros:
        dataset: conjunto PyTorch indexable que devuelve (imagen, etiqueta)
        out_dir: directorio destino para las imágenes PNG

    Retorna:
        dict de índices con metadatos por entrada.
    """
    out: Dict[int, Dict[str, Any]] = {}
    out_dir.mkdir(parents=True, exist_ok=True)

    for i in range(len(dataset)): # type: ignore
        x, y = dataset[i]                      # x: Tensor CxHxW, y: Tensor o int
        label = int(y) if not isinstance(y, int) else y
        fname = f"{i:06d}_{label}.png"
        fpath = (out_dir / fname).resolve()

        if not fpath.exists():
            _save_tensor_as_png(x, fpath)

        out[i] = {
            "image": str(fpath),
            "label": label,
            "is_synth": False
        }

        if (i + 1) % 1000 == 0:
            logger.info("  %d  processed samples ...", i + 1)

    return out
