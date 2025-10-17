# medsyn/models/ccDDPM/dataloader.py
# Purpose: Dataset from the JSON index produced by medsyn/cli/data.py
# Output tensors: pixel_values in [-1,1], labels as int64, and metadata.
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Tuple, List
import json
import logging
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

logger = logging.getLogger(__name__)

@dataclass
class Sample:
    path: Path
    label: int
    is_synth: bool

class PathMNISTIndexDataset(Dataset):
    """
    Dataset backed by a MedSyn JSON index:
      { "PathMNIST": { "train": { "0": {"image": "...", "label": int, "is_synth": bool}, ... } } }
    """
    def __init__(self, index_json: Path, split: str = "train", image_size: int = 128, normalize: bool = True):
        with open(index_json, "r", encoding="utf-8") as fh:
            data = json.load(fh)["PathMNIST"][split]
        self.items: List[Sample] = []
        for _, rec in data.items():
            self.items.append(Sample(Path(rec["image"]).resolve(), int(rec["label"]), bool(rec.get("is_synth", False))))
        self.transform = T.Compose([
            T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.ConvertImageDtype(torch.float32),
            T.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]) if normalize else (lambda x: x),
        ])
        logger.info("IndexDataset split=%s size=%d", split, len(self.items))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        s = self.items[i]
        img = Image.open(s.path).convert("RGB")
        x = self.transform(img)
        return {"pixel_values": x, "labels": torch.tensor(s.label, dtype=torch.long), "path": str(s.path), "is_synth": s.is_synth}

def build_loader(index_json: Path, split: str, image_size: int, batch_size: int, num_workers: int, normalize: bool = True) -> DataLoader:
    """
    Build torch DataLoader with sane defaults.
    """
    ds = PathMNISTIndexDataset(index_json, split=split, image_size=image_size, normalize=normalize)
    return DataLoader(ds, batch_size=batch_size, shuffle=(split=="train"), num_workers=num_workers, pin_memory=True, drop_last=(split=="train"))
