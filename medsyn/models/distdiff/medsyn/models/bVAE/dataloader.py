# medsyn/models/bVAE/dataloader.py
# Purpose: Torch Dataset/DataLoader over the PathMNIST JSON index.

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Literal, Optional
import json
import logging
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

logger = logging.getLogger(__name__)

# --------------------------- Dataset ------------------------------------------

class JSONIndexImageDataset(Dataset):
    # Returns (image_tensor, label_int) from a split entry in the JSON index.
    def __init__(self, index: Dict[str, Any], split: Literal["train","val","test"], img_size: int):
        super().__init__()
        self.split = split
        self._entries = index["PathMNIST"][split]
        self._keys = sorted(self._entries.keys(), key=lambda k: int(k))
        self.tfm = transforms.Compose([
            transforms.Resize(img_size) if img_size != 28 else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),  # [0,1]
        ])

    def __len__(self) -> int:
        return len(self._keys)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        k = self._keys[i]
        rec = self._entries[k]
        img = Image.open(rec["image"]).convert("RGB")
        x = self.tfm(img)
        y = int(rec["label"])
        return x, y

# --------------------------- Factory ------------------------------------------

@dataclass(frozen=True)
class LoaderPack:
    train: DataLoader
    val: DataLoader
    test: DataLoader

def make_loaders(index_json: str, img_size: int, batch_size: int, num_workers: int, seed: int) -> LoaderPack:
    # Build deterministic loaders for train/val/test from JSON index.
    with Path(index_json).open("r", encoding="utf-8") as fh:
        idx = json.load(fh)

    g = torch.Generator()
    g.manual_seed(seed)

    ds_train = JSONIndexImageDataset(idx, "train", img_size)
    ds_val   = JSONIndexImageDataset(idx, "val",   img_size)
    ds_test  = JSONIndexImageDataset(idx, "test",  img_size)

    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True, generator=g)
    dl_val   = DataLoader(ds_val,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    dl_test  = DataLoader(ds_test,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    logger.info("Loaders: train=%d val=%d test=%d", len(ds_train), len(ds_val), len(ds_test))
    return LoaderPack(train=dl_train, val=dl_val, test=dl_test)
