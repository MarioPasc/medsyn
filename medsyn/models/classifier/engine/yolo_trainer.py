from __future__ import annotations
from ultralytics.models.yolo.classify.train import ClassificationTrainer
from ultralytics.utils.torch_utils import de_parallel
from ultralytics.utils import RANK
from medsyn.models.classifier.dataloaders import build_npz_loader

class MedsynClassificationTrainer(ClassificationTrainer):
    """Ultralytics trainer that feeds from NPZ instead of folders."""

    def build_dataset(self, img_path: str, mode: str = "train", batch=None):
        # Not used; training/validation will call our get_dataloader
        return super().build_dataset(img_path, mode, batch)

    def get_dataloader(self, dataset_path: str, batch_size: int = 16, rank: int = 0, mode: str = "train"):
        args = self.args
        return build_npz_loader(
            npz_path=args.medsyn_npz_path,
            split={"train":"train","val":"val","test":"test"}[mode],
            imgsz=args.imgsz,
            batch=batch_size,
            workers=args.workers,
            training_images=args.medsyn_training_images,
            augment=(mode=="train"),
        )

    def set_model_attributes(self):
        # Let parent set class names from dataset if available
        super().set_model_attributes()
        # Attach transforms to model for export/val
        if RANK in (-1, 0):
            m = de_parallel(self.model)
            m.args = self.args
