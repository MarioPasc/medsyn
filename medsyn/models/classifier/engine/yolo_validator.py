from __future__ import annotations
from ultralytics.models.yolo.classify.val import ClassificationValidator
from medsyn.models.classifier.dataloaders import build_npz_loader

class MedsynClassificationValidator(ClassificationValidator):
    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="val"):
        args = self.args or self.trainer.args
        return build_npz_loader(
            npz_path=args.medsyn_npz_path,
            split={"train":"train","val":"val","test":"test"}[mode],
            imgsz=args.imgsz,
            batch=batch_size,
            workers=args.workers,
            training_images=args.medsyn_training_images,
            augment=False,
        )
