#!/usr/bin/env python
from __future__ import annotations
import argparse
from ultralytics.models.yolo.classify import ClassificationTrainer
from medsyn.models.classifier.config import load_cfg
from medsyn.models.classifier.engine.yolo_trainer import MedsynClassificationTrainer
from medsyn.models.classifier.engine.yolo_validator import MedsynClassificationValidator

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="medsyn_cfg.yaml")
    ap.add_argument("--hparams", required=True, help="yolo_hyperparameters.yaml")
    ap.add_argument("--val_only", action="store_true")
    args = ap.parse_args()

    cfg = load_cfg(args.config, args.hparams)

    # Trainer with NPZ-backed dataloaders
    trainer: ClassificationTrainer = MedsynClassificationTrainer(overrides=cfg.overrides)
    trainer.validator = MedsynClassificationValidator(args=trainer.args)
    if args.val_only:
        trainer.validate()
    else:
        trainer.train()

if __name__ == "__main__":
    main()
