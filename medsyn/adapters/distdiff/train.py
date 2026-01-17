# medsyn/adapters/distdiff/train.py
"""
DistDiff guide model training wrapper.

This module provides a standardized training interface for the DistDiff
guide model (classifier), accepting YAML configuration.

The guide model is used to extract prototypes for prototype-guided generation.
"""
from __future__ import annotations

import os
import sys
import shutil
import time
import random
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import yaml

from medsyn.adapters.distdiff.config import DistDiffTrainConfig, load_distdiff_train_config
from medsyn.adapters.distdiff.dataset import load_npz_dataset, get_class_names

logger = logging.getLogger(__name__)


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions."""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def set_random_seed(seed: Optional[int]):
    """Set random seed for reproducibility."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def create_model(
    arch: str,
    num_classes: int,
    pretrained: bool,
    class_names: list,
    cache_dir: Optional[str] = None,
    dataset_name: str = "pathmnist",
):
    """
    Create the guide model.

    Imports from DistDiff's model_utils to avoid code duplication.
    """
    # Add DistDiff model directory to path
    distdiff_path = Path(__file__).parent.parent.parent / "models" / "distdiff"
    if str(distdiff_path) not in sys.path:
        sys.path.insert(0, str(distdiff_path))

    from model_utils import create_model as distdiff_create_model

    return distdiff_create_model(
        arch,
        num_classes=num_classes,
        pretrained=pretrained,
        class_names=class_names,
        cache_dir=cache_dir,
        dataset_name=dataset_name,
    )


def train_epoch(
    train_loader: DataLoader,
    model: nn.Module,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    use_cuda: bool,
    accumulate: int = 0,
) -> tuple:
    """Train for one epoch."""
    model.train()

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        if accumulate == 0:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            loss = loss / accumulate
            loss.backward()
            if ((batch_idx + 1) % accumulate) == 0:
                optimizer.step()
                optimizer.zero_grad()

        if batch_idx % 50 == 0:
            logger.info(
                f'Train Epoch: {epoch} [{batch_idx}/{len(train_loader)}] '
                f'Loss: {losses.avg:.4f} Acc@1: {top1.avg:.2f}% Acc@5: {top5.avg:.2f}%'
            )

    torch.cuda.empty_cache()
    return losses.avg, top1.avg


def validate(
    val_loader: DataLoader,
    model: nn.Module,
    criterion: nn.Module,
    use_cuda: bool,
) -> tuple:
    """Validate the model."""
    model.eval()

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    with torch.no_grad():
        for inputs, targets in val_loader:
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))

    torch.cuda.empty_cache()
    return losses.avg, top1.avg


def save_checkpoint(state, is_best, checkpoint_dir, filename='checkpoint.pth.tar'):
    """Save checkpoint and optionally copy best model."""
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)
    if is_best:
        logger.info(f"New best model with accuracy: {state['best_acc']:.2f}%")
        shutil.copyfile(filepath, os.path.join(checkpoint_dir, 'model_best.pth.tar'))


def train(config_path: str) -> dict:
    """
    Train the DistDiff guide model.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Dictionary with training results (best_accuracy, last_accuracy)
    """
    # Load configuration
    cfg = load_distdiff_train_config(config_path)

    # Set random seed
    set_random_seed(cfg.seed)

    # Create output directory
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    log_file = output_dir / "training.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logging.getLogger().addHandler(file_handler)

    logger.info(f"Starting DistDiff guide model training")
    logger.info(f"Config: {cfg}")

    # Setup transforms
    transform_train = transforms.Compose([
        transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomRotation(15),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    transform_test = transforms.Compose([
        transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load datasets
    logger.info(f"Loading dataset from: {cfg.npz_path}")
    trainset = load_npz_dataset(
        cfg.npz_path, 'train', transform_train, filter_synthetic=True
    )
    testset = load_npz_dataset(
        cfg.npz_path, 'test', transform_test, filter_synthetic=False
    )

    # Get class names
    class_names = get_class_names(cfg.dataset_name)
    num_classes = len(class_names)
    logger.info(f"Dataset: {cfg.dataset_name}, num_classes: {num_classes}")

    # Create dataloaders
    trainloader = DataLoader(
        trainset,
        batch_size=cfg.train_batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
    )
    testloader = DataLoader(
        testset,
        batch_size=cfg.val_batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )

    # Create model
    logger.info(f"Creating model: {cfg.arch}")
    model = create_model(
        cfg.arch,
        num_classes=num_classes,
        pretrained=cfg.pretrained,
        class_names=class_names,
        cache_dir=cfg.cache_dir,
        dataset_name=cfg.dataset_name,
    )

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model = torch.nn.DataParallel(model).cuda()
        cudnn.benchmark = True

    logger.info(f'Total params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M')

    # Setup training
    criterion = nn.CrossEntropyLoss()

    if cfg.train_fc:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.module.fc.parameters():
            param.requires_grad = True
        optimizer = optim.SGD(
            model.module.fc.parameters(),
            lr=cfg.lr,
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay,
            nesterov=True,
        )
    else:
        optimizer = optim.SGD(
            model.parameters(),
            lr=cfg.lr,
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay,
            nesterov=True,
        )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.epochs)

    # Training loop
    best_acc = 0.0
    for epoch in range(cfg.epochs):
        lr = optimizer.param_groups[0]['lr']
        logger.info(f'Epoch: [{epoch + 1}/{cfg.epochs}] LR: {lr:.6f}')

        train_loss, train_acc = train_epoch(
            trainloader, model, criterion, optimizer, epoch, use_cuda
        )
        test_loss, test_acc = validate(testloader, model, criterion, use_cuda)

        logger.info(
            f'Epoch {epoch + 1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}%, '
            f'Test Loss={test_loss:.4f}, Test Acc={test_acc:.2f}%'
        )

        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)

        save_checkpoint(
            {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'acc': test_acc,
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
            },
            is_best,
            str(output_dir),
        )

        scheduler.step()

    logger.info(f'Training completed. Best accuracy: {best_acc:.2f}%')

    # Save results
    result = {
        "best_accuracy": best_acc,
        "last_accuracy": test_acc,
    }
    with open(output_dir / "results.yaml", 'w') as f:
        yaml.dump(result, f)

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train DistDiff guide model")
    parser.add_argument("config", help="Path to YAML configuration file")
    args = parser.parse_args()

    train(args.config)
