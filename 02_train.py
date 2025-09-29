#!/usr/bin/env python3
"""
train.py — universal image-classification trainer with Weights & Biases logging.

Requirements:
    pip install torch torchvision timm scikit-learn wandb

Expected data structure:
    data_dir/
      train/<class1|class2|...>/*.jpg
      val/<class1|class2|...>/*.jpg
      test/<class1|class2|...>/*.jpg
"""

import argparse
import os
import time
import json
import math
import random
from pathlib import Path
from typing import Tuple, Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

# timm for ConvNeXt, ViT, Swin, RegNet etc.
try:
    import timm
    TIMM_AVAILABLE = True
except Exception:
    TIMM_AVAILABLE = False

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

import wandb


# ------------------------------------------------------
# Utility functions
# ------------------------------------------------------
def set_seed(seed: int = 42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def count_trainable_params(model: nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_device():
    """Return GPU device if available, else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------------------------------
# Data
# ------------------------------------------------------
def build_transforms(img_size: int = 224, aug: bool = True):
    """Return torchvision transforms for training and evaluation."""
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    train_tf = transforms.Compose([
        transforms.Resize(int(img_size*1.15)),
        transforms.CenterCrop(img_size) if not aug else transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip() if aug else transforms.Lambda(lambda x: x),
        transforms.RandomApply([transforms.ColorJitter(0.2, 0.2, 0.2, 0.1)], p=0.3) if aug else transforms.Lambda(lambda x: x),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize(int(img_size*1.15)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return train_tf, eval_tf


def build_dataloaders(data_dir: str, img_size: int, batch_size: int, workers: int, aug: bool) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """
    Build DataLoaders for train, val, test splits.
    Returns loaders and list of class names.
    """
    train_tf, eval_tf = build_transforms(img_size, aug)

    train_ds = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=train_tf)
    val_ds   = datasets.ImageFolder(os.path.join(data_dir, "val"),   transform=eval_tf)
    test_ds  = datasets.ImageFolder(os.path.join(data_dir, "test"),  transform=eval_tf)

    class_names = train_ds.classes

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True, drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)

    return train_loader, val_loader, test_loader, class_names


# ------------------------------------------------------
# Model builders
# ------------------------------------------------------
TV_MODELS = {
    "resnet50": lambda num_classes: _tv_resnet50(num_classes),
    "densenet121": lambda num_classes: _tv_densenet121(num_classes),
    "efficientnet_b1": lambda num_classes: _tv_efficientnet_b1(num_classes),
    "mobilenet_v3_large": lambda num_classes: _tv_mobilenet_v3_large(num_classes),
    "shufflenet_v2_x1_0": lambda num_classes: _tv_shufflenet_v2(num_classes),
    "vgg16": lambda num_classes: _tv_vgg16(num_classes),
}

# ---- Torchvision wrappers ----
def _tv_resnet50(num_classes):
    m = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m

def _tv_densenet121(num_classes):
    m = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
    m.classifier = nn.Linear(m.classifier.in_features, num_classes)
    return m

def _tv_mobilenet_v3_large(num_classes):
    m = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
    m.classifier[3] = nn.Linear(m.classifier[3].in_features, num_classes)
    return m

def _tv_shufflenet_v2(num_classes):
    m = models.shufflenet_v2_x1_0(weights=models.ShuffleNet_V2_X1_0_Weights.DEFAULT)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m

def _tv_efficientnet_b1(num_classes):
    m = models.efficientnet_b1(weights=models.EfficientNet_B1_Weights.DEFAULT)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    return m

def _tv_vgg16(num_classes):
    m = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
    m.classifier[6] = nn.Linear(m.classifier[6].in_features, num_classes)
    return m

# ---- Generic builder ----
def build_model(model_name: str, num_classes: int):
    """
    Build a torchvision or timm model with the correct number of classes.
    """
    name = model_name.lower()
    if name in TV_MODELS:
        return TV_MODELS[name](num_classes)

    if not TIMM_AVAILABLE:
        raise ValueError(f"Model '{model_name}' requires timm package. Install: pip install timm")

    timm_map = {
        "convnext_tiny": "convnext_tiny.fb_in22k",
        "vit_b": "vit_base_patch16_224.augreg_in21k",
        "swin_t": "swin_tiny_patch4_window7_224.ms_in22k",
        "regnety_8gf": "regnety_008.pycls_in1k",
    }
    if name not in timm_map and name in timm.list_models(pretrained=True):
        backbone = name
    else:
        backbone = timm_map.get(name, None)
    if backbone is None:
        raise ValueError(f"Unknown model '{model_name}'. Available: {list(TV_MODELS.keys()) + list(timm_map.keys())}")

    m = timm.create_model(backbone, pretrained=True, num_classes=num_classes)
    return m


# ------------------------------------------------------
# Training & Evaluation loops
# ------------------------------------------------------
def train_one_epoch(model, loader, criterion, optimizer, device, scaler, log_images=False, class_names=None, max_log_images=8):
    """Train for one epoch."""
    model.train()
    losses = []
    all_preds, all_targets = [], []

    for step, (imgs, targets) in enumerate(loader):
        imgs, targets = imgs.to(device, non_blocking=True), targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            logits = model(imgs)
            loss = criterion(logits, targets)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        losses.append(loss.item())
        preds = logits.argmax(1)
        all_preds.append(preds.detach().cpu().numpy())
        all_targets.append(targets.detach().cpu().numpy())

        # Log sample images in the first batch of the epoch
        if log_images and step == 0 and class_names is not None:
            grid_imgs = []
            for i in range(min(max_log_images, imgs.size(0))):
                grid_imgs.append(wandb.Image(
                    imgs[i].detach().cpu(),
                    caption=f"target={class_names[targets[i].item()]}, pred={class_names[preds[i].item()]}"
                ))
            wandb.log({"train/sample_images": grid_imgs})

    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_preds)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    return float(np.mean(losses)), acc, prec, rec, f1


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluate on validation or test set."""
    model.eval()
    losses = []
    all_preds, all_targets = [], []

    for imgs, targets in loader:
        imgs, targets = imgs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        with torch.cuda.amp.autocast():
            logits = model(imgs)
            loss = criterion(logits, targets)
        losses.append(loss.item())
        preds = logits.argmax(1)
        all_preds.append(preds.detach().cpu().numpy())
        all_targets.append(targets.detach().cpu().numpy())

    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_preds)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    return float(np.mean(losses)), acc, prec, rec, f1, cm, y_true, y_pred


def save_checkpoint(state: dict, path: Path):
    """Save checkpoint dict to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: str, model: nn.Module, optimizer=None, scaler=None) -> int:
    """Load checkpoint into model and optionally optimizer & scaler."""
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scaler is not None and "scaler" in ckpt and scaler is not None:
        scaler.load_state_dict(ckpt["scaler"])
    start_epoch = ckpt.get("epoch", 0) + 1
    return start_epoch


# ------------------------------------------------------
# Main function
# ------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Universal image classifier trainer with W&B logging")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory with train/val/test folders")
    parser.add_argument("--model", type=str, default="resnet50",
                        help="Choose: resnet50, densenet121, efficientnet_b1, mobilenet_v3_large, "
                             "shufflenet_v2_x1_0, vgg16, convnext_tiny, vit_b, swin_t, regnety_8gf, ...")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=7, help="Early stopping patience")
    parser.add_argument("--freeze_backbone", action="store_true", help="Freeze backbone initially (head-only training)")
    parser.add_argument("--unfreeze_epoch", type=int, default=2, help="Unfreeze backbone after N epochs if frozen")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--project", type=str, default="image-classification")
    parser.add_argument("--entity", type=str, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint .pt file")
    parser.add_argument("--no_aug", action="store_true", help="Disable train-time augmentations")
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--adamw_beta1", type=float, default=0.9)
    parser.add_argument("--adamw_beta2", type=float, default=0.999)
    parser.add_argument("--log_cm_every", type=int, default=1, help="Log confusion matrix every N epochs")
    args = parser.parse_args()

    set_seed(42)
    device = get_device()

    # ----- Data -----
    train_loader, val_loader, test_loader, class_names = build_dataloaders(
        args.data_dir, args.img_size, args.batch_size, args.workers, aug=not args.no_aug
    )
    num_classes = len(class_names)

    # ----- Model -----
    model = build_model(args.model, num_classes=num_classes).to(device)

    # Optionally freeze backbone
    if args.freeze_backbone:
        for name, p in model.named_parameters():
            p.requires_grad = False
        for m in [getattr(model, 'fc', None),
                  getattr(model, 'classifier', None),
                  getattr(model, 'head', None)]:
            if m is None:
                continue
            for p in m.parameters():
                p.requires_grad = True

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(args.adamw_beta1, args.adamw_beta2)
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

    # ----- W&B -----
    wandb.init(
        project=args.project,
        entity=args.entity,
        name=args.run_name,
        config={
            "model": args.model,
            "img_size": args.img_size,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "patience": args.patience,
            "freeze_backbone": args.freeze_backbone,
            "unfreeze_epoch": args.unfreeze_epoch,
            "label_smoothing": args.label_smoothing,
            "optimizer": "AdamW",
            "scheduler": "CosineAnnealingLR",
            "classes": class_names,
        },
        settings=wandb.Settings(start_method="thread")
    )
    wandb.watch(model, log="all", log_freq=100)

    # ----- Checkpoints -----
    start_epoch = 0
    best_val_acc = -1.0
    ckpt_dir = Path(args.output_dir) / args.project / (args.run_name or wandb.run.name or "run")
    ckpt_best = ckpt_dir / "best.pt"
    ckpt_last = ckpt_dir / "last.pt"

    if args.resume and os.path.isfile(args.resume):
        start_epoch = load_checkpoint(args.resume, model, optimizer, scaler)
        print(f"Resumed from epoch {start_epoch} at {args.resume}")

    print(f"Model: {args.model} | Trainable params: {count_trainable_params(model):,}")

    epochs_no_improve = 0

    # ----- Training loop -----
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        # Unfreeze backbone after N epochs if selected
        if args.freeze_backbone and epoch == args.unfreeze_epoch:
            for p in model.parameters():
                p.requires_grad = True
            print("Backbone unfrozen.")

        train_loss, train_acc, train_prec, train_rec, train_f1 = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler,
            log_images=(epoch == start_epoch), class_names=class_names
        )
        val_loss, val_acc, val_prec, val_rec, val_f1, val_cm, y_true_val, y_pred_val = evaluate(
            model, val_loader, criterion, device
        )

        scheduler.step()

        wandb.log({
            "epoch": epoch,
            "lr": scheduler.get_last_lr()[0],
            "train/loss": train_loss,
            "train/acc": train_acc,
            "train/precision_macro": train_prec,
            "train/recall_macro": train_rec,
            "train/f1_macro": train_f1,
            "val/loss": val_loss,
            "val/acc": val_acc,
            "val/precision_macro": val_prec,
            "val/recall_macro": val_rec,
            "val/f1_macro": val_f1,
            "time/epoch_s": time.time() - t0,
        })

        if epoch % args.log_cm_every == 0:
            cm_table = wandb.plot.confusion_matrix(
                probs=None,
                y_true=y_true_val,
                preds=y_pred_val,
                class_names=class_names
            )
            wandb.log({"val/confusion_matrix": cm_table})

        # Save checkpoints
        save_checkpoint({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "val_acc": val_acc,
            "class_names": class_names,
            "args": vars(args)}, ckpt_last)

        improved = val_acc > best_val_acc
        if improved:
            best_val_acc = val_acc
            save_checkpoint({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict() if scaler is not None else None,
                "val_acc": val_acc,
                "class_names": class_names,
                "args": vars(args),
            }, ckpt_best)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        print(f"[{epoch+1}/{args.epochs}] "
              f"train_acc={train_acc:.4f} val_acc={val_acc:.4f} "
              f"val_f1={val_f1:.4f} best_val_acc={best_val_acc:.4f} "
              f"no_improve={epochs_no_improve}")

        if epochs_no_improve >= args.patience:
            print("Early stopping triggered.")
            break

    # ---------- Final test evaluation ----------
    if ckpt_best.exists():
        load_checkpoint(str(ckpt_best), model)

    test_loss, test_acc, test_prec, test_rec, test_f1, test_cm, y_true_test, y_pred_test = evaluate(
        model, test_loader, criterion, device
    )

    wandb.log({
        "test/loss": test_loss,
        "test/acc": test_acc,
        "test/precision_macro": test_prec,
        "test/recall_macro": test_rec,
        "test/f1_macro": test_f1,
    })

    cm_table_test = wandb.plot.confusion_matrix(
        probs=None,
        y_true=y_true_test,
        preds=y_pred_test,
        class_names=class_names
    )
    wandb.log({"test/confusion_matrix": cm_table_test})

    # ---------- Save classification report ----------
    report = classification_report(y_true_test, y_pred_test, target_names=class_names, digits=4, zero_division=0)
    print("\n=== Test Classification Report ===\n", report)

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (ckpt_dir / "classification_report.txt").write_text(report, encoding="utf-8")
    wandb.save(str(ckpt_dir / "classification_report.txt"))

    # Save class-index mapping
    with open(ckpt_dir / "class_index.json", "w", encoding="utf-8") as f:
        json.dump({i: c for i, c in enumerate(class_names)}, f, ensure_ascii=False, indent=2)
    wandb.save(str(ckpt_dir / "class_index.json"))

    wandb.finish()


if __name__ == "__main__":
    main()
