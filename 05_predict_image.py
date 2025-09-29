#!/usr/bin/env python3
"""
predict_image.py — single-image inference: per-model Top-K and ensemble Top-K.

Requirements:
    pip install torch torchvision timm numpy pillow

Examples:
    # Raw .pth (state_dict) — you must specify the architecture and classes (CSV)
    python predict_image.py --image path/to/img.jpg \
        --models_dir ./models_pth --select "*.pth" \
        --model_fallback resnet50 \
        --classes "classA,classB,classC" \
        --ensemble prob_avg

    # .pt checkpoints saved by train.py (contain class_names)
    python predict_image.py --image path/to/img.jpg \
        --models_dir ./ckpts --select "best.pt" \
        --ensemble majority --k 3

    # Mixed .pt + .pth, classes loaded from JSON (index->name)
    python predict_image.py --image path/to/img.jpg \
        --models_dir ./all --select "*.pt,*.pth" \
        --model_fallback convnext_tiny \
        --class_index_json ./class_index.json \
        --ensemble logits_avg --weights 0.5,0.3,0.2
"""

import argparse
import glob
import json
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms, models

# Optional timm backbones (ConvNeXt, ViT, Swin, RegNet, etc.)
try:
    import timm
    TIMM_AVAILABLE = True
except Exception:
    TIMM_AVAILABLE = False

import torch.nn.functional as F


# ----------------------- Torchvision model builders -----------------------
TV_MODELS = {
    "resnet50": lambda num_classes: _tv_resnet50(num_classes),
    "densenet121": lambda num_classes: _tv_densenet121(num_classes),
    "efficientnet_b1": lambda num_classes: _tv_efficientnet_b1(num_classes),
    "mobilenet_v3_large": lambda num_classes: _tv_mobilenet_v3_large(num_classes),
    "shufflenet_v2_x1_0": lambda num_classes: _tv_shufflenet_v2(num_classes),
    "vgg16": lambda num_classes: _tv_vgg16(num_classes),
}

def _tv_resnet50(num_classes):
    m = models.resnet50(weights=None)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m

def _tv_densenet121(num_classes):
    m = models.densenet121(weights=None)
    m.classifier = nn.Linear(m.classifier.in_features, num_classes)
    return m

def _tv_mobilenet_v3_large(num_classes):
    m = models.mobilenet_v3_large(weights=None)
    m.classifier[3] = nn.Linear(m.classifier[3].in_features, num_classes)
    return m

def _tv_shufflenet_v2(num_classes):
    m = models.shufflenet_v2_x1_0(weights=None)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m

def _tv_efficientnet_b1(num_classes):
    m = models.efficientnet_b1(weights=None)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    return m

def _tv_vgg16(num_classes):
    m = models.vgg16(weights=None)
    m.classifier[6] = nn.Linear(m.classifier[6].in_features, num_classes)
    return m

def build_model(model_name: str, num_classes: int):
    """
    Create a model (torchvision or timm) with the specified number of classes.
    """
    name = model_name.lower()
    if name in TV_MODELS:
        return TV_MODELS[name](num_classes)
    if not TIMM_AVAILABLE:
        raise ValueError(f"Model '{model_name}' requires 'timm' (pip install timm).")
    timm_map = {
        "convnext_tiny": "convnext_tiny.fb_in22k",
        "vit_b": "vit_base_patch16_224.augreg_in21k",
        "swin_t": "swin_tiny_patch4_window7_224.ms_in22k",
        "regnety_8gf": "regnety_008.pycls_in1k",
    }
    if name not in timm_map and name in timm.list_models(pretrained=False):
        backbone = name
    else:
        backbone = timm_map.get(name, None)
    if backbone is None:
        raise ValueError(f"Unknown model '{model_name}'.")
    return timm.create_model(backbone, pretrained=False, num_classes=num_classes)


# ----------------------- Preprocessing -----------------------
def build_transform(img_size: int):
    """
    Standard eval preprocessing aligned with common ImageNet-normalized models.
    """
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    return transforms.Compose([
        transforms.Resize(int(img_size * 1.15)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

def load_image_tensor(path: str, img_size: int, device: torch.device) -> torch.Tensor:
    """
    Load an image file and return a normalized tensor [1, C, H, W].
    """
    img = Image.open(path).convert("RGB")
    tf = build_transform(img_size)
    t = tf(img).unsqueeze(0).to(device)
    return t


# ----------------------- Checkpoint loading -----------------------
def try_read_trainpy_ckpt(obj) -> Optional[dict]:
    """
    Detect if 'obj' looks like a train.py checkpoint (dict with 'model' state dict).
    """
    if isinstance(obj, dict) and "model" in obj and isinstance(obj["model"], dict):
        return obj
    return None

def load_checkpoint_generic(path: str, device: torch.device, num_classes: int,
                            model_fallback: Optional[str]) -> Tuple[nn.Module, List[str], float, int, Dict]:
    """
    Load either:
      - train.py checkpoint (dict with 'model','args','class_names'), or
      - raw .pth state_dict (requires --model_fallback).
    Returns: (model, class_names, val_acc, img_size, args_dict)
    """
    obj = torch.load(path, map_location=device)

    # Case 1: our train.py-style checkpoint
    ckpt = try_read_trainpy_ckpt(obj)
    if ckpt is not None:
        args = ckpt.get("args", {})
        model_name = args.get("model", model_fallback)
        if model_name is None:
            raise ValueError(f"{path}: checkpoint lacks args['model']; provide --model_fallback.")
        class_names = ckpt.get("class_names", [])
        m = build_model(model_name, num_classes=num_classes).to(device)
        m.load_state_dict(ckpt["model"])
        m.eval()
        val_acc = float(ckpt.get("val_acc", float("nan")))
        img_size = int(args.get("img_size", 224))
        return m, class_names, val_acc, img_size, args

    # Case 2: raw .pth state_dict
    if model_fallback is None:
        raise ValueError(f"{path}: appears to be a raw .pth state_dict; provide --model_fallback.")
    m = build_model(model_fallback, num_classes=num_classes).to(device)
    try:
        m.load_state_dict(obj, strict=True)
    except Exception as e:
        # If the head differs, attempt a non-strict load (common when only backbone is saved)
        print(f"[WARN] {path}: strict load_state_dict failed ({e}); retrying with strict=False")
        m.load_state_dict(obj, strict=False)
    m.eval()
    return m, [], float("nan"), 224, {"model": model_fallback, "source": "raw_pth"}


# ----------------------- Inference helpers -----------------------
@torch.no_grad()
def predict_logits(model: nn.Module, input_tensor: torch.Tensor) -> np.ndarray:
    """
    Forward pass for a single image tensor. Returns logits as a 1D NumPy array [C].
    """
    with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
        logits = model(input_tensor)  # [1, C]
    return logits.detach().cpu().numpy()[0]

def softmax_np(x: np.ndarray) -> np.ndarray:
    """
    Numerically stable softmax for a 1D array.
    """
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()

def topk_from_probs(probs: np.ndarray, class_names: List[str], k: int) -> List[Tuple[str, float, int]]:
    """
    Return a list of (class_name, probability, class_index) for the top-k classes.
    """
    idx = np.argsort(probs)[::-1][:k]
    return [(class_names[i], float(probs[i]), int(i)) for i in idx]

def majority_vote(single_preds: List[int], single_probs: List[np.ndarray]) -> int:
    """
    Majority voting with a tie-breaker using the highest mean probability across tied classes.
    """
    values, counts = np.unique(single_preds, return_counts=True)
    top = counts.max()
    tied = values[counts == top]
    if len(tied) == 1:
        return int(tied[0])
    avg_conf = []
    for cls in tied:
        avg_conf.append(np.mean([p[cls] for p in single_probs]))
    return int(tied[int(np.argmax(avg_conf))])


# ----------------------- Main CLI -----------------------
def main():
    ap = argparse.ArgumentParser(description="Single-image Top-K predictions per-model and via ensemble")
    ap.add_argument("--image", required=True, type=str, help="Path to the image (jpg/png/tif...)")
    ap.add_argument("--models_dir", required=True, type=str, help="Folder containing .pt and/or .pth files")
    ap.add_argument("--select", type=str, default="*.pt,*.pth",
                    help="Glob patterns, e.g., '*.pt,*.pth' or 'best.pt'")
    ap.add_argument("--model_fallback", type=str, default=None,
                    help="Architecture for raw .pth (e.g., resnet50, convnext_tiny, vit_b, swin_t, regnety_8gf)")
    ap.add_argument("--classes", type=str, default=None,
                    help="Comma-separated class list, e.g., 'cat,dog,bird' (used if checkpoints lack class_names)")
    ap.add_argument("--class_index_json", type=str, default=None,
                    help="JSON mapping index->name (e.g., produced by train.py)")
    ap.add_argument("--img_size", type=int, default=224, help="Model input size used at training time")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--k", type=int, default=3, help="Top-K to display")
    ap.add_argument("--ensemble", choices=["none", "prob_avg", "logits_avg", "majority"], default="prob_avg")
    ap.add_argument("--weights", type=str, default=None,
                    help="Weights for prob_avg/logits_avg, e.g., '0.5,0.3,0.2'")
    ap.add_argument("--save_json", type=str, default=None, help="Path to save the result JSON")
    args = ap.parse_args()

    device = torch.device(args.device)

    # 1) Build class list:
    #    Priority: --class_index_json → --classes → class_names from first .pt checkpoint
    classes: List[str] = []
    if args.class_index_json:
        mp = json.loads(Path(args.class_index_json).read_text(encoding="utf-8"))
        # Accept keys as str or int; enforce index order
        classes = [mp[str(i)] if str(i) in mp else mp[i] for i in sorted([int(k) for k in mp.keys()], key=int)]
    elif args.classes:
        classes = [c.strip() for c in args.classes.split(",") if c.strip()]

    # 2) Collect model files
    patterns = [s.strip() for s in args.select.split(",")]
    ckpt_paths = []
    for pat in patterns:
        ckpt_paths.extend(glob.glob(str(Path(args.models_dir) / pat)))
    ckpt_paths = sorted(set(ckpt_paths))
    if not ckpt_paths:
        raise FileNotFoundError(f"No model files matching '{args.select}' in {args.models_dir}")

    # 3) Load models (accept both .pt with metadata and raw .pth)
    models_info = []
    for p in ckpt_paths:
        # If we don't yet know the number of classes, temporarily build with 1 class
        num_classes = len(classes) if classes else None
        try:
            tmp_num = num_classes if num_classes is not None else 1
            model, cls_names, val_acc, img_size_ckpt, args_ckpt = load_checkpoint_generic(
                p, device, num_classes=tmp_num, model_fallback=args.model_fallback
            )
            # If the checkpoint provides class_names and we didn't have them yet, adopt them
            if cls_names and not classes:
                classes = cls_names
            # If we initialized with 1 class but now know the correct count, reload the model
            if tmp_num != len(classes):
                model, cls_names, val_acc, img_size_ckpt, args_ckpt = load_checkpoint_generic(
                    p, device, num_classes=len(classes), model_fallback=args.model_fallback
                )
            models_info.append({
                "path": p, "model": model, "val_acc": val_acc,
                "img_size": img_size_ckpt, "args": args_ckpt
            })
        except Exception as e:
            print(f"[WARN] Skipping {p}: {e}")

    if not models_info:
        raise RuntimeError("Failed to load any model.")
    if not classes:
        raise ValueError("Could not determine the class list. Provide --classes or --class_index_json, "
                         "or use .pt checkpoints that include class_names.")

    # 4) Preprocess the image (use the maximum img_size among models; safe for smaller backbones)
    max_img_size = max([m["img_size"] for m in models_info] + [args.img_size])
    inp = load_image_tensor(args.image, max_img_size, device)

    # 5) Per-model predictions
    per_model = []
    for m in models_info:
        logits = predict_logits(m["model"], inp)
        probs = softmax_np(logits)
        topk = topk_from_probs(probs, classes, args.k)
        per_model.append({"path": m["path"], "val_acc": float(m["val_acc"]), "topk": topk, "probs": probs})

    # 6) Ensemble prediction (optional)
    ensemble_out = None
    if args.ensemble != "none":
        weights = None
        if args.weights:
            weights = np.array([float(x) for x in args.weights.split(",")], dtype=np.float64)
            if len(weights) != len(per_model):
                raise ValueError(f"--weights length ({len(weights)}) != number of models ({len(per_model)})")
            weights = weights / weights.sum()

        if args.ensemble == "prob_avg":
            P = np.stack([pm["probs"] for pm in per_model], axis=0)  # [M, C]
            probs_ens = (weights[:, None] * P).sum(axis=0) if weights is not None else P.mean(axis=0)
            topk = topk_from_probs(probs_ens, classes, args.k)
            ensemble_out = {"strategy": "prob_avg", "topk": topk, "probs": probs_ens.tolist()}

        elif args.ensemble == "logits_avg":
            # We do not have raw logits here after softmax_np; approximate via log-probs for combination
            L = np.stack([np.log(pm["probs"] + 1e-12) for pm in per_model], axis=0)  # [M, C]
            logits_ens = (weights[:, None] * L).sum(axis=0) if weights is not None else L.mean(axis=0)
            probs_ens = np.exp(logits_ens - logits_ens.max())
            probs_ens = probs_ens / probs_ens.sum()
            topk = topk_from_probs(probs_ens, classes, args.k)
            ensemble_out = {"strategy": "logits_avg", "topk": topk, "probs": probs_ens.tolist()}

        elif args.ensemble == "majority":
            preds = [int(np.argmax(pm["probs"])) for pm in per_model]
            probs_list = [pm["probs"] for pm in per_model]
            cls_idx = majority_vote(preds, probs_list)
            probs_mean = np.mean(np.stack(probs_list, axis=0), axis=0)
            topk = topk_from_probs(probs_mean, classes, args.k)
            ensemble_out = {"strategy": "majority", "topk": topk, "probs": probs_mean.tolist()}

    # 7) Display results
    print(f"\nImage: {args.image}")
    print(f"Classes ({len(classes)}): {classes}\n")
    for i, pm in enumerate(per_model):
        print(f"[MODEL {i}] {Path(pm['path']).name}  (val_acc={pm['val_acc']:.4f})")
        for rank, (name, prob, idx) in enumerate(pm["topk"], start=1):
            print(f"   {rank}. {name:20s}  p={prob:7.4f}  (idx={idx})")
        print()

    if ensemble_out:
        print(f"[ENSEMBLE:{ensemble_out['strategy']}] Top-{args.k}")
        for rank, (name, prob, idx) in enumerate(ensemble_out["topk"], start=1):
            print(f"   {rank}. {name:20s}  p={prob:7.4f}  (idx={idx})")
        print()

    # 8) Optional JSON dump
    if args.save_json:
        payload = {
            "image": args.image,
            "classes": classes,
            "per_model": [
                {
                    "path": pm["path"],
                    "val_acc": pm["val_acc"],
                    "topk": [{"rank": r+1, "class": name, "prob": prob, "index": idx}
                             for r, (name, prob, idx) in enumerate(pm["topk"])]
                } for pm in per_model
            ],
            "ensemble": ensemble_out
        }
        Path(args.save_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.save_json).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Saved JSON: {args.save_json}")


if __name__ == "__main__":
    main()
