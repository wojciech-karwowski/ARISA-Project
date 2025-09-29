"""
evaluate.py — CLI for evaluating single models and ensembles on the test split.

Requirements:
    pip install torch torchvision timm scikit-learn pandas wandb

Expected data structure:
    data_dir/
      test/<class1|class2|...>/*.jpg

Checkpoints:
    Folder with *.pt saved by train.py (contain keys: 'model', 'args', 'class_names', 'val_acc', ...)

Examples:
    # Evaluate each model separately
    python evaluate.py --data_dir /path/to/data --models_dir ./outputs/image-classification/runA

    # Soft-probability averaging ensemble over all checkpoints
    python evaluate.py --data_dir /path/to/data --models_dir ./outputs/image-classification/runA --ensemble prob_avg

    # Majority vote with confidence tie-breaker, only top-3 best ckpts (by val_acc stored in ckpt)
    python evaluate.py --data_dir /path/to/data --models_dir ./ckpts --ensemble majority --topk 3

    # Weighted probability averaging (weights match the order printed or --select pattern order)
    python evaluate.py --data_dir /path/to/data --models_dir ./ckpts --ensemble prob_avg --weights 0.5,0.3,0.2

    # Log to Weights & Biases (optional)
    python evaluate.py --data_dir /path/to/data --models_dir ./ckpts --ensemble prob_avg --wandb_project evals --wandb_run_name my_ensemble
"""

import argparse
import os
import json
import glob
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

try:
    import timm
    TIMM_AVAILABLE = True
except Exception:
    TIMM_AVAILABLE = False

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# ----------------------- Model builders  -----------------------
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

def build_model_for_eval(model_name: str, num_classes: int):
    name = model_name.lower()
    if name in TV_MODELS:
        return TV_MODELS[name](num_classes)

    if not TIMM_AVAILABLE:
        raise ValueError(f"Model '{model_name}' requires timm (pip install timm).")

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
    m = timm.create_model(backbone, pretrained=False, num_classes=num_classes)
    return m

# ----------------------- Data -----------------------
def build_test_loader(data_dir: str, img_size: int, batch_size: int, workers: int):
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    eval_tf = transforms.Compose([
        transforms.Resize(int(img_size*1.15)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_ds = datasets.ImageFolder(os.path.join(data_dir, "test"), transform=eval_tf)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)
    return test_loader, test_ds.classes, test_ds.samples  # samples: list[(path, label)]

# ----------------------- Utils -----------------------
def softmax_np(logits: np.ndarray) -> np.ndarray:
    x = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)

def load_checkpoint_model(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    args = ckpt.get("args", {})
    class_names = ckpt.get("class_names", None)
    model_name = args.get("model", None)
    if class_names is None or model_name is None:
        raise ValueError(f"Checkpoint {ckpt_path} missing 'class_names' or 'args[model]'.")
    num_classes = len(class_names)
    model = build_model_for_eval(model_name, num_classes=num_classes).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    val_acc = float(ckpt.get("val_acc", np.nan))
    img_size = int(args.get("img_size", 224))
    return model, class_names, val_acc, img_size, args

@torch.no_grad()
def predict_logits(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    logits_all = []
    for imgs, _ in loader:
        imgs = imgs.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            logits = model(imgs)
        logits_all.append(logits.detach().cpu().numpy())
    return np.concatenate(logits_all, axis=0)

def majority_vote(preds_list: List[np.ndarray], probs_list: Optional[List[np.ndarray]] = None) -> np.ndarray:
    """Tie-breaker: use highest average probability across tied classes."""
    preds_arr = np.stack(preds_list, axis=1)  # [N, M]
    N, M = preds_arr.shape
    out = np.empty(N, dtype=np.int64)
    if probs_list is not None:
        probs_arr = np.stack(probs_list, axis=1)  # [N, M, C]
    for i in range(N):
        votes, counts = np.unique(preds_arr[i], return_counts=True)
        top_count = counts.max()
        tied = votes[counts == top_count]
        if len(tied) == 1 or probs_list is None:
            out[i] = tied[0] if len(tied) == 1 else votes[np.argmax(counts)]
        else:
            # average prob over models for tied classes
            avg_conf = []
            for cls in tied:
                avg_conf.append(np.mean([probs_list[m][i, cls] for m in range(M)]))
            out[i] = tied[int(np.argmax(avg_conf))]
    return out

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str]) -> Dict:
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    rep = classification_report(y_true, y_pred, target_names=class_names, digits=4, zero_division=0)
    return {"acc": acc, "precision_macro": prec, "recall_macro": rec, "f1_macro": f1, "confusion_matrix": cm, "report": rep}

# ----------------------- Main -----------------------
def main():
    ap = argparse.ArgumentParser(description="Evaluate image models & ensembles on test split")
    ap.add_argument("--data_dir", required=True, type=str, help="Path containing 'test/' subfolder")
    ap.add_argument("--models_dir", required=True, type=str, help="Path with *.pt checkpoints (train.py format)")
    ap.add_argument("--ensemble", choices=["none", "prob_avg", "logits_avg", "majority"], default="none",
                    help="Ensemble strategy across all loaded checkpoints")
    ap.add_argument("--weights", type=str, default=None, help="Comma-separated weights for models (for prob_avg/logits_avg)")
    ap.add_argument("--select", type=str, default="*.pt", help="Glob pattern to select checkpoints in models_dir")
    ap.add_argument("--topk", type=int, default=None, help="Use only top-K checkpoints by stored val_acc")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--img_size", type=int, default=224, help="Eval resize/crop size (should match training)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--output_dir", type=str, default="./eval_outputs")
    ap.add_argument("--wandb_project", type=str, default=None)
    ap.add_argument("--wandb_entity", type=str, default=None)
    ap.add_argument("--wandb_run_name", type=str, default=None)
    args = ap.parse_args()

    device = torch.device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load test data
    test_loader, class_names_ds, samples = build_test_loader(args.data_dir, args.img_size, args.batch_size, args.workers)
    y_true = np.array([lbl for _, lbl in samples], dtype=np.int64)
    paths = [p for p, _ in samples]

    # 2) Collect checkpoints
    ckpt_paths = sorted(glob.glob(str(Path(args.models_dir) / args.select)))
    if not ckpt_paths:
        raise FileNotFoundError(f"No checkpoints matching '{args.select}' in {args.models_dir}")
    models_info = []
    for p in ckpt_paths:
        try:
            model, class_names_ckpt, val_acc, img_size_ckpt, args_ckpt = load_checkpoint_model(p, device)
            models_info.append({
                "path": p, "model": model, "class_names": class_names_ckpt,
                "val_acc": val_acc, "img_size": img_size_ckpt, "args": args_ckpt
            })
        except Exception as e:
            print(f"[WARN] Skipping {p}: {e}")

    if not models_info:
        raise RuntimeError("No valid checkpoints loaded.")

    # Ensure consistent classes across checkpoints and dataset
    base_classes = models_info[0]["class_names"]
    if base_classes != class_names_ds:
        raise ValueError("Class order mismatch between dataset test split and checkpoint class_names. "
                         "Rebuild dataset/test or map indices before evaluation.")
    for info in models_info[1:]:
        if info["class_names"] != base_classes:
            raise ValueError(f"Class order mismatch across checkpoints: {info['path']}")

    # 3) Optionally pick top-K by stored val_acc
    if args.topk is not None and args.topk < len(models_info):
        models_info = sorted(models_info, key=lambda d: (-(d["val_acc"] if not np.isnan(d["val_acc"]) else -1e9)))[:args.topk]

    print("Loaded checkpoints:")
    for i, m in enumerate(models_info):
        print(f"  [{i}] acc_val={m['val_acc']:.4f} | img_size={m['img_size']} | path={m['path']}")

    # 4) Optional W&B
    use_wandb = args.wandb_project is not None
    if use_wandb:
        import wandb
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=args.wandb_run_name,
                   config={"ensemble": args.ensemble, "num_models": len(models_info),
                           "img_size": args.img_size, "batch_size": args.batch_size})
        wandb.config.update({"ckpts": [m["path"] for m in models_info]})

    # 5) Per-model predictions
    per_model_logits = []
    per_model_probs = []
    per_model_preds = []
    for m in models_info:
        logits = predict_logits(m["model"], test_loader, device)
        probs = softmax_np(logits)
        preds = probs.argmax(axis=1)
        per_model_logits.append(logits)
        per_model_probs.append(probs)
        per_model_preds.append(preds)

    # 6) Evaluate single models
    single_metrics = []
    for i, m in enumerate(models_info):
        met = compute_metrics(y_true, per_model_preds[i], base_classes)
        single_metrics.append({
            "path": m["path"],
            "val_acc_stored": float(m["val_acc"]),
            "test_acc": float(met["acc"]),
            "test_precision_macro": float(met["precision_macro"]),
            "test_recall_macro": float(met["recall_macro"]),
            "test_f1_macro": float(met["f1_macro"]),
        })
        print(f"[SINGLE] {m['path']}  acc={met['acc']:.4f} f1={met['f1_macro']:.4f}")
        if use_wandb:
            import wandb
            wandb.log({f"single/{Path(m['path']).stem}/acc": met["acc"],
                       f"single/{Path(m['path']).stem}/f1_macro": met["f1_macro"]})

    # 7) Ensemble
    ens_results = {}
    if args.ensemble != "none":
        weights = None
        if args.weights:
            weights = np.array([float(x) for x in args.weights.split(",")], dtype=np.float64)
            if len(weights) != len(models_info):
                raise ValueError(f"--weights length {len(weights)} != num_models {len(models_info)}")
            weights = weights / weights.sum()

        if args.ensemble == "prob_avg":
            probs_stack = np.stack(per_model_probs, axis=0)  # [M, N, C]
            if weights is None:
                probs_ens = probs_stack.mean(axis=0)
            else:
                probs_ens = np.tensordot(weights, probs_stack, axes=(0, 0))  # [N, C]
            preds_ens = probs_ens.argmax(axis=1)
            ens_results["strategy"] = "prob_avg"
            ens_results["preds"] = preds_ens
            ens_results["probs"] = probs_ens

        elif args.ensemble == "logits_avg":
            logits_stack = np.stack(per_model_logits, axis=0)  # [M, N, C]
            if weights is None:
                logits_ens = logits_stack.mean(axis=0)
            else:
                logits_ens = np.tensordot(weights, logits_stack, axes=(0, 0))
            probs_ens = softmax_np(logits_ens)
            preds_ens = probs_ens.argmax(axis=1)
            ens_results["strategy"] = "logits_avg"
            ens_results["preds"] = preds_ens
            ens_results["probs"] = probs_ens

        elif args.ensemble == "majority":
            preds_ens = majority_vote(per_model_preds, per_model_probs)
            ens_results["strategy"] = "majority"
            ens_results["preds"] = preds_ens
            # derive mean probs for reporting only
            probs_ens = np.mean(np.stack(per_model_probs, axis=0), axis=0)
            ens_results["probs"] = probs_ens

        # metrics
        met_ens = compute_metrics(y_true, ens_results["preds"], base_classes)
        print(f"[ENSEMBLE:{ens_results['strategy']}] acc={met_ens['acc']:.4f} f1={met_ens['f1_macro']:.4f}")
        if use_wandb:
            import wandb
            wandb.log({f"ensemble/{ens_results['strategy']}/acc": met_ens["acc"],
                       f"ensemble/{ens_results['strategy']}/f1_macro": met_ens["f1_macro"]})
        ens_results["metrics"] = {
            "acc": float(met_ens["acc"]),
            "precision_macro": float(met_ens["precision_macro"]),
            "recall_macro": float(met_ens["recall_macro"]),
            "f1_macro": float(met_ens["f1_macro"]),
            "report": met_ens["report"]
        }

    # 8) Save artifacts
    results_dir = out_dir / ("ensemble_" + (ens_results.get("strategy", "none")))
    results_dir.mkdir(parents=True, exist_ok=True)

    # per-image predictions table
    df = pd.DataFrame({"path": paths, "y_true": y_true, "y_true_name": [base_classes[i] for i in y_true]})
    for i, m in enumerate(models_info):
        df[f"pred_{i}"] = per_model_preds[i]
        df[f"pred_{i}_name"] = [base_classes[j] for j in per_model_preds[i]]
        df[f"maxprob_{i}"] = per_model_probs[i].max(axis=1)

    if args.ensemble != "none":
        df["ens_pred"] = ens_results["preds"]
        df["ens_pred_name"] = [base_classes[j] for j in ens_results["preds"]]
        df["ens_maxprob"] = ens_results["probs"].max(axis=1)

    csv_path = results_dir / "predictions.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8")

    # save summaries
    summary = {
        "dataset_test_size": len(y_true),
        "classes": base_classes,
        "single_models": single_metrics,
        "ensemble": ens_results if args.ensemble != "none" else None,
        "args": vars(args),
    }
    (results_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    # W&B upload (optional)
    if use_wandb:
        import wandb
        wandb.save(str(csv_path))
        wandb.save(str(results_dir / "summary.json"))
        wandb.finish()

    print(f"\nSaved:\n- {csv_path}\n- {results_dir / 'summary.json'}")
    if args.ensemble != "none":
        print(f"Ensemble metrics: acc={summary['ensemble']['metrics']['acc']:.4f}, "
              f"f1_macro={summary['ensemble']['metrics']['f1_macro']:.4f}")

if __name__ == "__main__":
    main()
