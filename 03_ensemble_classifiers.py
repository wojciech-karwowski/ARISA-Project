
"""
ensemble_classifiers.py — production-grade CLI to build classifier ensembles from per-model prediction CSVs.

Core ideas
----------
- Input = prediction tables (CSV/Parquet) from individual classifiers.
  Each table must have an ID column (e.g., "image", "id", "filename") and either:
    a) per-class probabilities (sum to 1 per row), or
    b) logits (any real numbers; use --apply-softmax to convert)
  Optionally include the ground-truth column for evaluation.

- Outputs a CSV with ensemble probabilities and (if GT given) metrics.
- Supports multiple ensembling strategies:
    * soft: arithmetic mean of probabilities
    * weighted-soft: convex combination with user-provided weights
    * geometric: normalized geometric mean
    * rank-avg: average the per-class ranks (Borda-style)
    * hard: majority vote on argmax predictions (ties -> highest summed prob)
    * stacking: meta-learner (logistic regression) on concatenated probs/logits

- Optional per-model temperature scaling on a validation set to calibrate logits/probabilities.

- Robust alignment of rows by ID across files; will inner-join on IDs by default.

- Computes metrics if y_true is available (accuracy, balanced accuracy, macro precision/recall/F1,
  log loss). If scikit-learn is installed, adds ROC-AUC (OvR), confusion matrix.
  If not installed, these are skipped gracefully.

Usage examples
--------------
Soft voting over 4 models, inputs are probabilities with class columns C0..C6:

    python ensemble_classifiers.py \
      --inputs resnet50_val.csv regnety8gf_val.csv shufflenet_val.csv swint_val.csv \
      --id-col image --class-cols C0 C1 C2 C3 C4 C5 C6 \
      --strategy soft \
      --output ensemble_val.csv \
      --y-col label

Weighted soft voting with weights:

    python ensemble_classifiers.py \
      --inputs resnet50_val.csv regnety8gf_val.csv shufflenet_val.csv swint_val.csv \
      --weights 0.35 0.30 0.20 0.15 \
      --id-col image --class-cols C0 C1 C2 C3 C4 C5 C6 \
      --strategy weighted-soft \
      --output ensemble_val.csv \
      --y-col label

Stacking (logistic regression meta-learner) using probs as features:

    python ensemble_classifiers.py \
      --inputs resnet50_val.csv regnety8gf_val.csv shufflenet_val.csv swint_val.csv \
      --id-col image --class-cols C0 C1 C2 C3 C4 C5 C6 \
      --strategy stacking \
      --stacking-features probs \
      --cv 5 \
      --output ensemble_val.csv \
      --y-col label

Geometric mean with logit inputs + softmax + temperature scaling per model:

    python ensemble_classifiers.py \
      --inputs resnet50_logits.csv regnet_logits.csv shuff_logits.csv swint_logits.csv \
      --apply-softmax \
      --temp-scale \
      --id-col image --class-cols C0 C1 C2 C3 C4 C5 C6 \
      --strategy geometric \
      --output ensemble_val.csv \
      --y-col label

Notes
-----
- All inputs must share the same set of class columns (names must match).
- If your tables have different ID column names, use --rename-id to map them to a common name.
- Inner join on IDs ensures only overlapping samples are ensembled.
- For rank averaging, higher prob -> better rank (1 = best); ranks are averaged then inverted back to scores.

"""

import argparse
import logging
import sys
import math
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd

# Optional sklearn metrics
_SKLEARN = True
try:
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_recall_fscore_support, log_loss, roc_auc_score, confusion_matrix
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
except Exception:
    _SKLEARN = False

# -------------------------- Logging ---------------------------------

def setup_logging(verbosity: int) -> None:
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

# -------------------------- Utilities --------------------------------

def softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    x = logits - np.max(logits, axis=axis, keepdims=True)
    e = np.exp(x, dtype=np.float64)
    return e / np.sum(e, axis=axis, keepdims=True)

def normalize_probs(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    p = np.clip(p, eps, 1.0)
    p_sum = p.sum(axis=1, keepdims=True)
    p_sum = np.clip(p_sum, eps, None)
    return p / p_sum

def check_class_cols(df: pd.DataFrame, class_cols: List[str]) -> None:
    missing = [c for c in class_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing class columns: {missing}")

def temperature_scale(probs_or_logits: np.ndarray, y: np.ndarray, apply_softmax: bool, grid: Optional[List[float]] = None) -> float:
    """
    Fit a scalar temperature T >= 0.01 that minimizes NLL on (probs_or_logits, y).
    Uses simple grid search for robustness (no external deps). Returns T.
    """
    if grid is None:
        grid = [0.25, 0.33, 0.5, 0.67, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0]
    best_T = 1.0
    best_nll = float("inf")
    for T in grid:
        if apply_softmax:
            probs = softmax(probs_or_logits / T, axis=1)
        else:
            # if already probs, apply temperature via softmax inverse (logits ~ log p)
            logits = np.log(np.clip(probs_or_logits, 1e-12, 1.0))
            probs = softmax(logits / T, axis=1)
        # Negative log likelihood
        idx = np.arange(len(y))
        nll = -np.log(np.clip(probs[idx, y], 1e-12, 1.0)).mean()
        if nll < best_nll:
            best_nll = nll
            best_T = T
    return best_T

def rank_average(prob_list: List[np.ndarray]) -> np.ndarray:
    # Higher prob -> better rank (1). Use average rank across models, then invert to score and renormalize.
    ranks_sum = None
    for P in prob_list:
        # argsort gives ascending; we want descending ranks
        # compute ranks per row: highest prob gets rank 1
        order = np.argsort(-P, axis=1)
        ranks = np.empty_like(order, dtype=np.float64)
        # assign ranks 1..C per row
        n = np.arange(P.shape[0])[:, None]
        ranks[n, order] = np.arange(1, P.shape[1] + 1)[None, :]
        if ranks_sum is None:
            ranks_sum = ranks
        else:
            ranks_sum += ranks
    avg_rank = ranks_sum / len(prob_list)
    # Convert rank to score: lower rank -> higher score, use (C+1 - rank)
    C = avg_rank.shape[1]
    scores = (C + 1) - avg_rank
    return normalize_probs(scores)

def geometric_mean(prob_list: List[np.ndarray], eps: float = 1e-12) -> np.ndarray:
    log_sum = None
    for P in prob_list:
        P = np.clip(P, eps, 1.0)
        if log_sum is None:
            log_sum = np.log(P)
        else:
            log_sum += np.log(P)
    geom = np.exp(log_sum / len(prob_list))
    return normalize_probs(geom)

def weighted_soft(prob_list: List[np.ndarray], weights: List[float]) -> np.ndarray:
    w = np.array(weights, dtype=np.float64)
    if np.any(w < 0):
        raise ValueError("Weights must be non-negative.")
    if np.allclose(w.sum(), 0.0):
        raise ValueError("Weights sum to zero.")
    w = w / w.sum()
    out = np.zeros_like(prob_list[0], dtype=np.float64)
    for Wi, Pi in zip(w, prob_list):
        out += Wi * Pi
    return normalize_probs(out)

def hard_vote(prob_list: List[np.ndarray]) -> np.ndarray:
    # Majority vote on argmax; tie-breaker: sum of probabilities
    num_models = len(prob_list)
    C = prob_list[0].shape[1]
    votes = np.zeros((prob_list[0].shape[0], C), dtype=np.int32)
    sum_probs = np.zeros_like(prob_list[0], dtype=np.float64)
    for P in prob_list:
        preds = P.argmax(axis=1)
        votes[np.arange(len(preds)), preds] += 1
        sum_probs += P
    # For each row, find classes with max votes; tie-break by sum_probs
    winners = np.zeros(len(prob_list[0]), dtype=np.int32)
    for i in range(votes.shape[0]):
        max_votes = votes[i].max()
        candidates = np.where(votes[i] == max_votes)[0]
        if len(candidates) == 1:
            winners[i] = candidates[0]
        else:
            # choose candidate with highest summed prob
            winners[i] = candidates[np.argmax(sum_probs[i, candidates])]
    # Return one-hot probs
    out = np.zeros_like(prob_list[0], dtype=np.float64)
    out[np.arange(len(winners)), winners] = 1.0
    return out

def optimize_weights(prob_list: List[np.ndarray], y: np.ndarray, steps: int = 300, seed: int = 42) -> np.ndarray:
    """
    Simple projected coordinate ascent on the simplex to maximize accuracy on y.
    Initializes uniformly; iteratively perturbs weights and keeps improvements.
    """
    rng = np.random.default_rng(seed)
    M = len(prob_list)
    w = np.ones(M, dtype=np.float64) / M

    def acc_for(wvec: np.ndarray) -> float:
        P = weighted_soft(prob_list, wvec)
        preds = P.argmax(axis=1)
        return (preds == y).mean()

    best_acc = acc_for(w)
    for _ in range(steps):
        j = rng.integers(0, M)
        delta = rng.normal(scale=0.1)
        w_new = w.copy()
        w_new[j] = max(0.0, w_new[j] + delta)
        # renormalize to simplex
        s = w_new.sum()
        if s <= 0: 
            continue
        w_new /= s
        acc = acc_for(w_new)
        if acc >= best_acc:
            w, best_acc = w_new, acc
    return w

# -------------------------- IO & alignment ---------------------------

def read_table(path: Path, id_col: str, class_cols: List[str]) -> pd.DataFrame:
    if path.suffix.lower() in (".parquet", ".pq"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    if id_col not in df.columns:
        raise ValueError(f"{path}: missing id column '{id_col}'")
    check_class_cols(df, class_cols)
    return df[[id_col] + class_cols + [c for c in df.columns if c not in ([id_col] + class_cols)]]

def align_on_id(dfs: List[pd.DataFrame], id_col: str) -> List[pd.DataFrame]:
    # Inner join on common IDs
    ids = set(dfs[0][id_col])
    for df in dfs[1:]:
        ids &= set(df[id_col])
    if not ids:
        raise ValueError("No overlapping IDs across inputs.")
    ids = sorted(list(ids))
    aligned = [df.set_index(id_col).loc[ids].reset_index() for df in dfs]
    return aligned

# ----------------------------- Main ----------------------------------

def main():
    ap = argparse.ArgumentParser(description="Build classifier ensembles from per-model prediction tables.",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--inputs", "-i", nargs="+", required=True, help="Paths to per-model prediction CSV/Parquet files.")
    ap.add_argument("--id-col", required=True, help="Common ID column name present in all inputs (e.g., image).")
    ap.add_argument("--class-cols", nargs="+", required=True, help="Ordered list of class probability/logit columns.")
    ap.add_argument("--y-col", help="Ground-truth column name (optional; enables metrics).")
    ap.add_argument("--strategy", choices=["soft","weighted-soft","geometric","rank-avg","hard","stacking"], default="soft")
    ap.add_argument("--weights", nargs="*", type=float, help="Weights for weighted-soft (same order as --inputs).")
    ap.add_argument("--apply-softmax", action="store_true", help="Apply softmax to class columns (logit inputs).")
    ap.add_argument("--temp-scale", action="store_true", help="Apply per-model temperature scaling (requires y-col).")
    ap.add_argument("--stacking-features", choices=["probs","logits","probs+argmax"], default="probs",
                    help="Features for stacking (only if --strategy stacking).")
    ap.add_argument("--cv", type=int, default=5, help="CV folds for stacking meta-learner.")
    ap.add_argument("--optimize-weights", action="store_true", help="Optimize weights on validation set (requires y-col).")
    ap.add_argument("--output", "-o", required=True, help="Output CSV path for ensemble predictions.")
    ap.add_argument("--verbose", "-v", action="count", default=0, help="Increase verbosity.")
    args = ap.parse_args()

    setup_logging(args.verbose)

    class_cols = args.class_cols
    id_col = args.id_col

    # Load all tables
    paths = [Path(p) for p in args.inputs]
    dfs = [read_table(p, id_col, class_cols) for p in paths]
    dfs = align_on_id(dfs, id_col)

    # Extract ground truth if available from the first table (or any, but aligned)
    y = None
    if args.y_col:
        if args.y_col not in dfs[0].columns:
            logging.warning(f"y-col '{args.y_col}' not found in first table; searching others...")
            found = False
            for d in dfs:
                if args.y_col in d.columns:
                    y = d[args.y_col].to_numpy()
                    found = True
                    break
            if not found:
                logging.error(f"y-col '{args.y_col}' not found in any input table.")
                sys.exit(2)
        else:
            y = dfs[0][args.y_col].to_numpy()

        # Map y to 0..C-1 if it's not already numeric
        if not np.issubdtype(np.array(y).dtype, np.number):
            # assume labels equal to class column names; build mapping
            label_to_idx = {c: i for i, c in enumerate(class_cols)}
            try:
                y = np.array([label_to_idx[str(lbl)] for lbl in y])
            except Exception:
                logging.error("Non-numeric y provided; expected labels to match class column names.")
                sys.exit(3)
        y = y.astype(int)

    # Build per-model probability arrays (apply softmax if requested)
    prob_list = []
    raw_list = []
    for d in dfs:
        X = d[class_cols].to_numpy(dtype=np.float64)
        raw_list.append(X.copy())
        if args.apply_softmax:
            P = softmax(X, axis=1)
        else:
            # assume already probabilities, normalize defensively
            P = normalize_probs(X)
        prob_list.append(P)

    # Optional temperature scaling per model (needs y)
    if args.temp_scale:
        if y is None:
            logging.error("--temp-scale requires --y-col to fit calibration.")
            sys.exit(4)
        scaled = []
        for X, P in zip(raw_list, prob_list):
            T = temperature_scale(X if args.apply_softmax else P, y, apply_softmax=args.apply_softmax)
            logging.info(f"Fitted temperature T={T:.3f}")
            if args.apply_softmax:
                P_cal = softmax(X / T, axis=1)
            else:
                logits = np.log(np.clip(P, 1e-12, 1.0))
                P_cal = softmax(logits / T, axis=1)
            scaled.append(P_cal)
        prob_list = scaled

    # Optionally optimize weights
    weights = None
    if args.strategy == "weighted-soft":
        if args.weights is None and not args.optimize_weights:
            logging.error("For 'weighted-soft', provide --weights or enable --optimize-weights (requires --y-col).")
            sys.exit(5)
        if args.weights is not None:
            if len(args.weights) != len(prob_list):
                logging.error("Number of --weights must equal number of --inputs.")
                sys.exit(6)
            weights = args.weights
        elif args.optimize_weights:
            if y is None:
                logging.error("--optimize-weights requires --y-col.")
                sys.exit(7)
            weights = optimize_weights(prob_list, y)
            logging.info("Optimized weights: " + " ".join(f"{w:.4f}" for w in weights))

    # Compute ensemble probabilities
    if args.strategy == "soft":
        ensemble = normalize_probs(sum(prob_list) / len(prob_list))
    elif args.strategy == "weighted-soft":
        ensemble = weighted_soft(prob_list, weights)
    elif args.strategy == "geometric":
        ensemble = geometric_mean(prob_list)
    elif args.strategy == "rank-avg":
        ensemble = rank_average(prob_list)
    elif args.strategy == "hard":
        ensemble = hard_vote(prob_list)
    elif args.strategy == "stacking":
        if not _SKLEARN:
            logging.error("Stacking requires scikit-learn. Please install scikit-learn.")
            sys.exit(8)
        if y is None:
            logging.error("Stacking requires --y-col for training the meta-learner.")
            sys.exit(9)
        # Build features
        feats = []
        for i, (Xraw, P) in enumerate(zip(raw_list, prob_list)):
            if args.stacking-features == "probs":
                feats.append(P)
            elif args.stacking-features == "logits":
                feats.append(Xraw)
            else:  # probs+argmax
                argm = P.argmax(axis=1)[:, None]
                feats.append(np.concatenate([P, argm], axis=1))
        Xfeat = np.concatenate(feats, axis=1)

        # CV train Logistic Regression (multi-class OvR with liblinear or lbfgs)
        skf = StratifiedKFold(n_splits=max(2, args.cv), shuffle=True, random_state=42)
        oof = np.zeros_like(prob_list[0])
        for tr, va in skf.split(Xfeat, y):
            clf = LogisticRegression(max_iter=200, multi_class='auto', n_jobs=None)
            clf.fit(Xfeat[tr], y[tr])
            oof[va] = clf.predict_proba(Xfeat[va])
        ensemble = normalize_probs(oof)
    else:
        logging.error(f"Unknown strategy: {args.strategy}")
        sys.exit(10)

    # Build output DataFrame
    out = pd.DataFrame({id_col: dfs[0][id_col]})
    for c, col in enumerate(class_cols):
        out[col] = ensemble[:, c]
    out["pred"] = ensemble.argmax(axis=1)

    # Metrics if y is available
    if y is not None:
        preds = ensemble.argmax(axis=1)
        acc = (preds == y).mean()
        # confusion, precision/recall/f1, log_loss, balanced acc
        try:
            from sklearn.metrics import precision_recall_fscore_support, balanced_accuracy_score, log_loss, roc_auc_score, confusion_matrix
            pr, rc, f1, _ = precision_recall_fscore_support(y, preds, average="macro", zero_division=0)
            bacc = balanced_accuracy_score(y, preds)
            ll = log_loss(y, ensemble, labels=list(range(ensemble.shape[1])))
            out.attrs["metrics"] = {
                "accuracy": float(acc),
                "balanced_accuracy": float(bacc),
                "precision_macro": float(pr),
                "recall_macro": float(rc),
                "f1_macro": float(f1),
            }
            try:
                roc = roc_auc_score(y, ensemble, multi_class="ovr")
                out.attrs["metrics"]["roc_auc_macro_ovr"] = float(roc)
            except Exception:
                pass
            # Print metrics nicely
            print("Metrics:")
            print(f"  accuracy           : {acc:.4f}")
            print(f"  balanced_accuracy  : {bacc:.4f}")
            print(f"  precision_macro    : {pr:.4f}")
            print(f"  recall_macro       : {rc:.4f}")
            print(f"  f1_macro           : {f1:.4f}")
            try:
                print(f"  log_loss           : {ll:.4f}")
            except Exception:
                pass
            try:
                print(f"  roc_auc_macro_ovr  : {out.attrs['metrics']['roc_auc_macro_ovr']:.4f}")
            except Exception:
                pass
            try:
                cm = confusion_matrix(y, preds)
                print("Confusion matrix (rows=true, cols=pred):")
                # print as plain table
                for row in cm:
                    print("  " + " ".join(f"{int(x):4d}" for x in row))
            except Exception:
                pass
        except Exception as e:
            print(f"Basic accuracy: {acc:.4f} (install scikit-learn to see more metrics)")

    # Save
    out_path = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()
