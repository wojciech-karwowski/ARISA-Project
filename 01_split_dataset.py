
#!/usr/bin/env python3
"""
split_dataset.py — production-ready CLI to split a folder of class-subfolders into train/val/test.

Features
- Deterministic, stratified split by class (subfolder names are class labels)
- Validates split ratios (sum to 1.0 ± tolerance)
- Supports copy | move | symlink | hardlink modes
- Filters by file extensions
- Optional dry-run
- Clear logging + summary table
- Works on large datasets without extra dependencies

Usage
------
Basic 70/15/15 split:

    python split_dataset.py \
        --input /path/to/dataset \
        --output /path/to/output \
        --train 0.70 --val 0.15 --test 0.15

Copy vs move vs links:

    python split_dataset.py ... --mode copy
    python split_dataset.py ... --mode move
    python split_dataset.py ... --mode symlink
    python split_dataset.py ... --mode hardlink

Control randomness and file types:

    python split_dataset.py ... --seed 42 --extensions .jpg .jpeg .png .bmp

Dry run (prints what would happen):

    python split_dataset.py ... --dry-run

By default, the script expects the input directory to contain subfolders, each representing a class.
"""
import argparse
import logging
import math
import os
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

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
        handlers=[logging.StreamHandler(sys.stdout)],
    )

# ----------------------- Data classes --------------------------------

@dataclass
class SplitCounts:
    train: int
    val: int
    test: int

@dataclass
class ClassSummary:
    cls: str
    total: int
    train: int
    val: int
    test: int

# ----------------------- Helpers -------------------------------------

def validate_ratios(train: float, val: float, test: float, tol: float = 1e-6) -> None:
    s = train + val + test
    if not math.isclose(s, 1.0, rel_tol=0, abs_tol=tol):
        raise ValueError(f"Split ratios must sum to 1.0 (got {s:.6f}). "
                         f"Provided --train {train}, --val {val}, --test {test}.")

def find_class_dirs(root: Path) -> List[Path]:
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Input directory not found or not a directory: {root}")
    classes = [p for p in sorted(root.iterdir()) if p.is_dir()]
    if not classes:
        raise ValueError(f"No class subdirectories found under: {root}")
    return classes

def list_files(folder: Path, extensions: Tuple[str, ...]) -> List[Path]:
    files = []
    for p in folder.rglob("*"):
        if p.is_file():
            if extensions:
                if p.suffix.lower() in extensions:
                    files.append(p)
            else:
                files.append(p)
    return sorted(files)

def compute_counts(n: int, train: float, val: float, test: float) -> SplitCounts:
    # Round with deterministic rule: train gets floor, then val, test is the remainder
    n_train = int(n * train)
    n_val = int(n * val)
    n_test = n - n_train - n_val
    return SplitCounts(n_train, n_val, n_test)

def ensure_dir(path: Path, dry_run: bool) -> None:
    if dry_run:
        logging.debug(f"[dry-run] Would create directory: {path}")
        return
    path.mkdir(parents=True, exist_ok=True)

def transfer(src: Path, dst: Path, mode: str, dry_run: bool) -> None:
    if dry_run:
        logging.debug(f"[dry-run] Would {mode} -> {dst}")
        return
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "move":
        shutil.move(str(src), str(dst))
    elif mode == "symlink":
        # Relative symlink for portability
        rel = os.path.relpath(src, start=dst.parent)
        if dst.exists():
            dst.unlink()
        dst.symlink_to(rel)
    elif mode == "hardlink":
        if dst.exists():
            dst.unlink()
        os.link(src, dst)
    else:
        raise ValueError(f"Unsupported mode: {mode}")

def human_count(n: int) -> str:
    return f"{n:,}".replace(",", " ")

# ----------------------- Core logic ----------------------------------

def split_class(
    cls_dir: Path,
    out_root: Path,
    ratios: Tuple[float, float, float],
    mode: str,
    seed: int,
    exts: Tuple[str, ...],
    dry_run: bool,
) -> ClassSummary:
    class_name = cls_dir.name
    files = list_files(cls_dir, exts)
    rng = random.Random(seed + hash(class_name) % (2**32))
    rng.shuffle(files)

    counts = compute_counts(len(files), *ratios)
    logging.info(f"Class '{class_name}': total={human_count(len(files))} "
                 f"-> train={counts.train}, val={counts.val}, test={counts.test}")

    # Prepare output dirs
    for split_name in ("train", "val", "test"):
        ensure_dir(out_root / split_name / class_name, dry_run)

    # Do transfers
    idx_train_end = counts.train
    idx_val_end = counts.train + counts.val

    splits = [
        ("train", files[:idx_train_end]),
        ("val",   files[idx_train_end:idx_val_end]),
        ("test",  files[idx_val_end:]),
    ]

    for split_name, split_files in splits:
        target_dir = out_root / split_name / class_name
        for src in split_files:
            dst = target_dir / src.name
            transfer(src, dst, mode, dry_run)

    return ClassSummary(
        cls=class_name,
        total=len(files),
        train=counts.train,
        val=counts.val,
        test=counts.test,
    )

def print_summary_table(summaries: List[ClassSummary]) -> None:
    # Calculate column widths
    name_w = max(5, max(len(s.cls) for s in summaries))
    def w(field): return max(len(field), 7)
    cols = ["class", "total", "train", "val", "test"]
    widths = {
        "class": name_w,
        "total": w("total"),
        "train": w("train"),
        "val":   w("val"),
        "test":  w("test"),
    }

    sep = " | "
    header = sep.join([
        f"{'class':<{widths['class']}}",
        f"{'total':>{widths['total']}}",
        f"{'train':>{widths['train']}}",
        f"{'val':>{widths['val']}}",
        f"{'test':>{widths['test']}}",
    ])
    line = "-+-".join([
        "-" * widths['class'],
        "-" * widths['total'],
        "-" * widths['train'],
        "-" * widths['val'],
        "-" * widths['test'],
    ])
    print(header)
    print(line)
    total_total = total_train = total_val = total_test = 0
    for s in summaries:
        print(sep.join([
            f"{s.cls:<{widths['class']}}",
            f"{s.total:>{widths['total']}}",
            f"{s.train:>{widths['train']}}",
            f"{s.val:>{widths['val']}}",
            f"{s.test:>{widths['test']}}",
        ]))
        total_total += s.total
        total_train += s.train
        total_val += s.val
        total_test += s.test
    # Totals
    print(line)
    print(sep.join([
        f"{'TOTAL':<{widths['class']}}",
        f"{total_total:>{widths['total']}}",
        f"{total_train:>{widths['train']}}",
        f"{total_val:>{widths['val']}}",
        f"{total_test:>{widths['test']}}",
    ]))

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Split a dataset of class-subfolders into train/val/test.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", "-i", type=Path, required=True,
                   help="Path to INPUT dataset folder containing class subfolders.")
    p.add_argument("--output", "-o", type=Path, required=True,
                   help="Path to OUTPUT folder where train/val/test will be created.")
    p.add_argument("--train", type=float, default=0.70, help="Train ratio (0..1).")
    p.add_argument("--val", type=float, default=0.15, help="Val ratio (0..1).")
    p.add_argument("--test", type=float, default=0.15, help="Test ratio (0..1).")
    p.add_argument("--mode", choices=["copy", "move", "symlink", "hardlink"],
                   default="copy", help="How to place files into output splits.")
    p.add_argument("--extensions", "-e", nargs="*", default=[".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"],
                   help="File extensions to include (case-insensitive). Empty = all files.")
    p.add_argument("--seed", type=int, default=42, help="Random seed for shuffling within each class.")
    p.add_argument("--dry-run", action="store_true", help="Print actions without writing files.")
    p.add_argument("--verbose", "-v", action="count", default=0,
                   help="Increase verbosity (-v for INFO, -vv for DEBUG).")
    p.add_argument("--clean-output", action="store_true",
                   help="If set, the script will refuse to run if OUTPUT exists and is non-empty.")
    return p.parse_args()

def main() -> int:
    args = parse_args()
    setup_logging(args.verbose)

    # Validate ratios and paths
    try:
        validate_ratios(args.train, args.val, args.test)
    except ValueError as e:
        logging.error(str(e))
        return 2

    input_root: Path = args.input.resolve()
    output_root: Path = args.output.resolve()
    exts = tuple(e.lower() if e.startswith(".") else f".{e.lower()}" for e in (args.extensions or []))

    if output_root.exists():
        # If clean-output is set and output isn't empty, refuse to proceed to avoid mixing datasets.
        if args.clean_output:
            non_empty = any(output_root.iterdir())
            if non_empty:
                logging.error(f"Output directory exists and is not empty: {output_root}. "
                              f"Use a new path or empty it manually.")
                return 3
    else:
        if not args.dry_run:
            output_root.mkdir(parents=True, exist_ok=True)

    try:
        class_dirs = find_class_dirs(input_root)
    except (FileNotFoundError, ValueError) as e:
        logging.error(str(e))
        return 4

    logging.info(f"Found {len(class_dirs)} class folders under: {input_root}")
    summaries: List[ClassSummary] = []

    for cls_dir in class_dirs:
        s = split_class(
            cls_dir=cls_dir,
            out_root=output_root,
            ratios=(args.train, args.val, args.test),
            mode=args.mode,
            seed=args.seed,
            exts=exts,
            dry_run=args.dry_run,
        )
        summaries.append(s)

    print_summary_table(summaries)
    if args.dry_run:
        logging.info("Dry run completed. No files were written.")
    else:
        logging.info("Done.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
