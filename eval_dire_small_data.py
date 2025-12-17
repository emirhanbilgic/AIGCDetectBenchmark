"""
Evaluate DIRE on a custom 3-folder setup and decide which fake set is harder to detect.

Expected directory layout (default):
  test_data/
    real_images/          # Real images (may contain subfolders)
      ├── celebahq/
      ├── cityscapes_real/
      └── ...
    fake_ours/            # Your fake images (may contain subfolders)
      ├── celebahq_openjourney_ours/
      ├── cityscapes_kandindsky_ours/
      └── ...
    fake_semi-truths/     # Semi-truths fake images (may contain subfolders)
      ├── celebahq_openjourney/
      ├── cityscapes_kandindsky/
      └── ...

The script recursively searches all subfolders for images (.jpg, .jpeg, .png, etc.).

This script runs two binary evaluations:
  1) real_images (label=0) vs fake_ours (label=1)
  2) real_images (label=0) vs fake_semi-truths (label=1)

"Harder to detect" is reported as the fake set with **lower ROC-AUC** (tie-break: lower AP).

IMPORTANT: If you get numpy/sklearn compatibility errors, run this FIRST in your environment:
    pip uninstall -y numpy scikit-learn
    pip install numpy==1.23.5 scikit-learn==1.2.2
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageFile

# Import sklearn metrics (may fail if numpy/sklearn versions incompatible)
try:
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        roc_auc_score,
    )
except ImportError as e:
    print("❌ sklearn import failed. This is likely due to numpy/sklearn version incompatibility.")
    print("🔧 Run this FIRST in your environment:")
    print("   pip uninstall -y numpy scikit-learn")
    print("   pip install numpy==1.23.5 scikit-learn==1.2.2")
    raise e

from torch.utils.data import DataLoader, Dataset

from data.datasets import custom_augment, process_img
from util import get_model

ImageFile.LOAD_TRUNCATED_IMAGES = True

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def _collect_images(root: str) -> List[str]:
    paths: List[str] = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            ext = os.path.splitext(fn)[1].lower()
            if ext in IMG_EXTS:
                paths.append(os.path.join(dirpath, fn))
    paths.sort()
    return paths


class _BinaryFolderDataset(Dataset):
    def __init__(self, items: Sequence[Tuple[str, int]], opt):
        self.items = items
        self.opt = opt

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.items[idx]
        img = Image.open(path).convert('RGB')
        img, _ = process_img(img, self.opt, path, label)
        return img, label


def _eval_pair(
    model,
    opt,
    real_paths: List[str],
    fake_paths: List[str],
    batch_size: int = 64,
) -> dict:
    """Evaluate one real-vs-fake pair."""
    items = [(p, 0) for p in real_paths] + [(p, 1) for p in fake_paths]
    ds = _BinaryFolderDataset(items, opt)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    y_true, y_pred = [], []

    @torch.no_grad()
    def run_inference():
        nonlocal y_true, y_pred
        for x, y in dl:
            x = x.to(opt.device)
            logits = model(x)
            probs = logits.sigmoid().flatten().tolist()
            y_pred.extend(probs)
            y_true.extend(y.flatten().tolist())

    run_inference()

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Calculate metrics
    acc = accuracy_score(y_true, y_pred > 0.5)
    ap = average_precision_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred)

    r_acc = accuracy_score(y_true[y_true == 0], y_pred[y_true == 0] > 0.5)
    f_acc = accuracy_score(y_true[y_true == 1], y_pred[y_true == 1] > 0.5)

    return {
        "accuracy": acc,
        "avg_precision": ap,
        "auc": auc,
        "real_accuracy": r_acc,
        "fake_accuracy": f_acc,
        "y_true": y_true,
        "y_pred": y_pred,
    }


def _load_dire(args):
    """Load DIRE model with preprocessing setup."""
    # Build a TestOptions-like object so preprocessing matches the repo.
    class Opt:
        pass
    opt = Opt()
    opt.detect_method = "DIRE"
    opt.isTrain = False  # For evaluation/testing
    opt.isVal = False    # For evaluation/testing
    opt.model_path = args.model_path
    opt.noise_type = args.noise_type
    opt.no_crop = bool(args.no_crop)
    opt.no_resize = bool(args.no_resize)
    opt.batch_size = args.batch_size

    # DIRE-specific settings
    opt.DIRE_modelpath = args.dire_model_path

    # Set default preprocessing options
    opt.rz_interp = ["bilinear"]
    opt.blur_sig = [1.0]
    opt.jpg_method = ["pil"]
    opt.jpg_qual = [95]
    opt.loadSize = 256
    opt.CropSize = 224

    # Set device
    if args.device:
        opt.device = torch.device(args.device)
    else:
        opt.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[eval] device={opt.device}, batch_size={args.batch_size}")

    model = get_model(opt)

    # DIRE models typically don't need explicit weight loading here
    # The preprocessing setup happens in get_processing_model
    model.to(opt.device)
    model.eval()

    return model, opt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--small_data_root",
        type=str,
        default="test_data",
        help="Path to the test_data folder.",
    )
    ap.add_argument("--real_dir", type=str, default="real_images")
    ap.add_argument("--fake_ours_dir", type=str, default="fake_ours")
    ap.add_argument("--fake_semi_dir", type=str, default="fake_semi-truths")
    ap.add_argument(
        "--model_path",
        type=str,
        default="./weights/classifier/CNNSpot.pth",
        help="Path to the DIRE-trained classifier checkpoint.",
    )
    ap.add_argument(
        "--dire_model_path",
        type=str,
        default="./weights/preprocessing/lsun_bedroom.pt",
        help="Path to the DIRE diffusion model.",
    )
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--device", type=str, default=None, help="cuda, cuda:0, or cpu. Default: auto")
    ap.add_argument(
        "--noise_type",
        type=str,
        default=None,
        choices=["jpg", "blur", "resize"],
        help="Optional test-time corruption. Omit for no corruption.",
    )
    ap.add_argument("--no_crop", action="store_true")
    ap.add_argument("--no_resize", action="store_true")
    args = ap.parse_args()

    real_root = os.path.join(args.small_data_root, args.real_dir)
    fake_ours_root = os.path.join(args.small_data_root, args.fake_ours_dir)
    fake_semi_root = os.path.join(args.small_data_root, args.fake_semi_dir)

    for p in [real_root, fake_ours_root, fake_semi_root]:
        if not os.path.isdir(p):
            raise FileNotFoundError(f"Missing folder: {p}")

    if not os.path.isfile(args.model_path):
        raise FileNotFoundError(f"Missing classifier checkpoint: {args.model_path}. "
                               f"For DIRE, download CNNSpot.pth from the Baidu Pan link in weights/classifier/download.txt")
    if not os.path.isfile(args.dire_model_path):
        raise FileNotFoundError(f"Missing DIRE diffusion model: {args.dire_model_path}. "
                               f"Download lsun_bedroom.pt from the Baidu Pan link in weights/preprocessing/download.txt")

    print(f"[data] Loading images from {args.small_data_root}")
    real_paths = _collect_images(real_root)
    fake_ours_paths = _collect_images(fake_ours_root)
    fake_semi_paths = _collect_images(fake_semi_root)

    print(f"[data] {args.real_dir}: {len(real_paths)}")
    print(f"[data] {args.fake_ours_dir}: {len(fake_ours_paths)}")
    print(f"[data] {args.fake_semi_dir}: {len(fake_semi_paths)}")

    model, opt = _load_dire(args)

    print(f"\n[eval] Evaluating {args.fake_ours_dir} vs {args.real_dir}...")
    res_ours = _eval_pair(model, opt, real_paths, fake_ours_paths, batch_size=args.batch_size)

    print(f"\n[eval] Evaluating {args.fake_semi_dir} vs {args.real_dir}...")
    res_semi = _eval_pair(model, opt, real_paths, fake_semi_paths, batch_size=args.batch_size)

    print(f"\n{'='*60}")
    print(f"RESULTS SUMMARY (DIRE method)")
    print(f"{'='*60}")

    def print_metrics(name: str, res: dict):
        print(f"\n{name}:")
        print(".4f")
        print(".4f")
        print(".4f")
        print(".4f")

    print_metrics(f"{args.real_dir} vs {args.fake_ours_dir}", res_ours)
    print_metrics(f"{args.real_dir} vs {args.fake_semi_dir}", res_semi)

    # Compare which fake set is harder to detect
    auc_ours = res_ours["auc"]
    auc_semi = res_semi["auc"]

    print(f"\n{'='*60}")
    print("VERDICT: Which fake set is harder to detect?")
    print(f"{'='*60}")

    if auc_ours < auc_semi:
        print(f"[verdict] Harder to detect (lower AUC): {args.fake_ours_dir}")
        print(".4f")
        print(".4f")
    elif auc_semi < auc_ours:
        print(f"[verdict] Harder to detect (lower AUC): {args.fake_semi_dir}")
        print(".4f")
        print(".4f")
    else:
        # Tie - use AP as tiebreaker
        ap_ours = res_ours["avg_precision"]
        ap_semi = res_semi["avg_precision"]
        if ap_ours < ap_semi:
            print(f"[verdict] TIED AUC - Harder to detect (lower AP): {args.fake_ours_dir}")
            print(".4f")
            print(".4f")
        else:
            print(f"[verdict] TIED AUC - Harder to detect (lower AP): {args.fake_semi_dir}")
            print(".4f")
            print(".4f")


if __name__ == "__main__":
    main()
