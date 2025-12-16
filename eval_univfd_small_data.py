"""
Evaluate UnivFD on a custom 3-folder setup and decide which fake set is harder to detect.

Expected directory layout (default):
  small_data/
    real_images/
    fake_ours/
    fake_semi-truths/

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
        self.items = list(items)
        self.opt = opt

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        path, label = self.items[idx]
        img = Image.open(path).convert("RGB")
        # match repo eval behavior: apply noise_type augmentation only at test time
        img = custom_augment(img, self.opt)
        tens, _ = process_img(img, self.opt, path, label)
        return tens, int(label)


def _as_numpy(x: Sequence[float]) -> np.ndarray:
    return np.asarray(list(x), dtype=np.float64)


def _fpr_at_tpr(y_true: np.ndarray, y_score: np.ndarray, tpr_target: float = 0.95) -> float:
    """
    Compute FPR@TPR=tpr_target using a simple threshold sweep.
    y_true: {0,1}, y_score: higher => more fake.
    """
    # Sort by score desc, treat each unique score as a threshold.
    order = np.argsort(-y_score)
    y_true = y_true[order]
    y_score = y_score[order]

    P = float((y_true == 1).sum())
    N = float((y_true == 0).sum())
    if P == 0 or N == 0:
        return float("nan")

    tp = 0.0
    fp = 0.0
    best_fpr = 1.0

    prev_score = None
    for i in range(len(y_true)):
        score = y_score[i]
        if prev_score is None:
            prev_score = score
        # When score changes, evaluate current operating point.
        if score != prev_score:
            tpr = tp / P
            fpr = fp / N
            if tpr >= tpr_target:
                best_fpr = min(best_fpr, fpr)
            prev_score = score

        if y_true[i] == 1:
            tp += 1.0
        else:
            fp += 1.0

    # Evaluate final point (threshold below min score)
    tpr = tp / P
    fpr = fp / N
    if tpr >= tpr_target:
        best_fpr = min(best_fpr, fpr)

    return float(best_fpr)


@dataclass
class EvalResult:
    auc: float
    ap: float
    acc: float
    r_acc: float
    f_acc: float
    fpr_at_95_tpr: float
    mean_score_real: float
    mean_score_fake: float
    n_real: int
    n_fake: int


@torch.no_grad()
def _eval_pair(model, opt, real_paths: Sequence[str], fake_paths: Sequence[str], batch_size: int) -> EvalResult:
    items: List[Tuple[str, int]] = [(p, 0) for p in real_paths] + [(p, 1) for p in fake_paths]
    ds = _BinaryFolderDataset(items, opt)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    y_true: List[int] = []
    y_score: List[float] = []

    for x, y in dl:
        x = x.to(opt.device)
        logits = model(x).flatten()
        scores = torch.sigmoid(logits).detach().cpu().numpy().tolist()
        y_true.extend([int(v) for v in y])
        y_score.extend([float(s) for s in scores])

    yt = _as_numpy(y_true).astype(np.int32)
    ys = _as_numpy(y_score)

    auc = float(roc_auc_score(yt, ys)) if (yt.min() != yt.max()) else float("nan")
    ap = float(average_precision_score(yt, ys)) if (yt.min() != yt.max()) else float("nan")
    acc = float(accuracy_score(yt, ys > 0.5))
    r_acc = float(accuracy_score(yt[yt == 0], (ys[yt == 0] > 0.5))) if (yt == 0).any() else float("nan")
    f_acc = float(accuracy_score(yt[yt == 1], (ys[yt == 1] > 0.5))) if (yt == 1).any() else float("nan")
    fpr95 = _fpr_at_tpr(yt, ys, 0.95)
    mean_real = float(np.mean(ys[yt == 0])) if (yt == 0).any() else float("nan")
    mean_fake = float(np.mean(ys[yt == 1])) if (yt == 1).any() else float("nan")

    return EvalResult(
        auc=auc,
        ap=ap,
        acc=acc,
        r_acc=r_acc,
        f_acc=f_acc,
        fpr_at_95_tpr=fpr95,
        mean_score_real=mean_real,
        mean_score_fake=mean_fake,
        n_real=int((yt == 0).sum()),
        n_fake=int((yt == 1).sum()),
    )


def _load_univfd(model_path: str, opt):
    model = get_model(opt)

    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)

    # UnivFD checkpoints in this repo are typically just the fc state_dict.
    state_dict = ckpt
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and "fc" in ckpt and isinstance(ckpt["fc"], dict):
        state_dict = ckpt["fc"]
    elif isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
        # fallback: sometimes checkpoints store full model under "model"
        state_dict = ckpt["model"]

    try:
        model.fc.load_state_dict(state_dict, strict=True)
    except Exception:
        # handle DataParallel prefixes or partial dicts
        if isinstance(state_dict, dict):
            cleaned = {k.replace("module.", ""): v for k, v in state_dict.items()}
            model.fc.load_state_dict(cleaned, strict=False)
        else:
            raise

    model.to(opt.device)
    model.eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--small_data_root",
        type=str,
        default="small_data",
        help="Path to the small_data folder.",
    )
    ap.add_argument("--real_dir", type=str, default="real_images")
    ap.add_argument("--fake_ours_dir", type=str, default="fake_ours")
    ap.add_argument("--fake_semi_dir", type=str, default="fake_semi-truths")
    ap.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the UnivFD checkpoint (typically the fc head weights).",
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
        raise FileNotFoundError(f"Missing model checkpoint: {args.model_path}")

    # Build a TestOptions-like object so preprocessing matches the repo.
    class Opt:
        pass
    opt = Opt()
    opt.detect_method = "UnivFD"
    opt.isTrain = False  # For evaluation/testing
    opt.model_path = args.model_path
    opt.noise_type = args.noise_type
    opt.no_crop = bool(args.no_crop)
    opt.no_resize = bool(args.no_resize)
    opt.batch_size = args.batch_size
    # Set default preprocessing options
    opt.rz_interp = ["bilinear"]
    opt.blur_sig = [1.0]
    opt.jpg_method = ["pil"]
    opt.jpg_qual = [95]
    opt.loadSize = 256
    opt.CropSize = 224

    if args.device is None:
        opt.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        opt.device = torch.device(args.device)

    model = _load_univfd(args.model_path, opt)

    real_paths = _collect_images(real_root)
    fake_ours_paths = _collect_images(fake_ours_root)
    fake_semi_paths = _collect_images(fake_semi_root)

    if len(real_paths) == 0:
        raise RuntimeError(f"No images found under {real_root}")
    if len(fake_ours_paths) == 0:
        raise RuntimeError(f"No images found under {fake_ours_root}")
    if len(fake_semi_paths) == 0:
        raise RuntimeError(f"No images found under {fake_semi_root}")

    print(f"[data] real_images: {len(real_paths)}")
    print(f"[data] fake_ours: {len(fake_ours_paths)}")
    print(f"[data] fake_semi-truths: {len(fake_semi_paths)}")
    if args.noise_type is not None:
        print(f"[eval] noise_type={args.noise_type}")
    print(f"[eval] device={opt.device}, batch_size={args.batch_size}")

    res_ours = _eval_pair(model, opt, real_paths, fake_ours_paths, batch_size=args.batch_size)
    res_semi = _eval_pair(model, opt, real_paths, fake_semi_paths, batch_size=args.batch_size)

    def _print(name: str, r: EvalResult):
        print(
            f"\n[{name}] n_real={r.n_real} n_fake={r.n_fake}\n"
            f"  AUC={r.auc:.4f}  AP={r.ap:.4f}  ACC@0.5={r.acc:.4f}\n"
            f"  r_acc={r.r_acc:.4f}  f_acc={r.f_acc:.4f}  FPR@95TPR={r.fpr_at_95_tpr:.4f}\n"
            f"  mean_score(real)={r.mean_score_real:.4f}  mean_score(fake)={r.mean_score_fake:.4f}"
        )

    _print("real vs fake_ours", res_ours)
    _print("real vs fake_semi-truths", res_semi)

    # Decide "harder": lower AUC, tie-break by lower AP.
    harder = "fake_ours"
    if np.isnan(res_ours.auc) and not np.isnan(res_semi.auc):
        harder = "fake_ours"
    elif np.isnan(res_semi.auc) and not np.isnan(res_ours.auc):
        harder = "fake_semi-truths"
    else:
        if res_semi.auc < res_ours.auc:
            harder = "fake_semi-truths"
        elif res_semi.auc == res_ours.auc and res_semi.ap < res_ours.ap:
            harder = "fake_semi-truths"

    print(f"\n[verdict] Harder to detect (lower AUC): {harder}")


if __name__ == "__main__":
    main()


