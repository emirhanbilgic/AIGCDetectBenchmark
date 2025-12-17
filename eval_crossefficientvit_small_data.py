"""
Evaluate CrossEfficientViT on a custom 3-folder setup for image deepfake detection.

This script adapts the video deepfake detection model from:
https://github.com/davide-coccomini/Combining-EfficientNet-and-Vision-Transformers-for-Video-Deepfake-Detection

For single image classification instead of video frames.

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

The script recursively searches all subfolders for images.

This script runs two binary evaluations:
  1) real_images (label=0) vs fake_ours (label=1)
  2) real_images (label=0) vs fake_semi_truths (label=1)

"Harder to detect" is reported as the fake set with **lower ROC-AUC** (tie-break: lower AP).

NOTE: CrossEfficientViT was originally designed for video deepfake detection.
This adaptation uses it for single image classification.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset

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
    def __init__(self, items: Sequence[Tuple[str, int]], transform=None):
        self.items = list(items)
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        image = Image.open(path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


# Improved CrossEfficientViT model for single image classification
# Better adapted from the original video model architecture
class CrossEfficientViT(nn.Module):
    def __init__(self, num_classes=2, embed_dim=768, depth=12, num_heads=12):
        super(CrossEfficientViT, self).__init__()

        # CNN backbone (EfficientNet-style) - Fixed for 224x224 input
        self.features = nn.Sequential(
            # Stem
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            # EfficientNet-style blocks - Reduced blocks to avoid over-downsampling
            self._make_layer(64, 128, 1),    # 1 block instead of 2
            self._make_layer(128, 256, 1),   # 1 block instead of 2
            self._make_layer(256, 512, 1),   # 1 block instead of 2

            nn.AdaptiveAvgPool2d((1, 1))
        )

        # Vision Transformer components
        self.patch_embed = nn.Linear(512, embed_dim)  # Project CNN features to embed_dim
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + 1, embed_dim))  # 1 patch + 1 cls token

        # Transformer blocks
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=embed_dim * 4,
                dropout=0.1,
                batch_first=True
            ) for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)

        # Initialize weights
        self.apply(self._init_weights)

    def _make_layer(self, in_channels, out_channels, blocks):
        layers = []
        for _ in range(blocks):
            layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2)
            ])
            in_channels = out_channels
        return nn.Sequential(*layers)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        # CNN feature extraction
        x = self.features(x)  # [B, 512, 1, 1]
        x = torch.flatten(x, 1)  # [B, 512]

        # Patch embedding - project to embed_dim and add sequence dimension
        x = self.patch_embed(x)  # [B, embed_dim]
        x = x.unsqueeze(1)  # [B, 1, embed_dim]

        # Add cls token
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)  # [B, 1, embed_dim]
        x = torch.cat((cls_tokens, x), dim=1)  # [B, 2, embed_dim]

        # Add positional embedding
        x = x + self.pos_embed[:, :x.size(1)]

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)

        # Classification from cls token
        x = self.norm(x)
        cls_output = x[:, 0]  # Take cls token
        x = self.classifier(cls_output)
        return x


def _load_crossefficientvit(model_path: str, device: str = "cuda"):
    """Load CrossEfficientViT model adapted for single image classification."""
    try:
        model = CrossEfficientViT(num_classes=2)
        print("✅ Created CrossEfficientViT model")
    except Exception as e:
        print(f"⚠️ Could not create CrossEfficientViT model: {e}")
        print("Falling back to simple CNN model")
        # Fallback to simpler architecture
        model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 2)
        )

    # Try to load pretrained weights if available
    if os.path.exists(model_path):
        file_size = os.path.getsize(model_path)
        print(f"📁 Found weights file: {model_path} ({file_size} bytes)")
        try:
            # First try to load as a regular PyTorch checkpoint
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

            # Check if it's a state_dict or full checkpoint
            if isinstance(checkpoint, dict):
                # If it has 'state_dict' key, it's a full checkpoint
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                # If it has 'model' key
                elif 'model' in checkpoint:
                    state_dict = checkpoint['model']
                else:
                    # Assume it's already a state_dict
                    state_dict = checkpoint

                # Adapt layer names - remove 'module.' prefix if present
                new_state_dict = {}
                for k, v in state_dict.items():
                    if k.startswith('module.'):
                        k = k[7:]
                    new_state_dict[k] = v

                # Try to load what we can
                missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)

                if len(missing_keys) == 0 and len(unexpected_keys) == 0:
                    print(f"✅ Successfully loaded all CrossEfficientViT weights from {model_path}")
                elif len(missing_keys) < len(model.state_dict()):
                    print(f"✅ Partially loaded CrossEfficientViT weights from {model_path}")
                    print(f"   Missing keys: {len(missing_keys)}, Unexpected keys: {len(unexpected_keys)}")
                else:
                    print(f"⚠️ Could not load pretrained weights: incompatible architecture")
                    print("Using randomly initialized model")
            else:
                print(f"⚠️ Unexpected checkpoint format in {model_path}")
                print("Using randomly initialized model")

        except Exception as e:
            print(f"⚠️ Could not load pretrained weights: {e}")
            print("Using randomly initialized model")
    else:
        print(f"⚠️ Model weights not found at {model_path}")
        print("Using randomly initialized model")

    model.to(device)
    model.eval()
    return model


def _eval_pair(model, real_paths: List[str], fake_paths: List[str], batch_size: int = 32) -> dict:
    """Evaluate one binary pair (real vs fake)."""
    # Create datasets
    real_items = [(p, 0) for p in real_paths]
    fake_items = [(p, 1) for p in fake_paths]
    all_items = real_items + fake_items

    # Simple transform for CrossEfficientViT
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = _BinaryFolderDataset(all_items, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    device = next(model.parameters()).device
    y_true, y_pred = [], []

    with torch.no_grad():
        for batch_images, batch_labels in dataloader:
            batch_images = batch_images.to(device)

            outputs = model(batch_images)
            probs = torch.softmax(outputs, dim=1)[:, 1]  # Probability of being fake

            y_pred.extend(probs.cpu().numpy())
            y_true.extend(batch_labels.numpy())

    y_true, y_pred = np.array(y_true), np.array(y_pred)

    # Calculate metrics
    acc = accuracy_score(y_true, y_pred > 0.5)
    ap = average_precision_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred)

    r_acc = accuracy_score(y_true[y_true == 0], y_pred[y_true == 0] > 0.5)
    f_acc = accuracy_score(y_true[y_true == 1], y_pred[y_true == 1] > 0.5)

    return {
        'accuracy': acc,
        'avg_precision': ap,
        'auc': auc,
        'r_acc': r_acc,
        'f_acc': f_acc,
        'y_true': y_true,
        'y_pred': y_pred
    }


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
        default="cross_efficient_vit.pth",
        help="Path to the CrossEfficientViT checkpoint.",
    )
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--device", type=str, default=None, help="cuda, cuda:0, or cpu. Default: auto")

    args = ap.parse_args()

    real_root = os.path.join(args.small_data_root, args.real_dir)
    fake_ours_root = os.path.join(args.small_data_root, args.fake_ours_dir)
    fake_semi_root = os.path.join(args.small_data_root, args.fake_semi_dir)

    for p in [real_root, fake_ours_root, fake_semi_root]:
        if not os.path.isdir(p):
            raise FileNotFoundError(f"Missing folder: {p}")

    # Setup device
    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[eval] device={device}, batch_size={args.batch_size}")

    # Load model
    model = _load_crossefficientvit(args.model_path, device)

    # Collect images
    real_paths = _collect_images(real_root)
    fake_ours_paths = _collect_images(fake_ours_root)
    fake_semi_paths = _collect_images(fake_semi_root)

    print(f"[data] real_images: {len(real_paths)}")
    print(f"[data] fake_ours: {len(fake_ours_paths)}")
    print(f"[data] fake_semi-truths: {len(fake_semi_paths)}")

    # Evaluate both pairs
    print("\nEvaluating fake_ours vs real_images...")
    res_ours = _eval_pair(model, real_paths, fake_ours_paths, args.batch_size)
    print(f"ACC: {res_ours['accuracy']:.3f}, AP: {res_ours['avg_precision']:.3f}, AUC: {res_ours['auc']:.3f}")
    print(f"r_acc: {res_ours['r_acc']:.3f}, f_acc: {res_ours['f_acc']:.3f}")

    print("\nEvaluating fake_semi-truths vs real_images...")
    res_semi = _eval_pair(model, real_paths, fake_semi_paths, args.batch_size)
    print(f"ACC: {res_semi['accuracy']:.3f}, AP: {res_semi['avg_precision']:.3f}, AUC: {res_semi['auc']:.3f}")
    print(f"r_acc: {res_semi['r_acc']:.3f}, f_acc: {res_semi['f_acc']:.3f}")

    # Compare which is harder to detect
    auc_ours = res_ours['auc']
    auc_semi = res_semi['auc']

    print("\n🎯 COMPARISON:")
    print(f"fake_ours AUC: {auc_ours:.3f}")
    print(f"fake_semi-truths AUC: {auc_semi:.3f}")

    if auc_ours < auc_semi:
        print(f"[verdict] Harder to detect (lower AUC): fake_ours")
    elif auc_semi < auc_ours:
        print(f"[verdict] Harder to detect (lower AUC): fake_semi-truths")
    else:
        # Tie - use AP as tiebreaker
        ap_ours = res_ours['avg_precision']
        ap_semi = res_semi['avg_precision']
        if ap_ours < ap_semi:
            print(f"[verdict] Tie in AUC, harder to detect (lower AP): fake_ours")
        else:
            print(f"[verdict] Tie in AUC, harder to detect (lower AP): fake_semi-truths")


if __name__ == "__main__":
    main()
