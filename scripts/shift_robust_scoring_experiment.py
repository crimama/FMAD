#!/usr/bin/env python3
"""Shift-Robust Scoring: Why patch-level works and how to improve it.

Core hypothesis: Anomaly is LOCAL (specific patches), shift is GLOBAL (all patches).
  → Aggregation strategy determines shift robustness.
  → Score-level normalization can remove global shift floor without paired data.

Experiment matrix:
  Layer: {8, 11}
  Level: {image, patch}
  Aggregation: {mean, max, p95, p99}
  Normalization: {none, median_subtract, mad_normalize}
  Setting: few-shot K=4,8 from train objects (regular condition only)
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
from torchvision import transforms


# ─── Feature Extraction ───────────────────────────────────────────

class DINOv2Extractor:
    def __init__(self, model_name="dinov2_vitb14", device="cuda", target_layer=8):
        self.device = device
        self.model = torch.hub.load("facebookresearch/dinov2", model_name).to(device).eval()
        self.target_layer = target_layer
        self.feature = None
        self.model.blocks[self.target_layer].register_forward_hook(self._hook)
        self.transform = transforms.Compose([
            transforms.Resize(518, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(518),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def _hook(self, module, inp, output):
        self.feature = output.detach()

    @torch.no_grad()
    def extract_all(self, img):
        """Return (cls_patch_mean, all_patches)."""
        x = self.transform(img).unsqueeze(0).to(self.device)
        _ = self.model(x)
        feat = self.feature[0]
        cls = feat[0]
        patches = feat[1:]  # (M, dim)
        pm = patches.mean(0)
        img_feat = torch.cat([cls, pm])
        img_feat = img_feat / (img_feat.norm() + 1e-8)
        return img_feat.cpu().numpy(), patches.cpu().numpy()


# ─── Data Collection ──────────────────────────────────────────────

def collect_data(data_root, extractor, K_shot=8, seed=42):
    """Collect few-shot reference (regular only) + all test data."""
    root = Path(data_root)
    rng = np.random.RandomState(seed)
    cats = sorted(d.name for d in root.iterdir() if d.is_dir() and (d / "train").is_dir())

    dataset = {}
    for cat in cats:
        # Train objects (regular only) for few-shot reference
        train_dir = root / cat / "train" / "good"
        train_imgs, train_patches = [], []
        if train_dir.is_dir():
            paths = sorted(p for p in train_dir.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
            selected = rng.choice(len(paths), size=min(K_shot, len(paths)), replace=False)
            for idx in selected:
                img_f, pat_f = extractor.extract_all(Image.open(paths[idx]).convert("RGB"))
                train_imgs.append(img_f)
                train_patches.append(pat_f)

        # Test (all conditions mixed — realistic)
        test_n_imgs, test_n_patches = [], []
        test_a_imgs, test_a_patches = [], []
        test_dir = root / cat / "test_public"
        if test_dir.is_dir():
            for sub in sorted(test_dir.iterdir()):
                if not sub.is_dir():
                    continue
                is_normal = sub.name == "good"
                tgt_imgs = test_n_imgs if is_normal else test_a_imgs
                tgt_patches = test_n_patches if is_normal else test_a_patches
                for p in sorted(sub.iterdir())[:200]:
                    if p.suffix.lower() in (".png", ".jpg", ".jpeg"):
                        img_f, pat_f = extractor.extract_all(Image.open(p).convert("RGB"))
                        tgt_imgs.append(img_f)
                        tgt_patches.append(pat_f)

        dataset[cat] = {
            "train_imgs": np.stack(train_imgs) if train_imgs else np.zeros((0, 1536)),
            "train_patches": train_patches,  # list of (M, dim)
            "test_n_imgs": np.stack(test_n_imgs) if test_n_imgs else np.zeros((0, 1536)),
            "test_n_patches": test_n_patches,
            "test_a_imgs": np.stack(test_a_imgs) if test_a_imgs else np.zeros((0, 1536)),
            "test_a_patches": test_a_patches,
        }
        print(f"  {cat}: ref={len(train_imgs)}, test_n={len(test_n_imgs)}, test_a={len(test_a_imgs)}")
    return dataset


# ─── Scoring Functions ────────────────────────────────────────────

def image_mahal_score(train_imgs, test_imgs):
    mu = train_imgs.mean(0)
    if len(train_imgs) < 3:
        return np.linalg.norm(test_imgs - mu, axis=1)
    cov = np.cov((train_imgs - mu).T) + np.eye(train_imgs.shape[1]) * 1e-5
    try:
        inv = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        return np.linalg.norm(test_imgs - mu, axis=1)
    tc = test_imgs - mu
    return np.sqrt(np.sum(tc @ inv * tc, axis=1))


def patch_scores_knn(train_patch_list, test_patches, coreset_n=3000):
    """Per-patch kNN scores. Returns (M,) scores."""
    bank = np.vstack(train_patch_list)
    if coreset_n < len(bank):
        rng = np.random.RandomState(42)
        idx = rng.choice(len(bank), coreset_n, replace=False)
        bank = bank[idx]
    nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
    nn.fit(bank)
    d, _ = nn.kneighbors(test_patches)
    return d.ravel()


def aggregate_patch_scores(patch_scores, method="max", normalize="none"):
    """Aggregate M patch scores to single image score.

    normalize: how to handle global shift floor
      none: raw aggregation
      median_sub: subtract median (removes global floor)
      mad_norm: (score - median) / MAD (normalized deviation)
    """
    if normalize == "median_sub":
        median = np.median(patch_scores)
        patch_scores = patch_scores - median
    elif normalize == "mad_norm":
        median = np.median(patch_scores)
        mad = np.median(np.abs(patch_scores - median)) + 1e-8
        patch_scores = (patch_scores - median) / mad
    elif normalize == "q25_sub":
        q25 = np.percentile(patch_scores, 25)
        patch_scores = patch_scores - q25

    if method == "max":
        return np.max(patch_scores)
    elif method == "p99":
        return np.percentile(patch_scores, 99)
    elif method == "p95":
        return np.percentile(patch_scores, 95)
    elif method == "mean":
        return np.mean(patch_scores)
    else:
        raise ValueError(method)


# ─── Evaluation ──────────────────────────────────────────────────

def evaluate(dataset, level, agg="max", normalize="none", coreset_n=3000):
    """Evaluate a scoring configuration."""
    results = {}
    aurocs = []

    for cat in sorted(dataset.keys()):
        d = dataset[cat]

        if level == "image":
            scores_n = image_mahal_score(d["train_imgs"], d["test_n_imgs"])
            scores_a = image_mahal_score(d["train_imgs"], d["test_a_imgs"])
        elif level == "patch":
            if not d["train_patches"]:
                continue
            # Build bank once
            bank = np.vstack(d["train_patches"])
            if coreset_n < len(bank):
                rng = np.random.RandomState(42)
                idx = rng.choice(len(bank), coreset_n, replace=False)
                bank = bank[idx]
            nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
            nn.fit(bank)

            scores_n = []
            for patches in d["test_n_patches"]:
                ps = nn.kneighbors(patches)[0].ravel()
                scores_n.append(aggregate_patch_scores(ps, agg, normalize))

            scores_a = []
            for patches in d["test_a_patches"]:
                ps = nn.kneighbors(patches)[0].ravel()
                scores_a.append(aggregate_patch_scores(ps, agg, normalize))

            scores_n = np.array(scores_n)
            scores_a = np.array(scores_a)
        else:
            raise ValueError(level)

        if len(scores_n) == 0 or len(scores_a) == 0:
            continue

        labels = np.concatenate([np.zeros(len(scores_n)), np.ones(len(scores_a))])
        scores = np.concatenate([scores_n, scores_a])
        auroc = roc_auc_score(labels, scores)
        results[cat] = auroc
        aurocs.append(auroc)

    results["MEAN"] = np.mean(aurocs) if aurocs else 0.0
    return results


# ─── Main ────────────────────────────────────────────────────────

def run(args):
    print("=" * 60)
    print("Shift-Robust Scoring: Local vs Global Principle")
    print("=" * 60)

    all_results = {}

    for layer in [8, 11]:
        print(f"\n{'='*40}")
        print(f"Layer {layer}")
        print(f"{'='*40}")

        extractor = DINOv2Extractor(args.model_name, args.device, layer)
        print(f"\nCollecting data (K={args.K_shot})...")
        dataset = collect_data(args.data_root, extractor, K_shot=args.K_shot, seed=42)

        # Image-level baseline
        label = f"L{layer}_img_mahal"
        print(f"\n  --- {label} ---")
        res = evaluate(dataset, "image")
        all_results[label] = res
        print(f"  MEAN: {res['MEAN']*100:.1f}%")

        # Patch-level with different aggregations and normalizations
        for agg in ["mean", "p95", "p99", "max"]:
            for norm in ["none", "median_sub", "mad_norm", "q25_sub"]:
                label = f"L{layer}_patch_{agg}_{norm}"
                res = evaluate(dataset, "patch", agg=agg, normalize=norm, coreset_n=args.coreset_n)
                all_results[label] = res
                print(f"  {label}: {res['MEAN']*100:.1f}%")

    # Summary
    print("\n" + "=" * 60)
    print(f"SUMMARY (K={args.K_shot}-shot, AD2 I-AUROC %)")
    print("=" * 60)

    # Group by layer
    for layer in [8, 11]:
        print(f"\n  Layer {layer}:")
        print(f"  {'Method':<35s} {'AUROC':>8s}")
        print(f"  " + "-" * 45)

        img_key = f"L{layer}_img_mahal"
        print(f"  {'image_mahal':<35s} {all_results[img_key]['MEAN']*100:>7.1f}%")

        for agg in ["mean", "p95", "p99", "max"]:
            for norm in ["none", "median_sub", "mad_norm", "q25_sub"]:
                key = f"L{layer}_patch_{agg}_{norm}"
                val = all_results[key]["MEAN"] * 100
                marker = ""
                img_val = all_results[img_key]["MEAN"] * 100
                if val > img_val + 1:
                    marker = " ✅"
                elif val < img_val - 1:
                    marker = " ❌"
                print(f"  patch_{agg}_{norm:<12s}          {val:>7.1f}%{marker}")

    # Best per layer
    print(f"\n  Best configurations:")
    for layer in [8, 11]:
        best_key = max(
            (k for k in all_results if k.startswith(f"L{layer}_patch")),
            key=lambda k: all_results[k]["MEAN"]
        )
        best_val = all_results[best_key]["MEAN"] * 100
        img_val = all_results[f"L{layer}_img_mahal"]["MEAN"] * 100
        print(f"  L{layer}: {best_key} = {best_val:.1f}% (vs img {img_val:.1f}%, Δ={best_val-img_val:+.1f}pp)")

    # Save
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    serializable = {k: {kk: float(vv) for kk, vv in v.items()} for k, v in all_results.items()}
    with open(out / "shift_robust_scoring.json", "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\n  Saved to {out / 'shift_robust_scoring.json'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default="/home/hun/Volume/DATA/mvtec_ad_2")
    p.add_argument("--model_name", default="dinov2_vitb14")
    p.add_argument("--device", default="cuda")
    p.add_argument("--K_shot", type=int, default=8)
    p.add_argument("--coreset_n", type=int, default=3000)
    p.add_argument("--output_dir", default="results/shift_robust_scoring")
    run(p.parse_args())
