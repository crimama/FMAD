#!/usr/bin/env python3
"""Score Decomposition Experiment: Shift floor + sparse anomaly separation.

Core model: patch_score_i = shift_floor + normal_noise_i + anomaly_i
  - shift_floor: quasi-global (all patches elevated)
  - anomaly_i: sparse (only defect patches)

Experiments:
  1. D0: Score-space shift uniformity (vs feature-space 2.6%)
  2. Decomposition methods: various shift floor estimators
  3. Spatial-aware decomposition: per-region shift floor
  4. Comparison with existing best (patch max + q25_sub = 75.1%)

Setting: 8-shot, Layer 8 and Layer 11, no paired data, no shift knowledge.
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
    def extract_patches(self, img):
        x = self.transform(img).unsqueeze(0).to(self.device)
        _ = self.model(x)
        return self.feature[0, 1:].cpu().numpy()  # (M, dim)


# ─── Data Collection ──────────────────────────────────────────────

def collect_data(data_root, extractor, K_shot=8, seed=42):
    root = Path(data_root)
    rng = np.random.RandomState(seed)
    cats = sorted(d.name for d in root.iterdir() if d.is_dir() and (d / "train").is_dir())

    dataset = {}
    for cat in cats:
        train_patches = []
        train_dir = root / cat / "train" / "good"
        if train_dir.is_dir():
            paths = sorted(p for p in train_dir.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
            selected = rng.choice(len(paths), size=min(K_shot, len(paths)), replace=False)
            for idx in selected:
                train_patches.append(extractor.extract_patches(Image.open(paths[idx]).convert("RGB")))

        test_n_patches, test_a_patches = [], []
        test_dir = root / cat / "test_public"
        if test_dir.is_dir():
            for sub in sorted(test_dir.iterdir()):
                if not sub.is_dir():
                    continue
                tgt = test_n_patches if sub.name == "good" else test_a_patches
                for p in sorted(sub.iterdir())[:200]:
                    if p.suffix.lower() in (".png", ".jpg", ".jpeg"):
                        tgt.append(extractor.extract_patches(Image.open(p).convert("RGB")))

        dataset[cat] = {
            "train": train_patches,
            "test_n": test_n_patches,
            "test_a": test_a_patches,
        }
        print(f"  {cat}: train={len(train_patches)}, test_n={len(test_n_patches)}, test_a={len(test_a_patches)}")
    return dataset


# ─── Paired data for D0 diagnostic ───────────────────────────────

def collect_paired_scores(data_root, extractor, nn_model, max_pairs=100):
    """Get per-patch scores for regular and shifted versions of same objects."""
    root = Path(data_root)
    cats = sorted(d.name for d in root.iterdir() if d.is_dir() and (d / "test_public").is_dir())

    results = {}
    for cat in cats:
        good_dir = root / cat / "test_public" / "good"
        if not good_dir.is_dir():
            continue
        groups = defaultdict(dict)
        for p in sorted(good_dir.glob("*.png")):
            parts = p.stem.split("_", 1)
            if len(parts) == 2:
                groups[parts[0]][parts[1]] = p

        reg_scores_list, shift_scores_list = [], []
        count = 0
        for oid, conds in sorted(groups.items()):
            if "regular" not in conds:
                continue
            patches_reg = extractor.extract_patches(Image.open(conds["regular"]).convert("RGB"))
            scores_reg = nn_model.kneighbors(patches_reg)[0].ravel()

            for c, path in sorted(conds.items()):
                if c == "regular":
                    continue
                patches_s = extractor.extract_patches(Image.open(path).convert("RGB"))
                scores_s = nn_model.kneighbors(patches_s)[0].ravel()
                reg_scores_list.append(scores_reg)
                shift_scores_list.append(scores_s)
                count += 1
            if count >= max_pairs:
                break

        if reg_scores_list:
            results[cat] = {
                "reg": np.stack(reg_scores_list),    # (N_pairs, M)
                "shift": np.stack(shift_scores_list), # (N_pairs, M)
            }
    return results


# ─── Score Decomposition Methods ─────────────────────────────────

def decompose_scores(patch_scores, method="median"):
    """Estimate shift floor and return anomaly residual scores.

    Args:
        patch_scores: (M,) per-patch kNN distances
        method: shift floor estimation method

    Returns:
        anomaly_scores: (M,) residual after shift floor removal
    """
    M = len(patch_scores)

    if method == "none":
        return patch_scores

    elif method == "median_sub":
        floor = np.median(patch_scores)
        return patch_scores - floor

    elif method == "q25_sub":
        floor = np.percentile(patch_scores, 25)
        return patch_scores - floor

    elif method == "q10_sub":
        floor = np.percentile(patch_scores, 10)
        return patch_scores - floor

    elif method == "trimmed_mean":
        # Trimmed mean of bottom 80% as shift floor
        sorted_s = np.sort(patch_scores)
        n_keep = int(M * 0.8)
        floor = sorted_s[:n_keep].mean()
        return patch_scores - floor

    elif method == "mad_norm":
        # (score - median) / MAD
        med = np.median(patch_scores)
        mad = np.median(np.abs(patch_scores - med)) + 1e-8
        return (patch_scores - med) / mad

    elif method == "iqr_norm":
        # Normalize by interquartile range
        q25 = np.percentile(patch_scores, 25)
        q75 = np.percentile(patch_scores, 75)
        iqr = q75 - q25 + 1e-8
        return (patch_scores - q25) / iqr

    elif method == "spatial_4x4":
        # Per-region shift floor (4x4 grid)
        H = W = int(np.sqrt(M))
        if H * W != M:
            # Fallback to global median
            return patch_scores - np.median(patch_scores)
        scores_2d = patch_scores.reshape(H, W)
        result = np.zeros_like(scores_2d)
        grid = 4
        bh, bw = H // grid, W // grid
        for i in range(grid):
            for j in range(grid):
                block = scores_2d[i*bh:(i+1)*bh, j*bw:(j+1)*bw]
                floor = np.median(block)
                result[i*bh:(i+1)*bh, j*bw:(j+1)*bw] = block - floor
        return result.ravel()

    elif method == "spatial_8x8":
        H = W = int(np.sqrt(M))
        if H * W != M:
            return patch_scores - np.median(patch_scores)
        scores_2d = patch_scores.reshape(H, W)
        result = np.zeros_like(scores_2d)
        grid = 8
        bh, bw = max(1, H // grid), max(1, W // grid)
        for i in range(min(grid, H)):
            for j in range(min(grid, W)):
                r0, r1 = i*bh, min((i+1)*bh, H)
                c0, c1 = j*bw, min((j+1)*bw, W)
                block = scores_2d[r0:r1, c0:c1]
                if block.size > 0:
                    floor = np.median(block)
                    result[r0:r1, c0:c1] = block - floor
        return result.ravel()

    elif method == "robust_sparse":
        # Simple L1-inspired: iteratively estimate floor as median, residual as sparse
        floor = np.median(patch_scores)
        residual = patch_scores - floor
        # Second pass: re-estimate floor from low-residual patches
        low_mask = np.abs(residual) < np.percentile(np.abs(residual), 80)
        if low_mask.sum() > 0:
            floor = patch_scores[low_mask].mean()
        return patch_scores - floor

    else:
        raise ValueError(f"Unknown method: {method}")


# ─── Evaluation ──────────────────────────────────────────────────

def evaluate(dataset, decomp_method="none", agg="max", coreset_n=3000):
    results = {}
    aurocs = []

    agg_fn = {
        "max": np.max, "p99": lambda s: np.percentile(s, 99),
        "p95": lambda s: np.percentile(s, 95), "mean": np.mean,
    }[agg]

    for cat in sorted(dataset.keys()):
        d = dataset[cat]
        if not d["train"] or not d["test_n"] or not d["test_a"]:
            continue

        # Build patch memory bank
        bank = np.vstack(d["train"])
        if coreset_n < len(bank):
            rng = np.random.RandomState(42)
            idx = rng.choice(len(bank), coreset_n, replace=False)
            bank = bank[idx]
        nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
        nn.fit(bank)

        scores_n, scores_a = [], []
        for patches in d["test_n"]:
            raw = nn.kneighbors(patches)[0].ravel()
            decomposed = decompose_scores(raw, decomp_method)
            scores_n.append(agg_fn(decomposed))

        for patches in d["test_a"]:
            raw = nn.kneighbors(patches)[0].ravel()
            decomposed = decompose_scores(raw, decomp_method)
            scores_a.append(agg_fn(decomposed))

        labels = np.concatenate([np.zeros(len(scores_n)), np.ones(len(scores_a))])
        scores = np.concatenate([scores_n, scores_a])
        auroc = roc_auc_score(labels, scores)
        results[cat] = auroc
        aurocs.append(auroc)

    results["MEAN"] = np.mean(aurocs) if aurocs else 0.0
    return results


# ─── D0: Score Space Uniformity ──────────────────────────────────

def diagnose_score_uniformity(paired_scores):
    """Compare shift uniformity in SCORE space vs feature space (2.6%)."""
    print(f"\n  D0: Score-space shift uniformity")
    print(f"  {'Category':<16s} {'Uniformity':>12s} {'Mean shift':>12s} {'Std shift':>12s}")
    print(f"  " + "-" * 54)

    all_uniformities = []
    for cat, data in sorted(paired_scores.items()):
        # score_shift - score_reg per patch
        score_diffs = data["shift"] - data["reg"]  # (N_pairs, M)

        uniformities = []
        for i in range(len(score_diffs)):
            diff = score_diffs[i]
            global_mean = diff.mean()
            residual = diff - global_mean
            total_var = np.sum(diff ** 2)
            res_var = np.sum(residual ** 2)
            if total_var > 1e-10:
                uniformity = 1.0 - res_var / total_var
            else:
                uniformity = 1.0
            uniformities.append(uniformity)

        mean_u = np.mean(uniformities)
        mean_shift = np.mean(np.abs(score_diffs))
        std_shift = np.std(score_diffs.mean(axis=1))
        all_uniformities.extend(uniformities)
        print(f"  {cat:<16s} {mean_u:>11.3f} {mean_shift:>11.4f} {std_shift:>11.4f}")

    overall = np.mean(all_uniformities)
    print(f"\n  OVERALL score-space uniformity: {overall:.3f}")
    print(f"  (vs feature-space uniformity: 0.026)")
    print(f"  {'✅ Score > Feature' if overall > 0.026 else '❌ Not better'}")
    return overall


# ─── Main ────────────────────────────────────────────────────────

def run(args):
    print("=" * 60)
    print("Score Decomposition: Shift Floor + Sparse Anomaly")
    print("=" * 60)

    for layer in [11, 8]:
        print(f"\n{'='*50}")
        print(f"Layer {layer}")
        print(f"{'='*50}")

        extractor = DINOv2Extractor(args.model_name, args.device, layer)
        print(f"\nCollecting data (K={args.K_shot})...")
        dataset = collect_data(args.data_root, extractor, K_shot=args.K_shot, seed=42)

        # Build NN for D0 diagnostic
        if layer == 11:  # Only run D0 once
            print(f"\nD0: Score-space uniformity diagnostic...")
            # Need a NN model for score computation
            sample_cat = list(dataset.keys())[0]
            bank = np.vstack(dataset[sample_cat]["train"])
            if 3000 < len(bank):
                rng = np.random.RandomState(42)
                bank = bank[rng.choice(len(bank), 3000, replace=False)]
            nn_diag = NearestNeighbors(n_neighbors=1, metric="euclidean")
            nn_diag.fit(bank)

            paired = collect_paired_scores(args.data_root, extractor, nn_diag, max_pairs=50)
            if paired:
                score_uniformity = diagnose_score_uniformity(paired)

        # Pre-compute raw patch scores (cache) — build NN once per category
        print(f"\nPre-computing raw patch scores...")
        cached_scores = {}
        for cat in sorted(dataset.keys()):
            d = dataset[cat]
            if not d["train"] or not d["test_n"] or not d["test_a"]:
                continue
            bank = np.vstack(d["train"])
            if args.coreset_n < len(bank):
                rng = np.random.RandomState(42)
                bank = bank[rng.choice(len(bank), args.coreset_n, replace=False)]
            nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
            nn.fit(bank)

            raw_n = [nn.kneighbors(p)[0].ravel() for p in d["test_n"]]
            raw_a = [nn.kneighbors(p)[0].ravel() for p in d["test_a"]]
            cached_scores[cat] = {"test_n": raw_n, "test_a": raw_a}
            print(f"  {cat}: {len(raw_n)} normal + {len(raw_a)} anomaly images scored")

        # Evaluate all decomposition methods on cached scores
        methods = [
            ("none", "max"), ("none", "p99"), ("none", "p95"),
            ("median_sub", "max"), ("q25_sub", "max"), ("q10_sub", "max"),
            ("trimmed_mean", "max"), ("mad_norm", "max"), ("iqr_norm", "max"),
            ("spatial_4x4", "max"), ("spatial_8x8", "max"), ("robust_sparse", "max"),
            ("median_sub", "p99"), ("q25_sub", "p99"), ("mad_norm", "p99"),
            ("robust_sparse", "p99"),
        ]

        print(f"\nEvaluating {len(methods)} decomposition methods (on cached scores)...")
        all_results = {}
        for decomp, agg in methods:
            label = f"{decomp}+{agg}"
            agg_fn = {"max": np.max, "p99": lambda s: np.percentile(s, 99),
                      "p95": lambda s: np.percentile(s, 95), "mean": np.mean}[agg]

            cat_aurocs = {}
            for cat, scores in cached_scores.items():
                s_n = [agg_fn(decompose_scores(s, decomp)) for s in scores["test_n"]]
                s_a = [agg_fn(decompose_scores(s, decomp)) for s in scores["test_a"]]
                labels = np.concatenate([np.zeros(len(s_n)), np.ones(len(s_a))])
                vals = np.concatenate([s_n, s_a])
                cat_aurocs[cat] = roc_auc_score(labels, vals)

            cat_aurocs["MEAN"] = np.mean(list(v for k, v in cat_aurocs.items() if k != "MEAN"))
            all_results[label] = cat_aurocs

        # Summary
        print(f"\n  {'Method':<30s} {'AUROC':>8s} {'vs none+max':>12s}")
        print(f"  " + "-" * 52)
        baseline = all_results["none+max"]["MEAN"]
        for label in sorted(all_results.keys(), key=lambda k: all_results[k]["MEAN"], reverse=True):
            val = all_results[label]["MEAN"] * 100
            delta = (all_results[label]["MEAN"] - baseline) * 100
            marker = " ★" if all_results[label]["MEAN"] == max(r["MEAN"] for r in all_results.values()) else ""
            print(f"  {label:<30s} {val:>7.1f}% {delta:>+10.1f}pp{marker}")

        # Per-category for best method
        best_label = max(all_results, key=lambda k: all_results[k]["MEAN"])
        print(f"\n  Best: {best_label}")
        print(f"  {'Category':<16s} {'none+max':>10s} {best_label:>15s} {'Delta':>8s}")
        print(f"  " + "-" * 52)
        for cat in sorted(k for k in all_results["none+max"] if k != "MEAN"):
            base = all_results["none+max"].get(cat, 0) * 100
            best = all_results[best_label].get(cat, 0) * 100
            print(f"  {cat:<16s} {base:>9.1f}% {best:>14.1f}% {best-base:>+7.1f}")

    # Save
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "score_decomposition_results.json", "w") as f:
        serializable = {k: {kk: float(vv) for kk, vv in v.items()} for k, v in all_results.items()}
        json.dump(serializable, f, indent=2)
    print(f"\n  Saved to {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default="/home/hun/Volume/DATA/mvtec_ad_2")
    p.add_argument("--model_name", default="dinov2_vitb14")
    p.add_argument("--device", default="cuda")
    p.add_argument("--K_shot", type=int, default=8)
    p.add_argument("--coreset_n", type=int, default=3000)
    p.add_argument("--output_dir", default="results/score_decomposition")
    run(p.parse_args())
