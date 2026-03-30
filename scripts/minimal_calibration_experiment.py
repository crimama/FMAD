#!/usr/bin/env python3
"""Minimal Calibration Experiment: How few paired samples does NSP need?

Core question: NSP with full pairs (~315) gives 83.8% AD2.
  How many pairs are SUFFICIENT to approach this performance?

Experiment: Subsample paired shift vectors → NSP → AD scoring.
  N_pairs ∈ {1, 2, 3, 5, 10, 20, 50, 100, full}
  3 random seeds per N_pairs for variance estimation.

This directly addresses the "Minimal Sufficiency" contribution:
  "Paired calibration is necessary (impossibility) AND efficient (N pairs suffice)"
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score
from torchvision import transforms


# ─── Feature Extraction ───────────────────────────────────────────

class DINOv2Extractor:
    def __init__(self, model_name: str = "dinov2_vitb14", device: str = "cuda", target_layer: int = 8):
        self.device = device
        self.model = torch.hub.load("facebookresearch/dinov2", model_name).to(device).eval()
        self.target_layer = target_layer
        self.feature = None
        self.model.blocks[self.target_layer].register_forward_hook(self._hook)
        self.transform = transforms.Compose([
            transforms.Resize(518, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(518),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def _hook(self, module, inp, output):
        self.feature = output.detach()

    @torch.no_grad()
    def extract(self, img: Image.Image) -> np.ndarray:
        x = self.transform(img).unsqueeze(0).to(self.device)
        _ = self.model(x)
        feat = self.feature[0]
        cls_token = feat[0]
        patch_mean = feat[1:].mean(dim=0)
        combined = torch.cat([cls_token, patch_mean])
        combined = combined / (combined.norm() + 1e-8)
        return combined.cpu().numpy()


# ─── Data Collection ──────────────────────────────────────────────

def collect_shift_vectors_per_category(
    data_root: str,
    extractor: DINOv2Extractor,
) -> dict[str, np.ndarray]:
    """Collect ALL paired shift vectors per category."""
    root = Path(data_root)
    categories = sorted(
        d.name for d in root.iterdir()
        if d.is_dir() and (d / "test_public").is_dir()
    )
    all_shifts = {}
    for cat in categories:
        good_dir = root / cat / "test_public" / "good"
        if not good_dir.is_dir():
            continue

        groups: dict[str, dict[str, Path]] = defaultdict(dict)
        for img_path in sorted(good_dir.glob("*.png")):
            parts = img_path.stem.split("_", 1)
            if len(parts) == 2:
                groups[parts[0]][parts[1]] = img_path

        shifts = []
        for obj_idx, conditions in sorted(groups.items()):
            if "regular" not in conditions:
                continue
            feat_reg = extractor.extract(Image.open(conditions["regular"]).convert("RGB"))
            for cond, path in sorted(conditions.items()):
                if cond == "regular":
                    continue
                feat_shift = extractor.extract(Image.open(path).convert("RGB"))
                shifts.append(feat_shift - feat_reg)

        all_shifts[cat] = np.stack(shifts) if shifts else np.zeros((0, 1536))
        print(f"  {cat}: {len(all_shifts[cat])} shift vectors")
    return all_shifts


def collect_test_features(
    data_root: str,
    extractor: DINOv2Extractor,
    max_per_cat: int = 200,
) -> dict[str, dict[str, np.ndarray]]:
    """Collect train normal + test features per category."""
    root = Path(data_root)
    categories = sorted(
        d.name for d in root.iterdir()
        if d.is_dir() and (d / "train").is_dir()
    )
    dataset = {}
    for cat in categories:
        cat_data = {"train_normal": [], "test_normal": [], "test_anomaly": []}

        train_good = root / cat / "train" / "good"
        if train_good.is_dir():
            for p in sorted(train_good.iterdir())[:max_per_cat]:
                if p.suffix.lower() in (".png", ".jpg", ".jpeg"):
                    cat_data["train_normal"].append(extractor.extract(Image.open(p).convert("RGB")))

        test_dir = root / cat / "test_public"
        if test_dir.is_dir():
            for sub in sorted(test_dir.iterdir()):
                if not sub.is_dir():
                    continue
                label = "test_normal" if sub.name == "good" else "test_anomaly"
                for p in sorted(sub.iterdir())[:max_per_cat]:
                    if p.suffix.lower() in (".png", ".jpg", ".jpeg"):
                        cat_data[label].append(extractor.extract(Image.open(p).convert("RGB")))

        for k in cat_data:
            cat_data[k] = np.stack(cat_data[k]) if cat_data[k] else np.zeros((0, 1536))

        print(f"  {cat}: train={len(cat_data['train_normal'])}, "
              f"test_n={len(cat_data['test_normal'])}, test_a={len(cat_data['test_anomaly'])}")
        dataset[cat] = cat_data
    return dataset


# ─── NSP with Subsampled Pairs ───────────────────────────────────

def nsp_nuisance_basis(
    shift_vectors: np.ndarray,
    K: int = 100,
    n_pairs: int | None = None,
    seed: int = 42,
) -> np.ndarray:
    """Compute NSP nuisance basis from (optionally subsampled) shift vectors.

    Args:
        shift_vectors: (N, dim) all available shift vectors
        K: number of nuisance dimensions
        n_pairs: if set, subsample this many shift vectors
        seed: random seed for subsampling

    Returns: (dim, K) orthonormal nuisance basis
    """
    if n_pairs is not None and n_pairs < len(shift_vectors):
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(shift_vectors), size=n_pairs, replace=False)
        shift_vectors = shift_vectors[idx]

    actual_k = min(K, len(shift_vectors) - 1)
    if actual_k <= 0:
        return np.zeros((shift_vectors.shape[1], 0))

    centered = shift_vectors - shift_vectors.mean(axis=0)
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    return Vt[:actual_k].T


def hard_project(features: np.ndarray, nuisance_dirs: np.ndarray) -> np.ndarray:
    if nuisance_dirs.shape[1] == 0:
        return features
    proj = nuisance_dirs @ nuisance_dirs.T
    return features - features @ proj


def mahalanobis_score(train: np.ndarray, test: np.ndarray) -> np.ndarray:
    mu = train.mean(axis=0)
    cov = np.cov((train - mu).T) + np.eye(train.shape[1]) * 1e-6
    cov_inv = np.linalg.inv(cov)
    tc = test - mu
    return np.sqrt(np.sum(tc @ cov_inv * tc, axis=1))


# ─── Evaluation ──────────────────────────────────────────────────

def evaluate_nsp(
    dataset: dict[str, dict[str, np.ndarray]],
    all_shifts: dict[str, np.ndarray],
    K: int,
    n_pairs: int | None,
    seed: int,
    global_nuisance: bool = True,
) -> dict[str, float]:
    """Evaluate NSP with given number of pairs.

    Two modes:
      global_nuisance=True: pool all categories' shifts → single nuisance basis (original NSP)
      global_nuisance=False: per-category nuisance basis
    """
    if global_nuisance:
        # Pool all shift vectors across categories
        all_sv = []
        for cat in sorted(all_shifts.keys()):
            if len(all_shifts[cat]) > 0:
                all_sv.append(all_shifts[cat])
        pooled = np.vstack(all_sv)
        nuisance = nsp_nuisance_basis(pooled, K=K, n_pairs=n_pairs, seed=seed)
    else:
        nuisance = None  # computed per-category below

    results = {}
    aurocs = []
    for cat in sorted(dataset.keys()):
        train = dataset[cat]["train_normal"]
        test_n = dataset[cat]["test_normal"]
        test_a = dataset[cat]["test_anomaly"]
        if len(train) == 0 or len(test_n) == 0 or len(test_a) == 0:
            continue

        if not global_nuisance:
            cat_shifts = all_shifts.get(cat, np.zeros((0, train.shape[1])))
            nuisance = nsp_nuisance_basis(cat_shifts, K=K, n_pairs=n_pairs, seed=seed)

        train_p = hard_project(train, nuisance)
        test_n_p = hard_project(test_n, nuisance)
        test_a_p = hard_project(test_a, nuisance)

        scores_n = mahalanobis_score(train_p, test_n_p)
        scores_a = mahalanobis_score(train_p, test_a_p)
        labels = np.concatenate([np.zeros(len(scores_n)), np.ones(len(scores_a))])
        scores = np.concatenate([scores_n, scores_a])
        auroc = roc_auc_score(labels, scores)
        results[cat] = auroc
        aurocs.append(auroc)

    results["MEAN"] = np.mean(aurocs) if aurocs else 0.0
    return results


# ─── Main ────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> None:
    print("=" * 60)
    print("Minimal Calibration: How Few Paired Samples Suffice?")
    print("=" * 60)

    extractor = DINOv2Extractor(
        model_name=args.model_name, device=args.device, target_layer=args.layer,
    )

    print(f"\n[1/3] Collecting shift vectors (Layer {args.layer})...")
    all_shifts = collect_shift_vectors_per_category(args.data_root, extractor)
    total_shifts = sum(len(v) for v in all_shifts.values())
    print(f"  Total: {total_shifts} shift vectors")

    print(f"\n[2/3] Collecting train/test features...")
    dataset = collect_test_features(args.data_root, extractor, max_per_cat=args.max_per_cat)

    print(f"\n[3/3] Evaluating NSP with varying N_pairs...")

    n_pairs_list = [1, 2, 3, 5, 10, 20, 50, 100, None]  # None = full
    seeds = [42, 123, 456]
    K = args.K

    all_results = {}

    # Baseline (no projection)
    print(f"\n  --- baseline ---")
    res = evaluate_nsp(dataset, all_shifts, K=K, n_pairs=None, seed=42, global_nuisance=True)
    # Override: no projection
    res_base = {}
    aurocs_base = []
    for cat in sorted(dataset.keys()):
        train = dataset[cat]["train_normal"]
        test_n = dataset[cat]["test_normal"]
        test_a = dataset[cat]["test_anomaly"]
        if len(train) == 0 or len(test_n) == 0 or len(test_a) == 0:
            continue
        scores_n = mahalanobis_score(train, test_n)
        scores_a = mahalanobis_score(train, test_a)
        labels = np.concatenate([np.zeros(len(scores_n)), np.ones(len(scores_a))])
        scores = np.concatenate([scores_n, scores_a])
        auroc = roc_auc_score(labels, scores)
        res_base[cat] = auroc
        aurocs_base.append(auroc)
    res_base["MEAN"] = np.mean(aurocs_base)
    all_results["baseline"] = {"mean": res_base["MEAN"], "std": 0.0, "per_seed": [res_base]}
    print(f"  MEAN: {res_base['MEAN']*100:.1f}%")

    # NSP with varying N_pairs
    for n_pairs in n_pairs_list:
        label = f"N={n_pairs}" if n_pairs is not None else "N=full"
        print(f"\n  --- {label} (K={K}) ---")

        seed_results = []
        for seed in seeds:
            res = evaluate_nsp(
                dataset, all_shifts, K=K, n_pairs=n_pairs, seed=seed,
                global_nuisance=True,
            )
            seed_results.append(res)

        means = [r["MEAN"] for r in seed_results]
        mean_val = np.mean(means)
        std_val = np.std(means)
        all_results[label] = {
            "mean": float(mean_val),
            "std": float(std_val),
            "per_seed": seed_results,
        }
        print(f"  MEAN: {mean_val*100:.1f}% ± {std_val*100:.1f}%")

    # Per-category mode (for comparison)
    print(f"\n  --- Per-category NSP (N=full, K={K}) ---")
    res_percat = evaluate_nsp(
        dataset, all_shifts, K=K, n_pairs=None, seed=42, global_nuisance=False,
    )
    all_results["percat_full"] = {"mean": res_percat["MEAN"], "std": 0.0, "per_seed": [res_percat]}
    print(f"  MEAN: {res_percat['MEAN']*100:.1f}%")

    # Summary
    print("\n" + "=" * 60)
    print("SATURATION CURVE")
    print("=" * 60)
    print(f"\n  {'N_pairs':<12s} {'AD2 I-AUROC':>14s} {'Std':>8s} {'% of Full':>10s}")
    print("  " + "-" * 46)

    full_mean = all_results["N=full"]["mean"]
    base_mean = all_results["baseline"]["mean"]

    for label in ["baseline"] + [f"N={n}" if n is not None else "N=full" for n in n_pairs_list]:
        r = all_results[label]
        pct_full = (r["mean"] - base_mean) / (full_mean - base_mean) * 100 if full_mean > base_mean else 0
        print(f"  {label:<12s} {r['mean']*100:>12.1f}% {r['std']*100:>7.1f}% {pct_full:>9.0f}%")

    print(f"\n  Per-category NSP (full): {all_results['percat_full']['mean']*100:.1f}%")

    # Save
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "minimal_calibration_results.json"

    # Convert numpy to float for JSON serialization
    serializable = {}
    for k, v in all_results.items():
        serializable[k] = {
            "mean": float(v["mean"]),
            "std": float(v["std"]),
            "per_seed_means": [float(r["MEAN"]) for r in v["per_seed"]],
        }

    with open(result_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\n  Saved to {result_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Minimal Calibration Experiment")
    parser.add_argument("--data_root", type=str, default="/home/hun/Volume/DATA/mvtec_ad_2")
    parser.add_argument("--model_name", type=str, default="dinov2_vitb14")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--layer", type=int, default=8)
    parser.add_argument("--K", type=int, default=100)
    parser.add_argument("--max_per_cat", type=int, default=200)
    parser.add_argument("--output_dir", type=str, default="results/minimal_calibration")
    args = parser.parse_args()
    run(args)
