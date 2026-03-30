#!/usr/bin/env python3
"""NN Pseudo-Pairing Experiment: Unpaired nuisance estimation via NN matching.

Core idea:
  1. For each test image, find its NN in train at L8 (robust layer)
  2. pseudo_shift = f(test) - f(NN) → approximates real shift for normals
  3. Robust PCA on pseudo_shifts → nuisance subspace
  4. Hard projection → Mahalanobis scoring (same as NSP)

Comparison matrix:
  - Baseline (no projection)
  - NSP oracle (paired shift vectors)
  - NN Pseudo-Pairing (unpaired, L8 NN matching)
  - NN Pseudo-Pairing + robust filtering (remove outlier pseudo_shifts)
  - Random projection (control)
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

class MultiLayerExtractor:
    def __init__(
        self,
        model_name: str = "dinov2_vitb14",
        device: str = "cuda",
        target_layers: tuple[int, ...] = (8, 11),
    ) -> None:
        self.device = device
        self.model = torch.hub.load("facebookresearch/dinov2", model_name).to(device).eval()
        self.target_layers = target_layers
        self.features: dict[int, torch.Tensor] = {}
        for layer_idx in target_layers:
            self.model.blocks[layer_idx].register_forward_hook(self._make_hook(layer_idx))
        self.transform = transforms.Compose([
            transforms.Resize(518, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(518),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def _make_hook(self, layer_idx: int):
        def hook(module, inp, output):
            self.features[layer_idx] = output.detach()
        return hook

    @torch.no_grad()
    def extract(self, img: Image.Image) -> dict[int, np.ndarray]:
        x = self.transform(img).unsqueeze(0).to(self.device)
        self.features.clear()
        _ = self.model(x)
        result = {}
        for layer_idx in self.target_layers:
            feat = self.features[layer_idx][0]
            cls_token = feat[0]
            patch_mean = feat[1:].mean(dim=0)
            combined = torch.cat([cls_token, patch_mean])
            combined = combined / (combined.norm() + 1e-8)
            result[layer_idx] = combined.cpu().numpy()
        return result


# ─── Data Collection ──────────────────────────────────────────────

def collect_all_features(
    data_root: str,
    extractor: MultiLayerExtractor,
    layer: int,
    max_per_cat: int = 200,
) -> dict[str, dict[str, np.ndarray]]:
    """Collect train/test features at specified layer. Returns per-category data."""
    root = Path(data_root)
    categories = sorted(
        d.name for d in root.iterdir()
        if d.is_dir() and (d / "train").is_dir()
    )
    dataset = {}
    for cat in categories:
        cat_data = {"train_normal": [], "test_normal": [], "test_anomaly": []}

        # Train normal
        train_good = root / cat / "train" / "good"
        if train_good.is_dir():
            for p in sorted(train_good.iterdir())[:max_per_cat]:
                if p.suffix.lower() in (".png", ".jpg", ".jpeg"):
                    feats = extractor.extract(Image.open(p).convert("RGB"))
                    cat_data["train_normal"].append(feats[layer])

        # Test (good = normal, bad = anomaly)
        test_dir = root / cat / "test"
        if not test_dir.is_dir():
            test_dir = root / cat / "test_public"
        if test_dir.is_dir():
            for sub in sorted(test_dir.iterdir()):
                if not sub.is_dir():
                    continue
                label = "test_normal" if sub.name == "good" else "test_anomaly"
                for p in sorted(sub.iterdir())[:max_per_cat]:
                    if p.suffix.lower() in (".png", ".jpg", ".jpeg"):
                        feats = extractor.extract(Image.open(p).convert("RGB"))
                        cat_data[label].append(feats[layer])

        for k in cat_data:
            cat_data[k] = np.stack(cat_data[k]) if cat_data[k] else np.zeros((0, 1536))

        print(f"  {cat}: train={len(cat_data['train_normal'])}, "
              f"test_n={len(cat_data['test_normal'])}, test_a={len(cat_data['test_anomaly'])}")
        dataset[cat] = cat_data
    return dataset


def collect_oracle_shifts(
    data_root: str,
    extractor: MultiLayerExtractor,
    layer: int,
    max_pairs: int = 500,
) -> dict[str, np.ndarray]:
    """Collect oracle paired shift vectors per category."""
    root = Path(data_root)
    oracle = {}
    categories = sorted(
        d.name for d in root.iterdir()
        if d.is_dir() and (d / "test_public").is_dir()
    )
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
                shifts.append(feat_shift[layer] - feat_reg[layer])
            if len(shifts) >= max_pairs:
                break
        oracle[cat] = np.stack(shifts) if shifts else np.zeros((0, 1536))
    return oracle


# ─── Nuisance Estimation Methods ─────────────────────────────────

def nn_pseudo_shift_nuisance(
    train: np.ndarray,
    test_all: np.ndarray,
    K: int = 100,
    robust: bool = True,
    robust_percentile: float = 80.0,
) -> np.ndarray:
    """Estimate nuisance subspace via NN pseudo-pairing.

    1. For each test sample, find NN in train
    2. pseudo_shift = f(test) - f(NN_train)
    3. Optional robust filtering: keep pseudo_shifts with small norm (likely normals)
    4. PCA on pseudo_shifts → top-K = nuisance basis

    Returns (dim, K) orthonormal nuisance basis.
    """
    # NN matching
    nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
    nn.fit(train)
    distances, indices = nn.kneighbors(test_all)
    nn_idx = indices.ravel()

    # Pseudo-shift vectors
    pseudo_shifts = test_all - train[nn_idx]

    if robust:
        # Filter: keep pseudo_shifts with norm below percentile threshold
        # Anomalies tend to have large pseudo_shift norms (shift + anomaly signal)
        norms = np.linalg.norm(pseudo_shifts, axis=1)
        threshold = np.percentile(norms, robust_percentile)
        mask = norms <= threshold
        pseudo_shifts_filtered = pseudo_shifts[mask]
        n_kept = mask.sum()
        print(f"    robust: kept {n_kept}/{len(pseudo_shifts)} "
              f"(threshold={threshold:.4f}, percentile={robust_percentile})")
    else:
        pseudo_shifts_filtered = pseudo_shifts

    if len(pseudo_shifts_filtered) < K:
        print(f"    WARNING: only {len(pseudo_shifts_filtered)} pseudo_shifts, need {K}")
        return np.zeros((train.shape[1], 0))

    # PCA on pseudo_shifts → nuisance basis
    centered = pseudo_shifts_filtered - pseudo_shifts_filtered.mean(axis=0)
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    basis = Vt[:K].T  # (dim, K)

    total_var = (S ** 2).sum()
    explained = (S[:K] ** 2).sum() / total_var
    print(f"    nuisance basis: top-{K} explain {explained:.1%} of pseudo-shift variance")

    return basis


def oracle_nsp_nuisance(
    oracle_shifts: np.ndarray,
    K: int = 100,
) -> np.ndarray:
    """Oracle NSP: PCA on paired shift vectors."""
    centered = oracle_shifts - oracle_shifts.mean(axis=0)
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    return Vt[:K].T


def random_nuisance(dim: int, K: int) -> np.ndarray:
    """Random basis (control)."""
    rng = np.random.RandomState(42)
    dirs = rng.randn(dim, K)
    Q, _ = np.linalg.qr(dirs)
    return Q[:, :K]


# ─── Scoring ─────────────────────────────────────────────────────

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

def evaluate_method(
    dataset: dict[str, dict[str, np.ndarray]],
    oracle_shifts: dict[str, np.ndarray],
    method: str,
    K: int = 100,
    robust: bool = True,
    robust_percentile: float = 80.0,
) -> dict[str, float]:
    """Evaluate a method across all categories."""
    results = {}
    aurocs = []

    for cat in sorted(dataset.keys()):
        train = dataset[cat]["train_normal"]
        test_n = dataset[cat]["test_normal"]
        test_a = dataset[cat]["test_anomaly"]

        if len(train) == 0 or len(test_n) == 0 or len(test_a) == 0:
            continue

        test_all = np.vstack([test_n, test_a])

        # Estimate nuisance
        if method == "baseline":
            nuisance = np.zeros((train.shape[1], 0))
        elif method == "nn_pseudo":
            nuisance = nn_pseudo_shift_nuisance(
                train, test_all, K=K, robust=robust, robust_percentile=robust_percentile,
            )
        elif method == "nn_pseudo_norobust":
            nuisance = nn_pseudo_shift_nuisance(
                train, test_all, K=K, robust=False,
            )
        elif method == "oracle_nsp":
            cat_shifts = oracle_shifts.get(cat, np.zeros((0, train.shape[1])))
            if len(cat_shifts) == 0:
                nuisance = np.zeros((train.shape[1], 0))
            else:
                nuisance = oracle_nsp_nuisance(cat_shifts, K=K)
        elif method == "random":
            nuisance = random_nuisance(train.shape[1], K)
        else:
            raise ValueError(f"Unknown method: {method}")

        # Project + score
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
    print("NN Pseudo-Pairing Experiment: Unpaired Nuisance Projection")
    print("=" * 60)

    extractor = MultiLayerExtractor(
        model_name=args.model_name,
        device=args.device,
        target_layers=(8, 11),
    )

    layer = args.layer
    print(f"\n[1/3] Extracting features at Layer {layer}...")
    dataset = collect_all_features(args.data_root, extractor, layer=layer, max_per_cat=args.max_per_cat)

    print(f"\n[2/3] Collecting oracle shift vectors...")
    oracle_shifts = collect_oracle_shifts(args.data_root, extractor, layer=layer, max_pairs=args.max_pairs)

    print(f"\n[3/3] Evaluating methods...")

    methods = [
        ("baseline", {}),
        ("oracle_nsp", {"K": args.K}),
        ("nn_pseudo", {"K": args.K, "robust": True, "robust_percentile": 80}),
        ("nn_pseudo", {"K": args.K, "robust": True, "robust_percentile": 90}),
        ("nn_pseudo_norobust", {"K": args.K}),
        ("random", {"K": args.K}),
    ]

    method_labels = [
        "baseline",
        f"oracle_nsp_K{args.K}",
        f"nn_pseudo_K{args.K}_rob80",
        f"nn_pseudo_K{args.K}_rob90",
        f"nn_pseudo_K{args.K}_norobust",
        f"random_K{args.K}",
    ]

    all_results = {}
    for (method, kwargs), label in zip(methods, method_labels):
        print(f"\n  --- {label} ---")
        res = evaluate_method(dataset, oracle_shifts, method, **kwargs)
        all_results[label] = res
        print(f"  MEAN I-AUROC: {res['MEAN']*100:.1f}%")

    # K sweep for best method
    if args.k_sweep:
        print(f"\n  --- K sweep (nn_pseudo robust80) ---")
        for k_val in [30, 50, 100, 150, 200]:
            label = f"nn_pseudo_K{k_val}_rob80"
            print(f"\n  --- {label} ---")
            res = evaluate_method(
                dataset, oracle_shifts, "nn_pseudo",
                K=k_val, robust=True, robust_percentile=80,
            )
            all_results[label] = res
            print(f"  MEAN I-AUROC: {res['MEAN']*100:.1f}%")

    # Summary table
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY (Layer {})".format(layer))
    print("=" * 60)

    header = f"  {'Category':<16s}"
    for label in all_results:
        short = label.replace("nn_pseudo_", "NNP_").replace("oracle_nsp_", "NSP_").replace("baseline", "base")
        header += f" {short:>12s}"
    print(header)
    print("  " + "-" * (16 + 13 * len(all_results)))

    cats = sorted(k for k in next(iter(all_results.values())).keys() if k != "MEAN")
    for cat in cats:
        row = f"  {cat:<16s}"
        for label in all_results:
            val = all_results[label].get(cat, 0) * 100
            row += f" {val:>11.1f}%"
        print(row)

    row = f"  {'MEAN':<16s}"
    for label in all_results:
        val = all_results[label]["MEAN"] * 100
        row += f" {val:>11.1f}%"
    print(row)

    # Save
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "nn_pseudo_results.json"
    with open(result_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Saved to {result_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NN Pseudo-Pairing Experiment")
    parser.add_argument("--data_root", type=str, default="/home/hun/Volume/DATA/mvtec_ad_2")
    parser.add_argument("--model_name", type=str, default="dinov2_vitb14")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--layer", type=int, default=8)
    parser.add_argument("--K", type=int, default=100)
    parser.add_argument("--max_per_cat", type=int, default=200)
    parser.add_argument("--max_pairs", type=int, default=500)
    parser.add_argument("--k_sweep", action="store_true", help="Run K sweep")
    parser.add_argument("--output_dir", type=str, default="results/nn_pseudo")
    args = parser.parse_args()
    run(args)
