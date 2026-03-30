#!/usr/bin/env python3
"""SAPP: Spectral Anomaly-Preserving Projection.

Variance-gated nuisance removal that protects anomaly-sensitive directions.
Key insight: Normal covariance's low-variance directions are anomaly-sensitive.
Nuisance directions overlapping these should be preserved, not removed.

GO/NO-GO test: compute omega_k distribution to verify bimodality.
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
from tqdm import tqdm


# ─── Feature Extraction (reuse from phase3) ────────────────────────

class DINOv2Extractor:
    def __init__(self, model_name: str = "dinov2_vitb14", device: str = "cuda", target_layer: int = -1):
        self.device = device
        self.model = torch.hub.load("facebookresearch/dinov2", model_name).to(device).eval()
        self.n_layers = len(self.model.blocks)
        self.target_layer = target_layer if target_layer >= 0 else self.n_layers + target_layer
        self.feature = None
        self.model.blocks[self.target_layer].register_forward_hook(self._hook)
        self.transform = transforms.Compose([
            transforms.Resize(518, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(518),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def _hook(self, module, input, output):
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


# ─── Data Loading ───────────────────────────────────────────────────

def collect_shift_vectors(data_root: str, extractor, max_pairs: int = 300) -> np.ndarray:
    """Collect shift vectors from paired regular↔shifted images."""
    root = Path(data_root)
    categories = sorted(d.name for d in root.iterdir() if d.is_dir() and (d / "test_public").is_dir())

    vectors = []
    for cat in categories:
        good_dir = root / cat / "test_public" / "good"
        if not good_dir.is_dir():
            continue
        groups: dict[str, dict[str, Path]] = defaultdict(dict)
        for p in sorted(good_dir.glob("*.png")):
            parts = p.stem.split("_", 1)
            if len(parts) == 2:
                groups[parts[0]][parts[1]] = p
        for obj_idx, conds in groups.items():
            if "regular" not in conds:
                continue
            f_reg = extractor.extract(Image.open(conds["regular"]).convert("RGB"))
            for cond, path in conds.items():
                if cond == "regular":
                    continue
                f_shift = extractor.extract(Image.open(path).convert("RGB"))
                vectors.append(f_shift - f_reg)
            if len(vectors) >= max_pairs:
                break
        if len(vectors) >= max_pairs:
            break
    return np.stack(vectors)


def extract_dataset(data_root: str, extractor, max_per_cat: int = 200) -> dict:
    """Extract train/test features per category."""
    root = Path(data_root)
    categories = sorted(d.name for d in root.iterdir() if d.is_dir() and (d / "train").is_dir())
    dataset = {}
    for cat in categories:
        cat_data = {"train_normal": [], "test_normal": [], "test_anomaly": []}
        train_good = root / cat / "train" / "good"
        if train_good.is_dir():
            for p in sorted(train_good.iterdir())[:max_per_cat]:
                if p.suffix.lower() in (".png", ".jpg", ".jpeg"):
                    cat_data["train_normal"].append(extractor.extract(Image.open(p).convert("RGB")))
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
                        cat_data[label].append(extractor.extract(Image.open(p).convert("RGB")))
        for k in cat_data:
            cat_data[k] = np.stack(cat_data[k]) if cat_data[k] else np.array([]).reshape(0, 1536)
        dataset[cat] = cat_data
        print(f"  {cat}: train={len(cat_data['train_normal'])}, test_n={len(cat_data['test_normal'])}, test_a={len(cat_data['test_anomaly'])}")
    return dataset


# ─── SAPP Core ──────────────────────────────────────────────────────

def estimate_nuisance_directions(shift_vectors: np.ndarray, n_components: int) -> tuple[np.ndarray, np.ndarray]:
    """PCA on shift vectors → nuisance directions + eigenvalues."""
    centered = shift_vectors - shift_vectors.mean(axis=0)
    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    idx = np.argsort(eigenvalues)[::-1]
    return eigenvectors[:, idx[:n_components]], eigenvalues[idx]


def compute_omega(nuisance_dirs: np.ndarray, train_features: np.ndarray) -> np.ndarray:
    """Compute anomaly-sensitivity score ω_k for each nuisance direction.

    ω_k = n_k^T @ Σ_train^{-1} @ n_k
    High ω → low train variance along n_k → anomaly-sensitive → PROTECT
    Low ω → high train variance along n_k → anomaly-insensitive → REMOVE OK
    """
    centered = train_features - train_features.mean(axis=0)
    cov = np.cov(centered.T)
    cov += np.eye(cov.shape[0]) * 1e-6
    cov_inv = np.linalg.inv(cov)

    K = nuisance_dirs.shape[1]
    omega = np.zeros(K)
    for k in range(K):
        n_k = nuisance_dirs[:, k]
        omega[k] = n_k @ cov_inv @ n_k
    return omega


def compute_gating_weights(omega: np.ndarray, alpha: float = 5.0, tau: float = None) -> np.ndarray:
    """Sigmoid gating: g_k = σ(-α * (ω_k - τ)).

    g_k ≈ 1 when ω_k << τ (anomaly-insensitive → remove)
    g_k ≈ 0 when ω_k >> τ (anomaly-sensitive → protect)
    """
    if tau is None:
        tau = np.median(omega)
    return 1.0 / (1.0 + np.exp(alpha * (omega - tau)))


def sapp_project(features: np.ndarray, nuisance_dirs: np.ndarray, gating: np.ndarray) -> np.ndarray:
    """Gated nuisance projection: f_clean = f - Σ_k g_k * (f·n_k) * n_k."""
    projections = features @ nuisance_dirs  # (N, K)
    weighted = projections * gating[np.newaxis, :]  # (N, K)
    return features - weighted @ nuisance_dirs.T


def nsp_project(features: np.ndarray, nuisance_dirs: np.ndarray) -> np.ndarray:
    """Standard hard NSP projection (baseline)."""
    proj_matrix = nuisance_dirs @ nuisance_dirs.T
    return features - features @ proj_matrix


def mahalanobis_score(train: np.ndarray, test: np.ndarray) -> np.ndarray:
    """Mahalanobis anomaly score."""
    mean = train.mean(axis=0)
    centered = train - mean
    cov = np.cov(centered.T) + np.eye(train.shape[1]) * 1e-6
    cov_inv = np.linalg.inv(cov)
    test_c = test - mean
    return np.sqrt(np.sum(test_c @ cov_inv * test_c, axis=1))


# ─── Evaluation ─────────────────────────────────────────────────────

def evaluate_method(dataset: dict, nuisance_dirs: np.ndarray, method: str = "nsp",
                    omega: np.ndarray = None, alpha: float = 5.0, tau: float = None) -> dict:
    """Evaluate NSP or SAPP across all categories."""
    results = {}
    aurocs = []

    for cat in sorted(dataset.keys()):
        train = dataset[cat]["train_normal"]
        test_n = dataset[cat]["test_normal"]
        test_a = dataset[cat]["test_anomaly"]
        if len(train) == 0 or len(test_n) == 0 or len(test_a) == 0:
            continue

        if method == "sapp" and omega is not None:
            gating = compute_gating_weights(omega, alpha=alpha, tau=tau)
            train_p = sapp_project(train, nuisance_dirs, gating)
            test_n_p = sapp_project(test_n, nuisance_dirs, gating)
            test_a_p = sapp_project(test_a, nuisance_dirs, gating)
        elif method == "nsp":
            train_p = nsp_project(train, nuisance_dirs)
            test_n_p = nsp_project(test_n, nuisance_dirs)
            test_a_p = nsp_project(test_a, nuisance_dirs)
        else:  # baseline
            train_p, test_n_p, test_a_p = train, test_n, test_a

        scores_n = mahalanobis_score(train_p, test_n_p)
        scores_a = mahalanobis_score(train_p, test_a_p)
        labels = np.concatenate([np.zeros(len(scores_n)), np.ones(len(scores_a))])
        scores = np.concatenate([scores_n, scores_a])
        auroc = roc_auc_score(labels, scores)
        results[cat] = auroc
        aurocs.append(auroc)

    results["MEAN"] = np.mean(aurocs)
    return results


# ─── Main ───────────────────────────────────────────────────────────

def run(args):
    print("=" * 60)
    print("SAPP: Spectral Anomaly-Preserving Projection")
    print("=" * 60)

    extractor = DINOv2Extractor(model_name=args.model_name, device=args.device, target_layer=args.layer)
    print(f"\n[1/5] DINOv2 Layer {args.layer} loaded")

    # Shift vectors
    print(f"\n[2/5] Collecting shift vectors...")
    shift_vecs = collect_shift_vectors(args.data_root_original, extractor, max_pairs=300)
    print(f"  {len(shift_vecs)} shift vectors collected")
    nuisance_dirs, shift_eigenvalues = estimate_nuisance_directions(shift_vecs, args.n_components)

    # Dataset features
    print(f"\n[3/5] Extracting features...")
    print("  --- MVTec AD 2 ---")
    ds_ad2 = extract_dataset(args.data_root_ad2, extractor, max_per_cat=200)
    ds_ad1 = None
    if args.data_root_ad1:
        print("  --- MVTec AD ---")
        ds_ad1 = extract_dataset(args.data_root_ad1, extractor, max_per_cat=200)

    # Compute omega using ALL train features pooled
    print(f"\n[4/5] Computing omega (anomaly sensitivity per nuisance direction)...")
    all_train = np.concatenate([d["train_normal"] for d in ds_ad2.values() if len(d["train_normal"]) > 0])
    omega = compute_omega(nuisance_dirs, all_train)

    # ─── GO/NO-GO Analysis ───
    print(f"\n  === OMEGA DISTRIBUTION (GO/NO-GO) ===")
    print(f"  ω range: [{omega.min():.2f}, {omega.max():.2f}]")
    print(f"  ω mean: {omega.mean():.2f}, median: {np.median(omega):.2f}, std: {omega.std():.2f}")
    print(f"  ω quartiles: Q1={np.percentile(omega, 25):.2f}, Q2={np.percentile(omega, 50):.2f}, Q3={np.percentile(omega, 75):.2f}")

    # Bimodality check: ratio of high-omega to low-omega
    threshold = np.median(omega)
    n_high = (omega > threshold * 1.5).sum()
    n_low = (omega < threshold * 0.5).sum()
    print(f"  High-ω (>1.5×median): {n_high}/{len(omega)} ({n_high/len(omega)*100:.0f}%)")
    print(f"  Low-ω (<0.5×median): {n_low}/{len(omega)} ({n_low/len(omega)*100:.0f}%)")
    print(f"  Spread ratio (max/min): {omega.max()/omega.min():.1f}x")

    go = omega.max() / omega.min() > 5.0
    print(f"\n  GO/NO-GO: {'✅ GO' if go else '❌ NO-GO'} (spread ratio {'>' if go else '<'} 5x)")

    # ─── Evaluate all methods ───
    print(f"\n[5/5] Evaluating...")

    for ds_name, ds in [("MVTec AD 2", ds_ad2), ("MVTec AD", ds_ad1)]:
        if ds is None:
            continue
        print(f"\n  === {ds_name} ===")

        r_base = evaluate_method(ds, nuisance_dirs, method="baseline")
        r_nsp = evaluate_method(ds, nuisance_dirs, method="nsp")

        # SAPP with different alpha values
        sapp_results = {}
        for alpha in [1.0, 3.0, 5.0, 10.0]:
            r_sapp = evaluate_method(ds, nuisance_dirs, method="sapp", omega=omega, alpha=alpha)
            sapp_results[alpha] = r_sapp

        # Print comparison
        print(f"  {'Category':<20s} {'Baseline':>9s} {'NSP':>9s}", end="")
        for alpha in [1.0, 3.0, 5.0, 10.0]:
            print(f" {'SAPP-α'+str(alpha):>10s}", end="")
        print()
        print("  " + "-" * 80)

        for cat in sorted(r_base.keys()):
            b = r_base[cat] * 100
            n = r_nsp[cat] * 100
            print(f"  {cat:<20s} {b:>8.1f}% {n:>8.1f}%", end="")
            for alpha in [1.0, 3.0, 5.0, 10.0]:
                s = sapp_results[alpha][cat] * 100
                delta = s - n
                marker = "+" if delta > 0.5 else ("-" if delta < -0.5 else "=")
                print(f" {s:>8.1f}%{marker}", end="")
            print()

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    save_data = {
        "omega": omega.tolist(),
        "omega_stats": {
            "min": float(omega.min()), "max": float(omega.max()),
            "mean": float(omega.mean()), "median": float(np.median(omega)),
            "std": float(omega.std()), "spread_ratio": float(omega.max() / omega.min()),
        },
        "go_nogo": bool(go),
        "layer": args.layer,
        "n_components": args.n_components,
    }
    with open(output_dir / f"sapp_omega_L{args.layer}_K{args.n_components}.json", "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\n  Saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SAPP Experiment")
    parser.add_argument("--data_root_original", type=str, default="/home/hun/Volume/DATA/mvtec_ad_2")
    parser.add_argument("--data_root_ad2", type=str, default="/home/hun/Volume/DATA/mvtec_ad_2_compat")
    parser.add_argument("--data_root_ad1", type=str, default=None)
    parser.add_argument("--model_name", type=str, default="dinov2_vitb14")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--layer", type=int, default=11)
    parser.add_argument("--n_components", type=int, default=100)
    parser.add_argument("--output_dir", type=str, default="results/sapp")
    args = parser.parse_args()
    run(args)
