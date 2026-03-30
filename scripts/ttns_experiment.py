#!/usr/bin/env python3
"""TTNS: Test-Time Nuisance Subspace estimation for robust AD.

Core idea: ΔΣ = Σ_test - Σ_train → eigendecomposition →
positive eigenvalue directions = shift-induced variance = nuisance.
Hard orthogonal projection (same as NSP) on estimated nuisance basis.
No paired data needed — works with train normal + test batch only.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score
from torchvision import transforms
from tqdm import tqdm


class DINOv2Extractor:
    def __init__(self, model_name: str = "dinov2_vitb14", device: str = "cuda", target_layer: int = -1):
        self.device = device
        self.model = torch.hub.load("facebookresearch/dinov2", model_name).to(device).eval()
        n_layers = len(self.model.blocks)
        self.target_layer = target_layer if target_layer >= 0 else n_layers + target_layer
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


def extract_dataset(data_root: str, extractor, max_per_cat: int = 200) -> dict:
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


# ─── Nuisance Estimation Methods ───────────────────────────────────

def select_likely_normals(
    train: np.ndarray,
    test_all: np.ndarray,
    keep_ratio: float = 0.5,
    n_iters: int = 2,
) -> np.ndarray:
    """Iteratively keep low-score test samples as likely normals.

    This targets anomaly contamination in test-time statistics.
    """
    if len(test_all) == 0:
        return test_all

    keep_count = max(1, int(len(test_all) * keep_ratio))
    selected = test_all

    mu = train.mean(axis=0)
    cov = np.cov((train - mu).T) + np.eye(train.shape[1]) * 1e-6
    cov_inv = np.linalg.inv(cov)

    for _ in range(max(1, n_iters)):
        centered = selected - mu
        scores = np.sqrt(np.sum(centered @ cov_inv * centered, axis=1))
        idx = np.argsort(scores)
        selected = selected[idx[: min(keep_count, len(selected))]]

        mu = selected.mean(axis=0)
        cov = np.cov((selected - mu).T) + np.eye(train.shape[1]) * 1e-6
        cov_inv = np.linalg.inv(cov)

    return selected

def ttns_covariance_shift(train: np.ndarray, test_all: np.ndarray, robust: bool = True) -> np.ndarray:
    """Estimate nuisance from ΔΣ = Σ_test - Σ_train.

    Returns nuisance directions (positive eigenvalue eigenvectors of ΔΣ).
    """
    mu_train = train.mean(axis=0)
    sigma_train = np.cov(train.T) + np.eye(train.shape[1]) * 1e-6

    if robust and len(test_all) > 20:
        test_clean = select_likely_normals(train, test_all, keep_ratio=0.5, n_iters=2)
        print(f"    robust selection: kept {len(test_clean)}/{len(test_all)} likely-normal samples")
    else:
        test_clean = test_all

    sigma_test = np.cov(test_clean.T) + np.eye(test_clean.shape[1]) * 1e-6
    delta_sigma = sigma_test - sigma_train

    eigenvalues, eigenvectors = np.linalg.eigh(delta_sigma)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Keep only positive eigenvalue directions (variance INCREASED)
    positive_mask = eigenvalues > 0
    nuisance_dirs = eigenvectors[:, positive_mask]
    pos_eigenvalues = eigenvalues[positive_mask]

    n_pos = positive_mask.sum()
    if n_pos > 0:
        explained = pos_eigenvalues.sum() / np.abs(eigenvalues).sum()
        print(f"    ΔΣ: {n_pos} positive directions, explain {explained:.1%} of total |ΔΣ|")
    else:
        print(f"    ΔΣ: no positive eigenvalues — no shift detected")

    return nuisance_dirs


def ttns_mean_shift(train: np.ndarray, test_all: np.ndarray, robust: bool = True) -> np.ndarray:
    """Estimate nuisance from mean shift direction only (rank-1)."""
    mu_train = train.mean(axis=0)

    if robust and len(test_all) > 20:
        test_clean = select_likely_normals(train, test_all, keep_ratio=0.5, n_iters=2)
        mu_test = test_clean.mean(axis=0)
        print(f"    robust selection: kept {len(test_clean)}/{len(test_all)} likely-normal samples")
    else:
        mu_test = test_all.mean(axis=0)

    d = mu_test - mu_train
    norm = np.linalg.norm(d)
    if norm < 1e-8:
        return np.array([]).reshape(train.shape[1], 0)
    return (d / norm).reshape(-1, 1)


def ttns_combined(train: np.ndarray, test_all: np.ndarray, robust: bool = True) -> np.ndarray:
    """Mean shift + covariance shift combined."""
    mean_dir = ttns_mean_shift(train, test_all, robust=robust)
    cov_dirs = ttns_covariance_shift(train, test_all, robust=robust)

    if mean_dir.shape[1] == 0:
        return cov_dirs
    if cov_dirs.shape[1] == 0:
        return mean_dir

    # Combine and re-orthogonalize
    combined = np.hstack([mean_dir, cov_dirs])
    U, S, Vt = np.linalg.svd(combined, full_matrices=False)
    # Keep directions with significant singular values
    keep = S > S[0] * 0.01
    return U[:, keep]


def random_projection(dim: int, k: int) -> np.ndarray:
    """Random nuisance basis (control experiment)."""
    rng = np.random.RandomState(42)
    dirs = rng.randn(dim, k)
    Q, _ = np.linalg.qr(dirs)
    return Q[:, :k]


def adabn_normalize(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """AdaBN: replace test statistics with test batch statistics."""
    mu_test = test.mean(axis=0)
    std_test = test.std(axis=0) + 1e-8
    mu_train = train.mean(axis=0)
    std_train = train.std(axis=0) + 1e-8

    test_normed = (test - mu_test) / std_test * std_train + mu_train
    return train, test_normed


# ─── Scoring & Evaluation ──────────────────────────────────────────

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


def evaluate(dataset: dict, method: str, **kwargs) -> dict:
    results = {}
    aurocs = []
    for cat in sorted(dataset.keys()):
        train = dataset[cat]["train_normal"]
        test_n = dataset[cat]["test_normal"]
        test_a = dataset[cat]["test_anomaly"]
        if len(train) == 0 or len(test_n) == 0 or len(test_a) == 0:
            continue

        test_all = np.vstack([test_n, test_a])

        if method == "baseline":
            train_p, test_n_p, test_a_p = train, test_n, test_a

        elif method == "ttns_cov":
            nuisance = ttns_covariance_shift(train, test_all, robust=kwargs.get("robust", True))
            train_p = hard_project(train, nuisance)
            test_n_p = hard_project(test_n, nuisance)
            test_a_p = hard_project(test_a, nuisance)

        elif method == "ttns_mean":
            nuisance = ttns_mean_shift(train, test_all, robust=kwargs.get("robust", True))
            train_p = hard_project(train, nuisance)
            test_n_p = hard_project(test_n, nuisance)
            test_a_p = hard_project(test_a, nuisance)

        elif method == "ttns_combined":
            nuisance = ttns_combined(train, test_all, robust=kwargs.get("robust", True))
            train_p = hard_project(train, nuisance)
            test_n_p = hard_project(test_n, nuisance)
            test_a_p = hard_project(test_a, nuisance)

        elif method == "random":
            k = kwargs.get("k", 100)
            nuisance = random_projection(train.shape[1], k)
            train_p = hard_project(train, nuisance)
            test_n_p = hard_project(test_n, nuisance)
            test_a_p = hard_project(test_a, nuisance)

        elif method == "adabn":
            train_p, test_n_p = adabn_normalize(train, test_n)
            _, test_a_p = adabn_normalize(train, test_a)

        else:
            raise ValueError(f"Unknown method: {method}")

        scores_n = mahalanobis_score(train_p, test_n_p)
        scores_a = mahalanobis_score(train_p, test_a_p)
        labels = np.concatenate([np.zeros(len(scores_n)), np.ones(len(scores_a))])
        scores = np.concatenate([scores_n, scores_a])
        auroc = roc_auc_score(labels, scores)
        results[cat] = auroc
        aurocs.append(auroc)

    results["MEAN"] = np.mean(aurocs) if aurocs else 0.0
    return results


# ─── Main ───────────────────────────────────────────────────────────

def run(args):
    print("=" * 60)
    print("TTNS: Test-Time Nuisance Subspace Estimation")
    print("=" * 60)

    extractor = DINOv2Extractor(model_name=args.model_name, device=args.device, target_layer=args.layer)
    print(f"\n[1/3] DINOv2 Layer {args.layer}")

    print(f"\n[2/3] Extracting features...")
    print("  --- MVTec AD 2 ---")
    ds_ad2 = extract_dataset(args.data_root_ad2, extractor, max_per_cat=args.max_per_cat)
    ds_ad1 = None
    if args.data_root_ad1:
        print("  --- MVTec AD ---")
        ds_ad1 = extract_dataset(args.data_root_ad1, extractor, max_per_cat=args.max_per_cat)

    print(f"\n[3/3] Evaluating all methods...")

    methods = [
        ("baseline", {}),
        ("ttns_mean", {"robust": True}),
        ("ttns_cov", {"robust": True}),
        ("ttns_combined", {"robust": True}),
        ("random", {"k": 100}),
        ("adabn", {}),
    ]

    for ds_name, ds in [("MVTec AD 2", ds_ad2), ("MVTec AD", ds_ad1)]:
        if ds is None:
            continue
        print(f"\n  === {ds_name} ===")

        # Header
        method_names = [m[0] for m in methods]
        header = f"  {'Category':<20s}" + "".join(f" {m:>12s}" for m in method_names)
        print(header)
        print("  " + "-" * (20 + 13 * len(methods)))

        all_results = {}
        for method_name, kwargs in methods:
            print(f"  [running {method_name}...]", end="", flush=True)
            all_results[method_name] = evaluate(ds, method_name, **kwargs)
            print(f" done ({all_results[method_name]['MEAN']*100:.1f}%)")

        # Per-category table
        for cat in sorted(k for k in all_results["baseline"].keys() if k != "MEAN"):
            row = f"  {cat:<20s}"
            for method_name, _ in methods:
                val = all_results[method_name].get(cat, 0) * 100
                row += f" {val:>11.1f}%"
            print(row)

        # MEAN row
        row = f"  {'MEAN':<20s}"
        for method_name, _ in methods:
            val = all_results[method_name]["MEAN"] * 100
            row += f" {val:>11.1f}%"
        print(row)

    # Save
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "ttns_results.json", "w") as f:
        json.dump({"methods": {m: all_results.get(m, {}) for m, _ in methods}}, f, indent=2, default=str)
    print(f"\n  Saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root_ad2", type=str, default="/home/hun/Volume/DATA/mvtec_ad_2_compat")
    parser.add_argument("--data_root_ad1", type=str, default=None)
    parser.add_argument("--model_name", type=str, default="dinov2_vitb14")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--layer", type=int, default=8)
    parser.add_argument("--max_per_cat", type=int, default=200)
    parser.add_argument("--output_dir", type=str, default="results/ttns")
    args = parser.parse_args()
    run(args)
