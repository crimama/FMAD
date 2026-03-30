#!/usr/bin/env python3
"""IFR: Inter-layer Feature Residual for robust AD.

Core idea: DINOv2's residual connections mean f_L11 = f_L8 + Δ_9 + Δ_10 + Δ_11.
Since L8 is robust (RelShift 0.11) and L11 is fragile (0.26),
Δ_{9-11} must contain nuisance information.

PCA on layer residuals → nuisance subspace → hard projection (same as NSP).
No paired data, no test statistics, no anomaly contamination.
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


class DINOv2MultiLayerExtractor:
    """Extract features from multiple DINOv2 layers simultaneously."""

    def __init__(
        self,
        model_name: str = "dinov2_vitb14",
        device: str = "cuda",
        target_layers: tuple[int, ...] = (8, 9, 10, 11),
    ):
        self.device = device
        self.model = torch.hub.load("facebookresearch/dinov2", model_name).to(device).eval()
        self.target_layers = target_layers
        self.features: dict[int, torch.Tensor] = {}

        for layer_idx in target_layers:
            self.model.blocks[layer_idx].register_forward_hook(
                self._make_hook(layer_idx)
            )

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
        """Extract CLS + patch_mean features at each target layer."""
        x = self.transform(img).unsqueeze(0).to(self.device)
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


def extract_dataset_multilayer(
    data_root: str,
    extractor: DINOv2MultiLayerExtractor,
    max_per_cat: int = 200,
) -> dict:
    """Extract multi-layer features for all categories."""
    root = Path(data_root)
    categories = sorted(
        d.name for d in root.iterdir()
        if d.is_dir() and (d / "train").is_dir()
    )

    dataset = {}
    for cat in categories:
        cat_data = {
            "train_normal": {L: [] for L in extractor.target_layers},
            "test_normal": {L: [] for L in extractor.target_layers},
            "test_anomaly": {L: [] for L in extractor.target_layers},
        }

        # Train normal — try train/good first, then train/ directly
        train_good = root / cat / "train" / "good"
        if not train_good.is_dir():
            # AD2 compat: train/good may be a broken symlink; try original structure
            train_good = root / cat / "train"
        if train_good.is_dir():
            imgs = sorted(p for p in train_good.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
            for p in imgs[:max_per_cat]:
                feats = extractor.extract(Image.open(p).convert("RGB"))
                for L in extractor.target_layers:
                    cat_data["train_normal"][L].append(feats[L])

        # Test — try test_public first (AD2), then test/
        test_dir = root / cat / "test_public"
        if not test_dir.is_dir():
            test_dir = root / cat / "test"
        if test_dir.is_dir():
            for sub in sorted(test_dir.iterdir()):
                if not sub.is_dir():
                    continue
                label = "test_normal" if sub.name == "good" else "test_anomaly"
                imgs = sorted(p for p in sub.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
                for p in imgs[:max_per_cat]:
                    feats = extractor.extract(Image.open(p).convert("RGB"))
                    for L in extractor.target_layers:
                        cat_data[label][L].append(feats[L])

        # Stack
        for split in cat_data:
            for L in extractor.target_layers:
                arr = cat_data[split][L]
                cat_data[split][L] = np.stack(arr) if arr else np.array([]).reshape(0, 1536)

        n_train = len(cat_data["train_normal"][extractor.target_layers[0]])
        n_test_n = len(cat_data["test_normal"][extractor.target_layers[0]])
        n_test_a = len(cat_data["test_anomaly"][extractor.target_layers[0]])
        print(f"  {cat}: train={n_train}, test_n={n_test_n}, test_a={n_test_a}")

        dataset[cat] = cat_data
    return dataset


# ─── IFR Nuisance Estimation ─────────────────────────────────────

def ifr_nuisance_subspace(
    train_multilayer: dict[int, np.ndarray],
    anchor_layer: int = 8,
    deep_layers: tuple[int, ...] = (9, 10, 11),
    K: int = 100,
    mode: str = "stacked",
) -> np.ndarray:
    """Estimate nuisance subspace from inter-layer residuals.

    Args:
        train_multilayer: {layer_idx: (N, D) features} for train normals
        anchor_layer: robust layer (bottleneck)
        deep_layers: layers beyond bottleneck
        K: number of nuisance directions to extract
        mode: 'stacked' = PCA on [Δ_9; Δ_10; Δ_11] separately then merge
              'summed' = PCA on (Δ_9 + Δ_10 + Δ_11) = f_L11 - f_L8
              'per_layer' = PCA on each Δ_L separately, take top-k from each

    Returns:
        (D, K) nuisance basis matrix
    """
    f_anchor = train_multilayer[anchor_layer]  # (N, D)

    if mode == "summed":
        # Simplest: total residual = f_L11 - f_L8
        last_layer = max(deep_layers)
        delta_total = train_multilayer[last_layer] - f_anchor  # (N, D)
        delta_centered = delta_total - delta_total.mean(axis=0)
        U, S, Vt = np.linalg.svd(delta_centered, full_matrices=False)
        n_dirs = min(K, Vt.shape[0])
        return Vt[:n_dirs].T  # (D, K)

    elif mode == "stacked":
        # Stack all per-layer residuals, then joint PCA
        deltas = []
        prev = f_anchor
        for L in sorted(deep_layers):
            delta_L = train_multilayer[L] - prev
            deltas.append(delta_L)
            prev = train_multilayer[L]
        all_deltas = np.vstack(deltas)  # (N * num_layers, D)
        centered = all_deltas - all_deltas.mean(axis=0)
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        n_dirs = min(K, Vt.shape[0])
        return Vt[:n_dirs].T

    elif mode == "per_layer":
        # PCA on each layer's residual, take top-k/num_layers from each
        k_per = max(1, K // len(deep_layers))
        all_dirs = []
        prev = f_anchor
        for L in sorted(deep_layers):
            delta_L = train_multilayer[L] - prev
            centered = delta_L - delta_L.mean(axis=0)
            U, S, Vt = np.linalg.svd(centered, full_matrices=False)
            all_dirs.append(Vt[:k_per].T)
            prev = train_multilayer[L]
        combined = np.hstack(all_dirs)  # (D, k_per * num_layers)
        # Re-orthogonalize
        Q, _ = np.linalg.qr(combined)
        n_dirs = min(K, Q.shape[1])
        return Q[:, :n_dirs]

    else:
        raise ValueError(f"Unknown mode: {mode}")


def ifr_alignment_with_nsp(
    ifr_basis: np.ndarray,
    nsp_basis: np.ndarray,
) -> float:
    """Measure alignment between IFR and NSP nuisance subspaces.

    Returns average absolute cosine similarity (0=orthogonal, 1=aligned).
    """
    # Project IFR basis onto NSP subspace
    # Subspace alignment: Frobenius norm of V_ifr^T @ V_nsp / min(K1, K2)
    cross = ifr_basis.T @ nsp_basis  # (K1, K2)
    U, S, Vt = np.linalg.svd(cross, full_matrices=False)
    # Principal angles: cos(theta_i) = S_i
    return float(S.mean())


# ─── Scoring ─────────────────────────────────────────────────────

def hard_project(features: np.ndarray, nuisance_dirs: np.ndarray) -> np.ndarray:
    """Remove nuisance subspace via orthogonal projection."""
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
    dataset: dict,
    method_fn,
    target_layer: int = 11,
) -> dict[str, float]:
    """Evaluate a method across all categories.

    method_fn(cat_data) -> (projected_train, projected_test_n, projected_test_a)
    """
    results = {}
    aurocs = []
    for cat in sorted(dataset.keys()):
        cat_data = dataset[cat]
        train_L = cat_data["train_normal"][target_layer]
        test_n_L = cat_data["test_normal"][target_layer]
        test_a_L = cat_data["test_anomaly"][target_layer]

        if len(train_L) == 0 or len(test_n_L) == 0 or len(test_a_L) == 0:
            continue

        train_p, test_n_p, test_a_p = method_fn(cat_data)

        scores_n = mahalanobis_score(train_p, test_n_p)
        scores_a = mahalanobis_score(train_p, test_a_p)
        labels = np.concatenate([np.zeros(len(scores_n)), np.ones(len(scores_a))])
        scores = np.concatenate([scores_n, scores_a])
        auroc = roc_auc_score(labels, scores)
        results[cat] = auroc
        aurocs.append(auroc)

    results["MEAN"] = float(np.mean(aurocs)) if aurocs else 0.0
    return results


# ─── Experiment Configs ──────────────────────────────────────────

def make_baseline_fn(layer: int):
    """Baseline: no projection, Mahalanobis on target layer."""
    def fn(cat_data):
        return (
            cat_data["train_normal"][layer],
            cat_data["test_normal"][layer],
            cat_data["test_anomaly"][layer],
        )
    return fn


def make_ifr_fn(
    anchor_layer: int,
    deep_layers: tuple[int, ...],
    target_layer: int,
    K: int,
    mode: str,
):
    """IFR: project target layer features using inter-layer residual nuisance."""
    def fn(cat_data):
        nuisance = ifr_nuisance_subspace(
            cat_data["train_normal"],
            anchor_layer=anchor_layer,
            deep_layers=deep_layers,
            K=K,
            mode=mode,
        )
        train_p = hard_project(cat_data["train_normal"][target_layer], nuisance)
        test_n_p = hard_project(cat_data["test_normal"][target_layer], nuisance)
        test_a_p = hard_project(cat_data["test_anomaly"][target_layer], nuisance)
        return train_p, test_n_p, test_a_p
    return fn


def make_l8_only_fn():
    """L8-only baseline: use robust layer directly."""
    def fn(cat_data):
        return (
            cat_data["train_normal"][8],
            cat_data["test_normal"][8],
            cat_data["test_anomaly"][8],
        )
    return fn


# ─── Main ────────────────────────────────────────────────────────

def run(args):
    print("=" * 60)
    print("IFR: Inter-layer Feature Residual Experiment")
    print("=" * 60)

    layers = tuple(range(args.anchor_layer, 12))  # anchor through 11
    extractor = DINOv2MultiLayerExtractor(
        model_name=args.model_name,
        device=args.device,
        target_layers=layers,
    )
    print(f"\n[Config] Anchor: L{args.anchor_layer}, Layers: {layers}")

    # ── Extract features ──
    print(f"\n[1/3] Extracting features...")
    datasets = {}
    print("  --- MVTec AD 2 ---")
    datasets["AD2"] = extract_dataset_multilayer(args.data_root_ad2, extractor, max_per_cat=args.max_per_cat)

    if args.data_root_ad1:
        print("  --- MVTec AD ---")
        datasets["AD1"] = extract_dataset_multilayer(args.data_root_ad1, extractor, max_per_cat=args.max_per_cat)

    # ── Define experiments ──
    deep_layers = tuple(range(args.anchor_layer + 1, 12))
    experiments = [
        ("baseline_L11", make_baseline_fn(11)),
        ("baseline_L8", make_l8_only_fn()),
        # IFR variants
        ("IFR_summed_K30", make_ifr_fn(args.anchor_layer, deep_layers, 11, K=30, mode="summed")),
        ("IFR_summed_K50", make_ifr_fn(args.anchor_layer, deep_layers, 11, K=50, mode="summed")),
        ("IFR_summed_K100", make_ifr_fn(args.anchor_layer, deep_layers, 11, K=100, mode="summed")),
        ("IFR_summed_K200", make_ifr_fn(args.anchor_layer, deep_layers, 11, K=200, mode="summed")),
        ("IFR_stacked_K100", make_ifr_fn(args.anchor_layer, deep_layers, 11, K=100, mode="stacked")),
        ("IFR_perlayer_K100", make_ifr_fn(args.anchor_layer, deep_layers, 11, K=100, mode="per_layer")),
        # IFR on L8 (should have minimal effect — L8 is already robust)
        ("IFR_summed_K100_onL8", make_ifr_fn(args.anchor_layer, deep_layers, 8, K=100, mode="summed")),
    ]

    # ── Run experiments ──
    print(f"\n[2/3] Running {len(experiments)} experiments...")

    all_results = {}
    for ds_name, ds in datasets.items():
        print(f"\n  === {ds_name} ===")
        ds_results = {}
        for exp_name, method_fn in experiments:
            result = evaluate_method(ds, method_fn, target_layer=11 if "onL8" not in exp_name else 8)
            # Fix: for L8-only and onL8 methods, use appropriate target
            if exp_name == "baseline_L8":
                result = evaluate_method(ds, method_fn, target_layer=8)
            elif "onL8" in exp_name:
                result = evaluate_method(ds, method_fn, target_layer=8)
            ds_results[exp_name] = result
            mean_auroc = result["MEAN"]
            print(f"    {exp_name:<30s} → {mean_auroc*100:.1f}%")

        all_results[ds_name] = ds_results

        # Per-category breakdown for key methods
        key_methods = ["baseline_L11", "baseline_L8", "IFR_summed_K100", "IFR_stacked_K100"]
        cats = sorted(k for k in ds_results["baseline_L11"].keys() if k != "MEAN")
        print(f"\n  Per-category breakdown:")
        header = f"  {'Category':<20s}" + "".join(f" {m:>18s}" for m in key_methods)
        print(header)
        print("  " + "-" * (20 + 19 * len(key_methods)))
        for cat in cats:
            row = f"  {cat:<20s}"
            for m in key_methods:
                val = ds_results[m].get(cat, 0) * 100
                row += f" {val:>17.1f}%"
            print(row)
        row = f"  {'MEAN':<20s}"
        for m in key_methods:
            row += f" {ds_results[m]['MEAN']*100:>17.1f}%"
        print(row)

    # ── Nuisance subspace analysis ──
    print(f"\n[3/3] Nuisance subspace analysis...")
    for cat in sorted(datasets["AD2"].keys()):
        cat_data = datasets["AD2"][cat]
        train_ml = cat_data["train_normal"]
        if len(train_ml[args.anchor_layer]) == 0:
            continue

        # IFR nuisance directions
        nuisance_ifr = ifr_nuisance_subspace(
            train_ml, anchor_layer=args.anchor_layer, deep_layers=deep_layers, K=100, mode="summed"
        )

        # Variance explained by residuals
        f_anchor = train_ml[args.anchor_layer]
        f_deep = train_ml[11]
        delta_total = f_deep - f_anchor
        total_var = np.var(delta_total, axis=0).sum()
        projected_delta = hard_project(delta_total, nuisance_ifr)
        remaining_var = np.var(projected_delta, axis=0).sum()
        explained_ratio = 1.0 - remaining_var / (total_var + 1e-10)

        # Norm of residuals
        delta_norms = np.linalg.norm(delta_total, axis=1)

        print(f"  {cat}: residual_norm={delta_norms.mean():.3f}±{delta_norms.std():.3f}, "
              f"variance_explained_by_K100={explained_ratio:.1%}")

    # ── Save results ──
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "ifr_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Results saved to {output_dir}/ifr_results.json")

    # ── Summary ──
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    if "AD2" in all_results:
        ad2 = all_results["AD2"]
        print(f"  Baseline L11:       {ad2['baseline_L11']['MEAN']*100:.1f}%")
        print(f"  Baseline L8:        {ad2['baseline_L8']['MEAN']*100:.1f}%")
        best_ifr_name = max(
            (k for k in ad2 if k.startswith("IFR")),
            key=lambda k: ad2[k]["MEAN"],
        )
        print(f"  Best IFR:           {ad2[best_ifr_name]['MEAN']*100:.1f}% ({best_ifr_name})")
        print(f"  NSP oracle (ref):   83.8%")
        print(f"\n  GO/NO-GO:")
        best_val = ad2[best_ifr_name]["MEAN"] * 100
        if best_val >= 75:
            print(f"    ✅ STRONG GO — {best_val:.1f}% (≥75%)")
        elif best_val >= 70:
            print(f"    ✅ GO — {best_val:.1f}% (≥70%)")
        elif best_val >= 65:
            print(f"    🟡 CONDITIONAL GO — {best_val:.1f}% (65-70%)")
        elif best_val >= 60:
            print(f"    ⚠️ WEAK — {best_val:.1f}% (60-65%, baseline 수준)")
        else:
            print(f"    ❌ NO-GO — {best_val:.1f}% (<60%)")

    if "AD1" in all_results:
        ad1 = all_results["AD1"]
        print(f"\n  Clean performance (AD1):")
        print(f"  Baseline L11:       {ad1['baseline_L11']['MEAN']*100:.1f}%")
        best_ifr_ad1 = ad1.get(best_ifr_name, {}).get("MEAN", 0)
        print(f"  Best IFR:           {best_ifr_ad1*100:.1f}%")
        delta = best_ifr_ad1 * 100 - ad1["baseline_L11"]["MEAN"] * 100
        print(f"  Delta:              {delta:+.1f}pp {'✅' if delta >= -1 else '⚠️'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IFR: Inter-layer Feature Residual experiment")
    parser.add_argument("--data_root_ad2", type=str, default="/home/hun/Volume/DATA/mvtec_ad_2_compat")
    parser.add_argument("--data_root_ad1", type=str, default=None)
    parser.add_argument("--model_name", type=str, default="dinov2_vitb14")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--anchor_layer", type=int, default=8)
    parser.add_argument("--max_per_cat", type=int, default=200)
    parser.add_argument("--output_dir", type=str, default="results/ifr")
    args = parser.parse_args()
    run(args)
