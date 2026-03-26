#!/usr/bin/env python3
"""Phase 3: Nuisance Subspace Projection (NSP) Experiment.

Post-hoc linear disentanglement for robust anomaly detection.
1. Extract DINOv2 features from train normal (regular) and test images
2. Estimate nuisance subspace from regular↔shifted pairs
3. Project features onto nuisance-orthogonal subspace
4. kNN anomaly scoring on projected features
5. Compare with baseline (no projection)
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
        # CLS + patch mean, then L2 normalize
        cls_token = feat[0]
        patch_mean = feat[1:].mean(dim=0)
        combined = torch.cat([cls_token, patch_mean])
        combined = combined / (combined.norm() + 1e-8)
        return combined.cpu().numpy()

    @torch.no_grad()
    def extract_patches(self, img: Image.Image) -> np.ndarray:
        """Return all patch tokens (N_patches, dim)."""
        x = self.transform(img).unsqueeze(0).to(self.device)
        _ = self.model(x)
        return self.feature[0, 1:].cpu().numpy()  # skip CLS


def extract_dataset_features(data_root: str, extractor, max_per_cat: int = 100):
    """Extract features for all categories in MVTec AD 2 compat layout."""
    root = Path(data_root)
    categories = sorted(d.name for d in root.iterdir() if d.is_dir() and (d / "train").is_dir())

    dataset = {}
    for cat in categories:
        cat_data = {"train_normal": [], "test_normal": [], "test_anomaly": []}

        # Train normal
        train_good = root / cat / "train" / "good"
        if train_good.is_dir():
            for img_path in sorted(train_good.iterdir())[:max_per_cat]:
                if img_path.suffix.lower() in (".png", ".jpg", ".jpeg"):
                    feat = extractor.extract(Image.open(img_path).convert("RGB"))
                    cat_data["train_normal"].append(feat)

        # Test
        test_dir = root / cat / "test"
        if not test_dir.is_dir():
            test_dir = root / cat / "test_public"
        if test_dir.is_dir():
            for sub in sorted(test_dir.iterdir()):
                if not sub.is_dir():
                    continue
                label = "test_normal" if sub.name == "good" else "test_anomaly"
                for img_path in sorted(sub.iterdir())[:max_per_cat]:
                    if img_path.suffix.lower() in (".png", ".jpg", ".jpeg"):
                        feat = extractor.extract(Image.open(img_path).convert("RGB"))
                        cat_data[label].append(feat)

        for k in cat_data:
            cat_data[k] = np.stack(cat_data[k]) if cat_data[k] else np.array([]).reshape(0, 768)

        dataset[cat] = cat_data
        n_train = len(cat_data["train_normal"])
        n_test_n = len(cat_data["test_normal"])
        n_test_a = len(cat_data["test_anomaly"])
        print(f"  {cat}: train={n_train}, test_normal={n_test_n}, test_anomaly={n_test_a}")

    return dataset


def estimate_nuisance_subspace(data_root_original: str, extractor, n_components: int = 10, max_pairs: int = 200):
    """Estimate nuisance directions from regular↔shifted pairs in MVTec AD 2."""
    root = Path(data_root_original)
    categories = sorted(d.name for d in root.iterdir() if d.is_dir() and (d / "test_public").is_dir())

    shift_vectors = []
    for cat in categories:
        good_dir = root / cat / "test_public" / "good"
        if not good_dir.is_dir():
            continue

        groups = defaultdict(dict)
        for img_path in sorted(good_dir.glob("*.png")):
            parts = img_path.stem.split("_", 1)
            if len(parts) == 2:
                obj_idx, condition = parts
                groups[obj_idx][condition] = img_path

        for obj_idx, conditions in groups.items():
            if "regular" not in conditions:
                continue
            feat_reg = extractor.extract(Image.open(conditions["regular"]).convert("RGB"))
            for cond, path in conditions.items():
                if cond == "regular":
                    continue
                feat_shift = extractor.extract(Image.open(path).convert("RGB"))
                shift_vectors.append(feat_shift - feat_reg)

            if len(shift_vectors) >= max_pairs:
                break
        if len(shift_vectors) >= max_pairs:
            break

    shift_vectors = np.stack(shift_vectors)
    print(f"  Collected {len(shift_vectors)} shift vectors")

    # PCA on shift vectors
    mean_shift = shift_vectors.mean(axis=0)
    centered = shift_vectors - mean_shift
    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    nuisance_dirs = eigenvectors[:, :n_components]  # (dim, K)
    explained = eigenvalues[:n_components].sum() / eigenvalues.sum()
    print(f"  Top-{n_components} nuisance directions explain {explained:.1%} of shift variance")

    return nuisance_dirs


def project_out_nuisance(features: np.ndarray, nuisance_dirs: np.ndarray) -> np.ndarray:
    """Remove nuisance components from features.

    f_clean = f - N @ N^T @ f  where N is (dim, K) nuisance basis
    """
    projection = nuisance_dirs @ nuisance_dirs.T  # (dim, dim)
    return features - features @ projection


def knn_anomaly_score(train_features: np.ndarray, test_features: np.ndarray, k: int = 5) -> np.ndarray:
    """Compute kNN distance as anomaly score."""
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=k, metric="euclidean")
    nn.fit(train_features)
    distances, _ = nn.kneighbors(test_features)
    return distances.mean(axis=1)


def evaluate_category(cat_data: dict, nuisance_dirs: np.ndarray | None, k: int = 5) -> dict:
    """Evaluate one category with optional nuisance projection."""
    train = cat_data["train_normal"]
    test_n = cat_data["test_normal"]
    test_a = cat_data["test_anomaly"]

    if len(train) == 0 or len(test_n) == 0 or len(test_a) == 0:
        return {"auroc": None, "n_test": 0}

    # Apply projection
    if nuisance_dirs is not None:
        train = project_out_nuisance(train, nuisance_dirs)
        test_n = project_out_nuisance(test_n, nuisance_dirs)
        test_a = project_out_nuisance(test_a, nuisance_dirs)

    # kNN scoring
    scores_n = knn_anomaly_score(train, test_n, k=k)
    scores_a = knn_anomaly_score(train, test_a, k=k)

    labels = np.concatenate([np.zeros(len(scores_n)), np.ones(len(scores_a))])
    scores = np.concatenate([scores_n, scores_a])

    auroc = roc_auc_score(labels, scores)
    return {"auroc": auroc, "n_test": len(labels)}


def run(args):
    print("=" * 60)
    print("Phase 3: Nuisance Subspace Projection (NSP)")
    print("=" * 60)

    # Initialize extractor
    print(f"\n[1/4] Loading DINOv2 extractor (layer {args.layer})...")
    extractor = DINOv2Extractor(model_name=args.model_name, device=args.device, target_layer=args.layer)

    # Estimate nuisance subspace from original MVTec AD 2
    print(f"\n[2/4] Estimating nuisance subspace (K={args.n_components})...")
    nuisance_dirs = estimate_nuisance_subspace(
        args.data_root_original, extractor,
        n_components=args.n_components, max_pairs=300,
    )

    # Extract features from compat layout
    print(f"\n[3/4] Extracting dataset features...")
    print("  --- MVTec AD 2 ---")
    dataset_ad2 = extract_dataset_features(args.data_root_ad2, extractor, max_per_cat=200)

    if args.data_root_ad1:
        print("  --- MVTec AD ---")
        dataset_ad1 = extract_dataset_features(args.data_root_ad1, extractor, max_per_cat=200)
    else:
        dataset_ad1 = None

    # Evaluate
    print(f"\n[4/4] Evaluating...")
    results = {"nsp": {}, "baseline": {}, "config": {
        "n_components": args.n_components, "layer": args.layer, "k": args.k,
    }}

    for dataset_name, dataset in [("mvtec_ad_2", dataset_ad2), ("mvtec_ad", dataset_ad1)]:
        if dataset is None:
            continue
        print(f"\n  === {dataset_name} ===")
        print(f"  {'Category':<20s} {'Baseline':>10s} {'NSP':>10s} {'Δ':>8s}")
        print("  " + "-" * 50)

        aurocs_base = []
        aurocs_nsp = []
        for cat in sorted(dataset.keys()):
            r_base = evaluate_category(dataset[cat], nuisance_dirs=None, k=args.k)
            r_nsp = evaluate_category(dataset[cat], nuisance_dirs=nuisance_dirs, k=args.k)

            if r_base["auroc"] is not None:
                delta = (r_nsp["auroc"] - r_base["auroc"]) * 100
                print(f"  {cat:<20s} {r_base['auroc']*100:>9.1f}% {r_nsp['auroc']*100:>9.1f}% {delta:>+7.1f}pp")
                aurocs_base.append(r_base["auroc"])
                aurocs_nsp.append(r_nsp["auroc"])

                results[dataset_name] = results.get(dataset_name, {})
                results[dataset_name][cat] = {
                    "baseline": r_base["auroc"],
                    "nsp": r_nsp["auroc"],
                    "delta": r_nsp["auroc"] - r_base["auroc"],
                }

        if aurocs_base:
            mean_base = np.mean(aurocs_base) * 100
            mean_nsp = np.mean(aurocs_nsp) * 100
            delta_mean = mean_nsp - mean_base
            print(f"  {'MEAN':<20s} {mean_base:>9.1f}% {mean_nsp:>9.1f}% {delta_mean:>+7.1f}pp")
            results[dataset_name]["MEAN"] = {
                "baseline": np.mean(aurocs_base),
                "nsp": np.mean(aurocs_nsp),
                "delta": np.mean(aurocs_nsp) - np.mean(aurocs_base),
            }

    # Save
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outpath = output_dir / f"nsp_K{args.n_components}_L{args.layer}.json"
    with open(outpath, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Saved to {outpath}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root_original", type=str, default="/home/hun/Volume/DATA/mvtec_ad_2",
                        help="Original MVTec AD 2 root (for shift pair extraction)")
    parser.add_argument("--data_root_ad2", type=str, default="/home/hun/Volume/DATA/mvtec_ad_2_compat",
                        help="MVTec AD 2 compat layout (for AD evaluation)")
    parser.add_argument("--data_root_ad1", type=str, default=None,
                        help="MVTec AD root (for clean performance check)")
    parser.add_argument("--model_name", type=str, default="dinov2_vitb14")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--layer", type=int, default=11)
    parser.add_argument("--n_components", type=int, default=10, help="Number of nuisance directions to remove")
    parser.add_argument("--k", type=int, default=5, help="kNN k")
    parser.add_argument("--output_dir", type=str, default="results/phase3")
    args = parser.parse_args()
    run(args)
