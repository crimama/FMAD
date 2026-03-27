#!/usr/bin/env python3
"""Phase 2 Step 2.3: Entanglement Measurement.

Measures whether shift directions and anomaly directions overlap in FM feature space.
- PCA on normal features → find principal variation directions
- Project shift vectors and anomaly vectors onto PCA components
- Measure correlation: if same PCs respond to both shift and anomaly → entangled

Also produces t-SNE visualization of normal / shifted-normal / anomaly distributions.
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
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
        # CLS token + mean of patch tokens
        feat = self.feature[0]  # (n_tokens, dim)
        cls_token = feat[0]
        patch_mean = feat[1:].mean(dim=0)
        combined = torch.cat([cls_token, patch_mean])  # (2*dim,)
        return combined.cpu().numpy()


def collect_features(data_root: str, extractor, max_per_group: int = 50):
    """Collect features for three groups: normal-regular, normal-shifted, anomaly-regular."""
    root = Path(data_root)
    categories = sorted(d.name for d in root.iterdir() if d.is_dir() and (d / "test_public").is_dir())

    features = {"normal_regular": [], "normal_shifted": [], "anomaly_regular": [], "anomaly_shifted": []}
    metadata = {"normal_regular": [], "normal_shifted": [], "anomaly_regular": [], "anomaly_shifted": []}

    for cat in categories:
        for split, label in [("good", "normal"), ("bad", "anomaly")]:
            split_dir = root / cat / "test_public" / split
            if not split_dir.is_dir():
                continue

            groups = defaultdict(dict)
            for img_path in sorted(split_dir.glob("*.png")):
                parts = img_path.stem.split("_", 1)
                if len(parts) == 2:
                    obj_idx, condition = parts
                    groups[obj_idx][condition] = img_path

            count = 0
            for obj_idx, conditions in groups.items():
                if count >= max_per_group:
                    break
                if "regular" not in conditions:
                    continue

                # Regular
                img = Image.open(conditions["regular"]).convert("RGB")
                feat = extractor.extract(img)
                key_reg = f"{label}_regular"
                features[key_reg].append(feat)
                metadata[key_reg].append({"category": cat, "obj_idx": obj_idx, "condition": "regular"})

                # Shifted conditions
                for cond, path in conditions.items():
                    if cond == "regular":
                        continue
                    img_s = Image.open(path).convert("RGB")
                    feat_s = extractor.extract(img_s)
                    key_shift = f"{label}_shifted"
                    features[key_shift].append(feat_s)
                    metadata[key_shift].append({"category": cat, "obj_idx": obj_idx, "condition": cond})

                count += 1

    for k in features:
        features[k] = np.stack(features[k]) if features[k] else np.array([])
        print(f"  {k}: {len(features[k])} samples")

    return features, metadata


def entanglement_analysis(features: dict) -> dict:
    """Measure entanglement between shift and anomaly directions via PCA."""

    normal_reg = features["normal_regular"]  # (N, D)
    normal_shift = features["normal_shifted"]  # (M, D)
    anomaly_reg = features["anomaly_regular"]  # (K, D)

    if len(normal_reg) == 0 or len(normal_shift) == 0 or len(anomaly_reg) == 0:
        return {"error": "Insufficient samples"}

    # Center on normal-regular mean
    center = normal_reg.mean(axis=0)
    normal_reg_c = normal_reg - center
    normal_shift_c = normal_shift - center
    anomaly_reg_c = anomaly_reg - center

    # PCA on normal-regular features
    cov = np.cov(normal_reg_c.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    # Sort descending
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Explained variance
    total_var = eigenvalues.sum()
    explained_ratio = eigenvalues / total_var

    # Compute shift direction (mean direction from regular to shifted)
    # For each normal-shifted sample, we don't have exact pairs, so use group means
    shift_direction = normal_shift_c.mean(axis=0) - normal_reg_c.mean(axis=0)
    shift_direction_norm = shift_direction / (np.linalg.norm(shift_direction) + 1e-8)

    # Anomaly direction (mean direction from normal to anomaly)
    anomaly_direction = anomaly_reg_c.mean(axis=0) - normal_reg_c.mean(axis=0)
    anomaly_direction_norm = anomaly_direction / (np.linalg.norm(anomaly_direction) + 1e-8)

    # Project onto PCA components
    n_components = min(50, len(eigenvalues))
    top_pcs = eigenvectors[:, :n_components]  # (D, n_components)

    shift_proj = np.abs(top_pcs.T @ shift_direction_norm)  # (n_components,)
    anomaly_proj = np.abs(top_pcs.T @ anomaly_direction_norm)  # (n_components,)

    # Entanglement metrics
    # 1. Cosine similarity between shift and anomaly directions
    cos_sim = float(np.dot(shift_direction_norm, anomaly_direction_norm))

    # 2. Overlap in top PCs: correlation of projections
    proj_correlation = float(np.corrcoef(shift_proj, anomaly_proj)[0, 1])

    # 3. Shared top-K components: PCs where both shift and anomaly have high projection
    k = 10
    shift_topk = set(np.argsort(shift_proj)[-k:])
    anomaly_topk = set(np.argsort(anomaly_proj)[-k:])
    shared_topk = shift_topk & anomaly_topk
    shared_ratio = len(shared_topk) / k

    # 4. Per-PC analysis
    pc_analysis = []
    for i in range(n_components):
        pc_analysis.append({
            "pc_idx": i,
            "explained_variance_ratio": float(explained_ratio[i]),
            "cumulative_explained": float(explained_ratio[:i+1].sum()),
            "shift_projection": float(shift_proj[i]),
            "anomaly_projection": float(anomaly_proj[i]),
        })

    return {
        "entanglement_metrics": {
            "shift_anomaly_cosine_similarity": cos_sim,
            "pc_projection_correlation": proj_correlation,
            "shared_top10_ratio": shared_ratio,
            "shared_top10_indices": sorted(shared_topk),
        },
        "directions": {
            "shift_magnitude": float(np.linalg.norm(shift_direction)),
            "anomaly_magnitude": float(np.linalg.norm(anomaly_direction)),
        },
        "pca": {
            "n_components_for_90pct": int(np.searchsorted(np.cumsum(explained_ratio), 0.9) + 1),
            "n_components_for_95pct": int(np.searchsorted(np.cumsum(explained_ratio), 0.95) + 1),
            "top_eigenvalues": eigenvalues[:20].tolist(),
        },
        "pc_analysis": pc_analysis,
    }


def compute_tsne(features: dict, output_dir: Path):
    """Compute t-SNE and save coordinates for visualization."""
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        print("  sklearn not available, skipping t-SNE")
        return

    # Combine all features
    groups = []
    labels = []
    for key in ["normal_regular", "normal_shifted", "anomaly_regular", "anomaly_shifted"]:
        if len(features[key]) > 0:
            groups.append(features[key])
            labels.extend([key] * len(features[key]))

    if not groups:
        return

    all_features = np.concatenate(groups, axis=0)
    print(f"  Computing t-SNE on {all_features.shape[0]} samples, dim={all_features.shape[1]}...")

    tsne = TSNE(n_components=2, perplexity=min(30, len(all_features) - 1), random_state=42)
    coords = tsne.fit_transform(all_features)

    # Save
    tsne_data = {
        "coordinates": coords.tolist(),
        "labels": labels,
        "n_samples": len(labels),
    }
    with open(output_dir / "tsne_coordinates.json", "w") as f:
        json.dump(tsne_data, f, indent=2)
    print(f"  Saved t-SNE to {output_dir / 'tsne_coordinates.json'}")

    # Print overlap statistics
    label_set = sorted(set(labels))
    centroids = {}
    for lbl in label_set:
        mask = np.array(labels) == lbl
        centroids[lbl] = coords[mask].mean(axis=0)

    print("\n  t-SNE centroid distances:")
    for i, l1 in enumerate(label_set):
        for l2 in label_set[i+1:]:
            dist = np.linalg.norm(centroids[l1] - centroids[l2])
            print(f"    {l1} ↔ {l2}: {dist:.2f}")


def run(args):
    print("=" * 60)
    print("Phase 2 Step 2.3: Entanglement Analysis")
    print("=" * 60)

    # Extract features
    print(f"\n[1/3] Extracting features (layer {args.layer})...")
    extractor = DINOv2Extractor(
        model_name=args.model_name,
        device=args.device,
        target_layer=args.layer,
    )
    features, metadata = collect_features(args.data_root, extractor, max_per_group=args.max_per_group)

    # Entanglement analysis
    print("\n[2/3] Computing entanglement metrics...")
    results = entanglement_analysis(features)

    # Print key results
    em = results["entanglement_metrics"]
    print(f"\n  === ENTANGLEMENT METRICS ===")
    print(f"  Shift-Anomaly cosine similarity:   {em['shift_anomaly_cosine_similarity']:.4f}")
    print(f"  PC projection correlation:         {em['pc_projection_correlation']:.4f}")
    print(f"  Shared top-10 PC ratio:            {em['shared_top10_ratio']:.1%}")
    print(f"  Shared PC indices:                 {em['shared_top10_indices']}")
    print(f"\n  Shift magnitude:  {results['directions']['shift_magnitude']:.4f}")
    print(f"  Anomaly magnitude: {results['directions']['anomaly_magnitude']:.4f}")
    print(f"  PCs for 90% var:  {results['pca']['n_components_for_90pct']}")
    print(f"  PCs for 95% var:  {results['pca']['n_components_for_95pct']}")

    # Top PCs breakdown
    print(f"\n  === TOP PCs: Shift vs Anomaly Projection ===")
    print(f"  {'PC':>4} {'ExpVar%':>8} {'Shift':>8} {'Anomaly':>8} {'Dominant':>10}")
    for pc in results["pc_analysis"][:20]:
        s, a = pc["shift_projection"], pc["anomaly_projection"]
        dominant = "SHIFT" if s > a * 1.5 else ("ANOMALY" if a > s * 1.5 else "SHARED")
        print(f"  {pc['pc_idx']:>4} {pc['explained_variance_ratio']*100:>7.2f}% {s:>8.4f} {a:>8.4f} {dominant:>10}")

    # t-SNE
    print("\n[3/3] Computing t-SNE visualization...")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    compute_tsne(features, output_dir)

    # Save full results
    with open(output_dir / f"entanglement_{args.model_name}_layer{args.layer}.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="/home/hun/Volume/DATA/mvtec_ad_2")
    parser.add_argument("--model_name", type=str, default="dinov2_vitb14")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--layer", type=int, default=-1, help="Target layer (-1=last, 8=robust layer)")
    parser.add_argument("--output_dir", type=str, default="results/phase2")
    parser.add_argument("--max_per_group", type=int, default=50, help="Max samples per category per group")
    args = parser.parse_args()
    run(args)
