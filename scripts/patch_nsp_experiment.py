#!/usr/bin/env python3
"""Patch-level NSP: Spatially-Adaptive Nuisance Projection.

Key insight: Shift is spatially non-uniform in feature space (uniformity=2.6%).
Image-level NSP treats shift as a single direction → suboptimal.
Patch-level NSP captures position-specific shift structure.

Comparison:
  1. Baseline: image-level Mahalanobis (59.9%)
  2. Patch-kNN: patch-level scoring, no projection (66.2%)
  3. Image-NSP: image-level NSP K=100 (84.4%)
  4. Patch-NSP-global: patch-level scoring + global nuisance projection
  5. Patch-NSP-position: patch-level scoring + position-specific nuisance projection
  6. Patch-NSP-percat: per-category + position-specific (full)
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

class PatchExtractor:
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
        """(M, dim) patch tokens, M=37*37=1369."""
        x = self.transform(img).unsqueeze(0).to(self.device)
        _ = self.model(x)
        return self.feature[0, 1:].cpu().numpy()

    @torch.no_grad()
    def extract_image_feat(self, img):
        """CLS + patch_mean, L2 normalized. (dim*2,)"""
        x = self.transform(img).unsqueeze(0).to(self.device)
        _ = self.model(x)
        feat = self.feature[0]
        cls = feat[0]
        pm = feat[1:].mean(0)
        c = torch.cat([cls, pm])
        return (c / (c.norm() + 1e-8)).cpu().numpy()


# ─── Data Collection ──────────────────────────────────────────────

def collect_all(data_root, extractor, max_per_cat=100):
    """Collect patch-level + image-level features, plus paired shift vectors."""
    root = Path(data_root)
    cats = sorted(d.name for d in root.iterdir() if d.is_dir() and (d / "train").is_dir())

    dataset = {}
    for cat in cats:
        cd = {
            "train_patches": [], "train_images": [],
            "test_n_patches": [], "test_n_images": [],
            "test_a_patches": [], "test_a_images": [],
            "shift_patches": [],  # (N_pairs, M, dim) paired patch shifts
            "shift_images": [],   # (N_pairs, dim*2) paired image shifts
        }

        # Train
        tg = root / cat / "train" / "good"
        if tg.is_dir():
            for p in sorted(tg.iterdir())[:max_per_cat]:
                if p.suffix.lower() in (".png", ".jpg", ".jpeg"):
                    img = Image.open(p).convert("RGB")
                    cd["train_patches"].append(extractor.extract_patches(img))
                    cd["train_images"].append(extractor.extract_image_feat(img))

        # Test
        td = root / cat / "test_public"
        if td.is_dir():
            for sub in sorted(td.iterdir()):
                if not sub.is_dir():
                    continue
                key = "test_n" if sub.name == "good" else "test_a"
                for p in sorted(sub.iterdir())[:max_per_cat]:
                    if p.suffix.lower() in (".png", ".jpg", ".jpeg"):
                        img = Image.open(p).convert("RGB")
                        cd[f"{key}_patches"].append(extractor.extract_patches(img))
                        cd[f"{key}_images"].append(extractor.extract_image_feat(img))

        # Paired shift vectors
        gd = root / cat / "test_public" / "good"
        if gd.is_dir():
            groups = defaultdict(dict)
            for ip in sorted(gd.glob("*.png")):
                parts = ip.stem.split("_", 1)
                if len(parts) == 2:
                    groups[parts[0]][parts[1]] = ip

            for oid, conds in sorted(groups.items()):
                if "regular" not in conds:
                    continue
                img_reg = Image.open(conds["regular"]).convert("RGB")
                p_reg = extractor.extract_patches(img_reg)
                i_reg = extractor.extract_image_feat(img_reg)
                for c, path in sorted(conds.items()):
                    if c == "regular":
                        continue
                    img_s = Image.open(path).convert("RGB")
                    p_s = extractor.extract_patches(img_s)
                    i_s = extractor.extract_image_feat(img_s)
                    cd["shift_patches"].append(p_s - p_reg)     # (M, dim)
                    cd["shift_images"].append(i_s - i_reg)      # (dim*2,)

        # Stack
        cd["train_images"] = np.stack(cd["train_images"]) if cd["train_images"] else np.zeros((0, 1536))
        cd["test_n_images"] = np.stack(cd["test_n_images"]) if cd["test_n_images"] else np.zeros((0, 1536))
        cd["test_a_images"] = np.stack(cd["test_a_images"]) if cd["test_a_images"] else np.zeros((0, 1536))
        cd["shift_images"] = np.stack(cd["shift_images"]) if cd["shift_images"] else np.zeros((0, 1536))
        # shift_patches: list of (M, dim) → (N_pairs, M, dim)
        cd["shift_patches"] = np.stack(cd["shift_patches"]) if cd["shift_patches"] else np.zeros((0, 1369, 768))

        nt = len(cd["train_patches"])
        nn = len(cd["test_n_patches"])
        na = len(cd["test_a_patches"])
        ns = len(cd["shift_images"])
        print(f"  {cat}: train={nt}, test_n={nn}, test_a={na}, pairs={ns}")
        dataset[cat] = cd

    return dataset


# ─── Nuisance Basis ───────────────────────────────────────────────

def pca_basis(vectors, K):
    """PCA top-K basis from (N, dim) vectors. Returns (dim, K)."""
    if len(vectors) <= 1 or K <= 0:
        return np.zeros((vectors.shape[1], 0))
    K = min(K, len(vectors) - 1)
    centered = vectors - vectors.mean(0)
    _, S, Vt = np.linalg.svd(centered, full_matrices=False)
    return Vt[:K].T


def hard_project(features, basis):
    """Project out basis directions. features: (..., dim), basis: (dim, K)."""
    if basis.shape[1] == 0:
        return features
    proj = basis @ basis.T
    return features - features @ proj


# ─── Scoring Methods ─────────────────────────────────────────────

def score_image_mahal(train_imgs, test_imgs):
    mu = train_imgs.mean(0)
    cov = np.cov((train_imgs - mu).T) + np.eye(train_imgs.shape[1]) * 1e-6
    inv = np.linalg.inv(cov)
    tc = test_imgs - mu
    return np.sqrt(np.sum(tc @ inv * tc, axis=1))


def score_patch_knn(train_patch_list, test_patches, coreset_n=3000):
    """Patch-level kNN scoring. Returns per-patch scores (M,)."""
    bank = np.vstack(train_patch_list)
    if coreset_n < len(bank):
        rng = np.random.RandomState(42)
        idx = rng.choice(len(bank), coreset_n, replace=False)
        bank = bank[idx]
    nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
    nn.fit(bank)
    d, _ = nn.kneighbors(test_patches)
    return d.ravel()


def score_patch_nsp_global(train_patch_list, test_patches, shift_patches_3d, K=100, coreset_n=3000):
    """Patch-level scoring with GLOBAL nuisance projection.
    shift_patches_3d: (N_pairs, M, dim) → flatten to (N_pairs*M, dim) for global PCA.
    """
    # Global nuisance from all patches all positions
    flat_shifts = shift_patches_3d.reshape(-1, shift_patches_3d.shape[-1])  # (N*M, dim)
    basis = pca_basis(flat_shifts, K)

    # Project train patches
    proj_train = [hard_project(tp, basis) for tp in train_patch_list]
    proj_test = hard_project(test_patches, basis)

    return score_patch_knn(proj_train, proj_test, coreset_n)


def score_patch_nsp_position(train_patch_list, test_patches, shift_patches_3d, K=10, coreset_n=3000):
    """Patch-level scoring with POSITION-SPECIFIC nuisance projection.
    For each position p: PCA on shift_patches[:, p, :] → position-specific basis.
    """
    M = test_patches.shape[0]
    dim = test_patches.shape[1]
    N_pairs = shift_patches_3d.shape[0]

    # Per-position projection + scoring
    proj_test = np.zeros_like(test_patches)
    proj_trains = [np.zeros_like(tp) for tp in train_patch_list]

    for p in range(M):
        # Position-specific shifts: (N_pairs, dim)
        pos_shifts = shift_patches_3d[:, p, :]
        pos_K = min(K, N_pairs - 1, dim)
        if pos_K <= 0:
            proj_test[p] = test_patches[p]
            for i, tp in enumerate(train_patch_list):
                proj_trains[i][p] = tp[p]
            continue

        basis = pca_basis(pos_shifts, pos_K)
        proj_test[p] = hard_project(test_patches[p:p+1], basis)[0]
        for i, tp in enumerate(train_patch_list):
            proj_trains[i][p] = hard_project(tp[p:p+1], basis)[0]

    return score_patch_knn(proj_trains, proj_test, coreset_n)


# ─── Evaluation (optimized) ──────────────────────────────────────

def _build_patch_bank(patch_list, coreset_n=3000):
    """Build kNN-ready memory bank once per category."""
    bank = np.vstack(patch_list)
    if coreset_n < len(bank):
        rng = np.random.RandomState(42)
        idx = rng.choice(len(bank), coreset_n, replace=False)
        bank = bank[idx]
    nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
    nn.fit(bank)
    return nn


def _score_patches_with_bank(nn_model, test_patches):
    """Score test patches against pre-built bank."""
    d, _ = nn_model.kneighbors(test_patches)
    return d.ravel()


def evaluate(dataset, method, K=100, coreset_n=3000, aggregate="p95"):
    """Evaluate a method across categories. Optimized: build bank once per category."""
    results = {}
    aurocs = []

    agg_fn = {
        "max": lambda s: np.max(s),
        "p95": lambda s: np.percentile(s, 95),
        "p99": lambda s: np.percentile(s, 99),
        "mean": lambda s: np.mean(s),
    }[aggregate]

    for cat in sorted(dataset.keys()):
        cd = dataset[cat]
        if not cd["test_n_patches"] or not cd["test_a_patches"] or not cd["train_patches"]:
            continue

        # ── Image-level methods ───────────────────────────────────
        if method == "image_mahal":
            scores_n = score_image_mahal(cd["train_images"], cd["test_n_images"]).tolist()
            scores_a = score_image_mahal(cd["train_images"], cd["test_a_images"]).tolist()

        elif method == "image_nsp":
            basis = pca_basis(cd["shift_images"], K)
            tr_p = hard_project(cd["train_images"], basis)
            tn_p = hard_project(cd["test_n_images"], basis)
            ta_p = hard_project(cd["test_a_images"], basis)
            scores_n = score_image_mahal(tr_p, tn_p).tolist()
            scores_a = score_image_mahal(tr_p, ta_p).tolist()

        # ── Patch-level methods (build bank ONCE) ─────────────────
        elif method == "patch_knn":
            nn_model = _build_patch_bank(cd["train_patches"], coreset_n)
            scores_n = [agg_fn(_score_patches_with_bank(nn_model, p)) for p in cd["test_n_patches"]]
            scores_a = [agg_fn(_score_patches_with_bank(nn_model, p)) for p in cd["test_a_patches"]]

        elif method == "patch_nsp_global":
            # Project ALL patches once, then build bank once
            flat_shifts = cd["shift_patches"].reshape(-1, cd["shift_patches"].shape[-1])
            basis = pca_basis(flat_shifts, K)
            proj_train = [hard_project(tp, basis) for tp in cd["train_patches"]]
            nn_model = _build_patch_bank(proj_train, coreset_n)
            scores_n = [agg_fn(_score_patches_with_bank(nn_model, hard_project(p, basis)))
                        for p in cd["test_n_patches"]]
            scores_a = [agg_fn(_score_patches_with_bank(nn_model, hard_project(p, basis)))
                        for p in cd["test_a_patches"]]

        elif method == "patch_nsp_position":
            # Position-specific: project each position separately, then score
            M = cd["train_patches"][0].shape[0]  # 1369
            dim = cd["train_patches"][0].shape[1]  # 768
            N_pairs = cd["shift_patches"].shape[0]
            pos_K = min(K, max(N_pairs - 1, 1))

            # Precompute per-position basis
            bases = []
            for p_idx in range(M):
                pos_shifts = cd["shift_patches"][:, p_idx, :]
                bases.append(pca_basis(pos_shifts, pos_K))

            # Project all train patches
            proj_train_list = []
            for tp in cd["train_patches"]:
                proj_tp = np.zeros_like(tp)
                for p_idx in range(M):
                    proj_tp[p_idx] = hard_project(tp[p_idx:p_idx+1], bases[p_idx])[0]
                proj_train_list.append(proj_tp)

            nn_model = _build_patch_bank(proj_train_list, coreset_n)

            def _project_image(patches):
                proj = np.zeros_like(patches)
                for p_idx in range(M):
                    proj[p_idx] = hard_project(patches[p_idx:p_idx+1], bases[p_idx])[0]
                return proj

            scores_n = [agg_fn(_score_patches_with_bank(nn_model, _project_image(p)))
                        for p in cd["test_n_patches"]]
            scores_a = [agg_fn(_score_patches_with_bank(nn_model, _project_image(p)))
                        for p in cd["test_a_patches"]]
        else:
            raise ValueError(method)

        if not scores_n or not scores_a:
            continue

        labels = np.concatenate([np.zeros(len(scores_n)), np.ones(len(scores_a))])
        scores = np.concatenate([scores_n, scores_a])
        auroc = roc_auc_score(labels, scores)
        results[cat] = auroc
        aurocs.append(auroc)
        print(f"    {cat}: {auroc*100:.1f}%")

    results["MEAN"] = np.mean(aurocs) if aurocs else 0.0
    return results


# ─── Main ────────────────────────────────────────────────────────

def run(args):
    print("=" * 60)
    print("Patch-level NSP: Spatially-Adaptive Nuisance Projection")
    print("=" * 60)

    extractor = PatchExtractor(args.model_name, args.device, args.layer)

    print(f"\n[1/2] Collecting all features (Layer {args.layer})...")
    dataset = collect_all(args.data_root, extractor, args.max_per_cat)

    print(f"\n[2/2] Evaluating methods...")

    methods = [
        ("image_mahal", {"K": 0}),
        ("image_nsp", {"K": args.K}),
        ("patch_knn", {"K": 0}),
        ("patch_nsp_global", {"K": args.K}),
    ]
    labels = [
        "img_mahal",
        f"img_nsp_K{args.K}",
        "patch_knn",
        f"patch_nsp_global_K{args.K}",
    ]

    if args.position:
        methods.append(("patch_nsp_position", {"K": 10}))
        labels.append("patch_nsp_pos_K10")

    all_results = {}
    for (method, kw), label in zip(methods, labels):
        K_val = kw.get("K", args.K)
        print(f"\n  --- {label} ---")
        if method.startswith("image"):
            res = evaluate(dataset, method, K=K_val, coreset_n=args.coreset_n)
        else:
            res = evaluate(dataset, method, K=K_val, coreset_n=args.coreset_n, aggregate=args.aggregate)
        all_results[label] = res
        print(f"  MEAN: {res['MEAN']*100:.1f}%")

    # Summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    cats = sorted(k for k in next(iter(all_results.values())) if k != "MEAN")
    header = f"  {'Category':<16s}" + "".join(f" {l[:15]:>16s}" for l in all_results)
    print(header)
    print("  " + "-" * (16 + 17 * len(all_results)))

    for cat in cats:
        row = f"  {cat:<16s}"
        for label in all_results:
            v = all_results[label].get(cat, 0) * 100
            row += f" {v:>15.1f}%"
        print(row)

    row = f"  {'MEAN':<16s}"
    for label in all_results:
        v = all_results[label]["MEAN"] * 100
        row += f" {v:>15.1f}%"
    print(row)

    # Save
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "patch_nsp_results.json", "w") as f:
        json.dump({k: {kk: float(vv) for kk, vv in v.items()} for k, v in all_results.items()}, f, indent=2)
    print(f"\n  Saved to {out / 'patch_nsp_results.json'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default="/home/hun/Volume/DATA/mvtec_ad_2")
    p.add_argument("--model_name", default="dinov2_vitb14")
    p.add_argument("--device", default="cuda")
    p.add_argument("--layer", type=int, default=8)
    p.add_argument("--K", type=int, default=100)
    p.add_argument("--max_per_cat", type=int, default=100)
    p.add_argument("--coreset_n", type=int, default=3000)
    p.add_argument("--aggregate", default="p95", choices=["max", "p95", "p99", "mean"])
    p.add_argument("--output_dir", default="results/patch_nsp")
    p.add_argument("--position", action="store_true", help="Include position-specific NSP (slow)")
    run(p.parse_args())
