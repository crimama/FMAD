#!/usr/bin/env python3
"""Relative Representation Diagnostic for Shift-Agnostic AD.

Two diagnostic tests for the Relative Representation + Intra-Image Normalization method:

  D1: Shift Sensitivity Reduction
      - Compare degradation under shift: raw kNN score vs relative kNN score
      - GO: relative degradation < 50% of raw degradation

  D2: Intra-Image Normalization Benefit
      - Compare AUROC: raw, relative, relative+intra-image-norm
      - GO: relative+IIN AUROC(AD2) > raw AUROC(AD2) + 5pp

Theory:
  - Relative representation: encode each test patch as cosine similarity profile
    to K reference patches → shift affects all sims similarly → more robust
  - Intra-image normalization: anomaly is local, shift is global
    → normalize within image to remove global shift floor

Setting: 8-shot, multi-layer (4, 7, 8, 11), DINOv2 ViT-B/14 frozen.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
from torchvision import transforms


# ─── Feature Extraction (multi-layer, patch-level) ─────────────────

class MultiLayerPatchExtractor:
    """Extract DINOv2 patch features at multiple layers simultaneously."""

    def __init__(
        self,
        model_name: str = "dinov2_vitb14",
        device: str = "cuda",
        target_layers: tuple[int, ...] = (4, 7, 8, 11),
    ) -> None:
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
    def extract_patches(self, img: Image.Image) -> dict[int, np.ndarray]:
        """Return {layer_idx: (M, D)} patch features."""
        x = self.transform(img).unsqueeze(0).to(self.device)
        self.features.clear()
        _ = self.model(x)

        result = {}
        for layer_idx in self.target_layers:
            patches = self.features[layer_idx][0, 1:]  # skip CLS, (M, D)
            result[layer_idx] = patches.cpu().numpy()
        return result


# ─── Relative Representation ───────────────────────────────────────

def build_relative_repr(
    patches: np.ndarray,
    anchors: np.ndarray,
) -> np.ndarray:
    """Encode patches as cosine similarity profiles to anchor (reference) patches.

    Args:
        patches: (M, D) test or reference patch features
        anchors: (A, D) reference anchor features

    Returns:
        (M, A) cosine similarity matrix — the relative representation
    """
    # L2 normalize
    p_norm = patches / (np.linalg.norm(patches, axis=1, keepdims=True) + 1e-8)
    a_norm = anchors / (np.linalg.norm(anchors, axis=1, keepdims=True) + 1e-8)
    return p_norm @ a_norm.T  # (M, A)


def build_multilayer_relative_repr(
    ml_patches: dict[int, np.ndarray],
    ml_anchors: dict[int, np.ndarray],
    layers: tuple[int, ...],
) -> np.ndarray:
    """Build cross-layer relative representation by concatenating per-layer profiles.

    Returns:
        (M, A*L) concatenated relative representation across layers
    """
    profiles = []
    for l in layers:
        sim = build_relative_repr(ml_patches[l], ml_anchors[l])  # (M, A)
        profiles.append(sim)
    return np.concatenate(profiles, axis=1)  # (M, A*L)


# ─── Scoring Functions ─────────────────────────────────────────────

def raw_knn_patch_scores(
    test_patches: np.ndarray,
    ref_bank: np.ndarray,
    k: int = 3,
) -> np.ndarray:
    """Standard kNN distance scoring in raw feature space. Returns (M,)."""
    nn = NearestNeighbors(n_neighbors=k, metric="euclidean").fit(ref_bank)
    dists, _ = nn.kneighbors(test_patches)
    return dists.mean(axis=1)


def relative_knn_patch_scores(
    test_rel: np.ndarray,
    ref_rel: np.ndarray,
    k: int = 3,
) -> np.ndarray:
    """kNN distance scoring in relative representation space. Returns (M,)."""
    nn = NearestNeighbors(n_neighbors=k, metric="euclidean").fit(ref_rel)
    dists, _ = nn.kneighbors(test_rel)
    return dists.mean(axis=1)


def cosine_knn_patch_scores(
    test_patches: np.ndarray,
    ref_bank: np.ndarray,
    k: int = 3,
) -> np.ndarray:
    """kNN scoring using cosine distance in raw feature space. Returns (M,)."""
    nn = NearestNeighbors(n_neighbors=k, metric="cosine").fit(ref_bank)
    dists, _ = nn.kneighbors(test_patches)
    return dists.mean(axis=1)


def intra_image_normalize(scores: np.ndarray) -> np.ndarray:
    """Normalize scores within image: (s - median) / MAD."""
    med = np.median(scores)
    mad = np.median(np.abs(scores - med)) + 1e-8
    return (scores - med) / mad


def aggregate_image_score(patch_scores: np.ndarray, method: str = "max") -> float:
    """Aggregate patch-level scores to image-level."""
    if method == "max":
        return float(np.max(patch_scores))
    elif method == "p95":
        return float(np.percentile(patch_scores, 95))
    elif method == "mean":
        return float(np.mean(patch_scores))
    raise ValueError(f"Unknown aggregation: {method}")


# ─── Data Collection ───────────────────────────────────────────────

def collect_data(
    data_root: str,
    extractor: MultiLayerPatchExtractor,
    K_shot: int = 8,
    max_per_cat: int = 200,
    seed: int = 42,
    coreset_n: int = 2000,
) -> dict:
    """Collect few-shot reference + test data with multi-layer patch features."""
    root = Path(data_root)
    rng = np.random.RandomState(seed)
    cats = sorted(d.name for d in root.iterdir() if d.is_dir() and (d / "train").is_dir())

    dataset = {}
    for cat in cats:
        print(f"  Collecting {cat}...")

        # Train references (regular only)
        train_ml_patches: list[dict[int, np.ndarray]] = []
        train_dir = root / cat / "train" / "good"
        if train_dir.is_dir():
            paths = sorted(
                p for p in train_dir.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg")
            )
            selected = rng.choice(len(paths), size=min(K_shot, len(paths)), replace=False)
            for idx in selected:
                ml = extractor.extract_patches(Image.open(paths[idx]).convert("RGB"))
                train_ml_patches.append(ml)

        # Build reference banks per layer + coreset sampling
        ref_banks: dict[int, np.ndarray] = {}
        for l in extractor.target_layers:
            all_patches = np.concatenate([ml[l] for ml in train_ml_patches], axis=0)
            if len(all_patches) > coreset_n:
                idx = rng.choice(len(all_patches), size=coreset_n, replace=False)
                all_patches = all_patches[idx]
            ref_banks[l] = all_patches
        # Anchor features for relative repr = ref_banks (or subsample)
        anchor_n = min(500, min(len(ref_banks[l]) for l in ref_banks))
        anchors: dict[int, np.ndarray] = {}
        for l in extractor.target_layers:
            idx = rng.choice(len(ref_banks[l]), size=anchor_n, replace=False)
            anchors[l] = ref_banks[l][idx]

        # Reference relative representations
        ref_rel: dict[int, np.ndarray] = {}
        for l in extractor.target_layers:
            ref_rel[l] = build_relative_repr(ref_banks[l], anchors[l])

        # Test data
        test_n_ml: list[dict[int, np.ndarray]] = []
        test_a_ml: list[dict[int, np.ndarray]] = []
        test_dir = root / cat / "test_public"
        if test_dir.is_dir():
            for sub in sorted(test_dir.iterdir()):
                if not sub.is_dir():
                    continue
                is_normal = sub.name == "good"
                tgt = test_n_ml if is_normal else test_a_ml
                count = 0
                for p in sorted(sub.iterdir()):
                    if p.suffix.lower() not in (".png", ".jpg", ".jpeg"):
                        continue
                    if count >= max_per_cat:
                        break
                    ml = extractor.extract_patches(Image.open(p).convert("RGB"))
                    tgt.append(ml)
                    count += 1

        dataset[cat] = {
            "ref_banks": ref_banks,
            "anchors": anchors,
            "ref_rel": ref_rel,
            "test_n": test_n_ml,
            "test_a": test_a_ml,
        }
        print(f"    ref={len(train_ml_patches)}, anchors={anchor_n}, "
              f"test_n={len(test_n_ml)}, test_a={len(test_a_ml)}")
    return dataset


# ─── Paired Data for D1 ───────────────────────────────────────────

def collect_paired_data(
    data_root: str,
    extractor: MultiLayerPatchExtractor,
    max_pairs: int = 50,
) -> dict:
    """Collect paired (regular vs shifted) normal test images for D1 diagnostic."""
    root = Path(data_root)
    cats = sorted(d.name for d in root.iterdir() if d.is_dir() and (d / "test_public").is_dir())

    paired = {}
    for cat in cats:
        good_dir = root / cat / "test_public" / "good"
        if not good_dir.is_dir():
            continue

        # Group by object ID
        groups: dict[str, dict[str, Path]] = defaultdict(dict)
        for p in sorted(good_dir.glob("*.png")):
            parts = p.stem.split("_", 1)
            if len(parts) == 2:
                groups[parts[0]][parts[1]] = p

        reg_feats: list[dict[int, np.ndarray]] = []
        shift_feats: list[dict[int, np.ndarray]] = []
        count = 0

        for oid, conds in sorted(groups.items()):
            if "regular" not in conds:
                continue
            ml_reg = extractor.extract_patches(Image.open(conds["regular"]).convert("RGB"))

            for c, path in sorted(conds.items()):
                if c == "regular":
                    continue
                ml_shift = extractor.extract_patches(Image.open(path).convert("RGB"))
                reg_feats.append(ml_reg)
                shift_feats.append(ml_shift)
                count += 1
                if count >= max_pairs:
                    break
            if count >= max_pairs:
                break

        if reg_feats:
            paired[cat] = {"reg": reg_feats, "shift": shift_feats}
            print(f"  {cat}: {len(reg_feats)} pairs")

    return paired


# ─── D1: Shift Sensitivity Reduction ──────────────────────────────

def run_d1_diagnostic(
    paired: dict,
    dataset: dict,
    layers: tuple[int, ...],
    k: int = 3,
) -> dict:
    """Compare shift sensitivity: raw kNN vs relative kNN vs cosine kNN."""
    print("\n" + "=" * 70)
    print("D1: Shift Sensitivity Reduction")
    print("=" * 70)

    results = {}
    for cat in sorted(paired.keys()):
        if cat not in dataset:
            continue

        anchors = dataset[cat]["anchors"]
        ref_banks = dataset[cat]["ref_banks"]
        ref_rel = dataset[cat]["ref_rel"]

        raw_deltas, rel_deltas, cos_deltas = [], [], []
        raw_deltas_ml, rel_deltas_ml = [], []

        for ml_reg, ml_shift in zip(paired[cat]["reg"], paired[cat]["shift"]):
            for l in layers:
                # Raw L2 kNN
                nn_raw = NearestNeighbors(n_neighbors=k, metric="euclidean").fit(ref_banks[l])
                s_reg_raw = nn_raw.kneighbors(ml_reg[l])[0].mean(1)
                s_shift_raw = nn_raw.kneighbors(ml_shift[l])[0].mean(1)
                raw_delta = np.abs(s_shift_raw.mean() - s_reg_raw.mean())
                raw_deltas.append(raw_delta)

                # Cosine kNN
                nn_cos = NearestNeighbors(n_neighbors=k, metric="cosine").fit(ref_banks[l])
                s_reg_cos = nn_cos.kneighbors(ml_reg[l])[0].mean(1)
                s_shift_cos = nn_cos.kneighbors(ml_shift[l])[0].mean(1)
                cos_delta = np.abs(s_shift_cos.mean() - s_reg_cos.mean())
                cos_deltas.append(cos_delta)

                # Relative representation kNN
                rel_reg = build_relative_repr(ml_reg[l], anchors[l])
                rel_shift = build_relative_repr(ml_shift[l], anchors[l])
                nn_rel = NearestNeighbors(n_neighbors=k, metric="euclidean").fit(ref_rel[l])
                s_reg_rel = nn_rel.kneighbors(rel_reg)[0].mean(1)
                s_shift_rel = nn_rel.kneighbors(rel_shift)[0].mean(1)
                rel_delta = np.abs(s_shift_rel.mean() - s_reg_rel.mean())
                rel_deltas.append(rel_delta)

            # Multi-layer relative representation
            ml_rel_reg = build_multilayer_relative_repr(ml_reg, anchors, layers)
            ml_rel_shift = build_multilayer_relative_repr(ml_shift, anchors, layers)
            ref_ml_rel = np.concatenate(
                [ref_rel[l] for l in layers], axis=1
            )
            nn_ml = NearestNeighbors(n_neighbors=k, metric="euclidean").fit(ref_ml_rel)
            s_reg_ml = nn_ml.kneighbors(ml_rel_reg)[0].mean(1)
            s_shift_ml = nn_ml.kneighbors(ml_rel_shift)[0].mean(1)
            rel_deltas_ml.append(np.abs(s_shift_ml.mean() - s_reg_ml.mean()))

            # Multi-layer raw concat
            raw_reg_ml = np.concatenate([ml_reg[l] for l in layers], axis=1)
            raw_shift_ml = np.concatenate([ml_shift[l] for l in layers], axis=1)
            ref_raw_ml = np.concatenate([ref_banks[l] for l in layers], axis=1)
            nn_raw_ml = NearestNeighbors(n_neighbors=k, metric="euclidean").fit(ref_raw_ml)
            s_reg_raw_ml = nn_raw_ml.kneighbors(raw_reg_ml)[0].mean(1)
            s_shift_raw_ml = nn_raw_ml.kneighbors(raw_shift_ml)[0].mean(1)
            raw_deltas_ml.append(np.abs(s_shift_raw_ml.mean() - s_reg_raw_ml.mean()))

        mean_raw = np.mean(raw_deltas)
        mean_cos = np.mean(cos_deltas)
        mean_rel = np.mean(rel_deltas)
        mean_raw_ml = np.mean(raw_deltas_ml)
        mean_rel_ml = np.mean(rel_deltas_ml)

        reduction_pct = (1 - mean_rel / (mean_raw + 1e-8)) * 100
        reduction_ml_pct = (1 - mean_rel_ml / (mean_raw_ml + 1e-8)) * 100

        results[cat] = {
            "raw_delta": float(mean_raw),
            "cosine_delta": float(mean_cos),
            "relative_delta": float(mean_rel),
            "raw_ml_delta": float(mean_raw_ml),
            "relative_ml_delta": float(mean_rel_ml),
            "reduction_pct": float(reduction_pct),
            "reduction_ml_pct": float(reduction_ml_pct),
        }
        go = "GO" if reduction_pct > 50 else ("MARGINAL" if reduction_pct > 25 else "NO-GO")
        print(f"  {cat:15s}: raw={mean_raw:.4f}  cos={mean_cos:.4f}  "
              f"rel={mean_rel:.4f}  reduction={reduction_pct:.1f}%  [{go}]")
        print(f"  {'':15s}  raw_ml={mean_raw_ml:.4f}  rel_ml={mean_rel_ml:.4f}  "
              f"reduction_ml={reduction_ml_pct:.1f}%")

    # Summary
    all_reductions = [v["reduction_pct"] for v in results.values()]
    all_reductions_ml = [v["reduction_ml_pct"] for v in results.values()]
    mean_reduction = np.mean(all_reductions)
    mean_reduction_ml = np.mean(all_reductions_ml)
    print(f"\n  >>> OVERALL: per-layer reduction={mean_reduction:.1f}%, "
          f"multi-layer reduction={mean_reduction_ml:.1f}%")
    verdict = "GO" if mean_reduction > 50 else ("MARGINAL" if mean_reduction > 25 else "NO-GO")
    print(f"  >>> D1 VERDICT: {verdict}")

    results["_summary"] = {
        "mean_reduction_pct": float(mean_reduction),
        "mean_reduction_ml_pct": float(mean_reduction_ml),
        "verdict": verdict,
    }
    return results


# ─── D2: AUROC Comparison ─────────────────────────────────────────

def compute_auroc(
    dataset: dict,
    layers: tuple[int, ...],
    k: int = 3,
    agg: str = "max",
) -> dict:
    """Compute image-level AUROC with multiple scoring methods."""
    print("\n" + "=" * 70)
    print(f"D2: AUROC Comparison (k={k}, agg={agg})")
    print("=" * 70)

    methods = [
        "raw_L8",
        "raw_L11",
        "raw_ML",
        "cosine_L8",
        "cosine_L11",
        "rel_L8",
        "rel_L11",
        "rel_ML",
        "rel_L8_IIN",
        "rel_L11_IIN",
        "rel_ML_IIN",
        "raw_L8_IIN",
        "raw_L11_IIN",
    ]
    results: dict[str, dict[str, float]] = {m: {} for m in methods}

    for cat, data in sorted(dataset.items()):
        ref_banks = data["ref_banks"]
        anchors = data["anchors"]
        ref_rel = data["ref_rel"]

        if len(data["test_n"]) == 0 or len(data["test_a"]) == 0:
            continue

        # Pre-build reference multi-layer representations
        ref_ml_rel = np.concatenate([ref_rel[l] for l in layers], axis=1)
        ref_ml_raw = np.concatenate([ref_banks[l] for l in layers], axis=1)

        # Pre-fit NNs
        nns_raw = {l: NearestNeighbors(n_neighbors=k, metric="euclidean").fit(ref_banks[l])
                    for l in layers}
        nns_cos = {l: NearestNeighbors(n_neighbors=k, metric="cosine").fit(ref_banks[l])
                    for l in layers}
        nns_rel = {l: NearestNeighbors(n_neighbors=k, metric="euclidean").fit(ref_rel[l])
                    for l in layers}
        nn_ml_raw = NearestNeighbors(n_neighbors=k, metric="euclidean").fit(ref_ml_raw)
        nn_ml_rel = NearestNeighbors(n_neighbors=k, metric="euclidean").fit(ref_ml_rel)

        scores_by_method: dict[str, list[float]] = {m: [] for m in methods}
        labels: list[int] = []

        for ml_patches, label in [
            *[(ml, 0) for ml in data["test_n"]],
            *[(ml, 1) for ml in data["test_a"]],
        ]:
            labels.append(label)

            # Per-layer raw
            for l_tag, l_idx in [("L8", 8), ("L11", 11)]:
                if l_idx not in ref_banks:
                    continue
                # Raw L2
                s = nns_raw[l_idx].kneighbors(ml_patches[l_idx])[0].mean(1)
                scores_by_method[f"raw_{l_tag}"].append(aggregate_image_score(s, agg))
                scores_by_method[f"raw_{l_tag}_IIN"].append(
                    aggregate_image_score(intra_image_normalize(s), agg)
                )

                # Cosine
                s_cos = nns_cos[l_idx].kneighbors(ml_patches[l_idx])[0].mean(1)
                scores_by_method[f"cosine_{l_tag}"].append(aggregate_image_score(s_cos, agg))

                # Relative
                rel = build_relative_repr(ml_patches[l_idx], anchors[l_idx])
                s_rel = nns_rel[l_idx].kneighbors(rel)[0].mean(1)
                scores_by_method[f"rel_{l_tag}"].append(aggregate_image_score(s_rel, agg))
                scores_by_method[f"rel_{l_tag}_IIN"].append(
                    aggregate_image_score(intra_image_normalize(s_rel), agg)
                )

            # Multi-layer raw
            raw_ml = np.concatenate([ml_patches[l] for l in layers], axis=1)
            s_ml_raw = nn_ml_raw.kneighbors(raw_ml)[0].mean(1)
            scores_by_method["raw_ML"].append(aggregate_image_score(s_ml_raw, agg))

            # Multi-layer relative
            ml_rel = build_multilayer_relative_repr(ml_patches, anchors, layers)
            s_ml_rel = nn_ml_rel.kneighbors(ml_rel)[0].mean(1)
            scores_by_method["rel_ML"].append(aggregate_image_score(s_ml_rel, agg))
            scores_by_method["rel_ML_IIN"].append(
                aggregate_image_score(intra_image_normalize(s_ml_rel), agg)
            )

        labels_arr = np.array(labels)
        for m in methods:
            s = np.array(scores_by_method[m])
            if len(s) == len(labels_arr) and len(np.unique(labels_arr)) == 2:
                try:
                    results[m][cat] = float(roc_auc_score(labels_arr, s))
                except ValueError:
                    results[m][cat] = 0.5

    # Print table
    cats = sorted(set().union(*(results[m].keys() for m in methods)))
    header = f"{'Method':20s}" + "".join(f"{c:>12s}" for c in cats) + f"{'MEAN':>12s}"
    print(header)
    print("-" * len(header))

    for m in methods:
        vals = [results[m].get(c, float("nan")) for c in cats]
        mean_v = np.nanmean(vals) if vals else 0
        row = f"{m:20s}" + "".join(f"{v:>12.1f}" if not np.isnan(v) else f"{'N/A':>12s}" for v in [v * 100 for v in vals])
        row += f"{mean_v * 100:>12.1f}"
        print(row)

    # D2 verdict
    raw_l11_mean = np.nanmean([results["raw_L11"].get(c, float("nan")) for c in cats])
    rel_ml_iin_mean = np.nanmean([results["rel_ML_IIN"].get(c, float("nan")) for c in cats])
    improvement = (rel_ml_iin_mean - raw_l11_mean) * 100

    print(f"\n  >>> raw_L11 mean: {raw_l11_mean*100:.1f}%")
    print(f"  >>> rel_ML_IIN mean: {rel_ml_iin_mean*100:.1f}%")
    print(f"  >>> Improvement: {improvement:+.1f}pp")
    verdict = "GO" if improvement > 5 else ("MARGINAL" if improvement > 2 else "NO-GO")
    print(f"  >>> D2 VERDICT: {verdict}")

    return {
        "per_method": {m: {c: float(v) for c, v in results[m].items()} for m in methods},
        "summary": {
            "raw_L11_mean": float(raw_l11_mean),
            "rel_ML_IIN_mean": float(rel_ml_iin_mean),
            "improvement_pp": float(improvement),
            "verdict": verdict,
        },
    }


# ─── Main ─────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Relative Representation Diagnostic")
    parser.add_argument("--data_root", type=str, default="/Volume/DATA/mvtec_ad_2")
    parser.add_argument("--K_shot", type=int, default=8)
    parser.add_argument("--coreset_n", type=int, default=2000)
    parser.add_argument("--max_per_cat", type=int, default=200)
    parser.add_argument("--k_neighbors", type=int, default=3)
    parser.add_argument("--agg", type=str, default="max", choices=["max", "p95", "mean"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, default="results/relative_repr_diagnostic")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    layers = (4, 7, 8, 11)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Config: K_shot={args.K_shot}, layers={layers}, k={args.k_neighbors}, "
          f"agg={args.agg}, coreset={args.coreset_n}")

    # Extract features
    print("\n[1/4] Loading model and extracting features...")
    extractor = MultiLayerPatchExtractor(
        device=args.device,
        target_layers=layers,
    )

    print("\n[2/4] Collecting few-shot reference + test data...")
    dataset = collect_data(
        args.data_root, extractor, K_shot=args.K_shot,
        max_per_cat=args.max_per_cat, seed=args.seed,
        coreset_n=args.coreset_n,
    )

    print("\n[3/4] Collecting paired data for D1...")
    paired = collect_paired_data(args.data_root, extractor, max_pairs=50)

    # D1: Shift sensitivity
    print("\n[4a/4] Running D1: Shift Sensitivity Reduction...")
    d1_results = run_d1_diagnostic(paired, dataset, layers, k=args.k_neighbors)

    # D2: AUROC comparison
    print("\n[4b/4] Running D2: AUROC Comparison...")
    d2_results = compute_auroc(dataset, layers, k=args.k_neighbors, agg=args.agg)

    # Save results
    all_results = {
        "config": {
            "K_shot": args.K_shot,
            "layers": list(layers),
            "k_neighbors": args.k_neighbors,
            "agg": args.agg,
            "coreset_n": args.coreset_n,
            "seed": args.seed,
        },
        "d1_shift_sensitivity": d1_results,
        "d2_auroc": d2_results,
    }
    out_path = output_dir / "diagnostic_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"  D1 (Shift Reduction):  {d1_results['_summary']['verdict']}  "
          f"({d1_results['_summary']['mean_reduction_pct']:.1f}% reduction)")
    print(f"  D2 (AUROC Improvement): {d2_results['summary']['verdict']}  "
          f"({d2_results['summary']['improvement_pp']:+.1f}pp)")

    d1_go = d1_results["_summary"]["verdict"] in ("GO", "MARGINAL")
    d2_go = d2_results["summary"]["verdict"] in ("GO", "MARGINAL")
    if d1_go and d2_go:
        print("\n  >>> OVERALL: PROCEED with Relative Representation method")
    elif d1_go:
        print("\n  >>> OVERALL: PARTIAL — relative rep helps shift, but AUROC gain insufficient")
        print("  >>> Consider: combining with other scoring or pivoting to analysis paper")
    else:
        print("\n  >>> OVERALL: NO-GO — relative representation does not sufficiently reduce shift")
        print("  >>> Next: try Inter-Layer Feature Graph (Candidate C) diagnostic")


if __name__ == "__main__":
    main()
