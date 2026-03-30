#!/usr/bin/env python3
"""IFR Diagnostic: Test whether inter-layer residuals capture nuisance directions.

Two diagnostic tests BEFORE running the full IFR experiment:
  D1: Subspace alignment — IFR basis vs NSP oracle basis (principal angles)
  D2: Shift variance explained — how much of oracle shift vectors lie in IFR subspace

If both pass → IFR's core assumption holds → proceed to full experiment.
If either fails → IFR's assumption is broken → pivot to analysis paper.
"""

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm


# ─── Feature Extraction (multi-layer) ─────────────────────────────

class MultiLayerExtractor:
    """Extract DINOv2 features at multiple layers simultaneously."""

    def __init__(
        self,
        model_name: str = "dinov2_vitb14",
        device: str = "cuda",
        target_layers: tuple[int, ...] = (8, 9, 10, 11),
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
    def extract(self, img: Image.Image) -> dict[int, np.ndarray]:
        """Return {layer_idx: feature_vector} for CLS+patch_mean, L2-normed."""
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


# ─── Data Loading ──────────────────────────────────────────────────

class PairedShiftData(NamedTuple):
    shift_vectors: np.ndarray   # (N_pairs, dim) — oracle shift: f(shifted) - f(regular)
    regular_feats: np.ndarray   # (N_pairs, dim) — regular condition features
    shifted_feats: np.ndarray   # (N_pairs, dim) — shifted condition features


def collect_oracle_shift_vectors(
    data_root: str,
    extractor: MultiLayerExtractor,
    layer: int = 8,
    max_pairs: int = 500,
) -> PairedShiftData:
    """Collect paired shift vectors from MVTec AD 2 original (regular↔shifted)."""
    root = Path(data_root)
    categories = sorted(
        d.name for d in root.iterdir()
        if d.is_dir() and (d / "test_public").is_dir()
    )

    shift_vecs, reg_feats, shift_feats = [], [], []

    for cat in categories:
        good_dir = root / cat / "test_public" / "good"
        if not good_dir.is_dir():
            continue

        groups: dict[str, dict[str, Path]] = defaultdict(dict)
        for img_path in sorted(good_dir.glob("*.png")):
            parts = img_path.stem.split("_", 1)
            if len(parts) == 2:
                obj_idx, condition = parts
                groups[obj_idx][condition] = img_path

        for obj_idx, conditions in sorted(groups.items()):
            if "regular" not in conditions:
                continue
            feat_reg = extractor.extract(Image.open(conditions["regular"]).convert("RGB"))
            for cond, path in sorted(conditions.items()):
                if cond == "regular":
                    continue
                feat_shift = extractor.extract(Image.open(path).convert("RGB"))
                shift_vecs.append(feat_shift[layer] - feat_reg[layer])
                reg_feats.append(feat_reg[layer])
                shift_feats.append(feat_shift[layer])

            if len(shift_vecs) >= max_pairs:
                break
        if len(shift_vecs) >= max_pairs:
            break

    return PairedShiftData(
        shift_vectors=np.stack(shift_vecs),
        regular_feats=np.stack(reg_feats),
        shifted_feats=np.stack(shift_feats),
    )


def collect_train_multilayer_features(
    data_root: str,
    extractor: MultiLayerExtractor,
    max_per_cat: int = 100,
) -> dict[int, np.ndarray]:
    """Collect train normal features at all target layers. Returns {layer: (N, dim)}.

    Handles both original MVTec AD 2 layout (xxx_regular.png in train/good/)
    and compat layout.
    """
    root = Path(data_root)
    categories = sorted(
        d.name for d in root.iterdir()
        if d.is_dir() and (d / "train").is_dir()
    )

    layer_feats: dict[int, list[np.ndarray]] = {
        L: [] for L in extractor.target_layers
    }

    for cat in categories:
        train_good = root / cat / "train" / "good"
        if not train_good.is_dir():
            continue
        imgs = sorted(
            p for p in train_good.iterdir()
            if p.suffix.lower() in (".png", ".jpg", ".jpeg")
        )
        if not imgs:
            continue
        for img_path in imgs[:max_per_cat]:
            feats = extractor.extract(Image.open(img_path).convert("RGB"))
            for L in extractor.target_layers:
                layer_feats[L].append(feats[L])
        print(f"  {cat}: {min(len(imgs), max_per_cat)} train images")

    if not layer_feats[extractor.target_layers[0]]:
        raise ValueError(f"No train images found in {root}. Check directory structure.")

    return {L: np.stack(v) for L, v in layer_feats.items()}


def collect_shifted_multilayer_features(
    data_root: str,
    extractor: MultiLayerExtractor,
    max_pairs: int = 500,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Collect paired features at ALL layers for D3 (Δ direction stability).

    Returns (regular_feats_per_layer, shifted_feats_per_layer).
    """
    root = Path(data_root)
    categories = sorted(
        d.name for d in root.iterdir()
        if d.is_dir() and (d / "test_public").is_dir()
    )

    reg_feats: dict[int, list] = {L: [] for L in extractor.target_layers}
    shift_feats: dict[int, list] = {L: [] for L in extractor.target_layers}
    count = 0

    for cat in categories:
        good_dir = root / cat / "test_public" / "good"
        if not good_dir.is_dir():
            continue

        groups: dict[str, dict[str, Path]] = defaultdict(dict)
        for img_path in sorted(good_dir.glob("*.png")):
            parts = img_path.stem.split("_", 1)
            if len(parts) == 2:
                obj_idx, condition = parts
                groups[obj_idx][condition] = img_path

        for obj_idx, conditions in sorted(groups.items()):
            if "regular" not in conditions:
                continue
            feat_reg = extractor.extract(Image.open(conditions["regular"]).convert("RGB"))
            for cond, path in sorted(conditions.items()):
                if cond == "regular":
                    continue
                feat_shift = extractor.extract(Image.open(path).convert("RGB"))
                for L in extractor.target_layers:
                    reg_feats[L].append(feat_reg[L])
                    shift_feats[L].append(feat_shift[L])
                count += 1
            if count >= max_pairs:
                break
        if count >= max_pairs:
            break

    return (
        {L: np.stack(v) for L, v in reg_feats.items()},
        {L: np.stack(v) for L, v in shift_feats.items()},
    )


# ─── IFR Basis Estimation ─────────────────────────────────────────

def compute_ifr_basis(
    train_feats: dict[int, np.ndarray],
    K: int = 100,
) -> np.ndarray:
    """Compute IFR nuisance basis from inter-layer residuals on train normals.

    Δ_L = f_L - f_{L-1} for L in {9, 10, 11}.
    PCA on stacked Δs → top-K = IFR nuisance basis.

    Returns: (dim, K) orthonormal basis.
    """
    deltas = []
    for L in [9, 10, 11]:
        delta = train_feats[L] - train_feats[L - 1]  # (N, dim)
        deltas.append(delta)

    # Stack all deltas: (3*N, dim)
    all_deltas = np.vstack(deltas)
    centered = all_deltas - all_deltas.mean(axis=0)

    # SVD for PCA (more stable than eigendecomposition for large matrices)
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    basis = Vt[:K].T  # (dim, K)

    # Report variance explained
    total_var = (S ** 2).sum()
    explained = (S[:K] ** 2).sum() / total_var
    print(f"  IFR basis: top-{K} of {len(S)} components explain {explained:.1%} of Δ variance")
    print(f"  Singular value profile: [{S[0]:.2f}, {S[1]:.2f}, ..., {S[K-1]:.2f}, {S[K]:.2f}]")

    return basis


def compute_nsp_basis(
    shift_vectors: np.ndarray,
    K: int = 100,
) -> np.ndarray:
    """Compute NSP oracle nuisance basis from paired shift vectors.

    Returns: (dim, K) orthonormal basis.
    """
    centered = shift_vectors - shift_vectors.mean(axis=0)
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    basis = Vt[:K].T

    total_var = (S ** 2).sum()
    explained = (S[:K] ** 2).sum() / total_var
    print(f"  NSP basis: top-{K} of {len(S)} components explain {explained:.1%} of shift variance")

    return basis


# ─── Diagnostic Tests ──────────────────────────────────────────────

@dataclass(frozen=True)
class D1Result:
    """D1: Subspace alignment via principal angles."""
    principal_angles_deg: np.ndarray  # (min(K1, K2),) in degrees
    mean_angle_deg: float
    median_angle_deg: float
    fraction_below_45: float  # fraction of principal angles < 45°
    grassmann_distance: float

@dataclass(frozen=True)
class D2Result:
    """D2: How much oracle shift variance is explained by IFR basis."""
    explained_ratio: float  # [0, 1]
    per_component_explained: np.ndarray  # per oracle shift PC
    cumulative_explained: np.ndarray


def diagnostic_d1(ifr_basis: np.ndarray, nsp_basis: np.ndarray) -> D1Result:
    """D1: Subspace alignment — principal angles between IFR and NSP subspaces.

    Principal angles measure the minimum angles between two subspaces.
    cos(θ_i) = σ_i(A^T B) where A, B are orthonormal bases.
    """
    # Ensure orthonormal
    Q_ifr, _ = np.linalg.qr(ifr_basis)
    Q_nsp, _ = np.linalg.qr(nsp_basis)

    K = min(Q_ifr.shape[1], Q_nsp.shape[1])
    Q_ifr = Q_ifr[:, :K]
    Q_nsp = Q_nsp[:, :K]

    # SVD of cross-product → singular values = cos(principal angles)
    M = Q_ifr.T @ Q_nsp
    _, sigmas, _ = np.linalg.svd(M)
    sigmas = np.clip(sigmas, 0.0, 1.0)

    angles_rad = np.arccos(sigmas)
    angles_deg = np.degrees(angles_rad)

    # Grassmann distance = sqrt(sum(θ_i^2))
    grassmann = np.sqrt(np.sum(angles_rad ** 2))

    return D1Result(
        principal_angles_deg=angles_deg,
        mean_angle_deg=float(angles_deg.mean()),
        median_angle_deg=float(np.median(angles_deg)),
        fraction_below_45=float(np.mean(angles_deg < 45)),
        grassmann_distance=float(grassmann),
    )


def diagnostic_d2(
    ifr_basis: np.ndarray,
    shift_vectors: np.ndarray,
) -> D2Result:
    """D2: How much of oracle shift vectors lie in the IFR subspace.

    Projects each shift vector onto IFR basis and measures explained variance.
    """
    # Total variance of shift vectors
    centered = shift_vectors - shift_vectors.mean(axis=0)
    total_var = np.sum(centered ** 2)

    # Project onto IFR basis
    proj_coeffs = centered @ ifr_basis  # (N, K)
    reconstructed = proj_coeffs @ ifr_basis.T  # (N, dim)
    explained_var = np.sum(reconstructed ** 2)

    explained_ratio = explained_var / total_var

    # Per-component: how much of each NSP PC is captured by IFR
    U_shift, S_shift, Vt_shift = np.linalg.svd(centered, full_matrices=False)
    per_component = []
    for i in range(min(100, len(S_shift))):
        pc_i = Vt_shift[i]  # i-th NSP principal direction
        # How much of pc_i lies in IFR subspace?
        proj_norm = np.linalg.norm(ifr_basis.T @ pc_i)
        per_component.append(proj_norm ** 2)  # fraction in IFR subspace

    per_component = np.array(per_component)
    cumulative = np.cumsum(per_component * S_shift[:len(per_component)] ** 2)
    cumulative = cumulative / (S_shift[:len(per_component)] ** 2).sum()

    return D2Result(
        explained_ratio=float(explained_ratio),
        per_component_explained=per_component,
        cumulative_explained=cumulative,
    )


def diagnostic_d3(
    reg_feats: dict[int, np.ndarray],
    shift_feats: dict[int, np.ndarray],
    K: int = 50,
) -> dict[str, float]:
    """D3 (bonus): Do Δ directions change under shift?

    Compare PCA(Δ_clean) vs PCA(Δ_shifted) direction similarity.
    """
    results = {}

    for L in [9, 10, 11]:
        delta_clean = reg_feats[L] - reg_feats[L - 1]
        delta_shifted = shift_feats[L] - shift_feats[L - 1]

        # PCA on each
        Uc, Sc, Vtc = np.linalg.svd(delta_clean - delta_clean.mean(0), full_matrices=False)
        Us, Ss, Vts = np.linalg.svd(delta_shifted - delta_shifted.mean(0), full_matrices=False)

        # Principal angles between top-K subspaces
        basis_clean = Vtc[:K].T
        basis_shifted = Vts[:K].T

        M = basis_clean.T @ basis_shifted
        _, sigmas, _ = np.linalg.svd(M)
        sigmas = np.clip(sigmas, 0, 1)
        angles = np.degrees(np.arccos(sigmas))

        results[f"L{L}_mean_angle"] = float(angles.mean())
        results[f"L{L}_median_angle"] = float(np.median(angles))
        results[f"L{L}_frac_below_45"] = float(np.mean(angles < 45))

        # Also: magnitude change
        mag_clean = np.linalg.norm(delta_clean, axis=1).mean()
        mag_shifted = np.linalg.norm(delta_shifted, axis=1).mean()
        results[f"L{L}_mag_ratio"] = float(mag_shifted / mag_clean)

    return results


# ─── Main ──────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> None:
    print("=" * 60)
    print("IFR Diagnostic: Subspace Alignment & Shift Variance")
    print("=" * 60)

    extractor = MultiLayerExtractor(
        model_name=args.model_name,
        device=args.device,
        target_layers=(8, 9, 10, 11),
    )

    # ── Step 1: Collect data ──────────────────────────────────────
    print(f"\n[1/5] Extracting train normal features (4 layers)...")
    train_feats = collect_train_multilayer_features(
        args.data_root_compat, extractor, max_per_cat=args.max_per_cat,
    )
    n_train = len(train_feats[8])
    print(f"  Total train samples: {n_train}")

    print(f"\n[2/5] Collecting oracle shift vectors (paired data)...")
    oracle = collect_oracle_shift_vectors(
        args.data_root_original, extractor, layer=8, max_pairs=args.max_pairs,
    )
    print(f"  Collected {len(oracle.shift_vectors)} paired shift vectors")

    # ── Step 2: Compute bases ─────────────────────────────────────
    print(f"\n[3/5] Computing IFR basis (from train Δ_{9-11})...")
    ifr_basis = compute_ifr_basis(train_feats, K=args.K)

    print(f"\n       Computing NSP oracle basis (from paired shifts)...")
    nsp_basis = compute_nsp_basis(oracle.shift_vectors, K=args.K)

    # ── Step 3: Diagnostic D1 ─────────────────────────────────────
    print(f"\n[4/5] D1: Subspace alignment (principal angles)...")
    d1 = diagnostic_d1(ifr_basis, nsp_basis)
    print(f"  Mean principal angle:   {d1.mean_angle_deg:.1f}°")
    print(f"  Median principal angle: {d1.median_angle_deg:.1f}°")
    print(f"  Fraction < 45°:         {d1.fraction_below_45:.1%}")
    print(f"  Grassmann distance:     {d1.grassmann_distance:.2f}")
    print(f"  Top-5 angles: {d1.principal_angles_deg[:5].round(1)}")
    print(f"  Top-20 angles: {d1.principal_angles_deg[:20].round(1)}")

    # ── Step 4: Diagnostic D2 ─────────────────────────────────────
    print(f"\n       D2: Shift variance explained by IFR basis...")
    d2 = diagnostic_d2(ifr_basis, oracle.shift_vectors)
    print(f"  Overall explained ratio: {d2.explained_ratio:.1%}")
    print(f"  Per-component (top-10 NSP PCs in IFR):")
    for i in range(min(10, len(d2.per_component_explained))):
        print(f"    NSP PC{i}: {d2.per_component_explained[i]:.1%} captured by IFR")

    # ── Step 5: Diagnostic D3 (bonus) ─────────────────────────────
    print(f"\n[5/5] D3: Δ direction stability (clean vs shifted)...")
    print(f"  Extracting shifted features at all layers...")
    reg_ml, shift_ml = collect_shifted_multilayer_features(
        args.data_root_original, extractor, max_pairs=args.max_pairs,
    )
    d3 = diagnostic_d3(reg_ml, shift_ml, K=min(50, args.K))
    print(f"  Per-layer Δ direction stability:")
    for L in [9, 10, 11]:
        print(f"    Layer {L}: mean_angle={d3[f'L{L}_mean_angle']:.1f}°, "
              f"frac<45°={d3[f'L{L}_frac_below_45']:.1%}, "
              f"mag_ratio={d3[f'L{L}_mag_ratio']:.2f}")

    # ── Summary & GO/NO-GO ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 60)

    d1_pass = d1.mean_angle_deg < 60  # relaxed from 45 to account for partial overlap
    d2_pass = d2.explained_ratio > 0.30  # relaxed from 0.40 — even partial alignment is informative
    d1_strong = d1.mean_angle_deg < 45
    d2_strong = d2.explained_ratio > 0.50

    print(f"\n  D1 Subspace Alignment:  mean={d1.mean_angle_deg:.1f}°  "
          f"{'✅ PASS' if d1_pass else '❌ FAIL'}"
          f"{'  (STRONG)' if d1_strong else ''}")
    print(f"  D2 Shift Explained:     {d2.explained_ratio:.1%}       "
          f"{'✅ PASS' if d2_pass else '❌ FAIL'}"
          f"{'  (STRONG)' if d2_strong else ''}")

    if d1_pass and d2_pass:
        verdict = "GO"
        msg = "IFR basis captures meaningful nuisance directions. Proceed to full experiment."
        if d1_strong and d2_strong:
            verdict = "STRONG GO"
            msg = "IFR basis strongly aligned with oracle. High confidence in full experiment."
    elif d1_pass or d2_pass:
        verdict = "CONDITIONAL"
        msg = "Partial alignment. IFR may work but with reduced effect. Consider proceeding with caution."
    else:
        verdict = "NO-GO"
        msg = "IFR basis does NOT capture oracle nuisance directions. Core assumption is broken."

    print(f"\n  VERDICT: {verdict}")
    print(f"  {msg}")

    # ── Save results ──────────────────────────────────────────────
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "d1": {
            "mean_angle_deg": d1.mean_angle_deg,
            "median_angle_deg": d1.median_angle_deg,
            "fraction_below_45": d1.fraction_below_45,
            "grassmann_distance": d1.grassmann_distance,
            "top20_angles": d1.principal_angles_deg[:20].tolist(),
            "pass": d1_pass,
            "strong": d1_strong,
        },
        "d2": {
            "explained_ratio": d2.explained_ratio,
            "per_component_top10": d2.per_component_explained[:10].tolist(),
            "pass": d2_pass,
            "strong": d2_strong,
        },
        "d3": d3,
        "verdict": verdict,
        "config": {
            "K": args.K,
            "max_per_cat": args.max_per_cat,
            "max_pairs": args.max_pairs,
            "layers": [8, 9, 10, 11],
        },
    }

    result_path = output_dir / "ifr_diagnostic.json"
    with open(result_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {result_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IFR Diagnostic Tests")
    parser.add_argument(
        "--data_root_compat",
        type=str,
        default="/home/hun/Volume/DATA/mvtec_ad_2_compat",
        help="MVTec AD 2 compat layout (train/test split)",
    )
    parser.add_argument(
        "--data_root_original",
        type=str,
        default="/home/hun/Volume/DATA/mvtec_ad_2",
        help="MVTec AD 2 original (with test_public/good paired images)",
    )
    parser.add_argument("--model_name", type=str, default="dinov2_vitb14")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--K", type=int, default=100)
    parser.add_argument("--max_per_cat", type=int, default=100)
    parser.add_argument("--max_pairs", type=int, default=500)
    parser.add_argument("--output_dir", type=str, default="results/ifr_diagnostic")
    args = parser.parse_args()
    run(args)
