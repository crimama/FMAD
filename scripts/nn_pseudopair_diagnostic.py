#!/usr/bin/env python3
"""NN Pseudo-Pairing Diagnostic: SNR measurement.

Core question: Is the NN residual (noise) small enough relative to
the shift (signal) for pseudo_shift to be meaningful?

Measures:
  D0: NN distance vs oracle shift magnitude (per-category, per-layer)
  D1: Pseudo-shift basis vs NSP oracle basis alignment
  D2: Oracle shift variance explained by pseudo-shift basis

Two variants tested:
  - L8 match + L8 shift (original)
  - L8 match + L11 shift (cross-layer, 2.4x SNR expected)
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm


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


# ─── Data Collection ───────────────────────────────────────────────

def collect_paired_and_train(
    data_root: str,
    extractor: MultiLayerExtractor,
    max_per_cat: int = 100,
    max_pairs: int = 300,
) -> dict:
    """Collect train features + paired test features for all categories."""
    root = Path(data_root)
    categories = sorted(
        d.name for d in root.iterdir()
        if d.is_dir() and (d / "test_public").is_dir()
    )

    all_data = {}
    for cat in categories:
        cat_data = {
            "train": {L: [] for L in extractor.target_layers},
            "test_regular": {L: [] for L in extractor.target_layers},
            "test_shifted": {L: [] for L in extractor.target_layers},
            "oracle_shifts": {L: [] for L in extractor.target_layers},
        }

        # Train normal
        train_good = root / cat / "train" / "good"
        if train_good.is_dir():
            imgs = sorted(p for p in train_good.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
            for img_path in imgs[:max_per_cat]:
                feats = extractor.extract(Image.open(img_path).convert("RGB"))
                for L in extractor.target_layers:
                    cat_data["train"][L].append(feats[L])

        # Test paired (regular + shifted)
        good_dir = root / cat / "test_public" / "good"
        if not good_dir.is_dir():
            continue

        groups: dict[str, dict[str, Path]] = defaultdict(dict)
        for img_path in sorted(good_dir.glob("*.png")):
            parts = img_path.stem.split("_", 1)
            if len(parts) == 2:
                obj_idx, condition = parts
                groups[obj_idx][condition] = img_path

        pair_count = 0
        for obj_idx, conditions in sorted(groups.items()):
            if "regular" not in conditions:
                continue
            feat_reg = extractor.extract(Image.open(conditions["regular"]).convert("RGB"))
            for cond, path in sorted(conditions.items()):
                if cond == "regular":
                    continue
                feat_shift = extractor.extract(Image.open(path).convert("RGB"))
                for L in extractor.target_layers:
                    cat_data["test_regular"][L].append(feat_reg[L])
                    cat_data["test_shifted"][L].append(feat_shift[L])
                    cat_data["oracle_shifts"][L].append(feat_shift[L] - feat_reg[L])
                pair_count += 1
            if pair_count >= max_pairs:
                break

        # Stack arrays
        for key in cat_data:
            for L in extractor.target_layers:
                if cat_data[key][L]:
                    cat_data[key][L] = np.stack(cat_data[key][L])
                else:
                    cat_data[key][L] = np.zeros((0, 1536))

        n_train = len(cat_data["train"][extractor.target_layers[0]])
        n_pairs = len(cat_data["oracle_shifts"][extractor.target_layers[0]])
        print(f"  {cat}: train={n_train}, pairs={n_pairs}")
        all_data[cat] = cat_data

    return all_data


# ─── D0: SNR Measurement ──────────────────────────────────────────

def diagnostic_d0(all_data: dict, match_layer: int, shift_layer: int) -> dict:
    """D0: NN distance (noise) vs oracle shift magnitude (signal).

    For each shifted test image:
      1. Find NN in train at match_layer
      2. Compute NN distance at match_layer (matching quality)
      3. Compute oracle shift magnitude at shift_layer
      4. Compute pseudo_shift = f_shift_layer(x) - f_shift_layer(NN)
      5. Compare pseudo_shift direction with oracle shift direction
    """
    results = {}

    for cat, data in sorted(all_data.items()):
        train_match = data["train"][match_layer]  # (N_train, dim)
        test_shifted_match = data["test_shifted"][match_layer]
        test_shifted_shift = data["test_shifted"][shift_layer]
        train_shift = data["train"][shift_layer]
        oracle_shifts = data["oracle_shifts"][shift_layer]  # (N_pairs, dim)

        if len(train_match) == 0 or len(test_shifted_match) == 0:
            continue

        nn_dists = []
        shift_mags = []
        pseudo_oracle_cosines = []
        pseudo_shifts = []

        for i in range(len(test_shifted_match)):
            # NN matching at match_layer
            dists = np.linalg.norm(train_match - test_shifted_match[i], axis=1)
            nn_idx = np.argmin(dists)
            nn_dist = dists[nn_idx]

            # Oracle shift magnitude at shift_layer
            shift_mag = np.linalg.norm(oracle_shifts[i])

            # Pseudo-shift at shift_layer
            pseudo_shift = test_shifted_shift[i] - train_shift[nn_idx]

            # Cosine similarity between pseudo_shift and oracle shift
            cos_sim = 0.0
            if shift_mag > 1e-8 and np.linalg.norm(pseudo_shift) > 1e-8:
                cos_sim = np.dot(pseudo_shift, oracle_shifts[i]) / (
                    np.linalg.norm(pseudo_shift) * shift_mag
                )

            nn_dists.append(nn_dist)
            shift_mags.append(shift_mag)
            pseudo_oracle_cosines.append(cos_sim)
            pseudo_shifts.append(pseudo_shift)

        nn_dists = np.array(nn_dists)
        shift_mags = np.array(shift_mags)
        cosines = np.array(pseudo_oracle_cosines)

        snr = np.median(shift_mags) / (np.median(nn_dists) + 1e-8)

        results[cat] = {
            "nn_dist_median": float(np.median(nn_dists)),
            "nn_dist_mean": float(np.mean(nn_dists)),
            "shift_mag_median": float(np.median(shift_mags)),
            "shift_mag_mean": float(np.mean(shift_mags)),
            "snr": float(snr),
            "pseudo_oracle_cos_mean": float(np.mean(cosines)),
            "pseudo_oracle_cos_median": float(np.median(cosines)),
            "n_pairs": len(nn_dists),
        }
        pseudo_shifts_arr = np.stack(pseudo_shifts)
        results[cat]["_pseudo_shifts"] = pseudo_shifts_arr
        results[cat]["_oracle_shifts"] = oracle_shifts

    # Aggregate
    all_snr = [v["snr"] for v in results.values() if "snr" in v]
    all_cos = [v["pseudo_oracle_cos_mean"] for v in results.values() if "pseudo_oracle_cos_mean" in v]
    results["MEAN"] = {
        "snr": float(np.mean(all_snr)),
        "pseudo_oracle_cos_mean": float(np.mean(all_cos)),
    }

    return results


# ─── D1/D2: Subspace Alignment ────────────────────────────────────

def subspace_diagnostic(d0_results: dict, K: int = 100) -> dict:
    """D1+D2: Compare pseudo_shift basis with oracle basis."""
    # Collect all pseudo_shifts and oracle_shifts
    all_pseudo = []
    all_oracle = []
    for cat, v in d0_results.items():
        if cat == "MEAN" or "_pseudo_shifts" not in v:
            continue
        all_pseudo.append(v["_pseudo_shifts"])
        all_oracle.append(v["_oracle_shifts"])

    pseudo = np.vstack(all_pseudo)
    oracle = np.vstack(all_oracle)

    # PCA on pseudo_shifts
    pc = pseudo - pseudo.mean(axis=0)
    Up, Sp, Vtp = np.linalg.svd(pc, full_matrices=False)
    pseudo_basis = Vtp[:K].T  # (dim, K)

    # PCA on oracle shifts
    oc = oracle - oracle.mean(axis=0)
    Uo, So, Vto = np.linalg.svd(oc, full_matrices=False)
    oracle_basis = Vto[:K].T

    # D1: Principal angles
    Q_p, _ = np.linalg.qr(pseudo_basis)
    Q_o, _ = np.linalg.qr(oracle_basis)
    k = min(Q_p.shape[1], Q_o.shape[1])
    M = Q_p[:, :k].T @ Q_o[:, :k]
    _, sigmas, _ = np.linalg.svd(M)
    sigmas = np.clip(sigmas, 0, 1)
    angles = np.degrees(np.arccos(sigmas))

    # D2: Shift variance explained
    proj_coeffs = oc @ pseudo_basis
    reconstructed = proj_coeffs @ pseudo_basis.T
    explained = np.sum(reconstructed ** 2) / np.sum(oc ** 2)

    # Pseudo-shift variance explained by oracle basis (reverse)
    proj_coeffs_r = pc @ oracle_basis
    reconstructed_r = proj_coeffs_r @ oracle_basis.T
    reverse_explained = np.sum(reconstructed_r ** 2) / np.sum(pc ** 2)

    return {
        "d1_mean_angle": float(angles.mean()),
        "d1_median_angle": float(np.median(angles)),
        "d1_frac_below_45": float(np.mean(angles < 45)),
        "d1_top10_angles": angles[:10].tolist(),
        "d2_explained": float(explained),
        "d2_reverse_explained": float(reverse_explained),
        "pseudo_variance_top100": float((Sp[:K] ** 2).sum() / (Sp ** 2).sum()),
        "oracle_variance_top100": float((So[:K] ** 2).sum() / (So ** 2).sum()),
    }


# ─── Main ──────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> None:
    print("=" * 60)
    print("NN Pseudo-Pairing Diagnostic: SNR + Subspace Alignment")
    print("=" * 60)

    extractor = MultiLayerExtractor(
        model_name=args.model_name, device=args.device, target_layers=(8, 11),
    )

    print("\n[1/4] Collecting train + paired test features (L8, L11)...")
    all_data = collect_paired_and_train(
        args.data_root, extractor, max_per_cat=args.max_per_cat, max_pairs=args.max_pairs,
    )

    # ── D0: SNR at L8 match + L8 shift ──
    print("\n[2/4] D0-A: SNR (L8 match, L8 shift)...")
    d0_l8l8 = diagnostic_d0(all_data, match_layer=8, shift_layer=8)
    print(f"\n  Per-category (L8→L8):")
    print(f"  {'Category':<15s} {'NN dist':>10s} {'Shift mag':>10s} {'SNR':>8s} {'Cos(pseudo,oracle)':>18s}")
    print(f"  {'-'*65}")
    for cat in sorted(d0_l8l8.keys()):
        if cat == "MEAN" or cat.startswith("_"):
            continue
        v = d0_l8l8[cat]
        print(f"  {cat:<15s} {v['nn_dist_median']:>10.4f} {v['shift_mag_median']:>10.4f} "
              f"{v['snr']:>8.3f} {v['pseudo_oracle_cos_mean']:>18.3f}")
    v = d0_l8l8["MEAN"]
    print(f"  {'MEAN':<15s} {'':>10s} {'':>10s} {v['snr']:>8.3f} {v['pseudo_oracle_cos_mean']:>18.3f}")

    # ── D0: SNR at L8 match + L11 shift (cross-layer) ──
    print(f"\n[3/4] D0-B: SNR (L8 match, L11 shift — cross-layer)...")
    d0_l8l11 = diagnostic_d0(all_data, match_layer=8, shift_layer=11)
    print(f"\n  Per-category (L8→L11):")
    print(f"  {'Category':<15s} {'NN dist':>10s} {'Shift mag':>10s} {'SNR':>8s} {'Cos(pseudo,oracle)':>18s}")
    print(f"  {'-'*65}")
    for cat in sorted(d0_l8l11.keys()):
        if cat == "MEAN" or cat.startswith("_"):
            continue
        v = d0_l8l11[cat]
        print(f"  {cat:<15s} {v['nn_dist_median']:>10.4f} {v['shift_mag_median']:>10.4f} "
              f"{v['snr']:>8.3f} {v['pseudo_oracle_cos_mean']:>18.3f}")
    v = d0_l8l11["MEAN"]
    print(f"  {'MEAN':<15s} {'':>10s} {'':>10s} {v['snr']:>8.3f} {v['pseudo_oracle_cos_mean']:>18.3f}")

    # ── D1+D2: Subspace alignment for the better variant ──
    print(f"\n[4/4] D1+D2: Subspace alignment (both variants)...")
    sub_l8l8 = subspace_diagnostic(d0_l8l8, K=args.K)
    sub_l8l11 = subspace_diagnostic(d0_l8l11, K=args.K)

    for label, sub in [("L8→L8", sub_l8l8), ("L8→L11", sub_l8l11)]:
        print(f"\n  [{label}]")
        print(f"    D1 mean angle:    {sub['d1_mean_angle']:.1f}°")
        print(f"    D1 frac < 45°:    {sub['d1_frac_below_45']:.1%}")
        print(f"    D1 top-10 angles: {[round(a,1) for a in sub['d1_top10_angles']]}")
        print(f"    D2 oracle explained by pseudo: {sub['d2_explained']:.1%}")
        print(f"    D2 pseudo explained by oracle: {sub['d2_reverse_explained']:.1%}")

    # ── Summary ──
    print("\n" + "=" * 60)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 60)

    # GO/NO-GO criteria
    snr_l8 = d0_l8l8["MEAN"]["snr"]
    snr_l11 = d0_l8l11["MEAN"]["snr"]
    cos_l8 = d0_l8l8["MEAN"]["pseudo_oracle_cos_mean"]
    cos_l11 = d0_l8l11["MEAN"]["pseudo_oracle_cos_mean"]
    d1_l8 = sub_l8l8["d1_mean_angle"]
    d1_l11 = sub_l8l11["d1_mean_angle"]
    d2_l8 = sub_l8l8["d2_explained"]
    d2_l11 = sub_l8l11["d2_explained"]

    print(f"\n  {'Metric':<30s} {'L8→L8':>10s} {'L8→L11':>10s} {'GO criterion':>15s}")
    print(f"  {'-'*70}")
    print(f"  {'D0: SNR (shift/nn_dist)':<30s} {snr_l8:>10.3f} {snr_l11:>10.3f} {'> 0.3':>15s}")
    print(f"  {'D0: Cos(pseudo, oracle)':<30s} {cos_l8:>10.3f} {cos_l11:>10.3f} {'> 0.3':>15s}")
    print(f"  {'D1: Mean principal angle':<30s} {d1_l8:>9.1f}° {d1_l11:>9.1f}° {'< 60°':>15s}")
    print(f"  {'D2: Oracle shift explained':<30s} {d2_l8:>9.1%} {d2_l11:>9.1%} {'> 30%':>15s}")

    # Best variant
    best = "L8→L11" if d2_l11 > d2_l8 else "L8→L8"
    best_d1 = d1_l11 if best == "L8→L11" else d1_l8
    best_d2 = d2_l11 if best == "L8→L11" else d2_l8
    best_cos = cos_l11 if best == "L8→L11" else cos_l8

    d1_pass = best_d1 < 60
    d2_pass = best_d2 > 0.30
    cos_pass = best_cos > 0.3

    if d1_pass and d2_pass:
        verdict = "GO"
        msg = f"NN pseudo-pairing ({best}) captures meaningful shift directions."
    elif d1_pass or d2_pass or cos_pass:
        verdict = "CONDITIONAL"
        msg = f"Partial signal detected in {best}. Proceed with caution."
    else:
        verdict = "NO-GO"
        msg = "NN pseudo-shift does not align with oracle shift. SNR too low."

    print(f"\n  Best variant: {best}")
    print(f"  VERDICT: {verdict}")
    print(f"  {msg}")

    # ── Save ──
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Remove non-serializable arrays
    save_d0_l8l8 = {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
                     for k, v in d0_l8l8.items()}
    save_d0_l8l11 = {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
                      for k, v in d0_l8l11.items()}

    results = {
        "d0_l8l8": save_d0_l8l8,
        "d0_l8l11": save_d0_l8l11,
        "subspace_l8l8": sub_l8l8,
        "subspace_l8l11": sub_l8l11,
        "verdict": verdict,
        "best_variant": best,
    }
    with open(output_dir / "nn_pseudo_diagnostic.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {output_dir / 'nn_pseudo_diagnostic.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="/home/hun/Volume/DATA/mvtec_ad_2")
    parser.add_argument("--model_name", type=str, default="dinov2_vitb14")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--K", type=int, default=100)
    parser.add_argument("--max_per_cat", type=int, default=100)
    parser.add_argument("--max_pairs", type=int, default=300)
    parser.add_argument("--output_dir", type=str, default="results/nn_pseudo_diagnostic")
    args = parser.parse_args()
    run(args)
