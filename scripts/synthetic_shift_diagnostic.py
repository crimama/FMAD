#!/usr/bin/env python3
"""Synthetic Shift Codebook Diagnostic: Does synthetic shift span real shift?

Core question: DINOv2 features에서 synthetic augmentation으로 만든 shift vectors가
real distribution shift (MVTec AD 2의 paired shifts)와 같은 subspace에 있는가?

Measures:
  D1: Principal angle between synthetic basis and oracle basis
  D2: Real shift variance explained by synthetic subspace
  D3: Per-augmentation contribution (어떤 augmentation이 가장 유용한가)
  D4: DINOv2 invariance check (augmentation shift vector magnitude)
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from torchvision import transforms
from torchvision.transforms import functional as TF


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
    def extract_image(self, img):
        """CLS + patch_mean, L2 normalized."""
        x = self.transform(img).unsqueeze(0).to(self.device)
        _ = self.model(x)
        feat = self.feature[0]
        cls = feat[0]
        pm = feat[1:].mean(0)
        c = torch.cat([cls, pm])
        return (c / (c.norm() + 1e-8)).cpu().numpy()

    @torch.no_grad()
    def extract_patches(self, img):
        """All patch tokens (M, dim)."""
        x = self.transform(img).unsqueeze(0).to(self.device)
        _ = self.model(x)
        return self.feature[0, 1:].cpu().numpy()


# ─── Augmentations ────────────────────────────────────────────────

def get_augmentations():
    """Return diverse augmentations with names."""
    return {
        "brightness_up": lambda img: ImageEnhance.Brightness(img).enhance(1.5),
        "brightness_down": lambda img: ImageEnhance.Brightness(img).enhance(0.5),
        "brightness_strong_up": lambda img: ImageEnhance.Brightness(img).enhance(2.0),
        "brightness_strong_down": lambda img: ImageEnhance.Brightness(img).enhance(0.3),
        "contrast_up": lambda img: ImageEnhance.Contrast(img).enhance(1.5),
        "contrast_down": lambda img: ImageEnhance.Contrast(img).enhance(0.5),
        "contrast_strong": lambda img: ImageEnhance.Contrast(img).enhance(2.0),
        "saturation_up": lambda img: ImageEnhance.Color(img).enhance(1.5),
        "saturation_down": lambda img: ImageEnhance.Color(img).enhance(0.5),
        "sharpness_up": lambda img: ImageEnhance.Sharpness(img).enhance(2.0),
        "blur_light": lambda img: img.filter(ImageFilter.GaussianBlur(radius=1)),
        "blur_medium": lambda img: img.filter(ImageFilter.GaussianBlur(radius=2)),
        "blur_heavy": lambda img: img.filter(ImageFilter.GaussianBlur(radius=4)),
        "color_jitter_warm": lambda img: ImageEnhance.Color(
            ImageEnhance.Brightness(img).enhance(1.1)
        ).enhance(1.2),
        "color_jitter_cool": lambda img: ImageEnhance.Color(
            ImageEnhance.Brightness(img).enhance(0.9)
        ).enhance(0.8),
        "gamma_bright": lambda img: TF.adjust_gamma(img, gamma=0.7),
        "gamma_dark": lambda img: TF.adjust_gamma(img, gamma=1.5),
        "hue_shift": lambda img: TF.adjust_hue(img, hue_factor=0.05),
        "noise_gaussian": lambda img: _add_gaussian_noise(img, std=25),
    }


def _add_gaussian_noise(img, std=25):
    arr = np.array(img).astype(np.float32)
    noise = np.random.RandomState(42).randn(*arr.shape) * std
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


# ─── Data Collection ──────────────────────────────────────────────

def collect_synthetic_shifts(
    data_root: str,
    extractor: PatchExtractor,
    max_per_cat: int = 30,
    level: str = "image",
) -> dict[str, np.ndarray]:
    """Collect synthetic shift vectors per augmentation type.

    Returns: {aug_name: (N, dim) shift vectors}
    """
    root = Path(data_root)
    cats = sorted(d.name for d in root.iterdir() if d.is_dir() and (d / "train").is_dir())
    augmentations = get_augmentations()

    shifts_per_aug: dict[str, list] = {name: [] for name in augmentations}

    for cat in cats:
        train_dir = root / cat / "train" / "good"
        if not train_dir.is_dir():
            continue
        paths = sorted(p for p in train_dir.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg"))

        for img_path in paths[:max_per_cat]:
            img = Image.open(img_path).convert("RGB")

            if level == "image":
                f_orig = extractor.extract_image(img)
            else:
                f_orig = extractor.extract_patches(img).mean(axis=0)  # patch mean

            for aug_name, aug_fn in augmentations.items():
                try:
                    img_aug = aug_fn(img)
                    if level == "image":
                        f_aug = extractor.extract_image(img_aug)
                    else:
                        f_aug = extractor.extract_patches(img_aug).mean(axis=0)
                    shifts_per_aug[aug_name].append(f_aug - f_orig)
                except Exception:
                    pass

        print(f"  {cat}: {min(len(paths), max_per_cat)} images processed")

    result = {}
    for name, vecs in shifts_per_aug.items():
        if vecs:
            result[name] = np.stack(vecs)
            mag = np.linalg.norm(np.stack(vecs), axis=1).mean()
            print(f"  {name}: {len(vecs)} vectors, mean magnitude={mag:.4f}")
    return result


def collect_oracle_shifts(
    data_root: str,
    extractor: PatchExtractor,
    max_pairs: int = 500,
    level: str = "image",
) -> np.ndarray:
    """Collect oracle paired shift vectors from MVTec AD 2."""
    root = Path(data_root)
    cats = sorted(d.name for d in root.iterdir() if d.is_dir() and (d / "test_public").is_dir())

    all_shifts = []
    for cat in cats:
        good_dir = root / cat / "test_public" / "good"
        if not good_dir.is_dir():
            continue
        groups: dict[str, dict[str, Path]] = defaultdict(dict)
        for p in sorted(good_dir.glob("*.png")):
            parts = p.stem.split("_", 1)
            if len(parts) == 2:
                groups[parts[0]][parts[1]] = p

        for oid, conds in sorted(groups.items()):
            if "regular" not in conds:
                continue
            img_reg = Image.open(conds["regular"]).convert("RGB")
            if level == "image":
                f_reg = extractor.extract_image(img_reg)
            else:
                f_reg = extractor.extract_patches(img_reg).mean(axis=0)

            for c, path in sorted(conds.items()):
                if c == "regular":
                    continue
                img_s = Image.open(path).convert("RGB")
                if level == "image":
                    f_s = extractor.extract_image(img_s)
                else:
                    f_s = extractor.extract_patches(img_s).mean(axis=0)
                all_shifts.append(f_s - f_reg)

            if len(all_shifts) >= max_pairs:
                break
        if len(all_shifts) >= max_pairs:
            break

    return np.stack(all_shifts)


# ─── Diagnostics ──────────────────────────────────────────────────

def pca_basis(vectors, K):
    if len(vectors) <= 1:
        return np.zeros((vectors.shape[1], 0))
    K = min(K, len(vectors) - 1)
    centered = vectors - vectors.mean(0)
    _, S, Vt = np.linalg.svd(centered, full_matrices=False)
    return Vt[:K].T, S


def principal_angles(basis_a, basis_b):
    """Compute principal angles between two subspaces."""
    K = min(basis_a.shape[1], basis_b.shape[1])
    Qa, _ = np.linalg.qr(basis_a[:, :K])
    Qb, _ = np.linalg.qr(basis_b[:, :K])
    M = Qa.T @ Qb
    _, sigmas, _ = np.linalg.svd(M)
    sigmas = np.clip(sigmas, 0, 1)
    return np.degrees(np.arccos(sigmas))


def explained_variance(basis, target_vectors):
    """How much of target variance is explained by basis."""
    centered = target_vectors - target_vectors.mean(0)
    total_var = np.sum(centered ** 2)
    proj = centered @ basis @ basis.T
    explained = np.sum(proj ** 2)
    return explained / total_var


# ─── Main ────────────────────────────────────────────────────────

def run(args):
    print("=" * 60)
    print("Synthetic Shift Codebook Diagnostic")
    print("=" * 60)

    extractor = PatchExtractor(args.model_name, args.device, args.layer)

    # ── Step 1: Collect synthetic shifts ──────────────────────────
    print(f"\n[1/4] Collecting synthetic shift vectors (Layer {args.layer})...")
    synth_shifts = collect_synthetic_shifts(
        args.data_root, extractor, max_per_cat=args.max_per_cat, level="image",
    )

    # Pool all synthetic shifts
    all_synth = np.vstack(list(synth_shifts.values()))
    print(f"\n  Total synthetic vectors: {len(all_synth)}")

    # ── Step 2: Collect oracle shifts ─────────────────────────────
    print(f"\n[2/4] Collecting oracle shift vectors...")
    oracle_shifts = collect_oracle_shifts(
        args.data_root, extractor, max_pairs=500, level="image",
    )
    print(f"  Oracle vectors: {len(oracle_shifts)}")
    print(f"  Oracle mean magnitude: {np.linalg.norm(oracle_shifts, axis=1).mean():.4f}")

    # ── Step 3: Compute bases ─────────────────────────────────────
    K = args.K
    print(f"\n[3/4] Computing PCA bases (K={K})...")

    synth_basis, synth_S = pca_basis(all_synth, K)
    oracle_basis, oracle_S = pca_basis(oracle_shifts, K)

    synth_explained = (synth_S[:K] ** 2).sum() / (synth_S ** 2).sum()
    oracle_explained = (oracle_S[:K] ** 2).sum() / (oracle_S ** 2).sum()
    print(f"  Synthetic basis: top-{K} explain {synth_explained:.1%} of synthetic variance")
    print(f"  Oracle basis: top-{K} explain {oracle_explained:.1%} of oracle variance")

    # ── Step 4: Diagnostics ───────────────────────────────────────
    print(f"\n[4/4] Running diagnostics...")

    # D1: Principal angles
    angles = principal_angles(synth_basis, oracle_basis)
    d1_mean = angles.mean()
    d1_frac_45 = (angles < 45).mean()
    print(f"\n  D1: Principal angles (synthetic vs oracle)")
    print(f"    Mean angle: {d1_mean:.1f}°")
    print(f"    Fraction < 45°: {d1_frac_45:.1%}")
    print(f"    Top-10 angles: {angles[:10].round(1)}")
    print(f"    Top-20 angles: {angles[:20].round(1)}")

    # D2: Oracle shift variance explained by synthetic basis
    d2 = explained_variance(synth_basis, oracle_shifts)
    print(f"\n  D2: Oracle shift explained by synthetic: {d2:.1%}")

    # D2b: Synthetic shift variance explained by oracle basis
    d2b = explained_variance(oracle_basis, all_synth)
    print(f"  D2b: Synthetic shift explained by oracle: {d2b:.1%}")

    # D3: Per-augmentation contribution
    print(f"\n  D3: Per-augmentation alignment with oracle")
    print(f"  {'Augmentation':<28s} {'Magnitude':>10s} {'Oracle explained':>16s}")
    print(f"  " + "-" * 56)

    aug_results = {}
    for aug_name, vecs in sorted(synth_shifts.items()):
        mag = np.linalg.norm(vecs, axis=1).mean()
        aug_basis, _ = pca_basis(vecs, min(K, len(vecs) - 1))
        if aug_basis.shape[1] > 0:
            aug_explained = explained_variance(aug_basis, oracle_shifts)
        else:
            aug_explained = 0.0
        aug_results[aug_name] = {"magnitude": float(mag), "oracle_explained": float(aug_explained)}
        print(f"  {aug_name:<28s} {mag:>9.4f} {aug_explained:>15.1%}")

    # D4: DINOv2 invariance check
    print(f"\n  D4: DINOv2 invariance check (shift vector magnitude)")
    mags = {name: np.linalg.norm(vecs, axis=1).mean() for name, vecs in synth_shifts.items()}
    sorted_mags = sorted(mags.items(), key=lambda x: x[1], reverse=True)
    print(f"  Largest shift: {sorted_mags[0][0]} = {sorted_mags[0][1]:.4f}")
    print(f"  Smallest shift: {sorted_mags[-1][0]} = {sorted_mags[-1][1]:.4f}")
    print(f"  Oracle mean: {np.linalg.norm(oracle_shifts, axis=1).mean():.4f}")

    # ── Summary ───────────────────────────────────────────────────
    d1_pass = d1_mean < 60
    d2_pass = d2 > 0.30

    print(f"\n{'='*60}")
    print(f"DIAGNOSTIC SUMMARY")
    print(f"{'='*60}")
    print(f"  D1 Mean angle:          {d1_mean:.1f}°    {'✅ PASS' if d1_pass else '❌ FAIL'} (< 60°)")
    print(f"  D1 Fraction < 45°:      {d1_frac_45:.1%}")
    print(f"  D2 Oracle explained:    {d2:.1%}    {'✅ PASS' if d2_pass else '❌ FAIL'} (> 30%)")

    if d1_pass and d2_pass:
        verdict = "GO"
    elif d1_pass or d2_pass:
        verdict = "CONDITIONAL"
    else:
        verdict = "NO-GO"

    print(f"\n  VERDICT: {verdict}")

    # Save
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "synthetic_shift_diagnostic.json", "w") as f:
        json.dump({
            "d1_mean_angle": float(d1_mean),
            "d1_frac_below_45": float(d1_frac_45),
            "d1_top20_angles": angles[:20].tolist(),
            "d2_oracle_explained_by_synthetic": float(d2),
            "d2b_synthetic_explained_by_oracle": float(d2b),
            "d3_per_augmentation": aug_results,
            "d4_oracle_magnitude": float(np.linalg.norm(oracle_shifts, axis=1).mean()),
            "verdict": verdict,
            "config": {"K": K, "layer": args.layer, "n_augmentations": len(synth_shifts)},
        }, f, indent=2)
    print(f"\n  Saved to {out / 'synthetic_shift_diagnostic.json'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default="/home/hun/Volume/DATA/mvtec_ad_2")
    p.add_argument("--model_name", default="dinov2_vitb14")
    p.add_argument("--device", default="cuda")
    p.add_argument("--layer", type=int, default=8)
    p.add_argument("--K", type=int, default=50)
    p.add_argument("--max_per_cat", type=int, default=30)
    p.add_argument("--output_dir", default="results/synthetic_shift_diagnostic")
    run(p.parse_args())
