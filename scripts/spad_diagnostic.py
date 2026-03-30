#!/usr/bin/env python3
"""SPAD Diagnostic: Is distribution shift spatially uniform?

Core question: SPAD assumes shift affects all patches uniformly.
  If true → spatial mean subtraction removes shift cleanly.
  If false → spatial mean is a poor shift estimate → SPAD fails.

Measures:
  D0: Spatial uniformity of shift
      uniformity = 1 - ||residuals||² / ||per_patch_shift||²
      where residuals = per_patch_shift - mean(per_patch_shift)

  D1: How much of per-patch shift variance is explained by the global mean
      (high = uniform, low = spatially varying)

  D2: Quick AD scoring comparison
      - Baseline: patch-level kNN (PatchCore-style, no shift correction)
      - SPAD: patch kNN + spatial mean subtraction + max aggregation
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


# ─── Feature Extraction (patch-level) ────────────────────────────

class PatchExtractor:
    def __init__(
        self,
        model_name: str = "dinov2_vitb14",
        device: str = "cuda",
        target_layer: int = 8,
    ) -> None:
        self.device = device
        self.model = torch.hub.load("facebookresearch/dinov2", model_name).to(device).eval()
        self.target_layer = target_layer
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
    def extract_patches(self, img: Image.Image) -> np.ndarray:
        """Return patch tokens (M, dim) excluding CLS."""
        x = self.transform(img).unsqueeze(0).to(self.device)
        _ = self.model(x)
        return self.feature[0, 1:].cpu().numpy()  # (M, 768), M=37*37=1369

    @torch.no_grad()
    def extract_cls_patch(self, img: Image.Image) -> np.ndarray:
        """Return CLS + patch_mean, L2 normalized."""
        x = self.transform(img).unsqueeze(0).to(self.device)
        _ = self.model(x)
        feat = self.feature[0]
        cls = feat[0]
        pmean = feat[1:].mean(dim=0)
        combined = torch.cat([cls, pmean])
        combined = combined / (combined.norm() + 1e-8)
        return combined.cpu().numpy()


# ─── D0: Spatial Uniformity ──────────────────────────────────────

def measure_spatial_uniformity(
    data_root: str,
    extractor: PatchExtractor,
    max_pairs: int = 200,
) -> dict:
    """Measure how uniform the shift is across patches."""
    root = Path(data_root)
    categories = sorted(
        d.name for d in root.iterdir()
        if d.is_dir() and (d / "test_public").is_dir()
    )

    results = {}
    all_uniformities = []

    for cat in categories:
        good_dir = root / cat / "test_public" / "good"
        if not good_dir.is_dir():
            continue

        groups: dict[str, dict[str, Path]] = defaultdict(dict)
        for img_path in sorted(good_dir.glob("*.png")):
            parts = img_path.stem.split("_", 1)
            if len(parts) == 2:
                groups[parts[0]][parts[1]] = img_path

        cat_uniformities = []
        cat_global_norms = []
        cat_residual_norms = []
        pair_count = 0

        for obj_idx, conditions in sorted(groups.items()):
            if "regular" not in conditions:
                continue
            patches_reg = extractor.extract_patches(
                Image.open(conditions["regular"]).convert("RGB")
            )  # (M, 768)

            for cond, path in sorted(conditions.items()):
                if cond == "regular":
                    continue
                patches_shift = extractor.extract_patches(
                    Image.open(path).convert("RGB")
                )

                # Per-patch shift
                per_patch_shift = patches_shift - patches_reg  # (M, 768)

                # Global shift = spatial mean
                global_shift = per_patch_shift.mean(axis=0)  # (768,)

                # Residuals after global subtraction
                residuals = per_patch_shift - global_shift  # (M, 768)

                # Uniformity metric
                total_var = np.sum(per_patch_shift ** 2)
                residual_var = np.sum(residuals ** 2)

                if total_var > 1e-10:
                    uniformity = 1.0 - residual_var / total_var
                else:
                    uniformity = 1.0

                cat_uniformities.append(uniformity)
                cat_global_norms.append(np.linalg.norm(global_shift))
                cat_residual_norms.append(np.sqrt(np.mean(np.sum(residuals ** 2, axis=1))))

                pair_count += 1

            if pair_count >= max_pairs:
                break

        if cat_uniformities:
            results[cat] = {
                "mean_uniformity": float(np.mean(cat_uniformities)),
                "std_uniformity": float(np.std(cat_uniformities)),
                "min_uniformity": float(np.min(cat_uniformities)),
                "mean_global_norm": float(np.mean(cat_global_norms)),
                "mean_residual_norm": float(np.mean(cat_residual_norms)),
                "n_pairs": pair_count,
            }
            all_uniformities.extend(cat_uniformities)

    results["OVERALL"] = {
        "mean_uniformity": float(np.mean(all_uniformities)),
        "std_uniformity": float(np.std(all_uniformities)),
        "min_uniformity": float(np.min(all_uniformities)),
    }

    return results


# ─── D2: Quick AD Scoring Comparison ─────────────────────────────

def collect_patch_data(
    data_root: str,
    extractor: PatchExtractor,
    max_per_cat: int = 50,
) -> dict[str, dict]:
    """Collect patch-level features for train/test."""
    root = Path(data_root)
    categories = sorted(
        d.name for d in root.iterdir()
        if d.is_dir() and (d / "train").is_dir()
    )
    dataset = {}
    for cat in categories:
        cat_data = {
            "train_patches": [],  # list of (M, 768) arrays
            "test_normal_patches": [],
            "test_anomaly_patches": [],
        }

        train_good = root / cat / "train" / "good"
        if train_good.is_dir():
            for p in sorted(train_good.iterdir())[:max_per_cat]:
                if p.suffix.lower() in (".png", ".jpg", ".jpeg"):
                    cat_data["train_patches"].append(
                        extractor.extract_patches(Image.open(p).convert("RGB"))
                    )

        test_dir = root / cat / "test_public"
        if test_dir.is_dir():
            for sub in sorted(test_dir.iterdir()):
                if not sub.is_dir():
                    continue
                label = "test_normal_patches" if sub.name == "good" else "test_anomaly_patches"
                for p in sorted(sub.iterdir())[:max_per_cat]:
                    if p.suffix.lower() in (".png", ".jpg", ".jpeg"):
                        cat_data[label].append(
                            extractor.extract_patches(Image.open(p).convert("RGB"))
                        )

        n_tr = len(cat_data["train_patches"])
        n_tn = len(cat_data["test_normal_patches"])
        n_ta = len(cat_data["test_anomaly_patches"])
        print(f"  {cat}: train={n_tr}, test_n={n_tn}, test_a={n_ta}")
        dataset[cat] = cat_data

    return dataset


def score_patchcore_style(
    train_patches_list: list[np.ndarray],
    test_patches: np.ndarray,
    coreset_ratio: float = 0.1,
) -> float:
    """PatchCore-style: build patch memory bank, kNN per patch, max aggregation."""
    # Build memory bank
    all_train = np.vstack(train_patches_list)  # (N_train*M, 768)

    # Coreset subsampling for speed
    n_keep = max(1000, int(len(all_train) * coreset_ratio))
    if n_keep < len(all_train):
        rng = np.random.RandomState(42)
        idx = rng.choice(len(all_train), size=n_keep, replace=False)
        memory = all_train[idx]
    else:
        memory = all_train

    nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
    nn.fit(memory)

    # Per-patch kNN distance
    dists, _ = nn.kneighbors(test_patches)  # (M, 1)
    return dists.ravel()  # (M,) patch-level scores


def score_spad(
    train_patches_list: list[np.ndarray],
    test_patches: np.ndarray,
    coreset_ratio: float = 0.1,
) -> float:
    """SPAD: patch kNN → spatial mean subtraction → max aggregation."""
    all_train = np.vstack(train_patches_list)

    n_keep = max(1000, int(len(all_train) * coreset_ratio))
    if n_keep < len(all_train):
        rng = np.random.RandomState(42)
        idx = rng.choice(len(all_train), size=n_keep, replace=False)
        memory = all_train[idx]
    else:
        memory = all_train

    nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
    nn.fit(memory)

    # Per-patch NN matching
    dists, indices = nn.kneighbors(test_patches)  # (M, 1)
    nn_patches = memory[indices.ravel()]  # (M, 768)

    # Raw difference vectors
    raw_diff = test_patches - nn_patches  # (M, 768)

    # Global shift = spatial mean of differences
    global_shift = raw_diff.mean(axis=0)  # (768,)

    # Local residual
    local_residual = raw_diff - global_shift  # (M, 768)
    local_scores = np.linalg.norm(local_residual, axis=1)  # (M,)

    return local_scores


def evaluate_scoring(dataset: dict, method: str, coreset_ratio: float = 0.1) -> dict:
    """Evaluate a scoring method. Returns per-category AUROC."""
    results = {}
    aurocs = []

    for cat in sorted(dataset.keys()):
        train_list = dataset[cat]["train_patches"]
        test_n_list = dataset[cat]["test_normal_patches"]
        test_a_list = dataset[cat]["test_anomaly_patches"]

        if not train_list or not test_n_list or not test_a_list:
            continue

        scores_n = []
        scores_a = []

        for patches in test_n_list:
            if method == "patchcore":
                ps = score_patchcore_style(train_list, patches, coreset_ratio)
            elif method == "spad":
                ps = score_spad(train_list, patches, coreset_ratio)
            else:
                raise ValueError(method)

            # Image-level aggregation
            scores_n.append(np.percentile(ps, 95))  # top-5% percentile

        for patches in test_a_list:
            if method == "patchcore":
                ps = score_patchcore_style(train_list, patches, coreset_ratio)
            elif method == "spad":
                ps = score_spad(train_list, patches, coreset_ratio)
            else:
                raise ValueError(method)

            scores_a.append(np.percentile(ps, 95))

        labels = np.concatenate([np.zeros(len(scores_n)), np.ones(len(scores_a))])
        scores = np.concatenate([scores_n, scores_a])
        auroc = roc_auc_score(labels, scores)
        results[cat] = auroc
        aurocs.append(auroc)

    results["MEAN"] = np.mean(aurocs) if aurocs else 0.0
    return results


# ─── Main ────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> None:
    print("=" * 60)
    print("SPAD Diagnostic: Spatial Uniformity + Quick AD Comparison")
    print("=" * 60)

    extractor = PatchExtractor(
        model_name=args.model_name, device=args.device, target_layer=args.layer,
    )

    # ── D0: Spatial Uniformity ────────────────────────────────────
    print(f"\n[1/3] D0: Measuring spatial uniformity of shift (Layer {args.layer})...")
    uniformity = measure_spatial_uniformity(args.data_root, extractor, max_pairs=200)

    print(f"\n  Per-category:")
    print(f"  {'Category':<16s} {'Uniformity':>12s} {'Global norm':>12s} {'Residual norm':>14s} {'N_pairs':>8s}")
    print("  " + "-" * 64)
    for cat in sorted(k for k in uniformity if k != "OVERALL"):
        u = uniformity[cat]
        print(f"  {cat:<16s} {u['mean_uniformity']:>11.3f} {u['mean_global_norm']:>11.4f} "
              f"{u['mean_residual_norm']:>13.4f} {u['n_pairs']:>7d}")

    ov = uniformity["OVERALL"]
    print(f"\n  OVERALL: uniformity={ov['mean_uniformity']:.3f} ± {ov['std_uniformity']:.3f}"
          f" (min={ov['min_uniformity']:.3f})")

    d0_pass = ov["mean_uniformity"] > 0.3
    d0_strong = ov["mean_uniformity"] > 0.5
    print(f"  D0 {'✅ PASS' if d0_pass else '❌ FAIL'}"
          f"{'  (STRONG)' if d0_strong else ''}")

    # ── D2: Quick AD Scoring ──────────────────────────────────────
    print(f"\n[2/3] Collecting patch data (max {args.max_per_cat} per cat)...")
    dataset = collect_patch_data(args.data_root, extractor, max_per_cat=args.max_per_cat)

    print(f"\n[3/3] D2: Quick AD scoring comparison...")

    print(f"\n  --- PatchCore-style (no shift correction) ---")
    res_pc = evaluate_scoring(dataset, "patchcore", coreset_ratio=args.coreset_ratio)
    print(f"  MEAN I-AUROC: {res_pc['MEAN']*100:.1f}%")

    print(f"\n  --- SPAD (spatial mean subtraction) ---")
    res_spad = evaluate_scoring(dataset, "spad", coreset_ratio=args.coreset_ratio)
    print(f"  MEAN I-AUROC: {res_spad['MEAN']*100:.1f}%")

    # Summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"\n  D0: Spatial uniformity = {ov['mean_uniformity']:.3f}"
          f" {'✅' if d0_pass else '❌'}")

    print(f"\n  {'Category':<16s} {'PatchCore':>10s} {'SPAD':>10s} {'Delta':>8s}")
    print("  " + "-" * 46)
    for cat in sorted(k for k in res_pc if k != "MEAN"):
        pc = res_pc.get(cat, 0) * 100
        sp = res_spad.get(cat, 0) * 100
        delta = sp - pc
        marker = "✅" if delta > 0 else "❌" if delta < -1 else "—"
        print(f"  {cat:<16s} {pc:>9.1f}% {sp:>9.1f}% {delta:>+7.1f}  {marker}")

    pc_mean = res_pc["MEAN"] * 100
    sp_mean = res_spad["MEAN"] * 100
    print(f"  {'MEAN':<16s} {pc_mean:>9.1f}% {sp_mean:>9.1f}% {sp_mean-pc_mean:>+7.1f}")

    verdict = "GO" if sp_mean > pc_mean + 1 else "CONDITIONAL" if sp_mean >= pc_mean else "NO-GO"
    print(f"\n  VERDICT: {verdict}")

    # Save
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "spad_diagnostic.json", "w") as f:
        json.dump({
            "d0_uniformity": uniformity,
            "d2_patchcore": {k: float(v) for k, v in res_pc.items()},
            "d2_spad": {k: float(v) for k, v in res_spad.items()},
            "verdict": verdict,
        }, f, indent=2)
    print(f"\n  Saved to {output_dir / 'spad_diagnostic.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SPAD Diagnostic")
    parser.add_argument("--data_root", type=str, default="/home/hun/Volume/DATA/mvtec_ad_2")
    parser.add_argument("--model_name", type=str, default="dinov2_vitb14")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--layer", type=int, default=8)
    parser.add_argument("--max_per_cat", type=int, default=50)
    parser.add_argument("--coreset_ratio", type=float, default=0.1)
    parser.add_argument("--output_dir", type=str, default="results/spad_diagnostic")
    args = parser.parse_args()
    run(args)
