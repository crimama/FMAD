#!/usr/bin/env python3
"""Few-shot FM-AD under Distribution Shift.

Problem: Few-shot FM-AD (K reference images) fails under distribution shift.
Proposal: Multi-condition calibration — capture K references under C conditions.
  → K×(C-1) shift vectors → NSP projection → robust scoring.

Experiment matrix:
  K (shots): 1, 2, 4, 8, 16, full
  C (conditions): 1 (no calibration), 2, 3, 6 (all)
  Methods: baseline (no proj), NSP
  Layer: 8
  Scoring: Mahalanobis
  3 seeds per (K, C) combination
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


# ─── Feature Extraction ───────────────────────────────────────────

class DINOv2Extractor:
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
    def extract(self, img):
        x = self.transform(img).unsqueeze(0).to(self.device)
        _ = self.model(x)
        feat = self.feature[0]
        cls = feat[0]
        pm = feat[1:].mean(0)
        c = torch.cat([cls, pm])
        return (c / (c.norm() + 1e-8)).cpu().numpy()


# ─── Data Collection ──────────────────────────────────────────────

ALL_CONDITIONS = ["regular", "overexposed", "underexposed", "shift_1", "shift_2", "shift_3"]


def collect_multi_condition_references(
    data_root: str,
    extractor: DINOv2Extractor,
) -> dict[str, dict]:
    """Collect ALL multi-condition reference data per category.

    Returns per-category:
      objects: {obj_id: {condition: feature_vector}}
      test_normal: [features]  (all conditions mixed)
      test_anomaly: [features]
    """
    root = Path(data_root)
    cats = sorted(d.name for d in root.iterdir() if d.is_dir() and (d / "test_public").is_dir())

    dataset = {}
    for cat in cats:
        # Multi-condition references from test_public/good
        good_dir = root / cat / "test_public" / "good"
        objects: dict[str, dict[str, np.ndarray]] = defaultdict(dict)

        for img_path in sorted(good_dir.glob("*.png")):
            parts = img_path.stem.split("_", 1)
            if len(parts) == 2:
                obj_id, condition = parts
                feat = extractor.extract(Image.open(img_path).convert("RGB"))
                objects[obj_id][condition] = feat

        # Also collect from train/good (regular only, more objects)
        train_dir = root / cat / "train" / "good"
        train_objects: dict[str, np.ndarray] = {}
        if train_dir.is_dir():
            for img_path in sorted(train_dir.glob("*.png")):
                parts = img_path.stem.split("_", 1)
                if len(parts) == 2:
                    obj_id = parts[0]
                    train_objects[obj_id] = extractor.extract(Image.open(img_path).convert("RGB"))

        # Test anomaly
        test_anomaly = []
        test_dir = root / cat / "test_public"
        if test_dir.is_dir():
            for sub in sorted(test_dir.iterdir()):
                if not sub.is_dir() or sub.name == "good":
                    continue
                for img_path in sorted(sub.glob("*.png"))[:200]:
                    test_anomaly.append(extractor.extract(Image.open(img_path).convert("RGB")))

        # Test normal = all conditions from test_public/good
        test_normal = []
        for obj_id, conds in objects.items():
            for cond, feat in conds.items():
                test_normal.append(feat)

        n_obj = len(objects)
        n_train = len(train_objects)
        n_tn = len(test_normal)
        n_ta = len(test_anomaly)
        print(f"  {cat}: objects={n_obj}, train_objects={n_train}, test_n={n_tn}, test_a={n_ta}")

        dataset[cat] = {
            "objects": dict(objects),
            "train_objects": train_objects,
            "test_normal": np.stack(test_normal) if test_normal else np.zeros((0, 1536)),
            "test_anomaly": np.stack(test_anomaly) if test_anomaly else np.zeros((0, 1536)),
        }

    return dataset


# ─── Few-shot Sampling ────────────────────────────────────────────

def sample_fewshot(
    cat_data: dict,
    K: int,
    C: int,
    seed: int,
    conditions: list[str] = ALL_CONDITIONS,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample K objects × C conditions for few-shot reference + calibration.

    Returns:
      reference_feats: (K, dim) — regular condition features (for scoring)
      shift_vectors: (K*(C-1), dim) — shift vectors (for NSP)
    """
    rng = np.random.RandomState(seed)

    # Available objects with all requested conditions
    objects = cat_data["objects"]
    valid_obj_ids = [
        oid for oid, conds in objects.items()
        if "regular" in conds and sum(c in conds for c in conditions[:C]) >= C
    ]

    # If not enough multi-condition objects, also use train objects (regular only)
    if len(valid_obj_ids) < K:
        # Fallback: use train objects for reference (regular only, no shift vectors)
        train_ids = list(cat_data["train_objects"].keys())
        rng.shuffle(train_ids)
        extra_needed = K - len(valid_obj_ids)
        # Use all valid multi-condition objects + extra from train
        selected_multi = valid_obj_ids
        selected_train = train_ids[:extra_needed]
    else:
        selected_multi = list(rng.choice(valid_obj_ids, size=min(K, len(valid_obj_ids)), replace=False))
        selected_train = []

    # Build reference features (regular condition)
    ref_feats = []
    for oid in selected_multi:
        ref_feats.append(objects[oid]["regular"])
    for oid in selected_train:
        ref_feats.append(cat_data["train_objects"][oid])

    # Build shift vectors (multi-condition objects only)
    shift_vecs = []
    selected_conditions = conditions[:C]
    for oid in selected_multi:
        feat_reg = objects[oid]["regular"]
        for cond in selected_conditions:
            if cond == "regular":
                continue
            if cond in objects[oid]:
                shift_vecs.append(objects[oid][cond] - feat_reg)

    ref_feats = np.stack(ref_feats) if ref_feats else np.zeros((0, 1536))
    shift_vecs = np.stack(shift_vecs) if shift_vecs else np.zeros((0, 1536))

    return ref_feats, shift_vecs


# ─── NSP + Scoring ────────────────────────────────────────────────

def pca_basis(vectors, K):
    if len(vectors) <= 1 or K <= 0:
        return np.zeros((vectors.shape[1], 0))
    K = min(K, len(vectors) - 1)
    centered = vectors - vectors.mean(0)
    _, S, Vt = np.linalg.svd(centered, full_matrices=False)
    return Vt[:K].T


def hard_project(features, basis):
    if basis.shape[1] == 0:
        return features
    return features - features @ (basis @ basis.T)


def mahalanobis_score(ref, test):
    mu = ref.mean(0)
    if len(ref) < 3:
        # Too few samples for covariance → use L2 distance to mean
        return np.linalg.norm(test - mu, axis=1)
    cov = np.cov((ref - mu).T) + np.eye(ref.shape[1]) * 1e-5
    try:
        inv = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        return np.linalg.norm(test - mu, axis=1)
    tc = test - mu
    return np.sqrt(np.sum(tc @ inv * tc, axis=1))


# ─── Evaluation ──────────────────────────────────────────────────

def evaluate_fewshot(
    dataset: dict,
    K: int,
    C: int,
    use_nsp: bool,
    nsp_K: int = 50,
    seed: int = 42,
) -> dict[str, float]:
    """Evaluate few-shot AD with optional NSP."""
    results = {}
    aurocs = []

    conditions = ALL_CONDITIONS[:C] if C <= len(ALL_CONDITIONS) else ALL_CONDITIONS

    for cat in sorted(dataset.keys()):
        cat_data = dataset[cat]
        test_n = cat_data["test_normal"]
        test_a = cat_data["test_anomaly"]

        if len(test_n) == 0 or len(test_a) == 0:
            continue

        ref_feats, shift_vecs = sample_fewshot(cat_data, K, C, seed, conditions)

        if len(ref_feats) == 0:
            continue

        # NSP projection
        if use_nsp and len(shift_vecs) > 1:
            actual_K = min(nsp_K, len(shift_vecs) - 1)
            basis = pca_basis(shift_vecs, actual_K)
            ref_p = hard_project(ref_feats, basis)
            tn_p = hard_project(test_n, basis)
            ta_p = hard_project(test_a, basis)
        else:
            ref_p, tn_p, ta_p = ref_feats, test_n, test_a

        scores_n = mahalanobis_score(ref_p, tn_p)
        scores_a = mahalanobis_score(ref_p, ta_p)

        labels = np.concatenate([np.zeros(len(scores_n)), np.ones(len(scores_a))])
        scores = np.concatenate([scores_n, scores_a])

        if len(np.unique(labels)) < 2:
            continue

        auroc = roc_auc_score(labels, scores)
        results[cat] = auroc
        aurocs.append(auroc)

    results["MEAN"] = np.mean(aurocs) if aurocs else 0.0
    return results


# ─── Main ────────────────────────────────────────────────────────

def run(args):
    print("=" * 60)
    print("Few-shot FM-AD under Distribution Shift")
    print("=" * 60)

    extractor = DINOv2Extractor(args.model_name, args.device, args.layer)

    print(f"\n[1/2] Collecting multi-condition references (Layer {args.layer})...")
    dataset = collect_multi_condition_references(args.data_root, extractor)

    print(f"\n[2/2] Evaluating few-shot settings...")

    K_values = [1, 2, 4, 8, 16]
    C_values = [1, 2, 3, 6]  # 1=no calibration, 2=regular+1shift, 3=regular+2shifts, 6=all
    seeds = [42, 123, 456]

    all_results = {}

    for K in K_values:
        for C in C_values:
            for use_nsp in [False, True]:
                if C == 1 and use_nsp:
                    continue  # No shift vectors with C=1
                label = f"K{K}_C{C}_{'nsp' if use_nsp else 'base'}"

                seed_means = []
                for seed in seeds:
                    res = evaluate_fewshot(
                        dataset, K=K, C=C, use_nsp=use_nsp,
                        nsp_K=args.nsp_K, seed=seed,
                    )
                    seed_means.append(res["MEAN"])

                mean_val = np.mean(seed_means)
                std_val = np.std(seed_means)
                all_results[label] = {
                    "mean": float(mean_val),
                    "std": float(std_val),
                    "K": K, "C": C, "nsp": use_nsp,
                }
                nsp_tag = "NSP" if use_nsp else "base"
                print(f"  K={K:>2d} C={C} {nsp_tag:<4s}: {mean_val*100:.1f}% ± {std_val*100:.1f}%")

    # Summary table
    print("\n" + "=" * 60)
    print("SUMMARY: Few-shot NSP (AD2 I-AUROC %)")
    print("=" * 60)

    header = f"  {'':>8s}"
    for C in C_values:
        header += f" {'C='+str(C)+' base':>12s}"
        if C > 1:
            header += f" {'C='+str(C)+' NSP':>12s}"
    print(header)
    print("  " + "-" * (8 + 13 * (len(C_values) + sum(1 for c in C_values if c > 1))))

    for K in K_values:
        row = f"  K={K:>3d}  "
        for C in C_values:
            base_key = f"K{K}_C{C}_base"
            r = all_results.get(base_key, {})
            row += f" {r.get('mean',0)*100:>10.1f}% "
            if C > 1:
                nsp_key = f"K{K}_C{C}_nsp"
                r2 = all_results.get(nsp_key, {})
                row += f" {r2.get('mean',0)*100:>10.1f}% "
        print(row)

    # NSP gain table
    print(f"\n  NSP Gain (pp):")
    print(f"  {'':>8s}" + "".join(f" {'C='+str(C):>12s}" for C in C_values if C > 1))
    for K in K_values:
        row = f"  K={K:>3d}  "
        for C in C_values:
            if C <= 1:
                continue
            base_key = f"K{K}_C{C}_base"
            nsp_key = f"K{K}_C{C}_nsp"
            b = all_results.get(base_key, {}).get("mean", 0)
            n = all_results.get(nsp_key, {}).get("mean", 0)
            gain = (n - b) * 100
            row += f" {gain:>+11.1f}  "
        print(row)

    # Save
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "fewshot_nsp_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Saved to {out / 'fewshot_nsp_results.json'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default="/home/hun/Volume/DATA/mvtec_ad_2")
    p.add_argument("--model_name", default="dinov2_vitb14")
    p.add_argument("--device", default="cuda")
    p.add_argument("--layer", type=int, default=8)
    p.add_argument("--nsp_K", type=int, default=50, help="Max nuisance dims for NSP")
    p.add_argument("--output_dir", default="results/fewshot_nsp")
    run(p.parse_args())
