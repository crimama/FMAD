#!/usr/bin/env python3
"""Phase 2: Feature Space Analysis for FM-AD Robustness Study.

Step 2.1 — Feature Space Shift Quantification
Step 2.2 — Layer-wise Robustness Profiling

Extracts DINOv2 and CLIP features from MVTec AD 2 paired images (regular vs shifted),
measures feature distances per layer, and produces analysis outputs.
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


# ─── Feature Extractors ────────────────────────────────────────────

class DINOv2Extractor:
    """Extract per-layer features from DINOv2."""

    def __init__(self, model_name: str = "dinov2_vitb14", device: str = "cuda"):
        self.device = device
        self.model = torch.hub.load("facebookresearch/dinov2", model_name).to(device).eval()
        self.n_layers = len(self.model.blocks)
        self.features: dict[int, torch.Tensor] = {}
        self._register_hooks()
        self.transform = transforms.Compose([
            transforms.Resize(518, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(518),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def _register_hooks(self):
        for i, block in enumerate(self.model.blocks):
            block.register_forward_hook(self._make_hook(i))

    def _make_hook(self, layer_idx: int):
        def hook(module, input, output):
            self.features[layer_idx] = output.detach()
        return hook

    @torch.no_grad()
    def extract(self, img: Image.Image) -> dict[int, torch.Tensor]:
        """Return {layer_idx: feature_tensor} for all layers."""
        self.features.clear()
        x = self.transform(img).unsqueeze(0).to(self.device)
        _ = self.model(x)
        return {k: v.cpu() for k, v in self.features.items()}


class CLIPExtractor:
    """Extract per-layer features from CLIP ViT."""

    def __init__(self, device: str = "cuda"):
        try:
            import open_clip
            self.model, _, self.transform = open_clip.create_model_and_transforms(
                "ViT-L-14", pretrained="openai", device=device
            )
        except ImportError:
            # Fallback: use AnomalyCLIP's CLIP loader pattern
            import clip
            self.model, self.transform = clip.load("ViT-L/14@336px", device=device)

        self.device = device
        self.model.eval()
        self.features: dict[int, torch.Tensor] = {}
        self._register_hooks()

    def _register_hooks(self):
        visual = self.model.visual
        # Handle different CLIP implementations
        if hasattr(visual, "transformer"):
            blocks = visual.transformer.resblocks
        elif hasattr(visual, "trunk"):
            blocks = visual.trunk.blocks
        else:
            blocks = []
        self.n_layers = len(blocks)
        for i, block in enumerate(blocks):
            block.register_forward_hook(self._make_hook(i))

    def _make_hook(self, layer_idx: int):
        def hook(module, input, output):
            if isinstance(output, tuple):
                output = output[0]
            self.features[layer_idx] = output.detach()
        return hook

    @torch.no_grad()
    def extract(self, img: Image.Image) -> dict[int, torch.Tensor]:
        self.features.clear()
        x = self.transform(img).unsqueeze(0).to(self.device)
        _ = self.model.encode_image(x)
        return {k: v.cpu() for k, v in self.features.items()}


# ─── Image Pair Discovery ──────────────────────────────────────────

def discover_pairs(data_root: str, categories: list[str] | None = None) -> list[dict]:
    """Find paired images (same object, different conditions) in MVTec AD 2."""
    root = Path(data_root)
    if categories is None:
        categories = sorted(
            d.name for d in root.iterdir()
            if d.is_dir() and (d / "test_public").is_dir()
        )

    pairs = []
    for cat in categories:
        for split in ["good", "bad"]:
            split_dir = root / cat / "test_public" / split
            if not split_dir.is_dir():
                continue

            # Group by object index (prefix before first _)
            groups: dict[str, dict[str, Path]] = defaultdict(dict)
            for img_path in sorted(split_dir.glob("*.png")):
                parts = img_path.stem.split("_", 1)
                if len(parts) == 2:
                    obj_idx, condition = parts
                    groups[obj_idx][condition] = img_path

            for obj_idx, conditions in groups.items():
                if "regular" not in conditions:
                    continue
                for cond, path in conditions.items():
                    if cond == "regular":
                        continue
                    pairs.append({
                        "category": cat,
                        "split": split,
                        "obj_idx": obj_idx,
                        "condition": cond,
                        "regular_path": str(conditions["regular"]),
                        "shifted_path": str(path),
                    })

    return pairs


# ─── Analysis Functions ─────────────────────────────────────────────

def compute_shift_metrics(
    feat_regular: dict[int, torch.Tensor],
    feat_shifted: dict[int, torch.Tensor],
) -> dict[int, dict[str, float]]:
    """Compute per-layer shift metrics between regular and shifted features."""
    metrics = {}
    for layer_idx in feat_regular:
        fr = feat_regular[layer_idx].float().flatten()
        fs = feat_shifted[layer_idx].float().flatten()

        # L2 distance
        l2_dist = torch.norm(fr - fs).item()
        # Cosine similarity
        cos_sim = F.cosine_similarity(fr.unsqueeze(0), fs.unsqueeze(0)).item()
        # Relative shift: ||f_s - f_r|| / ||f_r||
        norm_r = torch.norm(fr).item()
        relative_shift = l2_dist / (norm_r + 1e-8)

        metrics[layer_idx] = {
            "l2_distance": l2_dist,
            "cosine_similarity": cos_sim,
            "relative_shift": relative_shift,
            "norm_regular": norm_r,
            "norm_shifted": torch.norm(fs).item(),
        }
    return metrics


def run_analysis(args):
    print("=" * 60)
    print("Phase 2: Feature Space Analysis")
    print("=" * 60)

    # Discover pairs
    print("\n[1/4] Discovering image pairs...")
    pairs = discover_pairs(args.data_root, args.categories)
    print(f"  Found {len(pairs)} pairs across {len(set(p['category'] for p in pairs))} categories")

    conditions = sorted(set(p["condition"] for p in pairs))
    print(f"  Conditions: {conditions}")

    # Subsample if too many
    if args.max_pairs and len(pairs) > args.max_pairs:
        rng = np.random.RandomState(42)
        indices = rng.choice(len(pairs), args.max_pairs, replace=False)
        pairs = [pairs[i] for i in sorted(indices)]
        print(f"  Subsampled to {len(pairs)} pairs")

    # Initialize extractor
    print(f"\n[2/4] Loading {args.backbone} extractor...")
    if args.backbone == "dinov2":
        extractor = DINOv2Extractor(model_name=args.model_name, device=args.device)
        n_layers = extractor.n_layers
    else:
        extractor = CLIPExtractor(device=args.device)
        n_layers = extractor.n_layers
    print(f"  {n_layers} layers")

    # Extract features and compute metrics
    print(f"\n[3/4] Extracting features and computing shift metrics...")
    all_metrics = []
    for pair in tqdm(pairs, desc="Processing pairs"):
        img_regular = Image.open(pair["regular_path"]).convert("RGB")
        img_shifted = Image.open(pair["shifted_path"]).convert("RGB")

        feat_r = extractor.extract(img_regular)
        feat_s = extractor.extract(img_shifted)

        layer_metrics = compute_shift_metrics(feat_r, feat_s)
        all_metrics.append({
            **pair,
            "layer_metrics": {str(k): v for k, v in layer_metrics.items()},
        })

    # Aggregate results
    print(f"\n[4/4] Aggregating results...")
    results = aggregate_results(all_metrics, n_layers, conditions)

    # Save
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / f"shift_metrics_{args.backbone}.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved to {output_dir / f'shift_metrics_{args.backbone}.json'}")

    # Print summary
    print_summary(results)

    return results


def aggregate_results(
    all_metrics: list[dict],
    n_layers: int,
    conditions: list[str],
) -> dict:
    """Aggregate per-pair metrics into per-layer, per-condition summaries."""

    # Per-layer aggregation
    layer_summary = {}
    for layer_idx in range(n_layers):
        vals = [m["layer_metrics"][str(layer_idx)] for m in all_metrics
                if str(layer_idx) in m["layer_metrics"]]
        if not vals:
            continue
        layer_summary[layer_idx] = {
            "relative_shift_mean": np.mean([v["relative_shift"] for v in vals]),
            "relative_shift_std": np.std([v["relative_shift"] for v in vals]),
            "cosine_sim_mean": np.mean([v["cosine_similarity"] for v in vals]),
            "cosine_sim_std": np.std([v["cosine_similarity"] for v in vals]),
            "l2_distance_mean": np.mean([v["l2_distance"] for v in vals]),
            "norm_regular_mean": np.mean([v["norm_regular"] for v in vals]),
            "n_samples": len(vals),
        }

    # Per-condition aggregation
    condition_summary = {}
    for cond in conditions:
        cond_metrics = [m for m in all_metrics if m["condition"] == cond]
        if not cond_metrics:
            continue
        layer_means = {}
        for layer_idx in range(n_layers):
            vals = [m["layer_metrics"][str(layer_idx)] for m in cond_metrics
                    if str(layer_idx) in m["layer_metrics"]]
            if vals:
                layer_means[layer_idx] = {
                    "relative_shift_mean": np.mean([v["relative_shift"] for v in vals]),
                    "cosine_sim_mean": np.mean([v["cosine_similarity"] for v in vals]),
                }
        condition_summary[cond] = {
            "n_pairs": len(cond_metrics),
            "layer_means": layer_means,
        }

    # Per-category aggregation
    category_summary = {}
    for cat in sorted(set(m["category"] for m in all_metrics)):
        cat_metrics = [m for m in all_metrics if m["category"] == cat]
        mean_shift = np.mean([
            m["layer_metrics"][str(n_layers - 1)]["relative_shift"]
            for m in cat_metrics
            if str(n_layers - 1) in m["layer_metrics"]
        ])
        category_summary[cat] = {
            "n_pairs": len(cat_metrics),
            "last_layer_relative_shift_mean": mean_shift,
        }

    return {
        "layer_summary": {str(k): v for k, v in layer_summary.items()},
        "condition_summary": condition_summary,
        "category_summary": category_summary,
        "n_total_pairs": len(all_metrics),
        "n_layers": n_layers,
    }


def print_summary(results: dict):
    print("\n" + "=" * 60)
    print("LAYER-WISE SHIFT PROFILE")
    print("=" * 60)
    print(f"{'Layer':>6} {'RelShift':>10} {'CosSim':>10}")
    print("-" * 30)
    for layer_str in sorted(results["layer_summary"].keys(), key=int):
        s = results["layer_summary"][layer_str]
        print(f"{layer_str:>6} {s['relative_shift_mean']:>10.4f} {s['cosine_sim_mean']:>10.4f}")

    print("\n" + "=" * 60)
    print("PER-CONDITION SHIFT (last layer)")
    print("=" * 60)
    n_layers = results["n_layers"]
    last = str(n_layers - 1)
    for cond, info in sorted(results["condition_summary"].items()):
        if last in info["layer_means"]:
            rs = info["layer_means"][last]["relative_shift_mean"]
            cs = info["layer_means"][last]["cosine_sim_mean"]
            print(f"  {cond:<20s} RelShift={rs:.4f}  CosSim={cs:.4f}  (n={info['n_pairs']})")

    print("\n" + "=" * 60)
    print("PER-CATEGORY SHIFT (last layer)")
    print("=" * 60)
    for cat, info in sorted(results["category_summary"].items()):
        print(f"  {cat:<20s} RelShift={info['last_layer_relative_shift_mean']:.4f}  (n={info['n_pairs']})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 2: Feature Space Analysis")
    parser.add_argument("--data_root", type=str, default="/home/hun/Volume/DATA/mvtec_ad_2")
    parser.add_argument("--backbone", type=str, default="dinov2", choices=["dinov2", "clip"])
    parser.add_argument("--model_name", type=str, default="dinov2_vitb14")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, default="results/phase2")
    parser.add_argument("--max_pairs", type=int, default=200, help="Max pairs to process (0=all)")
    parser.add_argument("--categories", nargs="+", default=None)
    args = parser.parse_args()
    run_analysis(args)
