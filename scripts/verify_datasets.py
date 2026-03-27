#!/usr/bin/env python3

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Iterable


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
MVTEC_CATEGORIES = [
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
]


def count_images(root: Path) -> int:
    if not root.exists():
        return 0
    return sum(1 for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def list_image_dirs(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(
        {
            path.parent
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        }
    )


def print_header(title: str) -> None:
    print(f"\n=== {title} ===")


def verify_mvtec(data_root: Path) -> None:
    print_header("MVTec AD")
    existing_categories = [name for name in MVTEC_CATEGORIES if (data_root / name).is_dir()]
    print(f"root: {data_root}")
    print(f"categories_found: {len(existing_categories)}/{len(MVTEC_CATEGORIES)}")

    if not existing_categories:
        print("status: not found")
        return

    missing = [name for name in MVTEC_CATEGORIES if name not in existing_categories]
    if missing:
        print(f"missing_categories: {', '.join(missing)}")

    total_images = 0
    for category in existing_categories:
        category_dir = data_root / category
        split_counts = {
            "train": count_images(category_dir / "train"),
            "test": count_images(category_dir / "test"),
            "ground_truth": count_images(category_dir / "ground_truth"),
        }
        total_images += sum(split_counts.values())
        missing_splits = [split for split in ("train", "test", "ground_truth") if not (category_dir / split).is_dir()]
        split_counts_text = ", ".join(f"{split}={count}" for split, count in split_counts.items())
        status = "ok" if not missing_splits else f"missing_dirs={','.join(missing_splits)}"
        print(f"- {category}: {split_counts_text} [{status}]")

    print(f"total_images: {total_images}")


def iter_robustad_categories(robustad_root: Path) -> Iterable[Path]:
    preferred = robustad_root / "images"
    if preferred.is_dir():
        yield from sorted(path for path in preferred.iterdir() if path.is_dir())
        return

    for child in sorted(robustad_root.iterdir()):
        if not child.is_dir():
            continue
        if child.name.startswith(".") or child.name in {"__pycache__", "metadata", "annotations"}:
            continue
        if any(grandchild.is_dir() for grandchild in child.iterdir()):
            yield child


def verify_robustad(robustad_root: Path) -> None:
    print_header("RobustAD")
    print(f"root: {robustad_root}")
    if not robustad_root.is_dir():
        print("status: not found")
        return

    categories = list(iter_robustad_categories(robustad_root))
    if not categories:
        image_count = count_images(robustad_root)
        print(f"status: directory found but no category structure detected, recursive_images={image_count}")
        return

    print(f"categories_found: {len(categories)}")
    total_images = 0
    for category_dir in categories:
        split_counts: dict[str, int] = defaultdict(int)
        structure_dirs = set()
        for image_dir in list_image_dirs(category_dir):
            try:
                relative_parts = image_dir.relative_to(category_dir).parts
            except ValueError:
                continue
            if not relative_parts:
                continue
            split_counts[relative_parts[0]] += count_images(image_dir)
            structure_dirs.add("/".join(relative_parts[:2]))

        category_total = sum(split_counts.values())
        total_images += category_total
        if split_counts:
            split_counts_text = ", ".join(f"{split}={count}" for split, count in sorted(split_counts.items()))
        else:
            split_counts_text = f"recursive={count_images(category_dir)}"
        sample_structure = ", ".join(sorted(structure_dirs)[:4]) if structure_dirs else "n/a"
        print(f"- {category_dir.name}: {split_counts_text} [sample_structure: {sample_structure}]")

    print(f"total_images: {total_images}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify local dataset layout and image statistics.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/home/hun/Volume/DATA"),
        help="Root directory containing MVTec AD categories and RobustAD.",
    )
    parser.add_argument(
        "--robustad-dir",
        type=Path,
        default=None,
        help="Explicit RobustAD directory. Defaults to <data-root>/RobustAD.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = args.data_root.expanduser().resolve()
    robustad_root = args.robustad_dir.expanduser().resolve() if args.robustad_dir else data_root / "RobustAD"

    verify_mvtec(data_root)
    verify_robustad(robustad_root)


if __name__ == "__main__":
    main()
