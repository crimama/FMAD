#!/usr/bin/env python3
"""Create MVTec AD compatible symlink layout for RobustAD.

RobustAD layout:
    {Category}/{prefix}_data_dir_train/normal/
    {Category}/{prefix}_data_dir_test{N}/normal/
    {Category}/{prefix}_data_dir_test{N}/anomaly/
    {Category}/{prefix}_data_dir_test{N}/masks/

Target (MVTec AD compatible, per test domain):
    {Category}_test{N}/train/good/        -> train/normal
    {Category}_test{N}/test/good/         -> test{N}/normal
    {Category}_test{N}/test/bad/          -> test{N}/anomaly
    {Category}_test{N}/ground_truth/bad/  -> test{N}/masks

Usage:
    python prepare_robustad_compat.py --src /path/to/RobustAD --dst /path/to/RobustAD_compat
    # Creates one "virtual category" per (Category, testN) pair
"""

import argparse
import os
import sys
from pathlib import Path


def prepare(src_root: str, dst_root: str):
    src = Path(src_root).resolve()
    dst = Path(dst_root).resolve()

    if not src.is_dir():
        print(f"Source not found: {src}", file=sys.stderr)
        sys.exit(1)

    categories = sorted(d.name for d in src.iterdir() if d.is_dir())
    total = 0

    for cat in categories:
        cat_dir = src / cat
        subdirs = sorted(d.name for d in cat_dir.iterdir() if d.is_dir())

        # Find train dir
        train_dirs = [d for d in subdirs if "train" in d]
        test_dirs = [d for d in subdirs if "test" in d and "train" not in d]

        if not train_dirs:
            print(f"  {cat}: No train dir found, skipping")
            continue

        train_dir = cat_dir / train_dirs[0]
        train_normal = train_dir / "normal"

        if not train_normal.is_dir():
            print(f"  {cat}: No normal/ in train dir, skipping")
            continue

        for test_name in test_dirs:
            # Extract test index (e.g., "test0" from "pcb_data_dir_test0")
            test_idx = test_name.split("test")[-1]
            virtual_cat = f"{cat}_test{test_idx}"

            test_dir = cat_dir / test_name
            test_normal = test_dir / "normal"
            test_anomaly = test_dir / "anomaly"
            test_masks = test_dir / "masks"

            vcat_dst = dst / virtual_cat

            # train/good -> train/normal
            _symlink(train_normal, vcat_dst / "train" / "good")

            # test/good -> testN/normal
            if test_normal.is_dir():
                _symlink(test_normal, vcat_dst / "test" / "good")

            # test/bad -> testN/anomaly
            if test_anomaly.is_dir():
                _symlink(test_anomaly, vcat_dst / "test" / "bad")

            # ground_truth/bad -> testN/masks
            if test_masks.is_dir():
                _symlink(test_masks, vcat_dst / "ground_truth" / "bad")

            total += 1
            print(f"  {virtual_cat}: OK")

    print(f"\nPrepared {total} virtual categories in {dst}")


def _symlink(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    os.symlink(src, dst)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="RobustAD root")
    parser.add_argument("--dst", required=True, help="Output compatible root")
    args = parser.parse_args()
    prepare(args.src, args.dst)
