#!/usr/bin/env python3
"""Create a symlink-based MVTec AD compatible layout for MVTec AD 2.

MVTec AD 2 layout:
    {category}/train/good/
    {category}/test_public/good/
    {category}/test_public/bad/
    {category}/test_public/ground_truth/bad/

Target (MVTec AD compatible):
    {category}/train/good/       -> original
    {category}/test/good/        -> test_public/good
    {category}/test/bad/         -> test_public/bad
    {category}/ground_truth/bad/ -> test_public/ground_truth/bad
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

    categories = sorted(
        d.name for d in src.iterdir()
        if d.is_dir() and (d / "train").is_dir()
    )

    if not categories:
        print(f"No categories found in {src}", file=sys.stderr)
        sys.exit(1)

    for cat in categories:
        cat_src = src / cat
        cat_dst = dst / cat

        # train/good -> symlink to original
        train_dst = cat_dst / "train" / "good"
        train_src = cat_src / "train" / "good"
        if train_src.is_dir():
            train_dst.parent.mkdir(parents=True, exist_ok=True)
            if train_dst.exists() or train_dst.is_symlink():
                train_dst.unlink()
            os.symlink(train_src, train_dst)

        # test/good, test/bad -> symlink to test_public/*
        test_pub = cat_src / "test_public"
        if test_pub.is_dir():
            for sub in ("good", "bad"):
                sub_src = test_pub / sub
                if sub_src.is_dir():
                    sub_dst = cat_dst / "test" / sub
                    sub_dst.parent.mkdir(parents=True, exist_ok=True)
                    if sub_dst.exists() or sub_dst.is_symlink():
                        sub_dst.unlink()
                    os.symlink(sub_src, sub_dst)

            # ground_truth/bad -> test_public/ground_truth/bad
            gt_src = test_pub / "ground_truth" / "bad"
            if gt_src.is_dir():
                gt_dst = cat_dst / "ground_truth" / "bad"
                gt_dst.parent.mkdir(parents=True, exist_ok=True)
                if gt_dst.exists() or gt_dst.is_symlink():
                    gt_dst.unlink()
                os.symlink(gt_src, gt_dst)

        print(f"  {cat}: OK")

    print(f"\nPrepared {len(categories)} categories in {dst}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="MVTec AD 2 root")
    parser.add_argument("--dst", required=True, help="Output compatible root")
    args = parser.parse_args()
    prepare(args.src, args.dst)
