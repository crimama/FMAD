#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASELINE_ROOT="${REPO_ROOT}/baselines/anomalydino"
UPSTREAM_DIR="${ANOMALYDINO_DIR:-${BASELINE_ROOT}/upstream}"
RESULTS_ROOT="${REPO_ROOT}/results/phase1"
TMP_ROOT="${TMPDIR:-/tmp}/anomalydino_phase1"

MODEL_NAME="${ANOMALYDINO_MODEL:-dinov2_vits14}"
RESOLUTION="${ANOMALYDINO_RESOLUTION:-448}"
NUM_SEEDS="${ANOMALYDINO_NUM_SEEDS:-3}"
SHOTS=(${ANOMALYDINO_SHOTS:-1 4 8})

MVTec_ROOT="${ANOMALYDINO_MVTEC_ROOT:-/home/hun/Volume/DATA}"
MVTecAD2_ROOT="${ANOMALYDINO_MVTECAD2_ROOT:-/home/hun/Volume/DATA/mvtec_ad2}"
ROBUSTAD_ROOT="${ANOMALYDINO_ROBUSTAD_ROOT:-/home/hun/Volume/DATA/RobustAD}"

mkdir -p "${RESULTS_ROOT}" "${TMP_ROOT}"

ensure_upstream_repo() {
    if [[ -f "${UPSTREAM_DIR}/run_anomalydino.py" ]]; then
        return 0
    fi

    if [[ -d "${UPSTREAM_DIR}" ]]; then
        echo "AnomalyDINO source directory exists but run_anomalydino.py is missing: ${UPSTREAM_DIR}" >&2
        echo "Populate ${UPSTREAM_DIR} with https://github.com/dammsi/AnomalyDINO or set ANOMALYDINO_DIR." >&2
        return 1
    fi

    echo "Cloning AnomalyDINO into ${UPSTREAM_DIR}" >&2
    git clone https://github.com/dammsi/AnomalyDINO "${UPSTREAM_DIR}"
}

prepare_dataset_root() {
    local source_root="$1"
    local normalized_root="$2"
    local dataset_label="$3"

    if [[ ! -d "${source_root}" ]]; then
        echo "Dataset root not found for ${dataset_label}: ${source_root}" >&2
        return 1
    fi

    python3 - "${source_root}" "${normalized_root}" "${dataset_label}" <<'PY'
import os
import shutil
import sys
from pathlib import Path

source_root = Path(sys.argv[1]).resolve()
normalized_root = Path(sys.argv[2]).resolve()
dataset_label = sys.argv[3]


def child_dirs(path: Path):
    return sorted([p for p in path.iterdir() if p.is_dir()])


def is_category_centric(path: Path) -> bool:
    categories = child_dirs(path)
    if not categories:
        return False
    return all((cat / "train").is_dir() and (cat / "test").is_dir() for cat in categories)


def is_split_centric(path: Path) -> bool:
    return (path / "train").is_dir() and (path / "test").is_dir()


def discover_root(path: Path) -> Path:
    current = path
    for _ in range(4):
        if is_category_centric(current) or is_split_centric(current):
            return current
        dirs = child_dirs(current)
        if len(dirs) != 1:
            break
        current = dirs[0]
    raise SystemExit(
        f"Unsupported layout for {dataset_label}: {path}. "
        "Expected object-centric '<root>/<object>/train|test' or split-centric '<root>/train/<object>'."
    )


def symlink(src: Path, dst: Path):
    if dst.exists() or dst.is_symlink():
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    os.symlink(src, dst, target_is_directory=True)


resolved_root = discover_root(source_root)
if normalized_root.exists():
    shutil.rmtree(normalized_root)
normalized_root.mkdir(parents=True, exist_ok=True)

if is_category_centric(resolved_root):
    for category in child_dirs(resolved_root):
        symlink(category, normalized_root / category.name)
else:
    train_root = resolved_root / "train"
    test_root = resolved_root / "test"
    gt_root = resolved_root / "ground_truth"
    categories = sorted({p.name for p in child_dirs(train_root)} | {p.name for p in child_dirs(test_root)})
    if not categories:
        raise SystemExit(f"No categories found under split-centric root: {resolved_root}")
    for category in categories:
        dst = normalized_root / category
        dst.mkdir(parents=True, exist_ok=True)
        if (train_root / category).is_dir():
            symlink(train_root / category, dst / "train")
        if (test_root / category).is_dir():
            symlink(test_root / category, dst / "test")
        if (gt_root / category).is_dir():
            symlink(gt_root / category, dst / "ground_truth")

print(normalized_root)
PY
}

collect_results() {
    local dataset_arg="$1"
    local dataset_slug="$2"
    local source_root="$3"
    local prepared_root="$4"
    local preprocess="$5"
    local tag="$6"

    local output_json="${RESULTS_ROOT}/anomalydino_${dataset_slug}.json"

    python3 - "${UPSTREAM_DIR}" "${dataset_arg}" "${dataset_slug}" "${source_root}" "${prepared_root}" "${MODEL_NAME}" "${RESOLUTION}" "${NUM_SEEDS}" "${preprocess}" "${tag}" "${output_json}" "${SHOTS[@]}" <<'PY'
import json
import sys
from pathlib import Path

upstream_dir = Path(sys.argv[1]).resolve()
dataset_arg = sys.argv[2]
dataset_slug = sys.argv[3]
source_root = sys.argv[4]
prepared_root = sys.argv[5]
model_name = sys.argv[6]
resolution = int(sys.argv[7])
num_seeds = int(sys.argv[8])
preprocess = sys.argv[9]
tag = sys.argv[10]
output_json = Path(sys.argv[11]).resolve()
shots = [int(v) for v in sys.argv[12:]]

results_base = upstream_dir / f"results_{dataset_arg}" / f"{model_name}_{resolution}"
runs = []
missing = []

for shot in shots:
    shot_dir = results_base / f"{shot}-shot_preprocess={preprocess}_{tag}"
    for seed in range(num_seeds):
        metrics_path = shot_dir / f"metrics_seed={seed}.json"
        if metrics_path.is_file():
            with metrics_path.open() as fh:
                metrics = json.load(fh)
            runs.append(
                {
                    "shot": shot,
                    "seed": seed,
                    "metrics_path": str(metrics_path),
                    "metrics": metrics,
                }
            )
        else:
            missing.append({"shot": shot, "seed": seed, "expected_metrics_path": str(metrics_path)})

payload = {
    "method": "AnomalyDINO",
    "dataset": dataset_slug,
    "upstream_dataset_arg": dataset_arg,
    "source_data_root": source_root,
    "prepared_data_root": prepared_root,
    "shots": shots,
    "num_seeds": num_seeds,
    "model_name": model_name,
    "resolution": resolution,
    "preprocess": preprocess,
    "tag": tag,
    "results_base": str(results_base),
    "completed_runs": len(runs),
    "expected_runs": len(shots) * num_seeds,
    "runs": runs,
    "missing_runs": missing,
}

output_json.parent.mkdir(parents=True, exist_ok=True)
with output_json.open("w") as fh:
    json.dump(payload, fh, indent=2)

print(output_json)
PY
}

run_dataset() {
    local dataset_arg="$1"
    local dataset_slug="$2"
    local source_root="$3"
    local prepared_root="$4"
    local preprocess="$5"

    local tag="phase1_${dataset_slug}"

    pushd "${UPSTREAM_DIR}" >/dev/null
    python3 run_anomalydino.py \
        --dataset "${dataset_arg}" \
        --shots "${SHOTS[@]}" \
        --num_seeds "${NUM_SEEDS}" \
        --preprocess "${preprocess}" \
        --data_root "${prepared_root}" \
        --model_name "${MODEL_NAME}" \
        --resolution "${RESOLUTION}" \
        --tag "${tag}"
    popd >/dev/null

    collect_results "${dataset_arg}" "${dataset_slug}" "${source_root}" "${prepared_root}" "${preprocess}" "${tag}"
}

ensure_upstream_repo

run_dataset "MVTec" "mvtec" "${MVTec_ROOT}" "${MVTec_ROOT}" "agnostic"

MVTecAD2_PREPARED="${TMP_ROOT}/mvtec_ad2"
prepare_dataset_root "${MVTecAD2_ROOT}" "${MVTecAD2_PREPARED}" "MVTec AD 2"
run_dataset "MVTecAD2" "mvtec_ad2" "${MVTecAD2_ROOT}" "${MVTecAD2_PREPARED}" "agnostic_no_mask"

ROBUSTAD_PREPARED="${TMP_ROOT}/robustad"
prepare_dataset_root "${ROBUSTAD_ROOT}" "${ROBUSTAD_PREPARED}" "RobustAD"
run_dataset "RobustAD" "robustad" "${ROBUSTAD_ROOT}" "${ROBUSTAD_PREPARED}" "agnostic_no_mask"
