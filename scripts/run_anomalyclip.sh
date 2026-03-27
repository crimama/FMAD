#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ANOMALYCLIP_DIR="${ROOT_DIR}/baselines/anomalyclip"
RESULTS_DIR="${ROOT_DIR}/results/phase1"
PYTHON_BIN="${PYTHON_BIN:-python3}"

mkdir -p "${RESULTS_DIR}"

resolve_path() {
  local candidate
  for candidate in "$@"; do
    if [[ -n "${candidate}" && -d "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  return 1
}

write_status_json() {
  local dataset="$1"
  local status="$2"
  local message="$3"
  DATASET="${dataset}" STATUS="${status}" MESSAGE="${message}" OUTPUT_PATH="${RESULTS_DIR}/anomalyclip_${dataset}.json" "${PYTHON_BIN}" - <<'PY'
import json
import os

payload = {
    "method": "anomalyclip",
    "dataset": os.environ["DATASET"],
    "status": os.environ["STATUS"],
    "error": os.environ["MESSAGE"],
    "metrics": {
        "I-AUROC": None,
        "P-AUROC": None,
        "AU-PRO": None,
    },
}
with open(os.environ["OUTPUT_PATH"], "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2)
    handle.write("\n")
PY
}

VISA_DATA_PATH="${VISA_DATA_PATH:-}"
if [[ -z "${VISA_DATA_PATH}" ]]; then
  VISA_DATA_PATH="$(resolve_path \
    "/home/hun/Volume/DATA/visa" \
    "/home/hun/Volume/DATA/VISA" \
    "/home/hun/Volume/RESEARCH/VAD/Data/VISA" || true)"
fi
if [[ -z "${VISA_DATA_PATH}" ]]; then
  echo "VisA dataset not found." >&2
  exit 1
fi

MVTec_DATA_PATH="${MVTEC_DATA_PATH:-}"
if [[ -z "${MVTec_DATA_PATH}" ]]; then
  MVTec_DATA_PATH="$(resolve_path \
    "/home/hun/Volume/DATA" \
    "/home/hun/Volume/DATA/mvtec" \
    "/home/hun/Volume/RESEARCH/VAD/Data/MVTecAD" || true)"
fi

MVTec_AD2_DATA_PATH="${MVTEC_AD2_DATA_PATH:-}"
if [[ -z "${MVTec_AD2_DATA_PATH}" ]]; then
  MVTec_AD2_DATA_PATH="$(resolve_path \
    "/home/hun/Volume/DATA/mvtec_ad_2" \
    "/home/hun/Volume/DATA/MVTecAD2" \
    "/home/hun/Volume/DATA/mvtec2d-sam-b" \
    "/home/hun/Volume/RESEARCH/VAD/Data/mvtec2d-sam-b" || true)"
fi

ROBUSTAD_DATA_PATH="${ROBUSTAD_DATA_PATH:-}"
if [[ -z "${ROBUSTAD_DATA_PATH}" ]]; then
  ROBUSTAD_DATA_PATH="$(resolve_path \
    "/home/hun/Volume/DATA/RobustAD" \
    "/home/hun/Volume/DATA/robustad" || true)"
fi

pushd "${ANOMALYCLIP_DIR}" >/dev/null

CHECKPOINT_DIR="${ANOMALYCLIP_DIR}/checkpoints"
CHECKPOINT_PATH="${CHECKPOINT_DIR}/epoch_15.pth"

"${PYTHON_BIN}" train.py \
  --dataset visa \
  --train_data_path "${VISA_DATA_PATH}" \
  --save_path "${CHECKPOINT_DIR}" \
  --features_list 24 \
  --image_size 518 \
  --batch_size 8 \
  --epoch 15 \
  --depth 9 \
  --n_ctx 12 \
  --t_n_ctx 4

run_eval() {
  local dataset="$1"
  local data_path="$2"
  local save_path="${ANOMALYCLIP_DIR}/results/${dataset}"
  local json_path="${RESULTS_DIR}/anomalyclip_${dataset}.json"

  if [[ -z "${data_path}" || ! -d "${data_path}" ]]; then
    write_status_json "${dataset}" "missing_dataset" "Dataset path not found"
    return 0
  fi

  "${PYTHON_BIN}" test.py \
    --dataset "${dataset}" \
    --data_path "${data_path}" \
    --checkpoint_path "${CHECKPOINT_PATH}" \
    --save_path "${save_path}" \
    --features_list 24 \
    --image_size 518 \
    --depth 9 \
    --n_ctx 12 \
    --t_n_ctx 4 \
    --results_json "${json_path}" \
    --method_name anomalyclip
}

run_eval "mvtec" "${MVTec_DATA_PATH}"
run_eval "mvtec_ad_2" "${MVTec_AD2_DATA_PATH}"
run_eval "robustad" "${ROBUSTAD_DATA_PATH}"

popd >/dev/null

"${PYTHON_BIN}" "${ROOT_DIR}/scripts/collect_results.py" \
  --results-dir "${RESULTS_DIR}" \
  --output "${RESULTS_DIR}/anomalyclip_summary.md"
