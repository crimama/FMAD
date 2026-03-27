#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_DIR="${ROOT_DIR}/baselines/dinomaly"
SOURCE_DIR="${REPO_DIR}/upstream"
RESULTS_DIR="${ROOT_DIR}/results/phase1"
PATCH_DIR="${ROOT_DIR}/baselines/dinomaly/patches"
PARSER="${ROOT_DIR}/scripts/collect_dinomaly_results.py"

PYTHON_BIN="${PYTHON_BIN:-python}"
BENCHMARK="${1:-all}"

MVTEC_DATA_PATH="${MVTEC_DATA_PATH:-/home/hun/Volume/DATA}"
MVTEC_AD2_DATA_PATH="${MVTEC_AD2_DATA_PATH:-/home/hun/Volume/DATA/mvtec_ad2}"
ROBUSTAD_DATA_PATH="${ROBUSTAD_DATA_PATH:-/home/hun/Volume/DATA/RobustAD}"
DINOMALY_REPO_URL="${DINOMALY_REPO_URL:-https://github.com/guojiajeremy/Dinomaly}"
MVTEC_CHECKPOINT_URL="${MVTEC_CHECKPOINT_URL:-https://drive.google.com/file/d/1UvpX0hTZ48FYTxlrDwYa-NZYZcFwjc8r/view}"

ensure_repo() {
  if [[ -f "${SOURCE_DIR}/requirements.txt" && -f "${SOURCE_DIR}/dinomaly_mvtec_uni.py" ]]; then
    return
  fi

  rm -rf "${SOURCE_DIR}"
  mkdir -p "${REPO_DIR}"
  git clone "${DINOMALY_REPO_URL}" "${SOURCE_DIR}"
}

apply_optional_patch() {
  local patch_file="${PATCH_DIR}/dataset_mvtecad2_robustad.patch"
  if [[ ! -f "${patch_file}" || ! -f "${SOURCE_DIR}/dataset.py" ]]; then
    return
  fi

  if grep -q "class MVTecAD2Dataset" "${SOURCE_DIR}/dataset.py"; then
    return
  fi

  (
    cd "${SOURCE_DIR}"
    patch -p1 < "${patch_file}" || true
  )
}

print_checkpoint_note() {
  local checkpoint_dir="${SOURCE_DIR}/checkpoints"
  mkdir -p "${checkpoint_dir}"
  if [[ ! -f "${checkpoint_dir}/mvtec_ad_dinomaly.pth" ]]; then
    cat <<EOF
[dinomaly] pretrained checkpoint not found:
  expected: ${checkpoint_dir}/mvtec_ad_dinomaly.pth
  download : ${MVTEC_CHECKPOINT_URL}
EOF
  fi
}

run_one() {
  local benchmark_name="$1"
  local data_path="$2"
  local entrypoint="$3"
  local result_json="${RESULTS_DIR}/dinomaly_${benchmark_name}.json"
  local log_file

  mkdir -p "$(dirname "${result_json}")"
  log_file="$(mktemp)"

  (
    cd "${SOURCE_DIR}"
    "${PYTHON_BIN}" "${entrypoint}" --data_path "${data_path}"
  ) | tee "${log_file}"

  "${PYTHON_BIN}" "${PARSER}" \
    --benchmark "${benchmark_name}" \
    --entrypoint "${entrypoint}" \
    --data-path "${data_path}" \
    --log "${log_file}" \
    --output "${result_json}"
}

run_selected() {
  case "${BENCHMARK}" in
    mvtec)
      run_one "mvtec" "${MVTEC_DATA_PATH}" "dinomaly_mvtec_uni.py"
      ;;
    mvtec_ad2)
      apply_optional_patch
      run_one "mvtec_ad2" "${MVTEC_AD2_DATA_PATH}" "dinomaly_mvtec_uni.py"
      ;;
    robustad)
      apply_optional_patch
      run_one "robustad" "${ROBUSTAD_DATA_PATH}" "dinomaly_mvtec_uni.py"
      ;;
    all)
      run_one "mvtec" "${MVTEC_DATA_PATH}" "dinomaly_mvtec_uni.py"
      apply_optional_patch
      run_one "mvtec_ad2" "${MVTEC_AD2_DATA_PATH}" "dinomaly_mvtec_uni.py"
      run_one "robustad" "${ROBUSTAD_DATA_PATH}" "dinomaly_mvtec_uni.py"
      ;;
    *)
      echo "usage: $0 [all|mvtec|mvtec_ad2|robustad]" >&2
      exit 1
      ;;
  esac
}

ensure_repo
print_checkpoint_note
run_selected
