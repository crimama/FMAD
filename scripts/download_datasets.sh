#!/usr/bin/env bash

set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/home/hun/Volume/DATA}"
ROBUSTAD_REPO="${ROBUSTAD_REPO:-AmazonScience/RobustAD}"
ROBUSTAD_DIR="${ROBUSTAD_DIR:-${DATA_ROOT}/RobustAD}"

MVTEC_CATEGORIES=(
  bottle
  cable
  capsule
  carpet
  grid
  hazelnut
  leather
  metal_nut
  pill
  screw
  tile
  toothbrush
  transistor
  wood
  zipper
)

print_mvtec_status() {
  local existing=0
  for category in "${MVTEC_CATEGORIES[@]}"; do
    if [[ -d "${DATA_ROOT}/${category}" ]]; then
      ((existing+=1))
    fi
  done

  echo "[MVTec AD] detected ${existing}/${#MVTEC_CATEGORIES[@]} expected category folders under ${DATA_ROOT}"
  if (( existing == ${#MVTEC_CATEGORIES[@]} )); then
    echo "[MVTec AD] existing dataset appears to be present. No automated download is performed."
  else
    echo "[MVTec AD] download manually from the official MVTec site:"
    echo "  https://www.mvtec.com/company/research/datasets/mvtec-ad-2"
    echo "[MVTec AD] place the extracted category folders (for example: bottle/, cable/) directly under ${DATA_ROOT}"
  fi
}

download_robustad() {
  if command -v huggingface-cli >/dev/null 2>&1; then
    mkdir -p "${ROBUSTAD_DIR}"
    echo "[RobustAD] downloading ${ROBUSTAD_REPO} to ${ROBUSTAD_DIR}"
    huggingface-cli download "${ROBUSTAD_REPO}" \
      --repo-type dataset \
      --local-dir "${ROBUSTAD_DIR}" \
      --local-dir-use-symlinks False
    echo "[RobustAD] download completed"
  else
    echo "[RobustAD] huggingface-cli not found."
    echo "[RobustAD] install it with: pip install -U \"huggingface_hub[cli]\""
    echo "[RobustAD] then run:"
    echo "  huggingface-cli download ${ROBUSTAD_REPO} --repo-type dataset --local-dir ${ROBUSTAD_DIR} --local-dir-use-symlinks False"
  fi
}

main() {
  mkdir -p "${DATA_ROOT}"
  print_mvtec_status
  download_robustad
}

main "$@"
