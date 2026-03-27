#!/usr/bin/env bash
# RobustAD 카테고리별 이어받기 스크립트
# - 이미 완료된 카테고리는 스킵
# - rate limit 회피를 위해 카테고리 간 딜레이
# - 실패 시 자동 재시도 (backoff)
#
# Usage:
#   bash scripts/download_robustad_resume.sh              # 전체 다운로드
#   bash scripts/download_robustad_resume.sh Zipper Nut   # 특정 카테고리만

set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/home/hun/Volume/DATA}"
ROBUSTAD_REPO="AmazonScience/RobustAD"
ROBUSTAD_DIR="${DATA_ROOT}/RobustAD"
DELAY_BETWEEN="${DELAY_BETWEEN:-120}"   # 카테고리 간 대기 (초)
MAX_RETRIES="${MAX_RETRIES:-5}"
RETRY_BACKOFF="${RETRY_BACKOFF:-300}"   # 실패 시 초기 대기 (초), 재시도마다 2배

# RobustAD 전체 카테고리 (HF repo 기준)
ALL_CATEGORIES=(
  Breakfast_box
  Juice_bottle
  Pushpins
  Screw_bag
  Splicing_connectors
  MetalParts
  PCB
  Zipper
  Nut
  Capacitor
  Casting
  Console
  Cylinder
  Electronics
  Groove
  Hemisphere
  Lens
  Magnetics
  MetalPlate
  PillBottle
  Ring
  ScrewNut
  Spring
  Tape
  Transistor
  Tube
  USB
  Washer
  Wood
)

# 완료 판정: 카테고리 폴더가 존재하고 파일이 10개 이상이면 완료로 간주
is_done() {
  local cat_dir="${ROBUSTAD_DIR}/$1"
  if [[ -d "$cat_dir" ]]; then
    local count
    count=$(find "$cat_dir" -type f 2>/dev/null | wc -l)
    if (( count >= 10 )); then
      return 0
    fi
  fi
  return 1
}

download_category() {
  local category="$1"
  local attempt=0
  local wait_time="$RETRY_BACKOFF"

  while (( attempt < MAX_RETRIES )); do
    echo "[$(date '+%H:%M:%S')] Downloading: ${category} (attempt $((attempt+1))/${MAX_RETRIES})"

    if hf download "$ROBUSTAD_REPO" \
        --repo-type dataset \
        --include "${category}/*" \
        --local-dir "$ROBUSTAD_DIR" 2>&1; then
      echo "[$(date '+%H:%M:%S')] ✓ ${category} done"
      return 0
    fi

    ((attempt+=1))
    if (( attempt < MAX_RETRIES )); then
      echo "[$(date '+%H:%M:%S')] Rate limited. Waiting ${wait_time}s before retry..."
      sleep "$wait_time"
      wait_time=$((wait_time * 2))
    fi
  done

  echo "[$(date '+%H:%M:%S')] ✗ ${category} FAILED after ${MAX_RETRIES} attempts"
  return 1
}

main() {
  mkdir -p "$ROBUSTAD_DIR"

  # 대상 카테고리 결정
  local categories=()
  if (( $# > 0 )); then
    categories=("$@")
  else
    categories=("${ALL_CATEGORIES[@]}")
  fi

  # 상태 출력
  local total=${#categories[@]}
  local skipped=0
  local todo=()

  for cat in "${categories[@]}"; do
    if is_done "$cat"; then
      ((skipped+=1))
    else
      todo+=("$cat")
    fi
  done

  echo "============================================"
  echo " RobustAD Resume Download"
  echo " Total: ${total} | Already done: ${skipped} | Remaining: ${#todo[@]}"
  echo " Delay between categories: ${DELAY_BETWEEN}s"
  echo " Target: ${ROBUSTAD_DIR}"
  echo "============================================"

  if (( ${#todo[@]} == 0 )); then
    echo "All categories already downloaded!"
    return 0
  fi

  echo "To download: ${todo[*]}"
  echo ""

  local failed=()
  for i in "${!todo[@]}"; do
    local cat="${todo[$i]}"

    if ! download_category "$cat"; then
      failed+=("$cat")
    fi

    # 마지막이 아니면 딜레이
    if (( i < ${#todo[@]} - 1 )); then
      echo "[$(date '+%H:%M:%S')] Waiting ${DELAY_BETWEEN}s before next category..."
      sleep "$DELAY_BETWEEN"
    fi
  done

  echo ""
  echo "============================================"
  echo " Download Summary"
  echo " Succeeded: $(( ${#todo[@]} - ${#failed[@]} )) / ${#todo[@]}"
  if (( ${#failed[@]} > 0 )); then
    echo " Failed: ${failed[*]}"
    echo " Re-run to retry failed categories:"
    echo "   bash scripts/download_robustad_resume.sh ${failed[*]}"
  fi
  echo "============================================"
}

main "$@"
