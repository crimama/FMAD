# program.md — NSP (Nuisance Subspace Projection) 자율 실험

## 연구 목표

FM-AD의 distribution shift 취약성을 post-hoc feature disentanglement로 해결한다.
MVTec AD 2에서의 kNN 기반 anomaly detection 성능을 개선하되, MVTec AD (clean) 성능은 유지한다.

**목표**: MVTec AD 2 I-AUROC를 최대화하되, MVTec AD I-AUROC 하락 <1pp
**지표**: `ad2_auroc` — higher is better (secondary: `ad1_auroc` — 하락 없어야 함)
**현재 베이스라인**:
- kNN baseline (no projection): AD2 54.4%, AD1 91.6%
- NSP K=30 linear: AD2 67.9% (+13.5pp), AD1 91.9%
- 측정일: 2026-03-26

## 파일 스코프

### 수정 가능 (에이전트의 샌드박스)

- `scripts/phase3_nsp_experiment.py` — NSP 방법론 코드 (feature extraction, nuisance estimation, projection, scoring)
  - `estimate_nuisance_subspace()` — nuisance direction 추정 방법
  - `project_out_nuisance()` — projection 방법
  - `knn_anomaly_score()` — anomaly scoring 방법
  - `DINOv2Extractor` — feature 추출 설정 (layer, pooling 등)

### 읽기 전용 (절대 수정 금지)

- `program.md` — 이 파일 자체
- `scripts/phase2_feature_analysis.py` — Phase 2 분석 코드
- `scripts/phase2_entanglement_analysis.py` — entanglement 분석 코드
- `scripts/prepare_mvtecad2_compat.py` — 데이터 레이아웃
- `scripts/prepare_robustad_compat.py` — 데이터 레이아웃

### 무시

- `baselines/` — baseline 코드 (이미 평가 완료)
- `docker/` — Docker 설정
- `skill_graph/` — 연구 노트 (실험 결과 기록은 별도)

## 실행 환경

### 실행 명령어
```bash
docker run --rm --gpus all --shm-size=4g \
  -v /home/hun/Volume/DATA:/home/hun/Volume/DATA \
  -v $(pwd)/scripts:/workspace/scripts \
  -v $(pwd)/results:/workspace/results \
  pilot-anomalydino bash -c "
    pip install scikit-learn tqdm -q 2>/dev/null
    python3 /workspace/scripts/phase3_nsp_experiment.py \
      --data_root_original /home/hun/Volume/DATA/mvtec_ad_2 \
      --data_root_ad2 /home/hun/Volume/DATA/mvtec_ad_2_compat \
      --data_root_ad1 /home/hun/Volume/DATA/MVTecAD \
      --layer 11 \
      --n_components 10 \
      --k 5 \
      --output_dir /workspace/results/phase3
  " > run.log 2>&1
```

### 시간 예산
- **실험 시간**: ~5분 (feature extraction + kNN scoring)
- **타임아웃**: 15분 초과 시 kill & discard

### 지표 추출
```bash
grep "MEAN" run.log | tail -2
# 첫 줄: MVTec AD 2 결과 (ad2_auroc)
# 둘째 줄: MVTec AD 결과 (ad1_auroc)
```

### 하드웨어
- **GPU**: NVIDIA RTX 4090
- **VRAM**: 24GB — feature extraction만 하므로 VRAM 걱정 없음

## 실험 전략

### 우선 탐색 영역

1. **Nuisance estimation 개선** — 현재: shift vector PCA. 시도: robust PCA, ICA, shift vector 가중치 (condition별), per-category nuisance 등
2. **Projection 방법 변경** — 현재: hard orthogonal projection. 시도: soft projection (부분 제거), learnable mixing ratio, whitening 후 projection 등
3. **Feature 표현 변경** — 현재: patch token mean. 시도: CLS token, multi-layer fusion, patch-level scoring, token norm 정규화 등
4. **Scoring 개선** — 현재: kNN k=5. 시도: k 조정, Mahalanobis distance, projected space에서 covariance 활용 등
5. **Multi-layer NSP** — 여러 layer에서 각각 nuisance 제거 후 fusion

### 시도하지 말 것

- 학습 기반 방법 (LoRA, fine-tuning 등) — 이 실험은 post-hoc, training-free에 집중
- CLIP backbone — DINOv2 ViT-B/14에서만 실험
- 데이터 augmentation — 실제 shift pair만 사용
- MVTec AD 2 외 데이터셋 — 다른 벤치마크는 나중에

### 단순함 기준 (Simplicity Criterion)

- 복잡한 변경이 +1pp 미만 개선 → discard
- 코드 50줄 이상 추가가 +3pp 미만 개선 → discard
- 단순화로 동등 성능 → keep

## 판정 기준

### Keep 조건
- `ad2_auroc`가 이전 최고 대비 **+0.5pp 이상** 개선
- 그리고 `ad1_auroc` 하락이 **1pp 이하**

### Discard 조건
- `ad2_auroc` 개선이 0.5pp 미만
- 또는 `ad1_auroc`가 1pp 이상 하락

### Crash 처리
- 단순 버그 → 수정 후 재실행 (최대 3회)
- OOM/설계 결함 → crash 기록, 다음 실험

## 자율성 규칙

1. **절대 멈추지 않는다** — 사용자가 수동 중단할 때까지 루프를 계속한다
2. **질문하지 않는다** — 사람이 자고 있을 수 있다
3. **아이디어 고갈 시** — program.md 재독, 이전 실패 패턴 분석, 더 급진적 변경 시도
4. **실패도 기록한다** — crash/discard도 results.tsv에 남긴다

## 현재까지의 발견 (Phase 1-3 요약)

- DINOv2 layer 8이 uniquely shift-robust (RelShift 0.11 vs 다른 layer 0.22-0.27)
- Layer 11에서 shift-anomaly PC projection correlation = 0.53 (중간 entanglement)
- Linear NSP K=30: +13.5pp AD2, +0.3pp AD1 (K에 대해 monotonic, saturation 미도달)
- vial/sheet_metal에서 효과 극대 (feature shift가 큰 카테고리)
- can/fruit_jelly에서 효과 미미 (nonlinear shift 가능성)
