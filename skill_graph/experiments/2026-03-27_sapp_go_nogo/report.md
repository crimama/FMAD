# SAPP GO/NO-GO: Spectral Anomaly-Preserving Projection — 2026-03-27

> **상태**: 🟡 진행중
> **실험 ID**: `exp_20260327_sapp`
> **keywords**: SAPP, variance-gated projection, anomaly-preserving, spectral gap

---

## 1. 문제 분석 (Problem Analysis)

### 현상
- NSP(PCA + hard projection)가 +29.4pp 개선을 달성하지만, Layer 8에서만 잘 작동
- Layer 11(entanglement 0.53)에서는 hard projection이 anomaly 정보도 함께 제거
- 10개 복잡한 변형이 모두 실패 → 하지만 이들은 모두 "nuisance estimation" 또는 "projection 방법"의 변형이지, "anomaly 구조를 고려한 projection"은 시도하지 않음

### 원인 추정
- NSP의 hard 0/1 projection이 entangled direction에서 anomaly info를 불가피하게 손실
- AD의 one-class 구조(normal covariance의 spectral gap)를 projection 시 활용하지 않음

### 관련 선행 실험
- [Phase 3 NSP 결과](../2026-03-26_phase3_linear_disentanglement/report.md)
- [Phase 2 Entanglement 분석](../2026-03-26_phase2_feature_shift_analysis/report.md)

---

## 2. 가설 (Hypothesis)

### 주 가설
> Nuisance direction 중 normal covariance에서 분산이 낮은 방향(anomaly-sensitive)은 보호하고, 분산이 높은 방향만 제거하면, Layer 11에서도 NSP 수준 이상의 성능을 달성할 수 있다.

### GO/NO-GO 기준
- **GO**: ω 분포의 spread ratio (max/min) > 5x → nuisance direction 간 anomaly sensitivity 차이가 유의
- **NO-GO**: spread ratio < 5x → 모든 nuisance direction이 비슷한 anomaly sensitivity → gating 무의미

### 예상 결과
| 지표 | Layer 11 NSP | Layer 11 SAPP | Layer 8 NSP | Layer 8 SAPP |
|------|-------------|--------------|-------------|-------------|
| AD2 I-AUROC | ~71% | 74-79% | ~84% | ~84% (동등) |

---

## 3. 실험 설정 (Experiment Design)

### 방법
1. Shift vector PCA → nuisance directions n_1,...,n_K (K=100)
2. 각 n_k의 anomaly sensitivity: ω_k = n_k^T · Σ_train^{-1} · n_k
3. Gated projection: g_k = σ(-α·(ω_k - τ)), f_clean = f - Σ g_k·(f·n_k)·n_k
4. Mahalanobis scoring

### 대조군
- Baseline (no projection)
- NSP (hard projection, K=100)

### 실험군
- SAPP α=1, 3, 5, 10 (gating sharpness sweep)
- Layer 8 + Layer 11 양쪽에서 실험

### 평가 지표
- Primary: AD2 I-AUROC (per-category + mean)
- Secondary: AD1 I-AUROC (clean 하락 없어야 함)

---

## 4. 결과 (Results)

### ω 분포 (GO/NO-GO)

| Layer | ω range | ω mean | Spread ratio | 판정 |
|-------|---------|--------|-------------|------|
| 11 | 40K ~ 193K | 137K | 4.8x | ❌ NO-GO |
| 8 | 110K ~ 366K | 275K | 3.3x | ❌ NO-GO |

ω 분포가 좁고 전체적으로 높음 → 모든 nuisance direction이 비슷한 anomaly sensitivity → gating 무의미

### AD 성능 비교

| Layer | Baseline | NSP | SAPP α=1 | SAPP α=3 | SAPP α=5 | SAPP α=10 |
|-------|---------|-----|---------|---------|---------|----------|
| 11 AD2 | 59.7% | **79.1%** | 70.2% | 70.2% | 70.2% | 70.2% |
| 11 AD1 | 97.4% | 97.4% | 97.5% | 97.5% | 97.5% | 97.5% |
| 8 AD2 | 59.9% | **83.8%** | 77.8% | 77.8% | 77.8% | 77.8% |

**SAPP는 모든 설정에서 NSP보다 나쁨**: L11 -8.9pp, L8 -6.0pp

---

## 5. 결과 분석 (Analysis)

### 가설 검증
> **결론**: ❌ 가설 기각 — ω 기반 gating은 anomaly-sensitive direction을 올바르게 식별하지 못함

### 실패 원인 분석

1. **ω의 차별력 부족**: L2 normalized feature에서 Σ_train^{-1}의 eigenvalue가 전체적으로 크고 균일. nuisance direction이 normal covariance의 특정 축과 정렬되지 않음.

2. **"Normal 저분산 = anomaly-sensitive" 가정의 한계**: 이 가정은 anomaly가 normal distribution의 tail에서만 발생할 때 성립. 실제로는 anomaly direction이 normal의 고분산 축과도 관련될 수 있음.

3. **α에 무관한 결과**: 4가지 α(1,3,5,10) 모두 동일 결과 → sigmoid의 입력값(ω - τ)이 극단적으로 한쪽으로 치우쳐서 gating이 실질적으로 binary(전부 보호 또는 전부 제거)

### 부수 발견
- NSP가 Layer 11에서 79.1% 달성 (이전 71.4%보다 높음) — CLS+L2norm + Mahalanobis 개선이 반영됨
- SAPP의 gating이 overflow warning 발생 (ω값이 10만 단위) → feature normalize 전 ω 계산 필요

---

## 6. 피드백 및 다음 단계 (Feedback & Next Steps)

### 교훈
- Normal covariance의 inverse는 AD sensitivity의 good proxy가 **아님** (이 feature space에서)
- "Anomaly-sensitive direction 보호"라는 아이디어 자체는 유효하나, ω가 올바른 측정 방법이 아님
- 단순 gating보다 더 근본적인 접근 필요

### 다음 실험 후보
1. **ARHD (Anchor-Residual)**: SAPP 대신 multi-layer 비대칭 결합 시도
2. **SEAS (Shift-Equivariant Scoring)**: projection 대신 scoring distribution transport
3. **ω 측정 방법 변경**: Σ^{-1} 대신 anomaly direction과의 직접 correlation 측정 (하지만 anomaly label 없이 어떻게?)
4. **Plan B 전환 검토**: 방법론 novelty 탐색의 연속 실패 → 분석+이론 논문 pivot 고려

### _lessons.md 승격
- [x] "Normal covariance inverse는 anomaly sensitivity proxy로 부적합 (L2 normalized DINOv2 features에서)"
- [x] "SAPP gating이 NSP보다 나쁨 — hard projection이 여전히 최적"

---

## 관련 노트
- 선행: [사고 실험 교차 비교](../../analysis/2026-03-27_method_thought_experiment.md)
- 코드: `scripts/sapp_experiment.py`
