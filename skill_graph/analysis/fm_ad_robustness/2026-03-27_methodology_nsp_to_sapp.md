# 현재 방법론 상세: NSP → SAPP → Test-time Covariance Shift — 2026-03-27

> **상태**: 방향 전환 중 (NSP=oracle, test-time method가 새 핵심)
> **keywords**: NSP, SAPP, test-time, covariance shift, unpaired, nuisance subspace, generality

---

## 0. 방향 전환 기록 (2026-03-27 오후)

**문제 발견**: NSP/SAPP 모두 MVTec AD 2의 paired multi-condition 구조에 의존.
paired data가 있는 AD 벤치마크는 MVTec AD 2뿐 → 1 데이터셋에 locked.
"FM-AD의 distribution shift 문제 해결"이라는 claim과 evidence scope 불일치.

**결정**: Test-time covariance shift decomposition으로 전환.
- NSP → oracle upper bound (paired data 있을 때의 최선)
- Test-time method → 실제 contribution (paired data 없이, 어디서든 작동)
- 상세 분석: [비교군 및 일반성 분석](2026-03-27_comparison_and_generality_analysis.md)

## 1. 방법론 진화 개요

```
Phase 3 (NSP)                  Phase 4-old (SAPP)              Phase 4-new (Test-time)
─────────────                  ──────────────────              ────────────────────────
PCA + Hard Projection          PCA + Variance-Gated            Covariance Shift Decomposition
paired data 필수               paired data 필수 (동일 문제)     paired data 불필요
MVTec AD 2 전용                MVTec AD 2 전용                 어디서든 적용 가능
+29.4pp on AD2                 SAPP: Layer 11 개선 예상         NSP의 80%+ 목표
→ oracle upper bound로 전환    → 보류 (일반성 문제 미해결)       → 새 핵심 contribution
```

---

## 2. NSP (Nuisance Subspace Projection) — 현재 최선

### 2.1 알고리즘

```
Input:
  - Paired multi-environment normal data: {(f_i^reg, f_i^shift)}
  - DINOv2 ViT-B/14, Layer 8 features (CLS + patch mean, L2 normalized)
  - K = 100 (nuisance dimensions)

Step 1: Shift Vector Estimation
  d_i = f_i^shift - f_i^reg    (300 pairs from MVTec AD 2)

Step 2: Nuisance Subspace via PCA
  PCA({d_i}) → {n_1, ..., n_K}  (top-K eigenvectors of Cov(d))

Step 3: Hard Orthogonal Projection
  f_clean = f - N @ N^T @ f    (where N = [n_1, ..., n_K])

Step 4: Mahalanobis Anomaly Scoring
  score(x) = sqrt((x - μ_train)^T @ Σ_train^{-1} @ (x - μ_train))
```

### 2.2 핵심 성과

| Config | MVTec AD 2 | MVTec AD | Delta (AD2) |
|--------|-----------|----------|-------------|
| Baseline (kNN, L11, no projection) | 54.4% | 91.6% | — |
| NSP K=30, L11 | 67.9% | 91.9% | +13.5pp |
| NSP K=30, L8, Mahalanobis | 75.0% | 96.4% | +20.6pp |
| **NSP K=100, L8, Mahalanobis** | **83.8%** | **96.3%** | **+29.4pp** |

### 2.3 컴포넌트별 기여 (Ablation)

| Component | 독립 기여 | 누적 |
|-----------|----------|------|
| NSP 도입 (K=30, L11) | +13.5pp | 67.9% |
| CLS + L2 norm | +0.1pp | 68.0% |
| Mahalanobis scoring | +3.4pp | 71.4% |
| Layer 8 선택 | +3.6pp | 75.0% |
| K 증가 (30→50→100) | +8.8pp | 83.8% |

### 2.4 NSP의 한계

1. **Layer 의존성**: Layer 8에서만 optimal. Layer 11 (entanglement 0.53)에서는 hard projection이 anomaly 정보도 제거
2. **0/1 Decision**: 모든 nuisance direction을 동일하게 100% 제거 — entangled direction에서 anomaly 손실 불가피
3. **K saturation 미확인**: K=100까지 단조 증가, 최적점 불명
4. **Paired data 필요**: Regular↔shifted 쌍 데이터 요구 (실용적 제약)

### 2.5 실패한 10개 변형과 교훈

| 변형 | AD2 결과 | 실패 원인 | 교훈 |
|------|---------|----------|------|
| Multi-layer L8+L11 fusion | 72.5% (-2.5pp) | L11의 noise가 L8을 오염 | Layer 선택 > fusion |
| Condition-별 union PCA | 77.1% (-6.7pp) | 표준 PCA가 더 안정적 | 복잡한 estimation ≠ 좋은 estimation |
| Whitened PCA | 83.8% (동등) | Whitening이 orthogonal projection과 수학적 등가 | 불필요한 복잡성 |
| Soft projection (λ uniform) | 72.5% (-11.3pp) | 모든 direction을 균일하게 약하게 제거 → 둘 다 불완전 | Uniform soft는 hard보다 나쁨 |
| Signal-only subspace (RANP) | 59.0% (-24.8pp) | Anomaly 방향을 label 없이 사전 지정 불가 | Anomaly subspace 직접 추정 실패 |
| Train-only nuisance | 53.7% (-30.1pp) | Normal variance ≠ shift variance | Paired data 필수적 |
| Hybrid NSP + filtering | — | NSP의 효과를 상쇄 | 후처리 가공은 효과 미미 |
| Per-category adaptive K | — | 카테고리 간 불안정 | Global K가 더 안정적 |
| Dual-space ensemble | 74.6% (-9.2pp) | Original의 nuisance가 앙상블 오염 | Projection 후 원본 혼합은 해로움 |
| PCA-whitened Mahalanobis | — | 이중 정규화로 정보 왜곡 | 단순한 Mahalanobis가 최선 |

**핵심 교훈**: 10개 변형이 모두 실패한 것은 **이 문제의 최적해가 linear hard projection** 임을 강하게 시사. 복잡성을 더하면 오히려 성능이 하락한다.

---

## 3. SAPP (Spectral Anomaly-Preserving Projection) — 차세대 방법론

### 3.1 NSP의 근본적 한계에서 출발

NSP는 nuisance direction을 무조건 100% 제거한다. 하지만:
- **순수 nuisance direction** → 100% 제거 OK
- **순수 anomaly direction** → 100% 보존 OK
- **혼합 direction** (shift + anomaly 겹침) → **부분 제거가 최적**, 하지만 NSP는 0/1 선택만 가능

Layer 11에서 shift-anomaly PC correlation = 0.53 → nuisance direction의 ~절반이 anomaly 정보와 겹침. 이것이 NSP가 Layer 11에서 Layer 8보다 열등한 이유.

**SAPP의 질문**: "어떤 nuisance direction이 anomaly에 중요한가?"를 anomaly label 없이 알 수 있는가?

### 3.2 핵심 통찰: Normal Covariance의 Spectral Gap

Normal data의 covariance Σ_N의 eigenvalue spectrum:
```
λ_1 ≥ λ_2 ≥ ... ≥ λ_r >> λ_{r+1} ≥ ... ≥ λ_d
                    ↑ spectral gap
```

- **고분산 방향** (λ_i 큼): normal variation (조명, 각도, 개체 차이)
- **저분산 방향** (λ_i 작음): normal이 거의 퍼지지 않는 방향

**핵심 주장**: Anomaly는 저분산 방향으로 feature를 밀어낸다.

왜? Anomaly = "normal에서 관측되지 않는 패턴" = normal distribution의 support 밖. Support는 고분산 방향으로 넓고 저분산 방향으로 좁다. 따라서 anomaly가 support를 벗어나는 방향은 주로 저분산 방향이다.

이것이 Mahalanobis distance가 AD에서 작동하는 이유이기도 하다 — 작은 eigenvalue로 나누므로 저분산 방향의 편차가 증폭된다.

### 3.3 알고리즘

```
Input: NSP의 모든 입력 + Normal train features

Step 1-2: NSP와 동일 (shift vector PCA → nuisance directions)

Step 3 (NEW): Anomaly Sensitivity Scoring
  각 nuisance direction n_k에 대해:
  ω_k = n_k^T @ Σ_train^{-1} @ n_k    (Mahalanobis norm)

  해석:
    - ω_k 크다 = n_k가 저분산 방향(anomaly-sensitive)과 겹침 → 보호
    - ω_k 작다 = n_k가 고분산 방향(anomaly-insensitive)과 정렬 → 제거 OK

Step 4 (NEW): Variance-Gated Projection
  g_k = σ(-α × (ω_k - τ))    (sigmoid gating)

  f_clean = f - Σ_k g_k × (f · n_k) × n_k

  g_k ≈ 1: 이 nuisance direction 완전 제거 (순수 nuisance)
  g_k ≈ 0: 이 direction 보존 (anomaly 정보와 겹침)
  중간값: 부분 제거

Step 5: Mahalanobis scoring (NSP와 동일)
```

### 3.4 행렬 형태

```
P_SAPP = I - N × G × N^T

where:
  N = [n_1, ..., n_K]          (nuisance basis)
  G = diag(g_1, ..., g_K)      (gating weights)

NSP는 G = I의 special case.
SAPP는 Σ_train의 spectral structure로 G를 결정.
→ "Normal precision으로 regularized된 nuisance projection"
```

### 3.5 Hyperparameters

| Param | 역할 | 설정 원리 | Default |
|-------|------|----------|---------|
| K | Nuisance dimensions | Shift variance explained ratio | 100 |
| α | Gating sharpness | α→∞: hard gating (threshold), α→0: uniform | 5.0 |
| τ | Gating threshold | median(ω) 또는 percentile | median(ω) |

**핵심**: 모든 hyperparameter가 anomaly label 없이, normal data의 통계만으로 설정 가능.

### 3.6 이론적 연결

**Information-Theoretic**:
```
min  I(f_SAPP; env)                    — nuisance 제거
s.t. I(f_SAPP; anomaly) ≥ threshold   — anomaly 보존
```
I(f_SAPP; anomaly)를 직접 측정할 수 없으므로, normal covariance의 precision을 proxy로 사용. ω_k가 이 proxy.

**Causal Framework**:
```
Env → Nuisance → Feature ← Structure ← Anomaly
```
SAPP = Env→Feature 경로 차단 + Anomaly→Feature 경로 보존.

**Nonlinear ICA (von Kügelgen NeurIPS 2021)**:
- Environment label = auxiliary variable → block identifiability
- Normal covariance의 spectral gap = anomaly-relevant block의 추가 proxy
- SAPP는 identifiability 조건을 만족하면서도 anomaly-preserving constraint 추가

### 3.7 기존 방법과의 차별성

| 방법 | SAPP와의 차이 |
|------|--------------|
| NSP | All-or-nothing. Anomaly sensitivity 무시 |
| FDA (Fourier Domain Adaptation) | Input-level. Feature-level이 아님 |
| PISCO (ICML 2023) | Style/content 분리. AD-specific 아님 |
| SubspaceAD | Residual scoring. Nuisance 처리 없음 |
| Domain Adaptation (DANN, CORAL) | Source/target alignment. One-class 구조 미활용 |
| FiCo | Architecture-specific. Post-hoc 아님 |

**SAPP의 고유성**: AD의 one-class structure (normal data만 존재)를 활용하여, anomaly label 없이 anomaly-preserving constraint를 도출. Classification DG도, 일반 DA도 하지 않는 AD-specific contribution.

---

## 4. GO/NO-GO 기준 및 실험 설계

### 4.1 SAPP 검증 기준

| 지표 | GO 기준 | NO-GO |
|------|---------|-------|
| ω spread ratio (max/min) | > 5x | < 5x → gating 무의미 |
| Layer 11 SAPP vs NSP | +3pp 이상 | < +1pp |
| Layer 8 SAPP vs NSP | 동등 (±1pp) | -3pp 이상 하락 |
| Clean performance | -1pp 이내 | -3pp 이상 |

### 4.2 실험 매트릭스

```
Layer: {8, 11}
Method: {Baseline, NSP, SAPP-α1, SAPP-α3, SAPP-α5, SAPP-α10}
Dataset: {MVTec AD 2, MVTec AD}
Metrics: I-AUROC (per-category + mean)
```

### 4.3 예상 결과

| 시나리오 | Layer 11 NSP | Layer 11 SAPP | Layer 8 NSP | Layer 8 SAPP |
|---------|-------------|--------------|-------------|-------------|
| 낙관적 | ~71% | 74-79% | ~84% | ~84% |
| 비관적 | ~71% | 72-73% | ~84% | 83-84% |

---

## 5. 방향 결정 프레임워크 (2026-03-27 updated)

### 현재 위치

```
Phase 1: Baseline Reproduction     ✅ 완료
Phase 2: Mechanism Analysis        ✅ 완료
Phase 3: NSP Method Design         ✅ 완료 (83.8% AD2) → oracle upper bound
Phase 4: Generality + Test-time    🟡 진행 중
  ├─ 비교군/일반성 분석 완료
  ├─ NSP의 MVTec AD 2 의존성 문제 식별
  ├─ Test-time covariance shift method 설계
  └─ 검증 실험 대기
Phase 5: Paper Writing             ⬜ 미착수
```

### 방향 전환: SAPP → Test-time Covariance Shift

**전환 이유**: SAPP도 NSP와 동일하게 paired data에 의존 → MVTec AD 2 전용이라는 근본 문제 미해결.
Test-time method는 paired data 불필요 → RobustAD 등 어디서든 적용 가능 → 논문 contribution으로 더 강함.

**새 방향 분기**:
```
Test-time method 검증 (MVTec AD 2에서)
  ├─ NSP의 80%+ 달성 (test-time cov shift ≥ 67%)
  │   └─ 논문 확정: Analysis + Test-time Method
  │       NSP = oracle, test-time = practical contribution
  │
  ├─ 60-80% (test-time cov shift 50-67%)
  │   └─ 추가 개선 탐색: robust estimation, iterative refinement 등
  │
  └─ 60% 미만 (test-time cov shift < 50%)
      └─ Pivot: scope 축소 (paired setting 한정) 또는 분석 논문
```

### 최우선 실험 (즉시)

| ID | 실험 | 목적 | 판단 기준 |
|----|------|------|----------|
| TT1 | Test-time mean shift (1-dim) | 가장 단순한 test-time, lower bound | baseline보다 나으면 방향 유효 |
| TT2 | Test-time covariance shift (K-dim) | 핵심 실험 | NSP의 80%+ 이면 GO |
| TT3 | Random projection K=100 | "방향이 의미있는가" 검증 | TT2 >> TT3 이어야 함 |
| TT4 | AdaBN baseline | 가장 직접적 경쟁자 | TT2 > TT4 이어야 차별성 |

### 이전 Quick Experiments (보류/재우선순위)

| ID | 실험 | 상태 | 비고 |
|----|------|------|------|
| ~~QE1~~ | Intrinsic Dimension | 보류 | 분석 보강에는 유용하나 test-time 방향과 직접 관련 없음 |
| ~~QE2~~ | NSP 잔차 MMD | 보류 | NSP가 oracle로 전환되어 우선순위 하락 |
| ~~QE3~~ | FDA vs PCA | 보류 | 동일 |
| ~~SAPP~~ | ω 분포 검증 | **보류** | paired data 의존 문제 미해결 → test-time 우선 |

---

## 6. 연구적 의미와 포지셔닝

### Claim
> FM-AD의 robustness 실패는 feature space에서 nuisance와 anomaly 정보의 linear entanglement에 기인하며, normal data의 spectral structure가 이 entanglement을 해소하는 sufficient information을 제공한다.

### Evidence
- Phase 1: 3 SOTA × 3 benchmark에서 25-34pp 하락 실증
- Phase 2: Layer별 비단조 shift 프로파일 + entanglement 정량화 (correlation 0.53)
- Phase 3: Post-hoc linear projection으로 +29.4pp 회복 (0 params)
- Phase 3-neg: 10개 nonlinear/complex 변형 모두 실패 → linear optimality

### Boundary
- Layer 8에서 이미 자연적으로 분리된 경우 NSP만으로 충분 (SAPP 불필요)
- Can/Wallplugs: 반사면 + 미세결함에서 전 방법 실패 (<10%) → 현재 기술 한계
- Paired multi-environment data 필요 (SANA로 완화 가능하나 성능 하락 예상)

### Positioning
- **기존 연구 0건**: FM-AD에서 feature disentanglement을 명시적으로 다룬 논문 없음
- **ICLR 적합**: Analysis(60%) + Lightweight Fix(40%) 구조
- **타이틀 후보**: "Are Foundation Model Features All You Need for Anomaly Detection?"

---

## 관련 노트

- [Phase 1 실험](../../experiments/2026-03-26_phase1_baseline_reproduction/report.md)
- [Phase 2 실험](../../experiments/2026-03-26_phase2_feature_shift_analysis/report.md)
- [Phase 3 실험](../../experiments/2026-03-26_phase3_linear_disentanglement/report.md)
- [SAPP GO/NO-GO](../../experiments/2026-03-27_sapp_go_nogo/report.md)
- [7가지 실패 메커니즘](2026-03-23_mechanism_analysis.md)
- [이론적 기반](2026-03-23_theoretical_foundations.md)
- [4팀 사고 실험](../2026-03-27_method_thought_experiment.md)
- [SAPP 원본 제안서](../../ideas/2026-03-26_ASAD_proposal.md)
- [검증된 패턴](_lessons.md)
