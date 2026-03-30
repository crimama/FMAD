# Spectral Anomaly-Preserving Projection (SAPP): 방법론 제안서

> **Date**: 2026-03-26
> **Status**: Proposal (미구현)
> **Keywords**: anomaly-aware disentanglement, spectral gap, variance-gated projection, robust AD

---

## 0. Executive Summary

NSP의 근본적 한계: nuisance direction을 "shift가 큰 방향"으로 정의하고 **무조건 제거**한다. 그런데 Layer 11에서 shift-anomaly PC projection correlation이 0.53이다 --- shift가 큰 방향의 절반 가까이가 anomaly detection에도 중요한 방향이다. NSP는 이 공유 방향을 무차별 제거하므로, anomaly 정보를 함께 파괴한다.

**SAPP의 핵심 통찰**: Normal data의 covariance 구조가 "어느 방향이 anomaly에 민감한가"를 알려준다. Normal data에서 분산이 극히 낮은 방향(spectral gap 아래)은 --- 정의상 --- normal variation으로는 접근하지 않는 방향이다. Anomaly는 바로 이 "normal이 가지 않는 방향"으로 feature를 밀어낸다. 따라서 **normal covariance의 spectral structure가 anomaly-sensitive subspace의 implicit prior**가 된다.

SAPP는 nuisance를 제거할 때, 각 nuisance direction이 anomaly-sensitive subspace와 얼마나 겹치는지를 측정하고, 겹침이 큰 방향은 **부분적으로만 제거하거나 보존**한다. "제거할 것과 보존할 것을 spectral structure로 구분한다"는 원리이다.

**Novelty Gate 통과 확인**:
- **기존 방법의 핵심 한계**: NSP/PCA-based nuisance removal은 shift와 anomaly가 겹치는 subspace에서 anomaly 정보를 함께 파괴
- **건드리는 메커니즘**: nuisance projection의 aggressiveness를 normal data의 spectral structure로 조절
- **Contribution 한 줄**: "Normal data의 spectral gap이 anomaly sensitivity의 sufficient statistic이며, 이를 이용해 nuisance removal 시 anomaly-preserving constraint를 부여할 수 있다"

---

## 1. 문제 정의: NSP는 왜 Layer 11에서 실패하고 Layer 8에서 성공하는가

### 1.1 경험적 관찰 요약

| Layer | PC proj. correlation (shift-anomaly) | NSP K=30 AD2 AUROC | NSP K=100 AD2 AUROC |
|-------|--------------------------------------|---------------------|----------------------|
| 8     | 0.12                                 | 75.0%               | **83.8%** (+24.0pp)  |
| 11    | **0.53**                             | 67.9%               | —                    |

Layer 8에서는 shift direction과 anomaly direction이 거의 직교(correlation 0.12) --- nuisance를 아무리 제거해도 anomaly 정보가 살아남는다. 그래서 K를 크게 올려도 monotonic improvement.

Layer 11에서는 correlation 0.53 --- nuisance direction의 상당 부분이 anomaly direction과 겹친다. K를 올리면 nuisance와 함께 anomaly 정보도 같이 제거된다. 이것이 Layer 11에서 NSP가 Layer 8보다 열등한 이유이다.

### 1.2 NSP의 수학적 한계

NSP의 projection:
```
f_clean = f - sum_{k=1}^{K} (f . n_k) n_k
```
여기서 `n_k`는 nuisance subspace의 k번째 basis vector.

이것은 binary decision이다: nuisance direction은 100% 제거, 나머지는 100% 보존. 하지만 현실에서는:
- 순수 nuisance direction (shift에만 반응) --- 100% 제거 OK
- 순수 anomaly direction (anomaly에만 반응) --- 100% 보존 OK
- **혼합 direction (shift에도 anomaly에도 반응)** --- 부분 제거가 최적

혼합 direction에 대해 NSP는 0/1 선택만 가능하다. 제거하면 anomaly 정보 손실, 보존하면 shift noise 잔존. **이것이 NSP의 근본적 한계이며, 10개 variant가 모두 실패한 이유이다** --- 모든 variant가 이 0/1 framework 안에서만 작동했기 때문.

### 1.3 왜 기존 시도가 실패했는가

Phase 3 autoresearch에서 시도된 방법들을 재분석:

| 시도 | 무엇을 바꿨는가 | 왜 실패했는가 |
|------|----------------|-------------|
| Soft projection (exp8, -11pp) | λ * projection | λ를 uniform하게 적용 --- 모든 direction을 동일하게 약하게 제거하므로, nuisance도 불완전 제거 + anomaly도 약간 손상 |
| Signal-only subspace (exp9, -25pp) | Anomaly subspace에서만 scoring | Anomaly subspace 추정이 부정확 --- anomaly label 없이 "anomaly 방향"을 정의할 수 없었음 |
| Train-only nuisance (exp11, -24pp) | Paired data 없이 nuisance 추정 | Normal train 내 variance = class identity + nuisance가 혼합 --- shift vector가 아닌 일반 variance를 제거 |
| Dual-space (exp14, -9pp) | Original + projected 앙상블 | Original에 남아있는 nuisance가 앙상블을 오염 |

**공통 실패 원인**: "어떤 방향이 anomaly에 중요한가"를 알 수 없었다. Label이 없으니 직접 정의할 수 없고, 간접 추정도 모두 실패했다.

---

## 2. 핵심 통찰: Normal Covariance의 Spectral Gap이 Anomaly Sensitivity를 Encode한다

### 2.1 이론적 근거

Normal data의 covariance matrix `Sigma_N`의 eigenvalue 분해:
```
Sigma_N = sum_i lambda_i * v_i * v_i^T
```

`lambda_i`가 큰 direction `v_i`: normal data가 **많이 퍼져있는** 방향. 이 방향의 variation은 normal variation이다 (조명, 각도, 개체 차이 등).

`lambda_i`가 매우 작은 direction `v_i`: normal data가 **거의 퍼지지 않는** 방향. 이 방향으로의 편차는 normal에서 관측된 적 없다.

**핵심 주장**: Anomaly는 이 low-variance direction으로 feature를 밀어낸다.

왜? Anomaly는 정의상 "normal에서 관측되지 않는 패턴"이다. Feature space에서 이것은 "normal distribution의 support 밖"이다. Normal distribution의 support는 high-variance direction을 따라 넓고, low-variance direction을 따라 좁다. 따라서 anomaly가 normal support를 벗어나는 방향은 주로 **low-variance direction**이다.

이것은 PatchCore, SPADE 등 distance-based AD가 작동하는 이유이기도 하다 --- normal feature 분포에서 거리가 먼 점이 anomaly이고, 이 거리는 low-variance direction에서의 편차에 의해 지배된다 (Mahalanobis distance에서 작은 eigenvalue로 나누므로).

### 2.2 Spectral Gap을 Anomaly Sensitivity Proxy로 사용

Normal covariance의 eigenvalue spectrum에서 **spectral gap**을 식별:

```
lambda_1 >= lambda_2 >= ... >= lambda_r >> lambda_{r+1} >= ... >= lambda_d
```

- `v_1, ..., v_r`: normal variation subspace (high variance)
- `v_{r+1}, ..., v_d`: anomaly-sensitive subspace (low variance)

이 분류는 anomaly label 없이, 오직 normal data의 structure만으로 도출된다.

### 2.3 왜 이것이 PISCO/SubspaceAD와 다른가

**SubspaceAD**: Normal PCA의 residual (orthogonal complement)에서 anomaly score를 계산. 하지만 nuisance direction 처리를 하지 않는다 --- shift가 있으면 residual이 오염된다.

**PISCO**: Style과 content를 post-hoc으로 분리. Domain label이 필요하고, anomaly detection이 아닌 classification용으로 설계되었다.

**NSP**: Nuisance direction을 제거하지만, 어떤 direction이 anomaly에 중요한지 고려하지 않는다.

**SAPP (본 제안)**: Nuisance 제거와 anomaly 보존을 **동시에** 고려한다. Normal covariance의 spectral structure가 anomaly sensitivity의 proxy이고, 이것이 nuisance 제거의 **soft constraint**로 작용한다.

---

## 3. 방법론: Spectral Anomaly-Preserving Projection (SAPP)

### 3.1 Setup

- Frozen DINOv2 feature extractor, layer L (default: 8)
- Multi-environment normal data: regular 조건 features `{f_i^reg}`, shifted 조건 features `{f_i^shift}`
- Paired data: `(f_i^reg, f_i^shift)` for same object i

### 3.2 Step 1: Nuisance Subspace Estimation (NSP와 동일)

Shift vectors:
```
d_i = f_i^shift - f_i^reg
```

PCA on `{d_i}` --> nuisance basis `{n_1, ..., n_K}` (top-K eigenvectors of shift covariance)

### 3.3 Step 2: Anomaly-Sensitive Subspace Estimation (새로운 부분)

Normal covariance (regular 조건만):
```
Sigma_N = Cov({f_i^reg})
```

Eigendecomposition:
```
Sigma_N = sum_j mu_j * u_j * u_j^T
```

Anomaly sensitivity score for direction `u_j`:
```
a(u_j) = 1 / (mu_j + epsilon)
```
(분산이 작을수록 anomaly에 민감)

### 3.4 Step 3: Variance-Gated Nuisance Projection (핵심 novelty)

각 nuisance direction `n_k`에 대해, 그것이 anomaly-sensitive subspace와 얼마나 겹치는지 계산:

**Anomaly overlap score of nuisance direction n_k**:
```
omega_k = sum_j a(u_j) * (n_k . u_j)^2
         = sum_j (n_k . u_j)^2 / (mu_j + epsilon)
         = n_k^T * Sigma_N^{-1} * n_k
```

이것은 nuisance direction `n_k`의 **Mahalanobis norm** (normal covariance 기준)이다.

직관:
- `omega_k`가 크다 = `n_k`가 normal에서 분산이 작은 방향(anomaly-sensitive)과 많이 겹친다 --> 이 direction을 제거하면 anomaly 정보 손실
- `omega_k`가 작다 = `n_k`가 normal에서 분산이 큰 방향(anomaly-insensitive)과 주로 정렬 --> 안전하게 제거 가능

**Gating function**:
```
g_k = sigmoid(-alpha * (omega_k - tau))
```
여기서:
- `alpha`: sharpness (크면 hard gating에 가까움)
- `tau`: threshold (anomaly overlap이 이 값보다 크면 보존)

**SAPP Projection**:
```
f_clean = f - sum_{k=1}^{K} g_k * (f . n_k) * n_k
```

`g_k`가 1에 가까우면: 이 nuisance direction을 완전히 제거 (순수 nuisance)
`g_k`가 0에 가까우면: 이 direction을 보존 (anomaly 정보와 겹침)
중간값: 부분 제거

### 3.5 왜 omega_k = n_k^T Sigma_N^{-1} n_k 인가: 기하학적 해석

`Sigma_N^{-1}`은 normal distribution의 **precision matrix**이다. `n_k^T Sigma_N^{-1} n_k`는 direction `n_k`를 따라 normal distribution이 얼마나 **concentrated** (좁게 분포)되어 있는지를 나타낸다.

- Concentrated (high precision) = 이 방향으로의 편차는 즉시 anomaly로 탐지됨 = **anomaly-sensitive**
- Spread (low precision) = 이 방향으로의 편차는 normal variation에 묻힘 = **anomaly-insensitive**

따라서 `omega_k`는 정확히 "nuisance direction n_k를 제거했을 때 anomaly detection 능력이 얼마나 감소하는가"의 proxy이다.

### 3.6 Closed-Form 해석: Regularized Inverse Projection

SAPP를 행렬 형태로 다시 쓰면:

```
P_SAPP = I - N * G * N^T
```
여기서 `N = [n_1, ..., n_K]` (nuisance basis), `G = diag(g_1, ..., g_K)` (gating)

NSP는 `G = I` (all ones).

SAPP의 `G`는 `Sigma_N`의 spectral structure에 의해 결정된다. 이것은 본질적으로 **normal precision으로 regularized된 nuisance projection**이다.

### 3.7 Alpha와 Tau의 설정

**Tau (threshold) 설정 원리**:

`omega_k`의 분포를 보면, nuisance direction들 사이에서도 anomaly overlap이 다양하다. `tau`는 이 분포의 중앙값(median) 또는 특정 분위수(quantile)로 설정:

```
tau = percentile({omega_1, ..., omega_K}, q)
```
- `q = 50`: 절반의 nuisance direction을 보존 (보수적)
- `q = 75`: 25%만 보존 (aggressive removal)
- `q = 25`: 75% 보존 (매우 보수적)

**Alpha (sharpness)**:
- `alpha = infinity`: hard gating (NSP와 동일, threshold 기반)
- `alpha = 0`: uniform scaling (모든 direction을 동일하게 부분 제거)
- `alpha = 5~10`: soft gating (gradient가 smooth)

Alpha와 tau 모두 **anomaly label 없이 설정 가능** --- normal data의 통계만으로 결정.

---

## 4. 이론적 정당화

### 4.1 Information-Theoretic Interpretation

NSP는 mutual information `I(f_clean; env)` 를 줄이려 한다 (environment에 대한 정보를 제거). 하지만 동시에 `I(f_clean; anomaly)`도 줄어들 수 있다.

SAPP의 목적함수를 implicitly 해석하면:

```
min  I(f_SAPP; env)                  -- nuisance 제거
s.t. I(f_SAPP; anomaly) >= threshold  -- anomaly 보존
```

`I(f_SAPP; anomaly)`를 직접 측정할 수 없으므로(anomaly label 없음), normal covariance의 precision을 proxy로 사용. 이것이 `omega_k` 제약의 information-theoretic 근거이다.

### 4.2 Causal Framework 연결

Structural Causal Model:
```
Environment (E) --> Nuisance (N) --> Feature (F) <-- Structure (S) <-- Anomaly (A)
```

SAPP가 하는 것:
1. `E --> N --> F` 경로를 차단 (nuisance projection)
2. `A --> S --> F` 경로를 보존 (anomaly-sensitive direction 보호)
3. Normal covariance의 spectral gap이 `S --> F` 경로의 proxy

이것은 von Kugelgen의 block identifiability와 연결된다:
- Environment가 auxiliary variable 역할
- Block-identifiable하게 nuisance를 식별
- **추가로** normal covariance가 anomaly-relevant block의 proxy를 제공

### 4.3 Identifiability Condition

**Claim**: Multi-environment normal data + normal covariance structure가 주어졌을 때, anomaly-preserving nuisance projection은 다음 조건 하에서 unique하게 결정된다:

1. Nuisance subspace와 anomaly-sensitive subspace의 principal angle이 0이 아닌 direction이 존재 (entanglement 존재)
2. Normal covariance의 spectral gap이 존재 (low-variance anomaly-sensitive direction이 identifiable)
3. Shift vector가 nuisance subspace를 span할 만큼 다양 (sufficient environment diversity)

조건 1은 Layer 11에서 correlation 0.53으로 확인.
조건 2는 PCA eigenvalue spectrum에서 확인 가능.
조건 3은 multi-environment data collection으로 보장.

---

## 5. NSP 실패 카테고리에 대한 예측

### 5.1 Can 카테고리 분석

Can은 현재 가장 어려운 카테고리 중 하나:
- Metallic reflective surface --> lighting shift에 극도로 민감
- Anomaly (dent, scratch) --> surface normal 변화를 통해 반사 패턴을 바꿈
- **문제**: lighting shift도 반사 패턴을 바꿈 --> shift direction과 anomaly direction이 강하게 겹침

NSP L8 K150에서 can: 0.485 --> 0.843 (+35.8pp). 이미 큰 개선이지만, 여전히 85% 미만.

**SAPP 예측**: Can에서는 반사 패턴 관련 direction이 nuisance이면서 동시에 anomaly-sensitive이다. SAPP의 gating이 이 방향을 부분 보존하여, shift noise는 줄이면서 dent/scratch의 반사 변화 신호는 유지할 수 있다.

예상 개선: can에서 추가 +3-8pp (0.87-0.92)

### 5.2 Fruit Jelly 카테고리 분석

Fruit jelly: 색상/투명도 변화가 anomaly의 핵심 신호. 동시에 조명 변화도 색상 feature에 영향.

L8 K150: 0.912 --> 0.973 (+6.1pp). NSP가 이미 잘 작동하는 경우.

**SAPP 예측**: 이미 NSP가 대부분 해결한 카테고리. SAPP의 추가 개선은 미미할 것 (+0-2pp). 이유: L8에서 color feature가 nuisance와 anomaly에서 이미 잘 분리되어 있기 때문 (correlation 0.12).

### 5.3 전체 예측

SAPP의 주요 개선 대상: **NSP에서 K를 올려도 saturation되거나 하락하는 카테고리** (shift-anomaly overlap이 큰 경우). 현재 data에서는:
- L11 전체 (correlation 0.53)에서 NSP 대비 개선 예상
- L8 (correlation 0.12)에서는 SAPP ≈ NSP (이미 거의 직교이므로 gating이 큰 차이를 만들지 않음)

**가장 흥미로운 예측**: SAPP는 Layer 11에서도 Layer 8 NSP에 필적하는 성능을 낼 수 있다. 왜냐하면 SAPP는 Layer 11의 높은 entanglement를 **해소**하기 때문.

이것이 검증되면: "SAPP가 layer-robust하다" = 어떤 layer를 선택하든 안정적으로 작동한다는 claim이 가능. 이는 "Layer 8이 sweet spot"이라는 NSP의 brittle한 결론보다 훨씬 강력한 contribution.

---

## 6. 구현 계획 (2주)

### Week 1: Core Implementation + Validation

**Day 1-2**: SAPP core 구현
- `phase3_nsp_experiment.py`에 SAPP projection 추가
- `omega_k` 계산 (Mahalanobis norm of nuisance directions)
- Gating function `g_k = sigmoid(-alpha * (omega_k - tau))`
- Config로 alpha, tau, K 조절 가능하게

**Day 3-4**: Layer 11에서 SAPP vs NSP 비교
- 주 가설 검증: "SAPP가 Layer 11에서 NSP를 이기는가?"
- K sweep: K=10, 20, 30, 50 at Layer 11
- Alpha sweep: 1, 5, 10, 50 at fixed K
- Tau sweep: 25th, 50th, 75th percentile

**Day 5**: Layer 8에서 SAPP = NSP 확인
- 예상: Layer 8에서는 거의 차이 없음 (이미 직교)
- 이 결과 자체가 "SAPP의 gating이 entanglement에 비례하여 작동한다"는 evidence

### Week 2: Analysis + Ablation + Paper Figures

**Day 6-7**: Ablation study
- SAPP vs NSP vs no projection at Layer 8, 11
- Hard gating (alpha=inf) vs soft gating vs no gating (NSP)
- Per-category omega_k 분포 분석: 어떤 카테고리에서 gating이 활성화되는가?

**Day 8-9**: Visualization + analysis
- Nuisance direction별 omega_k scatter plot (anomaly overlap 시각화)
- Gating 전후 feature 분포 변화 (t-SNE)
- Per-category SAPP improvement 분석

**Day 10**: K=150 at Layer 8 with SAPP (best NSP config에 SAPP 적용)
- "Even at the best NSP layer/K, SAPP provides marginal additional gain"
- 또는 "SAPP allows smaller K for same performance" (efficiency claim)

### 성공 기준

| 기준 | 최소 | 목표 | 비고 |
|------|------|------|------|
| Layer 11 SAPP vs Layer 11 NSP | +3pp | +8pp | SAPP의 핵심 contribution |
| Layer 11 SAPP vs Layer 8 NSP | 동등 (-2pp) | +2pp | Layer robustness |
| Clean performance (MVTec AD) | -1pp | 0pp | No degradation |
| 이론적 명확성 | omega_k와 성능의 상관 | omega_k 기반 자동 gating | Explainability |

---

## 7. 논문 Positioning 및 Story

### 7.1 Paper Message

> Foundation model features entangle nuisance variations with anomaly-discriminative information. Standard nuisance removal (PCA projection) destroys anomaly signals in the shared subspace. We show that the spectral structure of normal data's covariance provides an implicit map of anomaly sensitivity, enabling **anomaly-preserving nuisance projection** --- removing environmental shift while protecting the capacity to detect defects.

### 7.2 Contribution Structure

1. **Empirical finding**: Quantify shift-anomaly entanglement across DINOv2 layers (PC projection correlation 0.53 at L11, 0.12 at L8)
2. **Theoretical insight**: Normal covariance spectral gap as anomaly sensitivity proxy --- anomaly label 없이 anomaly-sensitive direction 식별
3. **Method (SAPP)**: Variance-gated nuisance projection with theoretical motivation
4. **Experimental evidence**: Layer-robust improvement, outperforms NSP at entangled layers, no clean degradation

### 7.3 왜 이것이 FDA/PCA/Domain Adaptation이 아닌가

| 기존 방법 | SAPP와의 차이 |
|-----------|--------------|
| FDA (Fourier Domain Adaptation) | Input-level style transfer. Feature-level이 아님. Anomaly 구조를 보존한다는 보장 없음 |
| PCA nuisance removal (NSP) | All-or-nothing removal. Anomaly sensitivity 무시 |
| Domain adaptation (DANN, CORAL) | Source/target domain alignment. AD의 one-class 구조를 활용하지 않음 |
| IRM (Invariant Risk Minimization) | Invariant predictor 학습. Label 필요. Post-hoc이 아님 |
| PISCO | Style/content linear decomposition. AD-specific anomaly sensitivity 개념 없음 |
| FiCo | Architecture-specific (reverse distillation). Post-hoc이 아님 |

SAPP의 고유성: **AD의 one-class structure (normal data만 존재)를 적극 활용하여, anomaly label 없이 anomaly-preserving constraint를 도출**. 이것은 classification DG도, 일반 domain adaptation도 하지 않는 AD-specific contribution이다.

### 7.4 잠재적 Figure 구성

| Figure | 내용 | Message |
|--------|------|---------|
| Fig 1 | Nuisance-anomaly entanglement 시각화 (PC projection overlap) | "The problem: shift and anomaly share directions" |
| Fig 2 | omega_k scatter: nuisance directions의 anomaly overlap score 분포 | "Some nuisance directions overlap with anomaly sensitivity" |
| Fig 3 | SAPP method overview (gating diagram) | "Variance-gated projection preserves anomaly information" |
| Fig 4 | Layer 11 SAPP vs NSP performance curve | "SAPP resolves the entanglement bottleneck" |
| Fig 5 | Per-category gating activation 분석 | "Gating automatically adapts to category-specific entanglement" |

---

## 8. Risk Analysis

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Layer 8에서 이미 correlation 0.12이므로 SAPP의 추가 개선이 미미 | 높음 | 중간 | Layer 11에서의 개선이 main story. Layer 8에서 SAPP=NSP는 "method correctly identifies no entanglement" |
| omega_k가 실제 anomaly sensitivity와 상관 없음 | 중간 | 높음 | Synthetic validation: 알려진 anomaly direction에 대해 omega_k와 실제 중요도 비교 |
| Sigma_N의 estimation이 불안정 (few-shot 상황) | 중간 | 중간 | Regularization (shrinkage estimator), minimum sample 요구사항 명시 |
| Clean performance 하락 | 낮음 | 중간 | Clean 조건에서는 nuisance direction에 signal이 없으므로 gating이 효과 없음 (SAPP = baseline) |
| 이론적 정당화가 약함 | 중간 | 중간 | Empirical validation이 이론보다 강력하면 "analysis + lightweight method" 구조로 |

---

## 9. Beyond SAPP: 확장 방향

### 9.1 Patch-Level SAPP
현재 image-level (CLS+patch mean). Patch-level로 확장하면 pixel-level AD에 적용 가능. 각 spatial position에서의 normal covariance가 다르므로, **spatially varying gating** 가능.

### 9.2 Adaptive K Selection
omega_k가 tau를 넘는 direction의 수가 자동으로 "effective K"를 결정. K를 hyperparameter에서 해방시키는 방법.

### 9.3 Multi-Layer SAPP
Layer별 entanglement가 다르므로 (L8: 0.12, L11: 0.53), 각 layer에 맞는 gating을 적용하고 layer를 결합. NSP에서 실패한 multi-layer fusion이 SAPP에서는 성공할 수 있다 --- 각 layer의 entanglement를 개별적으로 처리하기 때문.

### 9.4 Theoretical Paper로의 Pivot
만약 SAPP의 실험적 개선이 modest하더라도:
- "Normal covariance spectral structure = anomaly sensitivity proxy" 라는 이론적 결과 자체가 contribution
- 이 framework 위에서 NSP, SubspaceAD, Mahalanobis distance를 모두 통합적으로 해석 가능
- "Anomaly detection as constrained nuisance invariance" 라는 새로운 이론적 관점 제시

---

## 관련 노트

- `../fm_ad_robustness/2026-03-23_mechanism_analysis.md` -- 실패 메커니즘 분석
- `../fm_ad_robustness/2026-03-23_theoretical_foundations.md` -- 이론적 기반
- `../../experiments/2026-03-26_phase3_linear_disentanglement/report.md` -- Phase 3 NSP 실험 결과
- `../feature_disentanglement/2026-03-23_deep_survey.md` -- Disentanglement 서베이
