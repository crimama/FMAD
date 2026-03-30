# Shift-Equivariant Anomaly Scoring (SEAS) — 방법론 제안

> 작성: 2026-03-26
> 목적: NSP(Nuisance Subspace Projection)의 "nuisance 제거" 패러다임을 넘어서는 새로운 방법론 설계
> 상태: 제안 단계 (미검증)

---

## 0. 왜 새로운 패러다임이 필요한가

### NSP의 성공과 한계

NSP는 +29.4pp라는 놀라운 개선을 달성했지만, 본질적으로 **정보 파괴** 방법이다:

```
f_clean = f - sum_k (f . n_k) n_k     (hard projection)
```

이 접근의 구조적 한계:
1. **Nuisance subspace에 anomaly 정보가 있으면 함께 제거** -- K가 커질수록 이 위험 증가
2. **Paired multi-environment data가 필수** -- train-only nuisance estimation(exp11)은 -24pp로 실패
3. **Binary 결정**: 각 direction은 100% 제거 or 100% 보존 -- soft projection(exp8)도 실패했지만, 이는 soft의 방식이 잘못된 것이지 개념 자체의 문제는 아닐 수 있음
4. **Static projection**: test 샘플의 실제 shift magnitude와 무관하게 동일한 projection 적용

### 10개 복잡한 변형이 모두 실패한 이유 재해석

Phase 3에서 10개 변형(condition-union, whitened PCA, soft projection, signal-only, train-only, filtered, PCA-whitened Mahalanobis, dual-space, per-category adaptive K)이 모두 실패했다. 기존 해석은 "linear optimality → 단순함이 최적"이었지만, **다른 해석**이 가능하다:

> 10개 변형은 모두 **"nuisance 제거"라는 동일 패러다임 안에서의 변형**이었다. 실패한 것은 복잡성이 아니라, 패러다임 자체의 local optimum에 갇힌 것일 수 있다.

이를 뒷받침하는 증거:
- K=100에서도 saturation 미도달 → nuisance subspace가 100차원 이상이라면, 정보 파괴량이 상당
- Per-category adaptive K(-17pp)의 실패 → nuisance 구조가 category마다 다를 수 있는데 uniform K는 이를 무시
- Dual-space ensemble(-9pp)의 실패 → 제거 전/후 정보를 단순 결합하면 shift noise가 다시 유입

**새 패러다임의 가능성**: nuisance를 **제거**하지 않고, scoring function이 nuisance에 **equivariant**하게 만들면?

---

## 1. 핵심 아이디어: Distribution Transport for Anomaly Scoring

### 1.1 직관

현재 문제의 본질:

```
Train: P_N(z)          -- normal distribution in feature space
Test:  P_N'(z) = T(P_N(z))  -- shifted normal distribution
       P_A'(z)               -- shifted anomaly distribution

NSP approach:  project out T → P_N(z) ≈ P_N'(z) in projected space
Our approach:  estimate T → transport P_N to match test conditions → score against T(P_N)
```

NSP는 shift를 **무시**하려 하고, SEAS는 shift를 **이해하고 따라간다**.

비유: 온도계가 기압에 따라 오차가 생긴다고 하자.
- NSP = 기압 감응 부품을 아예 제거 → 기압 변화에 무관해지지만, 기압과 상관된 온도 정보도 잃음
- SEAS = 기압을 측정하여 보정 공식을 적용 → 기압 정보를 활용하면서 정확도 유지

### 1.2 수학적 정의

**Shift transformation model**:
```
z_shifted = z_regular + delta(z, e)
```
여기서 e는 environment condition (overexposed, underexposed, geometric shift 등),
delta는 shift field -- feature space의 각 위치에서 shift의 방향과 크기를 정의.

**가정**: delta가 locally linear하다면 (Phase 2에서 PCA가 잘 작동한 사실이 이를 지지):
```
delta(z, e) = A_e * z + b_e
```
A_e는 environment-dependent linear transform, b_e는 translation.

**Anomaly score under transport**:

기존 Mahalanobis:
```
s(z) = (z - mu)^T Sigma^{-1} (z - mu)
```

Transport-corrected Mahalanobis:
```
s_SEAS(z_test) = (z_test - T_e(mu))^T T_e(Sigma)^{-1} (z_test - T_e(mu))
```
여기서 T_e는 environment e에 의한 normal distribution의 transport:
```
T_e(mu) = mu + A_e * mu + b_e = (I + A_e) * mu + b_e
T_e(Sigma) = (I + A_e) * Sigma * (I + A_e)^T
```

### 1.3 NSP와의 이론적 관계

**정리 (비공식)**: NSP는 SEAS의 special case이다.

증명 스케치:
- NSP는 nuisance direction N = [n_1, ..., n_K]을 제거: `f_clean = (I - N N^T) f`
- 이는 모든 test sample에 동일한 transform 적용 → **shift magnitude를 무시**
- SEAS에서 A_e = 0 (no linear transform), b_e = shift mean이면, T_e(mu) = mu + b_e
- NSP의 projection은 이 b_e의 방향을 제거하는 것과 동치
- **그러나** SEAS는 A_e ≠ 0을 허용 → shift가 feature의 위치에 따라 다를 수 있음

이 관계가 중요한 이유: NSP가 잘 작동하는 이유를 설명하면서, NSP가 놓치는 것(non-uniform shift)을 정확히 지적할 수 있다.

---

## 2. 방법론: SEAS (Shift-Equivariant Anomaly Scoring)

### 2.1 Overview

```
Phase 1 (Offline): Multi-environment normal data로 shift model 학습
Phase 2 (Test-time): Test batch에서 environment 추정 → normal distribution transport → scoring

Training Pipeline:
  {regular images} → DINOv2 Layer 8 → features Z_reg
  {shifted images}  → DINOv2 Layer 8 → features Z_shift
  Paired (Z_reg, Z_shift) → Estimate shift model (A_e, b_e)
  Z_reg → Fit normal model (mu, Sigma)

Test Pipeline:
  {test images} → DINOv2 Layer 8 → features Z_test
  Z_test → Estimate test environment e_hat
  Transport: (mu', Sigma') = T_{e_hat}(mu, Sigma)
  Score: s = Mahalanobis(Z_test; mu', Sigma')
```

### 2.2 Shift Model Estimation (핵심 모듈 1)

**입력**: Paired features (z_reg^i, z_shift^i) for i = 1, ..., N

**방법 A: Global Affine Model** (가장 단순)
```
Minimize ||z_shift - (A * z_reg + b)||^2
```
Closed-form solution:
```
[A | b] = Z_shift * [Z_reg; 1]^T * ([Z_reg; 1] * [Z_reg; 1]^T)^{-1}
```

**방법 B: Subspace-Restricted Affine Model** (NSP 지식 활용)
```
A = sum_k alpha_k * n_k * n_k^T     (shift는 nuisance direction에서만 발생)
```
n_k는 NSP의 nuisance direction을 재활용. alpha_k와 b만 추정.

이 제약의 장점:
- 파라미터 수가 d^2 → K로 대폭 감소 (d=768, K=100이면 590K → 100)
- NSP의 분석 결과를 자연스럽게 통합
- Overfitting 위험 감소

**방법 C: Per-condition Model** (environment label 활용)
```
MVTec AD 2는 7개 condition: overexposed, underexposed, shift_1~5
각 condition별로 별도의 (A_e, b_e) 추정
```

### 2.3 Test-time Environment Estimation (핵심 모듈 2)

Test 시 environment label이 없을 때, test batch에서 environment를 추정해야 한다.

**방법 A: Batch Statistics Matching**
```
e_hat = argmin_e ||mean(Z_test) - T_e(mu)||^2
```
Test batch의 mean을 각 known environment의 transported mean과 비교.

**방법 B: Mixture Model**
```
P(Z_test) = sum_e pi_e * N(T_e(mu), T_e(Sigma))
```
EM으로 pi_e (environment mixture weight) 추정. 부분적으로 다른 environment가 섞여 있어도 처리 가능.

**방법 C: Continuous Interpolation**
```
e_hat = sum_e w_e * e     where w_e = softmax(-||mean(Z_test) - T_e(mu)||^2 / tau)
```
Discrete environment 사이를 연속적으로 interpolation. 학습에 없던 novel shift에도 대응 가능.

**방법 D: Projection-free (가장 단순)**
```
delta_test = mean(Z_test) - mu_train
T_test(mu) = mu + delta_test
T_test(Sigma) = Sigma + epsilon * I     (covariance는 shift에 덜 민감하다고 가정)
```
Environment model 자체가 불필요. Test batch mean drift만으로 transport.

> 주의: 방법 D는 사실상 Team B (SANA)의 아이디어. 하지만 SEAS framework 안에서 이것이 왜 작동하는지 (A_e ≈ 0, first-order approximation), 그리고 언제 실패하는지 (A_e가 significant할 때)를 설명할 수 있다.

### 2.4 Transport-Corrected Scoring (핵심 모듈 3)

Normal distribution parameters를 transport한 후 scoring:

**Mahalanobis scoring (기본)**:
```
mu' = T_e(mu) = (I + A_e) mu + b_e
Sigma' = T_e(Sigma) = (I + A_e) Sigma (I + A_e)^T + noise_term

s(z) = (z - mu')^T (Sigma')^{-1} (z - mu')
```

**핵심 차이 vs NSP**:
- NSP: feature를 변환 → scoring은 기존과 동일
- SEAS: feature는 그대로 → scoring distribution을 변환
- 정보 보존 관점에서 SEAS가 우월: feature에서 아무것도 제거하지 않음

**Adaptive regularization**:
```
Sigma'_reg = Sigma' + lambda(e) * I
```
lambda(e)는 shift magnitude에 비례 -- shift가 클수록 uncertainty가 크므로 더 강한 regularization.

### 2.5 Patch-level Extension

Image-level이 아닌 patch token별로 transport 적용:

```
For each spatial position (h, w):
  z_{h,w}^test → score against T_e(mu_{h,w}, Sigma_{h,w})
```

Patch-level normal model은 spatial position별로 유지. Transport는 global (모든 patch에 동일한 A_e, b_e 적용) 또는 spatially-varying.

---

## 3. 이론적 근거: 왜 Transport가 Projection보다 나을 수 있는가

### 3.1 Information-theoretic Argument

**NSP의 information loss**:

NSP가 K개의 nuisance direction을 제거하면, 남는 feature의 차원은 d-K.

```
I(f_clean; anomaly) = I(f; anomaly) - I(f_nuisance; anomaly | f_clean)
```

마지막 항이 0이 아니면 (nuisance subspace에 anomaly 정보가 있으면) 정보 손실 발생.

Phase 3에서 K=100까지도 clean 성능(-0.2pp)이 거의 유지된 것은 이 정보 손실이 작다는 뜻이지만, **0이라는 보장은 없다**. 특히:
- 어떤 anomaly type은 shift direction과 align될 수 있음 (e.g., 반사면의 scratch와 조명 변화)
- Per-category를 보면 특정 category에서 더 큰 손실이 있을 수 있음

**SEAS의 정보 보존**:

SEAS는 feature를 변형하지 않으므로:
```
I(f; anomaly) = I(f; anomaly)     (trivially preserved)
```

대신 scoring function이 shift를 보정:
```
s_SEAS(z) = Mahalanobis(z; T_e(mu, Sigma))
```

문제는 T_e 추정의 정확도에 의존한다는 것. T_e가 부정확하면 오히려 NSP보다 나쁠 수 있다.

### 3.2 Equivariance vs Invariance

**Invariance** (NSP): `score(z) = score(z + shift)` -- shift를 무시
**Equivariance** (SEAS): `score(z + shift) = score(z; shifted_model)` -- shift에 따라 model이 변환

Equivariance가 더 강력한 이유:
- Invariance는 shift 정보를 완전히 버림 → shift와 correlate된 모든 정보 소실
- Equivariance는 shift 정보를 **활용** → shift를 이해하고 보정

이는 vision에서의 invariance vs equivariance 논쟁과 동일한 구조:
- Pooling (invariance) vs Capsule Networks / Equivariant Networks (equivariance)
- Translation invariance (global average pooling) vs translation equivariance (conv features)

### 3.3 Generalization to Unseen Shifts

NSP의 한계: 학습된 nuisance direction과 다른 방향의 novel shift에 대응 불가.

SEAS의 가능성: 방법 B (subspace-restricted) + 방법 C (continuous interpolation)를 조합하면, known shift들의 interpolation/extrapolation으로 novel shift에 대응 가능.

```
Novel shift e_new ≈ sum_k w_k * e_k     (convex combination of known shifts)
T_{e_new} ≈ sum_k w_k * T_{e_k}
```

이는 NSP에서는 원리적으로 불가능한 기능이다 (projection은 방향만 정의하므로 magnitude 정보가 없음).

---

## 4. 예상 장점 및 NSP 대비 우위 조건

### 4.1 SEAS가 NSP를 이기는 경우 (예측)

| 조건 | 이유 |
|------|------|
| **Anomaly signal이 nuisance direction에 일부 포함** | NSP는 함께 제거, SEAS는 보존 |
| **Shift magnitude가 샘플마다 다를 때** | NSP는 고정 projection, SEAS는 adaptive transport |
| **Novel (unseen) shift 조건** | NSP는 학습된 direction만 제거, SEAS는 interpolation 가능 |
| **Category별 shift 구조가 다를 때** | SEAS의 per-condition model이 자연스럽게 처리 |
| **Anomaly localization (pixel-level)** | Patch-level transport가 더 fine-grained |

### 4.2 NSP가 SEAS를 이기는 경우 (예측)

| 조건 | 이유 |
|------|------|
| **Shift model 추정이 부정확** | Transport 오류가 scoring 오류로 직결 |
| **Test batch가 작을 때** | Environment 추정의 통계적 불안정 |
| **Shift가 순수 additive일 때** | A_e ≈ 0이면 NSP가 이미 optimal |
| **Nuisance와 anomaly가 완전 orthogonal** | NSP의 정보 손실 = 0, SEAS의 추가 복잡성만 남음 |

### 4.3 검증 가능한 예측

1. **K가 클수록 SEAS의 상대적 이점이 커진다**: K가 크면 NSP의 정보 손실 누적, SEAS는 무관
2. **Can, Wallplugs 같은 난이도 높은 category에서 SEAS 이점이 크다**: 반사면에서 shift-anomaly entanglement이 강함
3. **방법 D (mean-shift only)도 NSP의 80%+ 성능**: first-order approximation이 충분
4. **방법 B (subspace-restricted)가 방법 A (full affine)보다 우수**: parameter 절약 + overfitting 방지

---

## 5. 구현 계획 (2주)

### Week 1: Core Implementation + Validation

**Day 1-2: Shift Model Estimation**
```python
# 방법 A: Global Affine
def estimate_affine_shift(z_reg, z_shift):
    """Least squares: z_shift = A * z_reg + b"""
    # z_reg: (N, d), z_shift: (N, d)
    # Augment z_reg with ones for bias
    Z = np.hstack([z_reg, np.ones((N, 1))])  # (N, d+1)
    # Solve: z_shift = Z @ W, where W = [A^T; b^T]
    W, _, _, _ = np.linalg.lstsq(Z, z_shift, rcond=None)
    A = W[:d].T  # (d, d)
    b = W[d]     # (d,)
    return A, b

# 방법 B: Subspace-Restricted
def estimate_restricted_shift(z_reg, z_shift, nuisance_dirs):
    """A = sum_k alpha_k * n_k @ n_k^T, solve for alpha_k and b"""
    K = nuisance_dirs.shape[0]
    # Project shift residual onto each nuisance direction pair
    delta = z_shift - z_reg  # (N, d)
    # For each nuisance dir n_k: alpha_k = mean(<delta, n_k> / <z_reg, n_k>)
    # This is a simplified estimator; proper one uses least squares
    ...
```

**Day 3-4: Transport-Corrected Scoring**
```python
def seas_score(z_test, mu_train, sigma_train, A_e, b_e, reg_lambda):
    """Transport normal distribution and score"""
    transform = np.eye(d) + A_e
    mu_transported = transform @ mu_train + b_e
    sigma_transported = transform @ sigma_train @ transform.T + reg_lambda * np.eye(d)

    # Mahalanobis distance
    diff = z_test - mu_transported
    sigma_inv = np.linalg.inv(sigma_transported)
    scores = np.sum(diff @ sigma_inv * diff, axis=1)
    return scores
```

**Day 5: Environment Estimation**
```python
def estimate_environment(z_test_batch, mu_train, transport_models):
    """Match test batch to closest known environment"""
    test_mean = z_test_batch.mean(axis=0)
    distances = {}
    for env_name, (A_e, b_e) in transport_models.items():
        mu_transported = (np.eye(d) + A_e) @ mu_train + b_e
        distances[env_name] = np.linalg.norm(test_mean - mu_transported)

    # Soft assignment
    dists = np.array(list(distances.values()))
    weights = softmax(-dists / tau)

    # Interpolated transport
    A_hat = sum(w * A for w, (A, _) in zip(weights, transport_models.values()))
    b_hat = sum(w * b for w, (_, b) in zip(weights, transport_models.values()))
    return A_hat, b_hat
```

### Week 2: Experiments + Analysis

**Day 6-8: Main Experiments**

| 실험 | 내용 | 성공 기준 |
|------|------|----------|
| EXP-S1 | SEAS (방법 A + 방법 A env) vs NSP K=100 | AD2 >= 83.8% |
| EXP-S2 | SEAS (방법 B + 방법 C env) vs NSP K=100 | AD2 >= 83.8% |
| EXP-S3 | SEAS (방법 D, mean-only) vs NSP K=100 | AD2 >= 67% (80% of NSP) |
| EXP-S4 | Per-category breakdown | Can/Wallplugs에서 SEAS > NSP |
| EXP-S5 | K sweep에서 SEAS vs NSP crossover | 큰 K에서 SEAS 이점 확인 |

**Day 9-10: Ablation + Analysis**

| 분석 | 내용 |
|------|------|
| ABL-1 | Transport components: A만 / b만 / A+b |
| ABL-2 | Environment estimation 방법 비교 (A/B/C/D) |
| ABL-3 | Affine model residual 분석: ||z_shift - (Az_reg + b)||의 크기 |
| ABL-4 | NSP+SEAS 결합: NSP로 일부 제거 + SEAS로 나머지 보정 |
| ANA-1 | Information preservation: NSP vs SEAS에서 anomaly-relevant information 정량 비교 |

**Day 11-14: Paper Material**
- Figure: "정보 보존 관점에서의 NSP vs SEAS" 도식
- Table: Main results (NSP vs SEAS, per-category, per-condition)
- Figure: K sweep에서의 crossover point
- Analysis: "SEAS가 NSP를 이기는/지는 조건" empirical verification

---

## 6. Go/No-Go 기준

### Quick Experiment (Day 1-2): Affine Model Fitness

Phase 3의 paired data로 affine model을 fit하고 residual을 측정:

```
R^2 = 1 - ||z_shift - (A z_reg + b)||^2 / ||z_shift - z_reg||^2
```

| R^2 | 판정 | 이유 |
|-----|------|------|
| > 0.9 | **STRONG GO** | Shift가 affine으로 잘 모델링됨 → transport 정확 |
| 0.7 - 0.9 | **GO with 주의** | Affine이 대부분 설명하지만 nonlinear residual 존재 |
| 0.5 - 0.7 | **MODIFY** | Subspace-restricted (방법 B) or mean-only (방법 D)로 축소 |
| < 0.5 | **NO-GO** | Affine model 부적합 → 다른 패러다임 필요 |

### Quick Experiment 2: Information Loss 측정

NSP K=100 적용 후 제거된 100차원에서의 anomaly score를 별도 측정:

```
z_removed = sum_k (z . n_k) n_k
s_removed = Mahalanobis(z_removed; mu_removed, Sigma_removed)
```

`s_removed`가 normal과 anomaly를 유의미하게 구분하면 → NSP의 정보 손실이 있다 → SEAS의 동기가 강화된다.

---

## 7. Novelty 평가

### 7.1 기존 방법과의 차별성

| 기존 방법 | 핵심 전략 | SEAS와의 차이 |
|-----------|----------|--------------|
| NSP (ours Phase 3) | Nuisance direction 제거 | SEAS는 제거 대신 transport |
| AdaBN (Li et al., 2018) | BN statistics 적응 | Feature-level이 아닌 model parameter 적응 |
| TENT (Wang et al., 2021) | Entropy minimization TTA | Unsupervised, anomaly label 없음과 충돌 |
| PILOT (BMVC 2025) | Prompt TTA | Feature space가 아닌 prompt space |
| RoDA (2025) | Sinkhorn distance DA | Distribution alignment, not transport for scoring |
| FiCo (AAAI 2025) | Filter or compensate | Architecture-specific, implicit |

**SEAS의 unique contribution**:
> "Scoring function을 test distribution에 맞게 transport한다"는 개념은 AD에서 제안된 적이 없다. TTA는 model을 적응시키고, DA는 feature를 적응시키지만, SEAS는 **decision boundary (normal model)를 적응**시킨다.

### 7.2 Novelty Gate 체크

- **기존 방법의 핵심 한계**: Nuisance removal은 정보 파괴, TTA/DA는 AD의 one-class 특성과 충돌
- **SEAS가 건드리는 메커니즘**: Scoring function의 shift equivariance — feature가 아닌 normal model의 transport
- **논문 contribution 한 줄**: "We show that transporting the normal distribution to match test conditions preserves anomaly-discriminative information that nuisance removal necessarily destroys."

### 7.3 ICLR Positioning

```
논문 구조 (SEAS가 성공할 경우):

Section 1: Problem — FM-AD fails under distribution shift
Section 2: Analysis — Why NSP works, and its information-theoretic limit
  - PCA near-optimality (Phase 3 결과)
  - But: information loss in nuisance subspace (QE 측정)
Section 3: Method — SEAS
  - Shift model estimation
  - Distribution transport
  - Environment-adaptive scoring
  - Theoretical analysis: NSP as special case of SEAS
Section 4: Experiments
  - SEAS vs NSP: main results, per-category, per-condition
  - Ablation: transport components, environment estimation
  - Analysis: when SEAS wins, when NSP suffices
Section 5: Discussion
  - Information preservation vs estimation accuracy tradeoff
  - Practical guidelines: when to use NSP vs SEAS
```

---

## 8. 위험 요소 및 대응

| 위험 | 확률 | 대응 |
|------|------|------|
| Affine model이 shift를 잘 설명 못함 | 30% | 방법 D (mean-only)로 축소, 분석 논문 전환 |
| SEAS < NSP | 40% | "왜 transport가 projection보다 나쁜가" 분석 → 분석 논문의 core insight |
| Environment estimation 실패 | 25% | Oracle environment로 upper bound 측정 → 문제 분리 |
| SEAS ≈ NSP (유의차 없음) | 30% | "NSP ≈ first-order SEAS" 증명 → 이론적 contribution |

**핵심**: SEAS가 실패하더라도 **"왜 NSP가 optimal인가"의 deeper understanding**이라는 분석 contribution이 남는다. 이는 "4팀 사고 실험"의 결론 (분석 > 방법론)과도 일치하지만, 방법론이 성공하면 논문 impact가 훨씬 크다.

---

## 9. NSP+SEAS Hybrid: 안전한 확장

완전히 새로운 패러다임이 아닌, NSP를 기반으로 SEAS를 추가하는 hybrid:

```
Step 1: NSP로 K_small (e.g., 30) direction만 제거 (conservative)
Step 2: 남은 feature에서 SEAS transport scoring 적용

Rationale:
- K=30은 Phase 3에서 이미 +13.5pp (conservative but safe)
- 남은 feature에서의 residual shift를 SEAS가 보정
- K=100 NSP가 aggressive하게 제거하는 70개 direction의 정보를 SEAS가 보존
```

이 hybrid는 "NSP의 안전한 영역 + SEAS의 추가 이점"을 결합한다.

예상: K=30 NSP + SEAS >= K=100 NSP (83.8%), 동시에 clean 성능 유지

---

## 관련 노트
- 선행: [Phase 3 결과](../../experiments/2026-03-26_phase3_linear_disentanglement/report.md)
- 배경: [4팀 사고 실험](../2026-03-27_method_thought_experiment.md)
- 이론: [theoretical_foundations](2026-03-23_theoretical_foundations.md)
- 비교: [solution_survey](2026-03-23_solution_survey.md)
