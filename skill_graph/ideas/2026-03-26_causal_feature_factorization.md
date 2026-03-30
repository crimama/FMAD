# Causal Feature Factorization for Robust Anomaly Detection (CaFF-AD)

**Date**: 2026-03-26
**Status**: Proposal (detailed)
**Keywords**: causal inference, feature disentanglement, foundation model, anomaly detection, robustness, structural equation model, counterfactual

---

## Executive Summary

PCA 기반 nuisance removal (Phase 3 best: +29.4pp)은 **variance 기준**으로 shift direction을 제거한다. 이 방법은 놀라울 정도로 효과적이지만, 근본적 한계가 있다: PCA는 **통계적 상관(correlation)**만 포착하고, **인과 구조(causal structure)**를 무시한다. 이 제안은 FM feature의 생성 과정을 Structural Causal Model (SCM)로 모델링하고, **do-calculus에 기반한 interventional scoring**을 통해 PCA가 본질적으로 해결할 수 없는 문제를 해결하는 방법론을 제시한다.

---

## 1. PCA가 해결할 수 없는 것: 정밀한 문제 정의

### 1.1 현재 상태의 성공과 한계

Phase 3 결과 (NSP K=100, Layer 8, Mahalanobis):
- MVTec AD 2 I-AUROC: 54.4% -> 83.8% (+29.4pp)
- MVTec AD I-AUROC: 96.5% -> 96.3% (-0.2pp)

PCA 기반 NSP가 작동하는 이유: shift vector들의 PCA가 nuisance direction을 포착하고, orthogonal projection이 이를 제거한다. 그러나 Phase 3 실험에서 **10가지 변형이 모두 실패**했다:
- Soft projection, signal-only subspace, whitened PCA, dual-space scoring, per-category adaptive K...
- 결론: "standard PCA + hard orthogonal projection"이 local optimum이며, **PCA 패러다임 내에서의 개선 여지가 고갈됨**

### 1.2 PCA의 근본적 한계: 왜 더 이상 갈 수 없는가

**한계 1: Variance =/= Causality**
PCA는 maximum variance direction을 찾는다. 그러나 high-variance direction이 반드시 environment에 의해 "caused"된 것이 아니다. 예시:
- Object identity가 high variance를 만들 수 있다 (다른 object들이 섞여 있을 때)
- Anomaly 자체가 high variance를 만들 수 있다 (다양한 defect type)
- K를 키우면 이런 non-nuisance variance도 함께 제거될 위험

**한계 2: Entanglement 하에서의 정보 손실**
Phase 2에서 PC4가 shift(0.215)와 anomaly(0.707) 모두에 반응하는 "shared PC"임을 확인했다. PCA orthogonal projection은:
- 이 PC를 제거하면: anomaly 정보 0.707만큼 손실
- 이 PC를 유지하면: shift 정보 0.215만큼 오염

**Binary decision** (제거 or 유지)밖에 없다. Entangled PC에서 shift 기여분만 선택적으로 제거하는 것이 불가능하다.

**한계 3: Confounding 무시**
PCA는 E(environment)와 S(anomaly state)가 독립이라고 암묵적으로 가정한다. 그러나 현실에서:
- 특정 조명 조건에서 특정 defect가 더 잘 보임 (E -> visibility of S)
- 특정 환경이 특정 defect를 유발 (E -> S, e.g., 습도 -> 부식)
- 이런 confounding이 있으면 PCA의 "shift variance = nuisance" 가정이 무너짐

**한계 4: Paired data 필수**
NSP는 regular-shifted pair에서 shift vector를 계산한다. 이는:
- Paired data가 없는 환경에서 적용 불가
- Pair 수에 따른 nuisance estimation quality 변동
- 새로운 shift type에 대한 generalization 보장 없음

### 1.3 핵심 질문

> PCA가 "어떤 방향이 변하는가?"를 답한다면,
> 우리가 답해야 하는 질문은 "**왜** 그 방향이 변하는가, 그리고 그 변화의 **원인**이 environment인가 anomaly인가?"이다.

이 질문에 답하려면 **인과 추론(causal reasoning)**이 필요하다.

---

## 2. Causal Model: FM Feature의 생성 과정

### 2.1 Structural Causal Model (SCM)

FM feature `z`의 생성을 다음 SCM으로 모델링한다:

```
      O (Object Identity)
      |
      v
E --> Z <-- S
(Env)       (Anomaly State)
```

**Structural Equations:**
```
O := f_O(U_O)                    # Object identity (exogenous)
E := f_E(U_E)                    # Environment condition (exogenous)
S := f_S(U_S)                    # Anomaly state: normal or {scratch, dent, stain, ...}
Z := g(O, E, S) + epsilon        # FM feature = nonlinear mixing of three causal parents
```

여기서:
- `O`: object identity (category, instance). Screw인지 bottle인지, 같은 screw 중 어떤 instance인지.
- `E`: environment factor. Lighting intensity, direction, camera viewpoint, background.
- `S`: anomaly state. Normal / 구체적 defect type (scratch, missing part, contamination).
- `Z`: DINOv2 patch feature (R^768 for ViT-B/14).
- `g`: FM encoder가 구현하는 nonlinear mixing function.
- `epsilon`: noise (measurement noise + feature extraction stochasticity).

**핵심 가정**: O, E, S는 서로 independent exogenous variables에 의해 생성됨.
- O _||_ E: 어떤 object이든 어떤 환경에 놓일 수 있음
- O _||_ S: 어떤 object이든 어떤 defect가 생길 수 있음
- E _||_ S: environment는 anomaly state에 직접 영향을 주지 않음 (이 가정의 위반은 Section 4에서 다룸)

### 2.2 AD의 목표를 Causal Language로 재정의

**AD scoring function이 추정해야 하는 것:**

```
Score(x) = P(S = anomaly | Z = z, do(E = e_ref))
```

이것은 **interventional probability**이다. "환경을 reference 조건으로 **개입(intervene)**했을 때, 이 feature가 anomaly로부터 생성되었을 확률"을 묻는다.

**왜 interventional인가?** Observational probability `P(S=anomaly | Z=z, E=e)`는:
- 특정 환경 `e`에서의 조건부 확률만 알려줌
- 다른 환경에서는 score가 달라짐 (score의 non-invariance)
- 이것이 현재 FM-AD의 문제: score가 environment-dependent

Interventional probability `P(S=anomaly | Z=z, do(E=e_ref))`는:
- Environment를 **고정**한 반사실적 세계에서의 확률
- Environment가 무엇이든 동일한 scoring → **environment-invariant AD**

### 2.3 PCA vs Causal Approach의 차이

| 측면 | PCA (NSP) | Causal (CaFF) |
|------|-----------|---------------|
| 질문 | "어떤 방향이 환경에 따라 변하는가?" | "환경이 feature에 미치는 **인과적 효과**는 무엇인가?" |
| 분리 기준 | Variance | Causal mechanism |
| Entangled PC 처리 | Binary (제거 or 유지) | 해당 PC에서 E의 causal effect만 선택적 제거 |
| Confounding 처리 | 무시 | Explicit modeling |
| 보장 | 없음 (heuristic) | do-calculus에 의한 identifiability |
| Paired data | 필수 | 활용하되, unpaired 확장 가능 |

---

## 3. 제안 방법론: CaFF-AD (Causal Feature Factorization for AD)

### 3.1 Overview

세 단계로 구성:
1. **Causal Effect Estimation**: Multi-environment data로 E -> Z의 causal effect 추정
2. **Counterfactual Feature Construction**: 각 test feature를 "environment가 reference였다면"의 counterfactual feature로 변환
3. **Invariant Anomaly Scoring**: Counterfactual feature 공간에서 anomaly scoring

### 3.2 Stage 1: Environment Causal Effect Estimation

#### 3.2.1 Average Causal Effect (ACE) of Environment

Multi-environment paired data가 있을 때 (same object O=o, different environments E=e1, e2):

```
ACE(e1 -> e2 | O=o) = E[Z | O=o, do(E=e2)] - E[Z | O=o, do(E=e1)]
```

O와 E가 independent이므로, paired data에서:
```
ACE(e1 -> e2 | O=o) = z(o, e2) - z(o, e1)
```

이것은 Phase 3의 "shift vector"와 동일하다. **그러나 여기서 멈추지 않는다.**

#### 3.2.2 Conditional Average Causal Effect (CACE)

ACE는 환경 변화의 **평균** 효과만 포착한다. 그러나 환경 변화의 효과는 **object identity에 따라 다를 수 있다**:
- 금속 표면의 object: lighting에 매우 민감 (specular reflection)
- 직물 표면의 object: lighting에 덜 민감 (diffuse reflection)

```
CACE(e1 -> e2 | O) = E[Z | O, do(E=e2)] - E[Z | O, do(E=e1)]
```

이를 **object-conditional nuisance function** `h(O, E)`로 모델링:

```
Z = mu(O) + h(O, E) + delta(S) + epsilon
```

여기서:
- `mu(O)`: object identity에 의한 mean feature (object가 정상이고 reference 환경일 때의 feature)
- `h(O, E)`: object O에 대한 environment E의 causal effect. **이것이 nuisance.**
- `delta(S)`: anomaly state에 의한 feature deviation. **이것이 signal.**
- `epsilon`: residual noise.

**PCA와의 차이**: PCA는 `h(O, E)`와 `delta(S)`를 구분하지 못하고, 단순히 "큰 variance 방향 = nuisance"로 처리한다. CaFF는 `h(O, E)`를 **명시적으로 추정**하여 제거한다.

#### 3.2.3 `h(O, E)` 추정 방법

Multi-environment paired data `{(z_i^{e1}, z_i^{e2})}_{i=1}^N`에서:

**Option A: Direct Difference (Linear)**
```
h_hat(o_i, e2) - h_hat(o_i, e1) = z_i^{e2} - z_i^{e1}
```
Reference environment를 e1으로 설정하면:
```
h_hat(o_i, e) = z_i^{e} - z_i^{e_ref}
```
이것은 NSP의 shift vector와 동치. 그러나 이를 **object identity에 조건화**할 수 있다.

**Option B: Partially Linear Model (Double ML 영감)**

Feature를 두 부분으로 모델링:
```
Z = theta * E_repr + m(O_repr) + delta(S) + epsilon
```
여기서:
- `theta`: E의 linear causal effect (추정 대상)
- `E_repr`: environment representation (e.g., global image statistics, or environment label one-hot)
- `m(O_repr)`: object identity에 의한 confounding (nuisance function)
- `O_repr`: object representation (e.g., category label, or instance-level feature)

**Double ML procedure:**
1. **Stage 1a**: `O_repr`로 `Z`를 예측하는 모델 `m_hat` 학습 → residual `Z_res = Z - m_hat(O_repr)`
2. **Stage 1b**: `O_repr`로 `E_repr`를 예측하는 모델 `l_hat` 학습 → residual `E_res = E_repr - l_hat(O_repr)`
3. **Stage 2**: `theta_hat = (E_res' * E_res)^{-1} * E_res' * Z_res`

이 `theta_hat`이 **debiased estimate of E's causal effect on Z**, controlling for O.

**장점**: Object identity가 environment 인식에 미치는 confounding을 제거. PCA는 이 confounding을 무시.

**Option C: Representation-based Causal Effect (가장 실용적)**

FM feature 자체를 활용하여 implicit하게 O, E를 표현:

```
z = phi(x)     # DINOv2 feature
z_global = CLS token or global average pooling  # image-level representation
```

Global feature `z_global`에서 environment effect를 추출:
1. Multi-environment pairs에서 `Delta_z_global = z_global^{shifted} - z_global^{regular}`
2. `Delta_z_global`의 PCA → environment effect의 principal directions in global feature space
3. 이 direction들을 local (patch-level) feature에 project하여 nuisance 제거

**핵심 차이**: Global-level에서 추정한 environment effect를 local-level에 transfer.
- Global feature는 object identity를 strongly encode → environment effect를 더 cleanly 분리 가능
- Local feature에서 직접 추정하면 spatial variation이 noise로 작용

### 3.3 Stage 2: Counterfactual Feature Construction

#### 3.3.1 Counterfactual Question

> "이 이미지가 reference environment에서 촬영되었다면, feature가 어떻게 보였을까?"

```
Z_cf = Z_observed - h_hat(O, E_observed) + h_hat(O, E_ref)
```

Reference environment를 "regular" (standard lighting/viewpoint)로 설정하면:
```
Z_cf = Z_observed - h_hat(O, E_observed)
```
(h_hat(O, E_ref) = 0 by definition)

이것은 **Structural Counterfactual**이다 (Pearl's Level 3):
- Level 1 (Association): P(Z | E) — "이 환경에서 feature는 어떤가?" → 현재 FM-AD
- Level 2 (Intervention): P(Z | do(E)) — "환경을 바꾸면 feature가 어떻게 변하나?" → PCIR (NeurIPS 2023)
- Level 3 (Counterfactual): "이 **특정** 이미지가 다른 환경이었다면?" → **CaFF (제안)**

#### 3.3.2 왜 Counterfactual이 Intervention보다 강한가

Interventional approach (PCIR):
```
P(Z | do(E=e_ref)) = SUM_O P(Z | O, E=e_ref) * P(O)
```
이것은 **population-level** intervention — 모든 object에 대한 평균.

Counterfactual approach (CaFF):
```
Z_cf(o_i) = Z(o_i, E_observed) - h(o_i, E_observed)
```
이것은 **instance-level** — 이 특정 object o_i에 대한 환경 효과를 제거.

**차이가 중요한 이유**: 금속 object와 직물 object는 같은 lighting에 대해 매우 다르게 반응한다. Population-level intervention은 이 차이를 평균화하여 소실시킨다. Instance-level counterfactual은 각 object의 고유한 환경 반응을 정확히 보정한다.

#### 3.3.3 Practical Counterfactual Construction

**Method A: Linear Counterfactual (baseline)**
```
z_cf = z - Proj_{V_nuisance}(z)
```
여기서 `V_nuisance`는 shift vector PCA subspace. 이것은 Phase 3의 NSP와 동치.

**Method B: Object-Conditioned Counterfactual**
```
z_cf = z - h_hat(category(x), E_observed)
```
여기서 `h_hat(c, e) = mean_{i: cat(i)=c}[z_i^e - z_i^{ref}]` per-category mean shift.

**Method C: Instance-Level Counterfactual (핵심 제안)**
```
z_cf = z - h_hat_instance(z_global, E_repr)
```
여기서 `h_hat_instance`는 global feature `z_global`과 environment representation `E_repr`로부터 instance-specific environment effect를 예측하는 learned function.

구현:
1. Training pairs `{(z_i^{ref}, z_i^{shifted})}`: `delta_i = z_i^{shifted} - z_i^{ref}`
2. `z_global_i^{shifted}`에서 `delta_i`를 예측하는 lightweight regressor `h_hat` 학습
3. Test time: `z_cf = z - h_hat(z_global)`

이 regressor는:
- Input: CLS token or global-pooled feature (768-dim)
- Output: predicted nuisance shift delta (768-dim per patch, or shared across patches)
- Architecture: 1-2 layer MLP (< 1M parameters)
- Training: MSE loss on paired data, ~100 pairs 충분

### 3.4 Stage 3: Invariant Anomaly Scoring

Counterfactual feature `z_cf`에 대해 standard AD scoring 적용:

```
score(x) = d_Mahalanobis(z_cf(x), mu_normal, Sigma_normal)
```

여기서 `mu_normal`, `Sigma_normal`은 **reference environment**의 normal data에서 추정.

**핵심**: Scoring이 이제 single-environment 기준으로 수행됨. Multi-environment 변동은 Stage 2에서 이미 제거되었으므로, threshold가 stable.

---

## 4. Confounding 처리: E _||_ S 가정이 위반될 때

### 4.1 E -> S Confounding

현실에서 environment가 anomaly 관찰에 영향을 줄 수 있다:
- 밝은 조명에서 scratch가 더 잘 보임 → `E -> visibility(S) -> Z`
- 이것은 S의 Z에 대한 causal effect가 E에 의해 modulated되는 것

수정된 SCM:
```
Z = mu(O) + h(O, E) + delta(S, E) + epsilon
```

`delta(S, E)`: anomaly의 feature deviation이 environment에 의존.

이 경우 counterfactual:
```
z_cf = z - h_hat(O, E)
     = mu(O) + delta(S, E_observed) + epsilon
```

`delta(S, E_observed)`가 남는다. 이것은 "이 환경에서 보이는 anomaly signal"이므로 **제거하면 안 된다**. Environment가 anomaly visibility를 modulate하는 것은 자연스러운 현상이며, 이를 보존해야 한다.

**PCA와의 차이**: PCA는 E와 S의 shared variance를 무차별적으로 제거한다. CaFF는 h(O, E) (pure environment effect)만 제거하고, delta(S, E) (environment-modulated anomaly signal)는 보존한다.

### 4.2 Instrumental Variable Perspective

E를 S에 대한 **instrument**로 볼 수 있다:
- E -> Z (E가 Z에 영향)
- E _||_ S (E와 S 독립, 또는 E -> S 경로가 없음)
- E는 Z를 통해서만 S에 대한 정보를 전달

IV approach:
```
delta_hat(S) = Z - E[Z | E]     # residual after removing E's linear prediction
```

이것은 "E로 설명되지 않는 Z의 변동 = anomaly signal + object identity + noise"

**Object identity 처리**: `mu(O)`를 제거해야 한다.
```
delta_hat(S) = Z - E[Z | E] - (E[Z | O] - E[Z])
             = Z - E[Z | E] - mu(O) + E[Z]
```

Practical simplification: reference environment에서의 normal memory bank가 `mu(O)` 역할.

---

## 5. PCA가 "절대" 할 수 없고 CaFF만 할 수 있는 것

### 5.1 Entangled PC의 선택적 Deconfounding

Phase 2 결과에서 PC4: shift projection 0.215, anomaly projection 0.707.

**PCA**: PC4를 제거하면 anomaly 정보 0.707 손실. 유지하면 shift 오염 0.215 잔류.

**CaFF**: PC4에서 E의 causal contribution (0.215에 해당)만 제거하고, S의 causal contribution (0.707에 해당)은 보존.

```
z_cf[PC4] = z[PC4] - <z[PC4]의 E에 의한 부분>
```

이것이 가능한 이유: CaFF는 "왜 PC4가 이 값을 가지는가"를 decompose할 수 있기 때문이다. PCA는 "어떤 방향이 변하는가"만 알고, "왜 변하는가"를 모른다.

### 5.2 Instance-Adaptive Nuisance Removal

금속 can: lighting이 specular reflection을 유발 → 큰 nuisance shift
직물 fabric: lighting이 diffuse → 작은 nuisance shift

**PCA (K=100)**: 모든 object에 동일한 100개 direction 제거. Can에는 부족하고 fabric에는 과도할 수 있다.

**CaFF**: `h_hat(z_global, E)` 가 instance-specific nuisance를 예측. Can에는 큰 correction, fabric에는 작은 correction. 자동으로 적응.

### 5.3 Unpaired Environment 확장

PCA-NSP는 strictly paired data (same object under different conditions) 필요.

**CaFF의 unpaired 확장**: `h_hat(z_global, E_repr)`가 학습되면, E_repr만 있으면 prediction 가능. E_repr을 image statistics (brightness, contrast, color histogram)로 정의하면:
- Paired data로 `h_hat` 학습
- Test time에는 어떤 environment에서든 image statistics만 계산하여 correction 가능
- **새로운 unseen environment에도 generalization 가능** (학습된 environment effect function의 interpolation/extrapolation)

### 5.4 요약: PCA vs CaFF Capability Matrix

| Capability | PCA-NSP | CaFF-AD |
|-----------|---------|---------|
| Remove average shift | O | O |
| Object-specific shift removal | X | O |
| Selective deconfounding of entangled PCs | X | O |
| Handle E-S confounding | X | O (preserves delta(S,E)) |
| Instance-adaptive correction | X | O |
| Unpaired environment generalization | X | O (with learned h_hat) |
| Theoretical guarantee | None | Identifiability under SCM |

---

## 6. 이론적 보장: Identifiability Analysis

### 6.1 CaFF의 Identifiability

**Theorem (informal)**: 다음 조건 하에서 `h(O, E)`는 identifiable하다:
1. O, E, S가 mutually independent exogenous variables에 의해 생성
2. Training data에서 S = normal (one-class setting)
3. 최소 2개의 distinct environment values가 존재 (e_ref, e_shifted)
4. g(O, E, S) = mu(O) + h(O, E) + delta(S) + epsilon (additive decomposition)

**Proof sketch**:
- Normal training data에서 S = normal, delta(normal) = 0으로 설정.
- 그러면 `Z = mu(O) + h(O, E) + epsilon`
- Paired data (same O, different E): `Z^{e2} - Z^{e1} = h(O, e2) - h(O, e1) + epsilon'`
- E_ref를 기준으로: `h(O, E) = E_{O}[Z^E - Z^{ref} | O]`
- O가 주어지면 h(O, E)는 sample mean으로 consistently 추정 가능.

**한계**: Additive decomposition 가정. 실제로는 `g`가 nonlinear mixing이므로, additive는 local approximation. 그러나 DINOv2 feature space에서 이것이 reasonable한 근사인지는 실험적으로 검증해야 함.

### 6.2 von Kugelgen과의 연결

von Kugelgen (NeurIPS 2021)의 identifiability theorem:
- SSL feature는 "augmentation에 invariant한 것" (content)과 "augmentation에 따라 변하는 것" (style)을 block-identifiable하게 분리
- DINOv2는 crop, color jitter, blur에 invariant하도록 학습됨
- 따라서 DINOv2 feature에는 content (spatial structure, semantic) / style (appearance, texture) block이 존재

CaFF는 이 구조를 **AD에 맞게 재해석**:
- von Kugelgen의 "style" ~= CaFF의 "h(O, E)" (environment에 의해 변하는 부분)
- von Kugelgen의 "content" ~= CaFF의 "mu(O) + delta(S)" (invariant + anomaly)
- 그러나 von Kugelgen은 AD를 고려하지 않았고, delta(S)와 mu(O)의 분리는 다루지 않음
- CaFF는 **one-class constraint** (training에서 S=normal)를 활용하여 delta(S)를 implicitly 정의

### 6.3 PCIR (NeurIPS 2023)과의 차이

PCIR: invariant representation을 학습하여 AD robustness 향상. MMD regularizer로 multi-environment representation을 정렬.

차이점:
- PCIR은 **representation level**에서 invariance를 강제 (모든 environment에서 같은 representation)
- CaFF는 **scoring level**에서 counterfactual correction (representation은 유지하되, scoring 전에 environment effect 제거)
- PCIR은 training 필요 (regularizer), CaFF는 training-free 또는 minimal training
- PCIR의 invariance는 anomaly information도 함께 제거할 위험 (over-invariance). CaFF는 h(O,E)만 선택적으로 제거.
- **가장 큰 차이**: PCIR은 population-level invariance (Level 2), CaFF는 instance-level counterfactual (Level 3)

---

## 7. 구현 계획 (2주)

### Week 1: Core Method + Validation

**Day 1-2: Infrastructure**
- [ ] Phase 3 코드 기반으로 CaFF pipeline 구축
- [ ] SCM-based feature decomposition module
- [ ] Counterfactual feature constructor

**Day 3-4: Method A (Linear Counterfactual, = NSP baseline)**
- [ ] Phase 3 best config 재현 확인
- [ ] Per-category shift estimation 구현
- [ ] Global CLS-conditioned shift estimation 구현

**Day 5-7: Method C (Instance-Level Counterfactual)**
- [ ] MLP-based `h_hat(z_global, E_repr)` 학습
- [ ] E_repr = {brightness, contrast, color histogram} 또는 {environment label one-hot}
- [ ] Counterfactual feature로 Mahalanobis scoring
- [ ] Phase 3 best (NSP K=100)과 비교

### Week 2: Analysis + Ablation + Paper Figures

**Day 8-9: Entangled PC 분석**
- [ ] PC4 등 shared PC에서 CaFF vs PCA의 selective deconfounding 효과 시각화
- [ ] Per-PC shift contribution vs anomaly contribution decomposition

**Day 10-11: Ablation Study**
- [ ] Linear vs Instance-level counterfactual
- [ ] With vs without object-conditioning
- [ ] K (nuisance dim) sensitivity under CaFF vs PCA
- [ ] Unpaired evaluation: paired로 학습, unpaired로 test

**Day 12-14: Paper Figures & Writeup**
- [ ] Figure 1: SCM diagram + PCA vs CaFF comparison
- [ ] Figure 2: Entangled PC decomposition visualization
- [ ] Figure 3: Instance-adaptive correction (can vs fabric)
- [ ] Table 1: Main results (CaFF vs NSP vs baseline)

---

## 8. 예상 결과 및 성공 기준

### 8.1 정량적 목표

| Method | AD2 I-AUROC | AD1 I-AUROC | 비고 |
|--------|------------|------------|------|
| Baseline (no projection) | 54.4% | 96.5% | Phase 3 baseline |
| NSP K=100 (PCA) | 83.8% | 96.3% | Phase 3 best |
| CaFF-Linear (= NSP + category-conditioning) | 85-87% | 96.3% | +1-3pp over NSP |
| **CaFF-Instance (full method)** | **87-90%** | **96.3-96.5%** | **+3-6pp over NSP** |

### 8.2 정성적 목표

- [ ] Entangled PC에서 selective deconfounding이 실제로 작동하는 evidence
- [ ] Instance-adaptive correction이 category별로 다른 크기의 correction을 적용하는 visualization
- [ ] Clean 성능 하락 없이 shifted 성능 개선

### 8.3 Failure Modes 및 대응

**Failure 1**: Instance-level counterfactual이 NSP와 큰 차이 없음
- 원인: DINOv2 feature에서 h(O,E)의 O-dependency가 약함 (모든 object가 비슷하게 shift)
- 대응: "Linearity가 surprisingly 잘 성립함" → Occam's razor 스토리로 전환
- 논문 기여: causal framework 제시 + "linear가 optimal인 이유"의 이론적 설명

**Failure 2**: Counterfactual correction이 anomaly signal도 함께 제거
- 원인: h_hat이 delta(S)까지 학습 (anomaly가 environment effect로 오인됨)
- 대응: Training data에서 anomaly 없으므로 (one-class), h_hat은 normal variation만 학습. 검증: anomaly를 synthetic하게 추가하여 h_hat이 이를 무시하는지 확인.

**Failure 3**: MLP `h_hat`이 overfit
- 원인: Paired data가 적음 (category당 ~30 pairs)
- 대응: Strong regularization (weight decay, dropout), 또는 per-category linear model로 fallback

---

## 9. Novelty Gate 확인

### 기존 방법의 핵심 한계는 무엇인가?
PCA-based nuisance removal은 variance 기준으로만 작동하여, entangled feature에서 environment effect만 선택적으로 제거할 수 없다. 모든 object에 동일한 projection을 적용하여 instance-specific 환경 반응을 무시한다.

### 제안 방법이 건드리는 메커니즘은 무엇인가?
FM feature 생성의 인과 구조를 명시적으로 모델링하고, counterfactual reasoning으로 environment의 causal effect만 선택적으로 제거한다. 이를 통해 entangled dimension에서도 anomaly signal을 보존하면서 nuisance를 제거할 수 있다.

### 이 차이가 논문 contribution 한 줄로 요약 가능한가?
> "We propose CaFF-AD, a causal feature factorization framework that models the generative process of FM features as a structural causal model and performs counterfactual correction to remove environment effects while preserving anomaly signals, achieving robust AD where PCA-based methods fundamentally cannot."

---

## 10. Positioning

### ICLR Fit

1. **"통념 도전"**: "PCA/linear projection으로 충분하다"는 Phase 3의 결론을 "PCA는 local optimum이며, causal reasoning이 이를 넘어선다"로 발전
2. **분석 + 해결**: SCM 분석 (60%) + CaFF method (40%)
3. **이론적 깊이**: Identifiability, counterfactual, do-calculus
4. **실험적 simplicity**: 방법론 자체는 lightweight (MLP 또는 per-category linear model)
5. **Surprising finding 가능성**: "instance-level correction이 global correction보다 XX pp 좋다" 또는 "linear가 실은 optimal이며 그 이유는 causal structure가 additive이기 때문"

### 기존 연구 대비 차별성

| 연구 | Level | 방법 | 한계 |
|------|-------|------|------|
| PCIR (NeurIPS 2023) | 2 (Intervention) | MMD regularizer | Population-level, 학습 필요 |
| FiCo (AAAI 2025) | 1 (Association) | Filter/Compensate | Implicit disentanglement |
| NSP (ours Phase 3) | 1 (Association) | PCA projection | Variance-based, no causal reasoning |
| **CaFF-AD (proposed)** | **3 (Counterfactual)** | **SCM + counterfactual correction** | **Instance-level, causal** |

Pearl's causal hierarchy에서의 positioning이 명확한 차별점:
- Level 1 -> Level 2 -> **Level 3**으로의 progression이 자연스러운 narrative

---

## 관련 노트

- [../2026-03-23_연구방향_후보.md](../2026-03-23_연구방향_후보.md) — 초기 방향 탐색
- [../../analysis/fm_ad_robustness/2026-03-23_theoretical_foundations.md](../../analysis/fm_ad_robustness/2026-03-23_theoretical_foundations.md) — 이론적 기초
- [../../analysis/feature_disentanglement/2026-03-23_deep_survey.md](../../analysis/feature_disentanglement/2026-03-23_deep_survey.md) — Disentanglement survey
- [../../experiments/2026-03-26_phase3_linear_disentanglement/report.md](../../experiments/2026-03-26_phase3_linear_disentanglement/report.md) — PCA-NSP 실험 결과

## 참고문헌

### Causal Inference
- Pearl, J. (2009). Causality. Cambridge University Press.
- Chernozhukov et al. (2018). "Double/Debiased Machine Learning." Econometrics Journal.
- Lu et al. (NeurIPS 2023). "Invariant Anomaly Detection under Distribution Shifts: A Causal Perspective" (PCIR).

### Feature Identifiability
- Khemakhem et al. (AISTATS 2020). "Variational Autoencoders and Nonlinear ICA."
- von Kugelgen et al. (NeurIPS 2021). "Self-Supervised Learning Provably Isolates Content from Style."
- Zimmermann et al. (ICML 2021). "Contrastive Learning Inverts the Data Generating Process."

### AD Robustness
- Chen et al. (AAAI 2025). "FiCo: Filter or Compensate."
- Huang et al. (ICLR 2023). "HOOD: Harnessing OOD Examples via Augmenting Content and Style."
- Ngweta et al. (ICML 2023). "PISCO: Simple Disentanglement of Style and Content."
