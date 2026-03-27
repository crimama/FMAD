# Theoretical Foundations: Why FM Features May Fail for AD under Distribution Shift

> 작성: 2026-03-23
> 목적: FM-AD robustness 연구의 이론적 backbone 정리
> 관련: mechanism_analysis.md, failure_survey.md

---

## 1. Contrastive Learning and Information Theory

### 1.1 InfoNCE and Mutual Information

**핵심 결과 (van den Oord et al., 2018):**
InfoNCE loss는 두 변수 간 mutual information의 lower bound이다.

```
L_InfoNCE = -E[ log( exp(f(x,c)) / sum_j exp(f(x_j,c)) ) ]
```

최적화하면: `I(X; C) >= log(K) - L_InfoNCE` (K = negative samples 수)

수학적 직관: InfoNCE는 density ratio `p(x|c)/p(x)`를 추정하는 것과 동치. 즉 contrastive learning은 "context c가 주어졌을 때 x가 얼마나 예측 가능한가"를 학습한다.

**한계**: InfoNCE는 MI의 loose lower bound. K가 유한하면 `log(K)`로 상한이 제한되어, 실제 MI가 높은 경우 과소추정한다. 이는 representation이 이론적 최적에 도달하지 못할 수 있음을 의미.

### 1.2 Multiview Assumption and Minimal Sufficient Representation

**핵심 결과 (Tian et al., NeurIPS 2020 — "What Makes for Good Views for Contrastive Learning?"):**

Contrastive learning의 optimal representation은 **minimal sufficient statistic**이다:
- **Sufficient**: `I(Z; Y) = I(X; Y)` — representation Z가 task label Y에 대해 input X와 동일한 정보를 보존
- **Minimal**: `I(Z; X)`를 최소화 — task에 불필요한 정보는 버린다

이를 **InfoMin principle**이라 부른다: view 쌍 (v1, v2) 사이의 MI `I(v1; v2)`를 줄이되, task-relevant information은 보존하는 sweet spot이 존재.

수학적 직관: Information Bottleneck (IB) framework과 직결.
```
min I(Z; X) - beta * I(Z; Y)
```
beta가 클수록 task 정보를 더 보존, 작을수록 더 압축.

### 1.3 Task-Relevant Information 소실 문제

**핵심 결과 (Wang et al., CVPR 2022 Oral — "Rethinking Minimal Sufficient Representation in Contrastive Learning"):**

Contrastive learning이 minimal sufficient representation을 얻더라도, **downstream task에 필요한 정보가 view 간 shared information에 포함되지 않으면 소실**된다.

핵심 증명:
- View pair (v1, v2)의 shared information = `I(v1; v2)`
- Task Y에 필요하지만 shared되지 않는 정보 = `I(X; Y) - I(v1 ∩ v2; Y)` > 0 가능
- Contrastive learning은 이 non-shared task-relevant information을 원리적으로 복구 불가

**FM-AD 연결**: Augmentation이 color jittering, cropping 등으로 정의되면:
- Shared info = spatial structure, semantic content (augmentation에 invariant)
- Discarded info = exact color, texture detail, fine-grained appearance
- **AD에 필요한 defect signal은 대부분 discarded 영역에 존재** (scratch, stain, discoloration)

### 1.4 The Fundamental Tension: Invariance vs Sensitivity for AD

Classification은 **nuisance invariance**가 유리하다:
- Lighting, viewpoint, background에 invariant할수록 class label 예측이 좋아진다
- Augmentation이 가르치는 것: "이 변환들은 무시해도 된다"

Anomaly Detection은 **selective sensitivity**가 필요하다:
- Normal variation (lighting, pose)에는 invariant
- Anomalous variation (scratch, dent, stain)에는 sensitive
- 문제: **누가 "무시해야 할 변환"을 정의하는가?**

Contrastive learning의 augmentation set이 곧 invariance set을 정의한다.
AD에서는 이 invariance set이 anomaly signal과 겹치면 치명적:
- Color jittering → color anomaly에 insensitive
- Random cropping → small local defect 정보 소실
- **이것이 FM feature가 AD에서 근본적으로 실패할 수 있는 information-theoretic 이유**

---

## 2. Self-Supervised Learning Feature Properties

### 2.1 DINO/DINOv2: Self-Distillation이 만드는 Feature 특성

**메커니즘 (Caron et al., ICCV 2021 — DINO):**
Self-distillation = student-teacher framework without labels.
- Student와 teacher에 서로 다른 augmented view 입력
- Teacher는 student의 EMA (exponential moving average)
- Cross-entropy loss로 student가 teacher 출력을 모방

**Emergent properties:**
1. **Semantic segmentation 자발적 학습**: Self-attention map이 object boundary와 일치. Supervised ViT나 ConvNet에서는 이 정도로 나타나지 않음
2. **Scene layout 이해**: Patch feature가 자연스럽게 object/background 분리
3. **Visual prototype 형성**: Label 없이 semantic category 수준의 클러스터링

**DINOv2 layer별 특성:**
- **Shallow layers**: Low-level image characteristics (edges, textures, local patterns)
- **Intermediate layers**: 강한 positional information + structural correspondence. kNN matching에 최적
- **Deep layers**: Robust semantic representation. Degradation에 insensitive, high-frequency contour 보존
- **Last layer**: Positional information 감소, semantic abstraction 극대화

### 2.2 Supervised (ImageNet) vs Self-Supervised (DINO) vs CLIP Features

| 속성 | ImageNet Supervised | DINO/DINOv2 | CLIP |
|------|-------------------|-------------|------|
| **학습 목표** | Class label 예측 | Self-distillation (view consistency) | Image-text alignment |
| **Invariance 대상** | Class-irrelevant variation | Augmentation-defined variation | Modality gap |
| **Encode하는 것** | Class-discriminative features | Spatial structure + semantic | Global semantic + language-aligned |
| **Fine-grained detail** | 중간 (class에 필요한 것만) | 높음 (특히 중간 layer) | 낮음 (global alignment 우선) |
| **Texture/Material** | Class 관련 texture만 | 상대적으로 잘 보존 | 약함 (text로 표현 어려운 것 무시) |
| **AD 적합성** | Memory-bank 방식에서 검증됨 (WRN50) | Few-shot AD에서 SOTA | Zero-shot AD 가능하나 fine-grained 약함 |

**핵심 차이점:**
- CLIP: Text-image alignment 때문에 "언어로 기술 가능한" 특성에 편향. Subtle texture defect는 text로 기술하기 어려우므로 feature에 반영 안 됨
- DINOv2: Augmentation-invariance만 학습하므로, augmentation에 포함되지 않은 변환(e.g., subtle defect)은 feature에 보존될 가능성 있음
- ImageNet supervised: 1000-class 분류에 필요한 정보만 보존. AD와 무관한 정보도 많이 남아있어 PatchCore 같은 distance-based method에서 잘 작동

### 2.3 Which Features for Which Anomalies?

- **Structural anomaly** (missing part, wrong shape): Semantic feature가 유리 → CLIP, DINOv2 deep layer
- **Textural anomaly** (scratch, stain): Low-to-mid level feature가 유리 → DINOv2 intermediate layer, ImageNet supervised mid-layer
- **Logical anomaly** (wrong arrangement, constraint violation): Global context 이해 필요 → CLIP (text guidance), DINOv2 deep layer
- **Color anomaly** (discoloration): Color jittering augmentation을 쓴 모델은 모두 취약 → 이 문제를 정면으로 다루는 FM이 없음

---

## 3. Distribution Shift Theory

### 3.1 Shift의 종류와 정의

Joint distribution `p(x, y)`의 변화를 분해:

| Type | 정의 | 예시 (AD context) |
|------|------|------------------|
| **Covariate shift** | `p_train(x) != p_test(x)`, `p(y|x)` 불변 | 조명 변화, 카메라 교체, 계절 변화 |
| **Concept shift** | `p(y|x)` 변화 | 새로운 defect type 등장, 허용 기준 변경 |
| **Domain shift** | `p_source(x,y) != p_target(x,y)` 전체 | 다른 공장, 다른 제품 라인 |
| **Label shift** | `p(y)` 변화, `p(x|y)` 불변 | Anomaly 비율 변화 |

### 3.2 FM Feature가 Shift를 어떻게 변환하는가

FM의 feature extractor `phi: X -> Z`가 input space의 shift를 feature space로 mapping할 때:

**이론적 가능성 세 가지:**
1. **Shift 흡수 (바람직)**: `phi`가 nuisance variation에 invariant하면 feature space에서 shift가 사라짐. e.g., lighting change → DINOv2가 invariant → feature 불변
2. **Shift 보존**: Feature space에서 shift magnitude가 input space와 비슷. e.g., novel texture → feature에도 반영
3. **Shift 증폭 (위험)**: Feature space에서 shift가 오히려 확대. e.g., 특정 frequency band의 작은 변화가 feature의 큰 변화로 mapping

**핵심 문제**: FM은 대부분 case 1을 목표로 학습되었다. 그러나 AD에서는:
- Normal variation의 shift는 흡수되어야 (case 1)
- Anomaly signal의 shift는 보존되어야 (case 2)
- 동일한 `phi`로 두 가지를 동시에 달성할 수 없을 때 문제 발생

**Feature norm과 angle의 역할:**
- Covariate shift → feature norm에 영향 (activation 약해짐)
- Concept shift → feature angle에 영향 (방향 변화)
- 이 분리가 AD에서 각각 다른 failure mode를 유발

### 3.3 Method별 Robustness 차이: PatchCore vs EfficientAD

**PatchCore (Memory Bank) — 상대적으로 Robust:**
- Memory bank = normal feature의 explicit 저장. Decision boundary가 data-driven
- Feature extractor는 frozen pretrained model → test-time에도 동일한 mapping
- Feature space에서 shift가 "흡수"되면 → memory bank 거리가 안정
- Layer 2-3 사용 (mid-level feature) → semantic abstraction과 local detail의 균형

**EfficientAD (Knowledge Distillation) — 상대적으로 Fragile:**
- Student-teacher discrepancy로 anomaly 탐지 → student가 "normal을 얼마나 잘 모방하는가"에 의존
- **Normality forgetting**: Student가 anomaly도 잘 reconstruction → sensitivity 저하
- **Generalization-robustness tradeoff**: Student가 normal에 너무 잘 generalize하면 anomaly에도 generalize → miss. Normal에만 특화하면 slight domain shift에도 false positive
- Capacity gap (teacher-student) → fine-grained pattern에 inconsistent response → domain shift 시 불안정
- 학습된 mapping이 training distribution에 overfit → shift 시 전체 decision boundary가 흔들림

**이론적 해석:**
- PatchCore: Non-parametric method → decision boundary가 data에 직접 의존 → shift 시 점진적 열화
- EfficientAD: Parametric method → 학습된 함수가 training distribution의 manifold에 fit → shift 시 급격한 열화 가능

---

## 4. Anomaly Detection Theory

### 4.1 One-Class Classification Theory

**핵심 결과 (Scholkopf et al., 2001; Tax & Duin, 2004):**

One-Class SVM: Origin에서 data를 최대 margin으로 분리하는 hyperplane 학습.
SVDD: Data를 감싸는 최소 반경의 hypersphere 학습.

```
min R^2 + (1/nu*n) * sum(max(0, ||phi(x_i) - c||^2 - R^2))
```

RBF kernel 사용 시 두 방법은 수학적으로 동치.

수학적 직관: "Normal data의 support를 추정"하는 것. Support 밖의 점 = anomaly.

### 4.2 AD가 Classification과 근본적으로 다른 이유

**Classification의 feature 요구사항:**
- Inter-class separation 극대화
- Class-irrelevant variation에 invariant
- Decision boundary = class 간 경계

**AD의 feature 요구사항:**
- Normal data의 compact representation (intra-class compactness)
- Anomaly에 대한 sensitivity (compact region 밖으로 밀어내야)
- Decision boundary = normal data의 support boundary (단일 class의 윤곽)
- **Unknown anomaly**에 대해서도 작동해야 함 (open-set 문제)

| 차이점 | Classification | Anomaly Detection |
|--------|---------------|-------------------|
| 학습 데이터 | 모든 class | Normal class만 |
| Decision boundary | Class 간 | Normal의 support |
| Feature 목표 | Discriminative | Descriptive + Compact |
| Invariance 대상 | Class-irrelevant | Normal variation |
| Sensitivity 대상 | Class-relevant | Any deviation from normal |
| Open-set 여부 | Closed-set | Open-set (unknown anomaly) |

**핵심 통찰**: Classification feature는 "어떤 class인가?"를 답하도록 최적화. AD feature는 "normal인가 아닌가?"를 답해야 한다. 전자는 inter-class boundary, 후자는 intra-class boundary를 정의한다. FM은 전자를 위해 학습되었으므로, 후자에 suboptimal할 수밖에 없다.

### 4.3 Feature Space에서의 Anomaly-Normal Boundary under Shift

Normal data의 feature distribution을 `P_N`이라 하면:
- **Without shift**: `P_N` 고정 → support boundary 안정 → threshold 설정 가능
- **With shift**: `P_N` → `P_N'` 변화 → 두 가지 문제 발생:

1. **False positive 증가**: `P_N'`의 tail이 원래 boundary 밖으로 → normal인데 anomaly로 판정
2. **False negative 증가**: Anomaly distribution `P_A`도 shift → 원래 boundary 안으로 들어올 수 있음

Feature space에서 shift의 영향:
```
d(x, memory_bank) = d(phi(x), phi(x_train))
```
- `phi`가 shift에 invariant하면: `d` 안정 → 둘 다 해결
- `phi`가 shift에 sensitive하면: `d`가 변동 → threshold 재설정 필요
- **문제**: anomaly에 대한 sensitivity는 유지하면서 nuisance shift에만 invariant해야 함

### 4.4 Robust AD를 위한 Feature Space 조건

이상적인 feature space `Z`는:
1. **Anomaly sensitivity**: `||phi(x_normal) - phi(x_anomaly)|| >> epsilon` for some threshold
2. **Nuisance invariance**: `||phi(x; env1) - phi(x; env2)|| < delta` for same x under different environments
3. **Compactness**: Normal features가 compact cluster 형성
4. **Separability**: Normal cluster와 anomaly region이 분리 가능

조건 1과 2의 동시 충족이 핵심 난제:
- FM은 조건 2를 잘 만족 (invariance 학습)
- 그러나 조건 1을 희생할 수 있음 (invariance가 anomaly signal을 삼킴)
- 조건 3은 FM의 semantic clustering으로 어느 정도 충족
- 조건 4는 anomaly type에 따라 달라짐

---

## 5. Identifiability and Disentanglement Theory

### 5.1 Nonlinear ICA의 Identifiability Problem

**배경**: Linear ICA (Hyvarinen, 1999)는 독립 source를 non-Gaussian이면 identifiable하게 복원 가능. 그러나 nonlinear mixing에서는 일반적으로 **identifiable하지 않음** — 무한히 많은 해가 존재.

**핵심 결과 (Khemakhem et al., AISTATS 2020):**

Nonlinear ICA가 identifiable하려면 **auxiliary variable u** (e.g., class label, time index, domain)가 필요:

```
p(s | u) = prod_i p(s_i | u)  (conditionally factorial prior)
```

조건:
1. Mixing function `f`가 injective (invertible on its image)
2. Prior `p(s|u)`가 exponential family with sufficient statistics
3. Enough distinct values of `u` to span the parameter space

이 조건 하에서 latent variable `s`는 **component-wise invertible transformation까지** identifiable.

수학적 직관: Auxiliary variable `u`가 latent source의 분포를 "다양하게 변화"시켜야, 관측 데이터에서 mixing function을 역추론할 수 있다. Label이나 domain 정보가 없으면 해가 유일하지 않다.

### 5.2 AD에서의 Nuisance-Anomaly Disentanglement 연결

FM feature `z = phi(x)`를 두 부분으로 decompose한다고 가정:
```
z = (z_nuisance, z_anomaly)
```
- `z_nuisance`: Lighting, viewpoint, camera 등 nuisance factors encoding
- `z_anomaly`: Defect 관련 정보 encoding

**Identifiability 관점의 문제:**
1. FM은 auxiliary variable 없이 학습됨 (self-supervised) → nonlinear ICA 이론에 의하면 identifiable decomposition을 보장할 수 없음
2. 단, **domain label** (e.g., 어떤 환경에서 촬영했는가)을 auxiliary variable로 사용하면 identifiability 조건을 충족할 수 있음
3. AD에서는 anomaly label이 없으므로, domain/environment label만으로 nuisance factor를 identify해야 함

**Structured Nonlinear ICA (NeurIPS 2021)의 확장:**
- Noise가 있어도 identifiable
- Spatial/temporal dependency structure를 활용 가능
- AD에서 시간적으로 같은 생산 라인의 이미지 → temporal structure = auxiliary variable 후보

### 5.3 Practical Implications for FM Feature Manipulation

**Direct implication 1: Subspace Decomposition**
FM feature space에서 nuisance direction과 anomaly direction을 분리할 수 있다면:
- Nuisance direction을 suppress → shift-invariant representation
- Anomaly direction을 preserve → sensitivity 유지
- 이론적으로 가능하려면 multi-environment data가 필요 (identifiability condition)

**Direct implication 2: Domain을 Auxiliary Variable로 활용**
- 여러 environment에서 수집한 normal data가 있으면 → environment label = auxiliary variable
- 이를 통해 "환경에 따라 변하는 feature dimension" = nuisance로 identify 가능
- Nuisance를 제거한 residual = anomaly-relevant feature

**Direct implication 3: Identifiability 없이도 유용한 근사**
- 완전한 identifiability가 없어도, 통계적으로 "multi-domain에서 variance가 높은 direction" ≈ nuisance
- PCA/whitening 기반 방법으로 근사적 decomposition 가능
- 이론적 guarantee는 약하지만, 실용적으로 작동할 수 있음

---

## 종합: FM-AD Robustness의 이론적 Landscape

### The Core Dilemma

```
FM Training Objective:    max I(Z_v1; Z_v2)  s.t. min I(Z; X \ {shared info})
                          → Invariance to augmentation-defined nuisance
                          → But also invariance to anomaly signals that overlap with nuisance

AD Objective:             Detect any deviation from normal
                          → Requires sensitivity to ALL types of anomaly
                          → Including those in the "nuisance" space of FM
```

### Information Flow Analysis

```
Input x = (content, nuisance, anomaly_signal)
                    ↓ FM feature extraction phi
Feature z = phi(x) = f(content, partially_absorbed_nuisance, partially_lost_anomaly)
                    ↓ AD method (PatchCore, EfficientAD, etc.)
Score s = g(z)     → Normal vs anomaly decision
```

문제 지점:
1. `phi`에서 anomaly_signal이 partially lost → **Information-theoretic limit** (Section 1)
2. `phi`가 encode하는 것이 AD에 optimal하지 않음 → **Feature mismatch** (Section 2)
3. Nuisance가 완전히 흡수되지 않고 feature에 남아 shift를 유발 → **Residual shift** (Section 3)
4. AD의 one-class 특성으로 boundary 정의가 classification보다 어려움 → **Boundary fragility** (Section 4)
5. Nuisance와 anomaly의 분리가 이론적으로 어려움 → **Identifiability gap** (Section 5)

### Research Opportunity

**가장 큰 이론적 gap**: FM feature에서 nuisance-encoding subspace와 anomaly-encoding subspace를 principled하게 분리하는 방법이 없다. Nonlinear ICA 이론은 auxiliary variable (domain label)을 제공하면 identifiable decomposition이 가능함을 보여주지만, 이를 AD에 적용한 연구는 0편.

**제안 가능한 이론적 framework**:
1. Multi-environment normal data → environment label as auxiliary variable
2. FM feature space에서 environment-variant direction identify (≈ nuisance)
3. Environment-invariant residual = anomaly-sensitive feature
4. 이 residual feature로 AD 수행 → shift-robust

이는 Invariant Risk Minimization (IRM)의 AD 버전으로도 해석 가능하며, causal inference의 "invariant prediction" 원리와도 연결된다.

---

## 관련 노트
- [2026-03-23_mechanism_analysis.md](2026-03-23_mechanism_analysis.md)
- [2026-03-23_failure_survey.md](2026-03-23_failure_survey.md)
- [2026-03-23_solution_survey.md](2026-03-23_solution_survey.md)
- [2026-03-23_ICLR_positioning.md](2026-03-23_ICLR_positioning.md)

## 주요 참고문헌

### Contrastive Learning / Information Theory
- van den Oord et al. (2018). "Representation Learning with Contrastive Predictive Coding." arXiv:1807.03748
- Tian et al. (NeurIPS 2020). "What Makes for Good Views for Contrastive Learning?" — InfoMin principle, minimal sufficient representation
- Wang et al. (CVPR 2022 Oral). "Rethinking Minimal Sufficient Representation in Contrastive Learning" — task-relevant info 소실 증명

### Self-Supervised Features
- Caron et al. (ICCV 2021). "Emerging Properties in Self-Supervised Vision Transformers" (DINO)
- Oquab et al. (2024). "DINOv2: Learning Robust Visual Features without Supervision"

### Distribution Shift / AD Robustness
- Lu et al. (NeurIPS 2023). "Invariant Anomaly Detection under Distribution Shifts: A Causal Perspective" — PCIR
- Chen et al. (AAAI 2025). "Filter or Compensate: Towards Invariant Representation from Distribution Shift for AD" — FiCo
- RoDA (2025). "Robust Distribution Alignment for Industrial AD under Distribution Shift"

### One-Class Classification
- Scholkopf et al. (2001). "Estimating the Support of a High-Dimensional Distribution" — One-Class SVM
- Tax & Duin (2004). "Support Vector Data Description" — SVDD
- Ruff et al. (ICML 2018). "Deep One-Class Classification" — Deep SVDD

### Identifiability / Disentanglement
- Khemakhem et al. (AISTATS 2020). "Variational Autoencoders and Nonlinear ICA: A Unifying Framework"
- Structured Nonlinear ICA (NeurIPS 2021). "Disentangling Identifiable Features from Noisy Data"
