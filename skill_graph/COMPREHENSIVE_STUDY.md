# Foundation Model 기반 Few-shot Anomaly Detection의 Distribution Shift 취약성: 원인 분석, 해결 시도, 그리고 열린 문제

> 연구 기간: 2026-03-23 ~ 03-30 | 40+개 실험, 15개 검증 패턴, 5개 impossibility 증거
> 이 문서는 논문 형태가 아닌, **연구 과정 전체를 학습/참고할 수 있도록 자세히 정리한 study note**이다.

---

## 목차

1. [서론: FM-AD의 약속과 현실](#1-서론)
2. [관련 연구](#2-관련-연구)
3. [문제 정의: Few-shot FM-AD under Distribution Shift](#3-문제-정의)
4. [근본 원인 분석: 왜 FM Feature가 Shift에 취약한가](#4-근본-원인-분석)
5. [Phase 1: Robustness Stress Test — 문제의 실증](#5-phase-1)
6. [Phase 2: Feature Space 분석 — V-shape Profile과 Entanglement](#6-phase-2)
7. [Phase 3: Paired NSP — Oracle Solution과 Linear Optimality](#7-phase-3)
8. [Phase 4: Unpaired 해결 시도 — 5가지 전멸](#8-phase-4)
9. [Phase 4b: Minimal Calibration — 얼마나 필요한가](#9-phase-4b)
10. [Phase 4c-d: Patch-level 접근 — Spatial Structure 활용](#10-phase-4cd)
11. [Phase 4e: Shift-Robust Scoring — Structural Prior 기반](#11-phase-4e)
12. [Few-shot Multi-condition NSP — 효과와 한계](#12-fewshot-nsp)
13. [이론적 연결: Task Vector와 Shift Vector](#13-task-vector)
14. [종합 인사이트와 검증된 패턴](#14-종합-인사이트)
15. [열린 문제와 향후 연구 방향](#15-열린-문제)
16. [참고 문헌](#16-참고-문헌)

---

## 1. 서론: FM-AD의 약속과 현실 {#1-서론}

### 1.1 Visual Anomaly Detection의 패러다임 전환

Visual Anomaly Detection(VAD) 분야는 2024-2025년에 세 가지 패러다임 전환을 겪고 있다.

**Per-class → Unified Multi-class**: 카테고리별 개별 모델에서 단일 모델로 모든 카테고리를 처리하는 방향으로 전환. Dinomaly(CVPR 2025)가 최초의 multi-class SOTA를 달성하며 이 방향의 feasibility를 실증했다.

**Task-specific → Foundation Model 기반**: 전용 backbone(WideResNet, EfficientNet)에서 DINOv2, CLIP 등 사전학습된 Foundation Model(FM)로 전환. Feature engineering에서 adaptation engineering으로 연구의 초점이 이동했다. DINOv2의 self-supervised feature는 ImageNet-22K 수준의 범용 시각 표현을 제공하며, CLIP은 text-image alignment으로 zero-shot AD를 가능하게 한다.

**Supervised → Zero/Few-shot**: 대량의 이상치 라벨 없이 소수 정상 샘플만으로 탐지. 2024-2025년에만 9편 이상의 few-shot AD 논문이 CVPR/ICLR에 발표되었다. Zero-shot은 ~93-95% AUROC, 4-shot은 ~97%+를 달성한다.

이 전환의 중심에 Foundation Model이 있다. DINOv2와 CLIP 기반 방법들은 MVTec AD에서 91-99% I-AUROC를 달성하며, **"FM feature면 충분하다"**는 가정이 지배적이다.

### 1.2 문제 제기: 이 가정은 틀렸다

MVTec AD 2(2025), RobustAD(2024) 등 **현실적 벤치마크**에서 FM-AD 방법들은 25-34pp의 심각한 성능 하락을 보인다. 이는 단순한 성능 저하가 아닌, FM feature 자체의 구조적 한계를 시사한다.

| 벤치마크 | 특성 | FM-AD 성능 |
|----------|------|-----------|
| MVTec AD | Clean, 통제된 환경 | 91-99% I-AUROC |
| MVTec AD 2 | 조명/시점 변화 포함 | 57-72% (-25~34pp) |
| RobustAD | 9종 domain shift | 73-84% |
| Real-IAD Variety | 160 카테고리 | 10-20% 추가 열화 |

### 1.3 이 연구가 채우는 공백

기존 연구는 세 가지로 나뉜다:

- **벤치마크 구축**: MVTec AD 2, RobustAD — "깨진다"는 **사실만** 확인
- **현상 관찰**: Phi-eat, SINDER — FM feature의 **특이 행동만** 보고
- **Task-specific 해결**: PILOT(prompt tuning), EPHAD(score calibration) — 특정 **증상만** 완화

**이 연구가 채우는 공백**:
1. FM feature가 **왜** 깨지는지 — 메커니즘 진단 (7가지 실패 원인, entanglement 정량화)
2. **어떻게** 고치는지 — paired linear projection (+29.4pp, 0 params)
3. **왜 unpaired로는 고칠 수 없는지** — 5가지 독립적 impossibility evidence
4. **Scoring-level structural prior** — "anomaly는 local, shift는 global" 원리 (+22pp)

FM feature를 anomaly-relevant vs nuisance로 명시적으로 분리한 기존 연구 = **0건**.

---

## 2. 관련 연구 {#2-관련-연구}

### 2.1 Foundation Model 기반 Anomaly Detection

**CLIP 계열**:
- WinCLIP (CVPR 2023): Zero-shot text prompt로 AD. 91.8% AUROC on MVTec AD.
- AnomalyCLIP (ICLR 2024): Object-agnostic prompt 학습. 17개 데이터셋 평가.
- PromptAD (CVPR 2024): Few-shot prompt tuning. 97.3% AUROC.

**DINOv2 계열**:
- AnomalyDINO (WACV 2025): Few-shot (1-4장) patch-based. PCA-based background masking.
- Dinomaly (CVPR 2025): Multi-class unified. Noisy bottleneck + linear attention decoder.
- SuperAD (2025): Training-free DINOv2 memory bank. 가장 robust한 FM-AD 중 하나 (-3.5% on shift).

**전통 방법**:
- PatchCore (CVPR 2022): WideResNet-101 → greedy coreset → kNN. 가장 robust (-3pp on AD2).
- EfficientAD (WACV 2024): PDN teacher-student. 최고 속도(>600fps)이지만 가장 fragile.
- ResAD (NeurIPS 2024 Spotlight): Residual feature + NF scoring. Cross-class invariant.

### 2.2 Distribution Shift와 Robustness

**벤치마크**:
- MVTec AD 2 (2025): 8 categories, multi-lighting conditions. AU-PRO(0.05) = 기존 대비 6배 엄격.
- RobustAD (2024): 5 categories × 9 domain shift types (brightness, contrast, noise, blur 등).

**Robustness 평가**:
- SuperAD: Training-free DINOv2가 training-based보다 robust (3.5% vs 11.3% drop).
- RoBiS: DINOv2 + augmentation 기반. 절대 성능은 높으나 augmentation이 오히려 bias 도입.

**Domain Adaptation/Generalization for AD**: 거의 존재하지 않음.
- FiCo (AAAI 2025): Filter-compensate. AD-specific DA의 거의 유일한 시도.
- PILOT (BMVC 2025): Test-time prompt tuning. Prompt-level만, feature-level TTA 없음.

### 2.3 Feature Disentanglement 이론

**Nonlinear ICA Identifiability** (von Kügelgen, NeurIPS 2021):
- Self-supervised learning에서 content와 style의 block identifiability 증명.
- Auxiliary variable(환경 레이블)이 있으면 nuisance와 content를 분리 가능.
- **AD 적용**: Paired data = auxiliary variable → nuisance 분리의 이론적 보장.

**PISCO** (ICML 2023):
- CLIP representation에서 post-hoc linear decomposition이 유효함을 실증.
- Style/content 분리가 pretrained feature space에서 선형적으로 가능.
- DINOv2로 확장 가능성 시사.

**핵심 연결**: DINOv2의 self-distillation(DINO) + masked image modeling(iBOT) 학습은 multi-crop augmentation에 대한 view-invariant representation을 생성한다. 이 과정에서 anomaly detection에 필요한 fine-grained 외관 정보(색상 변화, blur, texture 차이)가 "같은 이미지의 다른 view"로 취급되어 소실된다. 이를 복원하려면 feature space에서 nuisance 방향을 식별하고 제거해야 한다 — 이것이 Nonlinear ICA 이론과 연결.

### 2.4 Task Arithmetic과의 연결

**Task Vector** (Ilharco et al., ICLR 2023):
- τ = θ_finetuned - θ_pretrained: weight space에서의 차이 벡터.
- 모델에서 τ를 빼면 해당 task 능력 제거, 더하면 추가.
- Weight space에서 task가 **선형적으로** 인코딩된다는 실증.

**Shift Vector와의 구조적 유사성**:
- d = f(shifted) - f(clean): feature space에서의 차이 벡터.
- Feature에서 d 방향을 빼면 shift 영향 제거.
- **Feature space에서 shift가 선형적으로 인코딩**된다는 것을 우리의 실험이 실증 (14개 nonlinear 변형 전멸 → linear optimality).

이 연결은 "왜 linear projection이 최적인가"에 대한 이론적 근거를 제공한다.

---

## 3. 문제 정의: Few-shot FM-AD under Distribution Shift {#3-문제-정의}

### 3.1 Setting

```
Given:  K장의 정상 reference 이미지 (K = 1, 2, 4, 8)
        DINOv2 ViT-B/14 (frozen, pretrained) as feature extractor
        Distribution shift type 미지 (조명? 시점? 포커스? 알 수 없음)
Test:   Unknown condition에서 촬영된 이미지들
Task:   Image-level anomaly detection (binary: 정상/이상)
Metric: I-AUROC on shifted test set
```

### 3.2 기존 Few-shot FM-AD 파이프라인

```
Reference images {x_1, ..., x_K} → DINOv2 Layer L → features {f_1, ..., f_K}
→ Normal distribution modeling (μ, Σ from K features)
→ Test image → DINOv2 → f_test
→ Score = Mahalanobis(f_test; μ, Σ) or kNN distance
→ High score = anomaly
```

### 3.3 왜 이 설정이 중요한가

- **FM-AD의 핵심 약속**: "소수 정상 이미지만으로 AD 가능"
- **현실**: 공장에서 reference를 촬영한 조건과 실제 검사 조건이 다름 (조명 변화, 카메라 이동 등)
- **결과**: 25-34pp 하락 — FM-AD의 약속이 무너짐
- **기존 연구**: 이 문제를 체계적으로 분석한 연구 없음

---

## 4. 근본 원인 분석: 왜 FM Feature가 Shift에 취약한가 {#4-근본-원인-분석}

### 4.1 원인 1: FM Feature가 실제 Shift에 "불변"이 아니다

DINOv2는 self-supervised 학습에서 augmentation(color jitter, blur, crop, rotation)에 invariant하도록 학습된다. 하지만 **학습 시 사용된 augmentation과 실제 distribution shift는 다르다**.

```
학습 시: color jitter σ=0.4 범위 내의 색상 변화에 invariant
실제:    factory 조명이 LED→형광등으로 교체 → 색온도 전체 변화
         → 학습한 invariance 범위 밖 → feature가 이동
```

이것은 DINOv2의 학습 구조에서 기인하는 근본적 tension이다. DINOv2는 self-distillation(DINO)으로 학습하며, student network가 teacher network(EMA)의 출력을 모방하도록 훈련된다. Multi-crop 전략에서 global crop과 local crop에 color jitter, blur, solarization 등의 augmentation이 적용되고, **다른 augmented view에서도 같은 teacher 출력을 재현**하도록 학습된다. 이것은 contrastive loss(InfoNCE)가 아닌 distillation loss(cross-entropy)지만, **결과적으로 augmentation에 대한 view invariance를 생성**한다. AD에서는 color shift가 anomaly일 수 있으므로 (변색 = 오염), AD에 필요한 fine-grained appearance 정보가 **학습 과정에서 "nuisance"로 취급되어 이미 손실**되어 있다.

**관련 연구**: Phi-eat(2025)은 DINOv2가 "무엇인가(object identity)"를 "어떻게 보이는가(material/texture)"보다 강하게 인코딩한다는 것을 실증. 이는 정확히 이 문제의 발현.

### 4.2 원인 2 (핵심): Nuisance-Anomaly Entanglement

FM feature는 여러 정보를 하나의 벡터에 **혼합**하여 인코딩한다:

```
f(image) = [semantic 정보] + [외관/조건 정보] + [결함 정보] + ...
```

**문제**: 이 성분들이 feature space에서 **직교하지 않는다 (entangled)**.

구체적으로: **shift vector의 방향과 anomaly 특징의 방향이 유사하여 분리가 어렵다.**

우리의 측정 (Phase 2):
- **Layer 11** (최종 layer): shift 방향과 anomaly 방향의 PC projection correlation = **0.53** → 심각하게 entangled. Top-10 principal component 중 40%가 shift와 anomaly 모두에 반응.
- **Layer 8** (중간 layer): correlation = **0.12** → 자연적으로 분리됨. 예외적.

**Entanglement의 결과**:

정상 이미지가 조건 변화(조명 등)로 feature space에서 이동할 때, 그 이동 방향이 anomaly가 존재할 때의 이동 방향과 겹침 → scoring function이 "조건 변화"를 "결함"으로 오인 → **false positive 폭증**.

**반례**: sheet_metal 카테고리는 shift magnitude가 크지만(0.33) 하락이 moderate(-20pp). 이유: 이 카테고리에서 shift 방향이 anomaly 방향과 **직교**. → **shift의 크기가 아니라 방향(anomaly와의 관계)이 성능 하락을 결정한다.**

### 4.3 원인 3: Few-shot에서 취약성 증폭

```
Full-shot (200장): Σ가 넓음 → 정상 변이 범위가 넓어 약간의 shift를 흡수
Few-shot (K=4장):  Σ가 매우 좁음 → 조금만 shift해도 "정상 범위 밖" 판정
```

Few-shot은 "정상의 정의"가 좁아서 동일한 shift에 더 민감하게 반응한다. 이것이 full-shot(-25pp)보다 few-shot에서 하락이 더 클 수 있는 이유.

### 4.4 세 원인의 상호작용 요약

```
원인 1: FM feature가 shift에 불변이 아님 → feature가 이동
   ×
원인 2: 이동 방향이 anomaly 방향과 entangled (corr 0.53) → 이동이 anomaly로 오인
   ×
원인 3: Few-shot에서 정상 범위가 좁음 → 작은 이동도 범위를 벗어남
   ↓
결과: 25-34pp 하락
```

**가장 근본적인 원인은 2 (entanglement)**. 원인 1(feature 이동)은 FM의 본질적 특성이고, 원인 3(few-shot 제약)은 setting의 제약. 하지만 **이동 방향이 anomaly와 직교하면 문제 없다** — entanglement만 해결하면 된다.

### 4.5 7가지 실패 메커니즘 (상세)

위의 3가지 근본 원인을 더 세분화하면 7가지 구체적 메커니즘을 식별할 수 있다:

| # | 메커니즘 | 핵심 | 근거 |
|---|---------|------|------|
| 1 | **Invariance-as-Information-Destruction** | Augmentation이 AD-relevant 외관 정보 파괴 | Contrastive learning 이론, Phi-eat |
| 2 | **Semantic Dominance** | "무엇인지" >> "어떻게 보이는지" | Phi-eat (2025) 실증 |
| 3 | **High-Norm Token Artifacts** | 특정 patch norm 7.5배 (434 vs 57.6) | SINDER (ECCV 2024) |
| 4 | **Inter-Class Interference** | Multi-class에서 클래스 간 분산이 anomaly 압도 | Real-IAD Variety: 160 cat에서 10-20% 추가 열화 |
| 5 | **Nuisance-Semantic Entanglement** | Shift ↔ anomaly 방향 비직교 (corr 0.53) | **Phase 2 직접 측정** |
| 6 | **Pretraining Distribution Mismatch** | ImageNet에 산업 결함 거의 부재 | 도메인 간극 |
| 7 | **Token-Level Shift Accumulation** | Patch shift가 attention으로 비균일 증폭 | SPAD 측정: spatial uniformity 2.6% |

**#1 상세**: DINOv2는 self-distillation(DINO) + masked image modeling(iBOT)으로 학습한다. DINO의 multi-crop 전략에서 color jitter, blur, solarization 등이 augmentation으로 적용되며, student가 teacher(EMA)의 출력을 모방하는 과정에서 이러한 augmentation에 **invariant한 representation**이 생성된다 (contrastive loss가 아닌 distillation loss이지만 결과는 유사). 그런데 AD에서는 변색=오염, blur=코팅결함, distortion=구조손상이므로, **anomaly-relevant appearance가 feature space에서 이미 소실**. Self-distillation의 view-invariance 목표와 AD의 fine-grained sensitivity 요구 사이에 근본적 tension이 존재한다.

**#2 상세**: Phi-eat(2025) 논문이 실증 — DINOv2 feature는 material과 texture보다 object identity를 강하게 인코딩. AD는 정반대(표면 이상 >> 물체 정체)를 요구. Feature space의 구조가 AD 요구와 정반대.

**#3 상세**: SINDER(ECCV 2024) 발견 — DINOv2의 특정 patch token이 norm 434 vs 정상 57.6 (7.5배). 이미지 독립적 방향(pairwise angle 3.1°), weight matrix의 leading singular vector에서 기인. Distance-based scoring이 이 artifact에 지배.

**#5 상세 (핵심)**: 이 연구에서 직접 정량화. Layer 11에서 Top-10 PC 중 4개가 shift와 anomaly 모두에 반응(PC4의 shift projection 0.215, anomaly projection 0.707). NSP의 이론적 동기이자 paired data가 필수적인 이유. **기존 연구에서 이를 명시적으로 다룬 논문 = 0건.**

**#7 상세**: 환경 변화 시 각 patch token이 local context로 소폭 shift → attention의 cross-patch interaction이 이 shift를 **비균일하게 증폭** → spatial grid 전체에서 누적. 같은 pixel-level 조명 변화라도 DINOv2 feature space에서 patch마다 다르게 인코딩됨 (SPAD 진단에서 확인: spatial uniformity = **2.6%**).

### 4.6 핵심 딜레마: Robustness ↔ Discriminability Trade-off

```
Layer 8:  Entanglement 0.12 → shift-robust하지만 discriminative power 부족 (AD2 59.9%)
Layer 11: Entanglement 0.53 → discriminative하지만 shift에 fragile (AD2 53.0%)
```

어떤 layer를 쓰든 robustness와 discriminability를 동시에 얻기 어렵다. 이 딜레마를 해결하는 것이 연구의 핵심 과제.

---

## 5. Phase 1: Robustness Stress Test — 문제의 실증 {#5-phase-1}

### 5.1 실험 설계

3가지 SOTA FM-AD 방법을 MVTec AD(clean)와 MVTec AD 2(shifted)에서 비교:

- **Dinomaly** (CVPR 2025): DINOv2-Reg → noisy bottleneck → decoder. Full-shot, unified multi-class.
- **AnomalyDINO** (WACV 2025): DINOv2 → few-shot (4장) patch-based.
- **AnomalyCLIP** (ICLR 2024): CLIP → zero-shot prompt learning.

### 5.2 결과

| Method | Backbone | MVTec AD | MVTec AD 2 | Drop |
|--------|----------|----------|-----------|------|
| Dinomaly | DINOv2 ViT-B/14-reg | 99.64% | 68.6% | **-31.0pp** |
| AnomalyDINO | DINOv2 ViT-B/14 | 97.7% | 72.7% | **-25.0pp** |
| AnomalyCLIP | CLIP ViT-L/14@336 | 91.5% | 57.9% | **-33.6pp** |

**카테고리별**:

| Category | Dinomaly | AnomalyDINO | AnomalyCLIP | 패턴 |
|----------|---------|------------|-------------|------|
| wallplugs | 44.1% | 41.8% | 41.7% | **전 방법 ~42% (random 수준)** |
| can | 52.8% | 56.7% | 54.4% | 전 방법 실패 (반사면) |
| vial | 79.2% | 85.8% | 54.9% | CLIP 특히 취약 |
| fruit_jelly | 85.9% | 88.2% | 68.3% | DINOv2 > CLIP |

### 5.3 핵심 인사이트

1. **Backbone 무관**: DINOv2든 CLIP이든 모두 실패 → FM 공통 구조적 문제.
2. **wallplugs = 전 방법 동일 실패 (~42%)**: 반사면 + 미세결함은 현재 기술의 근본 한계.
3. **학습 기반이 오히려 더 취약**: Dinomaly(-31pp, 학습) > AnomalyDINO(-25pp, few-shot). 학습이 source domain에 overfitting.
4. **DINOv2 > CLIP**: 평균 10pp+ 차이. CLIP의 text-image alignment이 AD에 불리.

---

## 6. Phase 2: Feature Space 분석 — V-shape Profile과 Entanglement {#6-phase-2}

### 6.1 실험 설계

DINOv2 ViT-B/14의 12개 layer(0-11)에서 200 paired samples(regular↔shifted)의 feature를 추출하여:
- **RelShift**: 각 layer에서 shift magnitude 측정
- **Cosine Similarity**: Clean ↔ shifted feature의 유사도
- **PC Projection Correlation**: Shift direction과 anomaly direction의 겹침도

### 6.2 V-shaped Layer Robustness Profile

```
RelShift:
  0.27  ────────────────          ────  0.26    (fragile)
                    ╲              ╱
                      ╲          ╱
                        ╲      ╱
  0.11                    ╲──╱                   (robust)
  ─────────── Layer 0  4  [8]  11 ──────────────
```

| Layer | RelShift | CosSim | Entanglement | 해석 |
|-------|----------|--------|-------------|------|
| 0-3 (shallow) | 0.22-0.25 | 0.96-0.97 | — | 중간 수준 |
| 4-7 (mid) | 0.26-0.27 | 0.94-0.95 | 높음 | **가장 취약** |
| **8** | **0.11** | **0.99** | **0.12** | **유일한 robust point** |
| 9-10 | 0.15-0.22 | 0.97-0.98 | 중간 | 회복 중 |
| 11 (last) | 0.26 | 0.95 | **0.53** | 다시 취약 |

**"Deep layer = robust"라는 통상적 가정은 틀렸다.** Layer robustness는 비단조적이며, Layer 8만 유일하게 robust.

### 6.3 Entanglement 정량화

Layer 11에서 Top-10 PC의 shift/anomaly projection을 분석:
- PC4: shift projection 0.215, anomaly projection 0.707 → **shift와 anomaly가 같은 PC에 동시 반응**
- Top-10 PC 중 40%가 이런 dual-response → **체계적 entanglement**
- Layer 8에서는 0.12 → 자연적으로 분리됨

### 6.4 t-SNE 시각화

| Pair | Centroid Distance | 의미 |
|------|------------------|------|
| normal_regular ↔ normal_shifted | **1.85** | Shift가 normal을 이동 |
| anomaly_shifted ↔ normal_shifted | **1.09** | **Shifted-normal이 anomaly에 접근** |
| normal_regular ↔ anomaly_regular | 0.87 | 원래도 가까움 |

**Shifted-normal이 anomaly보다 더 가까워진다** → false positive의 직접 원인.

### 6.5 카테고리별 패턴: 크기가 아니라 방향

| Category | Feature Shift | AD Drop | 분석 |
|----------|--------------|---------|------|
| fabric | 0.15 | -27pp | Shift와 하락 비례 |
| wallplugs | **0.36** | **-56pp** | 최대 shift, 최대 하락 |
| **sheet_metal** | **0.33** | **-20pp** | **이상치**: shift 크지만 하락 적음 |

sheet_metal의 경우, shift magnitude는 크지만(0.33) 하락이 moderate(-20pp). 이유: **shift 방향이 anomaly 방향과 직교**. → Entanglement이 낮으면 shift가 커도 문제없다.

---

## 7. Phase 3: Paired NSP — Oracle Solution과 Linear Optimality {#7-phase-3}

### 7.1 NSP (Nuisance Subspace Projection) 알고리즘

```
Input:
  - Paired multi-environment normal data: {(f_i^reg, f_i^shift)}
  - DINOv2 ViT-B/14, Layer 8 features (CLS + patch mean, L2 normalized)
  - K = 100 (nuisance dimensions)

Step 1: Shift Vector 계산
  d_i = f_i^shift - f_i^reg    (같은 물체의 차이 → 물체 정보 상쇄, 순수 shift만 남음)

Step 2: Nuisance Subspace via PCA
  PCA({d_i}) → {n_1, ..., n_K}  (top-K eigenvectors of Cov(d))

Step 3: Hard Orthogonal Projection
  f_clean = f - N @ N^T @ f    (N = [n_1, ..., n_K])

Step 4: Mahalanobis Anomaly Scoring
  score(x) = sqrt((x - μ_train)^T @ Σ_train^{-1} @ (x - μ_train))
```

**Task Arithmetic과의 관계**: NSP는 본질적으로 feature space에서의 task arithmetic이다. Task vector(τ = θ_ft - θ_pre)가 weight space에서 task를 제거하듯, shift vector(d = f_shift - f_clean)를 feature에서 빼면 shift를 제거한다. 둘 다 "고차원 공간에서 의미 있는 factor가 선형적으로 인코딩된다"는 가정에 기반.

### 7.2 결과

```
Best Config: Layer 8, K=100, Mahalanobis, CLS+patch_mean L2 normalized
  MVTec AD 2: 83.8% (+29.4pp from 54.4% baseline)
  MVTec AD:   96.3% (-0.2pp)
  Per-category: 87.4% (per-cat nuisance estimation)
```

### 7.3 컴포넌트별 기여 (Ablation)

| 변경 | 독립 기여 | 누적 AD2 | 역할 |
|------|----------|---------|------|
| kNN baseline (L11) | — | 54.4% | 출발점 |
| + NSP K=30 | +13.5pp | 67.9% | **핵심**: nuisance 방향 제거 |
| + CLS + L2 norm | +0.1pp | 68.0% | Feature 표현 |
| + Mahalanobis scoring | +3.4pp | 71.4% | Projected covariance 활용 |
| + Layer 8 | +3.6pp | 75.0% | Robust layer 선택 |
| + K=100 | +8.8pp | **83.8%** | 더 많은 nuisance 방향 제거 |

### 7.4 Linear Hard Projection이 Optimal: 14개 변형 전멸

| 카테고리 | 시도 | 교훈 |
|---------|------|------|
| Soft/partial removal (4개) | SAPP(-6pp), soft(-11pp), signal-only(-25pp), hybrid(-28pp) | **Hard > Soft**: 부분 제거는 항상 해로움 |
| Estimation 변경 (4개) | condition-union, whitened, train-only, adaptive K | Standard PCA가 최적 |
| Scoring/fusion (4개) | dual-space, PCA-whitened Mahal, L8+L11 concat, multi-layer | 단순함이 최적, fusion은 오염 유발 |
| Adaptive gating (2개) | SAPP ω-gated, per-category K | Uniform이 안정적 |

**SAPP (Spectral Anomaly-Preserving Projection)**: Normal covariance의 spectral gap으로 anomaly-sensitive direction을 보호하려는 시도. ω_k = n_k^T Σ^{-1} n_k (anomaly sensitivity). Sigmoid gating으로 ω가 큰 direction을 보존.

**SAPP 결과**: L8에서 77.8%(-6.0pp), L11에서 70.2%(-8.9pp). ω spread ratio(3.3-4.8x)가 좁아 gating 차별력 없음 → 사실상 uniform gating = NSP와 동일하되 regularization 추가 → 성능 하락.

**핵심 교훈**: **복잡성을 추가하면 항상 성능이 하락한다.** 이 문제의 structure가 linear이라는 강한 증거. Task arithmetic 관점에서: shift는 feature space에서 선형적으로 인코딩되어 있으므로, 선형 연산(projection)이 최적이고 비선형 방법은 과적합이나 정보 손실을 유발.

---

## 8. Phase 4: Unpaired 해결 시도 — 5가지 전멸 {#8-phase-4}

NSP는 MVTec AD 2의 paired multi-condition data에 의존 → 1 데이터셋에 locked. "Distribution shift 일반에 대한 해법"이라는 claim과 불일치. 5가지 unpaired 접근을 시도했고, **모두 실패**.

### 8.1 TTNS — Test-time Covariance Shift ❌

**아이디어**: ΔΣ = Σ_test - Σ_train → positive eigenvalue 방향 = nuisance.

**결과**:
| Method | AD2 | AD1 |
|--------|-----|-----|
| TTNS-cov | 50.6% | **30.8%** (붕괴) |
| TTNS-combined | **46.3%** | **24.9%** (최악) |

**실패 원인**: Test batch = normal + anomaly 혼합. Anomaly도 분산을 증가시킴 → ΔΣ의 positive eigenvalue에 anomaly variance가 포함 → 이 방향을 제거하면 anomaly detection 능력 자체가 파괴됨. AD1이 30.8%까지 떨어진 것이 결정적 증거.

**교훈**: "분산 증가 = nuisance"라는 가정이 one-class AD에서 **근본적으로** 틀림.

### 8.2 IFR — Inter-layer Feature Residual ❌

**아이디어**: f_L11 = f_L8 + Δ_{9-11}. L8이 robust, L11이 fragile이면, Δ(deep layer가 추가한 것)가 nuisance를 포함할 것.

**30분 사전 진단으로 NO-GO 확정**:

| 진단 | 결과 | GO 기준 |
|------|------|---------|
| D1: IFR basis vs NSP basis principal angle | **74.2°** (≈직교) | < 60° |
| D2: Oracle shift explained by IFR | **12.6%** | > 30% |

**확인 실험**: IFR K=100으로 residual variance의 94-98%를 capture → AD2 59.5% (baseline과 동일). **Residual variance ≠ shift-relevant variance**.

**실패 원인**: Train normal의 inter-layer residual = **semantic refinement** 방향 (object identity, texture detail). Distribution shift 방향과 거의 직교(74.2°). FM 내부 구조가 외부 shift를 예측하지 못함.

**부수 발견 (D3)**: Δ 방향은 clean↔shifted에서 안정적 (mean 24°, mag ratio 1.00). Deep layer는 **같은 방향으로 정보를 추가**하되, 그 방향이 shift 방향과 다를 뿐. → "Deep layer addition ≠ shift vulnerability". L11이 fragile한 이유는 Δ의 방향이 아니라, **L8의 미세 shift가 L11의 다른 covariance 구조에서 더 problematic하게 작용**하기 때문.

**진단 방법론의 가치**: 30분 진단으로 1-2일의 본 실험을 회피. **Subspace alignment 진단(principal angle + shift explained ratio)은 방법론 사전 검증의 효과적 도구.**

### 8.3 AdaBN — Adaptive Batch Normalization ❌

| Method | AD2 | AD1 |
|--------|-----|-----|
| AdaBN | 58.3% | **78.5%** (-18pp) |

**실패 원인**: 전체 feature 정규화가 anomaly signal도 함께 훼손. Global correction ≠ directional correction. AD에서 필요한 것은 **특정 방향만 선택적으로** 수정하는 것인데, AdaBN은 **모든 방향을 일괄** 정규화.

### 8.4 NN Pseudo-Pairing ❌

**아이디어**: Per-sample level에서 shift를 추정하면 TTNS의 batch-level 문제를 회피할 수 있을 것. Test image의 train NN을 pseudo-pair로 사용 → robust PCA.

**사전 진단 (D0-D2)**: 통과!
- SNR = 0.69 (shift signal이 NN noise 대비 충분)
- D2: Oracle shift explained = **87.6%** (IFR의 12.6%와 대조적)

**본 실험**: ❌ 실패

| Method | AD2 | vs Baseline |
|--------|-----|------------|
| NN Pseudo K=100 rob80 | **55.7%** | **-4.2pp** |
| NN Pseudo K=50 rob80 | 50.6% | -9.3pp |

**실패 원인**: Per-sample pseudo-shift는 normal test image에 대해 양호하나, **PCA aggregation 단계에서 anomaly pseudo-shift가 nuisance basis를 재오염**. 80th percentile robust filtering으로도 불충분 (MVTec AD 2 test의 anomaly ratio ~40-60%).

**핵심 교훈**:
1. **Normal-only 진단 ≠ AD 성능 보장**: 진단은 normal-only 조건에서 87.6% alignment을 보였지만, 실제 AD pipeline의 anomaly 혼합 조건에서는 실패. **진단은 필요 조건이지 충분 조건이 아님.**
2. **Per-sample estimation + batch PCA = 여전히 anomaly contamination**: TTNS의 실패 메커니즘(batch-level anomaly contamination)은 per-sample pseudo-shift를 거쳐도 **PCA 단계에서 재발**한다.

### 8.5 SPAD — Spatial Mean Subtraction ❌

**아이디어**: "Anomaly는 local, shift는 global" → 패치별 shift의 spatial mean을 빼면 global shift 제거.

**사전 진단**: Feature shift의 spatial uniformity = **2.6%**. Shift가 패치마다 다르게 작용.

**결과**: +0.5pp only.

**실패 원인**: Pixel space에서 균일한 조명 변화라도, feature space에서는 각 패치의 semantic content에 따라 DINOv2가 **다르게 인코딩** → spatial mean이 shift의 good proxy가 아님. Attention의 cross-patch interaction도 비균일성에 기여.

### 8.6 수렴하는 결론: 5가지 실패의 공통 패턴

| # | 접근 | Nuisance 추정 소스 | 실패 메커니즘 |
|---|------|------------------|-------------|
| 1 | TTNS | Test batch covariance | Anomaly variance ⊂ ΔΣ eigenspace |
| 2 | IFR | Train layer residuals | Train variation ⊥ shift direction (74.2°) |
| 3 | AdaBN | Test batch statistics | Global normalization → signal destruction |
| 4 | NN Pseudo | Per-sample NN + PCA | PCA aggregation re-contaminates |
| 5 | SPAD | Spatial mean | Feature-space spatial non-uniformity (2.6%) |

**경로별 정리**:
- **Test data 사용** (TTNS, NN Pseudo) → **Anomaly가 추정을 오염**
- **Train data만 사용** (IFR) → **Shift 정보 자체가 부재**
- **FM 내부 구조** (IFR D3) → **External shift를 예측 불가**
- **Global correction** (AdaBN, SPAD) → **Anomaly signal도 파괴**

> **결론: Feature-level에서 nuisance direction을 추정하려면 paired observation(같은 물체, 다른 환경)이 필수.**

---

## 9. Phase 4b: Minimal Calibration — 얼마나 필요한가 {#9-phase-4b}

### 9.1 실험 설계

Paired data가 필수라면, **최소 몇 쌍이면 충분한가?** NSP의 shift vector 수를 N=1부터 full(315)까지 변화.

### 9.2 Saturation Curve

| N (paired samples) | AD2 I-AUROC | Recovery | 실용적 의미 |
|--------------------|------------|----------|------------|
| 0 (baseline) | 59.9% | 0% | No projection |
| 5 | 60.6% | 3% | 미미 |
| 10 | 62.0% | 9% | K > N → 대부분 비활성 |
| 20 | 64.6% | 19% | |
| **50** | **68.0%** | **33%** | 첫 의미 있는 개선 |
| **100** | **75.0%** | **62%** | 실용적 최소선 (~12쌍/category) |
| Full (315) global | 84.4% | 100% | Global oracle |
| Full per-category | **87.4%** | 112% | **Per-cat > global +3pp** |

### 9.3 인사이트

1. **Saturation이 매우 느림**: 100쌍에서도 full의 62%만 회복. "5-10쌍이면 충분"이라는 기대와 불일치.
2. **고차원 nuisance subspace가 원인**: K=100까지 단조 증가, saturation 미도달 → nuisance subspace가 100+ 차원으로 다양. PCA의 sample complexity로 N ≫ K 필요.
3. **Per-category > Global (+3pp)**: 카테고리마다 shift 구조가 다름. Global pooling이 정보를 희석.
4. **Graceful degradation**: N < K이면 effective K = N-1로 자동 축소. Cliff 없는 점진적 하락.

---

## 10. Phase 4c-d: Patch-level 접근 — Spatial Structure 활용 {#10-phase-4cd}

### 10.1 Patch-level NSP

Image-level feature (CLS + patch_mean) 대신 개별 patch token(1369개, 768-dim)으로 NSP 수행.

| Method | AD2 | vs baseline |
|--------|-----|-----------|
| Image Mahalanobis | 58.7% | — |
| Image NSP K=100 | 84.9% | +26.2pp |
| Patch kNN (no proj) | 60.7% | +2.0pp |
| **Patch NSP global K=100** | **69.5%** | **+10.8pp** |

**카테고리별 핵심 발견 (walnuts)**:
- Image NSP: 72.9%
- **Patch NSP: 83.8%** → **+10.9pp 초과!**
- 이유: walnuts의 anomaly가 매우 국소적 → patch-level scoring이 spatial structure를 활용.

**전체 평균이 image NSP보다 낮은 이유**: Scoring 방식의 차이 (kNN vs Mahalanobis). Image NSP는 Mahalanobis(covariance 활용), Patch NSP는 kNN(local distance만).

### 10.2 Spatial Non-uniformity 발견 (SPAD 진단)

Feature shift의 spatial uniformity를 측정: **2.6%**. 이는 pixel space에서 균일한 조명 변화라도 feature space에서는 패치마다 다르게 작용한다는 것을 의미.

| Category | Uniformity | Global norm | Residual norm |
|---------|-----------|------------|--------------|
| sheet_metal | 0.045 (최고) | 0.767 | 3.290 |
| fabric | 0.027 | 0.286 | 1.637 |
| **OVERALL** | **0.026** | — | — |

→ "Global shift를 spatial mean으로 제거"하는 SPAD 접근은 **feature space에서 성립하지 않음**.

---

## 11. Phase 4e: Shift-Robust Scoring — Structural Prior 기반 {#11-phase-4e}

### 11.1 핵심 원리

Feature-level nuisance 제거가 불가능하다면, **scoring-level에서 구조적 prior를 활용**:

```
Distribution shift → 모든 patch score를 elevate (global effect)
Local anomaly → 특정 patch score만 spike (local effect)

Mean aggregation: global elevation이 누적 → false positive
Max aggregation: local spike만 잡아냄 → global elevation 무시
```

### 11.2 Scoring 전략 Ablation (8-shot, no paired data, no shift knowledge)

**Layer 8**:

| Aggregation | none | median_sub | mad_norm | q25_sub |
|------------|------|-----------|---------|---------|
| mean | 53.4% | 58.4% | 58.6% | 60.2% |
| p95 | 59.6% | 58.1% | 53.6% | 58.8% |
| p99 | **62.3%** | 60.9% | 57.2% | 62.5% |
| max | 62.3% | 60.1% | 57.3% | 62.4% |

**Layer 11**:

| Aggregation | none | median_sub | mad_norm | q25_sub |
|------------|------|-----------|---------|---------|
| mean | 54.4% | 64.1% | **64.5%** | 59.3% |
| p95 | 57.8% | 60.8% | 64.0% | 60.1% |
| p99 | 67.2% | 69.5% | 67.4% | 69.6% |
| max | 72.9% | 74.6% | 71.4% | **75.1%** |

**Best: L11 patch max + q25_sub = 75.1% (+22.1pp vs image baseline)**

### 11.3 해석

1. **Aggregation 효과가 지배적**: L11에서 mean(54.4%) → max(72.9%) = +18.5pp. Max가 global shift floor를 자연적으로 무시.
2. **Score normalization은 mean에서 효과적, max에서 제한적**: Max 자체가 이미 shift floor를 무시하므로 추가 normalization의 marginal gain이 작음 (+2.2pp).
3. **L11 > L8**: L11의 patch features가 더 discriminative. Shift floor만 제거하면 L11의 discriminative power가 살아남.
4. **Normalization이 max에서 오히려 해로운 경우**: mad_norm은 max에서 -1.5pp. 과도한 정규화가 anomaly signal도 약화.

### 11.4 한계

**+22.1pp는 기존 기법의 조합이지 새로운 원리가 아니다.** PatchCore가 이미 patch max를 사용하고, AnomalyDINO는 false positive 위험 때문에 max 대신 quantile(0.01)을 사용. 우리의 조합은 engineering이지 연구가 아님.

---

## 12. Few-shot Multi-condition NSP — 효과와 한계 {#12-fewshot-nsp}

### 12.1 Setting

Few-shot FM-AD 프로토콜에 multi-condition calibration을 결합:

```
기존: K장의 reference를 1가지 조건(regular)에서 촬영
제안: 같은 K장을 C가지 조건에서 촬영
  → K × (C-1)개의 shift vector → NSP
```

MVTec AD 2에서 C 조건: regular, overexposed, underexposed, shift_1, shift_2, shift_3 (총 6가지).

### 12.2 결과

| K (shots) | C=1 (baseline) | C=2 | C=3 | **C=6 (all)** | NSP Gain (C=6) |
|-----------|---------------|-----|-----|-------------|----------------|
| 1 | 61.6% | 61.6% | 62.3% | **65.2%** | +5.1pp |
| 2 | 62.0% | 61.9% | 62.6% | **65.9%** | +5.6pp |
| 4 | 72.2% | 72.4% | 73.3% | **82.2%** | **+12.7pp** |
| 8 | 77.0% | 77.1% | 78.0% | **92.1%** | **+18.5pp** |
| 16 | 77.7% | 78.0% | 79.1% | **98.1%** | **+22.3pp** |

### 12.3 핵심 분석

1. **C=6에서 폭발적 효과**: K=8에서 +18.5pp, K=16에서 +22.3pp → 5가지 shift type을 모두 관찰하면 nuisance subspace를 정확히 포착.
2. **C=2,3에서는 미미**: +0.1~1.3pp. 1-2가지 shift type만으로는 nuisance subspace의 일부만 커버.
3. **K 증가에 따라 NSP gain 증가**: 더 많은 paired samples = 더 정확한 nuisance 추정.

### 12.4 Critical Caveat: Cheating 문제

**C=6은 test shift type을 정확히 관찰한 oracle 설정이다.**

```
Calibration conditions = {overexposed, underexposed, shift_1, shift_2, shift_3}
Test conditions = {overexposed, underexposed, shift_1, shift_2, shift_3}  ← 동일!
```

**Test에 등장할 shift type을 calibration에서 정확히 알고 있음** → 일종의 oracle/cheating. C=6에서 92-98%가 나온 것은 "미리 정답을 알고 있을 때"의 성능.

**실제 배포에서는 어떤 shift가 올지 모른다.** 조명 shift만이 아니라, 시점 변화, 포커스, 센서 열화, 배경 변화 등 예측 불가능하게 다양.

**관찰하지 않은 shift는 제거할 수 없다** — 이것은 당연하지만, "method"로 제안하려면 이 한계를 넘어서야 한다.

---

## 13. 이론적 연결: Task Vector와 Shift Vector {#13-task-vector}

### 13.1 구조적 유사성

| | Task Vector | Shift Vector |
|--|------------|-------------|
| 공간 | Weight space (θ) | Feature space (f) |
| 정의 | θ_finetuned - θ_pretrained | f(shifted) - f(clean) |
| 연산 | 빼기 → 능력 제거 | Projection → nuisance 제거 |
| 핵심 가정 | Weight space에서 task가 linear | Feature space에서 shift가 linear |
| Paired 필요 | ✅ (pretrained ↔ finetuned) | ✅ (clean ↔ shifted) |

### 13.2 공통 기반: Linear Representation Hypothesis

두 연결 모두 **"고차원 공간에서 의미 있는 factor가 선형적으로 인코딩된다"**는 가정에 기반.

우리의 실험이 이것을 실증: **14개 nonlinear 변형이 모두 linear hard projection보다 열등** → feature space에서 shift가 선형적으로 인코딩되어 있다.

### 13.3 이 연결의 의미

- **이론적 framing**: NSP를 "feature space의 task arithmetic"으로 설명 → ICLR 독자에게 친숙한 언어
- **Linear optimality의 근거**: Task vector 문헌에서도 linear operation이 surprisingly effective
- **방법론적 한계의 공유**: Task vector도 paired data(pretrained ↔ finetuned)가 필수 → 우리의 impossibility와 일치

---

## 14. 종합 인사이트와 검증된 패턴 {#14-종합-인사이트}

### 14.1 검증된 패턴 15개

| # | 패턴 | 함의 |
|---|------|------|
| 1 | FM Invariance ≠ AD Robustness | Contrastive invariance가 fine-grained 정보 파괴 |
| 2 | MVTec AD 성능 ≠ Robustness | 99%+ → AD2에서 60% 추락 |
| 3 | Layer 8 = V-shape sweet spot | DINOv2 robustness는 비단조, L8만 robust |
| 4 | Feature disentanglement = 최대 novelty gap | FM-AD에서 분리한 연구 0건 |
| 5 | Linear hard projection이 optimal | 14 nonlinear 변형 전멸 |
| 6 | Unpaired nuisance estimation 구조적 불가 | 5가지 독립 경로 전멸 |
| 7 | Per-category > Global NSP (+3pp) | Category-specific shift 존재 |
| 8 | Nuisance subspace 고차원 (≥100) | Minimal calibration에 한계, saturation 느림 |
| 9 | Shift는 feature space에서 spatially non-uniform (2.6%) | Spatial mean subtraction 무효 |
| 10 | Patch-level max가 image-level보다 shift-robust | "Anomaly=local, shift=global" 원리 |
| 11 | Score normalization (q25_sub)이 L11에서 효과적 | Shift floor 제거, +2.2pp |
| 12 | Normal-only 진단 ≠ AD 성능 보장 | NN Pseudo: D0 pass → 본 실험 fail |
| 13 | 진단 실험으로 사전 검증 → 시간 절약 | IFR: 30분 진단으로 1-2일 절약 |
| 14 | Few-shot NSP는 C=6(oracle)에서만 강력 | Shift type 사전 지식 없으면 +0.1~1.3pp |
| 15 | Deep layer addition ≠ shift vulnerability | IFR D3: Δ 안정적이나 shift와 무관 |

### 14.2 주요 수치 정리

| 측정 | 값 | 의미 |
|------|---|------|
| FM-AD shift 하락 | -25~34pp | 문제의 심각성 |
| Layer 8 entanglement | 0.12 | 자연 분리 (예외적) |
| Layer 11 entanglement | 0.53 | 심각한 entanglement |
| NSP best (paired, full-shot) | 87.4% (+27.5pp) | Oracle upper bound |
| NSP best (paired, 8-shot, C=6) | 92.1% (+18.5pp) | Few-shot oracle |
| Unpaired best (patch max L11) | 75.1% (+22.1pp) | Shift-agnostic best |
| Spatial uniformity | 2.6% | Feature-space shift는 non-uniform |
| IFR vs NSP angle | 74.2° | Train residual ⊥ shift direction |
| NN Pseudo D0 alignment | 87.6% | 진단 pass → 본 실험 fail |
| Minimal cal 100쌍 | 75.0% (62% recovery) | Saturation 느림 |

---

## 15. 열린 문제와 향후 연구 방향 {#15-열린-문제}

### 15.1 핵심 미해결 문제

> **"관찰하지 않은 distribution shift에 대해 FM-AD를 robust하게 만드는 원리적 방법은 무엇인가?"**

현재 status:
- **Shift를 관찰하면 해결**: NSP +29pp (하지만 paired 필수, shift type 사전 지식 필수)
- **Shift를 모르면**: Patch-level scoring +22pp (하지만 기존 기법 조합, novelty 부족)
- **Feature-level unpaired 제거**: 구조적으로 불가능 (5가지 증명)

### 15.2 가능한 연구 방향

**방향 1: Feature 추출 방식 변경** (Backbone 내부 활용)
- L8이 왜 robust한지 이해 → 더 나은 추출 원리 도출
- Cross-layer combination: L8의 robustness + L11의 discriminability
- Token/head 선택: shift에 덜 민감한 구성 요소 선별

**방향 2: Feature-level 보정** (paired 필요)
- NSP + task vector transfer: 한 번 학습한 shift vector를 재사용 가능?
- Minimal calibration의 효율 개선 (adaptive K, per-category 등)

**방향 3: Scoring-level robustness** (unpaired)
- "Anomaly=local, shift=global" 원리에서 genuinely novel한 method 도출
- Per-image self-referential scoring
- Distribution-free scoring

**방향 4: Image-level 전처리**
- Pixel-space에서 photometric shift 제거 후 feature extraction
- 시점/구조적 shift에는 적용 불가

**방향 5: Representation 학습** (adapter/projection head)
- Self-supervised objective로 shift-invariant representation 학습
- Few-shot에서 adapter 학습 가능한가?
- 어떤 objective? Augmentation = domain-specific 문제

### 15.3 Robustness ↔ Discriminability 딜레마의 해결 가능성

이 딜레마를 해결하는 것이 궁극적 목표:
- L8 수준의 robustness (entanglement 0.12)
- L11 수준의 discriminability (풍부한 texture/structure 인코딩)
- Shift type 사전 지식 불필요

현재까지의 탐색으로 이것이 **feature-level manipulation으로는 어렵다**는 강한 증거를 확보. **다른 수준의 개입**(scoring, representation learning, image preprocessing)이 필요할 수 있음.

---

## 16. 참고 문헌 {#16-참고-문헌}

### Tier 1: 정독 필수

1. MVTec AD 2 (2025) — 핵심 벤치마크, AU-PRO(0.05) 정의
2. SuperAD (2025) — FM-AD 대규모 비교, robustness 순위
3. RoBiS (2025) — DINOv2 robustness 정량 평가
4. RobustAD (2024) — 9종 domain shift 벤치마크
5. Phi-eat (2025) — DINOv2 semantic dominance 실증
6. SINDER (ECCV 2024) — High-norm token artifact
7. PISCO (ICML 2023) — Post-hoc linear decomposition
8. von Kügelgen (NeurIPS 2021) — Nonlinear ICA identifiability
9. FiCo (AAAI 2025) — Filter-compensate domain adaptation
10. ResAD (NeurIPS 2024) — Residual feature

### Tier 2: 핵심 개념 참조

Dinomaly (CVPR 2025), AnomalyCLIP (ICLR 2024), PatchCore (CVPR 2022), EfficientAD (WACV 2024), AnomalyDINO (WACV 2025), StyLIP, CausalCLIP, DINOv2 Registers (ICLR 2024), PILOT (BMVC 2025), WinCLIP (CVPR 2023), PromptAD (CVPR 2024), Task Arithmetic (ICLR 2023)

### Tier 3: 이론적 배경

Self-distillation 이론 (DINO, BYOL), Minimal sufficient representation (Tian et al., NeurIPS 2020), Masked image modeling (iBOT, MAE), Real-IAD Variety (2025), MMAD (ICLR 2025)

---

## 부록: 실험 환경

```
Hardware: NVIDIA RTX 4090 (24GB VRAM)
Runtime: Docker (Project_LG_2nd, PyTorch 2.9.1)
Backbone: DINOv2 ViT-B/14 (frozen, 12 layers, dim=768)
Feature: CLS + patch_mean, L2 normalized (dim=1536 for image-level)
         or raw patch tokens (dim=768 × 1369 patches for patch-level)
CUDA: 12.8, Driver: 570.133.07
총 실험: 40+개 (14 keep, 26+ discard)
연구 기간: 2026-03-23 ~ 2026-03-30
```
