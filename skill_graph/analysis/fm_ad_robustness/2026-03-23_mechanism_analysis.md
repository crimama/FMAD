# FM Feature Robustness 실패 메커니즘 분석

작성일: 2026-03-23
분석 목적: DINOv2/CLIP feature가 AD에서 distribution shift 하에 왜 실패하는지 근본 원인 규명

---

## 핵심 통찰

> **FM을 좋은 범용 feature extractor로 만드는 속성(invariance, semantic abstraction, 대규모 pretraining)이
> 정확히 distribution shift 하에서의 AD를 방해하는 속성이다.**

AD는 미세한 편차에 대한 **민감도**를 요구하지만, FM은 그런 편차에 대한 **둔감도**를 최적화한다.

---

## 메커니즘 1: Invariance-as-Information-Destruction

### 원리
DINOv2는 self-supervised distillation + multi-crop augmentation으로 학습됨. 학습 중 heavy geometric/photometric transformation을 적용하고, 서로 다른 augmented view가 같은 representation을 갖도록 강제.

### AD에서의 문제
| 학습 시 nuisance로 취급되는 변환 | AD에서 anomaly일 수 있는 변환 |
|--------------------------------|----------------------------|
| Color jitter | 변색, 오염 |
| Blur | 코팅 결함, 표면 열화 |
| Geometric distortion | 구조적 손상 |
| Rotation | 부품 오배치 |

**결과**: DINOv2 feature space에서 anomaly-relevant한 appearance 변화가 이미 소실됨

### 이론적 근거
- Contrastive learning theory (arXiv 2007.13916): multiview assumption이 성립하지 않으면, minimal sufficient representation이 task-relevant information을 덜 포함
- AD에서 multiview assumption이 **체계적으로 위반**됨 — color jitter된 view가 "같은 이미지"로 취급되지만, AD에서는 color shift 자체가 anomaly일 수 있음
- 이것은 구현 디테일이 아닌 **근본적 tension**

---

## 메커니즘 2: Semantic Dominance over Material/Texture

### 증거
**Phi-eat** (arXiv 2511.11270, Nov 2025) — 핵심 참조 논문:
- DINOv2/DINOv3 feature는 **semantically/spatially related 영역**을 그룹핑 — 같은 object에 속하는 패치를 묶음 (재질이 달라도)
- Physically-grounded feature (Phi-eat)는 **material boundary에 정렬** — 같은 reflectance/texture를 가진 패치를 묶음
- Phi-eat가 material 관련 메트릭에서 DINOv2를 크게 상회

### 의미
DINOv2 feature의 우선순위:
```
"이것이 무엇인가?" (object identity) >> "이것이 어떻게 보이는가?" (appearance/texture)
```

AD가 필요로 하는 우선순위:
```
"이것의 표면/외관이 정상과 다른가?" (appearance deviation) >> "이것이 무엇인가?"
```

**Feature space의 구조가 AD의 요구와 정반대**

---

## 메커니즘 3: High-Norm Token Artifacts (SINDER)

### 발견
**SINDER** (ECCV 2024, arXiv 2407.16826):
- DINOv2의 특정 patch token이 **비정상적으로 높은 norm** 보유: 평균 434.0 vs 정상 57.6 (**7.5배**)
- 이 defective token들의 방향이 **이미지 독립적**: 500개 이미지에서 평균 pairwise angle 3.1°
- 원인: 네트워크 weight matrix의 **leading left singular vector**에서 기인

### AD에서의 영향
- Distance 기반 anomaly scoring (PatchCore k-NN, PaDiM Mahalanobis 등)이 high-norm token에 의해 **지배됨**
- 환경 변화가 attention pattern을 바꾸면 → 어떤 token이 high-norm artifact가 되는지 재분배 → anomaly와 무관한 score 변동
- **체계적 false positive 또는 real anomaly masking 유발**

---

## 메커니즘 4: Inter-Class Interference (Multi-Class 설정)

### 원리
Unified model이 여러 카테고리를 동시에 처리할 때, feature space가 class 간 **entangle**됨.

### 핵심 논문
- **MINT-AD** (arXiv 2403.14213, 2024): "vanilla unified model은 inter-class 영역으로 reconstructive capability가 일반화되어 compact boundary를 얻을 수 없다"
- **Class-Aware Contrastive Learning** (arXiv 2412.04769, 2024): Multi-class 직접 학습 시 catastrophic forgetting + inter-class confusion

### FM feature에서 해결되지 않는 이유
FM feature는 category-level semantic 정보를 강하게 인코딩 → 15개 카테고리의 normal distribution을 만들면:
```
Inter-category variance (screw vs bottle vs cable) >> Intra-category anomaly signal (scratched screw vs normal screw)
```

**카테고리 수가 증가하면 inter-category variance가 feature space를 지배**, 미세한 intra-category anomaly signal이 noise에 파묻힘

### 해결 시도
- **UniMMAD** (arXiv 2509.25934): MoE로 heterogeneous input 분리
- **ROADS** (WACV 2025): Dynamic class-specific prompts
- **Contrastive separation**: Within-class align + between-class push

---

## 메커니즘 5: Nuisance-Semantic Entanglement

### 핵심 문제
FM feature가 **high-level semantics와 low-level physical factors를 entangle**하여 인코딩.

### 증거
- **Phi-eat** (2025): Self-supervised feature가 geometry/illumination 정보를 semantics와 혼합
- Lighting 변화가 semantic dimension도 함께 shift시킴
- **조명 변화로 인한 feature shift와 실제 anomaly로 인한 feature shift가 직교하지 않음**

### 결과
```
Feature distance = f(anomaly signal) + f(domain shift signal)
```
이 두 신호가 entangle되어 있으므로:
- Simple threshold로는 "domain shift"와 "anomaly"를 구분 불가
- Domain shift가 큰 환경에서 anomaly detection의 precision이 급락

---

## 메커니즘 6: Pretraining Distribution Mismatch

### 원리
FM Pretraining에 사용된 데이터셋 (ImageNet 규모)에 **산업 결함 패턴이 극히 희소**:
- Scratch, dent, contamination, discoloration → web-crawled data에서 거의 등장하지 않음
- Feature space가 결함 관련 영역에서 **저해상도** (poor resolution)

### 증거
- Foundation Models for AD Survey (arXiv 2502.06911, 2025): "FM representations underperform when upstream data distribution misaligns with downstream requirements, especially for rare classes"
- ADPretrain (NeurIPS 2025): AD 전용 pretraining으로 이 gap을 줄이려는 시도

### AD-specific 문제
- MVTec AD에서의 성능은 "유사한 object가 ImageNet에 있어서" 우연히 잘 되는 것일 수 있음
- **진정한 domain gap이 있는 산업 환경** (반도체 웨이퍼, 투명 바이알 등)에서는 feature quality 급락

---

## 메커니즘 7: Token-Level Distribution Shift Accumulation

### 원리
FM feature는 patch 단위(token)로 계산됨. 각 token의 feature는 local context + global context(attention) 모두에 의존.

### 환경 변화 시:
1. 각 token의 feature vector가 local context 변화로 **소폭 shift**
2. Attention을 통해 global context도 변화 → shift 증폭
3. 정상 feature distribution (memory bank, Gaussian)은 학습 조건에서 추정됨
4. Test 시 token별 소폭 shift가 spatial grid 전체에서 **누적** → 정상 분포로부터 체계적 이탈

**결과**: Domain shift에 의한 feature deviation이 anomaly signal과 구별 불가

---

## 종합: 실패 메커니즘 계층 구조

```
┌─────────────────────────────────────────────────────┐
│              Pretraining Distribution Mismatch        │  ← 근본 원인 (Layer 0)
│   (산업 결함이 학습 데이터에 부재)                        │
├─────────────────────────────────────────────────────┤
│  Invariance-as-          Semantic Dominance           │  ← 설계 원인 (Layer 1)
│  Information-Destruction  (what > how it looks)       │
├─────────────────────────────────────────────────────┤
│  Nuisance-Semantic    High-Norm Token    Inter-Class  │  ← 구조적 결함 (Layer 2)
│  Entanglement         Artifacts (SINDER)  Interference│
├─────────────────────────────────────────────────────┤
│  Token-Level Distribution Shift Accumulation          │  ← 발현 (Layer 3)
│  → False Positives, Missed Anomalies, Score Drift     │
└─────────────────────────────────────────────────────┘
```

---

## 논문 Contribution으로의 연결

이 메커니즘 분석이 논문에서 할 수 있는 기여:
1. **"FM feature가 AD에 robust하다"는 가정을 체계적으로 반박** — 메커니즘별 증거 제시
2. **실패의 taxonomy 제공** — 어떤 조건에서 어떤 메커니즘이 지배적인지 분류
3. **해결 방향 제시** — 각 메커니즘에 대응하는 원리적 해결책 도출

가장 설명력 있는 핵심 메커니즘 후보:
- **Invariance-Information Destruction** + **Nuisance-Semantic Entanglement** 조합
- 이론적으로 cleanㅎ고, 실험적으로 검증 가능하며, 해결책으로 자연스럽게 이어짐

---

## 관련 노트

- [2026-03-23_failure_survey.md](2026-03-23_failure_survey.md) — 실패 사례 종합
- [2026-03-23_solution_survey.md](2026-03-23_solution_survey.md) — 해결 방안 서베이
- [2026-03-23_ICLR_positioning.md](2026-03-23_ICLR_positioning.md) — 논문 포지셔닝
