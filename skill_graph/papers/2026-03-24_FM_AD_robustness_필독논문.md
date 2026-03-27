# FM-AD Robustness 연구 — 필독 논문 가이드

작성일: 2026-03-24
목적: 연구 진행 시 실제로 정독해야 할 논문을 역할별/우선순위별 정리

---

## 읽기 전략

논문을 3개 그룹으로 나눔:
- **Tier 1 (정독)**: 방법론 설계와 논문 작성에 직접 영향. 전문 + 실험 + 부록까지 읽기
- **Tier 2 (핵심 파악)**: 관련 기법/벤치마크 이해. Abstract + Method + Key Results 중심
- **Tier 3 (참조)**: Related work 작성 또는 실험 비교 시 참조. Abstract + Tables 중심

---

## Tier 1: 정독 (10편)

### 1-A. 우리 논문의 핵심 주장을 뒷받침할 증거

#### ① MVTec AD 2 Dataset (arXiv:2503.21622)
- **왜 읽는가**: 우리 주요 벤치마크. 데이터 구조, 평가 프로토콜(AU-PRO 0.05), multi-lighting 설계를 정확히 이해해야 함
- **핵심 확인**: 3-tier test set (TESTpriv vs TESTpriv,mix), 카테고리별 도전 요소, baseline 결과
- **주의**: 평가 서버 2026-04-01 재오픈 예정 (VAND 4.0)

#### ② SuperAD (arXiv:2505.19750)
- **왜 읽는가**: MVTec AD 2에서 training-free DINOv2 memory bank의 robustness 증거. DINOv2가 학습 기반보다 조명 변화에 더 robust하다는 우리 Lesson #6의 직접 근거
- **핵심 확인**: TESTpriv→mix 하락 폭 (3.5%), per-category 결과, Can/Wallplugs 실패 분석

#### ③ RoBiS (arXiv:2505.21152)
- **왜 읽는가**: Augmentation으로 robustness를 시도했지만 오히려 더 큰 하락(11.3%)을 보인 사례. "Augmentation이 domain-specific bias를 도입할 수 있다"는 반직관적 결과
- **핵심 확인**: Augmentation 전략, MVTec AD 2 전체 결과, SuperAD와의 비교

#### ④ RobustAD (CVPR-W 2025, Amazon Science)
- **왜 읽는가**: Domain shift 전용 벤치마크. ARD(Average Relative Drop) metric 설계 이해
- **HuggingFace**: AmazonScience/RobustAD
- **핵심 확인**: 9 anomaly types × 8 test domains, shift type별 method robustness 차이

---

### 1-B. 메커니즘 이해 (왜 FM feature가 실패하는가)

#### ⑤ Phi-eat: Physically-Grounded Feature Representation (arXiv:2511.11270)
- **왜 읽는가**: DINOv2 feature가 semantic grouping을 하지 material/texture grouping을 하지 않는다는 **가장 직접적인 실험 증거**. "Semantic dominance" 메커니즘의 핵심 참조
- **핵심 확인**: DINOv2 vs Phi-eat의 clustering 비교 (Table/Figure), material IoU 차이 (0.566 vs 0.776)
- **우리 논문에서의 역할**: "FM feature가 AD에 부적합한 이유"의 핵심 증거

#### ⑥ SINDER: Repairing the Singular Defects of DINOv2 (ECCV 2024 Oral, arXiv:2407.16826)
- **왜 읽는가**: DINOv2의 high-norm token artifact (434 vs 57.6)의 원인과 구조. 이 artifact이 distance-based AD scoring을 왜곡할 가능성
- **핵심 확인**: Leading left singular vector 분석, layer별 artifact 발생 패턴, SINDER fix의 효과
- **우리 논문에서의 역할**: FM feature의 구조적 결함 증거 + SINDER 전처리가 AD robustness에 미치는 영향 실험 가능

---

### 1-C. 해결 방법론의 이론적 기반

#### ⑦ PISCO: Simple Disentanglement of Style and Content (ICML 2023)
- **왜 읽는가**: **Post-hoc linear disentanglement** — frozen FM feature에 직접 적용 가능한 유일한 provable 방법. 우리 방법론의 직접적 출발점
- **핵심 확인**: Linear mixing assumption `f = W_c * z_c + W_s * z_s`, content/style 분리 알고리즘, DG 결과
- **핵심 질문**: DINOv2 feature에 linear assumption이 유효한가?

#### ⑧ Self-Supervised Learning Provably Isolates Content from Style (NeurIPS 2021, von Kugelgen et al.)
- **왜 읽는가**: SSL feature의 content/style 분리에 대한 **이론적 identifiability guarantee**. Augmentation이 partition을 정의한다는 핵심 정리
- **핵심 확인**: Block identifiability 정리, augmentation → partition 관계, 실용적 한계
- **우리 논문에서의 역할**: "이론적으로 FM feature에서 style/content 분리가 가능하다"는 근거

#### ⑨ FiCo: Filter or Compensate (AAAI 2025, arXiv:2412.10115)
- **왜 읽는가**: AD에서 distribution-invariant feature를 추출하려는 **가장 직접적인 선행 연구**. 우리 논문의 핵심 비교 대상
- **핵심 확인**: DiSCo/DiIFi 구조, MVTec OOD corruption 결과, 이론적 근거(or 부재)
- **우리 방법과의 차이**: FiCo는 implicit separation + 특정 architecture 의존 / 우리는 principled disentanglement + architecture-agnostic (가능하다면)

#### ⑩ ResAD: A Simple Framework for Class Generalizable AD (NeurIPS 2024 Spotlight)
- **왜 읽는가**: Residual feature로 class identity를 제거하는 **implicit disentanglement**. Multi-class scaling 문제의 핵심 비교 대상
- **핵심 확인**: Feature Converter (residual 계산), Feature Constraintor (hypersphere), generalist AD 성능
- **우리 방법과의 차이**: ResAD는 inter-class variation 제거 / 우리는 intra-class nuisance variation(lighting, pose) 제거

---

## Tier 2: 핵심 파악 (12편)

### 2-A. FM-AD Baseline Methods (비교 실험 대상)

| # | 논문 | Venue | 읽는 이유 |
|---|------|-------|----------|
| 11 | **Dinomaly** | CVPR 2025 | DINOv2 기반 SOTA. 4 essentials 설계 이해. Multi-class AD 비교 대상 |
| 12 | **AnomalyCLIP** | ICLR 2024 | CLIP-AD의 기준점. Object-agnostic prompt의 robustness 한계 확인 |
| 13 | **PatchCore** | CVPR 2022 | 근본 baseline. Memory bank의 robustness 특성 (상대적으로 robust) |
| 14 | **EfficientAD** | — | Teacher-student 기반. Robustness 취약 (>10pp 하락). 비교 대상 |
| 15 | **AnomalyDINO** | WACV 2025 | DINOv2 few-shot AD. 1-shot 한계, rotation 문제 |

### 2-B. Disentanglement / Feature 분석

| # | 논문 | Venue | 읽는 이유 |
|---|------|-------|----------|
| 16 | **StyLIP** | WACV 2024 | CLIP ViT에서 multi-scale style/content 분리. DG 성공 사례 |
| 17 | **CausalCLIP** | arXiv 2512.13285 | CLIP feature의 causal/non-causal factorization. HSIC constraint |
| 18 | **Vision Transformers Need Registers** | ICLR 2024 | DINOv2 artifact의 대안 해결책. DINOv2-reg 사용 여부 결정 |
| 19 | **AD-DINOv3** | arXiv 2025 | CLS token bias, multi-layer analysis, layer별 특성 |

### 2-C. Robustness / Domain Adaptation

| # | 논문 | Venue | 읽는 이유 |
|---|------|-------|----------|
| 20 | **PILOT** | BMVC 2025 | TTA for zero-shot AD (prompt adaptation). 유일한 TTA-AD 논문 |
| 21 | **ROADS** | WACV 2025 | Robust prompt-driven multi-class AD under domain shift |
| 22 | **Closer Look at CLIP Robustness** | NeurIPS 2023 | FM robustness 분석 논문의 모범 사례. 실험 설계 참고 |

---

## Tier 3: 참조 (8편)

### 3-A. 이론

| # | 논문 | 읽는 이유 |
|---|------|----------|
| 23 | **Contrastive Learning Inverts DGP** (Zimmermann, ICML 2021) | InfoNCE ↔ nonlinear ICA 연결 |
| 24 | **What Makes Good Views for CL** (Tian, NeurIPS 2020) | InfoMin principle, minimal sufficient representation |
| 25 | **Rethinking Minimal Sufficient Representation in CL** (Wang, CVPR 2022) | Task-relevant info 소실 증명 |

### 3-B. 벤치마크/데이터

| # | 논문 | 읽는 이유 |
|---|------|----------|
| 26 | **Real-IAD Variety** (arXiv:2511.00540) | 160 cat 스케일링 데이터. 실험에 포함 여부 결정 |
| 27 | **MIRAD** (arXiv:2510.16370) | Individualized manufacturing robustness |
| 28 | **Beyond Academic Benchmarks** (CVPR-W 2025) | Lab-production gap 인용 |

### 3-C. ICLR 프레이밍 참고

| # | 논문 | 읽는 이유 |
|---|------|----------|
| 29 | **MMAD** (ICLR 2025) | "FM 한계 노출" 논문의 ICLR 수락 사례 |
| 30 | **HOOD** (ICLR 2023) | Causal content/style for OOD. AD 확장 가능성 |

---

## 추천 읽기 순서

### Phase 1: 문제 정의 (1-2일)
```
① MVTec AD 2 → ② SuperAD → ③ RoBiS → ④ RobustAD
```
**목표**: "무엇이 실패하는가"의 정량적 증거를 내 손으로 확인

### Phase 2: 메커니즘 이해 (1-2일)
```
⑤ Phi-eat → ⑥ SINDER → ⑧ von Kugelgen (NeurIPS 2021)
```
**목표**: "왜 실패하는가"의 이론적/실험적 근거 확보

### Phase 3: 해결 방향 (2-3일)
```
⑦ PISCO → ⑨ FiCo → ⑩ ResAD → 16 StyLIP → 17 CausalCLIP
```
**목표**: 기존 방법의 장단점 파악, 우리 방법의 차별점 정의

### Phase 4: 비교 대상 & 포지셔닝 (1-2일)
```
11 Dinomaly → 12 AnomalyCLIP → 13 PatchCore → 22 Closer Look at CLIP → 29 MMAD
```
**목표**: 실험 비교 대상 선정, 논문 프레이밍 확정

---

## 논문별 핵심 질문 체크리스트

읽을 때 아래 질문에 답할 수 있어야 함:

### 문제 정의 논문 (①-④)
- [ ] 어떤 shift type에서 어떤 method가 얼마나 떨어지는가? (정량)
- [ ] FM-based method vs classical method의 robustness 차이가 있는가?
- [ ] Per-category 분석에서 공통 실패 패턴이 보이는가?

### 메커니즘 논문 (⑤-⑥, ⑧)
- [ ] FM feature space의 어떤 구조가 AD 실패를 유발하는가?
- [ ] 이 구조가 post-hoc으로 수정 가능한가, 아니면 재학습이 필요한가?
- [ ] 우리 claim의 이론적 근거로 직접 인용 가능한 정리/결과는?

### 해결 방향 논문 (⑦, ⑨-⑩)
- [ ] 이 방법이 FM-AD에 직접 적용 가능한가? 불가능하면 무엇을 바꿔야 하는가?
- [ ] 이 방법의 가정(assumption)이 우리 setting에서 성립하는가?
- [ ] 우리 방법과의 명확한 차별점은 무엇인가?

---

## 관련 노트

- [2026-03-23_FM_AD_robustness_핵심논문.md](2026-03-23_FM_AD_robustness_핵심논문.md) — 전체 논문 목록 (30편)
- [../analysis/fm_ad_robustness/2026-03-23_ICLR_positioning.md](../analysis/fm_ad_robustness/2026-03-23_ICLR_positioning.md) — 포지셔닝 전략
- [../analysis/feature_disentanglement/2026-03-23_deep_survey.md](../analysis/feature_disentanglement/2026-03-23_deep_survey.md) — Disentanglement 심층 서베이
- [../analysis/fm_ad_robustness/2026-03-23_theoretical_foundations.md](../analysis/fm_ad_robustness/2026-03-23_theoretical_foundations.md) — 이론적 기반
