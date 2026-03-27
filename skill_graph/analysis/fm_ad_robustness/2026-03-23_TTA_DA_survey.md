# TTA & Domain Adaptation for FM-based Anomaly Detection — Deep Survey

> 작성일: 2026-03-23
> 목적: Test-Time Adaptation (TTA) 및 Domain Adaptation 방법론의 FM-AD robustness 적용 가능성 조사
> 분류: analysis / survey

---

## Part 1: TTA for Foundation Models (General)

### 1.1 TENT — Fully Test-Time Adaptation by Entropy Minimization (ICLR 2021)

**Core Method**: 테스트 배치의 prediction entropy를 최소화하여 모델을 적응시킨다. BatchNorm layer의 statistics와 channel-wise affine parameters만 업데이트하며, 배치당 한 번의 update로 inference를 거의 방해하지 않는다.

**Key Results**: ImageNet-C에서 SOTA error 달성. CIFAR-10/100-C에서도 일관된 개선.

**Strengths**:
- Source data 불필요 (fully test-time)
- 극히 경량: BN affine parameters만 업데이트
- 어떤 모델에든 plug-and-play 가능

**Weaknesses**:
- Entropy minimization은 noisy/ambiguous sample에 취약 — confident but wrong prediction으로 수렴 가능
- BN statistics가 batch size에 민감
- Long-term adaptation 시 error accumulation (catastrophic adaptation)

**FM-AD 연결**: AD에서는 classification entropy가 정의되지 않으므로 직접 적용 불가. 하지만 "anomaly score의 sharpness"를 entropy 대체 목표로 설계할 수 있음. BN update 아이디어는 FM feature extractor의 normalization layer adaptation에 활용 가능.

---

### 1.2 TTT — Test-Time Training with Self-Supervision (ICML 2020)

**Core Method**: Y-shaped 아키텍처에서 main task (classification)와 auxiliary self-supervised task (rotation prediction)가 shared feature를 사용. 테스트 시 auxiliary task의 loss로 shared encoder를 업데이트하여 test distribution에 적응.

**Key Results**: CIFAR-10-C에서 baseline 대비 significant improvement. TTT-Online (streaming adaptation) 추가 개선.

**Strengths**:
- Self-supervised signal이 label 없이도 의미 있는 adaptation 제공
- Training과 test 모두에서 일관된 프레임워크

**Weaknesses**:
- Training 시 auxiliary task를 함께 학습해야 함 (training pipeline 수정 필요)
- Rotation prediction이 모든 domain shift에 유효하지 않음
- FM에 적용하려면 pretrained model을 재학습해야 하는 문제

**FM-AD 연결**: AD에서는 reconstruction, jigsaw, contrastive prediction 등이 auxiliary task 후보. 핵심 insight는 "test sample 자체에서 학습 signal을 추출"하는 것. FM feature를 고정하고 lightweight adapter를 auxiliary task로 학습하는 변형이 유망.

---

### 1.3 TPT — Test-Time Prompt Tuning (NeurIPS 2022)

**Core Method**: CLIP의 text prompt를 테스트 시 최적화. 단일 test image의 여러 augmented view에서 prediction이 일관되도록 entropy를 최소화. Confidence selection으로 noisy augmentation 필터링.

**Key Results**: CLIP zero-shot top-1 accuracy +3.6% 평균, ImageNet-A에서 최대 +6.9%.

**Strengths**:
- Training data 불필요 — 순수 test-time
- CLIP의 text-image alignment을 활용한 prompt 최적화
- 단일 sample에서도 동작

**Weaknesses**:
- Prompt space의 최적화는 local minima에 빠지기 쉬움
- Entropy-based loss에 의존 — classification setting 한정
- 다수의 augmented view 필요 → inference 비용 증가

**FM-AD 연결**: AD에서 text prompt를 "normal"/"anomalous" description으로 사용하는 zero-shot AD (AnomalyCLIP, WinCLIP 등)에 직접 확장 가능. 하지만 AD에서 entropy의 의미가 다름 — "normal vs anomalous"의 이진 결정에 대한 confidence 최적화가 필요.

---

### 1.4 LoRA-TTT — Low-Rank Test-Time Training for VLMs (2025)

**Core Method**: CLIP의 **image encoder**에만 LoRA를 삽입하고, test-time에 reconstruction loss로 LoRA parameters만 업데이트. Text prompt가 아닌 visual feature 자체를 적응시키는 접근.

**Key Results**: CLIP-ViT-B/16 기준 OOD benchmark +5.79%, fine-grained benchmark +1.36%.

**Strengths**:
- Prompt tuning 대비 더 직접적인 feature adaptation
- LoRA의 low-rank constraint가 과적합 방지
- Memory/runtime overhead 최소

**Weaknesses**:
- Reconstruction loss의 설계가 task-specific
- Image encoder만 적응 — text-image alignment 변화 가능성
- Classification 기준 평가 — AD task에서의 효과 미검증

**FM-AD 연결**: **매우 유망**. AD는 주로 visual feature quality에 의존하므로 image encoder adaptation이 핵심. LoRA의 low-rank constraint는 pretrained feature의 catastrophic forgetting 방지에 적합. AD-specific reconstruction loss (e.g., normal feature reconstruction, patch-level consistency) 설계가 필요.

---

### 1.5 FoCAL — Test-Time Canonicalization by FM Priors (ICML 2025)

**Core Method**: Mental rotation에서 영감. 테스트 시 FM (CLIP/SAM)의 likelihood를 기준으로 input image의 여러 transformation 중 가장 "canonical"한 (typical한) view를 선택. 모델 재학습 없이 input 자체를 정규화.

**Key Results**: 2D/3D rotation, illumination shift, day-night variation에서 CLIP/SAM robustness 개선.

**Strengths**:
- Model-agnostic — 어떤 FM이든 적용 가능
- Training 불필요, input space에서만 동작
- 직관적 원리 (canonicalization)

**Weaknesses**:
- Transformation search space 설계가 domain-specific
- FM의 likelihood가 반드시 AD에 유용한 방향을 가리키지 않을 수 있음
- Inference 비용: 여러 candidate transformation 평가 필요

**FM-AD 연결**: AD에서 "canonical view"의 개념이 의미가 있는가? Normal sample에 대해서는 canonical view가 존재할 수 있으나, anomaly의 존재 자체가 "비전형적"이므로 canonicalization이 anomaly signal을 약화시킬 위험. 오히려 **normal training image를 canonicalize**하여 memory bank의 일관성을 높이는 방향이 더 적합할 수 있음.

---

### 1.6 ProtoDCS — Prototype-based Open-Set TTA for VLMs (2026)

**Core Method**: Open-set test-time adaptation에서 covariate-shifted ID와 OOD sample을 GMM 기반 double-check mechanism으로 분리. ID sample만으로 prototype-level update를 수행하며, uncertainty-aware loss로 overconfidence 방지.

**Key Results**: CIFAR-10/100-C, Tiny-ImageNet-C에서 SOTA. Known-class accuracy와 OOD detection 동시 개선.

**Strengths**:
- Open-set 가정 — 현실적 시나리오
- Prototype-level update → sample-level보다 안정적
- GMM separation이 임계값 방식보다 robust

**Weaknesses**:
- GMM fitting이 소수 sample에서 불안정할 수 있음
- AD 특유의 one-class 설정과 다름 (multi-class 분류 전제)

**FM-AD 연결**: AD에서의 핵심 과제인 "normal의 distribution shift vs actual anomaly" 구분과 직접 연결. ProtoDCS의 ID/OOD 분리 메커니즘을 "shifted-normal vs anomaly" 분리로 재해석할 수 있음. Prototype-based update는 memory bank 기반 AD (PatchCore 등)와 자연스럽게 결합 가능.

---

### 1.7 Summary: FM TTA의 주요 패러다임

| 패러다임 | 대표 방법 | 업데이트 대상 | AD 적합성 |
|---------|----------|-------------|----------|
| Entropy minimization | TENT, TPT | BN params / prompts | 낮음 (classification entropy 필요) |
| Self-supervised auxiliary | TTT | Shared encoder | 중간 (auxiliary task 설계 필요) |
| Prompt tuning | TPT, CLIPArTT | Text prompts | 중간 (AD prompt 특수성) |
| Visual adapter | LoRA-TTT | LoRA in image encoder | **높음** (feature quality 직접 개선) |
| Input canonicalization | FoCAL | Input image | 조건적 (anomaly signal 보존 필요) |
| Prototype update | ProtoDCS | Class prototypes | **높음** (memory bank과 자연스럽게 결합) |

---

## Part 2: TTA for Anomaly Detection (Sparse but Growing)

### 2.1 PILOT — Zero-Shot AD with Dual-Branch Prompt Selection (BMVC 2025)

**Core Method**: Learnable prompt pool + attribute memory bank의 dual-branch 구조. (1) Learnable prompt pool이 다양한 anomaly 유형에 적응적 가중치 부여, (2) Attribute memory bank이 고정된 semantic anchor (normal/anomalous state 설명) 제공. **Label-free TTA**: 고신뢰 pseudo-label로 learnable prompt parameters 업데이트.

**Key Results**: 13개 industrial + medical benchmark에서 SOTA. Domain shift 조건에서 기존 ZSAD 대비 유의미한 개선.

**Strengths**:
- AD-specific TTA의 최초 체계적 시도 중 하나
- Dual-branch 구조가 overfitting 방지
- Attribute memory bank이 semantic stability 유지

**Weaknesses**:
- Prompt-level adaptation만 — feature extractor는 고정
- Pseudo-label quality에 의존 — contaminated test batch에서 위험
- Text prompt 최적화의 한계 (visual domain gap은 직접 해소하지 않음)

**FM-AD 연결**: PILOT은 "prompt만 적응"이라는 제한이 있음 (`_lessons.md` 미검증 항목과 일치). Visual feature adapter + prompt의 joint adaptation이 다음 단계. 또한 prompt pool의 크기와 diversity가 성능을 좌우 — 이 부분의 원리적 분석이 부족.

---

### 2.2 Test-Time Training for Industrial Anomaly Segmentation (CVPRW 2024, Costanzino)

**Core Method**: Anomaly detection은 unsupervised라 test 시 anomaly label이 없는데, test-time에 anomalous sample의 rich feature를 활용해 classifier를 학습. Anomaly map을 binarization하는 threshold를 test-time에 최적화하는 전략.

**Key Results**: Standard threshold 방식 대비 segmentation 성능 개선.

**Strengths**:
- AD-specific TTT의 선구적 시도
- Anomaly sample 자체를 활용한다는 실용적 접근

**Weaknesses**:
- Threshold 최적화에 초점 — 근본적 feature adaptation이 아님
- Test에 anomaly가 있다는 가정 필요

**FM-AD 연결**: Threshold calibration은 post-hoc 접근. Feature-level adaptation과 결합하면 더 강력.

---

### 2.3 EPHAD — Evidence-Based Post-Hoc Adjustment (NeurIPS 2025)

**Core Method**: 오염된 training data로 학습된 AD model의 output을 test-time에 보정. **Exponential tilting**으로 AD model의 prior와 auxiliary evidence function (CLIP, LOF 등)의 output을 결합. Training pipeline/data 접근 불필요.

**Key Results**: 8개 visual AD dataset, 26개 tabular AD dataset, 1개 real-world industrial dataset에서 CFLOW, PaDiM, RD 등의 성능 향상. CLIP evidence가 단독으로 best가 아닐 때도 EPHAD 프레임워크 내에서 유효.

**Strengths**:
- Model-agnostic, plug-and-play
- Training pipeline 접근 불필요 — 실용성 높음
- Exponential tilting의 이론적 근거 명확

**Weaknesses**:
- "Evidence function"의 quality가 성능 상한 결정
- Domain shift 자체를 해결하는 것이 아니라 output 보정
- Feature-level adaptation이 아님 — fine-grained anomaly에 한계

**FM-AD 연결**: EPHAD는 "FM을 evidence source로 활용"하는 패러다임. 우리 연구와의 차이: EPHAD는 output-level 보정이고, feature-level adaptation은 다루지 않음. 두 접근의 결합이 가능 — feature adaptation + output calibration.

---

### 2.4 TUNE — Adapting Graph Anomaly Detectors at Test Time (2025)

**Core Method**: Graph Anomaly Detection에서 normality shift 문제를 해결. (1) Semantic confusion: 새로운 normal이 anomaly로 오인, (2) Aggregation contamination: message passing으로 seen normal의 representation이 오염. Graph aligner로 shifted data를 original distribution에 align하고, representation-level shift 최소화를 supervision signal로 사용.

**Key Results**: 10개 real-world dataset에서 pre-trained GAD model의 generalizability 유의미하게 향상.

**Strengths**:
- Lightweight, plug-and-play
- Normality shift의 원인을 두 가지로 분리하여 각각 대응

**Weaknesses**:
- Graph domain 특화 — visual AD로의 직접 전이 어려움
- Graph aggregation contamination은 visual AD에 해당하지 않음

**FM-AD 연결**: "Normality shift" 개념은 visual AD에서도 핵심적. Training에서 본 normal과 test의 normal이 다를 때 false positive 급증. Graph aligner의 아이디어를 **feature space aligner**로 변환 가능 — target domain의 normal feature를 source domain normal distribution에 align.

---

### 2.5 Source-Free TTA for Online Surface-Defect Detection (ICPR 2024)

**Core Method**: Pre-trained model을 inference 시 새로운 domain/class에 적응. (1) Supervisor가 high-confidence sample만 필터링하여 model update, (2) Augmented mean prediction으로 robust pseudo label 생성, (3) Dynamically-balancing loss로 classification과 segmentation 통합.

**Key Results**: Online surface defect detection에서 domain/class 변화에 적응.

**Strengths**:
- Source data 완전 불필요
- Pseudo-label quality를 supervisor로 관리

**Weaknesses**:
- Supervised AD 설정 (classification + segmentation) — unsupervised AD와 다름
- High-confidence filtering이 shift가 심할 때 거의 모든 sample을 거부할 수 있음

**FM-AD 연결**: Supervisor-based filtering은 prototype update에서 noise sample 배제에 활용 가능. Augmented mean prediction은 memory bank update에서 robust estimation으로 확장 가능.

---

### 2.6 Selective TTA for Unsupervised AD using Neural Implicit Representations (MICCAI-W 2024, Best Paper)

**Core Method**: Deep pretrained feature의 inherent characteristics를 활용하여 **zero-shot**으로 test image에 selective adaptation. Model-agnostic lightweight MLP (neural implicit representation)를 사용.

**Key Results**: Brain anomaly detection에서 enlarged ventricles +78%, edemas +24% detection rate 향상.

**Strengths**:
- Zero-shot, model-agnostic
- "Selective" adaptation — 모든 feature가 아닌 특정 component만 적응

**Weaknesses**:
- Medical imaging 특화 — industrial AD 검증 없음
- Neural implicit representation의 fitting 비용

**FM-AD 연결**: "Selective adaptation"이 핵심 insight. FM feature의 모든 dimension을 적응시키면 anomaly-sensitive information도 변형될 위험. **어떤 feature subspace를 적응시키고, 어떤 것을 보존할지**의 분리가 중요 — 이는 `_lessons.md`의 "Feature Disentanglement이 가장 높은 Novelty Gap" 발견과 직접 연결.

---

### 2.7 TTA+AD 논문 현황 요약

| Paper | Venue | Adaptation Target | AD Type | Key Idea |
|-------|-------|-------------------|---------|----------|
| PILOT | BMVC 2025 | Text prompts | Zero-shot VLM-AD | Dual-branch prompt pool + TTA |
| Costanzino | CVPRW 2024 | Threshold/classifier | Unsupervised AD | Test-time threshold calibration |
| EPHAD | NeurIPS 2025 | Output scores | Any AD | FM evidence + exponential tilting |
| TUNE | arXiv 2025 | Graph features | Graph AD | Normality shift alignment |
| Source-Free | ICPR 2024 | Full model | Supervised AD | Pseudo-label + supervisor |
| Selective TTA | MICCAI-W 2024 | Implicit repr. | Medical AD | Zero-shot selective feature adaptation |

**Gap 관찰**: Visual feature adapter (LoRA 등)를 AD에서 test-time에 적응시키는 연구가 **부재**. Prompt-level (PILOT), output-level (EPHAD), threshold-level (Costanzino)은 있으나, feature extractor 자체의 test-time adaptation for visual AD는 미개척 영역.

---

## Part 3: Domain Adaptation for Anomaly Detection

### 3.1 FiCo — Filter or Compensate (AAAI 2025)

**Core Method**: Reverse Distillation 프레임워크에서 teacher-student 간 distribution shift로 인한 misalignment을 해결. (1) **DiSCo** (Distribution-Specific Compensation): student feature에 distribution-specific 정보를 보상하여 teacher와 align, (2) **DiIFi** (Distribution-Invariant Filter): abnormal 정보 + distribution-specific 정보를 모두 필터링하여 distribution-invariant normality만 보존.

**Key Results**: OOD scenario에서 SOTA, ID scenario에서도 RD 기반 방법 대비 개선.

**Strengths**:
- "Filter vs Compensate"라는 명확한 dual mechanism
- RD 프레임워크 위에서 plug-in 가능
- ID/OOD 모두에서 효과적

**Weaknesses**:
- RD 프레임워크에 종속적
- FM feature 대신 ResNet/WideResNet backbone 사용
- Distribution shift의 유형을 명시적으로 모델링하지 않음 — 모든 shift를 동일하게 처리

**FM-AD 연결**: DiSCo/DiIFi의 filter-compensate 패러다임을 FM feature에 적용하는 것이 유망. FM의 경우 distribution-specific component가 무엇인지 (BN stats? attention pattern? token norm?)를 먼저 식별해야 함.

---

### 3.2 ROADS — Robust Prompt-Driven Multi-Class AD under Domain Shift (WACV 2025)

**Core Method**: Multi-class Unified AD (MUAD)에서 (1) **Hierarchical class-aware prompt**: class-specific prompt token을 multi-scale로 학습하여 inter-class interference 감소, (2) **Domain adapter**: student decoder의 residual block에 AdaIN layer를 장착하여 OOD target style을 ID source style로 rectify.

**Key Results**: MVTec-AD, VISA에서 detection + localization SOTA (특히 OOD 조건).

**Strengths**:
- Class-aware + domain-invariant의 orthogonal 설계
- AdaIN 기반 style alignment은 경량이면서 효과적
- Multi-class 설정에서의 inter-class interference 해결

**Weaknesses**:
- AdaIN은 style (mean/variance) 수준의 alignment만 가능 — structural shift에는 한계
- Source domain data로 학습 필요 (test-time 적응 아님)
- Domain adapter의 학습에 target domain 일부 data 필요 여부 불명확

**FM-AD 연결**: AdaIN 기반 style transfer를 FM feature의 test-time normalization으로 확장 가능. Feature의 channel-wise mean/variance를 source domain statistics로 re-normalize하는 것은 TENT의 BN adaptation과 유사한 원리.

---

### 3.3 ADPretrain — AD-Specific Pretraining (NeurIPS 2025)

**Core Method**: ImageNet pretrained feature의 한계를 지적 — ImageNet pretraining은 normal/abnormal 구분을 목표로 하지 않으며, natural image와 industrial image 간 domain gap 존재. (1) Angle-oriented + norm-oriented contrastive loss로 normal/abnormal feature의 angle과 norm 동시 분리, (2) RealIAD 대규모 AD dataset으로 pretraining, (3) Residual feature 기반 class-generalizable representation 학습.

**Key Results**: 5개 embedding 기반 AD 방법에 pretrained feature 대체 시 5개 AD dataset, 5개 backbone에서 일관된 개선.

**Strengths**:
- ImageNet → AD-specific pretraining이라는 패러다임 제안
- 기존 AD 방법에 plug-in 가능
- Residual feature로 domain shift 완화

**Weaknesses**:
- AD pretraining에 대규모 AD data 필요 — data-hungry
- Pretraining이 모든 domain shift를 해결하지 못함 (training domain과 다른 test domain)
- Feature space의 근본적 변화 — 기존 FM (CLIP/DINOv2) 대비 장점이 항상 보장되지 않음

**FM-AD 연결**: ADPretrain의 핵심 insight — "pretrained feature가 AD에 최적이 아니다"는 우리 `_lessons.md`의 "FM Feature Invariance != AD Robustness"와 정확히 일치. 다만 ADPretrain은 pretraining 단계 해결, 우리는 test-time 해결을 목표로 함 — 상호보완적.

---

### 3.4 Robust Distribution Alignment (RoDA) — Sinkhorn Distance for AD (2025)

**Core Method**: Memory bank 기반 AD에서 source-target distribution alignment. (1) **Robust Sinkhorn distance**: entropy-regularized Wasserstein distance로 target domain의 anomalous sample 영향 완화, (2) Assignment discretization + target data augmentation으로 robustness 강화.

**Key Results**: MVTec, RealIAD (2D), MVTec 3D에서 simulated distribution shift 하에서 SOTA 대비 우수한 robustness. OT alignment이 KL-Div, Moment Matching을 일관되게 outperform.

**Strengths**:
- 수학적으로 rigorous한 OT 프레임워크
- Anomaly sample의 영향을 자연스럽게 완화
- 2D/3D 모두 검증

**Weaknesses**:
- Limited target training data 필요 — 완전한 test-time 적응은 아님
- Sinkhorn iteration의 computational cost
- Simulated distribution shift — real-world shift 검증 부족

**FM-AD 연결**: Memory bank의 OT alignment은 PatchCore 등 FM-based AD에 직접 적용 가능. Test-time에 target normal sample만으로 memory bank을 realign하는 것은 prototype update와 유사한 원리. OT의 robustness guarantee는 이론적 novelty가 높음.

---

### 3.5 ADShift — Anomaly Detection under Distribution Shift (ICCV 2023)

**Core Method**: Training과 inference 모두에서 ID-OOD normal sample 간 distribution gap을 unsupervised하게 최소화. Generalized normal learning approach.

**Key Results**: Distribution shift가 있는 data에서 SOTA AD + OOD generalization 방법 모두 대비 유의미한 개선. ID data에서도 성능 유지.

**Strengths**:
- AD에서 distribution shift를 명시적으로 다룬 최초의 체계적 연구 (ICCV)
- Training + inference 양단에서 대응
- AeBAD dataset 함께 제안

**Weaknesses**:
- Pre-FM 시대 방법 (ResNet 기반)
- FM feature에서의 효과 미검증

**FM-AD 연결**: ADShift가 정의한 "normality shift" 문제 프레임이 우리 연구의 핵심 문제와 정확히 일치. FM feature 위에서의 ADShift 재현/확장이 baseline으로 적합.

---

### 3.6 AeBAD — Aero-Engine Blade AD with Domain Shift (2023)

**Core Method (MMR)**: Masked Multi-scale Reconstruction. Patch masking으로 normal sample 간 인과관계 학습 능력 강화.

**Dataset 특성**: Illumination + view 변화로 인한 domain shift. Unaligned, multi-scale. 1228 normal + 4342 abnormal (4 defect types).

**Key Results**: AeBAD-S에서 PRO 89.1% (PatchCore 87.8% 대비), MVTec에서 competitive.

**FM-AD 연결**: AeBAD는 real-world domain shift가 내재된 AD benchmark으로서 가치가 높음. MVTec/VisA에서 simulated shift만 테스트하는 것보다 AeBAD에서의 검증이 real-world 설득력을 높임.

---

## Part 4: Prompt/Adapter Robustness for VLMs

### 4.1 APT — Adversarial Prompt Tuning (CVPR 2024)

**Core Method**: VLM의 adversarial robustness를 text prompt 관점에서 접근. 단 하나의 learned prompt word 추가만으로 adversarial attack에 대한 robustness 대폭 향상.

**Key Results**: Accuracy +13%, robustness +8.5% 평균. 최대 accuracy +26.4%, robustness +16.7%.

**Strengths**:
- 극도로 경량 (1 word)
- Adversarial + clean accuracy 동시 개선

**Weaknesses**:
- Adversarial robustness ≠ distribution shift robustness
- AD-specific 검증 없음

**FM-AD 연결**: "Robust prompt"의 개념을 AD로 전이 가능. AD에서의 "adversarial" = distribution shift로 재해석. Prompt robustness와 AD feature robustness의 관계 분석이 필요.

---

### 4.2 Benchmarking Robustness of Adaptation Methods on Pre-trained VLMs (NeurIPS 2023)

**Core Method**: 11개 adaptation 방법을 4개 VL dataset에서 96 visual + 87 textual corruption으로 평가.

**Key Findings**:
1. **Text corruption에 더 민감** — visual corruption보다 text corruption이 성능 저하 심각
2. **Full fine-tuning ≠ best robustness** — adapter가 comparable clean performance에서 더 나은 robustness 달성
3. **Data/parameter 증가 ≠ robustness 향상** — 오히려 robustness 하락 가능

**Strengths**:
- 대규모 체계적 벤치마크
- 반직관적이지만 중요한 발견들

**Weaknesses**:
- Classification 기준 — AD로의 직접 전이 가능성 미검증
- Corruption types가 industrial domain shift와 다를 수 있음

**FM-AD 연결**: **매우 중요한 시사점**:
- AD에서도 adapter가 full fine-tuning보다 robust할 가능성 → LoRA/lightweight adapter 우선 탐색
- Data 많다고 robust해지지 않음 → few-shot adaptation이 오히려 유리할 수 있음
- AD에서 text corruption (prompt 변화)과 visual corruption (domain shift)의 차등 민감도 분석 필요

---

### 4.3 관련 TTA Robustness 방법들

**CLIPArTT (WACV 2025)**: Text prompt를 test-time에 transductive하게 재구성. Top-K predicted class를 합친 새 prompt로 re-classify. Training-free.

**R-TPT (2025)**: Adversarial attack에 대한 test-time prompt tuning. Inference 단계에서 adversarial 영향 완화.

**Any-Shift Prompting (CVPR 2024)**: Hierarchical probabilistic framework로 training-test distribution 연결. 다양한 shift 유형 동시 대응.

**Robust Fine-tuning via Variance Reduction (NeurIPS 2024)**: Ensemble prediction의 variance를 줄여 OOD accuracy +3.6%, ID accuracy +1.6%.

---

## Part 5: Synthesis — FM-AD Robustness를 위한 TTA의 기회와 도전

### 5.1 현재 Landscape의 Gap

```
                       Output-level    Prompt-level    Feature-level
                       ───────────     ────────────    ─────────────
TTA for Classification    TENT           TPT, CLIPArTT   LoRA-TTT
TTA for AD                EPHAD          PILOT           ???
                                                         ↑
                                                    OPEN OPPORTUNITY
```

**Feature-level TTA for Visual AD는 아직 누구도 하지 않았다.** PILOT은 prompt만, EPHAD는 output만, Costanzino는 threshold만 적응시킨다. FM의 visual feature를 test-time에 AD-aware하게 적응시키는 연구가 부재.

### 5.2 Why Feature-Level TTA Matters for AD

1. **Prompt adaptation의 한계**: Text prompt는 global semantic을 제어하지만, AD에 필요한 local, fine-grained feature에는 간접적 영향만 미침.
2. **Output adaptation의 한계**: Score 보정은 근본적 feature quality 문제를 해결하지 못함.
3. **Feature quality가 AD 성능의 upper bound**: Memory bank, normalizing flow, distillation 모두 feature quality에 의존.

### 5.3 설계 원칙 (TTA for FM-AD)

`_lessons.md`의 기존 발견과 본 survey의 종합:

1. **Selective Adaptation**: 모든 feature dimension이 아닌, nuisance-encoding subspace만 적응 (Selective TTA + Feature Disentanglement 결합)
2. **Anomaly Signal 보존**: Adaptation이 anomaly-sensitive feature를 파괴하지 않아야 함 (FoCAL의 교훈)
3. **Low-Rank Constraint**: LoRA-TTT의 원리 — pretrained knowledge의 catastrophic forgetting 방지
4. **Prototype-Guided**: ProtoDCS의 원리 — sample-level이 아닌 prototype-level update로 안정성 확보
5. **Robust Estimation**: RoDA의 원리 — test batch에 anomaly가 섞여 있으므로 robust alignment 필요

### 5.4 우리 연구와의 Positioning

| 기존 접근 | 우리의 차별화 |
|----------|-------------|
| PILOT: prompt만 적응 | Feature extractor 자체를 적응 |
| EPHAD: output 보정 | Feature-level 적응 (upstream) |
| FiCo: training-time DA | Test-time adaptation |
| ROADS: style transfer (AdaIN) | Principled feature subspace adaptation |
| RoDA: memory bank alignment | Feature extractor adaptation (더 근본적) |
| LoRA-TTT: classification용 | AD-specific objective 설계 |

---

## 관련 노트
- [_lessons.md](_lessons.md) — FM-AD robustness 검증된 패턴
- [2026-03-23_solution_survey.md](2026-03-23_solution_survey.md) — 기존 robustness 해결책 survey
- [2026-03-23_mechanism_analysis.md](2026-03-23_mechanism_analysis.md) — FM feature의 shift 메커니즘 분석
