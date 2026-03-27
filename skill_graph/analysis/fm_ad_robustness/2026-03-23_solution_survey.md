# FM-based AD Robustness 해결 방안 서베이

작성일: 2026-03-23
목적: 기존/잠재 솔루션 정리 + novelty gap 식별

---

## Novelty Opportunity Map (요약)

| 방향 | AD 적용? | FM-AD 적용? | Robustness 각도? | Novelty |
|------|---------|------------|-----------------|---------|
| Test-Time Adaptation | 극소수 | PILOT만 (prompt) | 부분적 | **HIGH** |
| Domain Adaptation | 성장 중 | 거의 없음 | 핵심 주제 | **HIGH** |
| Feature Disentanglement | 다수 (일반 AD) | FB-CLIP만 | Robustness 아님 | **VERY HIGH** |
| Prompt/Adapter Robustness | AD에 없음 | 없음 | Classification에 확립 | **HIGH** |
| Multi-Scale Fusion | 확립 | 핵심 패러다임 | 미연구 | MEDIUM |
| Score Calibration | 극소수 | 1편 | 거의 없음 | **HIGH** |
| Augmentation for Robustness | 소수 | 거의 없음 | 초기 | MEDIUM |

---

## 1. Test-Time Adaptation (TTA) for AD

**성숙도: 초기 (AD에서 극소수)**

### 핵심 논문

| 논문 | Venue | 내용 | FM-AD? |
|------|-------|------|--------|
| **PILOT** | BMVC 2025 | High-confidence pseudo-label로 prompt parameter 업데이트, 13 benchmarks | ✅ (prompt만) |
| **TopoOT** | arXiv 2026.01 | Persistence diagram + OT로 anomaly segmentation TTA, F1 +24.1% | ❌ |
| **EPHAD** | NeurIPS 2025 | CLIP+classical post-hoc adjustment, contaminated output 교정 | 부분적 |
| **TUNE** | arXiv 2025.11 | Plug-and-play graph alignment, false alarm 교정 | ❌ |
| **Source-Free TTA** | ICPR 2024 | Augmented mean prediction, online defect detection | ❌ |

### FM 일반 TTA

| 논문 | Venue | 내용 |
|------|-------|------|
| **LoRA-TTT** | arXiv 2025.02 | Image encoder에 LoRA test-time training, OOD +5.79% |
| **FOCAL** | ICML 2025 | FM prior로 rotation/lighting/day-night canonicalization |
| **ProtoDCS** | arXiv 2026.02 | Prototype 기반 VLM TTA, open-set |

### Gap 분석
- ⚠️ **Feature-level TTA (memory bank 통계, feature normalization 파라미터 적응)는 전무**
- PILOT은 prompt만 조정 → feature space 자체는 고정
- TTA + memory-bank method (PatchCore 스타일) 조합은 **완전 미탐색**
- DINOv2 기반 AD (Dinomaly, AnomalyDINO)에 대한 TTA 없음
- **ProxyCore 경험과 자연스럽게 연결 가능**

---

## 2. Domain Adaptation for AD

**성숙도: 성장 중 (classical AD), FM-AD에는 거의 없음**

### 핵심 논문

| 논문 | Venue | 내용 |
|------|-------|------|
| **Robust Distribution Alignment** | arXiv 2025.03 | Sinkhorn distance로 2D/3D AD domain shift 해결 |
| **FiCo** | AAAI 2025 | Distribution-invariant normality capture, filter or compensate |
| **ADPretrain** | NeurIPS 2025 | AD 전용 pretraining으로 natural-industrial gap 축소 |
| **ROADS** | WACV 2025 | Robust prompt-driven multi-class AD under domain shift |

### Gap 분석
- FM feature에 대한 DA는 **거의 미탐색** — 대부분 classical method 대상
- 어떤 FM feature가 어떤 shift type에 더 robust한지 **체계적 연구 없음**
- Source-free DA for AD는 1편(ICPR)뿐
- Feature adaptation (SimpleNet, CFA 스타일)과 robustness의 연결 미연구

---

## 3. Feature Disentanglement / Invariant Feature Learning

**성숙도: 일반 CV에서 확립, AD에서 초기, FM-AD robustness에는 미탐색**

### AD 관련

| 논문 | Venue | 내용 |
|------|-------|------|
| **FB-CLIP** | arXiv 2026.03 | Foreground-background 분리 for zero-shot CLIP AD |
| **IB-IUMAD** | arXiv 2026.03 | Inter-object feature coupling 분리 in multimodal AD |
| **BridgeNet** | arXiv 2025.07 | Depth/appearance 분리 for multimodal industrial AD |
| **HOOD** | ICML 2022 area | Content/style causal disentanglement for OOD detection |

### Style-Content Disentanglement (일반, AD 적용 가능)

| 논문 | Venue | 내용 |
|------|-------|------|
| **StyLIP** | WACV 2024 | CLIP vision encoder에서 style/content 분리 → DG 향상 |
| **Simple Disentanglement** | ICML 2023 | Post-processing, linear disentanglement → DG 향상 |
| **Self-Supervised Isolates Content from Style** | NeurIPS 2021 | **이론적 근거** — identifiability 증명 |

### Gap 분석 — **가장 높은 novelty 잠재력**
- ⚠️ **Style-content disentanglement을 FM-based AD robustness에 적용한 논문 0편**
- StyLIP가 CLIP DG에서 성공했지만 AD에는 미적용
- 이론적 근거 존재 (NeurIPS 2021 identifiability)
- DINOv2/CLIP feature space에서 어떤 dimension/subspace가 nuisance vs semantic 정보를 인코딩하는지 연구 없음
- **"anomaly-relevant feature"와 "anomaly-irrelevant feature" 분리는 open problem with clear novelty**

---

## 4. Prompt/Adapter Robustness

**성숙도: Classification에서 확립, AD에 적용 0편**

### Classification 관련

| 논문 | 내용 | 결과 |
|------|------|------|
| **Craft** (2024) | Cross-modal alignment + MMD | Base-to-Novel +6.1% |
| **APT** (2024) | Single learned prompt word | Accuracy +13%, Robustness +8.5% |
| **CAPT** (2024) | Multi-modal prompt learning | Clean + adversarial alignment 유지 |
| **ACAVP** (2025) | Expanded visual prompt transformation | Distribution shift robustness 향상 |
| **DRiFt** (2025) | LoRA + learnable prompts | Medical robustness |

### Adapter 관련
- **Benchmarking Robustness of Adaptation Methods** (2023): "adapters가 full fine-tuning보다 더 robust"
- AD에서 adapter robustness 연구 = **0편**

### Gap 분석
- Robust prompt/adapter → AD 전이 자체가 novelty
- AD의 one-class 특성이 classification과 다른 challenge 생성
- Domain shift 의미의 "robustness" (adversarial이 아닌)는 완전히 open

---

## 5. Multi-Scale / Multi-Layer Feature Fusion

**성숙도: AD에서 확립, robustness 관점은 미연구**

### 현황
- Dinomaly/Dinomaly2가 multi-layer DINOv2 feature + reconstruction 패러다임 정의
- PromptMAD (2026): Multi-scale convolutional + Transformer attention
- Pyramid-based Mamba (2025): Pyramidal scanning for multi-scale

### Gap 분석
- 어떤 DINOv2/CLIP layer가 어떤 distribution shift에 더 robust한지 **체계적 연구 없음**
- Feature fusion은 성능 최적화 목적으로만 설계 → robustness 목적 설계 없음
- **Domain shift 탐지 기반 adaptive layer selection** = 완전 새로운 방향
- 가설: Fine layer = texture-sensitive = lighting-vulnerable? 검증 필요

---

## 6. Score Calibration / Confidence

**성숙도: 극초기 (visual FM-AD에 1편)**

### 핵심
| 논문 | 내용 |
|------|------|
| **Khan & Krawczyk** (2025) | DINOv2 AD에 Platt scaling, 큰 calibration gap 발견 |
| **StructCore** (2026) | Diagonal Mahalanobis calibration for image scoring |
| **CoCAI** (2025) | Conformal prediction for statistically valid AD |

### Gap 분석
- Cross-domain score comparability (다른 조명 조건의 score를 비교 가능한가?) = **open + 실용적**
- Visual AD에서 conformal prediction 적용 = 미탐색
- Domain-aware score normalization = 미연구

---

## 7. Data Augmentation for Robustness

**성숙도: 일반 AD에서 확립, robustness 지향은 초기**

### 핵심
| 논문 | 내용 |
|------|------|
| **RoBiS** (2025) | Noise + lighting simulation augmentation, **+29.2% 개선** |
| **FOCAL** (ICML 2025) | FM prior로 test-time canonicalization |
| **FAST** (NeurIPS 2025) | Foreground-aware diffusion synthesis |

### Gap 분석
- Domain shift simulation augmentation (조명/viewpoint/material 변화)로 robust AD 학습 = 소수
- Style transfer / domain randomization for AD = 부재
- Synthesis + disentanglement 결합 = 미탐색

---

## 가장 유망한 Novel Direction (순위)

### 1위: FM Feature Space에서의 Feature Disentanglement for Robust AD
- **아이디어**: DINOv2/CLIP feature에서 nuisance-encoding subspace와 anomaly-encoding subspace를 분리
- **근거**: 이론(NeurIPS 2021 identifiability) + 실증(StyLIP for CLIP DG) 존재
- **Gap**: AD에 적용 0편
- **Novelty-to-Feasibility 비율 최고**

### 2위: Test-Time Feature Adaptation for FM-based AD
- **아이디어**: Prompt뿐 아니라 feature distribution 자체(memory bank 통계, normalization 파라미터)를 test time에 적응
- **근거**: PILOT(prompt TTA) 성공, ProxyCore 경험
- **Gap**: Feature-level TTA for FM-AD = 전무

### 3위: Cross-Domain Anomaly Score Calibration
- **아이디어**: 조건 변화에도 score가 comparable하도록 domain-aware normalization
- **Gap**: Visual AD에서 거의 미연구
- **실용적 가치 높음**

### 4위: Robustness-Aware Layer/Feature Selection
- **아이디어**: FM layer별 shift-type에 대한 robustness를 측정하고, shift 탐지 결과에 따라 adaptive fusion
- **Gap**: 기존 연구 0편

### 5위: Domain-Shift-Simulating Augmentation
- **아이디어**: Style transfer/domain randomization으로 shift-invariant representation 학습
- **Gap**: 기존 두 아이디어의 미탐색 교차점

---

## 타겟 벤치마크

1. **MVTec AD 2** — 조명 shift 평가가 명시적으로 포함됨
2. **RobustAD** — domain shift 전용
3. **Real-IAD Variety** — 스케일링 평가
4. **VAND 3.0 Challenge** — robustness 경쟁

---

## 관련 노트

- [2026-03-23_failure_survey.md](2026-03-23_failure_survey.md) — 실패 사례 종합
- [2026-03-23_mechanism_analysis.md](2026-03-23_mechanism_analysis.md) — 메커니즘 분석
- [2026-03-23_ICLR_positioning.md](2026-03-23_ICLR_positioning.md) — 논문 포지셔닝
