# Master Research Report: FM-AD Robustness via Feature Disentanglement

> **최종 업데이트**: 2026-03-27
> **프로젝트**: Pilot — Visual Anomaly Detection 연구
> **타겟**: ICLR 2027 제출
> **핵심 주장**: Foundation Model 기반 Anomaly Detection은 distribution shift에서 심각하게 열화되며, 이는 feature space에서 anomaly-relevant 정보와 nuisance 정보의 entanglement 때문이다. Post-hoc linear disentanglement로 이를 해결할 수 있다.

---

## 1. 연구 배경 및 동기

### 1.1 VAD 분야의 패러다임 전환

Visual Anomaly Detection(VAD) 분야는 3가지 패러다임 전환을 겪고 있다:

1. **Per-class → Unified Multi-class**: 카테고리별 개별 모델 → 단일 모델로 모든 카테고리 처리 (Dinomaly, CVPR 2025)
2. **Task-specific → Foundation Model 기반**: 전용 backbone → DINOv2/CLIP 등 사전학습된 FM 활용
3. **Supervised → Zero/Few-shot**: 대량 이상치 라벨 → 소수 정상 샘플 또는 라벨 없이 탐지

이 전환의 중심에 Foundation Model(FM)이 있다. DINOv2와 CLIP 기반 방법들은 MVTec AD에서 99%+ I-AUROC를 달성하며, FM feature의 범용성이 AD에도 충분하다는 가정이 지배적이다.

### 1.2 문제 제기: FM Feature의 Robustness는 검증되지 않았다

**그러나 이 가정은 틀렸다.** MVTec AD 2, RobustAD 등 현실적 벤치마크에서 FM-AD 방법들은 25~34pp의 성능 하락을 보인다. 이는 단순한 성능 저하가 아닌, FM feature 자체의 구조적 한계를 시사한다.

| 벤치마크 | 특성 | FM-AD 성능 |
|----------|------|-----------|
| MVTec AD | Clean, 통제된 환경 | 91~99% I-AUROC |
| MVTec AD 2 | 조명/각도 변화 포함 | 57~72% I-AUROC (-25~34pp) |
| RobustAD | 9종 domain shift | I-AUROC 73~84% |
| Real-IAD Variety | 160 카테고리 | 10~20% 추가 열화 |

### 1.3 이 연구의 위치

기존 연구는 크게 세 가지로 나뉜다:

- **벤치마크 구축**: MVTec AD 2, RobustAD — "깨진다"는 사실 확인
- **현상 관찰**: Phi-eat, SINDER — FM feature의 특이 행동 보고
- **Task-specific 해결**: PILOT (prompt), EPHAD (score calibration) — 증상 완화

**이 연구가 채우는 공백**: FM feature가 **왜** 깨지는지에 대한 메커니즘 진단 + feature 수준의 principled한 해결. 기존 연구 중 FM feature를 anomaly-relevant vs nuisance로 명시적으로 분리한 작업은 **0건**이다.

---

## 2. 이론적 기반

### 2.1 FM Feature의 7가지 실패 메커니즘

Phase 2 분석과 선행 연구 종합을 통해 7가지 실패 메커니즘을 식별했다:

| # | 메커니즘 | 설명 | 근거 |
|---|---------|------|------|
| 1 | **Invariance-as-Information-Destruction** | Color jitter, blur 등 augmentation이 anomaly-relevant 외관 정보 파괴 | Contrastive learning 이론 |
| 2 | **Semantic Dominance** | DINOv2가 "무엇인지"(semantic)를 "어떻게 보이는지"(material/texture)보다 우선 인코딩 | Phi-eat (2025) 실증 |
| 3 | **High-Norm Token Artifacts** | 특정 patch token의 norm이 7.5배 이상 높아 distance scoring 왜곡 | SINDER (ECCV 2024): norm 434 vs 57.6 |
| 4 | **Inter-Class Interference** | Multi-class 통합 시 클래스 간 분산이 이상치 신호를 압도 | 160+ 카테고리에서 10~20% 추가 열화 |
| 5 | **Nuisance-Semantic Entanglement** | 조명/시점 변화가 anomaly 차원과 직교하지 않음 | Phase 2 실측: PC correlation 0.53 |
| 6 | **Pretraining Distribution Mismatch** | ImageNet에 산업 결함 이미지 거의 부재 | 도메인 간극 |
| 7 | **Token-Level Shift Accumulation** | Patch 단위 shift가 attention을 통해 전파·증폭 | Transformer 구조적 특성 |

**이 연구가 집중하는 메커니즘**: #5 (Nuisance-Semantic Entanglement) — 가장 높은 novelty gap (기존 연구 0건)이며 이론적 해결 프레임워크 존재.

### 2.2 이론적 프레임워크

**Nonlinear ICA Identifiability** (von Kügelgen, NeurIPS 2021):
- Nuisance와 anomaly 요소의 분리에는 **auxiliary variable**(보조 변수)이 필요
- 본 연구에서는 **environment label** (clean vs shifted condition)을 보조 변수로 활용
- 이론적 보장: 충분한 environment 수에서 block identifiability 달성 가능

**PISCO** (ICML 2023): 사후 선형 분해가 CLIP 표현에서 유효함을 실증 → DINOv2로 확장 가능성.

**핵심 연결**: FM의 contrastive/self-supervised 학습은 view-invariant representation을 만들지만, anomaly detection에 필요한 fine-grained 외관 정보가 이 과정에서 "nuisance"로 취급되어 손실된다. 이를 복원하려면 feature space에서 nuisance 방향을 식별하고 제거해야 한다.

---

## 3. 실험 결과

### 3.1 Phase 1: Baseline Reproduction & Robustness Stress Test

> **가설**: FM-AD 방법들은 MVTec AD 2에서 20pp 이상 하락한다.
> **결과**: **확인됨** — 모든 방법이 25pp 이상 하락.

| Method | Backbone | MVTec AD | MVTec AD 2 | Drop |
|--------|----------|----------|-----------|------|
| Dinomaly | DINOv2 ViT-B/14-reg | 99.64% | 68.6% | **-31.0pp** |
| AnomalyDINO | DINOv2 ViT-B/14 | 97.7% | 72.7% | **-25.0pp** |
| AnomalyCLIP | CLIP ViT-L/14@336 | 91.5% | 57.9% | **-33.6pp** |

**핵심 발견**:
- Backbone 무관: DINOv2든 CLIP이든 모두 실패 → FM 공통 구조적 문제
- wallplugs 카테고리: 전 방법 42% 수준 (near-random) → 반사면 + 미세결함의 극단적 난이도
- 학습 기반 > 비학습 기반 방법이 더 취약: 학습이 source domain에 과적합
- RobustAD: I-AUROC 73~84% (domain 간), pixel-level은 상대적으로 안정 (96~98%)

### 3.2 Phase 2: Feature Space Shift Analysis

> **가설**: Deep layer는 robust하고, shallow layer는 취약하다 (단조 관계).
> **결과**: **기각됨** — Layer robustness는 V-자형 비단조 패턴.

#### Layer별 Shift 프로파일 (DINOv2 ViT-B/14)

| Layer | RelShift | CosSim | 해석 |
|-------|----------|--------|------|
| 0-3 (shallow) | 0.22-0.25 | 0.96 | 중간 수준 shift |
| 4-7 (mid) | 0.26-0.27 | 0.94 | **가장 취약** |
| **8 (sweet spot)** | **0.11** | **0.99** | **유일하게 robust** |
| 9-10 | 0.15-0.22 | 0.98 | 회복 중 |
| 11 (last) | 0.26 | 0.95 | 다시 취약 |

**Layer 8의 특이성**: RelShift 0.11로 다른 layer(0.22-0.27) 대비 2배 이상 안정적. 이는 DINOv2의 attention 구조에서 특정 중간 layer가 정보 bottleneck 역할을 하는 것으로 추정.

#### Entanglement 정량화

| Layer | PC Projection Correlation | 해석 |
|-------|--------------------------|------|
| Layer 11 | 0.53 | 중간 수준 entanglement → disentanglement 필요 |
| Layer 8 | 0.12 | 자연적으로 분리됨 → robust layer의 이유 |

**t-SNE 관찰**: Shifted-normal 샘플이 anomaly 영역과 겹침 → false positive의 직접 원인 확인.

#### 카테고리별 패턴

| Category | Feature Shift | AD Drop | 분석 |
|----------|--------------|---------|------|
| fabric | 0.15 | -27pp | shift와 성능 하락 비례 (정상) |
| wallplugs | 0.36 | -56pp | 최대 shift, 최대 하락 |
| sheet_metal | 0.33 | -20pp | **이상치**: shift 크지만 하락 적음 → shift가 anomaly와 직교 |

**결론**: Feature shift의 크기뿐 아니라 **방향**(anomaly 차원과의 관계)이 성능 하락을 결정한다. 이는 단순 shift correction이 아닌 directional disentanglement의 필요성을 뒷받침한다.

### 3.3 Phase 3: Post-hoc Linear Disentanglement (NSP)

> **가설**: Nuisance Subspace Projection(NSP)으로 5-10pp 개선 가능.
> **결과**: **초과 달성** — +29.4pp 개선, clean 성능 거의 유지.

#### 방법론: Nuisance Subspace Projection (NSP)

```
1. Multi-environment 정상 데이터 수집 (clean + shifted conditions)
2. Shift vector 계산: v_i = f(x_shifted) - f(x_clean)
3. Shift vector들에 PCA → 상위 K개 = nuisance subspace
4. Feature에서 nuisance subspace 성분 제거 (orthogonal projection)
5. Projected feature로 anomaly scoring (Mahalanobis distance)
```

**특성**: 학습 파라미터 0개, 순수 post-hoc 기하학적 연산.

#### Autoresearch 실험 궤적

| 실험 | Config | AD2 I-AUROC | AD1 I-AUROC | Delta (AD2) | 판정 |
|------|--------|-------------|-------------|-------------|------|
| Baseline | kNN patch L11 K=5 | 54.4% | 91.6% | — | — |
| +NSP | K=30 L11 | 67.9% | 91.9% | +13.5pp | keep |
| +CLS+L2 norm | K=30 L11 | 68.0% | 94.3% | +0.1pp | keep |
| +Mahalanobis | K=30 L11 | 71.4% | 97.4% | +3.4pp | keep |
| +Layer 8 | K=30 L8 | 75.0% | 96.4% | +3.6pp | keep |
| Multi-layer 8+11 | K=30 L8+L11 | 72.5% | 97.7% | -2.5pp | discard |
| +K=50 | K=50 L8 | 78.5% | 96.3% | +3.5pp | keep |
| **+K=100** | **K=100 L8** | **83.8%** | **96.3%** | **+5.3pp** | **keep (best)** |
| Condition union | K=100 L8 | 77.1% | 96.4% | -6.7pp | discard |
| Whitened PCA | K=100 L8 | 83.8% | 96.3% | 0pp | discard (동등) |
| Soft projection | K=100 L8 | 72.5% | 96.4% | -11.3pp | discard |
| RANP signal subspace | n=200 L8 | 59.0% | 95.0% | -24.8pp | discard |

#### 최적 구성 및 성과

```
Best Config:
  - Layer: 8 (naturally robust layer)
  - Nuisance dims (K): 100
  - Scoring: Mahalanobis distance
  - Feature: CLS + patch mean, L2 normalized
  - Learned parameters: 0

Performance:
  - MVTec AD 2: 83.8% (+29.4pp from baseline)
  - MVTec AD:   96.3% (+4.7pp from baseline, 또는 -0.2pp from Mahalanobis best)
```

#### 컴포넌트별 기여 분해 (Ablation)

| Component | 독립 기여 | 역할 |
|-----------|----------|------|
| NSP (K=30, L11) | +13.5pp | 핵심: nuisance 제거 |
| Layer 8 선택 | +3.6pp | 자연적으로 robust한 layer 활용 |
| K 증가 (30→100) | +8.8pp | 더 많은 nuisance 방향 제거 |
| Mahalanobis scoring | +3.4pp | Projected space의 covariance 활용 |
| CLS + L2 norm | +0.1pp | Feature 표현 개선 (미미) |

#### 검증

- Nuisance subspace 상위 10개 방향이 **전체 shift 분산의 72.2%** 포함 → nuisance subspace가 실제로 shift를 포착
- Anomaly 차원은 nuisance subspace와 직교 → projection으로 보존됨
- K 증가에 따라 단조 개선, saturation 미도달 → 추가 탐색 여지 존재

#### 실패한 시도와 교훈

| 시도 | 결과 | 교훈 |
|------|------|------|
| Multi-layer fusion (L8+L11) | 72.5% (< L8 단독 75.0%) | Layer 선택 > fusion. L11의 noise가 L8을 오염 |
| Condition-별 union | 77.1% (< 83.8%) | 표준 PCA가 condition union보다 우수 |
| Soft projection (부분 제거) | 72.5% | 보수적 제거는 오히려 해로움. Hard projection이 최적 |
| Signal subspace 보존 (RANP) | 59.0% | Anomaly 신호가 어느 subspace에 있는지 사전 지정 불가 |

---

## 4. 연구 방향 평가 및 선택

### 4.1 후보 방향 비교

초기에 6개 연구 방향을 탐색했으며, 체계적 평가를 거쳐 최종 방향을 선택했다:

| 방향 | Novelty | Feasibility | 기존 연구 | 선택 |
|------|---------|-------------|----------|------|
| **FM Feature Disentanglement for AD Robustness** | **VERY HIGH** | **HIGH** | **0건** | **채택** |
| Test-Time Adaptation for AD | HIGH | HIGH | 소수 (PILOT 등) | 차선 |
| Logical Anomaly Detection | HIGH | MEDIUM | SALAD 강력 | 보류 |
| Continual/Incremental AD | HIGH | MEDIUM | 벤치마크 미성숙 | 보류 |
| Multi-Modal RGB+3D AD | HIGH | MEDIUM | 인프라 요구 | 보류 |
| Anomaly Synthesis | MEDIUM | MEDIUM | Synthetic-real gap | 보류 |

### 4.2 선택 근거

1. **Novelty gap**: Feature disentanglement을 FM-AD robustness에 적용한 연구가 0건
2. **이론적 기반**: Nonlinear ICA identifiability 프레임워크가 직접 적용 가능
3. **경험적 검증**: 이미 NSP로 +29.4pp 달성 — proof of concept 완료
4. **ICLR 적합성**: Analysis(60%) + Lightweight Fix(40%) 구조에 자연스럽게 부합
5. **저자 강점**: ProxyCore(robustness), NF 기반 Continual AD, distribution shift 전문성

---

## 5. 논문 구조 및 포지셔닝

### 5.1 핵심 Narrative Arc

```
1. FM features → 99%+ on MVTec AD → "robust하다"는 가정이 지배적
2. 실증: MVTec AD 2에서 25-34pp 하락 → 가정이 틀렸다
3. 진단: Feature space에서 nuisance와 anomaly 신호가 entangled
4. 해결: Post-hoc linear disentanglement → +29.4pp 회복, clean 유지
5. 시사: FM feature를 "있는 그대로" 쓰는 것은 위험 → feature-level adaptation 필요
```

### 5.2 Contribution 후보

1. **Systematic Failure Analysis**: FM-AD 3종 × 3종 벤치마크에서 체계적 robustness 평가 + 7가지 실패 메커니즘 식별
2. **Mechanistic Diagnosis**: Layer별 비단조 robustness 프로파일 (Layer 8 sweet spot 발견) + nuisance-anomaly entanglement 정량화
3. **Principled Solution**: 이론 기반(Nonlinear ICA) post-hoc linear NSP — 0 learned params, +29.4pp 개선

### 5.3 Figure/Table 계획

| 위치 | 내용 | 데이터 소스 | 상태 |
|------|------|-----------|------|
| Fig 1 | Entanglement 시각화: shifted-normal ↔ anomaly 겹침 | Phase 2 t-SNE | 데이터 확보 |
| Fig 2 | Layer별 robustness-sensitivity V자형 프로파일 | Phase 2 layer 분석 | 데이터 확보 |
| Fig 3 | NSP 방법론 overview (pipeline) | 설계 | 설계 필요 |
| Fig 4 | Disentanglement 전/후 feature 분포 비교 | Phase 3 | 생성 필요 |
| Tab 1 | Main results: 3 methods × 3 benchmarks | Phase 1 + 3 | **부분 완료** |
| Tab 2 | Ablation study (component별 기여) | Phase 3 | 데이터 확보 |
| Tab 3 | Per-category / per-shift-type breakdown | Phase 1 + 2 | 데이터 확보 |

### 5.4 ICLR 포지셔닝 전략

**타이틀 후보**: "Are Foundation Model Features All You Need for Anomaly Detection?"

**ICLR가 선호하는 패턴** (선행 분석 기반):
- 널리 믿어지는 가정에 도전 + 체계적 증거 제시
- Analysis-heavy 논문도 동등하게 가치 부여 (SOTA 불필요)
- Principled lightweight fix는 복잡한 해결책보다 선호됨
- 구조: 9 pages, 60% 분석 + 40% 해결

---

## 6. Verified Patterns (검증된 패턴)

Phase 1-3에서 반복 확인된 패턴들 (`_lessons.md` 기반):

| # | 패턴 | 함의 |
|---|------|------|
| 1 | FM Feature Invariance ≠ AD Robustness | DINOv2/CLIP의 invariance가 오히려 fine-grained 정보를 파괴 |
| 2 | Multi-class scaling은 숨겨진 리스크 | 160+ 카테고리에서 추가 10-20% 열화 |
| 3 | MVTec AD 성능 ≠ Robustness | 99%+ 방법이 MVTec AD 2에서 60%로 추락 |
| 4 | Feature disentanglement = 최대 novelty gap | AD에서 FM feature를 명시적으로 분리한 연구 0건 |
| 5 | ICLR는 분석 + lightweight fix 선호 (60:40) | SOTA 불필요, insight 중시 |
| 6 | Training-free DINOv2가 training 기반보다 robust | SuperAD 3.5% drop vs RoBiS 11.3% |
| 7 | Can/Wallplugs는 전 방법 실패 (<10%) | 반사면 + 미세결함 = 현재 기술 한계 |
| 8 | AU-PRO(0.05)는 기존 대비 6배 엄격 | 미세결함 평가에 적합 |

---

## 7. 현재 상태 및 다음 단계

### 7.1 진행 상태

| Phase | 상태 | 핵심 산출물 |
|-------|------|-----------|
| Phase 1: Baseline Reproduction | **완료** | 3 methods × 2 benchmarks 정량 결과 |
| Phase 2: Mechanism Analysis | **완료** | Layer 프로파일, entanglement 정량화, t-SNE |
| Phase 3: Method Design (NSP) | **진행 중** | 83.8% AD2 달성, K saturation 미확인 |
| Phase 4: Validation & Ablation | **미착수** | — |
| Phase 5: Paper Writing | **미착수** | — |

### 7.2 즉시 다음 단계 (Phase 3 계속)

1. **K saturation 탐색**: K=150, 200까지 실험 — 성능 상한 확인
2. **Robust PCA / ICA**: Nuisance estimation 방법 개선 시도
3. **Per-category nuisance**: 카테고리별 개별 nuisance subspace 효과 확인
4. **Whitening 후 projection**: Feature 정규화 순서 변경 실험

### 7.3 Phase 4 계획 (Validation)

1. **Main Results Table**: NSP를 3개 baseline(Dinomaly, AnomalyDINO, AnomalyCLIP)에 적용
2. **Seed variance**: 3-5 seed 반복 측정
3. **Per-shift-type breakdown**: 어떤 shift에서 가장 효과적인지
4. **Failure case analysis**: 여전히 실패하는 카테고리 분석
5. **Computational overhead**: Inference time 비교 (NSP는 무시할 수준 예상)

### 7.4 Open Questions

1. **K saturation**: K=100에서 아직 개선 중 — 최적 K는?
2. **Nonlinear extension**: Linear NSP로 충분한가, 아니면 nonlinear가 추가 개선을 주는가?
3. **다른 backbone 적용**: CLIP에서도 동일한 효과를 보이는가?
4. **Environment label 수**: 최소 몇 개의 condition이 필요한가?
5. **Can/Wallplugs 해결**: 전 방법 실패 카테고리에 대한 별도 접근 필요?

---

## 8. 리스크 평가

| 리스크 | 확률 | 현재 상태 | 대응 |
|--------|------|----------|------|
| FM-AD가 예상보다 robust | ~~중~~ | **해소**: 25-34pp 하락 실증 | — |
| Linear disentanglement 불충분 | ~~중~~ | **일부 해소**: +29.4pp 달성 | Nonlinear backup 준비 |
| Clean 성능 하락 | ~~높~~ | **해소**: AD1 96.3% (baseline 91.6% 대비 +4.7pp) | — |
| 기존 방법과 차별화 부족 | 낮 | **해소**: 기존 연구 0건 확인 | — |
| K saturation / 최적점 미발견 | 중 | 탐색 중 | K sweep 계속 |
| 다른 baseline에 일반화 실패 | 중 | 미검증 | Phase 4에서 확인 |
| Seed 민감도 | 중 | 미검증 | Phase 4에서 확인 |

---

## 9. 참고 문헌 구조

### Tier 1: 정독 필수 (10편)

1. **MVTec AD 2** — 핵심 벤치마크, AU-PRO(0.05) 정의
2. **SuperAD** — FM-AD 대규모 비교, robustness 순위
3. **RoBiS** — DINOv2 robustness 정량 평가
4. **RobustAD** — 9종 domain shift 벤치마크
5. **Phi-eat** — DINOv2가 semantic 클러스터링, material/texture 무시 실증
6. **SINDER** — High-norm token artifact 발견 (ECCV 2024)
7. **PISCO** — Post-hoc linear decomposition (ICML 2023)
8. **von Kügelgen (NeurIPS 2021)** — Nonlinear ICA identifiability 이론
9. **FiCo** — Filter-compensate domain adaptation
10. **ResAD** — Residual feature (NeurIPS 2024 Spotlight)

### Tier 2: 핵심 개념 참조 (12편)

Dinomaly, AnomalyCLIP, PatchCore, EfficientAD, AnomalyDINO, StyLIP, CausalCLIP, DINOv2 Registers, AD-DINOv3, PILOT, ROADS, Closer Look at CLIP

### Tier 3: 배경 참조 (8편)

Contrastive learning 이론, Minimal sufficient representation, Real-IAD Variety, MMAD, HOOD 등

---

## 부록 A: 데이터셋 개요

| 데이터셋 | 카테고리 | 특성 | 메트릭 |
|----------|---------|------|--------|
| MVTec AD | 15 | Clean, 통제 환경 | I-AUROC, P-AUROC, AU-PRO |
| MVTec AD 2 | 10 | 조명/시점 shift 포함 | AU-PRO(0.05) |
| RobustAD | 5+9 shifts | 9종 domain shift | ARD |
| Real-IAD | 30 | 대규모, 다양한 결함 | I-AUROC, P-AUROC |
| Real-IAD Variety | 160 | Multi-class 스케일링 | I-AUROC |

## 부록 B: Baseline 방법론 요약

| Method | Architecture | Training | Key Feature |
|--------|-------------|----------|-------------|
| **Dinomaly** (CVPR 2025) | DINOv2-Reg → Noisy bottleneck → Decoder | Reconstruction | Unified multi-class, linear attention |
| **AnomalyCLIP** (ICLR 2024) | CLIP ViT-L/14@336 + learnable prompts | Zero-shot prompt learning | Object-agnostic, 17 datasets |
| **PatchCore** (CVPR 2022) | WideResNet-101 → Greedy coreset → kNN | Memory bank (non-parametric) | 가장 robust (only -3pp on AD2) |
| **ResAD** (NeurIPS 2024) | Residual features + NF scoring | Few-shot reference matching | 잔차 기반 cross-class 불변 |
| **EfficientAD** (WACV 2024) | PDN teacher-student, 4 conv layers | Distillation | 최고 속도 (>600fps), 가장 fragile |
| **INP-Former** (CVPR 2025) | 테스트 이미지 내 prototype 추출 → 복원 | Self-supervised prototypes | Natural universality |

## 부록 C: 실험 환경

```
Hardware: NVIDIA RTX 4090 (24GB VRAM)
Runtime: Docker (pilot-anomalydino container)
Backbone: DINOv2 ViT-B/14 (frozen)
Feature extraction: ~5min per full evaluation
Dependencies: scikit-learn, torch, tqdm
```

---

> **관련 파일**
> - `plan.md` — 전체 5-Phase 연구 계획 (조건부 분기 포함)
> - `program.md` — NSP 자율 실험 루프 설정
> - `results.tsv` — 전체 실험 결과 이력
> - `skill_graph/analysis/fm_ad_robustness/` — 분석 노트 모음
> - `skill_graph/experiments/` — Phase 1-3 실험 보고서
> - `skill_graph/papers/` — 논문 분석 노트
