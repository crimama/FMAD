# FM-AD / Few-shot AD / Vision-Encoder-Only AD 트렌드 조사

> **생성일**: 2026-03-30
> **소스**: 3개 병렬 에이전트, 60+편 조사
> **범위**: 2024-2026 주요 논문, 방법론, 인사이트

---

## Executive Summary

VAD(Visual Anomaly Detection) 분야는 **5가지 메가트렌드**가 동시 진행 중:

1. **DINOv2 → DINOv3**: Self-supervised ViT가 backbone 표준. CLIP text encoder는 불필요해지는 추세
2. **Training-free의 부상**: Frozen FM + 단순 scoring(kNN, PCA)이 학습 기반과 동등
3. **MVTec AD 포화 → Shift robustness가 새 frontier**: 99.9% 달성 후 MVTec AD 2, M2AD에서 급락
4. **Subspace/Projection 방법 등장**: SubspaceAD, FoundAD, ResAD — feature 변환 후 scoring
5. **Few-shot + Unified**: 1-4장으로 full-shot 수준 달성, multi-class 단일 모델

**우리 연구와의 직접 연결**: (4)의 subspace 방법들 중 **nuisance subspace를 명시적으로 식별하고 제거**하는 연구는 0건 → 우리의 gap.

---

## 1. Foundation Model AD (2024-2026)

### 1.1 Reconstruction 기반 (DINOv2)

| 논문 | Venue | Backbone | 핵심 | I-AUROC |
|------|-------|----------|------|---------|
| **Dinomaly** | CVPR 2025 | DINOv2-Reg | Noisy bottleneck + linear attention + loose reconstruction. "Less is more" | 99.6% (MVTec) |
| **Dinomaly2** | arXiv 2025 | DINOv2 | Dinomaly 확장, 99.9% 달성 | 99.9% |
| **INP-Former** | CVPR 2025 | DINOv2 | Test image 내부에서 intrinsic normal prototype 추출 → 외부 데이터 불필요 | 98.6% |

**Insight**: Reconstruction 방법은 decoder를 의도적으로 **약하게** 설계해야 작동 — FM feature가 이미 충분히 강력하므로 bottleneck이 핵심.

### 1.2 Zero-shot CLIP 기반

| 논문 | Venue | 핵심 | 특징 |
|------|-------|------|------|
| **AnomalyCLIP** | ICLR 2024 | Object-agnostic text prompt 학습 | 17 datasets, zero-shot |
| **AA-CLIP** | CVPR 2025 | Anomaly-aware CLIP fine-tuning | Zero-shot 강화 |
| **AdaCLIP** | ECCV 2024 | Static + dynamic (per-instance) prompts | Instance-level TTA |
| **VCP-CLIP** | ECCV 2024 | Visual context prompting | Language-free 방향 |

**Insight**: CLIP 기반 방법의 트렌드는 **text encoder 의존도 감소**. VisualAD (CVPR 2026)는 text encoder 제거해도 성능 동등. "Language is unnecessary for AD."

### 1.3 Language-Free / Vision-Only (2026 신흥)

| 논문 | Venue | 핵심 | 의미 |
|------|-------|------|------|
| **VisualAD** | CVPR 2026 | 2 learnable tokens + frozen ViT, text encoder 제거 | Trainable params >99% 감소 |
| **UniADet** | arXiv 2026 | Language encoder의 역할 = decision weight → 0.002M params로 대체 | "Embarrassingly simple" |

**Insight**: 2026년의 핵심 발견 — **CLIP의 text encoder는 AD에서 단순히 decision boundary를 정의할 뿐**, vision feature 자체가 anomaly 정보를 충분히 담고 있음.

### 1.4 Training-Free FM-AD

| 논문 | Venue | 핵심 | 성능 |
|------|-------|------|------|
| **SubspaceAD** | CVPR 2026 | PCA on DINOv2 patch features, normal subspace residual = anomaly | 1-shot MVTec 98.0% |
| **MuSc** | ICLR 2024 | Mutual scoring between unlabeled test images (no train needed) | Zero-shot SOTA |
| **SuperAD** | VAND 2025 | Training-free DINOv2 memory bank, MVTec AD 2 challenge | Shift에 가장 robust |

**Insight**: **SubspaceAD가 우리와 가장 개념적으로 가까움** — PCA로 normal subspace를 모델링하고 residual로 scoring. 차이: 그들은 normal subspace를 정의, 우리는 nuisance subspace를 제거. 관점이 반대이지만 수학적으로 관련.

### 1.5 DINOv3 시대 개막 (2026)

| 논문 | 핵심 |
|------|------|
| **DINO-AD** (arXiv 2026) | Frozen DINOv3 + clustering-based AD |
| **Spatial AR on DINOv3** (arXiv 2026) | 2D autoregressive CNN으로 patch 간 spatial dependency 모델링 |
| **AD-DINOv3** (arXiv 2025) | Lightweight adapter로 DINOv3 calibration |
| **FoundAD** (ICLR 2026) | DINOv3 지원, manifold projection |

**Insight**: DINOv3(7B params)가 새 backbone 표준. 우리 방법론의 backbone generality 검증 필요.

---

## 2. Few-shot AD (2024-2026)

### 2.1 주요 접근별 분류

#### VLM + Prompt Learning

| 논문 | Venue | K-shot | 핵심 |
|------|-------|--------|------|
| **PromptAD** | CVPR 2024 | 1-4 | Normal prompt → anomaly prompt 변환 (semantic concatenation) |
| **One-for-All (IIPAD)** | ICLR 2025 | 1-4 | Class-shared prompt generator (instance-adaptive) |
| **AA-CLIP** | CVPR 2025 | 0 | Anomaly-aware CLIP fine-tuning |

**Insight**: Prompt 방법의 한계 — class별 prompt 학습 필요, CLIP spatial resolution 제약.

#### Pure Vision FM (DINOv2)

| 논문 | Venue | K-shot | 핵심 | 성능 |
|------|-------|--------|------|------|
| **AnomalyDINO** | WACV 2025 | 1-4 | Patch kNN on DINOv2, training-free | 1-shot 96.6% |
| **VisionAD** | MM 2025 | 1-4 | NN search + dual augmentation | Single/multi-class SOTA |
| **SubspaceAD** | CVPR 2026 | 1-4 | PCA on DINOv2, subspace residual | 1-shot 98.0% |

**Insight**: **DINOv2 + 단순 NN/PCA가 VLM + prompt engineering을 이김**. "Search is All You Need." 복잡한 multi-modal alignment 없이 vision feature만으로 충분.

#### Residual / Registration 기반

| 논문 | Venue | 핵심 |
|------|-------|------|
| **ResAD** | NeurIPS 2024 Spotlight | Residual feature distribution 학습 → cross-class 일반화 |
| **RegAD+** | TNNLS 2024 | Category-agnostic registration, 공간 정렬 실패 = anomaly |
| **GraphCore** | ICLR 2023 | GNN으로 patch 간 structural relationship 모델링 |

#### Generative Augmentation

| 논문 | Venue | 핵심 |
|------|-------|------|
| **One-to-Normal** | NeurIPS 2024 | Diffusion으로 personalized normal 생성, triplet contrastive 추론 |
| **SeaS** | ICCV 2025 | Few abnormal + normal → 다양한 anomaly 합성 |
| **MetaUAS** | NeurIPS 2024 | Synthetic task meta-learning → 1-prompt universal segmenter |

#### 최신 SOTA (2026)

| 논문 | Venue | K-shot | 핵심 | 의미 |
|------|-------|--------|------|------|
| **FoundAD** | ICLR 2026 | 1-4 | Nonlinear projection onto natural image manifold | Anomaly = manifold deviation |
| **UniVAD** | CVPR 2025 | 1-4 | Component-level matching (structural + logical) | Training-free cross-domain unified |

### 2.2 Few-shot AD 성능 추이 (MVTec AD 1-shot I-AUROC)

```
2023: WinCLIP 93.1% → GraphCore 94.2%
2024: PromptAD 94.4% → AnomalyDINO 96.6% → ResAD 97.0%
2025: One-for-All 97.3% → VisionAD 97.8%
2026: SubspaceAD 98.0% → FoundAD 98.2%
      ────────────── gap closing with full-shot (99.6%) ──────
```

**Insight**: 1-shot에서 98%+ 달성. **Few-shot AD on clean benchmarks는 거의 solved** → 남은 frontier는 **shift robustness**.

---

## 3. Vision-Encoder-Only AD 트렌드

### 3.1 Backbone 진화

```
2022: WideResNet-50 (ImageNet supervised)
2023: ViT (ImageNet/CLIP pretrained)
2024: DINOv2 ViT-B/L (self-supervised) ← 현재 표준
2025: DINOv2-Reg, SigLIP
2026: DINOv3 (7B), DFN                 ← 새 표준
```

### 3.2 Memory Bank vs Subspace vs Parametric

| 접근 | 대표 방법 | 장점 | 단점 |
|------|---------|------|------|
| **Memory Bank** | PatchCore, AnomalyDINO | 단순, training-free | Storage, 느린 NN search |
| **Subspace** | SubspaceAD, Null Subspace PCA | 빠름, 해석 가능 | Linear 가정 |
| **Parametric** | Spatial AR, Gaussian | 빠른 inference | 학습 필요 |
| **Reconstruction** | Dinomaly | Multi-class 강점 | Decoder 필요 |

**Insight**: **Subspace 방법이 2025-2026에 급부상**. SubspaceAD(CVPR 2026)가 PCA만으로 memory bank을 이김. 우리의 NSP와 같은 "projection 후 scoring" 패러다임.

### 3.3 Scoring 방법 진화

```
kNN distance (PatchCore) → Mahalanobis distance → PCA residual (SubspaceAD)
                                                  → Manifold projection (FoundAD)
                                                  → Null subspace distance
```

**Insight**: **Scoring이 단순 distance에서 subspace-aware distance로 진화 중**. 이것이 우리의 nuisance subspace projection이 자연스럽게 들어갈 자리.

---

## 4. 우리 연구 포지셔닝

### 4.1 Gap Map (업데이트)

```
                     Shift-Robust?
                     No              Yes
                 ┌────────────────┬──────────────────┐
  Subspace /     │ SubspaceAD     │                  │
  Projection     │ FoundAD        │  ★ OUR WORK ★   │
  scoring        │ Null Subspace  │                  │
                 ├────────────────┼──────────────────┤
  Memory         │ PatchCore      │ SuperAD          │
  Bank           │ AnomalyDINO    │ (training-free)  │
                 ├────────────────┼──────────────────┤
  Reconstruction │ Dinomaly       │ FiCo             │
                 │ EfficientAD    │ (shift augment)  │
                 └────────────────┴──────────────────┘
```

**★ OUR WORK**: Subspace/Projection scoring × Shift-robust = **0건**

### 4.2 가장 가까운 논문들과의 차별화

| 논문 | 유사점 | 차이점 |
|------|--------|--------|
| **SubspaceAD** (CVPR 2026) | PCA on FM features, subspace-based AD | Normal subspace 정의 vs **nuisance subspace 제거** |
| **FoundAD** (ICLR 2026) | Manifold projection on FM features | Normal manifold projection vs **shift direction projection** |
| **Null Subspace PCA** | Null space scoring | Normal null space vs **nuisance-specific projection** |
| **Stylist** (WACV 2025) | Post-hoc nuisance removal on frozen FM | Feature dropping vs **subspace projection** |
| **ResAD** (NeurIPS 2024) | Residual features for cross-class generalization | Class residual vs **shift residual** |

### 4.3 우리만의 고유 contribution 영역

1. **"왜 깨지는가"의 메커니즘 분석**: V-shape, entanglement 정량화 — 기존 연구 0건
2. **Shift robustness + Subspace method**: SubspaceAD류가 normal subspace를 다루지만 shift는 미고려
3. **Impossibility of unpaired**: 5가지 독립 실패 — 이론(Locatello) + 실증 결합
4. **Practical calibration guide**: Saturation curve, per-category > global

---

## 5. 핵심 테이크어웨이

### 방법론 설계에 주는 시사점

1. **Training-free가 대세**: FM feature가 충분히 강력하므로, 학습 없이 scoring만 개선하는 접근이 점점 강해짐. 우리의 post-hoc projection은 이 트렌드에 정확히 부합.

2. **Subspace 방법이 검증됨**: SubspaceAD(CVPR 2026)가 PCA만으로 memory bank을 이긴 것은, feature space의 subspace structure를 활용하는 것이 유효하다는 강한 증거. 우리의 nuisance subspace projection은 이 패러다임의 자연스러운 확장.

3. **Language-free 전환**: Text encoder 없이도 충분. 우리 방법은 vision-only → 트렌드에 부합.

4. **Shift robustness = 미해결 frontier**: Clean에서 98-99%를 달성하는 방법들이 MVTec AD 2에서 급락. 이 gap을 메우는 것이 가장 가치 있는 연구 방향.

5. **DINOv3 검증 필요**: 우리 실험은 DINOv2 ViT-B/14 기반. DINOv3에서도 동일 패턴(V-shape, entanglement)이 재현되는지 검증하면 generality 크게 강화.

### 비교군 확정

| 비교군 | 유형 | 이유 |
|--------|------|------|
| **SubspaceAD** | Subspace scoring (CVPR 2026) | 가장 가까운 방법론 패러다임 |
| **PatchCore** | Memory bank (가장 robust) | Shift robustness baseline |
| **AnomalyDINO** | Few-shot DINOv2 kNN | 우리와 같은 backbone |
| **Stylist** | Post-hoc nuisance removal | 가장 유사한 nuisance 처리 |
| **FiCo** | Shift-aware AD (AAAI 2025) | Shift robustness 직접 비교 |

---

## 관련 노트

- [Related Work Survey (shift/entanglement 초점)](2026-03-30_related_work_survey.md)
- [FM-AD 핵심논문 10선](2026-03-23_FM_AD_robustness_핵심논문.md)
- [MASTER_REPORT](../MASTER_REPORT.md)
