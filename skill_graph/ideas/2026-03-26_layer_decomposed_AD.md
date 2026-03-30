# Layer-Decomposed Anomaly Detection: Residual Hierarchy with Selective Nuisance Removal

**작성일**: 2026-03-26
**Status**: PROPOSAL (미구현)
**Keywords**: layer decomposition, residual anomaly signal, selective nuisance removal, hierarchical scoring

---

## 0. Executive Summary

**한 줄 요약**: Layer 8의 nuisance-removed feature를 "anchor"로 사용하고, 다른 layer의 feature에서 Layer 8이 이미 설명하는 정보를 빼서(residual) texture/structure-specific anomaly signal만 추출하는 방법.

**왜 기존 방법이 안 되는가**:
- Single-layer (L8): robust하지만 texture anomaly에 둔감 (shallow 정보 부재)
- Multi-layer concat (L8+L11): L11의 noise가 L8의 clean signal을 희석 (exp4에서 확인: 75.0% → 72.5%)
- Feature pyramid / multi-scale AD: layer를 동등하게 취급 → shift-sensitive layer가 전체를 오염

**핵심 아이디어**: Layer 간의 관계를 "동등한 정보원"이 아니라 "anchor + residual"로 재정의한다.

---

## 1. 문제 정의: L8+L11 Concat이 실패한 진짜 이유

### 1.1 실패 메커니즘 분석

Phase 3 exp4 결과: Layer 8+11 concat = 72.5% vs Layer 8 단독 = 75.0% (AD2)

**Concat이 실패하는 세 가지 이유**:

1. **Dimension Ratio 불균형**: L8(768d) + L11(768d) = 1536d에서 L11의 768d가 shift-contaminated. Mahalanobis scoring에서 covariance matrix가 contaminated dimension의 variance를 반영 → anomaly score에 shift noise가 직접 유입.

2. **Redundant Information 증폭**: L8과 L11은 같은 네트워크를 거치므로 공유 정보가 많음. Concat하면 공유 정보(주로 semantic identity)가 2배로 가중 → relative anomaly signal 비중 감소.

3. **Nuisance Direction 비정렬**: L8의 nuisance subspace와 L11의 nuisance subspace가 다름. Concat 후 단일 nuisance projection을 적용하면, 각 layer의 nuisance를 최적으로 제거할 수 없음.

### 1.2 이것이 시사하는 원리

> **Layer 간 정보 결합의 올바른 방법은 "합치기"가 아니라 "차이 추출"이다.**

L11이 L8에 비해 추가로 가진 정보 = deep semantic + shift noise
L2가 L8에 비해 추가로 가진 정보 = fine texture + shift noise

문제: 추가 정보에서 useful signal(texture, semantic)과 noise(shift)가 섞여 있음.
해결: Layer-specific nuisance removal 후 residual만 사용.

---

## 2. 제안 방법: Anchor-Residual Hierarchical Detection (ARHD)

### 2.1 핵심 구조

```
Input Image
    ↓
DINOv2 ViT-B/14 (frozen)
    ↓
┌──────────────────────────────────────────────────┐
│  Layer 2 features (f₂)  — texture-rich           │
│  Layer 8 features (f₈)  — shift-robust (ANCHOR)  │
│  Layer 11 features (f₁₁) — semantic-rich          │
└──────────────────────────────────────────────────┘
    ↓
┌──────────────────────────────────────────────────────┐
│  Step 1: ANCHOR SCORING                               │
│    f₈_clean = NSP(f₈, K₈=100)                        │
│    s_anchor = Mahalanobis(f₈_clean, memory_bank₈)     │
│    → Shift-robust base anomaly score                  │
└──────────────────────────────────────────────────────┘
    ↓
┌──────────────────────────────────────────────────────┐
│  Step 2: TEXTURE RESIDUAL EXTRACTION                  │
│    f₂_clean = NSP(f₂, K₂=K₂*)                        │
│    r_texture = f₂_clean - Proj(f₂_clean → span(f₈_clean)) │
│    s_texture = ||r_texture|| (또는 Mahalanobis)       │
│    → L8이 놓친 texture anomaly signal                 │
└──────────────────────────────────────────────────────┘
    ↓
┌──────────────────────────────────────────────────────┐
│  Step 3: SEMANTIC RESIDUAL EXTRACTION                 │
│    f₁₁_clean = NSP(f₁₁, K₁₁=K₁₁*)                  │
│    r_semantic = f₁₁_clean - Proj(f₁₁_clean → span(f₈_clean)) │
│    s_semantic = ||r_semantic||                         │
│    → L8이 놓친 semantic/structural anomaly signal     │
└──────────────────────────────────────────────────────┘
    ↓
┌──────────────────────────────────────────────────────┐
│  Step 4: HIERARCHICAL SCORE FUSION                    │
│    s_final = s_anchor + α·s_texture + β·s_semantic   │
│    α, β = confidence-weighted (아래 2.4 참조)         │
└──────────────────────────────────────────────────────┘
```

### 2.2 각 단계의 수학적 정의

**Notation**:
- `f_l ∈ R^d`: Layer l의 feature (CLS+patch mean, L2 normalized)
- `N_l = {n_l^1, ..., n_l^K_l}`: Layer l의 nuisance basis vectors (shift PCA)
- `M_l ∈ R^{d×d}`: Layer l의 anchor subspace projection matrix

**Step 1: Anchor Scoring** (기존 NSP, 이미 검증됨)
```
f₈_clean = f₈ - Σ_{k=1}^{K₈} (f₈ · n₈ᵏ) n₈ᵏ
s_anchor = (f₈_clean - μ₈)ᵀ Σ₈⁻¹ (f₈_clean - μ₈)
```

**Step 2: Texture Residual**
```
# Layer 2에 대해 별도의 nuisance removal
f₂_clean = f₂ - Σ_{k=1}^{K₂} (f₂ · n₂ᵏ) n₂ᵏ

# f₂_clean에서 f₈_clean이 이미 설명하는 성분 제거
# → Layer 8이 놓친 texture-specific 정보만 남김
P₈ = f₈_clean · f₈_clean^T / (f₈_clean^T · f₈_clean)  (rank-1 approx)
# 또는 memory bank의 principal subspace로 확장:
P₈ = V₈ · V₈ᵀ  (V₈ = top-m eigenvectors of Σ₈)
r_texture = f₂_clean - P₈ · f₂_clean
```

**핵심 insight**: `r_texture`는 세 가지 필터를 통과한 신호다:
1. Layer 2의 shift noise 제거 (NSP)
2. Layer 8과 공유하는 semantic/structural 정보 제거 (anchor projection)
3. 남은 것 = **texture anomaly에만 반응하는 pure signal**

**Step 3: Semantic Residual** (동일 구조, Layer 11)
```
f₁₁_clean = f₁₁ - Σ_{k=1}^{K₁₁} (f₁₁ · n₁₁ᵏ) n₁₁ᵏ
r_semantic = f₁₁_clean - P₈ · f₁₁_clean
```

### 2.3 Layer-Specific Nuisance Removal이 핵심인 이유

Phase 2에서 확인된 layer-wise shift profile:
```
Layer 2:  RelShift = 0.24, CosSim = 0.96  (texture-rich, shift-sensitive)
Layer 8:  RelShift = 0.11, CosSim = 0.99  (robust anchor)
Layer 11: RelShift = 0.26, CosSim = 0.95  (semantic-rich, shift-sensitive)
```

**각 layer의 shift pattern이 다르다**. 따라서:
- L2의 nuisance direction ≠ L8의 nuisance direction ≠ L11의 nuisance direction
- 단일 nuisance subspace로 모든 layer를 처리하면 suboptimal
- **Layer-specific K 값**: L8은 K₈=100 (검증됨), L2와 L11은 독립적으로 sweep 필요

이것이 concat 후 단일 NSP가 실패하는 근본 이유: concat된 공간에서의 nuisance direction은 각 layer의 nuisance direction의 비최적 혼합.

### 2.4 Score Fusion: Confidence-Weighted Combination

단순한 α, β 가중은 위험하다. Shift가 큰 환경에서 s_texture, s_semantic의 신뢰도가 낮아지기 때문.

**제안 1: Shift-Aware Adaptive Weighting**

```python
# 각 layer의 shift magnitude를 test time에 추정
shift_l = ||f_l - f_l_clean|| / ||f_l||  # NSP projection magnitude

# Shift가 작은 layer에 더 큰 가중치
w_anchor = 1.0  # anchor는 항상 신뢰
w_texture = exp(-λ · shift_2)  # L2의 shift가 크면 texture score 신뢰도 하락
w_semantic = exp(-λ · shift_11)  # L11의 shift가 크면 semantic score 신뢰도 하락

s_final = s_anchor + w_texture · s_texture + w_semantic · s_semantic
```

**이 가중치의 의미**:
- Shift가 거의 없는 clean 환경: w_texture ≈ w_semantic ≈ 1 → 모든 layer 활용 → 최대 sensitivity
- Shift가 큰 환경: w_texture → 0, w_semantic → 0 → anchor만 사용 → 최대 robustness
- **환경에 따라 자동으로 sensitivity-robustness tradeoff 조절**

**제안 2: Residual Norm Gating**

Anchor score가 이미 높은 (명확한 anomaly) 영역에서는 residual을 추가할 필요가 없고, anchor score가 경계선인 영역에서만 residual이 의미있다.

```python
# Anchor score의 불확실성이 높은 영역에서만 residual 활성화
gate = sigmoid(τ · (s_anchor - threshold))  # soft gate
s_final = s_anchor + (1 - gate) · (α·s_texture + β·s_semantic)
```

Anomaly가 확실한 곳: gate ≈ 1 → residual 무시 (noise 방지)
경계선 영역: gate ≈ 0 → residual이 결정적 역할

---

## 3. 이 방법이 해결하는 구체적 시나리오

### 3.1 Texture Anomaly on Shift (L8-only의 약점)

**시나리오**: Fabric 카테고리, 조명 변화 환경에서 미세한 scratch 탐지

- L8 feature: shift-robust하지만 texture 정보가 추상화되어 scratch signal 약함
- L2 feature: scratch를 잘 잡지만 조명 shift에 의한 false positive도 높음
- **ARHD**: L2에서 shift를 제거하고(NSP), L8이 이미 설명하는 부분을 제거(anchor projection), 남은 r_texture가 scratch만의 순수 신호

### 3.2 Structural Anomaly (모든 layer에서 잡히는 경우)

**시나리오**: Wallplugs, missing part

- L8과 L11 모두 structural anomaly를 잡음
- anchor score만으로도 충분 → residual의 기여가 작음
- **ARHD**: Residual norm gating이 자동으로 residual 비활성화 → unnecessary noise 없음

### 3.3 Semantic/Logical Anomaly (L8이 놓치는 경우)

**시나리오**: 부품 배치 오류, wrong color

- L8: mid-level feature로 local structure 파악하지만 global arrangement 못 봄
- L11: global semantic context에서 배치 오류 감지 가능
- **ARHD**: r_semantic이 L8이 놓친 global arrangement signal을 보충

---

## 4. Novelty 분석: 기존 방법과의 차별점

### 4.1 기존 Multi-Scale AD와의 차이

| 속성 | Feature Pyramid (기존) | ARHD (제안) |
|------|----------------------|-------------|
| Layer 결합 방식 | Concat / Average | Anchor + Residual |
| Nuisance 처리 | 없음 또는 global 1회 | Layer-specific NSP |
| Layer 간 관계 | 동등 (symmetric) | 비대칭 (anchor vs auxiliary) |
| Shift adaptation | 없음 | Shift-aware weight |
| 정보 중복 처리 | 무시 (중복 = 증폭) | Projection으로 제거 |

### 4.2 기존 NSP (Phase 3)와의 차이

| 속성 | NSP (Phase 3) | ARHD (제안) |
|------|-------------|-------------|
| Layer | Single (L8) | Multi-layer (L2, L8, L11) |
| Anomaly type coverage | L8이 잡는 것만 | Texture + Structure + Semantic |
| Shift 처리 | Single nuisance space | Layer-specific nuisance |
| Score | Single Mahalanobis | Hierarchical fusion |
| Texture anomaly | 약함 | r_texture로 보완 |

### 4.3 Feature Pyramid Network (FPN)과의 차이

FPN은 computer vision에서 multi-scale detection에 널리 사용되지만:
- FPN은 top-down pathway로 semantic info를 shallow layer로 propagate
- ARHD는 반대: bottom-up residual로 shallow layer의 **unique** info를 추출
- FPN은 object detection용 (class-agnostic localization) / ARHD는 anomaly detection용 (class-specific deviation)
- FPN은 nuisance/shift를 고려하지 않음

### 4.4 Novelty Gate 충족 확인

1. **기존 방법의 핵심 한계**: Single-layer NSP는 anchor layer(L8)가 감지 못하는 anomaly type에 blind. Multi-layer concat은 shift noise를 amplify.

2. **제안 방법이 건드리는 메커니즘**: Layer 간 정보의 비대칭성을 활용. Anchor가 이미 설명하는 정보를 제거한 residual에서만 auxiliary signal을 추출.

3. **Contribution 한 줄 요약**: "Layer 간 residual을 통해 shift-robust anchor와 shift-sensitive auxiliary를 안전하게 결합하여, robustness를 유지하면서 anomaly type coverage를 확장하는 최초의 방법"

---

## 5. L8+L11 Concat 실패에 대한 설명과 ARHD의 회피

### 5.1 Concat 실패의 정량적 설명

L8(768d) + L11(768d) = 1536d feature에서:
- L8: shift contamination ≈ 100/768 = 13% (NSP K=100 기준)
- L11: shift contamination ≈ 250+/768 = 32%+ (Phase 2: L11 RelShift = 0.26)
- Concat: shift contamination ≈ (100+250)/1536 = 23%

L8 단독: shift contamination 13%, useful signal 87%
Concat: shift contamination 23%, useful signal 77%

→ Mahalanobis scoring에서 covariance estimation이 23% contaminated dimensions의 영향을 받아 anomaly score 정밀도 하락.

### 5.2 ARHD가 이를 회피하는 방법

ARHD에서 L11의 정보는 r_semantic으로만 유입:
```
r_semantic = NSP(f₁₁) - Proj(NSP(f₁₁) → anchor_space)
```

이 residual의 특성:
1. L11의 nuisance가 NSP로 제거됨 (K₁₁ 별도 최적화)
2. L8과 공유하는 정보(shift에 오염될 수 있는 공유 semantic)가 projection으로 제거됨
3. 남은 차원이 훨씬 적음 → contamination ratio 극소

즉, **concat은 정보를 무차별 합치므로 contamination이 전파**되지만, **ARHD는 clean residual만 선택적으로 추출**하므로 contamination이 차단됨.

---

## 6. 구현 계획 (2주)

### Week 1: Core Implementation + Validation

**Day 1-2: Layer-Specific NSP**
- L2, L11에 대해 별도 shift vector 수집 및 PCA
- K₂, K₁₁ sweep (K=10, 30, 50, 100, 150)
- **검증 기준**: L2_clean, L11_clean 각각의 shift reduction 확인

**Day 3-4: Anchor Projection + Residual Extraction**
- L8 anchor subspace 정의 (memory bank PCA top-m vectors)
- r_texture, r_semantic 계산
- **검증 기준**: residual의 shift magnitude < original의 50%

**Day 5: Hierarchical Scoring**
- 개별 score 계산: s_anchor, s_texture, s_semantic
- Oracle fusion (per-category best weight) → upper bound 확인
- **GO/NO-GO**: oracle fusion이 L8-only(83.8%)보다 유의하게 좋아야 함

### Week 2: Refinement + Ablation

**Day 6-7: Adaptive Weighting**
- Shift-aware weighting 구현 및 λ 튜닝
- Residual norm gating 구현 및 τ 튜닝
- Cross-validation으로 hyperparameter 안정성 확인

**Day 8-9: Ablation Study**
- Component ablation: anchor-only, +texture, +semantic, +both
- Per-anomaly-type 분석: texture vs structural vs logical
- Per-category breakdown

**Day 10: Integration + Paper Figure**
- 결과 정리, figure 생성
- Failure case 분석
- Paper positioning 결정

### 핵심 GO/NO-GO 기준

| 체크포인트 | 기준 | 실패 시 전환 |
|-----------|------|------------|
| Day 2 | L2/L11 layer-specific NSP가 shift를 50%+ 줄임 | Layer-specific NSP 자체가 분석 contribution |
| Day 5 | Oracle fusion > L8-only by 2pp+ on AD2 | Analysis paper로 pivot (L8 optimality 증명) |
| Day 7 | Adaptive weight > fixed weight by 1pp+ | Fixed weight로 단순화 |

---

## 7. 예상 결과 및 위험

### 7.1 낙관적 시나리오 (30%)

| 메트릭 | L8-only (현재 best) | ARHD (예상) |
|--------|-------------------|------------|
| AD2 I-AUROC | 83.8% | 87-90% |
| AD1 I-AUROC | 96.3% | 96-97% |
| Texture anomaly (fabric) | ~75% | ~85% |

**이 경우 논문 구조**: 방법론 60% + 분석 40%

### 7.2 현실적 시나리오 (50%)

| 메트릭 | L8-only | ARHD |
|--------|---------|------|
| AD2 I-AUROC | 83.8% | 85-87% |
| AD1 I-AUROC | 96.3% | 96% |
| Texture anomaly | ~75% | ~80% |

**이 경우 논문 구조**: 분석 60% + 방법론 40% (why L8 is special + why residual helps for specific types)

### 7.3 비관적 시나리오 (20%)

ARHD ≈ L8-only 또는 worse
→ **이것도 가치 있는 결과**: "Layer 8이 sufficient statistic for anomaly detection under shift"
→ Analysis paper로 pivot: 왜 단일 layer가 최적인가에 대한 정보이론적 설명

### 7.4 위험 요인

1. **Anchor projection이 anomaly signal도 제거**: L8과 다른 layer가 anomaly에 대해서도 correlated → residual에서 anomaly가 사라짐
   - **완화**: projection의 rank(m)를 조절하여 partial projection

2. **Layer-specific NSP의 K 최적화 어려움**: 3개 layer × K sweep = 큰 search space
   - **완화**: L8의 K=100 고정, L2와 L11만 sweep

3. **Score fusion의 hyperparameter sensitivity**: α, β, λ, τ 등 파라미터 많음
   - **완화**: adaptive weighting을 먼저 시도, 실패 시 fixed weight로 단순화

---

## 8. 이론적 배경 연결

### 8.1 Information Bottleneck 관점

Layer 8이 robust한 이유를 information bottleneck(IB)으로 설명:
- ViT의 각 layer는 input에서 output task로 정보를 전달하는 과정
- Layer 8 = IB의 최소점 → minimal sufficient statistic에 가까움
- Nuisance(shift) 정보가 최소 → robust
- 하지만 minimal이므로 일부 anomaly-relevant detail도 손실

ARHD는 IB의 한계를 보완:
- Anchor(L8) = minimal sufficient statistic (robust but lossy)
- Residuals(L2, L11) = IB에서 버려진 정보 중 anomaly-relevant part만 복원

### 8.2 Sufficient Statistics Decomposition

정보이론적으로:
```
I(f_l; Anomaly) = I(f₈; Anomaly) + I(f_l; Anomaly | f₈)
```

- `I(f₈; Anomaly)`: anchor가 제공하는 anomaly 정보 (robust, 이미 확보)
- `I(f_l; Anomaly | f₈)`: anchor가 놓친, layer l만의 추가 anomaly 정보

ARHD의 residual은 정확히 `I(f_l; Anomaly | f₈)`를 추정하는 것.
Concat은 `I(f₈∥f_l; Anomaly)`를 추정 → contamination 없이 I를 늘려야 하지만, shift noise도 함께 증가.

### 8.3 Von Kugelgen Identifiability와의 연결

- DINOv2의 augmentation이 정의한 content/style partition이 layer마다 다르게 나타남
- Shallow layer: style이 지배적 (texture, color → augmentation에 의해 변하는 것)
- Deep layer: content가 지배적 (semantic identity → augmentation에 불변)
- **Layer 8 = content와 style이 가장 잘 분리된 지점** (lowest entanglement → lowest shift)

ARHD의 residual = 한 layer의 "style" 중에서 anchor의 content와 직교하는 성분
= anomaly-relevant texture (content의 일부이지만 anchor에서 누락된 것) + noise
→ NSP가 noise를 선제거하므로, 남는 것은 anomaly-relevant texture

---

## 9. Paper Positioning

### 9.1 One-Paragraph Summary

> Foundation model features for anomaly detection exhibit a layer-wise robustness-sensitivity tradeoff: robust layers (Layer 8 in DINOv2) resist distribution shift but miss texture anomalies, while sensitive layers capture fine-grained defects but suffer from shift contamination. We propose Anchor-Residual Hierarchical Detection (ARHD), which uses the robust layer as an anchor and extracts shift-cleaned residual signals from other layers to capture anomaly types the anchor misses. By applying layer-specific nuisance removal and anchor projection, ARHD safely combines multi-layer information without the contamination that plagues naive concatenation. On MVTec AD 2, ARHD improves over the anchor-only baseline by X pp while maintaining robustness under distribution shift.

### 9.2 Contribution List

1. **분석**: DINOv2 layer-wise shift profile의 non-monotonicity 발견 및 Layer 8의 information bottleneck 해석
2. **방법론**: Anchor-Residual Hierarchical Detection — multi-layer 정보를 contamination 없이 결합하는 원리적 방법
3. **실험**: Multi-layer concat 실패의 정량적 설명 + ARHD의 anomaly-type-specific 개선 증거
4. **Negative results**: 10개 복잡한 방법이 왜 단순한 linear projection에 미치지 못하는지 설명

### 9.3 ICLR Fit

- **분석 depth**: layer-wise shift profile + IB 해석 + concat 실패 설명 = ICLR의 "understanding" 논문 구조
- **Practical method**: ARHD는 분석에서 자연스럽게 도출 = "insight-driven solution"
- **Simplicity as virtue**: training-free, linear operations only = reproducible + interpretable

---

## 10. 관련 노트

- 선행: [Phase 3 결과](../experiments/2026-03-26_phase3_linear_disentanglement/report.md) — NSP + L8 baseline
- 선행: [Phase 2 결과](../experiments/2026-03-26_phase2_feature_shift_analysis/report.md) — Layer-wise shift profile
- 선행: [4팀 사고실험](../analysis/2026-03-27_method_thought_experiment.md) — 방법론 비교
- 이론: [Feature Disentanglement Survey](../analysis/feature_disentanglement/2026-03-23_deep_survey.md)
- 이론: [Mechanism Analysis](../analysis/fm_ad_robustness/2026-03-23_mechanism_analysis.md)
