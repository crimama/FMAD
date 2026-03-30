# Master Research Report: Robust Few-shot FM-AD under Distribution Shift

> **최종 업데이트**: 2026-03-30
> **프로젝트**: Pilot — Visual Anomaly Detection 연구
> **타겟**: ICLR 2027 제출

---

## 연구 한 줄 요약

FM-AD는 distribution shift에서 25-34pp 하락하며, 이는 nuisance-anomaly entanglement 때문이다. Feature-level에서 nuisance를 제거하려면 shift를 사전 관찰해야 하지만(5가지 unpaired 실패), scoring-level에서 "anomaly는 local, shift는 global"이라는 구조적 prior를 활용하면 shift를 관찰하지 않고도 robustness를 개선할 수 있다 (+22pp, 0 params, any shift type). 현재 이 원리에서 기존에 없는 방법론을 도출하는 것이 핵심 과제.

---

## 0. 근본 원인 분석: Few-shot FM-AD는 왜 Distribution Shift에 취약한가

### 0.1 Few-shot FM-AD가 하는 일

```
1. K장의 정상 reference를 FM(DINOv2)으로 feature 추출 → 정상 분포 모델링 (μ, Σ)
2. Test image의 feature 추출
3. 정상 분포로부터의 거리 → anomaly score
```

핵심: **"정상이 feature space에서 어디에 있는가"를 소수 이미지로 정의하고, 거기서 벗어나면 이상.**

### 0.2 근본 원인 1: FM Feature가 실제 Shift에 "불변"이 아니다

DINOv2는 self-distillation(DINO) + masked image modeling(iBOT)으로 학습된다. DINO의 multi-crop 전략에서 student가 teacher(EMA)의 출력을 모방하며, color jitter, blur, crop 등의 augmentation에 invariant한 representation이 생성된다. 하지만:

```
학습 시 invariance:   crop, color jitter, gaussian blur 등 (학습에 사용된 augmentation)
실제 shift:           조명 변화, 시점 변화, 포커스 변화, 센서 노이즈 등

학습한 invariance ⊄ 실제 shift
→ FM feature는 실제 distribution shift에 대해 "불변"이 아님
→ 같은 정상 물체라도 조건이 바뀌면 feature가 이동
```

### 0.3 근본 원인 2 (핵심): Nuisance-Anomaly Entanglement

FM feature는 여러 정보를 **하나의 벡터에 혼합**하여 인코딩:

```
f(image) = [semantic 정보] + [외관/조건 정보] + [결함 정보] + ...
            "무엇인가"       "어떻게 보이는가"   "이상한가"
```

**이 성분들이 feature space에서 직교하지 않음 = ENTANGLEMENT.**

Shift vector의 방향과 anomaly 특징의 방향이 유사하여 분리가 어렵다:
- **Layer 11**: shift 방향 ≈ anomaly 방향 (correlation **0.53**) → 심각하게 entangled
- **Layer 8**: shift 방향 ⊥ anomaly 방향 (correlation **0.12**) → 자연적으로 분리됨 (유일한 예외)

**결과**: 정상 이미지가 조건 변화로 feature space에서 이동할 때, 그 이동 방향이 anomaly 방향과 겹침 → scoring이 "조건 변화"를 "결함"으로 오인 → **false positive 폭증**.

단, Layer 8은 entanglement이 낮아 이 문제가 덜함 → **Layer 8이 shift-robust한 근본 이유**.

### 0.4 근본 원인 3: Few-shot에서 취약성 증폭

```
Full-shot Σ (200장): 정상 변이 폭이 넓음 → shift가 이 폭 안에 들어올 수 있음
Few-shot Σ (K=4장):  Σ가 매우 좁음 → 조금만 shift해도 "비정상" 판정
```

Few-shot은 "정상의 정의"가 좁아서 동일한 shift에 더 민감하게 반응.

### 0.5 세 원인의 상호작용

```
원인 1: FM feature가 shift에 불변이 아님
  → 같은 물체라도 조건이 바뀌면 feature가 이동
    ×
원인 2: 이동 방향이 anomaly 방향과 entangled (correlation 0.53)
  → 이동이 anomaly로 오인됨
    ×
원인 3: Few-shot에서 정상 범위가 좁음
  → 작은 이동도 범위를 벗어남
    ↓
  결과: 25-34pp 하락
```

**이 중 가장 근본적인 것은 원인 2 (entanglement)**. 원인 1은 피할 수 없고, 원인 3은 setting의 제약. 하지만 움직이더라도 **anomaly 방향과 직교하는 방향으로만 움직이면 문제 없음** (sheet_metal이 그 예: shift 크지만 하락 moderate — shift가 anomaly와 직교하기 때문).

### 0.6 핵심 딜레마: Robustness ↔ Discriminability Trade-off

```
Layer 8:  Entanglement 0.12 → robust하지만 discriminative power 부족 (AD2 59.9%)
Layer 11: Entanglement 0.53 → discriminative하지만 shift에 fragile (AD2 53.0%)

→ 어떤 layer를 쓰든 robustness와 discriminability를 동시에 얻기 어려움
```

이 딜레마를 해결하는 것이 연구의 핵심 과제:
- NSP: L8의 잔여 entanglement를 paired data로 제거 → 83.8% (하지만 paired 필수)
- Patch max: L11을 쓰되 scoring에서 shift 영향 회피 → 75.1% (하지만 기존 기법 조합)
- **미해결**: Entanglement 자체를 shift 관찰 없이 줄이는 원리적 방법

---

## 1. 실증: FM-AD Robustness 실패 + 메커니즘 분석 (Phase 1-2)

> §0의 근본 원인을 실험적으로 실증한 결과. §0이 "왜"의 직관적 요약이라면, §1은 상세한 정량적 evidence.

### 1.1 Phase 1: Robustness Stress Test

3개 SOTA FM-AD가 MVTec AD 2(조명/시점 변화)에서:

| Method | Backbone | MVTec AD | MVTec AD 2 | Drop |
|--------|----------|----------|-----------|------|
| Dinomaly | DINOv2 ViT-B/14-reg | 99.64% | 68.6% | **-31.0pp** |
| AnomalyDINO | DINOv2 ViT-B/14 | 97.7% | 72.7% | **-25.0pp** |
| AnomalyCLIP | CLIP ViT-L/14@336 | 91.5% | 57.9% | **-33.6pp** |

**핵심 발견**:
- **Backbone 무관**: DINOv2든 CLIP이든 모두 실패 → FM 공통 구조적 문제
- **wallplugs: 전 방법 ~42%** (near-random) → 반사면 + 미세결함의 극단적 난이도
- **학습 기반 > 비학습 기반 방법이 더 취약**: 학습이 source domain에 overfitting
- **DINOv2 > CLIP**: 평균 10pp+ 차이

### 1.2 7가지 실패 메커니즘

> FM을 좋은 범용 feature extractor로 만드는 속성(invariance, semantic abstraction)이 **정확히** distribution shift 하에서의 AD를 방해한다. AD는 미세한 편차에 대한 **민감도**를 요구하지만, FM은 그런 편차에 대한 **둔감도**를 최적화한다.

| # | 메커니즘 | 핵심 | 근거 |
|---|---------|------|------|
| 1 | Invariance-as-Information-Destruction | Augmentation이 anomaly-relevant 외관 정보 파괴 | Contrastive learning 이론 |
| 2 | Semantic Dominance | "무엇인지" > "어떻게 보이는지" | Phi-eat (2025) |
| 3 | High-Norm Token Artifacts | 특정 patch norm 7.5배 이상 | SINDER (ECCV 2024) |
| 4 | Inter-Class Interference | 클래스 간 분산이 anomaly 신호 압도 | Real-IAD Variety |
| 5 | **Nuisance-Semantic Entanglement** | **조명/시점 ↔ anomaly 비직교 (corr 0.53)** | **Phase 2 실측 ← 핵심** |
| 6 | Pretraining Distribution Mismatch | ImageNet에 산업 결함 부재 | 도메인 간극 |
| 7 | Token-Level Shift Accumulation | Patch shift가 attention으로 비균일 증폭 | SPAD 실측 (2.6%) |

**#1. Invariance-as-Information-Destruction**: DINOv2는 self-distillation(DINO)의 multi-crop 전략에서 color jitter, blur, solarization 등을 augmentation으로 적용하며, student가 teacher(EMA)의 출력을 모방하는 과정에서 이러한 변환에 invariant한 representation을 학습. 그런데 AD에서는 변색=오염, blur=코팅결함, distortion=구조손상이므로, **anomaly-relevant appearance가 feature space에서 이미 소실**. Self-distillation의 view-invariance 목표와 AD의 fine-grained sensitivity 요구 사이에 근본적 tension.

**#2. Semantic Dominance over Material/Texture**: Phi-eat(2025) 실증 — DINOv2는 `"무엇인가"(object identity) >> "어떻게 보이는가"(texture)`를 우선 인코딩. AD는 정반대(`표면 이상 >> 물체 정체`)를 필요로 함. Feature space의 구조가 AD 요구와 정반대.

**#3. High-Norm Token Artifacts**: SINDER(ECCV 2024) — DINOv2의 특정 patch token이 norm 434 vs 정상 57.6(7.5배). 이미지 독립적 방향(pairwise angle 3.1°), weight matrix의 leading singular vector에서 기인. Distance 기반 scoring이 이 artifact에 지배되어, 환경 변화 시 anomaly와 무관한 score 변동 유발.

**#4. Inter-Class Interference**: Unified multi-class AD에서 15→160+ 카테고리 시 10-20pp 추가 열화(Real-IAD Variety). FM feature가 category-level semantic을 강하게 인코딩 → `inter-category variance >> intra-category anomaly signal` → 미세한 anomaly가 noise에 파묻힘.

**#5. Nuisance-Semantic Entanglement** ← **이 연구의 핵심 초점**: 조명/시점 변화가 feature space에서 anomaly 방향과 직교하지 않고 겹침. `Feature distance = f(anomaly) + f(shift)`, 이 둘이 entangled. **우리 실험에서 직접 정량화**: Layer 11에서 PC projection correlation 0.53, Top-10 PC 중 40%가 shift+anomaly 공유. NSP의 이론적 동기이자 paired data가 필수적인 이유. 기존 연구에서 이를 명시적으로 다룬 논문 = **0건**.

**#6. Pretraining Distribution Mismatch**: ImageNet에 산업 결함(scratch, dent, contamination)이 극히 희소. Feature space가 결함 관련 영역에서 저해상도. MVTec AD에서 잘 되는 것은 "유사 object가 ImageNet에 있어서" 우연일 수 있음. 실제 산업 환경에서 feature quality 급락.

**#7. Token-Level Shift Accumulation**: 환경 변화 시 각 patch token이 local context로 소폭 shift → attention cross-patch interaction이 shift를 **비균일하게 증폭** → spatial grid 전체에서 누적 → 정상 분포로부터 체계적 이탈. **SPAD 진단에서 확인**: feature shift의 spatial uniformity = 2.6%. 같은 pixel-level 조명 변화라도 patch마다 다르게 인코딩됨.

### 1.3 V-shaped Layer Robustness Profile

DINOv2 12개 layer의 shift sensitivity를 측정한 결과, **비단조적 V-자형 패턴** 발견:

```
Layer:     0    1    2    3    4    5    6    7   [8]   9   10   11
RelShift: .22  .25  .24  .24  .26  .27  .25  .27 [.11] .15  .22  .26
```

**Layer 8만 유일하게 shift-robust** (RelShift 0.11, 타 layer 0.22-0.27).

### 1.4 Entanglement 정량화

| Layer | Entanglement (PC correlation) | 해석 |
|-------|-------------------------------|------|
| Layer 11 | **0.53** | Top-10 PC 중 40%가 shift+anomaly 공유. 심각한 entanglement |
| **Layer 8** | **0.12** | 자연적으로 분리됨 — Layer 8이 robust한 이유 |

**t-SNE 확인**: Shifted-normal이 anomaly 방향으로 이동 (centroid distance 1.09) → false positive의 직접 원인.

**카테고리별 패턴**: Shift의 **크기뿐 아니라 방향**이 중요. sheet_metal은 shift가 크지만(0.33) 하락이 moderate(-20pp) — shift가 anomaly와 직교하기 때문.

---

## 2. Oracle Solution: Paired NSP

### 2.1 알고리즘

```
1. Paired data: 같은 물체의 clean/shifted 이미지 수집
2. Shift vector: d_i = f(obj_i, shifted) - f(obj_i, clean)  ← 물체 정보 상쇄, 순수 shift
3. PCA on {d_i} → 상위 K개 = nuisance subspace
4. f_clean = f - N @ N^T @ f   (hard orthogonal projection)
5. Mahalanobis anomaly scoring on projected features
```

**특성**: 학습 파라미터 0개, 순수 post-hoc 기하학적 연산.

### 2.2 결과

```
Best Config:
  Layer: 8, K: 100, Scoring: Mahalanobis, Feature: CLS+patch_mean L2norm
  MVTec AD 2: 83.8% (+29.4pp from 54.4% baseline)
  MVTec AD:   96.3% (-0.2pp from clean)
  Per-category: 87.4% (per-cat > global +3pp)
```

### 2.3 컴포넌트별 기여 분해 (Ablation)

| 변경 | 독립 기여 | 누적 AD2 | 역할 |
|------|----------|---------|------|
| kNN baseline (L11) | — | 54.4% | 출발점 |
| + NSP K=30 | +13.5pp | 67.9% | **핵심**: nuisance 제거 |
| + CLS + L2 norm | +0.1pp | 68.0% | Feature 표현 |
| + Mahalanobis | +3.4pp | 71.4% | Projected covariance 활용 |
| + Layer 8 | +3.6pp | 75.0% | Robust layer 선택 |
| + K=100 | +8.8pp | **83.8%** | 더 많은 nuisance 제거 |

**K Saturation**: K=30→50→100에서 단조 증가, saturation 미도달. Clean 성능은 K 무관 안정(96.3±0.1%).

### 2.4 Linear Hard Projection이 Optimal — 14개 변형 전멸

| 카테고리 | 시도 (14개) | 대표 결과 | 교훈 |
|---------|-----------|---------|------|
| **Estimation 변경** (4개) | condition-union, whitened, train-only, adaptive K | 모두 ≤ 83.8% | Standard PCA가 최적 |
| **Projection 변경** (4개) | soft, signal-only, SAPP ω-gated, hybrid filter | 모두 < 83.8% | Hard 0/1 projection이 최적 |
| **Scoring 변경** (2개) | PCA-whitened Mahal, dual-space | 모두 < 83.8% | Standard regularized Mahal 최적 |
| **Layer 결합** (2개) | L8+L11 concat, multi-layer fusion | 72.5% | Layer 선택 > fusion |
| **Adaptive** (2개) | per-category K, per-direction gating | ≤ 77.8% | Uniform이 최적 |

**SAPP (variance-gated projection)**: L8에서 77.8%(-6.0pp), L11에서 70.2%(-8.9pp). ω spread ratio(3.3-4.8x)가 좁아 gating 차별력 없음.

**핵심 결론**: 복잡성을 추가하면 **항상** 성능이 하락한다. 이 문제의 structure가 linear이라는 강한 증거.

---

## 3. Unpaired Nuisance Estimation — 5가지 전멸

NSP는 MVTec AD 2의 paired data에 의존 → 일반화 불가. 5가지 unpaired 접근을 시도, **모두 실패**.

### 3.1 TTNS — Test-time Covariance Shift ❌

```
ΔΣ = Σ_test - Σ_train → positive eigenvalue directions → nuisance 추정
```

| Method | AD2 | AD1 | 판정 |
|--------|-----|-----|------|
| Baseline | 59.9% | 96.5% | — |
| TTNS-cov | **50.6%** | **30.8%** | ❌ AD1 붕괴 |
| TTNS-combined | **46.3%** | **24.9%** | ❌ 최악 |
| TTNS-mean | 58.3% | 93.0% | ❌ 미미 |

**실패 원인**: Test batch = normal + anomaly 혼합. ΔΣ의 positive eigenvalue에 anomaly variance 포함 → 제거 시 anomaly detection 자체 파괴.

**핵심 교훈**: "분산 증가 = nuisance"라는 가정이 AD에서 근본적으로 틀림. Anomaly도 분산을 증가시키기 때문.

### 3.2 IFR — Inter-layer Feature Residual ❌

```
Δ_{9-11} = f_L11 - f_L8 → PCA on train Δ → nuisance 추정
```

**30분 사전 진단으로 NO-GO 확정** (본 실험 불필요):

| 진단 | 결과 | GO 기준 | 판정 |
|------|------|---------|------|
| D1: IFR basis vs NSP basis principal angle | **74.2°** | < 60° | ❌ |
| D2: Oracle shift explained by IFR | **12.6%** | > 30% | ❌ |

**확인 실험**: K=100으로 residual variance의 94-98% capture → AD2 59.5% (baseline과 동일). Residual variance ≠ shift-relevant variance.

**실패 원인**: Train normal의 inter-layer residual = semantic refinement 방향. Distribution shift 방향과 거의 직교(74.2°). FM 내부 구조가 외부 shift를 예측하지 못함.

**부수 발견 (D3)**: Δ 방향은 clean↔shifted에서 안정적 (mean 24°, mag ratio 1.00). Deep layer는 같은 방향으로 정보를 추가하되, 그 방향이 shift 방향과 다를 뿐. → "Deep layer addition ≠ shift vulnerability"

### 3.3 AdaBN — Adaptive Batch Normalization ❌

| Method | AD2 | AD1 |
|--------|-----|-----|
| AdaBN | 58.3% | **78.5%** (-18pp) |

**실패 원인**: 전체 feature 정규화가 anomaly signal도 함께 훼손. Global correction ≠ directional correction.

### 3.4 NN Pseudo-Pairing ❌

```
For test x: NN(x) in train at L8 → pseudo_shift = f(x) - f(NN(x))
Robust PCA on pseudo_shifts → nuisance basis → hard projection
```

**사전 진단(D0-D2)**:
| 진단 | 결과 | 판정 |
|------|------|------|
| D0: SNR (shift/nn_dist) | 0.69 | ✅ |
| D1: Mean principal angle | 50.6° | ✅ |
| D2: Oracle shift explained | **87.6%** | ✅ |

**진단은 통과(87.6%)했으나 본 실험에서 실패**:

| Method | AD2 | vs Baseline |
|--------|-----|------------|
| Baseline | 59.9% | — |
| **NN Pseudo K=100 rob80** | **55.7%** | **-4.2pp** |
| NN Pseudo K=50 rob80 | 50.6% | -9.3pp |

**K sweep**: 어떤 K에서도 baseline 초과 불가. K↑ → anomaly 정보 파괴, K↓ → 과공격적 제거.

**실패 원인**: Per-sample pseudo-shift는 normal에 대해 양호하나, **PCA aggregation 단계에서 anomaly pseudo-shift가 nuisance basis를 오염**. 80th percentile robust filtering으로도 불충분 (AD2 test의 anomaly ratio ~40-60%).

**핵심 교훈**: Normal-only 진단은 필요 조건이지 충분 조건이 아님. AD pipeline의 anomaly 혼합 조건은 별도 검증 필요.

### 3.5 SPAD — Spatial Mean Subtraction ❌

Feature shift의 spatial uniformity를 측정한 결과 = **2.6%**. Shift가 patch마다 다르게 작용 → spatial mean subtraction +0.5pp only.

**실패 원인**: Pixel space에서 균일한 shift라도 feature space에서는 patch마다 다르게 인코딩 (각 patch의 semantic content에 따라). "Global shift를 spatial mean으로 제거"하는 가정이 feature space에서 성립하지 않음.

### 3.6 수렴하는 결론

| 경로 | 실패 메커니즘 |
|------|-------------|
| Test data 사용 (TTNS, NN Pseudo) | **Anomaly가 추정을 오염** |
| Train data만 사용 (IFR) | **Shift 정보 자체가 부재** |
| FM 내부 구조 (IFR D3) | **External shift를 예측 불가** |
| Global correction (AdaBN, SPAD) | **Anomaly signal도 파괴** |

> **Feature-level에서 nuisance direction을 추정하려면 paired observation(같은 물체, 다른 환경)이 필수.**

---

## 4. Calibration 효율성: 얼마나 필요한가

### 4.1 Minimal Calibration Saturation Curve (Full-shot)

| N (paired samples) | AD2 I-AUROC | Recovery | 의미 |
|--------------------|------------|----------|------|
| 0 (baseline) | 59.9% | 0% | No projection |
| 5 | 60.6% | 3% | 미미 |
| 10 | 62.0% | 9% | K > N → 대부분 비활성 |
| 20 | 64.6% | 19% | |
| **50** | **68.0%** | **33%** | 첫 의미 있는 개선 |
| **100** | **75.0%** | **62%** | **실용적 최소선** (~12쌍/category) |
| Full (315) global | 84.4% | 100% | Global oracle |
| **Full per-category** | **87.4%** | **112%** | **Best: per-cat > global +3pp** |

**핵심 insight**:
- **고차원 nuisance subspace (K≥100)**: PCA sample complexity로 N ≫ K 필요. "5-10쌍이면 충분"이라는 기대와 불일치.
- **Per-category > Global (+3pp)**: 카테고리마다 shift 구조가 다름 → category-specific calibration이 효율적.
- **Graceful degradation**: N < K이면 effective K = N-1로 자동 축소, cliff 없는 점진적 하락.

### 4.2 Few-shot Multi-condition NSP

K장의 reference를 C가지 조건에서 촬영 → K×(C-1) shift vectors → NSP.

| K (shots) | C=1 (no calib) | C=2 | C=3 | **C=6 (all)** | NSP Gain (C=6) |
|-----------|---------------|-----|-----|-------------|----------------|
| 1 | 61.6% | 61.6% | 62.3% | **65.2%** | +5.1pp |
| 2 | 62.0% | 61.9% | 62.6% | **65.9%** | +5.6pp |
| 4 | 72.2% | 72.4% | 73.3% | **82.2%** | **+12.7pp** |
| 8 | 77.0% | 77.1% | 78.0% | **92.1%** | **+18.5pp** |
| 16 | 77.7% | 78.0% | 79.1% | **98.1%** | **+22.3pp** |

**⚠️ Critical caveat**: C=6은 test shift type을 정확히 관찰한 **oracle 설정** (MVTec AD 2의 6가지 condition을 모두 사전에 알고 있음). C=2에서는 +0.1pp, C=3에서 +1pp — **미리 모르는 shift type에는 효과 미미.**

**관찰하지 않은 shift는 제거할 수 없다** — 이는 당연하지만, 논문에서 method로 제안하려면 이 한계를 넘어서야 함.

---

## 5. Shift-Robust Scoring: Structural Prior 기반 접근

### 5.1 핵심 원리: "Anomaly는 Local, Shift는 Global"

Feature-level nuisance 제거가 불가능하다면, **scoring-level에서 구조적 prior를 활용**:
- Distribution shift: 모든 patch에 (비균일하게나마) 영향 → 전반적 score elevation
- Anomaly: 특정 patch만 이상 → 국소적 score spike

→ Aggregation과 normalization 전략으로 global elevation을 무시하고 local spike를 잡아냄.

### 5.2 Scoring 전략 Ablation (8-shot, no paired data, no shift knowledge)

| Level | Aggregation | Normalization | L8 | **L11** |
|-------|------------|--------------|-----|---------|
| Image | Mahalanobis | — | 53.8% | 53.0% |
| Patch | mean | none | 53.4% | 54.4% |
| Patch | mean | median_sub | 58.4% | 64.1% |
| Patch | mean | mad_norm | 58.6% | **64.5%** |
| Patch | mean | q25_sub | 60.2% | 59.3% |
| Patch | p95 | none | 59.6% | 57.8% |
| Patch | p95 | mad_norm | 53.6% | **64.0%** |
| Patch | p99 | none | **62.3%** | **67.2%** |
| Patch | p99 | q25_sub | 62.5% | **69.6%** |
| Patch | p99 | median_sub | 60.9% | **69.5%** |
| Patch | max | none | 62.3% | **72.9%** |
| Patch | max | median_sub | 60.1% | **74.6%** |
| **Patch** | **max** | **q25_sub** | 62.4% | **75.1%** |

**Best: L11 patch max + q25_sub = 75.1% (+22.1pp vs image baseline)**

### 5.3 해석

1. **Aggregation 효과**: mean(54.4%) → max(72.9%) = +18.5pp at L11. Max가 global shift floor를 자연적으로 무시.
2. **Score normalization 효과**: max(72.9%) → max+q25_sub(75.1%) = +2.2pp. Per-image q25 제거가 shift floor를 추가 제거.
3. **L11 > L8**: L11의 patch features가 더 discriminative. Shift floor만 제거하면 discriminative power가 살아남.
4. **Normalization은 mean에서 효과적, max에서 제한적**: Max 자체가 이미 shift floor를 무시하므로 추가 normalization의 marginal gain이 작음.

### 5.4 현재 한계

**+22.1pp는 기존 기법(PatchCore max, AnomalyDINO quantile)의 조합**이지 새로운 원리가 아님. 논문 contribution으로 제출하려면:
- "왜 patch max가 shift-robust한가"의 이론적 formalization
- 기존 기법을 넘어서는 원리적 개선
- 또는 분석 논문으로서의 가치 (entanglement + impossibility + structural prior 설명)

---

## 6. 연구 흐름도

```
Phase 1: "FM-AD가 shift에서 깨지는가?"
    → ✅ 3종 모두 -25~34pp 하락 확인
    ↓
Phase 2: "왜 깨지는가?"
    → ✅ Entanglement(L11=0.53) + V-shaped profile + Layer 8(0.12)
    ↓
Phase 3: "고칠 수 있는가?" (paired data)
    → ✅ NSP +29.4pp, 0 params, linear optimal (14 variants 전멸)
    ↓
Phase 4: "Paired data 없이 고칠 수 있는가?" (feature-level)
    → ❌ 5가지 unpaired 전멸 (TTNS, IFR, AdaBN, NN Pseudo, SPAD)
    → 교훈: Feature-level nuisance 제거에는 shift 관찰 필수
    ↓
Phase 4b: "얼마나 많은 paired data가 필요한가?"
    → Saturation curve: 100쌍→75%, per-cat>global
    ↓
Phase 4c: "Few-shot에서 multi-condition calibration?"
    → K=8 C=6: 92.1% — 하지만 shift type oracle (cheating 문제)
    → C=2,3에서는 +0.1~1.3pp (미미)
    ↓
Phase 4d: "Patch-level NSP?"
    → Global patch NSP 69.5% (+10.8pp), walnuts에서 image NSP 초과
    → 하지만 전체 평균은 image NSP 열등 (scoring 방식 차이)
    ↓
Phase 4e: "Scoring-level structural prior?"
    → ★ Patch max + q25_sub: 75.1% (+22.1pp, no paired data)
    → "Anomaly=local, shift=global" 원리가 작동
    → 하지만 기존 기법 조합, novelty 부족
    ↓
Phase 5: "이 원리에서 genuinely novel한 방법을 도출할 수 있는가?" ← 현재
```

---

## 7. 현재 상태 및 다음 단계

### 7.1 진행 상태

| Phase | 상태 | 핵심 산출물 |
|-------|------|-----------|
| Phase 1: Baseline Reproduction | **✅ 완료** | 3 methods × 2 benchmarks, -25~34pp |
| Phase 2: Mechanism Analysis | **✅ 완료** | V-shape, entanglement, Layer 8 |
| Phase 3: NSP Oracle | **✅ 완료** | +29.4pp, 14 variants failed → linear optimal |
| Phase 4: Unpaired Estimation | **✅ 완료** | 5가지 전멸 → impossibility |
| Phase 4b: Minimal Calibration | **✅ 완료** | Saturation curve |
| Phase 4c: Few-shot NSP | **✅ 완료** | K=8 C=6: 92.1% (oracle), C=2: +0.1pp |
| Phase 4d: Patch-level NSP | **✅ 완료** | 69.5% (+10.8pp) |
| Phase 4e: Shift-Robust Scoring | **✅ 완료** | L11 max+q25_sub: 75.1% (+22.1pp) |
| Phase 5: Novel Method / Paper | **⬜ 미착수** | 방향 결정 중 |

**전체 실험 수**: 40+개 (14 keep, 26+ discard)

### 7.2 보강 필요 항목

| 항목 | 우선순위 | 상태 | 비고 |
|------|---------|------|------|
| **Novel method (structural prior 기반)** | **최고** | 탐색 중 | 기존 기법 조합을 넘어서는 원리적 방법 |
| **PatchCore 비교군** | 높 | 미실시 | 가장 robust한 기존 방법 비교 필수 |
| **RobustAD 평가** | 중 | 미실시 | AD2 외 데이터셋 일반성 |
| Seed 민감도 | 중 | 미실시 | 주요 실험 3-seed 평균 |
| 다른 backbone (CLIP, ViT-L) | 중 | 미검증 | V-shape 일반성 |
| Per-category detailed analysis | 낮 | 부분 확보 | wallplugs/can 심층 분석 |

### 7.3 Open Questions

1. ~~FM-AD가 shift에서 깨지는가?~~ → **✅ -25~34pp** (Phase 1)
2. ~~왜 깨지는가?~~ → **✅ Entanglement + V-shape** (Phase 2)
3. ~~고칠 수 있는가?~~ → **✅ Paired NSP +29.4pp** (Phase 3)
4. ~~Unpaired로 고칠 수 있는가?~~ → **❌ 5가지 전멸** (Phase 4)
5. ~~Shift는 spatially uniform한가?~~ → **❌ 2.6%** (Phase 4e)
6. ~~Few-shot multi-condition으로?~~ → **C=6: 92.1%이나 shift type oracle** (Phase 4c)
7. **"Anomaly=local, shift=global" 원리에서 genuinely novel한 method는?** ← **현재 핵심 질문**
8. **PatchCore와의 직접 비교에서 우위 가능한가?** ← 미검증

---

## 8. Verified Patterns (15개)

| # | 패턴 | 함의 |
|---|------|------|
| 1 | FM Feature Invariance ≠ AD Robustness | Contrastive invariance가 fine-grained 정보 파괴 |
| 2 | MVTec AD 성능 ≠ Robustness | 99%+ → AD2에서 60% 추락 |
| 3 | Layer 8 = V-shape sweet spot | DINOv2 layer robustness는 비단조, L8만 robust |
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
| 15 | "Deep layer addition ≠ shift vulnerability" | IFR D3: Δ 안정적이나 shift와 무관 |

---

## 9. 리스크 평가

| 리스크 | 상태 | 대응 |
|--------|------|------|
| FM-AD가 예상보다 robust | **해소**: -25~34pp 실증 | — |
| Linear disentanglement 불충분 | **해소**: +29.4pp | — |
| Clean 성능 하락 | **해소**: AD1 96.3% | — |
| 기존 방법과 차별화 부족 | **해소**: 기존 연구 0건 | — |
| Unpaired fix 가능 | **해소**: 5가지 전멸 | — |
| MVTec AD 2 의존성 | **해소**: impossibility + scoring approach | — |
| **Novel method 부재** | **높** | 현재 탐색 중 |
| **비교군 부재 (PatchCore)** | **높** | 필수 확보 예정 |
| **RobustAD 일반성 미검증** | **중** | Phase 5 전 실시 |
| **Seed 민감도** | **낮** | 3-seed 평균 필요 |

---

## 10. 참고 문헌

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

---

## 부록 A: 전체 실험 이력

### 성공한 실험 — Image-level NSP

| 실험 | AD2 | AD1 | 기여 |
|------|-----|-----|------|
| Per-cat NSP full | **87.4%** | — | Per-category oracle best |
| Global NSP full(315) | 84.4% | — | Global oracle |
| NSP K=100 L8 Mahal | 83.8% | 96.3% | Autoresearch best |
| NSP K=50 L8 | 78.5% | 96.3% | K 효과 |
| NSP K=30 L8 | 75.0% | 96.4% | Layer 효과 |
| NSP K=30 L11 Mahal | 71.4% | 97.4% | Scoring 효과 |

### 성공한 실험 — Minimal Calibration

| N (쌍) | AD2 | Recovery |
|--------|-----|---------|
| 10 | 62.0% | 9% |
| 50 | 68.0% | 33% |
| 100 | 75.0% | 62% |
| Full (315) | 84.4% | 100% |

### 성공한 실험 — Few-shot NSP

| K × C | AD2 | NSP Gain |
|-------|-----|----------|
| K=4 × C=6 | 82.2% | +12.7pp |
| K=8 × C=6 | **92.1%** | +18.5pp |
| K=16 × C=6 | **98.1%** | +22.3pp |

### 성공한 실험 — Shift-Robust Scoring (unpaired)

| Method | Layer | AD2 | vs img baseline |
|--------|-------|-----|----------------|
| Patch max + q25_sub | L11 | **75.1%** | **+22.1pp** |
| Patch max + median_sub | L11 | 74.6% | +21.6pp |
| Patch max (no norm) | L11 | 72.9% | +19.9pp |
| Patch p99 + q25_sub | L11 | 69.6% | +16.6pp |
| Patch mean + mad_norm | L11 | 64.5% | +11.5pp |

### 실패한 실험 — 방법론 변형 (14개)

| 변형 | AD2 | 실패 유형 |
|------|-----|----------|
| SAPP ω-gated L8 | 77.8% | Gating < hard (-6pp) |
| Multi-layer L8+L11 | 72.5% | Fusion < selection |
| Soft projection | 72.5% | Soft < hard |
| Dual-space α=0.7 | 74.8% | Original 오염 |
| Condition union | 77.1% | Standard PCA가 더 나음 |
| + 9개 추가 | 모두 열등 | 복잡성 ↑ = 성능 ↓ |

### 실패한 실험 — Unpaired 추정 (5가지 + controls)

| 접근 | AD2 | AD1 | 실패 메커니즘 |
|------|-----|-----|-------------|
| TTNS-cov | 50.6% | **30.8%** | Anomaly ⊂ ΔΣ |
| TTNS-combined | 46.3% | **24.9%** | Worst: anomaly 방향 파괴 |
| IFR K=100 | 59.5% | 97.5% | Train Δ ⊥ shift (74.2°) |
| AdaBN | 58.3% | **78.5%** | Global normalization |
| NN Pseudo K=100 | **55.7%** | — | PCA aggregation contamination |
| SPAD subtraction | 66.8% | — | Spatial non-uniformity (2.6%) |
| Random K=100 | 59.8% | 96.4% | 방향 무의미 (control) |

## 부록 B: 실험 보고서 인덱스

| 보고서 | Phase | 상태 | 핵심 결과 |
|--------|-------|------|----------|
| [Phase 1: Baseline](experiments/2026-03-26_phase1_baseline_reproduction/report.md) | 1 | ✅ | 3 SOTA × 2 benchmarks, -25~34pp |
| [Phase 2: Feature Analysis](experiments/2026-03-26_phase2_feature_shift_analysis/report.md) | 2 | ✅ | V-shape, L8, entanglement 0.53 |
| [Phase 3: NSP](experiments/2026-03-26_phase3_linear_disentanglement/report.md) | 3 | ✅ | +29.4pp, 14 variants failed |
| [SAPP GO/NO-GO](experiments/2026-03-27_sapp_go_nogo/report.md) | 3b | ❌ | ω gating -6~9pp vs NSP |
| [TTNS GO/NO-GO](experiments/2026-03-27_ttns_go_nogo/report.md) | 4 | ❌ | AD1 30.8% — anomaly contamination |
| [IFR Diagnostic](experiments/2026-03-28_ifr_diagnostic/report.md) | 4 | ❌ | 74.2° — train Δ ⊥ shift |
| [NN Pseudo-Pairing](experiments/2026-03-28_nn_pseudopair/report.md) | 4 | ❌ | D0 pass → -4.2pp — PCA re-contamination |
| [Minimal Calibration](experiments/2026-03-28_minimal_calibration/report.md) | 4b | ✅ | 100쌍→75%, saturation curve |

## 부록 C: 실험 환경

```
Hardware: NVIDIA RTX 4090 (24GB VRAM)
Runtime: Docker (Project_LG_2nd, PyTorch 2.9.1)
Backbone: DINOv2 ViT-B/14 (frozen, 12 layers, dim=768)
Feature: CLS + patch_mean, L2 normalized (dim=1536 for image-level)
         or raw patch tokens (dim=768 × 1369 patches for patch-level)
Dependencies: scikit-learn, torch, tqdm, numpy
CUDA: 12.8, Driver: 570.133.07
```

---

> **관련 파일**
> - `results.tsv` — 전체 실험 결과 이력
> - `skill_graph/experiments/` — 실험 보고서 (8개)
> - `skill_graph/analysis/fm_ad_robustness/` — 분석 노트 (10+ 문서)
> - `skill_graph/analysis/fm_ad_robustness/_lessons.md` — 검증된 패턴 (15개)
> - `skill_graph/ideas/` — 방법론 제안서
> - `scripts/` — 실험 스크립트 (17개)
