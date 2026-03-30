# IFR Diagnostic: Inter-layer Feature Residual의 핵심 가정 검증 — 2026-03-28

> **상태**: 🟢 완료
> **실험 ID**: `exp_20260328_ifr_diag`
> **코드**: [`scripts/ifr_diagnostic.py`](../../../scripts/ifr_diagnostic.py)
> **결과**: [`results/ifr_diagnostic/ifr_diagnostic.json`](../../../results/ifr_diagnostic/ifr_diagnostic.json)
> **keywords**: IFR, inter-layer residual, subspace alignment, principal angle, nuisance estimation, negative result

---

## 1. 문제 분석 (Problem Analysis)

### 현상

Paired NSP는 +29.4pp 개선을 달성했으나 MVTec AD 2의 paired data에 의존한다. TTNS(test-time unpaired)는 anomaly contamination으로 완전 실패했다. **FM 내부 구조만으로** nuisance direction을 추정하는 방법이 필요하다.

### 원인 추정

DINOv2 ViT-B/14의 residual connection으로 인해:
```
f_L11 = f_L8 + Δ_9 + Δ_10 + Δ_11    (항등식)
```
Layer 8이 robust(RelShift 0.11)하고 Layer 11이 fragile(0.22-0.27)하므로, Δ_{9-11}이 nuisance를 포함할 것이라는 가설.

**핵심 의문**: Train normal의 Δ가 distribution shift 방향을 포착하는가? Train에는 shift가 없으므로 Δ의 고분산 방향이 shift 방향과 같다는 보장이 없다.

### 관련 선행 실험/분석

- [TTNS GO/NO-GO](../2026-03-27_ttns_go_nogo/report.md) — test-time 추정 실패 (anomaly contamination)
- [FM 내부 구조 분석](../../analysis/fm_ad_robustness/2026-03-28_fm_internal_nuisance_identification.md) — IFR 포함 5가지 아이디어 심층 분석
- [Phase 3 NSP](../2026-03-26_phase3_linear_disentanglement/report.md) — oracle paired NSP +29.4pp
- [방법론 상세](../../analysis/fm_ad_robustness/2026-03-27_methodology_nsp_to_sapp.md) — NSP→SAPP→Test-time 진화

---

## 2. 가설 (Hypothesis)

### 주 가설

> "DINOv2 Layer 9-11이 Layer 8 위에 추가하는 정보(Δ_{9-11})의 PCA 고분산 방향은 paired NSP의 oracle nuisance subspace와 유의하게 정렬되어 있으며, oracle shift variance의 40% 이상을 설명할 수 있다."

### 근거

1. **대수적 분해**: f_11 = f_8 + Σ(Δ_L)은 항등식. L8이 robust, L11이 fragile이면 Δ가 fragility의 원인.
2. **Jacobian 해석**: Deep layer 계산의 Jacobian의 top singular vectors ≈ 최대 증폭 방향 ≈ shift sensitivity 방향.
3. **Residual stream 이론** (Elhage et al., 2021): ViT는 shared residual stream에 additive하게 기록. 각 layer의 기여를 분리 가능.

### 예상 결과

| 지표 | 현재 (random) | 예상 (IFR) | GO 기준 |
|------|-------------|-----------|---------|
| D1: Principal angle | 90° (직교) | < 45° (정렬) | < 60° |
| D2: Shift explained | ~8% (random 100/1536) | > 40% | > 30% |
| D3: Δ direction stability | ? | clean ≈ shifted | mean < 45° |

---

## 3. 실험 설정 (Experiment Design)

### 진단 구조

IFR 본 실험(AD scoring) 전에, 핵심 가정의 타당성을 먼저 검증하는 **사전 진단** 설계.

### 대조군 (Control)

- **NSP oracle basis**: MVTec AD 2 paired shift vectors (315쌍)의 PCA top-100 → oracle nuisance subspace

### 실험군 (Treatment)

| 진단 | 측정 대상 | 의미 |
|------|----------|------|
| **D1: Subspace alignment** | IFR basis vs NSP basis의 principal angles | 같은 방향을 포착하는가? |
| **D2: Shift variance explained** | Oracle shift vectors의 IFR basis projection | Shift 정보가 IFR에 얼마나 포함되는가? |
| **D3: Δ direction stability** | Clean Δ vs Shifted Δ의 PCA 방향 비교 | Δ 자체는 shift에 안정적인가? |

### 데이터 및 설정

```
Backbone: DINOv2 ViT-B/14 (frozen)
Layers: 8, 9, 10, 11
Feature: CLS + patch_mean, L2 normalized (dim=1536)
K: 100 (nuisance dimensions)
Train: 8 categories × 100 images = 800 samples
Oracle pairs: 315 shift vectors (regular↔shifted)
```

### IFR basis 추정 방법

```python
# 1. 각 train image에서 L8, L9, L10, L11 features 추출
# 2. Δ_L = f_L - f_{L-1} for L in {9, 10, 11}
# 3. 모든 Δ를 stack (3*800 = 2400 vectors)
# 4. PCA → top-100 = IFR nuisance basis
```

### 평가 지표

- **D1**: Principal angles (via SVD of cross-product), Grassmann distance
- **D2**: Explained variance ratio of oracle shift vectors projected onto IFR basis
- **D3**: Principal angles between PCA(Δ_clean) and PCA(Δ_shifted) per layer

### 실행 커맨드

```bash
docker exec -w /Volume/RESEARCH/Pilot Project_LG_2nd python3 scripts/ifr_diagnostic.py \
  --data_root_compat /Volume/DATA/mvtec_ad_2 \
  --data_root_original /Volume/DATA/mvtec_ad_2 \
  --K 100 --max_per_cat 100 --max_pairs 500
```

---

## 4. 결과 (Results)

### D1: Subspace Alignment (IFR basis vs NSP oracle basis)

| 지표 | 값 | 해석 |
|------|-----|------|
| Mean principal angle | **74.2°** | 거의 직교 (90° = 완전 직교) |
| Median principal angle | 75.4° | |
| Fraction < 45° | **0.0%** | 100개 방향 중 0개가 정렬 |
| Grassmann distance | 13.08 | 매우 큼 |
| Best angle (PC0) | 48.4° | 가장 정렬된 방향도 45° 초과 |

**Top-20 principal angles** (degrees):
```
48.4, 52.4, 53.3, 54.3, 55.6, 56.0, 56.8, 57.7, 58.3, 59.1,
59.3, 59.9, 61.0, 61.2, 61.9, 62.2, 62.6, 63.3, 63.8, 64.1
```

**판정**: ❌ **FAIL** — IFR subspace와 NSP subspace가 사실상 직교.

### D2: Oracle Shift Variance Explained by IFR Basis

| 지표 | 값 | 해석 |
|------|-----|------|
| Overall explained ratio | **12.6%** | Random K=100의 기대값 ~6.5% 대비 약 2배 |

**Per-component** (각 NSP PC가 IFR subspace에 포함되는 비율):

| NSP PC | IFR에 포착 | 해석 |
|--------|-----------|------|
| PC0 (최중요) | 16.1% | 대부분 IFR 밖 |
| PC1 | 13.2% | |
| PC2 | 13.0% | |
| PC3 | 11.1% | |
| PC4 | 7.1% | |
| PC5-9 | 10.8~13.1% | 일관되게 낮음 |

**참고**: Random K=100/1536 = 6.5% expected. IFR의 12.6%는 random보다는 약간 높지만 의미 있는 alignment은 아님.

**판정**: ❌ **FAIL** — Shift variance의 87.4%가 IFR subspace 밖에 존재.

### D3: Δ Direction Stability (clean vs shifted)

| Layer | Mean angle | Median angle | Fraction < 45° | Magnitude ratio |
|-------|-----------|-------------|----------------|-----------------|
| L9 | **27.0°** | 15.2° | 78% | 1.00 |
| L10 | **24.8°** | 14.4° | 82% | 1.00 |
| L11 | **23.7°** | 13.4° | 82% | 1.00 |

**판정**: ✅ **PASS** — Δ의 방향과 크기 모두 shift에 대해 매우 안정적.

**해석**: Deep layer는 clean이든 shifted든 **같은 방향**으로 정보를 추가한다. 하지만 그 "같은 방향"이 shift direction과 다르다는 것이 D1+D2의 결론.

---

## 5. 결과 분석 (Analysis)

### 가설 검증

> **결론**: ❌ **가설 기각** — IFR basis는 oracle nuisance subspace를 포착하지 못한다. 핵심 가정 "deep layer addition direction ≈ shift-sensitive direction"이 틀렸다.

### 상세 분석

#### 5.1 왜 IFR이 실패하는가

**근본 원인**: Train normal 데이터의 inter-layer residual은 **object identity / semantic refinement** 방향의 분산을 포착하지, **환경 조건 (조명/시점) 변화** 방향의 분산을 포착하지 않는다.

구체적으로:
- **IFR의 top PCs**: 카테고리 간 차이 (can vs fabric vs rice), 텍스처 복잡도 변화, 개체별 형상 차이 등
- **NSP의 top PCs**: 조명 강도 변화, 색온도 shift, overexposure/underexposure 효과 등

이 두 정보 유형은 feature space에서 거의 직교한다 (74.2°).

#### 5.2 Jacobian 해석의 한계

이론적 논증은 "Jacobian J의 top singular vectors = sensitivity direction"이었으나, 이는 **J가 모든 방향의 perturbation에 대해 동일하게 반응**할 때만 성립. 실제로는:

- Clean train variation의 방향 (object identity) → J가 이 방향으로 amplify
- Shift perturbation의 방향 (조명 변화) → J가 다른 방향으로 amplify
- J의 eigenstructure가 입력 방향에 따라 다름 (nonlinearity of attention + LayerNorm)

#### 5.3 D3의 의미: Δ는 안정적이되 "엉뚱한 방향"

D3 결과는 Δ_clean ≈ Δ_shifted (방향/크기 모두). 이는:
- Deep layer의 **computation 자체**는 shift에 robust하다
- 하지만 **Layer 8의 입력이 미세하게 shift** → 이 미세 shift가 deep layer를 통과하면서 **증폭**된다
- 증폭이 Δ의 방향이 아닌, **f_8의 shift 방향에 aligned된 방향**으로 발생

따라서 nuisance = "Δ가 가리키는 방향"이 아니라 "f_8의 shift가 증폭되는 방향" — 이는 f_8의 shift를 관찰하지 않고는 추정 불가능.

#### 5.4 Random 대비 IFR의 미미한 우위 (12.6% vs ~6.5%)

IFR이 random보다 약간 높은 이유: inter-layer residual의 subspace가 전체 feature space(1536차원)의 일부(~100차원 집중)에 집중되어 있고, shift direction도 전체보다는 특정 영역에 집중 → **우연한 부분 겹침**. 의미 있는 alignment이 아님.

### 부수 발견 (Side Findings)

1. **IFR basis가 Δ variance의 98.3%를 설명**: 2400개 Δ vector가 ~100차원 subspace에 강하게 집중. Deep layer addition은 매우 저차원.
2. **NSP basis도 shift variance의 98.6%를 설명**: Shift도 ~100차원 subspace에 집중. 하지만 IFR의 100차원과 다른 100차원.
3. **Δ direction stability**: D3 결과 (mean 24°, mag_ratio 1.00)는 ViT의 deep layer가 **function적으로 안정적**임을 보여줌. Shift가 representation을 바꾸는 것은 deep layer의 computation이 아닌, shallow/mid layer에서의 input shift가 propagation되기 때문.

---

## 6. 피드백 및 다음 단계 (Feedback & Next Steps)

### 교훈 (Lessons Learned)

1. **"Deep layer addition ≠ shift vulnerability"**: Layer 9-11이 추가하는 것(semantic refinement)과 shift에 취약한 것(환경 변화 amplification)은 다른 현상이다. f_11이 fragile한 이유는 Δ의 방향이 아니라 f_8의 미세 shift가 Δ를 통과하며 증폭되기 때문.

2. **Train 내 variation으로 cross-condition shift를 예측하는 것은 불가능**: Object identity variance (train에 존재) ⊥ environmental shift variance (train에 부재). 관찰하지 않은 shift의 방향을 추정하려면 shift 자체를 관찰하거나 (paired data), shift의 물리적 성질을 사전 지식으로 주입해야 (augmentation) 한다.

3. **진단 먼저, 실험 나중**: D1+D2 진단으로 30분 만에 IFR 본 실험의 가치를 판정했다. 본 실험(AD scoring + ablation + 보고서)을 진행했다면 1-2일이 소요되었을 것. 사전 진단이 연구 시간을 크게 절약.

4. **Negative result의 가치**: 이 결과는 "FM 내부 구조만으로는 외부 distribution shift를 예측할 수 없다"는 중요한 negative evidence. 논문의 "왜 unpaired estimation이 근본적으로 어려운가" 섹션의 핵심 증거.

### 누적 negative evidence 정리

| 시도 | 접근 | 결과 | 실패 이유 |
|------|------|------|----------|
| Phase 3 | 10개 nonlinear 변형 | 모두 NSP 이하 | Linear hard projection이 optimal |
| TTNS | Test-time ΔΣ | AD1 30.8% 붕괴 | Anomaly contamination |
| **IFR** | **Train Δ_{9-11} PCA** | **D1=74.2°, D2=12.6%** | **Train Δ ⊥ shift direction** |

공통 패턴: **Paired observation 없이 nuisance direction을 추정하는 모든 시도가 실패**. 이는 우연이 아니라 AD의 one-class 구조에서 기인하는 구조적 한계.

### 다음 단계 제안

1. **분석 논문 (Direction C) 확정 고려**: NSP = oracle (paired 필수), 모든 unpaired 시도 실패 → "왜 paired data가 필수적인가"가 핵심 메시지
2. **Cross-layer Disagreement (Idea 4)**: Subspace projection이 아닌 per-sample adaptive scoring — IFR과 다른 메커니즘이므로 별도 검토 가치 있음
3. **Intrinsic Dimension (QE-D)**: Layer 8 특수성의 기하학적 설명 — 분석 논문 보강용으로 여전히 유효
4. **이 실험의 negative evidence를 논문에 포함**: "§5: Why unpaired nuisance estimation fails" 섹션의 Fig/Table 후보

### _lessons.md 승격 여부

- [x] 승격 필요:
  - "Train inter-layer residual의 PCA direction ≠ distribution shift direction (principal angle 74°, shift explained 12.6%)"
  - "FM 내부 구조만으로 외부 distribution shift를 예측하는 것은 불가능 — paired observation 또는 shift 사전 지식이 필수"
  - "진단 실험(subspace alignment)으로 사전 검증하면 1-2일의 무의미한 실험을 방지할 수 있음"

---

## 관련 노트

- 선행: [TTNS GO/NO-GO](../2026-03-27_ttns_go_nogo/report.md) — 또 다른 unpaired 실패
- 선행: [FM 내부 구조 분석](../../analysis/fm_ad_robustness/2026-03-28_fm_internal_nuisance_identification.md) — IFR 아이디어 도출
- 선행: [Phase 3 NSP](../2026-03-26_phase3_linear_disentanglement/report.md) — oracle baseline
- 선행: [4팀 사고 실험](../../analysis/2026-03-27_method_thought_experiment.md) — "분석이 방법론보다 가치 있다"
- 후속: 분석 논문 방향 확정 결정
- 분석: [비교군/일반성 분석](../../analysis/fm_ad_robustness/2026-03-27_comparison_and_generality_analysis.md)
