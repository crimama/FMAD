# NN Pseudo-Pairing: Unpaired Nuisance Estimation via NN Matching — 2026-03-28

> **상태**: 🟢 완료
> **실험 ID**: `exp_20260328_nnpseudo`
> **코드**: [`scripts/nn_pseudopair_experiment.py`](../../../scripts/nn_pseudopair_experiment.py), [`scripts/nn_pseudopair_diagnostic.py`](../../../scripts/nn_pseudopair_diagnostic.py)
> **결과**: [`results/nn_pseudo/nn_pseudo_results.json`](../../../results/nn_pseudo/nn_pseudo_results.json), [`results/nn_pseudopair_diagnostic/nn_pseudo_diagnostic.json`](../../../results/nn_pseudopair_diagnostic/nn_pseudo_diagnostic.json)
> **keywords**: NN pseudo-pairing, unpaired, nuisance estimation, robust PCA, anomaly contamination, negative result

---

## 1. 문제 분석 (Problem Analysis)

### 현상

Paired NSP는 +29.4pp를 달성하지만 MVTec AD 2의 paired data에 의존한다. 이전 unpaired 시도 3건이 모두 실패:
- TTNS: batch covariance에 anomaly 혼입 → AD1 30.8%
- IFR: train inter-layer residual ⊥ shift direction (principal angle 74.2°)
- AdaBN: 전체 정규화가 anomaly 신호 파괴 → AD1 78.5%

### 원인 추정

TTNS의 근본 문제는 **batch-level 통계**에서 anomaly를 분리할 수 없다는 것. NN pseudo-pairing은 **per-sample** 수준에서 shift를 추정하므로, anomaly pseudo-shift가 **개별 outlier**가 되어 robust PCA로 필터 가능할 것이라는 가설.

### 관련 선행 실험/분석

- [TTNS GO/NO-GO](../2026-03-27_ttns_go_nogo/report.md) — batch-level 실패
- [IFR Diagnostic](../2026-03-28_ifr_diagnostic/report.md) — train Δ ⊥ shift
- [FM 내부 구조 분석](../../analysis/fm_ad_robustness/2026-03-28_fm_internal_nuisance_identification.md)
- [비교군/일반성 분석](../../analysis/fm_ad_robustness/2026-03-27_comparison_and_generality_analysis.md) — unpaired 방법 후보 비교

---

## 2. 가설 (Hypothesis)

### 주 가설

> "Test image의 train NN을 pseudo-pair로 사용하면, pseudo_shift의 PCA가 oracle nuisance subspace를 근사하며, robust PCA로 anomaly outlier를 필터링한 후 hard projection하면 baseline 대비 +10pp 이상 개선할 수 있다."

### 근거

1. **Per-sample 추정**: TTNS(batch covariance)와 달리, per-sample pseudo_shift에서 anomaly는 개별 outlier → robust PCA로 분리 가능
2. **L8 robust NN**: L8이 robust(RelShift 0.11)하므로 shifted image도 올바른 NN 매칭 기대
3. **D0 진단 통과**: SNR=0.69, D2 oracle explained=87.6% → nuisance 방향을 포착

### 예상 결과

| 지표 | Baseline | 예상 (NN Pseudo) | Oracle NSP |
|------|---------|-----------------|-----------|
| AD2 I-AUROC | 59.9% | 70-80% (+10-20pp) | 83.8% |

---

## 3. 실험 설정 (Experiment Design)

### 알고리즘

```
1. Train features at L8: {f(z_j)} for all train normals
2. For each test image x_i:
   NN(x_i) = argmin_j ||f_L8(x_i) - f_L8(z_j)||
   pseudo_shift_i = f_L8(x_i) - f_L8(NN(x_i))
3. Robust filtering: keep pseudo_shifts with norm < p-th percentile
4. PCA on filtered pseudo_shifts → top-K = nuisance basis
5. Hard projection: f_clean = f - N @ N^T @ f
6. Mahalanobis anomaly scoring
```

### 대조군 (Control)

| 방법 | 설명 |
|------|------|
| Baseline | No projection, Mahalanobis at L8 |
| Oracle NSP K=100 | Paired shift vectors PCA (upper bound) |
| Random K=100 | Random basis projection (lower bound) |

### 실험군 (Treatment)

| 조건명 | K | Robust filtering | 비고 |
|--------|---|-----------------|------|
| nn_pseudo_K100_rob80 | 100 | 80th percentile | 주 실험 |
| nn_pseudo_K100_rob90 | 100 | 90th percentile | 덜 공격적 필터 |
| nn_pseudo_K100_norobust | 100 | None | Ablation: robust 효과 |
| K sweep (K=30,50,100,150,200) | 다양 | 80th percentile | K 민감도 |

### 데이터

```
Dataset: MVTec AD 2 (8 categories)
Backbone: DINOv2 ViT-B/14, Layer 8
Feature: CLS + patch_mean, L2 normalized (dim=1536)
Train: 200 images/category (1537 total)
Test: normal (72-200) + anomaly (60-105) per category
```

### 실행 커맨드

```bash
docker exec -w /Volume/RESEARCH/Pilot Project_LG_2nd python3 scripts/nn_pseudopair_experiment.py \
  --data_root /Volume/DATA/mvtec_ad_2 --layer 8 --K 100 --k_sweep
```

---

## 4. 결과 (Results)

### 사전 진단 (D0+D1+D2) — ✅ PASS

| 지표 | L8→L8 | GO 기준 |
|------|-------|---------|
| D0: SNR (shift/nn_dist) | **0.69** | > 0.3 |
| D0: Cos(pseudo, oracle) | **0.47** | > 0.3 |
| D1: Mean principal angle | **50.6°** | < 60° |
| D2: Oracle shift explained | **87.6%** | > 30% |

**진단은 통과했으나 본 실험에서 실패** — 진단과 본 실험의 괴리가 핵심 교훈.

### 본 실험 — ❌ FAIL

| Method | AD2 I-AUROC | vs Baseline |
|--------|------------|------------|
| **Baseline** | **59.9%** | — |
| **Oracle NSP K=100** | **87.4%** | **+27.5pp** |
| NN Pseudo K=100 rob80 | **55.7%** | **-4.2pp** |
| NN Pseudo K=100 rob90 | 54.9% | -5.0pp |
| NN Pseudo K=100 norobust | 53.4% | -6.5pp |
| Random K=100 | 59.8% | -0.1pp |

### K Sweep

| K | AD2 I-AUROC | 비고 |
|---|------------|------|
| 30 | 51.5% | -8.4pp (더 많이 하락) |
| 50 | 50.6% | -9.3pp (최악) |
| 100 | 55.7% | -4.2pp |
| 150 | 59.9% | 0pp (K > N_samples → projection 비활성) |
| 200 | 59.9% | 0pp (동일) |

**K가 작을수록 악화**: 소수의 방향에 anomaly 정보가 집중 → 제거 시 anomaly detection 파괴.

### 카테고리별 결과 (nn_pseudo K=100 rob80)

| Category | Baseline | Oracle NSP | NN Pseudo | Delta |
|----------|---------|-----------|----------|-------|
| rice | 61.2% | 100.0% | **68.8%** | **+7.6pp** ✅ |
| can | 48.5% | 84.9% | **54.2%** | **+5.7pp** ✅ |
| fruit_jelly | 91.2% | 97.0% | 91.2% | 0pp |
| sheet_metal | 59.2% | 85.8% | 59.2% | 0pp |
| fabric | 59.0% | 84.0% | 55.0% | -4.0pp ❌ |
| walnuts | 55.1% | 79.9% | 42.1% | **-13.0pp** ❌ |
| wallplugs | 47.4% | 71.7% | 38.0% | **-9.4pp** ❌ |
| vial | 57.4% | 95.9% | 37.1% | **-20.3pp** ❌ |

**패턴**: 2개 카테고리(rice, can)에서만 개선, 나머지 6개에서 동등 또는 악화. 특히 vial(-20.3pp), walnuts(-13.0pp)에서 심각한 하락.

---

## 5. 결과 분석 (Analysis)

### 가설 검증

> **결론**: ❌ **가설 기각** — NN Pseudo-Pairing은 baseline보다 나쁘다 (-4.2pp). Robust PCA filtering이 anomaly contamination을 충분히 제거하지 못한다.

### 5.1 진단 통과 vs 본 실험 실패의 괴리

**진단(D0-D2)은 normal test images만 사용**:
- D0: shifted normal의 pseudo_shift vs oracle shift → 높은 alignment
- 이 조건에서 pseudo_shift basis는 oracle과 잘 정렬됨

**본 실험은 normal + anomaly 혼합 test batch 사용**:
- pseudo_shift = test_all - NN(test_all) → anomaly images의 pseudo_shift가 포함
- Anomaly pseudo_shift = shift + anomaly signal → nuisance basis를 오염
- Robust filtering (80th percentile by norm)으로 일부 제거되나 불충분

**교훈**: Subspace alignment 진단은 필요 조건이지 충분 조건이 아니다. **실제 AD pipeline에서의 anomaly contamination은 별도로 검증해야 한다.**

### 5.2 TTNS와 동일한 근본 문제의 재발

```
TTNS:          Σ_test - Σ_train      ← batch-level aggregation → anomaly inseparable
NN Pseudo:     PCA({pseudo_shift_i}) ← sample → batch aggregation → anomaly leaks into PCA
```

Per-sample pseudo-shift 자체는 anomaly를 outlier로 만들지만, PCA 단계에서 batch aggregation이 필요하므로 anomaly의 영향이 재유입된다. **Outlier가 PCA의 principal direction을 왜곡**하기 때문.

TTNS에서는 anomaly가 covariance의 eigenvalue에 영향. NN Pseudo에서는 anomaly가 PCA의 singular vector에 영향. 메커니즘은 다르지만 결과는 동일: **nuisance basis에 anomaly 정보 혼입 → projection 시 anomaly detection 파괴**.

### 5.3 카테고리별 분석

**개선된 카테고리 (rice, can)**:
- 이 카테고리들의 공통점: shift magnitude가 크고 (SNR > 0.8), anomaly ratio가 상대적으로 낮음
- pseudo_shift에서 shift signal이 anomaly signal을 압도 → PCA가 올바른 nuisance를 포착

**악화된 카테고리 (vial, wallplugs, walnuts)**:
- Anomaly가 다양하고 강함 → pseudo_shift PCA에서 anomaly 방향이 top PC에 포함
- 이 방향을 제거하면 anomaly detection 능력 파괴

### 5.4 Robust filtering의 한계

| Variant | AD2 | 분석 |
|---------|-----|------|
| norobust | 53.4% | 모든 anomaly 포함 → 최악 |
| rob90 | 54.9% | 10% 제거 → 부족 |
| rob80 | 55.7% | 20% 제거 → 여전히 부족 |
| K=150+ | 59.9% | K > N이라 projection 없음 → baseline 회귀 |

Robust filtering은 anomaly ratio에 비례하는 제거가 필요하지만, MVTec AD 2의 anomaly ratio ≈ 40-60% (test batch 내)로 매우 높음. 80th percentile에서도 anomaly의 상당수가 남아 PCA를 오염.

**산업 환경에서 anomaly ratio < 5%라면** robust filtering이 효과적일 수 있지만, 벤치마크 평가에서는 anomaly ratio가 인위적으로 높음.

### 부수 발견

1. **Oracle NSP가 이전 실험(83.8%)보다 높음 (87.4%)**: max_per_cat=200으로 더 많은 train data 사용 → per-category NSP 추정 및 Mahalanobis가 개선됨.
2. **K=150+에서 baseline 회귀**: Robust filtering 후 N_samples < K가 되어 projection이 비활성화됨 → 자연스러운 safeguard.
3. **Rice 카테고리에서 oracle NSP = 100%**: 이 카테고리의 anomaly가 nuisance와 거의 직교하여 projection이 완벽하게 작동.

---

## 6. 피드백 및 다음 단계 (Feedback & Next Steps)

### 교훈 (Lessons Learned)

1. **"Per-sample estimation + batch PCA" = 여전히 batch aggregation 문제**: TTNS의 실패 메커니즘(batch-level anomaly contamination)은 per-sample pseudo-shift를 거쳐도 PCA 단계에서 재발한다. **Anomaly contamination은 aggregation 단계에서 발생하며, estimation 단계를 바꿔도 해결되지 않는다.**

2. **Subspace alignment 진단 ≠ AD 성능 보장**: D0-D2 진단은 normal-only 조건에서 측정. 실제 AD pipeline의 anomaly 혼합 조건은 별도 검증 필요. 향후 진단에는 "anomaly가 추정을 오염시키는 정도" 측정을 포함해야 함.

3. **Anomaly ratio가 높은 벤치마크의 특수성**: MVTec AD 2 test batch에서 anomaly ratio ≈ 40-60%. 산업 현장의 <5%와 크게 다름. Robust estimation의 효과는 anomaly ratio에 강하게 의존.

4. **4가지 unpaired 시도의 수렴**: TTNS, IFR, AdaBN, NN Pseudo-Pairing 모두 실패. 접근 방식은 모두 달랐지만 결과는 동일. **One-class AD에서 paired data 없이 nuisance direction을 추정하는 것은 구조적으로 불가능**하다는 결론에 4가지 독립적 증거가 수렴.

### 누적 negative evidence 종합

| # | 접근 | 메커니즘 | 실패 원인 | 논문 메시지 |
|---|------|---------|----------|-----------|
| 1 | TTNS (test ΔΣ) | Batch covariance | Anomaly variance ⊂ ΔΣ | Test statistics ≠ shift statistics |
| 2 | IFR (train Δ) | Layer residual PCA | Train variation ⊥ shift | Internal structure ≠ external shift |
| 3 | AdaBN | Feature normalization | Signal destruction | Global correction ≠ directional correction |
| 4 | **NN Pseudo** | **Per-sample NN + PCA** | **Anomaly leaks into PCA** | **Even sample-level fails at aggregation** |

### 다음 단계 제안

1. **분석 논문 (Direction C) 확정**: 4가지 independent negative evidence가 강력한 contribution.
   - "왜 FM-AD가 shift에서 깨지는가" (mechanisms)
   - "왜 unpaired estimation이 구조적으로 불가능한가" (4가지 실패 분석)
   - "paired data가 있으면 어떻게 고치는가" (NSP oracle)
   - "왜 linear가 optimal인가" (10개 nonlinear 실패)

2. **NN Pseudo-Pairing의 "normal-only" 변형**: Test batch에서 anomaly를 사전에 제거할 수 있다면? → 이것은 AD 문제 자체를 풀어야 하므로 순환 논리. 하지만 iterative approach (1차 scoring → 정상 추정 → nuisance 추정 → 2차 scoring) 가능성은 존재.

3. **Intrinsic Dimension 분석 (QE-D)**: Layer 8의 기하학적 특수성 설명 → 분석 논문 보강.

### _lessons.md 승격 여부

- [x] 승격 필요:
  - "Per-sample estimation + batch PCA = 여전히 anomaly contamination (TTNS와 동일 근본 문제)"
  - "Subspace alignment 진단은 normal-only 조건 → AD pipeline의 anomaly 혼합과 괴리 가능"
  - "4가지 unpaired 시도 전멸 → one-class AD에서 paired data 없이 nuisance 추정 구조적 불가능"

---

## 관련 노트

- 선행: [IFR Diagnostic](../2026-03-28_ifr_diagnostic/report.md) — FM 내부 구조 시도
- 선행: [TTNS GO/NO-GO](../2026-03-27_ttns_go_nogo/report.md) — test-time batch 실패
- 선행: [Phase 3 NSP](../2026-03-26_phase3_linear_disentanglement/report.md) — oracle baseline
- 분석: [FM 내부 구조 분석](../../analysis/fm_ad_robustness/2026-03-28_fm_internal_nuisance_identification.md)
- 후속: 분석 논문 방향 최종 결정
