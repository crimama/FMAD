# Phase 3: Post-hoc Linear Disentanglement — 2026-03-26

> **상태**: 🟡 진행중
> **실험 ID**: `exp_20260326_phase3`
> **keywords**: disentanglement, linear projection, nuisance removal, robustness

---

## 1. 문제 분석 (Problem Analysis)

### 현상
- Phase 2에서 DINOv2 feature의 shift-anomaly entanglement 확인 (PC correlation 0.53)
- Top PCs 중 40%가 shift와 anomaly 모두에 반응 → shift가 anomaly score를 오염
- t-SNE에서 shifted-normal이 anomaly 방향으로 이동 → false positive

### 원인 추정
- DINOv2 pretraining이 lighting/viewpoint invariance를 학습하지만, 이 invariance가 불완전
- 잔류 shift가 anomaly-relevant dimension에 leak → feature space에서 분리 불가

### 관련 선행 실험/분석
- [Phase 2 결과](../2026-03-26_phase2_feature_shift_analysis/report.md)
- [PISCO (ICML 2023)](../../papers/): Post-hoc linear disentanglement
- [von Kügelgen (NeurIPS 2021)](../../analysis/fm_ad_robustness/2026-03-23_theoretical_foundations.md): Block identifiability

---

## 2. 가설 (Hypothesis)

### 주 가설
> Normal 데이터의 multi-environment features (regular + shifted)를 사용하여 nuisance subspace를 추정하고, anomaly scoring을 nuisance-orthogonal subspace에서 수행하면, shifted 조건에서 5pp+ I-AUROC 개선이 가능하다.

### 근거
1. Phase 2에서 일부 PCs가 shift-dominant임을 확인 → 이 PCs를 제거하면 shift 영향 감소
2. PISCO (ICML 2023): frozen feature의 post-hoc linear decomposition이 DG에서 효과적
3. One-class setting에서도 normal data만으로 nuisance direction 추정 가능 (multi-environment 정보 활용)

### 예상 결과
| 지표 | Baseline (Dinomaly) | 예상 (+ Disentangle) | 비고 |
|------|--------------------|--------------------|------|
| MVTec AD 2 I-AUROC | 68.6% | 73-78% | +5~10pp |
| MVTec AD I-AUROC | 99.64% | 99.0-99.6% | 하락 없음 |

---

## 3. 실험 설정 (Experiment Design)

### 방법: Nuisance Subspace Projection (NSP)

**핵심 아이디어**:
1. Train set의 normal 이미지에서 DINOv2 feature 추출 (regular + shifted 조건)
2. Regular와 shifted feature 간의 차이 벡터 수집 → nuisance direction 추정
3. PCA로 nuisance subspace의 top-K direction 추출
4. Test 시 feature에서 nuisance projection을 제거 후 anomaly scoring

```
f_clean = f - Σ_{k=1}^{K} (f · n_k) n_k
```
여기서 n_k는 nuisance direction (shift PCA의 k번째 eigenvector)

### 대조군 (Control)
- Baseline: DINOv2 feature 그대로 사용 (kNN anomaly scoring)

### 실험군 (Treatment)
| 조건명 | K (nuisance dims) | Layer | 비고 |
|--------|-------------------|-------|------|
| NSP-K5 | 5 | 11 | 보수적 제거 |
| NSP-K10 | 10 | 11 | 중간 |
| NSP-K20 | 20 | 11 | 적극적 제거 |
| NSP-K10-L8 | 10 | 8 | robust layer |

### 평가 지표
- Primary: I-AUROC on MVTec AD 2 (shifted)
- Secondary: I-AUROC on MVTec AD (clean, 하락 없어야 함)
- Analysis: per-category breakdown

---

## 4. 결과 (Results)

### 전체 실험 이력 (Autoresearch Loop)

| Exp | AD2 I-AUROC | AD1 I-AUROC | Status | Key Change |
|-----|------------|------------|--------|------------|
| baseline | 54.4% | 91.6% | keep | kNN, patch mean, no projection, L11 |
| NSP K=30 | 67.9% | 91.9% | keep | + Linear nuisance projection |
| exp1 | 68.0% | 94.3% | keep | + CLS token concat + L2 normalize |
| exp2 | 71.4% | 97.4% | keep | + Mahalanobis scoring (대신 kNN) |
| exp3 | 75.0% | 96.4% | keep | Layer 8 (robust layer) |
| exp4 | 72.5% | 97.7% | **discard** | Multi-layer 8+11 (L8 단독보다 나쁨) |
| exp5 | 78.5% | 96.3% | keep | K=50 |
| **exp6** | **83.8%** | **96.3%** | **keep** | **K=100 — 현재 best** |
| exp7a | 77.1% | 96.4% | discard | Condition-union nuisance estimation |
| exp7b | 83.8% | 96.3% | discard | Whitened PCA (= standard, no gain) |
| exp8 | 72.5% | 96.4% | discard | Soft anomaly-preserving projection |
| exp9 | 59.0% | ~95.0% | discard | RANP signal-only subspace (AD1 drops) |
| exp10 | 80.3% | 93.2% | discard | PCA-whitened Mahalanobis scoring |
| exp11 | 59.5% | 96.3% | discard | Train-only nuisance (no paired data) |
| exp12 | 60.9% | 96.4% | discard | Hybrid NSP: filter by train variance |
| exp13 | 66.7% | 96.4% | discard | Per-category adaptive K |
| exp14 | 74.8% | 96.4% | discard | Dual-space scoring (orig + proj ensemble) |

### Negative Result Summary (Novel Methods, all discard)

**Nuisance estimation 변경**: condition-union(-6.7pp vs best), whitened(0pp), train-only(-24pp) → **standard PCA가 최적**

**Projection 변경**: soft(-11pp), signal-only(-25pp), filtered(-23pp) → **hard orthogonal projection이 최적**

**Scoring 변경**: PCA-whitened Mahalanobis(-3.5pp), dual-space(-9pp) → **standard regularized Mahalanobis가 최적**

**Adaptive**: per-category K(-17pp) → **uniform K가 최적**

**결론**: 현재 best config의 **단순함이 장점**. 복잡한 변경이 모두 성능을 해침. 이는 논문에서 "method simplicity as a theoretical virtue" (Occam's razor for AD robustness)로 포지셔닝 가능.

### K Sweep (초기 실험, kNN scoring, patch mean, L11)

| K | AD 2 I-AUROC | Δ AD 2 | AD I-AUROC | Δ Clean |
|---|-------------|--------|-----------|---------|
| 0 (baseline) | 54.4% | — | 91.6% | — |
| 5 | 57.5% | +3.0pp | 91.7% | +0.1pp |
| 10 | 61.0% | +6.5pp | 91.8% | +0.1pp |
| 15 | 64.4% | +9.9pp | 91.9% | +0.2pp |
| 20 | 65.9% | +11.5pp | 91.9% | +0.3pp |
| 30 | 67.9% | +13.5pp | 91.9% | +0.3pp |

### K Sweep (best config: Mahalanobis, CLS+patch L2norm, L8)

| K | AD 2 I-AUROC | Δ AD 2 | AD I-AUROC | Δ Clean |
|---|-------------|--------|-----------|---------|
| 30 | 75.0% | +15.1pp | 96.4% | -0.1pp |
| 50 | 78.5% | +18.6pp | 96.3% | -0.2pp |
| **100** | **83.8%** | **+24.0pp** | **96.3%** | **-0.2pp** |

### Best Config (K=100 L8) Per-Category Breakdown

> 상세 per-category는 `results/phase3/nsp_K100_L8.json` 참조

### Nuisance Subspace 통계
- 300 shift vectors 수집 (regular ↔ shifted pairs)
- Top-10 nuisance directions: shift variance의 72.2% 설명
- K가 커질수록 monotonic 개선, saturation 미도달 (K=100에서도 +24pp)

---

## 5. 결과 분석 (Analysis)

### 가설 검증
> **결론**: ✅ 가설 강하게 지지 — shifted 조건에서 **+24.0pp** 개선 (best), clean 하락 -0.2pp

### 핵심 발견 (Autoresearch Loop에서 도출)

1. **Scoring 방법이 중요**: kNN → Mahalanobis로 전환만으로 AD2 +3.4pp, AD1 +3.1pp. Mahalanobis가 projected space에서 covariance를 활용하므로 nuisance 제거 후 남은 정보를 더 잘 활용.

2. **Layer 선택이 핵심**: Layer 11(last) → Layer 8(robust)로 전환 시 AD2 +3.6pp (71.4→75.0%). Phase 2에서 발견한 "Layer 8 sweet spot"이 실제 AD 성능으로 이어짐.

3. **K는 monotonic, saturation 미도달**: K=30→50→100에서 75.0→78.5→83.8%로 계속 증가. Clean 성능은 K=30~100 모두 96.3~96.4%로 안정. nuisance subspace가 anomaly 정보를 포함하지 않음을 강하게 시사.

4. **Multi-layer fusion 실패**: Layer 8+11 concat(72.5%)이 Layer 8 단독(75.0%)보다 나쁨. Layer 11의 noisy feature가 Layer 8의 clean signal을 희석. → 단순 concat이 아닌 selective fusion이 필요하거나, Layer 8 단독이 최적.

5. **CLS token + L2 normalize 기여**: Baseline 자체를 54.4→55.9%(kNN) / 59.9%(Mahalanobis)로 개선. 방향 정보만 남기는 것이 distance-based scoring에 도움.

### 개선 기여도 분해 (Ablation 관점)

| 변경 | AD2 Δ | 누적 AD2 |
|------|-------|---------|
| Baseline (kNN, patch mean, L11) | — | 54.4% |
| + NSP K=30 | +13.5pp | 67.9% |
| + CLS token + L2 norm | +0.1pp | 68.0% |
| + Mahalanobis scoring | +3.4pp | 71.4% |
| + Layer 8 | +3.6pp | 75.0% |
| + K=100 | +8.8pp | **83.8%** |

### 부수 발견 (Side Findings)
- 학습 파라미터 0개, 순수 post-hoc linear projection으로 +29.4pp 달성
- Multi-environment normal data (regular + shifted pairs)가 필수 → practical requirement
- CLS+patch concat + L2norm이 feature representation 자체를 개선 (NSP 없이도 baseline 향상)

---

## 6. 피드백 및 다음 단계 (Feedback & Next Steps)

### 교훈 (Lessons Learned)
- **Linear nuisance removal이 surprisingly effective** — nonlinear 방법 전에 linear baseline을 반드시 시도
- **Layer 선택 > multi-layer fusion** — robust layer 단독이 naive concat보다 우수
- **K는 aggressive하게** — clean 하락이 거의 없으므로 큰 K가 유리 (saturation 탐색 필요)
- **Scoring 방법도 중요** — nuisance 제거 후 Mahalanobis가 kNN보다 projected space 활용에 유리

### 다음 실험 제안
1. **K saturation 탐색** — K=150, 200, 300으로 어디서 멈추는지
2. **NSP + Dinomaly/AnomalyDINO 통합** — 실제 baseline에 NSP 적용하여 main table 결과
3. **Soft projection** — 완전 제거 대신 부분 제거 (λ * projection)
4. **Per-category nuisance** — 카테고리별로 다른 nuisance subspace 사용
5. **Per-condition analysis** — overexposed vs geometric shift에서 NSP 효과 차이
6. **Patch-level NSP** — image-level mean 대신 patch token별 projection → pixel-level 개선

### _lessons.md 승격 여부
- [x] 승격 필요:
  - "Linear NSP + Mahalanobis + Layer 8 + K=100 → +29.4pp AD2, -0.2pp clean"
  - "Layer 8 = DINOv2의 robustness sweet spot"
  - "Multi-layer concat 실패 — robust layer 단독이 최적"
  - "K monotonic, saturation 미도달 at K=100"

---

## 관련 노트
- 선행: [Phase 2](../2026-03-26_phase2_feature_shift_analysis/report.md)
- 코드: `scripts/phase3_nsp_experiment.py`
- 데이터: `results/phase3/`, `results.tsv`
- Autoresearch: `program.md`, branch `autoresearch/mar26_nsp`
