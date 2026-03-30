# IFR GO/NO-GO: Inter-layer Feature Residual — 2026-03-28

> **상태**: ❌ NO-GO
> **실험 ID**: `exp_20260328_ifr`
> **keywords**: IFR, inter-layer residual, layer decomposition, nuisance estimation, FM internal

---

## 1. 문제 분석

- NSP(+29.4pp)는 paired data 필요 → MVTec AD 2 전용
- TTNS(test-time)는 anomaly contamination으로 실패
- FM의 내부 구조(layer residual)로 nuisance를 추정할 수 있는가?

## 2. 가설

> DINOv2의 residual connection으로 f_L11 = f_L8 + Δ_9 + Δ_10 + Δ_11.
> L8이 robust하고 L11이 fragile하므로, Δ_{9-11}이 nuisance를 포함한다.
> PCA on Δ → nuisance subspace → hard projection으로 AD2 65%+ 달성 가능.

## 3. 실험 설정

9개 방법 비교 (DINOv2 ViT-B/14, Mahalanobis scoring):

| Method | 설명 |
|--------|------|
| baseline_L11 | No projection, L11 features |
| baseline_L8 | No projection, L8 features |
| IFR_summed_K{30,50,100,200} | PCA on (f_L11 - f_L8), K nuisance dims |
| IFR_stacked_K100 | PCA on [Δ_9; Δ_10; Δ_11] separately stacked |
| IFR_perlayer_K100 | PCA per layer, merge |
| IFR_summed_K100_onL8 | IFR applied to L8 features (control) |

## 4. 결과

### MVTec AD 2

| Method | AD2 I-AUROC |
|--------|------------|
| baseline_L11 | 59.7% |
| baseline_L8 | 59.9% |
| IFR_summed_K30 | 59.7% |
| IFR_summed_K50 | 59.6% |
| **IFR_summed_K100** | **59.5%** |
| IFR_summed_K200 | 59.4% |
| IFR_stacked_K100 | 59.7% |
| IFR_perlayer_K100 | 59.7% |
| IFR_summed_K100_onL8 | 59.8% |
| NSP oracle (참고) | 83.8% |

**판정: ❌ NO-GO** — 모든 IFR 변형이 baseline과 ±0.3pp 이내. 사실상 무효과.

### MVTec AD (clean)

| Method | AD1 I-AUROC |
|--------|------------|
| baseline_L11 | 97.4% |
| IFR_summed_K100 | 97.5% (+0.1pp) |
| IFR_stacked_K100 | 97.5% (+0.1pp) |

Clean performance 유지 (무해하지만 무의미).

### Per-category (AD2)

| Category | baseline_L11 | baseline_L8 | IFR_summed_K100 |
|----------|-------------|-------------|-----------------|
| can | 44.0% | 48.5% | 43.8% |
| fabric | 55.3% | 59.0% | 54.6% |
| fruit_jelly | 80.3% | 91.2% | 80.0% |
| rice | 65.8% | 61.2% | 65.6% |
| sheet_metal | 59.9% | 59.2% | 59.4% |
| vial | 61.1% | 57.4% | 61.4% |
| wallplugs | 47.1% | 47.4% | 47.6% |
| walnuts | 64.1% | 55.1% | 63.9% |

### Nuisance Subspace Analysis

| Category | Residual norm | Variance explained by K=100 |
|----------|--------------|---------------------------|
| can | 1.360±0.016 | 98.7% |
| fabric | 1.264±0.008 | 97.2% |
| fruit_jelly | 1.329±0.012 | 94.9% |
| rice | 1.265±0.004 | 94.0% |
| sheet_metal | 1.279±0.017 | 98.6% |
| wallplugs | 1.284±0.011 | 95.8% |
| walnuts | 1.327±0.019 | 97.8% |

K=100이 residual variance의 94-98%를 포착하지만 AD2 성능에 영향 없음.
→ **Residual variance ≠ shift-relevant variance**

## 5. 분석

### 가설 검증
> ❌ **가설 기각** — Layer residuals는 nuisance(shift) 방향과 무관.

### 실패 원인

**핵심: Layer residual ≠ shift direction**

```
NSP:  d = f(obj, shifted) - f(obj, clean)  → 관측된 shift 차이
IFR:  Δ = f_L11(obj) - f_L8(obj)           → 단일 조건에서의 layer 처리 패턴
```

1. **Shift 미관측**: IFR은 train normal(단일 조건)에서만 계산. Shift가 어떤 방향인지 모름.
2. **Residual = 일반 처리**: Δ_{9-11}은 모든 이미지에 공통으로 적용되는 deep layer의 처리(semantic refinement, attention reweighting 등). Shift-specific이 아님.
3. **K=100이 variance 94-98% 포착하지만 무효**: Residual의 주성분은 "deep layer가 무엇을 추가하는가"이지, "shift가 어디로 발생하는가"가 아님.
4. **Phase 3 train-only nuisance(59.5%)와 동일 패턴**: 단일 조건 데이터로는 shift 방향 추정 불가.

### 근본적 한계

> **Shift 방향을 식별하려면 shift를 관측해야 한다.**

이것은 FM 내부 구조의 한계가 아니라 **정보 이론적 한계**:
- Shift가 feature space의 어느 방향으로 일어나는지는 shift를 관측해야만 알 수 있음
- FM의 layer 구조, attention, residual stream 어느 것도 "아직 관측하지 않은 shift"를 예측할 수 없음
- 이는 TTNS(test-time), IFR(layer residual), train-only covariance가 모두 실패하는 공통 원인

### Negative Evidence 계열 정리

| 시도 | 방법 | AD2 | 실패 원인 |
|------|------|-----|----------|
| NSP (oracle) | Paired shift vectors | 83.8% | **성공** — shift 직접 관측 |
| TTNS-cov | Test batch ΔΣ | 50.6% | Anomaly contamination |
| IFR | Layer residuals | 59.5% | Shift 미관측 |
| Train-only PCA | Train covariance low-var | 59.5% | Shift 미관측 |
| Random projection | Random K=100 | 59.8% | 방향 무의미 |
| AdaBN | Test BN statistics | 58.3% | 전체 정규화가 anomaly 훼손 |

→ **Paired data만이 shift를 순수하게 관측하는 유일한 방법**

## 6. 다음 단계

### 교훈
- **Layer residual은 shift proxy가 아니다** (일반 처리 패턴 = shift-agnostic)
- **단일 조건 데이터에서 shift 방향 추정은 불가능** (정보 이론적 한계)
- **이 일련의 negative results 자체가 논문 contribution** — "왜 paired data가 필수적인가"

### 방향 확정: Analysis Paper (Direction C)

4번의 체계적 실패(TTNS, IFR, train-only, random)가 하나의 결론을 가리킴:
> FM feature의 shift robustness 문제를 해결하려면 shift를 관측해야 하며,
> one-class AD에서 이를 깨끗하게 관측하는 유일한 방법은 paired data이다.

이것 자체가 분석 논문의 핵심 contribution:
1. FM-AD가 왜 깨지는가 (7 mechanisms)
2. Paired data로 고칠 수 있다 (NSP +29.4pp = oracle)
3. Unpaired로 고칠 수 없다 (4가지 시도 + 이론적 설명)
4. Linear가 optimal이다 (10가지 비선형 실패)

### _lessons.md 승격
- [x] "Layer residual은 shift direction의 proxy가 아니다"
- [x] "단일 조건 데이터에서 shift 방향 추정은 정보 이론적으로 불가능"

---

## 관련 노트
- 선행: [TTNS GO/NO-GO](../2026-03-27_ttns_go_nogo/report.md)
- 선행: [FM-internal 분석](../../analysis/fm_ad_robustness/2026-03-28_fm_internal_nuisance_identification.md)
- 코드: `scripts/ifr_experiment.py`
- 결과: `results/ifr/ifr_results.json`
