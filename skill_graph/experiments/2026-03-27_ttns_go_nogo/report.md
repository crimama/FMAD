# TTNS GO/NO-GO: Test-Time Nuisance Subspace — 2026-03-27

> **상태**: 🟡 진행중
> **실험 ID**: `exp_20260327_ttns`
> **keywords**: TTNS, test-time, covariance shift, unpaired, nuisance estimation

---

## 1. 문제 분석

- NSP(+29.4pp)는 MVTec AD 2의 paired multi-condition 구조에 완전 의존 → 1 데이터셋 솔루션
- Paired data 없이 nuisance subspace를 추정해야 → generality 확보

## 2. 가설

> ΔΣ = Σ_test - Σ_train의 positive eigenvalue 방향이 nuisance subspace를 근사하며, 이를 hard projection하면 NSP의 80%+ 성능(≥67%)을 달성할 수 있다.

## 3. 실험 설정

6개 방법 비교 (DINOv2 Layer 8, Mahalanobis scoring):
1. **Baseline**: no projection
2. **TTNS-mean**: mean shift direction only (rank-1)
3. **TTNS-cov**: ΔΣ positive eigenvalue directions
4. **TTNS-combined**: mean + cov directions
5. **Random K=100**: 방향의 의미 검증
6. **AdaBN**: 가장 단순한 test-time 대안

## 4. 결과

| Method | AD2 I-AUROC | AD1 I-AUROC |
|--------|------------|------------|
| Baseline | 59.9% | 96.5% |
| TTNS-mean | 58.3% | 93.0% |
| **TTNS-cov** | **50.6%** | **30.8%** |
| **TTNS-combined** | **46.3%** | **24.9%** |
| Random K=100 | 59.8% | 96.4% |
| AdaBN | 58.3% | 78.5% |

**판정: ❌ NO-GO** — TTNS-cov가 baseline보다 -9.3pp 나쁨, AD1 30.8%까지 붕괴.

## 5. 분석

### 가설 검증
> ❌ **가설 기각** — ΔΣ의 positive eigenvalue 방향은 nuisance subspace의 good proxy가 아님.

### 실패 원인

1. **Anomaly contamination**: test batch = normal + anomaly 혼합. Anomaly가 만드는 high-variance direction이 ΔΣ의 positive eigenvalue에 포함됨. 이를 제거하면 anomaly detection 능력 자체를 파괴.

2. **AD1에서 30.8% 붕괴가 결정적 증거**: AD1(clean, anomaly ratio ~50%)에서 ΔΣ의 positive direction을 제거하면 anomaly와 normal을 구분하는 핵심 축이 사라짐. 이는 "분산 증가 = nuisance"라는 가정이 AD에서 근본적으로 틀렸음을 의미.

3. **Mean shift도 비효과적**: TTNS-mean은 AD2에서 -1.6pp로 미미하지만, AD1에서 -3.5pp. Mean shift direction도 anomaly 방향과 겹칠 수 있음.

4. **AdaBN도 유해**: AD1 78.5% (-18pp). 전체 feature 정규화가 anomaly 신호를 훼손.

5. **Random projection은 중립**: AD2 59.8% ≈ baseline. 아무 방향이나 제거하면 약간 정보 손실 but anomaly-specific 파괴는 없음.

### 근본적 문제

**Paired NSP가 작동하는 이유**: `d_i = f(obj, shifted) - f(obj, clean)` — **같은 물체**의 차이이므로 anomaly 정보가 완벽히 상쇄되고 순수 shift만 남음.

**Unpaired TTNS가 실패하는 이유**: `ΔΣ = Σ_test - Σ_train` — test에 anomaly가 섞여 있어 anomaly의 분산 증가도 포함. **Anomaly 분산과 shift 분산을 분리할 방법이 없음** (one-class이므로 anomaly label 없음).

이것은 "paired data가 왜 필수적인가"에 대한 강력한 negative evidence. NSP의 성공은 paired structure 자체의 가치이지, projection method의 가치가 아님.

## 6. 다음 단계

### 교훈
- **ΔΣ positive eigenvalue ≠ nuisance** in AD (anomaly contamination)
- **Paired data의 가치는 "object pairing"에 있다** — shift를 pure하게 추출하는 유일한 방법
- **Test-time unpaired estimation은 AD의 one-class 구조와 근본적으로 충돌**

### 방향 결정 필요
1. **Paired setting에 scope 한정**: NSP를 핵심 method로, "paired data가 왜 필수적인가"를 분석 contribution으로
2. **분석 논문 pivot**: "FM feature의 entanglement 분석 + 왜 linear fix가 optimal + 왜 unpaired는 실패하는가"
3. **다른 unpaired 접근**: robust covariance (MCD), anomaly-free test subset 추정 등

### _lessons.md 승격
- [x] "ΔΣ positive eigenvalue는 AD에서 nuisance proxy가 아님 (anomaly contamination)"
- [x] "Paired data의 핵심 가치는 object pairing (shift를 pure하게 추출)"

---

## 관련 노트
- 선행: [방향 전환 분석](../../analysis/fm_ad_robustness/2026-03-27_comparison_and_generality_analysis.md)
- 코드: `scripts/ttns_experiment.py`
