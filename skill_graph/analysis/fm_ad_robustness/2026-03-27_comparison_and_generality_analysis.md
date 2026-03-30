# 비교군 설계 및 방법론 일반성 분석 — 2026-03-27

> **상태**: Critical path — 논문 설계에 직접 영향
> **keywords**: comparison, baseline, generality, MVTec AD 2 dependency, paired data, test-time adaptation

---

## 1. 현재 비교군의 Gap 분석

### 1.1 Claim과 Evidence의 불일치

| Claim | Scope | Evidence | Scope |
|-------|-------|----------|-------|
| FM-AD는 distribution shift에 취약 | General (모든 shift) | 3 methods × MVTec AD 2 | 조명 shift만 |
| Entanglement이 원인 | General | Layer 분석 on DINOv2 | 1 backbone |
| Post-hoc projection으로 해결 | General | NSP on MVTec AD 2 TESTpub | **1 데이터셋, 독자 파이프라인** |

**핵심 문제**: Claim은 general한데, 핵심 evidence(NSP +29.4pp)는 MVTec AD 2에 locked.

### 1.2 Reviewer가 반드시 물을 질문

**Q1: "기존 robustness 방법과 비교하지 않았다"**

| 비교 대상 | 유형 | 현재 상태 |
|-----------|------|----------|
| PatchCore (CVPR 2022) | Training-free, 가장 robust | **미비교** — 반드시 포함 필요 |
| FiCo (AAAI 2025) | DA for AD | **미비교** |
| PILOT (BMVC 2025) | TTA via prompt tuning | **미비교** |
| SuperAD (DINOv2) | Training-free FM-AD | TESTpub 76.7% 보고됨, 우리 조건과 비교 필요 |

**Q2: "NSP를 기존 FM-AD 방법 위에 적용하면?"**
- 독자 파이프라인에서만 검증 → "당신의 파이프라인이 좋은 것이지 NSP가 좋은 것이 아니다"
- Dinomaly + NSP, AnomalyCLIP + NSP 등 generality 검증 필수

**Q3: "단순 대안과 비교하지 않았다"**
- Random projection (아무 방향 K개 제거), data augmentation, feature normalization, AdaBN
- 특히 **random projection**은 "PCA가 의미 있는 방향을 잡는가?" 검증에 필수

**Q4: "RobustAD에서는?"**
- MVTec AD 2만으로는 generality 부족
- RobustAD에서 결과 없으면 "조명 shift 전용 방법" 취급

### 1.3 메트릭/Split 불일치

| 방법 | Split | 메트릭 | 직접 비교 가능? |
|------|-------|--------|---------------|
| SuperAD | TESTpub | AU-ROC_0.05 | 메트릭 다름 |
| RoBiS | TESTpriv/mix | AucPro_0.05 | Split 다름 |
| PatchCore | TESTpriv/mix | SegF1 | Split + 메트릭 다름 |
| **NSP (ours)** | **TESTpub** | **I-AUROC** | — |

→ Apple-to-apple 비교 불가. 통일 필요.

---

## 2. MVTec AD 2 의존성 문제

### 2.1 방법론이 데이터셋 구조에 의존

NSP의 shift vector 추출:
```python
# MVTec AD 2 고유 구조: 같은 물체의 multi-condition 이미지
# test_public/good/001_regular.png
# test_public/good/001_overexposed.png
d_i = f(obj_i, envB) - f(obj_i, envA)  ← 물체 정보 완벽 상쇄, 순수 환경 효과
```

이 구조가 있는 AD 벤치마크: **MVTec AD 2만 해당**.

### 2.2 RobustAD에서 NSP가 안 되는 구체적 이유

RobustAD 구조:
```
train/    ← source domain, 물체 {a, b, c}
test0/    ← target domain 0, 물체 {x, y, z}  ← 다른 물체
test1/    ← target domain 1, 물체 {p, q, r}  ← 또 다른 물체
```

**문제 1: Shift vector 오염**
```
MVTec AD 2: d_i = f(obj_i, envB) - f(obj_i, envA) = [순수 환경 효과]
RobustAD:   d = E[f(obj_j, envB)] - E[f(obj_k, envA)] = [환경 효과] + [물체 차이] + [샘플링 노이즈]
```
물체가 다르기 때문에 inter-object variance가 shift vector에 섞임 → nuisance subspace에 물체 구조 정보가 포함 → projection 시 anomaly 정보까지 제거 위험.

**문제 2: Shift vector 수 부족**
- MVTec AD 2: 물체 300개 × 조명 조건 수 = **수백 개** shift vectors → PCA 가능
- RobustAD: 도메인 쌍 간 mean shift = **6-7개** vectors → 1536차원에서 PCA 무의미

### 2.3 의존성 요소 정리

| 요소 | MVTec AD 2 의존도 | 다른 벤치마크 적용 가능? |
|------|------------------|----------------------|
| Paired shift vector | **완전 의존** | ❌ |
| Shift 유형 = 조명만 | 강하게 의존 | viewpoint, camera 등 미검증 |
| TESTpub 평가 | TESTpub만 사용 | TESTpriv 서버 제출 필요 |
| K=100 최적값 | 이 데이터셋에서 튜닝 | 다른 데이터셋 최적 K 불명 |

---

## 3. Unpaired Nuisance 추정 방법 후보

### 3.1 후보 비교

| 방법 | 추가 데이터 요구 | Shift vector 수 | 오염도 | 적용 범위 |
|------|----------------|----------------|--------|----------|
| A. NN pseudo-pairing (ResAD식) | 없음 (train + test normal) | test normal 수 (수십~수백) | 중 (NN 품질 의존) | General |
| B. Augmentation-based synthetic | 없음 | 무제한 | 중 (shift 유형 사전 지식 필요) | Shift 유형 알 때만 |
| C. Multi-domain statistics | 여러 target domain | 도메인 수 | 중 (물체 차이 섞임) | Multi-domain 있을 때 |
| **D. Test-time covariance shift** | **없음 (train + test batch)** | **N/A (covariance 직접)** | **낮 (통계적 추정)** | **가장 General** |

### 3.2 D (Test-time Covariance Shift)가 가장 강한 이유

```
NSP:   paired data 필요 → MVTec AD 2 전용 → "데이터셋 솔루션"
D:     test batch만 필요 → 어디서든 적용 → "문제 솔루션"
```

| 차원 | NSP | D (Test-time) |
|------|-----|---------------|
| 추가 데이터 요구 | paired multi-condition | **없음** |
| 적용 벤치마크 | MVTec AD 2만 | MVTec AD 2, RobustAD, 어디든 |
| Shift 유형 | 사전에 알려진 shift만 | **unknown shift에도 적용** |
| 실용성 | calibration 수집 필요 | **배포 후 즉시 작동** |
| 이론적 흥미 | PCA on known shift | **test-time에 shift 발견+제거** |

### 3.3 D의 구체적 알고리즘 (Test-time Covariance Shift Decomposition)

```
Train phase:
  μ_train = mean(f_train_normal)
  Σ_train = cov(f_train_normal)

Test phase (batch-level):
  μ_test = mean(f_test_batch)
  Σ_test = cov(f_test_batch)

Nuisance 추정:
  방법 1 — Mean shift direction:
    d = μ_test - μ_train  (1차원, 가장 단순)

  방법 2 — Covariance shift의 spectral decomposition:
    ΔΣ = Σ_test - Σ_train
    eigendecomposition(ΔΣ) → 고유값 큰 방향 = shift에 의해 분산이 변한 방향
    top-K eigenvectors of ΔΣ = nuisance subspace

  방법 3 — Mean + Covariance 결합:
    mean shift direction + covariance shift directions 결합

Projection + Scoring:
  NSP와 동일 (orthogonal projection → Mahalanobis)
```

### 3.4 AdaBN과의 차별점

| | AdaBN | Test-time Covariance Shift |
|---|-------|---------------------------|
| 하는 일 | BN statistics re-normalize | Nuisance direction 식별 + 선택적 제거 |
| Anomaly 신호 | **함께 정규화됨** (손실) | **보존됨** (nuisance만 제거) |
| 방향 선택성 | 없음 (전체 feature 정규화) | 있음 (shift 방향만 제거) |
| AD 적합성 | 낮음 | 높음 |

### 3.5 고려사항: Test batch에 anomaly가 섞임

Test batch = normal + anomaly 혼합 → μ_test, Σ_test가 anomaly에 의해 오염.

완화 전략:
- 산업 AD에서 anomaly ratio는 대개 <5% → 영향 제한적
- Robust mean/covariance 추정 (median, trimmed mean, minimum covariance determinant)
- Score-based filtering: 1차 scoring 후 high-score sample 제외하고 통계 재추정

---

## 4. 논문 구조 재설계

### 4.1 기존 구조 (NSP 중심 — 문제 있음)

```
§1. FM-AD가 shift에서 깨진다
§2. 원인: entanglement
§3. 해결: NSP (paired data 필요)
§4. 실험: MVTec AD 2에서 +29.4pp
→ 문제: 1 데이터셋, paired data 의존, 비교군 부재
```

### 4.2 새 구조 (Test-time 중심 — 더 강함)

```
§1. FM-AD가 shift에서 깨진다 (3 methods × 3 benchmarks)
§2. 원인: nuisance-anomaly entanglement (general insight)
§3. Oracle: paired data 있으면 PCA projection으로 해결 (NSP = upper bound)
§4. Practical: paired data 없이 test-time covariance shift로 해결
§5. 실험: MVTec AD 2 + RobustAD 모두에서 검증
§6. Analysis: oracle vs practical gap, 언제/왜 gap이 생기는지
```

**핵심 변화**: NSP가 "방법론"에서 "이론적 상한(oracle)"으로 역할 변경. Test-time method가 실제 contribution.

### 4.3 비교군 설계 (새 구조)

**Main Results Table**:
```
Rows:
  Existing methods: PatchCore, Dinomaly, AnomalyDINO, AnomalyCLIP
  + Robustness methods: FiCo or PILOT (최소 1개)
  + Simple baselines: AdaBN, Random projection
  + Ours (oracle): NSP (paired, upper bound)
  + Ours (practical): Test-time covariance shift

Cols:
  MVTec AD (clean), MVTec AD 2 (shift), RobustAD (multi-domain shift)
  메트릭 통일: I-AUROC (image-level)
```

---

## 5. 즉시 검증 실험

MVTec AD 2에서 세 방법 비교 → 방향 판단의 근거:

```
1. NSP (paired, K=100)              → 83.8% (이미 있음)
2. Test-time mean shift (1-dim)     → ?%   ← 가장 단순, lower bound
3. Test-time covariance shift (K-dim) → ?%  ← 핵심 실험
4. Random projection (K=100)        → ?%   ← "방향이 의미있는가" 검증

판단 기준:
  3이 1의 80% 이상 → test-time 방향으로 논문 진행
  3이 1의 60-80%  → 추가 개선 탐색 필요
  3이 1의 60% 미만 → paired data 필수, scope 축소 또는 Plan B
  4가 3과 비슷    → nuisance 추정 자체가 무의미 (critical failure)
```

---

## 관련 노트

- 선행: [방법론 상세 (NSP→SAPP)](2026-03-27_methodology_nsp_to_sapp.md)
- 선행: [4팀 사고 실험](../2026-03-27_method_thought_experiment.md)
- 선행: [벤치마크 기술 분석](2026-03-23_benchmark_analysis.md)
- 코드: `scripts/sapp_experiment.py` (SAPP), `scripts/phase3_nsp_experiment.py` (NSP)
- 후속: Test-time covariance shift 실험 보고서
