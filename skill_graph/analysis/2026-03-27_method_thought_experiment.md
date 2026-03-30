# 방법론 사고 실험 — 4팀 교차 비교 분석 (2026-03-27)

> 4개 전문 팀이 독립적으로 방법론을 제안하고, 내부 평가자가 비판한 후 교차 비교.

---

## 팀별 제안 요약

### Team A: Information-Theoretic Nuisance Decomposition (ITND)
- **핵심**: FDA(Fisher Discriminant Analysis)를 environment label 기반으로 적용. Generalized eigenvalue problem으로 nuisance/content direction 분리. Soft suppression `w_i = 1/(1+α·λ_i)`.
- **이론**: MI(Z;E) 최소화 under information preservation constraint. PCA가 ITND의 special case임을 증명.
- **평가**: Novelty 6, Feasibility 9, ICLR 5. "결국 FDA를 포장한 것" — 이론적 depth가 Gaussian assumption에 의존.
- **판정**: MODIFY — 분석 도구로 활용하거나, non-Gaussian 확장 필요.

### Team B: Self-Aligned Nuisance Adaptation (SANA)
- **핵심**: Test-time에 normal feature의 mean drift를 robust하게 추정하여 nuisance direction으로 사용. Paired data 불필요. Adaptive α로 shift 없으면 projection 비활성화.
- **이론**: Population drift ≈ nuisance subspace의 proxy. Nonlinear ICA의 auxiliary variable로 해석.
- **평가**: Novelty 7, Feasibility 9, ICLR 5. NSP의 60-80% 성능 예상. "Conceptually too simple" — AdaBN의 AD 버전으로 읽힐 위험.
- **판정**: MODIFY — PISCO-style disentanglement과 결합하여 이론적 depth 강화 필요.

### Team C: Geodesic Anomaly Detection (GAID)
- **핵심**: Layer 8이 information bottleneck의 최소점이라는 기하학적 가설. Intrinsic dimension, curvature, geodesic distance로 layer별 manifold 구조 분석. Curvature-adaptive layer fusion.
- **이론**: ViT의 information flow에서 Layer 8 = 최대 압축점. Geodesic distance가 Euclidean보다 shift-invariant.
- **평가**: Novelty 7, Feasibility 6, ICLR 6. **분석 프레임워크로서의 가치가 방법론보다 높음**. 가설 의존성이 높아 Week 1에서 조기 검증 필수.
- **판정**: MODIFY — 분석 70% + 방법론 30% 구조로 재포지셔닝.

### Team D: Environment-Contrastive Nuisance Disentanglement (ECND)
- **핵심**: Multi-environment normal data에서 MMD loss로 invariant/variant component 분리. Residual ECND: PCA 후 잔차에만 학습 적용.
- **이론**: 직접 invariance 최적화 (PCA의 간접 proxy 대비). Category-adaptive nuisance 학습.
- **평가**: Novelty 6, Feasibility 7, ICLR 4. **"PCA가 이미 충분하다"를 이기기 어렵다**. Day 1-2 MMD 분석이 GO/NO-GO 기준.
- **판정**: MODIFY — MMD 분석 자체가 "PCA near-optimality" 증거가 될 수 있음.

---

## 교차 비교 매트릭스

| 차원 | Team A (ITND) | Team B (SANA) | Team C (GAID) | Team D (ECND) |
|------|-------------|-------------|-------------|-------------|
| **Novelty** | 6 | 7 | 7 | 6 |
| **Feasibility** | 9 | 9 | 6 | 7 |
| **Expected Perf** | same/+1-3pp | worse (60-80%) | same/+2-5pp | same/+1-3pp |
| **ICLR Ready** | 5 | 5 | 6 | 4 |
| **Paired Data 필요** | Yes | **No** | No | Yes |
| **학습 필요** | No | No | No | **Yes** |
| **분석 가치** | 높음 (MI 분해) | 중간 (drift 분석) | **매우 높음** (기하학) | 중간 (MMD 분석) |
| **내부 판정** | MODIFY | MODIFY | MODIFY | MODIFY |

---

## 핵심 발견: 4팀의 공통 결론

### 1. "방법론 단독으로는 ICLR 부족" — 전원 일치

4개 팀 모두 자기 제안의 **방법론 novelty만으로는 ICLR에 불충분**하다고 평가. 이는 우연이 아니라 구조적 이유가 있다:
- PCA+hard projection이 이미 near-optimal (+29.4pp)
- 10개 복잡한 변형이 모두 실패 → **이 문제의 최적해가 linear**이라는 강한 증거
- Linear 문제에 non-linear 방법을 적용하면 overfitting/underfitting만 발생

### 2. "분석이 방법론보다 가치 있다" — 3/4팀 동의

| 팀 | 분석 contribution |
|----|------------------|
| A | "MI 분해로 nuisance/anomaly information 정량화" |
| C | "intrinsic dimension + curvature로 Layer 8 특수성 설명" |
| D | "MMD 분석으로 PCA near-optimality 증명" |

→ **분석 논문(Plan B)이 실제로 더 강한 논문**일 수 있다는 신호.

### 3. "이론적 framework + 단순한 해법" 구조가 최적

Team A와 C가 공통으로 제안: 이론적 framework(MI 분해 / 기하학적 분석)이 **"왜 단순한 방법이 최적인가"를 설명**하는 논문. 방법론은 framework에서 자연스럽게 도출되는 부산물.

---

## 팀 간 시너지 분석

### 최강 조합: Team A(이론) + Team C(분석) + Team B(실용성)

```
논문 구조:
§1. Problem: FM-AD가 shift에서 깨진다 (Phase 1)
§2. Analysis A (Team C): Layer별 기하학적 분석
    → intrinsic dimension, curvature profile
    → "Layer 8 = information bottleneck의 최소점" 발견
§3. Analysis B (Team A): Information-theoretic 분해
    → MI(Z;E) vs MI(Z;Anomaly) per layer
    → "왜 PCA가 작동하는가"의 formal 설명
§4. Method: NSP (단순하지만 원리에서 도출)
    + SANA variant (paired data 불필요, Team B)
§5. Experiments: +29.4pp, ablation, negative results
§6. Discussion: "10개 복잡한 방법이 왜 실패했는가"
    → 분석 결과에서 설명 (entanglement이 linear → linear fix가 optimal)
```

이 구조의 강점:
- **분석 depth**: 기하학(C) + 정보이론(A) 두 축의 독립적 증거
- **실용성**: SANA(B)로 "paired data 없이도 가능" 확장
- **Negative results**: 10개 실패가 "linear optimality" 가설의 증거
- **단순한 방법이 weakness가 아닌 strength**: 이론이 "왜 단순해야 하는가"를 설명

---

## 즉시 실행 가능한 Quick Experiments (1-2일)

### QE1: Intrinsic Dimension 측정 (Team C 핵심 가설 검증)
```python
# scikit-dimension으로 Layer 0-11의 intrinsic dim 측정
# 가설: Layer 8에서 최소
# 결과에 따라 전체 방향 결정
```
**GO/NO-GO**: Layer 8의 ID가 다른 layer 대비 유의하게 낮으면 GO

### QE2: NSP 잔차의 MMD 분석 (Team D 핵심 검증)
```python
# NSP K=100 적용 후 잔차에서 environment 간 MMD 측정
# 가설: 잔차에 유의한 shift가 남아있지 않음 → PCA가 충분
# 결과: PCA near-optimality의 직접 증거
```

### QE3: FDA vs PCA eigenvector 비교 (Team A 핵심 검증)
```python
# FDA eigenvectors와 PCA eigenvectors의 cosine similarity
# 가설: 높은 유사도 → FDA가 PCA와 거의 같음 → soft suppression의 이점 적음
```

---

## 최종 방향 제안

### 현실적 판단

| 경로 | 확률 | 조건 |
|------|------|------|
| **Plan A 성공** (방법론 novelty) | **20%** | QE1-3에서 예상 외 발견 + novel mechanism 도출 |
| **Plan B 전환** (분석+이론 논문) | **70%** | QE1-3 결과가 "linear optimality" 지지 |
| **Hybrid** (분석 60% + lightweight method 40%) | **10%** | SANA가 NSP의 80%+ 성능 달성 |

### 권장 액션

1. **QE1 (intrinsic dimension) 즉시 실행** — 1일 소요, 전체 방향 결정
2. QE1 결과에 따라:
   - Layer 8 ID가 확실히 낮으면 → Team C의 기하학적 분석 확장
   - 아니면 → Team A의 MI 분석으로 전환
3. **2주 후 checkpoint**: 방법론 novelty 확보 여부 판정 → Plan B pivot 결정

---

## 관련 노트
- 선행: [Phase 3 결과](../experiments/2026-03-26_phase3_linear_disentanglement/report.md)
- 방향 결정: memory/project_direction_decision.md
- 후속: QE1-3 실험 보고서
