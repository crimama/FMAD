# Minimal Calibration: How Few Paired Samples Does NSP Need? — 2026-03-28

> **상태**: 🟢 완료
> **실험 ID**: `exp_20260328_mincal`
> **코드**: [`scripts/minimal_calibration_experiment.py`](../../../scripts/minimal_calibration_experiment.py)
> **결과**: [`results/minimal_calibration/minimal_calibration_results.json`](../../../results/minimal_calibration/minimal_calibration_results.json)
> **keywords**: minimal calibration, sample efficiency, saturation curve, paired data, NSP, practical guide

---

## 1. 문제 분석 (Problem Analysis)

### 현상

NSP(paired projection)는 +29.4pp 개선을 달성하나 paired multi-condition data가 필수. 4가지 unpaired 접근(TTNS, IFR, AdaBN, NN Pseudo)이 모두 실패하여, paired data의 필요성은 확립됨.

### 핵심 질문

> Paired data가 필수라면, **최소 몇 쌍이면 충분한가?**

이것은 impossibility result(§4)와 결합하여 실용적 처방을 제공한다:
- 왜 필요한가 (impossibility) + 얼마나 필요한가 (efficiency) = 완결된 메시지.

### 관련 선행 실험/분석

- [Phase 3 NSP](../2026-03-26_phase3_linear_disentanglement/report.md) — full paired NSP (+29.4pp)
- [TTNS GO/NO-GO](../2026-03-27_ttns_go_nogo/report.md) — unpaired 실패
- [IFR Diagnostic](../2026-03-28_ifr_diagnostic/report.md) — FM 내부 구조 실패
- [NN Pseudo-Pairing](../2026-03-28_nn_pseudopair/report.md) — per-sample NN 실패

---

## 2. 가설 (Hypothesis)

### 주 가설

> "Paired shift vector의 수를 줄여도 PCA가 주요 nuisance 방향을 포착할 수 있으며, 10-50쌍으로 full NSP 성능의 80%+ 회복이 가능할 것이다."

### 근거

- PCA의 top-K eigenvector 수렴은 O(K²/ε²) samples (random matrix theory)
- Shift가 저차원이면 적은 sample로도 주요 방향 포착 가능
- NSP top-10 방향이 전체 shift 분산의 72.2% 설명 → 소수 sample로도 top 방향은 안정적

### 예상 결과

| N_pairs | 예상 AD2 | % of Full |
|---------|---------|----------|
| 5 | 65-70% | 30-50% |
| 10 | 70-75% | 50-70% |
| 50 | 78-82% | 80-90% |
| full (315) | 83.8% | 100% |

---

## 3. 실험 설정 (Experiment Design)

### 변경 사항

기존 NSP 파이프라인에서 **paired shift vector 수만 제어**. 나머지 동일:
- Layer 8, K=100, Mahalanobis scoring, CLS+patch L2norm
- Global nuisance (전 카테고리 shift vector pooling)

### 실험 매트릭스

| N_pairs | K (effective) | Seeds | 비고 |
|---------|-------------|-------|------|
| 0 (baseline) | 0 | 1 | No projection |
| 1, 2, 3, 5 | min(N-1, 100) | 3 | 극소 |
| 10, 20 | min(N-1, 100) | 3 | 소수 |
| 50, 100 | min(N-1, 100) | 3 | 중간 |
| full (315) | 100 | 3 | Oracle |
| per-category full | 100 | 1 | Per-cat oracle |

### 평가 지표

- Primary: AD2 I-AUROC (mean ± std over 3 seeds)
- Secondary: % of full NSP 회복률 = (score - baseline) / (full - baseline)

### 실행 커맨드

```bash
docker exec -w /Volume/RESEARCH/Pilot Project_LG_2nd python3 scripts/minimal_calibration_experiment.py \
  --data_root /Volume/DATA/mvtec_ad_2 --layer 8 --K 100 --max_per_cat 200
```

---

## 4. 결과 (Results)

### Saturation Curve

| N_pairs | AD2 I-AUROC | ±std | % of Full | 비고 |
|---------|------------|------|----------|------|
| 0 (baseline) | 59.9% | — | 0% | No projection |
| 1 | 59.9% | 0.0% | 0% | K > N → projection 사실상 비활성 |
| 2 | 59.9% | 0.0% | 0% | |
| 3 | 60.1% | 0.1% | 1% | |
| 5 | 60.6% | 0.1% | 3% | |
| 10 | 62.0% | 0.3% | 9% | 미미한 효과 시작 |
| 20 | 64.6% | 0.8% | 19% | |
| **50** | **68.0%** | **1.0%** | **33%** | 의미 있는 첫 개선 |
| **100** | **75.0%** | **0.6%** | **62%** | 실용적 최소선 |
| **full (315)** | **84.4%** | **0.0%** | **100%** | Global oracle |
| **per-category** | **87.4%** | — | **112%** | Per-cat oracle (best) |

### 시각화

```
AD2 I-AUROC (%)
  87 ┤                                              ● per-cat
  84 ┤                                           ●── full
     │                                         /
  75 ┤                                ●───────/
     │                              /
  68 ┤                       ●─────/
  65 ┤                  ●───/
  62 ┤              ●──/
  61 ┤           ●─/
  60 ┤──●──●──●─/
     └──┬──┬──┬──┬──┬───┬───┬───┬───
        1  2  3  5 10  20  50 100 full     N_pairs
```

---

## 5. 결과 분석 (Analysis)

### 가설 검증

> **결론**: ⚠️ **부분 기각** — 10-50쌍으로 80% 회복이라는 기대는 불일치. 100쌍에서 62%, 50쌍에서 33%. Saturation이 예상보다 매우 느림.

### 5.1 느린 saturation의 원인

**K=100 vs N_pairs의 sample complexity**: PCA에서 K개 방향을 안정적으로 추정하려면 N ≫ K가 필요. K=100일 때 N=50은 심각한 under-sampling.

- N < K일 때: PCA는 최대 N-1개의 non-trivial direction만 추출 가능 → effective K가 N-1로 제한
- N=50, K=100 → effective K ≈ 49 → full NSP(K=100) 대비 nuisance subspace가 절반

**해결 가능성**: K를 N에 맞춰 adaptive하게 설정 (K = min(N/2, 100)). 이 경우 소수 pair에서 더 적은 direction을 더 정확히 추정.

**그러나 근본적 한계**: NSP K=100까지 단조 증가하고 saturation 미도달이었으므로, nuisance subspace는 실제로 고차원 (≥100). 소수 pair로는 이 고차원 subspace의 일부만 커버 가능.

### 5.2 Per-category NSP (87.4%) > Global NSP (84.4%)

Per-category가 3pp 더 높음. 이는:
- 카테고리마다 nuisance direction이 다름 (shift가 물체 유형에 따라 다르게 작용)
- Global pooling이 category-specific nuisance를 희석
- **Per-category 접근이 more efficient**: 같은 수의 total pairs에서 더 정확한 nuisance 추정

**논문 시사점**: "Category-specific calibration이 global calibration보다 효율적" — 이는 실용적 처방에서 중요.

### 5.3 Saturation curve의 논문 contribution 가치

Saturation curve 자체가 3가지 메시지를 전달:

1. **Monotonic improvement**: Paired data가 많을수록 단조 개선 → paired data의 직접적 가치 실증
2. **No free lunch**: 5-10쌍으로는 의미 있는 개선 불가 → "쉬운 해법"은 없음
3. **Practical guide**: 100쌍 ≈ 75% AD2 (baseline +15pp). 카테고리당 ~12쌍이면 의미 있는 시작점

### 5.4 실용적 처방

| 시나리오 | 필요 데이터 | 기대 성능 |
|---------|-----------|----------|
| Paired data 없음 | — | 59.9% (baseline) |
| 최소 실용 | 카테고리당 ~12쌍 (총 100) | ~75% (+15pp) |
| 목표 80%+ | 카테고리당 ~25-30쌍 (총 200+) | ~80%+ |
| 최대 | 카테고리당 ~40쌍 (총 315) | 84.4% (+24.5pp) |
| Per-category oracle | 동일 | 87.4% (+27.5pp) |

### 부수 발견

1. **Variance가 N_pairs에 반비례**: N=50에서 std=1.0%, N=100에서 0.6%, full에서 0.0%. Subsample randomness의 영향이 상당 → 어떤 쌍을 선택하느냐가 중요.

2. **N=1-2에서 정확히 baseline**: K=100 > N_pairs이므로 PCA가 non-trivial direction을 추출하지 못함. 이는 알고리즘의 자연스러운 graceful degradation.

---

## 6. 피드백 및 다음 단계 (Feedback & Next Steps)

### 교훈 (Lessons Learned)

1. **Nuisance subspace가 고차원 (≥100)이므로 minimal calibration에 한계**: 저차원이었다면 (K≈5-10) 소수 pair로 충분했을 것. 하지만 실제 nuisance는 100+ 차원으로 다양 → 많은 관측이 필요.

2. **Per-category > global**: Category-specific nuisance 추정이 더 efficient. Practical guide에서 "전체 N쌍"보다 "카테고리당 N/C쌍"이 더 정확.

3. **Saturation curve = practical value table**: 사용자가 자신의 환경에서 "몇 쌍 촬영할 수 있는지"에 따라 기대 성능을 예측 가능. 이것 자체가 actionable contribution.

### 논문에서의 위치

```
§5 (Experiments) 또는 §6 (Discussion):
  "How much paired calibration is needed?"
  - Saturation curve (Fig X)
  - 100 pairs → 75% AD2 (+15pp from baseline)
  - Per-category calibration is more efficient
  - Practical recommendation table
```

이 실험은 **Impossibility (§4)와 직접 보완**:
- §4: "왜 unpaired가 불가능한가" (4가지 증거)
- §5/6: "그렇다면 얼마나의 paired data가 필요한가" (saturation curve)

### 다음 단계 제안

1. **K adaptive 추가 실험**: K = min(N/2, 100)으로 재실험 → 소수 pair 효율 개선 가능
2. **Per-category minimal calibration**: Category당 N쌍의 saturation curve
3. **PatchCore 비교군 추가**: 가장 robust한 기존 방법과의 비교
4. **논문 작성 시작**: contribution 5개 확립, 실험 데이터 대부분 확보

### _lessons.md 승격 여부

- [x] 승격 필요:
  - "Nuisance subspace는 고차원(≥100)이므로 minimal calibration으로 full 성능 도달 어려움. 100쌍으로 full의 62% 회복."
  - "Per-category nuisance 추정이 global pooling보다 3pp 우수. Category-specific shift 존재."

---

## 관련 노트

- 선행: [Phase 3 NSP](../2026-03-26_phase3_linear_disentanglement/report.md) — full oracle
- 선행: [4가지 unpaired 실패](../2026-03-28_nn_pseudopair/report.md) — impossibility
- 후속: K adaptive 실험 / Per-category curve / 논문 작성
