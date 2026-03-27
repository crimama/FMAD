# Phase 2: Feature Space Shift Analysis — 2026-03-26

> **상태**: 🟡 진행중
> **실험 ID**: `exp_20260326_phase2`
> **keywords**: feature analysis, DINOv2, layer-wise, distribution shift, entanglement

---

## 1. 문제 분석 (Problem Analysis)

### 현상
- Phase 1에서 3개 FM-AD 방법 모두 MVTec AD 2에서 25~34pp 하락 확인
- backbone(DINOv2/CLIP), 방법론(학습/few-shot/zero-shot) 무관하게 하락 → FM feature 자체의 문제

### 원인 추정
1. **Invariance-Information Destruction**: FM pretraining 시 조명/viewpoint를 nuisance로 학습 → anomaly와 관련된 appearance 정보도 함께 제거
2. **Nuisance-Semantic Entanglement**: shift와 anomaly가 같은 feature dimension을 공유 → 분리 불가
3. **Layer별 sensitivity trade-off**: shallow=texture-sensitive, deep=semantic → robustness와 discriminability 동시 확보 불가?

### 관련 선행 실험/분석
- [Phase 1 결과](../2026-03-26_phase1_baseline_reproduction/report.md)
- [analysis/fm_ad_robustness/2026-03-23_mechanism_analysis.md](../../analysis/fm_ad_robustness/2026-03-23_mechanism_analysis.md)
- [analysis/fm_ad_robustness/2026-03-23_theoretical_foundations.md](../../analysis/fm_ad_robustness/2026-03-23_theoretical_foundations.md)

---

## 2. 가설 (Hypothesis)

### 주 가설
> DINOv2 feature space에서 distribution shift는 layer별로 비균일하게 작용하며, deep layer가 shallow layer보다 shift에 robust할 것이다 (semantic invariance 때문).

### 부 가설
> Shift가 큰 카테고리(wallplugs, vial)가 Phase 1에서 AD 성능 하락이 크고, shift가 작은 카테고리(fabric)는 하락이 작을 것이다.

### 예상 결과
| 지표 | Shallow (0-3) | Mid (4-7) | Deep (8-11) |
|------|-------------|-----------|-------------|
| Relative Shift | 높음 (>0.3) | 중간 | 낮음 (<0.15) |
| Cosine Similarity | 낮음 | 중간 | 높음 (>0.98) |

---

## 3. 실험 설정 (Experiment Design)

### Step 2.1 — Feature Space Shift 정량화
- MVTec AD 2 paired images: regular vs {overexposed, underexposed, shift_1~5}
- DINOv2 ViT-B/14 feature 추출, layer 0~11 (12 layers)
- 200 pairs subsampled (8 categories × ~25 pairs)

### Step 2.2 — Layer별 Robustness 프로파일링
- Per-layer metrics: L2 distance, cosine similarity, relative shift
- Per-condition, per-category breakdown

### 평가 지표
- Relative Shift: `||f(shifted) - f(regular)|| / ||f(regular)||`
- Cosine Similarity: `cos(f(regular), f(shifted))`

### 실행 커맨드
```bash
python3 scripts/phase2_feature_analysis.py \
  --data_root /home/hun/Volume/DATA/mvtec_ad_2 \
  --backbone dinov2 --model_name dinov2_vitb14 \
  --output_dir results/phase2 --max_pairs 200
```

---

## 4. 결과 (Results)

### Step 2.1 + 2.2: DINOv2 Layer별 Shift 프로파일

| Layer | Relative Shift | Cosine Similarity | 구간 |
|-------|---------------|-------------------|------|
| 0 | 0.2184 | 0.9672 | shallow |
| 1 | 0.2464 | 0.9591 | shallow |
| 2 | 0.2406 | 0.9604 | shallow |
| 3 | 0.2424 | 0.9590 | shallow |
| 4 | 0.2605 | 0.9517 | mid |
| 5 | 0.2678 | 0.9480 | mid |
| 6 | 0.2543 | 0.9533 | mid |
| 7 | 0.2700 | 0.9478 | mid |
| **8** | **0.1063** | **0.9923** | **deep (most robust)** |
| 9 | 0.1533 | 0.9837 | deep |
| 10 | 0.2164 | 0.9671 | deep |
| 11 | 0.2551 | 0.9537 | deep (last) |

### Per-category Shift (last layer, layer 11)

| Category | Relative Shift | Phase 1 I-AUROC Drop | 상관 |
|----------|---------------|---------------------|------|
| fabric | 0.153 | -27pp (Dinomaly) | 낮은 shift = 작은 하락 |
| walnuts | 0.177 | -27pp | |
| fruit_jelly | 0.192 | -14pp | |
| can | 0.236 | -47pp | 높은 shift = 큰 하락 |
| rice | 0.237 | -35pp | |
| sheet_metal | 0.332 | -20pp | ⚠️ 예외 |
| wallplugs | 0.362 | -56pp | 가장 높은 shift = 가장 큰 하락 |
| vial | 0.365 | -20pp | ⚠️ 예외 |

---

## 5. 결과 분석 (Analysis)

### 가설 검증

**주 가설**: ❌ **기각** — Deep layer가 shallow보다 반드시 robust하지 않음
- Layer 8이 uniquely robust (RelShift 0.11), 하지만 layer 10-11은 shallow만큼 취약
- 예상: monotonic decrease (shallow→deep) / 실제: **V-shaped profile** (mid 취약, layer 8 robust, last layer 다시 취약)

**부 가설**: ⚠️ **부분 지지** — wallplugs(highest shift, worst AD)는 일치하지만, sheet_metal/vial은 높은 shift에도 AD 하락이 moderate

### 상세 분석

1. **Layer 8의 특이성**: 12개 layer 중 유일하게 RelShift < 0.15. DINOv2 ViT-B/14의 layer 8은 attention pattern이 다를 가능성 → register token 효과? 또는 information bottleneck?

2. **Last layer의 취약성**: Layer 11(last)이 layer 0(first)과 비슷한 shift를 보임 (0.255 vs 0.218). CLS token aggregation이 shift를 amplify하는 것으로 추정.

3. **Non-monotonic profile의 의미**:
   - 단순히 "deep layer를 쓰면 robust"가 아님
   - Layer selection만으로는 robustness-sensitivity trade-off 해결 불가
   - → **Dimension-level disentanglement이 필요** (plan.md 경로 α 지지)

4. **Category-shift correlation이 불완전한 이유**:
   - sheet_metal, vial: 높은 feature shift에도 AD 하락이 moderate → shift 방향이 anomaly 방향과 orthogonal할 수 있음
   - → Step 2.3 (entanglement 측정)에서 확인 필요

### 부수 발견 (Side Findings)
- DINOv2 ViT-B/14의 per-condition shift는 아직 집계 안 됨 (코드 버그: condition 파싱 로직에서 빈 결과)
- 이미지 크기에 따라 patch token 수 변화 → feature flatten 방식에 따른 차이 존재 가능

---

## 6. 피드백 및 다음 단계 (Feedback & Next Steps)

### 교훈 (Lessons Learned)
- Layer-wise robustness는 monotonic이 아니라 non-monotonic → "deep=robust" 가정 위험
- Feature 전체의 relative shift만으로는 AD 성능 하락을 완전히 설명 못함 → shift **방향**이 중요

### 다음 실험 제안
1. ~~**Step 2.3: Entanglement 측정**~~ → 완료 (아래 참조)
2. **Per-condition breakdown 수정** — overexposed vs underexposed vs geometric shift 비교
3. **CLIP feature 동일 분석** — DINOv2와 CLIP의 layer 프로파일 차이

### _lessons.md 승격 여부
- [x] 승격 필요 → "DINOv2 layer 8 robustness sweet spot", "layer-wise shift는 non-monotonic"

---

## Step 2.3 결과: Entanglement 측정 (추가, 2026-03-26)

### Layer 11 (Last) vs Layer 8 (Robust) 비교

| Metric | Layer 11 | Layer 8 |
|--------|---------|---------|
| Shift-Anomaly cosine sim | -0.188 | 0.063 |
| PC projection correlation | **0.528** | **0.116** |
| Shared top-10 PC ratio | **40%** | **30%** |
| Shift magnitude | 2.12 | 0.28 |
| Anomaly magnitude | 5.50 | 0.52 |

### Layer 11 Top PC 분석

| PC | ExpVar% | Shift | Anomaly | Dominant |
|----|---------|-------|---------|----------|
| 0 | 19.97% | 0.177 | 0.101 | SHIFT |
| 1 | 18.65% | 0.037 | 0.214 | ANOMALY |
| 2 | 14.17% | 0.123 | 0.046 | SHIFT |
| 3 | 12.87% | 0.007 | 0.178 | ANOMALY |
| 4 | 11.37% | 0.215 | **0.707** | SHARED/ANOMALY |
| 5 | 6.24% | 0.167 | 0.371 | ANOMALY |
| 6 | 5.25% | **0.463** | 0.267 | SHIFT |

### t-SNE centroid distances (Layer 8)

| Pair | Distance |
|------|----------|
| normal_regular ↔ anomaly_regular | 0.87 |
| normal_regular ↔ normal_shifted | **1.85** |
| anomaly_shifted ↔ normal_shifted | **1.09** |
| normal_regular ↔ anomaly_shifted | 2.93 |

**핵심**: shifted-normal이 anomaly 방향으로 이동 (normal_shifted ↔ anomaly_shifted = 1.09로 가까움)

### 결론
- Layer 11에서 **PC projection correlation 0.53** → 중간 수준 entanglement 확인
- Layer 8은 **0.12**로 자연 disentangled → layer 8이 robust한 이유
- t-SNE에서 shifted-normal ↔ anomaly 겹침 확인 → false positive 원인
- **Phase 3 경로 α (Post-hoc Linear Disentanglement) 정당화됨**

---

## 관련 노트
- 선행: [Phase 1 결과](../2026-03-26_phase1_baseline_reproduction/report.md)
- 후속: Phase 3 method design
- 분석: [plan.md](../../../plan.md) Step 2.1, 2.2, 2.3
- 데이터: `results/phase2/shift_metrics_dinov2.json`, `results/phase2/entanglement_*.json`, `results/phase2/tsne_coordinates.json`
