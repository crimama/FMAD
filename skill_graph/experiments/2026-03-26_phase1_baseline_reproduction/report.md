# Phase 1: Baseline Reproduction & Robustness Stress Test — 2026-03-26

> **상태**: 🟢 완료 (RobustAD 일부 미완)
> **실험 ID**: `exp_20260326_phase1`
> **keywords**: baseline, reproduction, robustness, MVTec AD, MVTec AD 2, distribution shift

---

## 1. 문제 분석 (Problem Analysis)

### 현상
- FM-AD(Foundation Model 기반 AD) 방법들이 MVTec AD에서 near-perfect 성능을 보이지만, 실제 환경(조명 변화, domain shift)에서의 robustness는 검증되지 않음
- MVTec AD 2 벤치마크가 이를 테스트하기 위해 설계되었으나, 기존 방법들의 체계적 비교가 부족

### 원인 추정
- FM feature(DINOv2, CLIP)가 pretraining 시 학습한 invariance가 AD에 필요한 sensitivity와 충돌
- 조명/환경 변화가 feature space에서 anomaly signal과 겹칠 가능성

### 관련 선행 실험/분석
- [skill_graph/analysis/fm_ad_robustness/2026-03-23_failure_survey.md](../../analysis/fm_ad_robustness/2026-03-23_failure_survey.md)
- [skill_graph/analysis/fm_ad_robustness/2026-03-23_benchmark_analysis.md](../../analysis/fm_ad_robustness/2026-03-23_benchmark_analysis.md)

---

## 2. 가설 (Hypothesis)

### 주 가설
> FM-AD 방법들은 MVTec AD에서 높은 성능을 보이지만, MVTec AD 2의 distribution shift 조건에서 20pp 이상 하락할 것이다.

### 근거
- MVTec AD 2 논문에서 보고된 기존 방법들의 성능 하락 (EfficientAD -48%, MSFlow -59%)
- FM feature의 invariance-information destruction 이론 (Tian et al., NeurIPS 2020)

### 예상 결과
| 지표 | MVTec AD (baseline) | MVTec AD 2 (예상) | 비고 |
|------|--------------------|--------------------|------|
| I-AUROC | 91~99% | 60~80% | 20pp+ 하락 예상 |

---

## 3. 실험 설정 (Experiment Design)

### 대조군 (Control)
MVTec AD 15 카테고리 표준 평가

### 실험군 (Treatment)
| 조건명 | 데이터셋 | 비고 |
|--------|----------|------|
| MVTec AD | 15 cat, clean | 재현 확인 |
| MVTec AD 2 | 8 cat, lighting shift | test_public 사용 |
| RobustAD | MetalParts 7 domains | AnomalyCLIP만 완료 |

### 방법
| Method | Backbone | Type | Docker Image |
|--------|----------|------|-------------|
| Dinomaly | DINOv2-Reg ViT-B/14 | Unified zero-shot (10k iter 학습) | pilot-anomalydino (cu124) |
| AnomalyDINO | DINOv2 ViT-S/14 | Few-shot (4-shot) | pilot-anomalydino |
| AnomalyCLIP | CLIP ViT-L/14@336px | Zero-shot (VisA 학습→transfer) | pilot-anomalyclip |

### 평가 지표
- Primary: I-AUROC (Image-level AUROC)
- Secondary: P-AUROC, AU-PRO

### 환경
- GPU: NVIDIA RTX 4090 (24GB), Driver 570, CUDA 12.8
- PyTorch 2.5.1+cu124
- Docker 기반 격리 실행

---

## 4. 결과 (Results)

### Step 1.1 — MVTec AD 재현 결과

| Method | I-AUROC | P-AUROC | AU-PRO | 논문값 | 재현 |
|--------|---------|---------|--------|--------|------|
| Dinomaly | **99.64%** | 98.28% | 94.51% | ~99.6% | ✅ |
| AnomalyDINO (4-shot) | **97.7%** | - | - | ~97% | ✅ |
| AnomalyCLIP | **91.5%** | 91.0% | 81.4% | ~91.5% | ✅ |

### Step 1.2 — MVTec AD 2 Stress Test (I-AUROC)

| Category | Dinomaly | AnomalyDINO | AnomalyCLIP |
|----------|---------|------------|-------------|
| can | 52.8% | 56.7% | 54.4% |
| fabric | 69.8% | 66.6% | 73.0% |
| fruit_jelly | 85.9% | 88.2% | 68.3% |
| rice | 64.6% | 84.6% | 72.6% |
| sheet_metal | 79.5% | 80.8% | 54.4% |
| vial | 79.2% | 85.8% | 54.9% |
| wallplugs | **44.1%** | **41.8%** | **41.7%** |
| walnuts | 73.0% | 77.4% | 44.2% |
| **MEAN** | **68.6%** | **72.7%** | **57.9%** |

### 하락 요약

| Method | MVTec AD | MVTec AD 2 | 하락 |
|--------|---------|-----------|------|
| Dinomaly | 99.64% | 68.6% | **-31.0pp** |
| AnomalyDINO | 97.7% | 72.7% | **-25.0pp** |
| AnomalyCLIP | 91.5% | 57.9% | **-33.6pp** |

### RobustAD 부분 결과 (AnomalyCLIP, MetalParts)

| Domain | I-AUROC | P-AUROC | AU-PRO |
|--------|---------|---------|--------|
| test0 (ID) | 81.2% | 96.8% | 79.8% |
| test1 | 83.9% | 96.8% | 78.7% |
| test2 | 84.3% | 96.4% | 83.3% |
| test3 | 78.0% | 97.1% | 81.6% |
| test4 | 73.9% | 97.5% | 82.0% |
| test5 | 82.1% | 97.8% | 87.8% |
| test6 | 74.8% | 97.9% | 89.2% |

---

## 5. 결과 분석 (Analysis)

### 가설 검증
> **결론**: ✅ 가설 지지 — 3개 방법 모두 25~34pp 하락

### 상세 분석

1. **Backbone 무관 하락**: DINOv2 기반(Dinomaly, AnomalyDINO)과 CLIP 기반(AnomalyCLIP) 모두 심각 하락. FM feature의 공통 한계.

2. **wallplugs는 근본적 한계**: 3개 방법 모두 ~42% (random 수준). 반사 표면 + sub-pixel defect → 현재 FM feature로는 불가능.

3. **방법론 타입별 차이**:
   - Few-shot(AnomalyDINO, -25pp) > Zero-shot 학습(Dinomaly, -31pp) > Zero-shot transfer(AnomalyCLIP, -34pp)
   - 학습 기반 방법이 오히려 더 큰 하락 → **학습이 source domain에 overfitting**

4. **카테고리별 패턴**:
   - fruit_jelly, sheet_metal: DINOv2 방법 우위 (texture-aware)
   - fabric: AnomalyCLIP 우위 (semantic matching이 도움)
   - wallplugs, can: 모든 방법 실패

5. **RobustAD MetalParts**: domain 간 I-AUROC 편차 ~10pp, pixel-level은 안정적(96-98%)

### 부수 발견 (Side Findings)
- Dinomaly Docker 환경에서 CUDA 11.3 + RTX 4090 비호환 (sm_89 미지원) → cu124로 통합 필요
- MVTec AD 2의 test_public 구조가 MVTec AD와 다름 (test → test_public, good/bad) → compat layer 필요
- RobustAD PiledBags에 pixel mask 없음 → pixel 평가 불가
- pandas 2.x에서 `df.append` 제거됨 → `pd.concat` 마이그레이션 필요

---

## 6. 피드백 및 다음 단계 (Feedback & Next Steps)

### 교훈 (Lessons Learned)
- Docker 환경에서 CUDA 호환성은 사전 검증 필수 (driver version → CUDA toolkit → PyTorch wheel)
- 벤치마크 데이터셋 구조가 다르면 compat layer를 먼저 만들 것 (symlink 기반)
- 3개 이상 baseline 비교 시 공통 container(pilot-anomalydino)를 재활용하면 의존성 관리가 간결해짐

### 다음 실험 제안
1. **Phase 2: Feature Space 분석** — "왜 깨지는가" 메커니즘 규명
2. **RobustAD full 평가** — eval adapter .jpg 대응 (후순위)
3. **MVTec AD 2 test_private / test_private_mixed** 추가 평가 (lighting mix 조건)

### _lessons.md 승격 여부
- [x] 승격 필요 → Docker CUDA 호환성, 데이터셋 compat layer 패턴

---

## 관련 노트
- 선행: [analysis/fm_ad_robustness/](../../analysis/fm_ad_robustness/)
- 후속: [2026-03-26_phase2_feature_shift_analysis/](../2026-03-26_phase2_feature_shift_analysis/)
- 분석: [plan.md](../../../plan.md) Step 1.1, 1.2
