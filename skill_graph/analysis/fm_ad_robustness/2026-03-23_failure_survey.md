# FM-based AD Robustness Failure Survey

작성일: 2026-03-23
분석 범위: DINOv2/CLIP 기반 AD 방법론의 robustness 실패 사례 종합

---

## 핵심 요약

Foundation Model(FM) 기반 AD 방법들은 통제된 벤치마크(MVTec AD, VisA)에서 near-perfect하지만, **현실적 조건에서 심각하게 열화**된다. 실패는 크게 4가지 축으로 분류된다: (1) 조명/환경 변화, (2) 카테고리 스케일링, (3) 미세 결함, (4) adversarial 취약성.

---

## 1. DINOv2 기반 AD 실패 사례

### 1.1 Adversarial 취약성
- **출처**: "Towards Adversarial Robustness in DINOv2-based Few-Shot AD" (arXiv 2510.13643, 2024)
- **결과**: FGSM 공격 시 MVTec AD AUROC **97.55% → 59.68%** (4-shot), **38pp 하락**
- **원인**: 임퍼셉터블 perturbation이 nearest-neighbor 관계를 뒤집음
- **부가 발견**: Raw anomaly score의 calibration이 매우 불량 (높은 ECE)

### 1.2 AnomalyDINO 한계
- **출처**: AnomalyDINO (WACV 2025)
- **실패 모드**:
  - 1-shot에서 foreground masking이 관련 패치를 모두 잡지 못함
  - 전처리 시 잘못된 rotation → 회전된 anomaly 미탐지
  - Natural-image pretraining ↔ industrial domain 간 feature misalignment

### 1.3 CLS Token Bias
- **출처**: AD-DINOv3 (arXiv 2025)
- **발견**: CLS token이 generic foreground object에 편향 → 미세한 결함 인지 불가
- **메커니즘**: Natural-image pretraining에서 salient object에 attend하도록 학습됨

### 1.4 Dinomaly 스케일링 한계
- **출처**: Dinomaly (CVPR 2025) / ADNet (arXiv 2511.20169)
- **결과**: One-for-one 90.6% → Multi-class 380 categories에서 **78.5%** (12pp 하락)
- **비고**: Dinomaly 자체는 robustness 평가를 수행하지 않음

---

## 2. CLIP 기반 AD 실패 사례

### 2.1 Anomaly-Unawareness (근본적 한계)
- **출처**: AA-CLIP (CVPR 2025)
- **핵심**: CLIP은 anomaly를 본 적이 없음 → normal/abnormal 의미 구분 능력 결여
- **결과**: Foreground object semantics에 attend하지, defect에 attend하지 않음

### 2.2 WinCLIP 한계
- **출처**: WinCLIP (CVPR 2023)
- **결과**: Zero-shot VisA에서 Image AUROC 78.1% / Pixel AUROC 79.6%
- **원인**: Handcrafted text prompt의 한계, 카테고리별 다른 prompt 생성 불가

### 2.3 AnomalyCLIP 한계
- **출처**: AnomalyCLIP (ICLR 2024)
- **한계**: Pixel-level annotation 필요, global alignment이 local anomaly segmentation에 부정확

### 2.4 CLIP-AD 공통 실패 패턴
- Image-level vs. pixel-level 간 성능 격차 큼
- 도메인 간 일반화 불안정
- Upstream pretraining bias가 downstream AD에 직접 전파

---

## 3. 벤치마크별 실패 증거

### 3.1 MVTec AD 2 (2025)
- **출처**: arXiv 2503.21622, CVPR 2025 VAND Workshop
- **데이터**: 8 scenarios, 8,000+ high-res images
- **도전 요소**: 투명/겹치는 물체, dark-field/backlight 조명, 극소 결함

| Method | Avg AU-PRO(0.05) | 비고 |
|--------|-------------------|------|
| EfficientAD | ~58.7% | |
| PatchCore | ~53.8% | |
| Best SOTA | < 60% | MVTec AD v1에서는 >90% |
| Can (worst) | **4-9%** | 반사 금속 위 미세 인쇄 오류 |

- **조명 변화 영향**: EfficientAD, MSFlow > 10pp AU-PRO 하락 / RD, PatchCore ≤ 3pp
- **평가 기준**: AU-PRO(0.05) 사용 (기존 0.30 대비 엄격 → 미세 결함 중요도 동등화)

### 3.2 RobustAD (CVPR 2025 Workshop, Amazon Science)
- **출처**: CVPR 2025 VAND 3.0, HuggingFace: AmazonScience/RobustAD
- **내용**: 제약/자동차/반도체 제조 환경, 9 anomaly types × 8 test domains
- **측정**: Average Relative Drop (ARD) — target domain vs. source domain 상대 성능 저하
- **결과**: EfficientAD/MSFlow > 10pp 하락 / PatchCore/RD ≤ 3pp

### 3.3 Real-IAD Variety (Pattern Recognition 2025)
- **출처**: arXiv 2511.00540
- **데이터**: 198,950 images, 160 categories, 28 industries
- **핵심 발견**: MUAD 방법들이 30 → 160 카테고리 스케일링 시 **10-20% 성능 저하**
- **흥미로운 대조**: Zero/few-shot 방법은 스케일링에 거의 영향 없음

### 3.4 MIRAD (arXiv 2510.16370, 2025)
- **내용**: 개별 맞춤 제조 환경, 10 categories, 6개 지리적 분산 노드
- **결과**: 모든 모델이 기존 벤치마크 대비 심각한 성능 하락

### 3.5 ADNet (arXiv 2511.20169, 2025)
- **내용**: 380 categories multi-domain benchmark
- **결과**: SOTA 90.6% (per-class) → **78.5%** (multi-class) — 12pp gap

---

## 4. 산업-학계 갭 (Beyond Academic Benchmarks)

- **출처**: "Beyond Academic Benchmarks" (CVPR 2025 VAND Workshop, Valeo/Intel)
- **핵심**: 통제된 벤치마크에서 잘 되는 방법이 실제 생산 환경에서 실패
- **원인**: 인위적 결함, 제한된 학습 이미지(15-30장), 추론 속도 요구(50-100ms)
- **경고**: 학계-산업 괴리가 AD 연구 방향을 왜곡할 위험

---

## 5. 실패 패턴 요약 (Taxonomy)

| 실패 모드 | 증거 | 심각도 |
|-----------|------|--------|
| **Adversarial 취약성** (DINOv2) | AUROC 97.5→59.7 (FGSM) | Critical |
| **MVTec AD 2 hard scenarios** | Best <60% AU-PRO; Can: 4-9% | Critical |
| **카테고리 스케일링** (MUAD) | 10-20% 하락 @160 categories | Severe |
| **조명 변화** | EfficientAD: >10pp 하락 | Moderate-Severe |
| **CLIP anomaly-unawareness** | Normal/abnormal 구분 불가 | Fundamental |
| **Pretraining domain gap** | CLS token foreground 편향 | Fundamental |
| **Multi-domain 스케일링** (ADNet) | 90.6→78.5% @380 categories | Severe |
| **학계-산업 괴리** | Lab→production 전이 실패 | Systemic |

---

## 6. 주요 관찰

1. **FM feature가 robust하다는 가정은 검증되지 않았다** — MVTec AD/VisA에서의 near-perfect 성능이 robustness로 오해되고 있음
2. **DINOv2와 CLIP은 다른 방식으로 실패한다** — DINOv2는 invariance로 인한 정보 소실, CLIP은 anomaly-unawareness
3. **스케일링이 숨겨진 위험** — 15 categories에서 잘 되는 unified 모델이 160+에서 무너짐
4. **새 벤치마크들이 문제를 가시화** — MVTec AD 2, RobustAD, Real-IAD Variety가 현실 조건 평가 가능하게 함

---

## 관련 노트

- [2026-03-23_mechanism_analysis.md](2026-03-23_mechanism_analysis.md) — FM feature 실패의 기술적 메커니즘
- [2026-03-23_solution_survey.md](2026-03-23_solution_survey.md) — 해결 방안 서베이
- [2026-03-23_ICLR_positioning.md](2026-03-23_ICLR_positioning.md) — 논문 포지셔닝 전략
- [../../papers/2026-03-23_핵심논문_목록.md](../../papers/2026-03-23_핵심논문_목록.md) — 전체 논문 목록
