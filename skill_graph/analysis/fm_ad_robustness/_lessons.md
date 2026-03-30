# FM-AD Robustness — 검증된 패턴

> 분석 출처: `2026-03-23_failure_survey.md`, `2026-03-23_mechanism_analysis.md`, `2026-03-23_solution_survey.md`, `2026-03-23_theoretical_foundations.md`
> 최종 갱신: 2026-03-23

---

## 1. FM Feature Invariance ≠ AD Robustness
- **발견**: DINOv2/CLIP의 invariance는 classification robustness와 AD robustness를 동시에 보장하지 않는다. 오히려 AD에 필요한 fine-grained appearance 정보를 소실시킨다.
- **근거**: Phi-eat (2025) — DINOv2가 semantic grouping 우선, material/texture 무시. Contrastive learning theory (2020) — multiview assumption 위반 시 task-relevant info 소실.
- **적용 조건**: Texture/color/geometric defect 탐지가 목표인 모든 FM-based AD
- **주의**: Semantic anomaly (object 자체가 다른 경우)에서는 invariance가 오히려 유리할 수 있음

## 2. 카테고리 스케일링은 숨겨진 위험
- **발견**: Unified multi-class AD 모델은 15 → 160+ 카테고리에서 10-20pp 열화. Inter-class variance가 feature space를 지배하여 intra-class anomaly signal이 묻힘.
- **근거**: Real-IAD Variety (2025) — 160 cat에서 10-20% 하락. ADNet (2025) — 380 cat에서 12pp 하락. MINT-AD (2024) — inter-class interference 정의.
- **적용 조건**: Multi-class unified AD 모든 설정
- **주의**: Zero/few-shot 방법은 스케일링에 덜 민감 (category-specific adaptation이 있으므로)

## 3. MVTec AD/VisA 성능은 Robustness를 의미하지 않음
- **발견**: MVTec AD 99.9%, VisA 99.3%를 달성하는 방법이 MVTec AD 2에서 60% 미만, Can 카테고리에서 4-9%까지 추락.
- **근거**: MVTec AD 2 (2025), Beyond Academic Benchmarks (CVPR-W 2025)
- **적용 조건**: AD 방법의 실질적 성능을 판단할 때
- **주의**: MVTec AD 2의 AU-PRO(0.05) 기준이 기존보다 엄격 — 직접 비교 시 주의

## 4. Feature Disentanglement이 가장 높은 Novelty Gap
- **발견**: FM feature에서 nuisance-encoding과 anomaly-encoding subspace를 분리하는 연구가 0편. 이론(NeurIPS 2021)과 classification DG 실증(StyLIP)은 존재.
- **근거**: Solution survey 전체 검토 결과
- **적용 조건**: FM-AD robustness 연구 방향 설정 시
- **주의**: AD의 one-class 특성으로 인해 classification용 disentanglement을 직접 적용하기 어려울 수 있음

## 5. ICLR은 "분석 + 경량 해결" 구조를 선호
- **발견**: ICLR은 기존 통념 도전 논문을 명시적으로 환영. SOTA 불필요. 분석:해결 비율 ~60:40이 적절. 해결책은 단순하고 해석 가능해야 함.
- **근거**: ICLR 2026 Reviewer Guide 직접 인용. MMAD(ICLR 2025), Closer Look at CLIP(NeurIPS 2023) 등 선례.
- **적용 조건**: ICLR 논문 구조 설계 시
- **주의**: 분석만으로도 가능하지만 해결책이 있으면 수락 확률 크게 증가

## 6. Training-free DINOv2가 학습 기반 방법보다 조명 변화에 더 robust
- **발견**: SuperAD (training-free DINOv2 memory bank)는 TESTpriv→mix에서 3.5% AU-PRO 하락. RoBiS (학습+augmentation)는 11.3% 하락. EfficientAD/MSFlow/SimpleNet은 48-59% 하락.
- **근거**: SuperAD (arXiv:2505.19750), RoBiS (arXiv:2505.21152) — MVTec AD 2 벤치마크 결과
- **적용 조건**: 조명 변화 환경에서의 AD robustness 평가
- **주의**: Augmentation이 오히려 domain-specific bias를 도입할 수 있음 (RoBiS > SuperAD 하락). 그러나 RoBiS의 절대 성능은 더 높음.

## 7. Can/Wallplugs 카테고리는 현재 모든 방법이 실패
- **발견**: Can은 최고 방법(RoBiS)에서도 AucPro_0.05 30%/20%, SegF1 2%/1%. Wallplugs도 유사. 반사면 + 극소 결함 조합이 근본적 도전.
- **근거**: SuperAD Table 2, RoBiS Table 1,2 — 8개 방법 모두 일관된 실패
- **적용 조건**: 연구에서 "전체 카테고리 평균"만 보고하면 이 실패가 가려짐 → per-category 분석 필수
- **주의**: Can의 실패가 feature 문제인지 후처리/binarization 문제인지 분리 분석 필요

## 8. AU-PRO(0.05)는 기존 벤치마크 대비 6배 엄격한 기준
- **발견**: FPR 0~0.05 구간만 평가하므로, AUROC가 높아도 AU-PRO(0.05)는 매우 낮을 수 있음. MVTec AD에서 97%+ → MVTec AD 2에서 60% 미만.
- **근거**: MVTec AD 2 논문, SuperAD/RoBiS 실험 결과
- **적용 조건**: MVTec AD 2 결과를 MVTec AD와 비교할 때 반드시 메트릭 차이 명시
- **주의**: SegF1과 AU-PRO(0.05)의 상관관계도 카테고리마다 다름 (binarization threshold 의존)

## 9. Paired Observation 없이 Nuisance Direction 추정은 구조적으로 불가능
- **발견**: **4가지** unpaired 접근이 모두 실패. (1) TTNS: test-time ΔΣ → anomaly contamination으로 AD1 30.8%. (2) IFR: train inter-layer residual PCA → oracle과 74.2° (≈직교). (3) AdaBN: 전체 정규화 → AD1 78.5%. **(4) NN Pseudo-Pairing: per-sample NN + PCA → baseline보다 -4.2pp 악화. D0 진단 통과(87.6% alignment)에도 불구하고 PCA aggregation에서 anomaly 재오염.**
- **근거**: IFR diagnostic (2026-03-28), TTNS GO/NO-GO (2026-03-27), NN Pseudo-Pairing (2026-03-28)
- **적용 조건**: AD에서 nuisance subspace를 추정하려는 모든 시도. One-class AD의 구조적 한계.
- **이유**: (a) Test data에는 anomaly가 섞여 anomaly variance ⊂ shift estimate — batch-level이든 per-sample→PCA이든 aggregation 단계에서 재발. (b) Train data에는 shift가 없어 train variance ⊥ shift direction. (c) FM 내부 layer 구조(Δ)도 shift가 아닌 semantic refinement를 포착.
- **함의**: Nuisance direction 추정에는 paired observation(같은 물체, 다른 환경) 또는 shift의 물리적 사전 지식(augmentation 유형) 중 하나가 필수적이다.

## 11. Subspace Alignment 진단(normal-only)은 AD 성능을 보장하지 않는다
- **발견**: NN Pseudo-Pairing의 D0 진단에서 oracle shift explained 87.6%로 통과했으나, 본 실험에서 -4.2pp 악화. 진단은 normal test만 사용, 본 실험은 normal + anomaly 혼합.
- **근거**: NN Pseudo-Pairing diagnostic vs experiment (2026-03-28)
- **적용 조건**: Subspace 기반 nuisance 방법의 사전 검증 설계 시.
- **함의**: 진단에 "anomaly 혼합 시 추정 오염도" 측정을 반드시 포함해야 한다. Normal-only alignment만으로는 GO 판정에 불충분.

## 10. 진단 실험(Subspace Alignment)으로 사전 검증하면 연구 시간 절약
- **발견**: IFR의 핵심 가정을 30분짜리 진단(principal angle + shift variance explained)으로 사전 검증하여, 1-2일의 본 실험을 회피.
- **근거**: IFR diagnostic D1=74.2°, D2=12.6% → 본 실험 NO-GO 확정.
- **적용 조건**: Subspace 기반 방법론을 제안할 때, 추정된 subspace와 oracle subspace의 alignment을 먼저 확인.
- **방법**: `cos(principal_angles) = SVD(Q_est^T @ Q_oracle)`. Mean angle < 45° = GO, > 60° = NO-GO.

## 12. Nuisance subspace는 고차원(≥100)이므로 minimal paired calibration에 한계
- **발견**: NSP K=100에서 paired sample 수를 줄이면 saturation이 매우 느림. 100쌍→75% AD2(full의 62%), 50쌍→68%(33%), 10쌍→62%(9%). "5-10쌍이면 충분"이라는 기대는 불일치.
- **근거**: Minimal Calibration experiment (2026-03-28)
- **이유**: Nuisance subspace가 100+ 차원으로 고차원. PCA의 sample complexity로 인해 N ≫ K가 필요. K=100이면 최소 100-200쌍 필요.
- **함의**: Practical guide로서 saturation curve 제공. 100쌍(카테고리당 ~12) = 실용적 최소선.

## 13. Per-category nuisance 추정이 global pooling보다 우수
- **발견**: Per-category NSP(87.4%) > Global NSP(84.4%). 3pp 차이. Category-specific shift 존재.
- **근거**: Minimal Calibration experiment (2026-03-28)
- **함의**: Paired calibration 시 global이 아닌 category별로 수행하는 것이 더 efficient.

## 14. Feature space에서 shift는 spatially non-uniform (uniformity=2.6%)
- **발견**: Pixel space에서 균일한 조명 변화라도 DINOv2 feature space에서는 patch마다 다르게 영향. Spatial mean subtraction 효과 미미(+0.5pp).
- **근거**: SPAD diagnostic (2026-03-28), uniformity = 1 - residual_var/total_var = 0.026
- **이유**: 각 patch의 semantic content가 다르므로 DINOv2가 같은 pixel-level shift를 다르게 인코딩. Attention cross-patch interaction도 기여.
- **함의**: Image-level shift를 spatial mean으로 근사하는 접근은 무효. Patch별 nuisance 처리 필요.

## 15. Patch-level NSP가 일부 카테고리에서 image-level NSP를 초과
- **발견**: Patch NSP global(69.5%) < Image NSP(84.9%) 전체 평균. 그러나 walnuts에서 patch NSP(83.8%) > image NSP(72.9%) = +10.9pp 초과.
- **근거**: Patch-level NSP experiment (2026-03-28)
- **이유**: Walnuts의 anomaly가 매우 국소적 → patch-level scoring이 spatial structure를 활용. Image-level은 global aggregation에서 local anomaly 신호 희석.
- **함의**: Patch-level + image-level의 결합, 또는 scoring 방식(kNN vs Mahalanobis)이 핵심 개선 방향.

---

## 미검증
- ~~DINOv2 feature의 어떤 layer가 lighting shift에 가장 취약한지~~ → **검증됨**: Layer 4-7 (mid) 가장 취약, Layer 8 유일하게 robust (Phase 2)
- CLIP vs DINOv2 중 어떤 것이 어떤 shift type에 더 robust한지
- ~~Feature disentanglement이 AD 성능을 유지하면서 robustness를 개선할 수 있는지~~ → **검증됨**: NSP +29.4pp, AD1 유지 (Phase 3). 단, paired data 필수.
- ~~Test-time feature adaptation의 실제 효과~~ → **검증됨**: TTNS, AdaBN 모두 실패 또는 유해 (TTNS GO/NO-GO)
- SINDER의 high-norm token 처리가 robustness에 미치는 영향
- RobustAD에서의 FM-based 방법 성능 (논문 미출판, 결과 미확인)
- Can 카테고리 실패가 feature 문제 vs binarization 문제 어디에 기인하는지
- ~~FM 내부 layer 구조(Δ)로 nuisance direction 추정 가능한지~~ → **기각됨**: IFR diagnostic D1=74.2°, D2=12.6% (2026-03-28)
- ~~Shift는 spatially uniform한가~~ → **기각됨**: uniformity=2.6% (SPAD diagnostic 2026-03-28)
- **Patch-level NSP + Mahalanobis scoring 결합 시 image NSP 초과 가능한가** → 미시도

---

## 관련 노트
- [2026-03-23_failure_survey.md](2026-03-23_failure_survey.md)
- [2026-03-23_mechanism_analysis.md](2026-03-23_mechanism_analysis.md)
- [2026-03-23_solution_survey.md](2026-03-23_solution_survey.md)
- [2026-03-23_ICLR_positioning.md](2026-03-23_ICLR_positioning.md)
- [2026-03-23_theoretical_foundations.md](2026-03-23_theoretical_foundations.md)
- [2026-03-23_benchmark_analysis.md](2026-03-23_benchmark_analysis.md)
