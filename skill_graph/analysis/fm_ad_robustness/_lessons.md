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

---

## 미검증
- DINOv2 feature의 어떤 layer가 lighting shift에 가장 취약한지
- CLIP vs DINOv2 중 어떤 것이 어떤 shift type에 더 robust한지
- Feature disentanglement이 AD 성능을 유지하면서 robustness를 개선할 수 있는지
- Test-time feature adaptation의 실제 효과 (PILOT은 prompt만 적응)
- SINDER의 high-norm token 처리가 robustness에 미치는 영향
- RobustAD에서의 FM-based 방법 성능 (논문 미출판, 결과 미확인)
- Can 카테고리 실패가 feature 문제 vs binarization 문제 어디에 기인하는지

---

## 관련 노트
- [2026-03-23_failure_survey.md](2026-03-23_failure_survey.md)
- [2026-03-23_mechanism_analysis.md](2026-03-23_mechanism_analysis.md)
- [2026-03-23_solution_survey.md](2026-03-23_solution_survey.md)
- [2026-03-23_ICLR_positioning.md](2026-03-23_ICLR_positioning.md)
- [2026-03-23_theoretical_foundations.md](2026-03-23_theoretical_foundations.md)
- [2026-03-23_benchmark_analysis.md](2026-03-23_benchmark_analysis.md)
