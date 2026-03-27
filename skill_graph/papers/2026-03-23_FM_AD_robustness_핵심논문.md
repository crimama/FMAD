# FM-AD Robustness 관련 핵심 논문

작성일: 2026-03-23

---

## A. Robustness 실패 증거 (Benchmarks & Analysis)

| # | 논문 | Venue | 핵심 내용 | 중요도 |
|---|------|-------|----------|--------|
| 1 | **MVTec AD 2** | arXiv 2503.21622, CVPR-W 2025 | 8 hard scenarios, SOTA <60% AU-PRO | ★★★ |
| 2 | **RobustAD** | CVPR-W 2025 (Amazon) | Domain shift benchmark, ARD metric | ★★★ |
| 3 | **Real-IAD Variety** | Pattern Recognition 2025 | 160 cat, 10-20% scaling degradation | ★★★ |
| 4 | **ADNet** | arXiv 2511.20169 | 380 cat, 90.6→78.5% multi-class | ★★☆ |
| 5 | **MIRAD** | arXiv 2510.16370 | Individualized manufacturing, 6 nodes | ★★☆ |
| 6 | **Beyond Academic Benchmarks** | CVPR-W 2025 (Valeo/Intel) | Lab→production gap analysis | ★★☆ |
| 7 | **Adversarial DINOv2 AD** | arXiv 2510.13643 | AUROC 97.5→59.7 under FGSM | ★★☆ |

## B. FM Feature 분석 (Mechanisms)

| # | 논문 | Venue | 핵심 내용 | 중요도 |
|---|------|-------|----------|--------|
| 8 | **Phi-eat** | arXiv 2511.11270 | DINOv2 = semantic grouping, not material/texture | ★★★ |
| 9 | **SINDER** | ECCV 2024 | DINOv2 high-norm token artifacts (434 vs 57.6) | ★★★ |
| 10 | **Closer Look at CLIP Robustness** | NeurIPS 2023 | 83 CLIP models, data > method for robustness | ★★★ |
| 11 | **Data Determines CLIP Robustness** | ICML 2022 | 데이터 다양성이 robustness 원인, 방법론 아님 | ★★☆ |
| 12 | **SSL Isolates Content from Style** | NeurIPS 2021 | Disentanglement identifiability 이론 | ★★☆ |
| 13 | **FM for AD Survey** | arXiv 2502.06911 | FM이 upstream-downstream mismatch 시 열화 | ★★☆ |
| 14 | **AD-DINOv3** | arXiv 2025 | CLS token foreground bias | ★☆☆ |

## C. Robustness 해결 시도

| # | 논문 | Venue | 핵심 내용 | 중요도 |
|---|------|-------|----------|--------|
| 15 | **PILOT** | BMVC 2025 | TTA for zero-shot AD (prompt adaptation) | ★★★ |
| 16 | **ROADS** | WACV 2025 | Robust prompt-driven multi-class AD under shift | ★★★ |
| 17 | **FiCo** | AAAI 2025 | Distribution-invariant normality capture | ★★☆ |
| 18 | **Robust Distribution Alignment** | arXiv 2025.03 | Sinkhorn distance for industrial AD shift | ★★☆ |
| 19 | **FOCAL** | ICML 2025 | FM prior test-time canonicalization | ★★☆ |
| 20 | **FB-CLIP** | arXiv 2026.03 | FG-BG disentanglement for zero-shot CLIP AD | ★★☆ |
| 21 | **StyLIP** | WACV 2024 | CLIP style/content disentanglement for DG | ★★☆ |
| 22 | **ADPretrain** | NeurIPS 2025 | AD 전용 pretraining | ★★☆ |
| 23 | **EPHAD** | NeurIPS 2025 | Post-hoc evidence adjustment | ★☆☆ |

## D. Multi-Class / Scaling 해결 시도

| # | 논문 | Venue | 핵심 내용 | 중요도 |
|---|------|-------|----------|--------|
| 24 | **MINT-AD** | arXiv 2024 | Inter-class interference 정의 + 해결 | ★★☆ |
| 25 | **UniMMAD** | arXiv 2025 | MoE-driven feature decompression | ★★☆ |
| 26 | **Class-Aware Contrastive** | arXiv 2024 | Contrastive separation for multi-class | ★☆☆ |

## E. ICLR 포지셔닝 참고

| # | 논문 | Venue | 참고 이유 |
|---|------|-------|----------|
| 27 | **AnomalyCLIP** | ICLR 2024 | "문제 진단 + fix" 프레이밍 참고 |
| 28 | **MMAD** | ICLR 2025 | "FM 한계 노출 benchmark" 선례 |
| 29 | **Can LLMs Generate Ideas** | ICLR 2025 | "통념 도전" 프레이밍 참고 |
| 30 | **Language Modeling is Compression** | ICLR 2024 | 우아한 reframing 참고 |

---

## 관련 노트

- [../analysis/fm_ad_robustness/2026-03-23_failure_survey.md](../analysis/fm_ad_robustness/2026-03-23_failure_survey.md)
- [../analysis/fm_ad_robustness/2026-03-23_mechanism_analysis.md](../analysis/fm_ad_robustness/2026-03-23_mechanism_analysis.md)
- [../analysis/fm_ad_robustness/2026-03-23_solution_survey.md](../analysis/fm_ad_robustness/2026-03-23_solution_survey.md)
- [../analysis/fm_ad_robustness/2026-03-23_ICLR_positioning.md](../analysis/fm_ad_robustness/2026-03-23_ICLR_positioning.md)
- [2026-03-23_핵심논문_목록.md](2026-03-23_핵심논문_목록.md) — 전체 VAD 논문 목록
