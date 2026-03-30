# Skill Graph Index

> 프로젝트 연구 기록의 키워드 기반 인덱스.
> 키워드 → 문서 링크로 그래프 탐색 가능.
> 새 문서 추가 시 이 파일도 함께 갱신할 것.

---

## 키워드 그래프

```
                  ┌──────────────────┐
                  │  FM-AD Robustness │
                  └────────┬─────────┘
          ┌────────────────┼────────────────┐
          ▼                ▼                ▼
  ┌───────────────┐ ┌──────────────┐ ┌────────────────┐
  │ Entanglement  │ │ Layer 8      │ │ Spectral Gap   │
  │ (shift↔anom)  │ │ Sweet Spot   │ │ (normal cov)   │
  └───────┬───────┘ └──────┬───────┘ └───────┬────────┘
          │                │                  │
          ▼                ▼                  ▼
  ┌───────────────┐ ┌──────────────┐ ┌────────────────┐
  │ NSP (PCA +    │ │ DINOv2 ViT   │ │ SAPP (variance │
  │ hard proj)    │ │ Feature Anal │ │ -gated proj)   │
  └───────┬───────┘ └──────────────┘ └───────┬────────┘
          │                                   │
          └──────────────┬────────────────────┘
                         ▼
              ┌──────────────────┐
              │ ICLR 2027 Paper  │
              │ (Analysis 60% +  │
              │  Method 40%)     │
              └──────────────────┘
```

---

## 키워드 → 문서 매핑

| 키워드 | 문서 | 카테고리 |
|--------|------|----------|
| **FM-AD robustness** | [Phase 1 Baseline](experiments/2026-03-26_phase1_baseline_reproduction/report.md), [실패 메커니즘](analysis/fm_ad_robustness/2026-03-23_mechanism_analysis.md) | experiments, analysis |
| **entanglement** | [Phase 2 분석](experiments/2026-03-26_phase2_feature_shift_analysis/report.md), [방법론 상세](analysis/fm_ad_robustness/2026-03-27_methodology_nsp_to_sapp.md) | experiments, analysis |
| **test-time adaptation** | [비교군/일반성 분석](analysis/fm_ad_robustness/2026-03-27_comparison_and_generality_analysis.md), [방법론 상세](analysis/fm_ad_robustness/2026-03-27_methodology_nsp_to_sapp.md) | analysis |
| **comparison/baseline** | [비교군/일반성 분석](analysis/fm_ad_robustness/2026-03-27_comparison_and_generality_analysis.md) | analysis |
| **dataset dependency** | [비교군/일반성 분석](analysis/fm_ad_robustness/2026-03-27_comparison_and_generality_analysis.md) | analysis |
| **NSP** | [Phase 3 실험](experiments/2026-03-26_phase3_linear_disentanglement/report.md), [방법론 상세](analysis/fm_ad_robustness/2026-03-27_methodology_nsp_to_sapp.md) | experiments, analysis |
| **SAPP** | [SAPP 제안서](ideas/2026-03-26_ASAD_proposal.md), [GO/NO-GO](experiments/2026-03-27_sapp_go_nogo/report.md), [방법론 상세](analysis/fm_ad_robustness/2026-03-27_methodology_nsp_to_sapp.md) | ideas, experiments |
| **spectral gap** | [SAPP 제안서](ideas/2026-03-26_ASAD_proposal.md), [이론적 기반](analysis/fm_ad_robustness/2026-03-23_theoretical_foundations.md) | ideas, analysis |
| **DINOv2 layer** | [Phase 2 분석](experiments/2026-03-26_phase2_feature_shift_analysis/report.md), [Shift-Equivariant](analysis/fm_ad_robustness/2026-03-26_shift_equivariant_scoring_proposal.md) | experiments, analysis |
| **linear optimality** | [Phase 3 실험](experiments/2026-03-26_phase3_linear_disentanglement/report.md), [4팀 사고실험](analysis/2026-03-27_method_thought_experiment.md) | experiments, analysis |
| **ICLR positioning** | [포지셔닝 전략](analysis/fm_ad_robustness/2026-03-23_ICLR_positioning.md), [4팀 사고실험](analysis/2026-03-27_method_thought_experiment.md) | analysis |
| **nonlinear ICA** | [이론적 기반](analysis/fm_ad_robustness/2026-03-23_theoretical_foundations.md), [SAPP 제안서](ideas/2026-03-26_ASAD_proposal.md) | analysis, ideas |
| **causal feature** | [CFF 제안](ideas/2026-03-26_causal_feature_factorization.md) | ideas |
| **layer decomposition** | [Layer-Decomposed AD](ideas/2026-03-26_layer_decomposed_AD.md) | ideas |

---

## 문서 → 키워드 역매핑

### experiments/

| 문서 | 상태 | 키워드 |
|------|------|--------|
| [Phase 1: Baseline Reproduction](experiments/2026-03-26_phase1_baseline_reproduction/report.md) | ✅ 완료 | `FM-AD` `robustness` `MVTec AD 2` `baseline` |
| [Phase 2: Feature Shift Analysis](experiments/2026-03-26_phase2_feature_shift_analysis/report.md) | ✅ 완료 | `entanglement` `layer analysis` `DINOv2` `t-SNE` |
| [Phase 3: Linear Disentanglement](experiments/2026-03-26_phase3_linear_disentanglement/report.md) | ✅ 완료 | `NSP` `PCA` `Mahalanobis` `linear optimality` |
| [SAPP GO/NO-GO](experiments/2026-03-27_sapp_go_nogo/report.md) | ⏸ 보류 | `SAPP` `omega` `variance-gated` `spectral gap` |
| [TTNS GO/NO-GO](experiments/2026-03-27_ttns_go_nogo/report.md) | ❌ NO-GO | `TTNS` `test-time` `covariance shift` `anomaly contamination` |
| [IFR Diagnostic](experiments/2026-03-28_ifr_diagnostic/report.md) | ❌ NO-GO | `IFR` `inter-layer residual` `subspace alignment` `negative result` |
| [NN Pseudo-Pairing](experiments/2026-03-28_nn_pseudopair/report.md) | ❌ NO-GO | `NN` `pseudo-pairing` `robust PCA` `anomaly contamination` `negative result` |
| [Minimal Calibration](experiments/2026-03-28_minimal_calibration/report.md) | ✅ 완료 | `minimal calibration` `sample efficiency` `saturation curve` `paired data` |
| SPAD Diagnostic | ❌ NO-GO | `SPAD` `spatial uniformity` `patch-level` `uniformity=2.6%` |
| Patch-level NSP | 🟡 진행중 | `patch NSP` `global projection` `69.5%` `walnuts +10.9pp` |

### analysis/

| 문서 | 키워드 |
|------|--------|
| [방법론 상세: NSP→Test-time](analysis/fm_ad_robustness/2026-03-27_methodology_nsp_to_sapp.md) | `NSP` `SAPP` `test-time` `covariance shift` `direction decision` |
| [비교군/일반성 분석](analysis/fm_ad_robustness/2026-03-27_comparison_and_generality_analysis.md) | `comparison` `baseline` `MVTec AD 2 dependency` `generality` `test-time` |
| [7가지 실패 메커니즘](analysis/fm_ad_robustness/2026-03-23_mechanism_analysis.md) | `invariance` `semantic dominance` `high-norm token` `entanglement` |
| [이론적 기반](analysis/fm_ad_robustness/2026-03-23_theoretical_foundations.md) | `nonlinear ICA` `information theory` `identifiability` |
| [ICLR 포지셔닝](analysis/fm_ad_robustness/2026-03-23_ICLR_positioning.md) | `ICLR` `paper structure` `analysis paper` |
| [4팀 사고 실험](analysis/2026-03-27_method_thought_experiment.md) | `ITND` `SANA` `GAID` `ECND` `direction decision` |
| [Shift-Equivariant Scoring](analysis/fm_ad_robustness/2026-03-26_shift_equivariant_scoring_proposal.md) | `SEAS` `equivariance` `shift` |
| [FM 내부 구조 분석](analysis/fm_ad_robustness/2026-03-28_fm_internal_nuisance_identification.md) | `IFR` `residual stream` `attention` `cross-layer` `nuisance` |
| [검증된 패턴](analysis/fm_ad_robustness/_lessons.md) | `lessons` `verified patterns` |
| [Failure Survey](analysis/fm_ad_robustness/2026-03-23_failure_survey.md) | `failure cases` `benchmark` |
| [Solution Survey](analysis/fm_ad_robustness/2026-03-23_solution_survey.md) | `novelty gap` `existing methods` |
| [Disentanglement Deep Survey](analysis/feature_disentanglement/2026-03-23_deep_survey.md) | `disentanglement` `PISCO` `ICA` |
| [VAD Trends](analysis/vad_trend/2026-03-23_최신트렌드분석.md) | `trend` `zero-shot` `multi-class` |
| [Dataset Survey](analysis/vad_datasets/2025-03-23_comprehensive_dataset_survey.md) | `MVTec` `RobustAD` `Real-IAD` |

### papers/

| 문서 | 구분 | 키워드 |
|------|------|--------|
| [FM-AD 핵심 논문 10선](papers/2026-03-23_FM_AD_robustness_핵심논문.md) | Tier 1 | `MVTec AD 2` `SuperAD` `Phi-eat` `SINDER` `PISCO` |
| [필독 논문 목록](papers/2026-03-24_FM_AD_robustness_필독논문.md) | Tier 2 | `Dinomaly` `AnomalyCLIP` `PatchCore` `ResAD` |
| [Baseline 심층 분석](papers/2026-03-23_baseline_papers_deep_analysis.md) | baseline | `Dinomaly` `AnomalyDINO` `AnomalyCLIP` |

### ideas/

| 문서 | 상태 | 키워드 |
|------|------|--------|
| [SAPP (ASAD) 제안서](ideas/2026-03-26_ASAD_proposal.md) | 🟡 검증중 | `SAPP` `spectral gap` `variance-gated` |
| [Layer-Decomposed AD](ideas/2026-03-26_layer_decomposed_AD.md) | screened | `layer decomposition` `multi-layer` |
| [Causal Feature Factorization](ideas/2026-03-26_causal_feature_factorization.md) | screened | `causal` `factorization` `invariance` |
| [연구 방향 후보 6개](ideas/2026-03-23_연구방향_후보.md) | 선별완료 | `direction` `robustness` `disentanglement` |

---

## 문서 간 연결 (관련 노트)

```
papers/핵심논문_10선 ──────► analysis/mechanism_analysis
        │                          │
        │                          ▼
        │                   analysis/theoretical_foundations
        │                          │
        ▼                          ▼
experiments/phase1_baseline ──► experiments/phase2_feature_shift
        │                          │
        │                          ▼
        │              analysis/방법론_상세(NSP→SAPP) ◄── ideas/ASAD_proposal
        │                     │    ▲
        ▼                     │    │
experiments/phase3_NSP ───────┘    │
        │                          │
        ▼                          │
analysis/4팀_사고실험 ─────────────┘
        │
        ▼
analysis/비교군_일반성_분석
        │
        ▼
experiments/test-time_cov_shift (다음 실험)
        │
        ├─ NSP의 80%+ ──► 논문: Analysis + Test-time Method
        └─ 60% 미만   ──► Pivot: scope 축소 or 분석 논문
```

---

## 타임라인

| 날짜 | 문서 | 요약 |
|------|------|------|
| 2026-03-23 | [연구 방향 후보](ideas/2026-03-23_연구방향_후보.md) | 6개 방향 탐색, FM-AD Robustness 선정 |
| 2026-03-23 | [핵심논문 10선](papers/2026-03-23_FM_AD_robustness_핵심논문.md) | Tier 1 논문 깊이 분석 |
| 2026-03-23 | [7가지 실패 메커니즘](analysis/fm_ad_robustness/2026-03-23_mechanism_analysis.md) | FM-AD 실패 원인 체계적 분류 |
| 2026-03-23 | [이론적 기반](analysis/fm_ad_robustness/2026-03-23_theoretical_foundations.md) | Nonlinear ICA, 정보이론 프레임워크 |
| 2026-03-23 | [ICLR 포지셔닝](analysis/fm_ad_robustness/2026-03-23_ICLR_positioning.md) | Analysis 60% + Method 40% 구조 |
| 2026-03-26 | [Phase 1](experiments/2026-03-26_phase1_baseline_reproduction/report.md) | 3 SOTA × 2 benchmarks, 25-34pp 하락 확인 |
| 2026-03-26 | [Phase 2](experiments/2026-03-26_phase2_feature_shift_analysis/report.md) | Layer 8 sweet spot, entanglement 0.53 (L11) |
| 2026-03-26 | [Phase 3](experiments/2026-03-26_phase3_linear_disentanglement/report.md) | NSP +29.4pp, 10개 변형 실패 → linear optimality |
| 2026-03-26 | [SAPP 제안서](ideas/2026-03-26_ASAD_proposal.md) | Variance-gated projection 설계 |
| 2026-03-27 | [4팀 사고 실험](analysis/2026-03-27_method_thought_experiment.md) | "방법론 단독 ICLR 부족" 전원 일치, Plan B 시그널 |
| 2026-03-27 | [방법론 상세](analysis/fm_ad_robustness/2026-03-27_methodology_nsp_to_sapp.md) | NSP→SAPP 진화, 방향 결정 프레임워크 |
| 2026-03-27 | [비교군/일반성 분석](analysis/fm_ad_robustness/2026-03-27_comparison_and_generality_analysis.md) | MVTec AD 2 의존성 문제 식별, test-time 방향 전환 |
| 2026-03-27 | [SAPP GO/NO-GO](experiments/2026-03-27_sapp_go_nogo/report.md) | ⏸ 보류 (test-time 우선) |
| 2026-03-27 | [TTNS GO/NO-GO](experiments/2026-03-27_ttns_go_nogo/report.md) | ❌ NO-GO: test-time ΔΣ → anomaly contamination, AD1 30.8% |
| 2026-03-28 | [FM 내부 구조 분석](analysis/fm_ad_robustness/2026-03-28_fm_internal_nuisance_identification.md) | IFR 포함 5가지 FM-internal 아이디어 심층 분석 |
| 2026-03-28 | [IFR Diagnostic](experiments/2026-03-28_ifr_diagnostic/report.md) | ❌ NO-GO: train Δ ⊥ shift (74.2°, 12.6%) → unpaired 구조적 한계 |
| 2026-03-28 | [NN Pseudo-Pairing](experiments/2026-03-28_nn_pseudopair/report.md) | ❌ NO-GO: D0 통과(87.6%)에도 본 실험 -4.2pp → PCA aggregation에서 anomaly 재오염 |
| 2026-03-28 | [Minimal Calibration](experiments/2026-03-28_minimal_calibration/report.md) | Saturation curve: 100쌍→75%(62% recovery), per-cat(87.4%)>global(84.4%) |
| 2026-03-28 | SPAD Diagnostic | ❌ Shift spatially non-uniform (2.6%), spatial subtraction 무효 |
| 2026-03-28 | Patch-level NSP | 🟡 Global patch NSP 69.5% (+10.8pp), walnuts에서 img NSP 초과(+10.9pp) |
