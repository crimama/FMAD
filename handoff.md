# Handoff — 인수인계 메모 (2026-03-27)

## 지금까지 끝난 일

- **Phase 1**: 3개 baseline(Dinomaly, AnomalyDINO, AnomalyCLIP) MVTec AD 재현 완료 + MVTec AD 2 stress test 완료
- **Phase 2**: DINOv2 feature space 분석 — layer별 shift profile, entanglement 정량화, t-SNE
- **Phase 3**: NSP(Nuisance Subspace Projection) 방법론 개발 + autoresearch 16 experiments
  - Best: AD2 83.8% (+29.4pp), AD1 96.3% (-0.2pp), K=100 L8 Mahalanobis
  - 10개 novel method 시도 → 모두 discard (단순 NSP가 최적)
- **방향 결정**: 방법론 novelty 우선 탐색 → 안 되면 분석+이론 논문으로 pivot

## 지금 닿은 지점

- Branch: `autoresearch/mar26_nsp` (코드 변경은 exp2까지 keep, 나머지 discard하여 reset됨)
- `scripts/phase3_nsp_experiment.py`: CLS+patch L2norm + Mahalanobis 까지 반영된 상태
- `results.tsv`: 16 experiments 전체 기록
- `results/phase2/`: shift_metrics, entanglement, tsne JSON
- `results/phase3/`: NSP 결과 JSON들
- 실험 보고서 3개: `skill_graph/experiments/2026-03-26_phase{1,2,3}_*/report.md`

## 다음에 볼 파일

- `plan.md` — 전체 연구 계획 (Phase 4: Validation 아직 미진입)
- `program.md` — autoresearch 설정
- `results.tsv` — 전체 실험 이력
- `skill_graph/experiments/2026-03-26_phase3_linear_disentanglement/report.md` — Phase 3 상세 결과

## 다음에 할 일

### Plan A 진행 (방법론 novelty)
- NSP를 넘어서는 새로운 방법론 탐색 (단순 hyperparameter 변형 X)
- 후보: test-time adaptive projection, learnable nuisance basis, patch-level spatial disentanglement, nonlinear ICA
- 2주 내 +3pp 또는 novelty 확보 못하면 Plan B로 pivot

### 미완성 작업 (두 방향 공통)
- RobustAD 평가 완성 (eval adapter .jpg 대응 필요)
- 기존 robustness method 비교 (SuperAD, RoBiS, FiCo)
- Pixel-level 평가 추가
- 이론적 formal statement 작성

## 열린 질문

- 방법론 novelty의 구체적 방향은? (test-time adaptation? learnable? theoretical?)
- Pivot 판단 기준: +3pp이면 충분한가? novelty의 정성적 기준은?
- DINOv2 ViT-B/14 외 다른 backbone (ViT-L, CLIP)에서도 동일 효과인지?

## 관련 파일

- `plan.md` — 전체 5-phase 연구 계획
- `program.md` — autoresearch 루프 설정
- `results.tsv` — 실험 이력
- `skill_graph/experiments/` — 실험 보고서 3개
- `skill_graph/analysis/fm_ad_robustness/` — Phase 1-2 사전 분석 7개 문서
- `memory/project_direction_decision.md` — 방향 결정 기록
