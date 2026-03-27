# Research Plan — FM-AD Robustness & Feature Disentanglement

> **핵심 주장**: FM(DINOv2/CLIP) 기반 AD는 distribution shift에서 심각하게 열화되며, 이는 feature space에서 anomaly-relevant 정보와 nuisance 정보가 entangle되어 있기 때문이다. 이를 principled하게 disentangle하면 robust AD가 가능하다.

> **타겟**: ICLR 2027 (구조: 60% 분석 + 40% 해결책)

---

## Phase 1: Empirical Foundation (1–2주)

> **목적**: "FM-AD가 깨진다"는 주장의 정량적 증거 확보

### Step 1.1 — Baseline 재현 (MVTec AD)

- **방법**: Dinomaly, AnomalyDINO(4-shot), AnomalyCLIP → MVTec AD 평가
- **메트릭**: I-AUROC, P-AUROC, AU-PRO
- **예상 결과**: 논문 수치와 ±2pp 이내 재현 (Dinomaly ~99.6%, AnomalyCLIP ~91.5%)

```
결과 A: 재현 성공 (±2pp) → Step 1.2로 진행
결과 B: 재현 실패 (>3pp 차이) → 환경/config 디버깅, 원인 파악 후 재시도
```

### Step 1.2 — Robustness Stress Test

- **방법**: 동일 3개 방법을 MVTec AD 2 + RobustAD에서 평가
- **메트릭**: AU-PRO(0.05), ARD (RobustAD), category별 breakdown
- **예상 결과**:
  - MVTec AD 2: 전반적으로 큰 하락 (특히 lighting shift 시나리오에서 -10~50pp)
  - RobustAD: domain별 편차 큼 (color shift > geometric shift 순으로 취약)

```
결과 A: 명확한 failure pattern 발견 (특정 shift 유형에서 일관된 하락)
  → Phase 2로 진행 — 메커니즘 분석 대상 확정

결과 B: 예상보다 robust (하락 <5pp)
  → 연구 방향 재검토 필요
  ├── B-1: 더 어려운 shift 조건 탐색 (multi-factor shift, 극단 조건)
  └── B-2: category scaling 문제로 pivot (multi-class 160+ 카테고리)

결과 C: 모든 방법이 동일하게 실패 (방법론 무관)
  → 데이터셋 자체 난이도 문제 vs feature 문제 분리 필요
  → oracle feature (ground truth 근처 feature) 실험 추가
```

### Step 1.3 — Category별 Failure Taxonomy 구축

- **방법**: Step 1.2 결과에서 category × shift type × method 3차원 분석
- **산출물**: Failure heatmap (어떤 category가 어떤 shift에 취약한지)
- **예상 결과**: texture 카테고리(carpet, leather)는 lighting에 강하나 color shift에 약함. object 카테고리(screw, metal_nut)는 geometric shift에 약함

```
결과 A: category 특성별 failure pattern 분류 가능
  → Phase 2에서 feature 분석 시 representative category 선택 가이드

결과 B: pattern 없이 랜덤하게 실패
  → feature space 분석이 더 중요 (category 수준이 아닌 token 수준 분석 필요)
```

---

## Phase 2: Mechanism Analysis (2–3주)

> **목적**: "왜 깨지는가"에 대한 메커니즘 수준 증거 확보 (논문 Section 3–4의 핵심)

### Step 2.1 — Feature Space Shift 정량화

- **방법**:
  - Clean vs shifted 이미지 pair에서 DINOv2/CLIP feature 추출
  - Layer별 (shallow/mid/deep) feature distance 측정
  - t-SNE/UMAP으로 normal vs anomaly vs shifted-normal 분포 시각화
- **예상 결과**: shifted-normal이 anomaly와 겹치는 영역 존재 (false positive의 원인)

```
결과 A: shifted-normal ↔ anomaly 겹침 확인
  → "nuisance-semantic entanglement" 가설 지지
  → Step 2.2에서 어떤 차원이 겹치는지 분석

결과 B: shifted-normal은 별도 클러스터 (겹치지 않음, 단순히 threshold 문제)
  → score calibration 방향으로 pivot
  └── 간단한 domain-aware normalization으로 해결 가능할 수 있음
      → 해결되면: lightweight contribution (workshop급)
      → 해결 안 되면: 더 깊은 feature 분석 필요
```

### Step 2.2 — Layer별 Robustness 프로파일링

- **방법**:
  - DINOv2 layer 1–24에서 각각 feature 추출
  - 각 layer의 shift sensitivity 측정: `||f(x_clean) - f(x_shifted)|| / ||f(x_clean)||`
  - 동시에 각 layer의 anomaly discriminability 측정: AUROC per layer
- **예상 결과**:
  - Shallow layers (1–6): shift에 민감하지만 anomaly detection도 잘함
  - Deep layers (19–24): shift에 강하지만 texture anomaly 놓침
  - → robustness-sensitivity trade-off 존재

```
결과 A: 명확한 trade-off curve 존재
  → "단일 layer 선택으로는 해결 불가" → disentanglement 필요성 근거
  → Step 2.3으로 진행

결과 B: 특정 mid-layer가 둘 다 괜찮음
  → adaptive layer selection만으로 충분할 수 있음
  ├── B-1: 그 layer로 baseline 재평가 → 충분하면 contribution 약함
  └── B-2: category에 따라 optimal layer 다름 → layer selection + disentanglement 결합
```

### Step 2.3 — Entanglement 직접 측정

- **방법**:
  - PCA로 feature space의 주요 variation direction 추출
  - 각 principal component가 반응하는 것 분류:
    - semantic (object identity) / nuisance (lighting, viewpoint) / anomaly (defect)
  - Mutual information 또는 correlation 측정: `MI(PC_i, shift_label)` vs `MI(PC_i, anomaly_label)`
- **예상 결과**: 상위 PC들이 semantic + nuisance를 동시에 인코딩 (entangled)

```
결과 A: 명확한 entanglement 확인 (같은 PC가 shift와 anomaly 모두에 반응)
  → Phase 3의 disentanglement 접근 직접 정당화
  → 논문 Figure 1 후보: entanglement visualization

결과 B: 이미 어느 정도 분리되어 있음 (다른 PC가 다른 역할)
  → linear projection으로 간단히 분리 가능
  → 방법론이 더 간단해지지만 novelty도 줄어듦
  ├── B-1: 간단한 projection이 robustness 크게 개선 → "숨겨진 구조 발견" 스토리
  └── B-2: 개선 미미 → nonlinear entanglement → 더 강력한 방법 필요
```

### Step 2.4 — High-Norm Token Artifact 분석 (SINDER 재현)

- **방법**: DINOv2 patch token norm 분포 분석, high-norm token이 shift 시 어떻게 변하는지
- **예상 결과**: high-norm token이 shift amplifier 역할 (norm 434 vs 57)

```
결과 A: high-norm token 제거만으로 robustness 개선
  → register token 또는 norm clipping을 baseline에 포함

결과 B: high-norm token과 robustness 무관
  → artifact 문제와 robustness 문제는 독립 → 분석에서 분리
```

---

## Phase 3: Method Design (2–3주)

> **목적**: 분석에서 도출된 원인에 대한 principled solution 설계

### 경로 결정 (Phase 2 결과에 따라)

```
Phase 2 종합 결과에 따른 방법론 선택:

경로 α: Entanglement 심각 + Linear 분리 가능
  → Step 3A: Post-hoc Linear Disentanglement (PISCO 계열)

경로 β: Entanglement 심각 + Nonlinear
  → Step 3B: Lightweight Nonlinear Disentanglement (LoRA-based)

경로 γ: Layer별 trade-off가 핵심
  → Step 3C: Robustness-Aware Multi-Layer Fusion

경로 δ: Score-level 문제가 지배적
  → Step 3D: Domain-Aware Score Calibration (backup, contribution 작음)
```

### Step 3A — Post-hoc Linear Disentanglement (Primary Path)

- **방법**:
  - Multi-environment normal data (clean + shifted)에서 feature 수집
  - Environment label을 auxiliary variable로 사용
  - Linear projection으로 anomaly-relevant subspace와 nuisance subspace 분리
  - Anomaly scoring을 anomaly-relevant subspace에서만 수행
- **이론적 근거**: von Kügelgen (NeurIPS 2021) block identifiability + PISCO (ICML 2023)
- **예상 결과**: shifted 조건에서 AU-PRO 5–15pp 개선, clean에서는 ±1pp

```
결과 A: Shifted에서 큰 개선 + Clean 유지
  → 핵심 contribution 확보 → Phase 4 ablation으로 진행

결과 B: Shifted 개선되었으나 Clean 하락 (-3pp 이상)
  → information loss 문제
  ├── B-1: subspace dimension 조절로 trade-off 완화 시도
  └── B-2: anomaly-relevant subspace 정의 재검토 (너무 aggressive한 제거)

결과 C: 개선 미미 (<3pp)
  → linear disentanglement 불충분
  → 경로 β (nonlinear) 또는 경로 γ (multi-layer fusion) 시도
```

### Step 3B — Lightweight Nonlinear Disentanglement (Backup)

- **방법**:
  - Frozen FM + learnable lightweight projector (1–2 layer MLP or LoRA)
  - Contrastive objective: same object, different environment → nuisance direction
  - Anomaly scoring은 nuisance-orthogonal projection에서 수행
- **예상 결과**: Step 3A 대비 추가 3–5pp 개선, 학습 비용 소폭 증가

```
결과 A: 3A 대비 의미 있는 추가 개선
  → nonlinear component가 필요하다는 evidence → 논문 ablation 포인트

결과 B: 3A와 큰 차이 없음
  → linear로 충분 → Occam's razor, 3A 채택 (더 해석 가능)
```

### Step 3C — Robustness-Aware Multi-Layer Fusion (Alternative)

- **방법**:
  - Layer별 robustness score 사전 측정 (Step 2.2 활용)
  - Shift 감지 → 자동으로 robust layer에 가중치 증가
  - 간단한 attention-based fusion 또는 gating mechanism
- **예상 결과**: 방법론이 단순하지만 설명력 높음

```
결과 A: disentanglement 없이도 상당한 개선
  → contribution: "layer selection이 underexplored" 스토리
  → 단, novelty 상대적으로 낮음 → 3A/3B와 결합하면 더 강력

결과 B: 개선 제한적
  → layer 수준이 아닌 dimension 수준 분리 필요 확인 → 3A 강화 근거
```

---

## Phase 4: Validation & Ablation (2–3주)

> **목적**: 방법론의 효과를 체계적으로 증명 + 논문 table/figure 생성

### Step 4.1 — Main Results Table

- **방법**:
  - 선택된 방법 (3A/3B/3C 중) + 3개 baseline × 3개 벤치마크 전체 평가
  - Metric: I-AUROC, P-AUROC, AU-PRO(0.05), ARD
  - MVTec AD (clean), MVTec AD 2 (hard + lighting), RobustAD (9 shift types)
- **성공 기준**:
  - Shifted: baseline 대비 5pp+ 개선
  - Clean: 하락 없음 (±1pp)
  - 최소 2/3 벤치마크에서 일관된 개선

```
결과 A: 성공 기준 충족
  → Step 4.2 ablation 진행

결과 B: 일부 벤치마크에서만 개선
  → boundary 명확히 규정 (어떤 shift에서 효과적이고 어떤 것에서 아닌지)
  → "limitation" 섹션에 솔직하게 기술 — ICLR는 이를 긍정적으로 봄

결과 C: 전반적으로 개선 미미
  → 방법론 재설계 필요 → Phase 3로 회귀
  └── 단, Phase 2의 분석 contribution만으로도 workshop/analysis track 가능
```

### Step 4.2 — Ablation Study

- **설계 원칙**: one-factor change, 각 모듈의 독립 기여 증명
- **실험**:
  1. Disentanglement ON/OFF → 핵심 모듈 기여
  2. Linear vs Nonlinear projection → complexity 정당화
  3. Subspace dimension sweep → sensitivity 분석
  4. Single-layer vs Multi-layer feature 입력 → layer 선택 효과
  5. Environment label 수 (1, 2, 4, 8) → practical requirement 분석
- **예상 결과**: disentanglement가 가장 큰 기여, 나머지는 부가적

### Step 4.3 — Analysis Experiments (논문 설명력 강화)

- **실험**:
  1. Feature visualization: 전/후 t-SNE (shifted-normal이 정상으로 돌아오는지)
  2. Per-shift-type breakdown: 어떤 shift에서 가장 효과적인지
  3. Failure case analysis: 여전히 실패하는 경우는 무엇인지
  4. Computational overhead: inference time 비교
- **산출물**: Figure 2–4 후보

### Step 4.4 — Seed & Variance 검증

- **방법**: 최종 설정으로 seed 3–5개 반복
- **보고**: mean ± std, worst-case seed 결과도 포함
- **목적**: "이 결과가 seed에 의존하지 않음" 증명

---

## Phase 5: Paper Writing (3–4주)

> **목적**: 결과를 ICLR 논문으로 구조화

### Step 5.1 — Story Crystallization

```
Narrative arc:
1. "FM features are assumed robust → evidence they are not"
2. "We diagnose WHY: entanglement of nuisance and anomaly signals"
3. "Principled disentanglement resolves this"
4. "Clean performance preserved, robustness significantly improved"
```

### Step 5.2 — Figure & Table 우선 작성

| 위치 | 내용 | 소스 |
|------|------|------|
| Fig 1 | Entanglement 시각화 (shifted-normal ↔ anomaly 겹침) | Step 2.1, 2.3 |
| Fig 2 | Layer별 robustness-sensitivity trade-off | Step 2.2 |
| Fig 3 | 방법론 overview | Step 3 설계 |
| Fig 4 | Disentanglement 전/후 feature 분포 | Step 4.3 |
| Tab 1 | Main results (3 methods × 3 benchmarks × clean/shifted) | Step 4.1 |
| Tab 2 | Ablation study | Step 4.2 |
| Tab 3 | Per-shift-type breakdown | Step 4.3 |

### Step 5.3 — 초안 작성 → 피드백 → 수정

---

## 제약

- GPU: 사용 가능 자원에 따라 병렬 실험 수 결정
- 데이터: MVTec AD 2 다운로드 필요 (진행 중)
- 시간: ~10주 (Phase 1–4) + 3–4주 (Phase 5)

## 바꾸지 않을 것

- 연구 원칙: novelty + 메커니즘 설명 > 단순 성능
- 벤치마크: MVTec AD + MVTec AD 2 + RobustAD (3종 고정)
- Baseline: Dinomaly, AnomalyDINO, AnomalyCLIP (3종 고정)
- 평가 메트릭: I-AUROC, P-AUROC, AU-PRO (고정)

## 성공 기준

1. **Phase 2 완료 시점**: "왜 깨지는지"에 대한 정량적 + 시각적 증거 3개 이상
2. **Phase 4 완료 시점**: shifted 조건에서 baseline 대비 5pp+ 개선, clean 유지
3. **최종**: ICLR contribution으로 (a) 체계적 failure 분석, (b) 메커니즘 진단, (c) principled fix 3개 모두 포함

## Risk Mitigation

| 리스크 | 확률 | 대응 |
|--------|------|------|
| FM-AD가 예상보다 robust | 중 | category scaling / harder shift 조건으로 pivot |
| Linear disentanglement 불충분 | 중 | Nonlinear (3B) 또는 Multi-layer (3C) 대안 |
| Clean 성능 하락 | 높 | subspace dimension tuning, residual connection |
| 기존 방법과 차별화 부족 | 낮 | 이론적 근거(identifiability) + 분석 depth로 차별화 |
| 실험 시간 부족 | 중 | Phase 2 분석만으로도 workshop급 contribution 가능 |
