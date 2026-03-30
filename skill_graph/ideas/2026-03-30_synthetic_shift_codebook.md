# Synthetic Shift Codebook + Cross-patch Consistency AD — 2026-03-30

> **상태**: 🔴 제안 (핵심 가정 검증 중)
> **keywords**: synthetic shift, codebook, MoE, cross-patch consistency, task arithmetic, shift-agnostic

---

## 1. 동기

### 1.1 현재까지의 결론

- **Paired NSP**: +29.4pp. Shift vector를 paired data로 추출 → feature space에서 linear projection으로 제거. Task arithmetic과 동일 원리.
- **Unpaired 불가**: 5가지 독립 시도 전멸. Feature-level에서 shift direction을 추정하려면 shift 관찰(paired data) 필수.
- **Scoring-level**: Patch max + q25_sub = +22pp. "Anomaly=local, shift=global" 원리. 하지만 기존 기법 조합.

### 1.2 핵심 딜레마

```
NSP: shift를 관찰해야 제거 가능 → paired data 필수, shift type 사전 지식 필수
Scoring: shift를 모르더라도 +22pp → 하지만 기존 기법 조합, novelty 부족
```

### 1.3 이 아이디어의 질문

> "Shift vector를 실제 환경에서 관찰하지 않고, **synthetic augmentation으로 shift space의 basis를 미리 구축**해두면,
> test 시 shift type을 모르더라도 자동으로 보정할 수 있지 않은가?"

---

## 2. 방법론 제안

### 2.1 핵심 원리

**Task Arithmetic 확장**: Shift vector가 feature space에서 선형이라는 것은 이미 실증 (14 nonlinear 변형 전멸). 따라서:

```
실제 shift = Σ_k α_k × d_k    (shift basis vectors의 선형 조합)

만약 synthetic augmentation으로 {d_1, ..., d_K}를 구축해두면,
실제 shift가 이 basis의 span 안에 있는 한 → 자동 보정 가능
```

**Cross-patch Consistency**: Shift는 global (모든 patch에 일관적), anomaly는 local (특정 patch에 국한). Codebook matching의 patch 간 일관성이 shift와 anomaly를 자연적으로 구분.

### 2.2 알고리즘

```
[Offline: Codebook 구축]
  1. Multi-class train images 수집 (few-shot K장 × C categories)
  2. 각 이미지에 다양한 augmentation 적용:
     {brightness±, contrast±, blur, noise, color_jitter, perspective, ...}
  3. Per-patch shift vector 추출:
     d_{img,aug,patch} = f_L8(patch_augmented) - f_L8(patch_original)
  4. Shift vectors를 codebook으로 구조화:
     - Clustering (k-means) → K개의 prototype shift directions
     - 또는 PCA → top-K principal shift directions
     - 또는 MoE의 expert vectors로 학습

[Test-time: Per-image Inference]
  Step 1: 각 patch의 deviation 계산
    δ_i = f(test_patch_i) - μ_normal_i    (train reference 대비 차이)

  Step 2: 각 patch를 codebook에 매칭
    g_i = softmax(similarity(δ_i, {d_k}))    (어떤 shift type에 가까운가)

  Step 3: Cross-patch consensus → global shift 추정
    g_consensus = aggregate(g_i across all patches)
    → "이 이미지 전체에 걸린 shift의 codebook 가중치"
    → Anomaly patch는 소수 → consensus에 미미한 영향 (1369 patch 중 ~수십)

  Step 4: Consensus 기반 보정
    correction = Σ_k g_consensus_k × d_k    (모든 patch에 동일한 consensus 보정)
    corrected_i = f(patch_i) - correction

  Step 5: Anomaly scoring
    score_i = ||δ_i - correction||    (보정 후 잔차)
    → Consensus와 일치 (shift만): 잔차 작음 → 정상
    → Consensus와 불일치 (anomaly): 잔차 큼 → 이상!
    image_score = max(score_i) or percentile_95(score_i)
```

### 2.3 왜 이것이 이전 실패를 회피하는가

| 이전 실패 | 실패 이유 | 이 방법의 대응 |
|----------|----------|--------------|
| TTNS | Test batch에서 anomaly 오염 | Per-image consensus → anomaly는 소수 patch → 영향 미미 |
| IFR | Train Δ ⊥ shift direction | Augmentation으로 shift 방향을 직접 생성 |
| AdaBN | Global normalization → signal 파괴 | Consensus correction = directional, anomaly 보존 |
| NN Pseudo | PCA aggregation에서 anomaly 재오염 | Patch-level majority = 자연적 robust aggregation |
| SPAD | Feature-space spatial non-uniformity (2.6%) | Codebook은 shift TYPE 매칭 (magnitude 비균일해도 type은 일관) |

### 2.4 핵심 차별점: Cross-patch Consistency

기존 접근은 **모든 deviation을 동일하게 처리** (전부 nuisance로 가정하거나, 전부 anomaly candidate로 처리).

이 방법은 **deviation의 cross-patch 일관성으로 shift와 anomaly를 구분**:
- 일관된 deviation (많은 patch에서 동일 codebook entry) → shift → 보정
- 비일관적 deviation (소수 patch에서만 특이한 entry) → anomaly → 잔차로 탐지

---

## 3. 성립 조건

### 3.1 핵심 가정 (make-or-break)

> **"Synthetic augmentation shift vectors가 span하는 subspace ⊃ real shift subspace"**

검증 방법: Synthetic shift vectors의 PCA basis vs Oracle(MVTec AD 2 paired) shift vectors의 PCA basis 사이의 principal angle + explained variance.

### 3.2 우려: DINOv2가 augmentation에 이미 invariant

DINOv2는 self-distillation 학습에서 color jitter, blur 등에 invariant하도록 학습. 따라서 synthetic augmentation으로 만든 shift vector가 **너무 작을** 수 있음.

하지만:
- DINOv2의 invariance는 **학습 시 사용된 augmentation 강도 범위** 내에서만 완벽
- 강도를 높이면 (예: brightness ×2.0) invariance 범위를 벗어남 → shift vector가 생김
- Real shift (공장 조명 교체 등)도 학습 augmentation 범위 밖일 수 있음

### 3.3 추가 가정

- Shift space의 basis가 유한하고 저차원 (물리적 basis: RGB, spatial, frequency 등)
- Codebook이 이 basis를 커버할 만큼 다양한 augmentation 포함
- Cross-patch consensus가 anomaly에 robust (anomaly ratio < 5% at patch level)

---

## 4. 관련 연구 연결

| 연결 | 내용 |
|------|------|
| **Task Arithmetic** (Ilharco, ICLR 2023) | Shift vector = task vector의 feature-space 버전. Linear composition 원리 공유. |
| **PISCO** (ICML 2023) | Post-hoc linear decomposition이 pretrained feature에서 유효. |
| **PatchCore** (CVPR 2022) | Patch-level kNN scoring. 이 방법에 codebook correction을 추가하는 형태. |
| **MoE for Domain Adaptation** | Expert selection으로 domain-specific processing. 하지만 AD에 적용한 연구 없음. |
| **Concept Erasure** (Belrose, 2023) | Feature space에서 특정 concept을 linear하게 제거. NSP와 동일 원리. |

---

## 5. 검증 실험 계획

### 5.1 핵심 진단: Synthetic ↔ Real Subspace Alignment

```python
# Synthetic shift vectors: train images × augmentations
# Real shift vectors: MVTec AD 2 paired (oracle)
# 비교: principal angle, explained variance
```

**GO 기준**: Real shift variance의 >40%를 synthetic subspace가 설명 → GO

### 5.2 후속 실험 (진단 통과 시)

1. Codebook 구축 (k-means or PCA on synthetic shifts)
2. Cross-patch consensus AD 전체 파이프라인
3. Ablation: codebook size, augmentation types, consensus method

---

## 관련 노트

- 선행: [Phase 3 NSP](../experiments/2026-03-26_phase3_linear_disentanglement/report.md)
- 선행: [5가지 unpaired 실패](../experiments/2026-03-28_nn_pseudopair/report.md)
- 선행: [SPAD 진단](../experiments/) — spatial non-uniformity 2.6%
- 이론: [Task vector 연결](../../.claude/projects/-home-hun-Volume-RESEARCH-Pilot/memory/project_task_vector_connection.md)
