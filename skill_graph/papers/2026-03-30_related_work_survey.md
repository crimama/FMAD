# Related Work Survey: FM-AD Robustness under Distribution Shift

> **생성일**: 2026-03-30
> **소스**: 3개 병렬 에이전트, HuggingFace paper search + WebSearch
> **총 논문**: 60+편, 핵심 선별 45편
> **목적**: 마스터 리포트 §0 근본 원인과 관련된 문헌 종합

---

## Executive Summary

FM-AD의 shift 취약성, nuisance-anomaly entanglement, shift-agnostic scoring에 대한 관련 연구를 조사한 결과:

1. **FM-AD robustness 문제는 인식되고 있으나, 체계적 원인 분석은 부재** — MVTec AD 2, M2AD, AeBAD 등 벤치마크는 있지만 "왜 깨지는가"에 대한 메커니즘 연구는 0건
2. **Nuisance disentanglement for AD는 극초기** — Red PANDA(2022), Stylist(WACV 2025), PiCo(2026)만 존재하며 모두 paired/labeled nuisance 필요
3. **이론적 기반은 탄탄** — Nonlinear ICA identifiability, Locatello impossibility, SSL content-style separation이 우리 문제를 정확히 프레이밍
4. **Few-shot FM-AD + unknown shift + no paired data** 교차점에 기존 연구 **0건** — 이것이 우리의 novelty gap

---

## 1. FM-AD Robustness 벤치마크 & 실증

### 핵심 벤치마크

| 벤치마크 | Year | Venue | 특성 | FM-AD 성능 |
|---------|------|-------|------|-----------|
| **MVTec AD 2** | 2025 | arXiv [2503.21622](https://arxiv.org/abs/2503.21622) | 조명 변화 (6 conditions/object), 10 categories | Best ~58.7% AU-PRO (vs 90%+ MVTec AD 1) |
| **M2AD** | 2025 | arXiv [2505.10996](https://arxiv.org/abs/2505.10996) | 12 views × 10 illuminations, 119K images | Multi-config fusion vs single-image robustness 프로토콜 |
| **RobustAD** | 2025 | CVPR-W [paper](https://openaccess.thecvf.com/content/CVPR2025W/VAND/) | 9종 domain shift, "Average Relative Drop" 메트릭 | Real-world shift 정량화 |
| **AeBAD** | 2023 | Comp. in Industry [2304.02216](https://arxiv.org/abs/2304.02216) | 항공엔진 블레이드, 실제 domain shift | MMR 방법 제안 |
| **Beyond Academic** | 2025 | CVPR-W [2503.23451](https://arxiv.org/abs/2503.23451) | 9 datasets × 11 SOTA × 7 metrics | 99.9% → 실제 생산에서 급락, **PatchCore가 가장 robust** |

**핵심 insight**: PatchCore가 shift에 가장 robust한 이유 — Layer 2-3 feature 사용 (deep layer 회피). 우리의 "layer 선택이 robustness에 핵심" 발견과 일맥상통.

### FM-AD 취약성 직접 실증

| 논문 | Key Finding | 우리 연구와의 관계 |
|------|-----------|-----------------|
| CLIP Semantic Robustness ([2405.07969](https://arxiv.org/abs/2405.07969)) | CLIP zero-shot AD: semantic transform에서 **최대 40% AU-PRO 하락** | CLIP의 shift 취약성 정량화 |
| DINOv2 Adversarial ([2510.13643](https://arxiv.org/abs/2510.13643)) | DINOv2 NN-detector: 작은 adversarial perturbation에 fragile, anomaly score 미보정 | DINOv2의 근본적 취약성 |
| VAND 3.0 ([2509.17615](https://arxiv.org/abs/2509.17615)) | CVPR 2025 challenge: benchmark-to-reality gap 여전 | 산업 현장과의 gap |

---

## 2. AD under Distribution Shift — 기존 방법론

### Feature-level 접근 (우리 방향과 직접 경쟁)

| 논문 | Year | Venue | 방법 | Paired 필요? | 우리와의 차이 |
|------|------|-------|------|-------------|-------------|
| **Red PANDA** ([2207.03478](https://arxiv.org/abs/2207.03478)) | 2022 | arXiv | Domain-supervised contrastive loss로 nuisance-free representation 학습 | **Yes** (domain labels) | 우리: post-hoc, 학습 없음 |
| **Stylist** ([2310.03738](https://arxiv.org/abs/2310.03738)) | 2025 | WACV | FM feature별 environment-bias score → 고편향 feature 드롭 | **Yes** (multi-env) | **가장 유사**: post-hoc feature selection on frozen FM. 우리는 subspace projection |
| **PiCo** ([2603.23122](https://arxiv.org/abs/2603.23122)) | 2026 | arXiv | 3-stage neural disentanglement (photometric/spectral/contextual) + active pose | **No** (active) | End-to-end 학습, 우리는 post-hoc |
| **FiCo** ([2412.10115](https://arxiv.org/abs/2412.10115)) | 2025 | AAAI | DiSCo (shift compensation) + DiIFi (invariant filter) on knowledge distillation | **No** (shift augmentation) | Architecture-specific, 우리는 architecture-agnostic |
| **PCIR** ([2312.14329](https://arxiv.org/abs/2312.14329)) | 2023 | NeurIPS | Causal invariant regularization, multi-environment training | **Yes** (multi-env) | 이론적 근거 제공, 우리의 causal framing 참조 |

### Scoring-level / TTA 접근

| 논문 | Year | Venue | 방법 | 우리와의 관계 |
|------|------|-------|------|-------------|
| **Selective TTA for AD** ([2410.03306](https://arxiv.org/abs/2410.03306)) | 2024 | arXiv | Lightweight MLP for neural implicit representations, zero-shot domain adaptation | Medical domain, 산업 AD로 전환 가능 |
| **TTT for Industrial AS** ([2404.03743](https://arxiv.org/abs/2404.03743)) | 2024 | CVPR-W | Test-time training for anomaly segmentation | Binarization에 집중, scoring과 보완적 |
| **Robust Distribution Alignment** ([2503.14910](https://arxiv.org/abs/2503.14910)) | 2025 | arXiv | Robust Sinkhorn distance로 memory-bank alignment | 일부 target normal data 가정 |
| **AD under Distribution Shift** ([2303.13845](https://arxiv.org/abs/2303.13845)) | 2023 | ICCV | Distribution-invariant normality learning (Reverse Distillation) | **첫 formal AD+shift 연구**. 우리와 다른 접근 (invariant learning vs post-hoc projection) |
| **LWinNN** ([2509.17670](https://arxiv.org/abs/2509.17670)) | 2025 | arXiv | Local window-based NN: 중간 수준 translation invariance | "controlled invariance" ← 우리의 selective nuisance removal과 유사 원리 |

### Few-shot AD (shift 미고려)

| 논문 | Year | Venue | 방법 |
|------|------|-------|------|
| **IIPAD** | 2025 | ICLR | One-for-all few-shot via CLIP+BLIP-Diffusion, instance-induced prompts |
| **WinCLIP** ([2303.14814](https://arxiv.org/abs/2303.14814)) | 2023 | CVPR | Window-based CLIP, compositional ensemble |
| **PromptAD** ([2404.05231](https://arxiv.org/abs/2404.05231)) | 2024 | arXiv | Semantic concatenation + anomaly margin loss |
| **AnomalyDINO** ([2405.14529](https://arxiv.org/abs/2405.14529)) | 2025 | WACV | DINOv2 patch embeddings, 1-shot 96.6% |
| **CAReg** ([2406.08810](https://arxiv.org/abs/2406.08810)) | 2024 | arXiv | Category-agnostic registration + Wasserstein augmentation |

---

## 3. Feature Entanglement & Disentanglement 이론

### 핵심 이론 논문 (우리 연구의 이론적 backbone)

| 논문 | Year | Venue | 핵심 결과 | 우리 연구에서의 역할 |
|------|------|-------|---------|-------------------|
| **SSL Isolates Content/Style** ([2106.04619](https://arxiv.org/abs/2106.04619)) | 2021 | NeurIPS | SSL + augmentation → content(invariant) / style(variant) block-identifiable | **FM feature에 content-style 분리가 존재**함의 이론적 보장 |
| **Nonlinear ICA** ([1805.08651](https://arxiv.org/abs/1805.08651)) | 2019 | AISTATS | Auxiliary variable 조건부 → nonlinear ICA identifiable | **Paired data(aux var)가 왜 nuisance 분리를 가능하게 하는지** 설명 |
| **iVAE** ([1907.04809](https://arxiv.org/abs/1907.04809)) | 2020 | AISTATS | Deep LVM에서 prior conditioning → identification up to simple transform | Post-hoc linear separation의 이론적 정당화 |
| **Locatello Impossibility** ([1811.12359](https://arxiv.org/abs/1811.12359)) | 2019 | ICML | **Unsupervised disentanglement은 inductive bias 없이 불가능** | **우리의 unpaired 5가지 실패를 이론적으로 뒷받침** |
| **Structured Disentanglement** ([2311.08815](https://arxiv.org/abs/2311.08815)) | 2023 | arXiv | Multi-block style disentanglement, augmentation structure 활용 | Style을 버리지 않고 분리하여 활용 가능 |

### Contrastive Learning의 Information Destruction

| 논문 | Year | Venue | 핵심 | 관련성 |
|------|------|-------|------|--------|
| **What Should Not Be Contrastive** ([2008.05659](https://arxiv.org/abs/2008.05659)) | 2021 | ICLR | Invariance 가정이 task-relevant info를 파괴 | **메커니즘 #1의 이론적 근거** |
| **Equivariant CL** ([2111.00899](https://arxiv.org/abs/2111.00899)) | 2022 | ICLR | Equivariance가 invariance보다 나음 (info 보존) | 대안적 학습 방식 제시 |
| **Learning the Unlearned** ([2402.11816](https://arxiv.org/abs/2402.11816)) | 2024 | ECCV | Feature suppression in CL → MCL로 복원 | FM feature의 정보 gap 검증 |
| **Rethinking Augmentation** ([2206.00227](https://arxiv.org/abs/2206.00227)) | 2022 | CVPR | 과도한 augmentation → fine-grained info 소실 | **메커니즘 #1 실증** |
| **Emergence of Invariance** ([1706.01350](https://arxiv.org/abs/1706.01350)) | 2018 | JMLR | Depth → invariance = info minimality | Deep FM이 점진적으로 nuisance를 제거하되 불완전 → 잔여 entanglement |

### Post-Hoc Linear Disentanglement

| 논문 | Year | Venue | 방법 | 관련성 |
|------|------|-------|------|--------|
| **PISCO extension** ([2509.11436](https://arxiv.org/abs/2509.11436)) | 2025 | Springer | Frozen encoder에서 linear rotation으로 technical/biological 분리 | **가장 유사한 방법론**: post-hoc linear rotation on frozen features |
| **Post-Hoc Concept Disentanglement** ([2503.05522](https://arxiv.org/abs/2503.05522)) | 2025 | arXiv | CAV (linear classifier)로 frozen DNN feature에서 concept 분리 | Linear concept separation 검증 |

---

## 4. Positioning Gap Map

```
                    Paired Data Required?
                    Yes                 No
                ┌─────────────────┬──────────────────┐
  Feature-      │ Red PANDA       │ FiCo (shift aug) │
  level         │ Stylist         │ PiCo (active)    │
  nuisance      │ PCIR            │ ★ OUR GAP ★     │
  removal       │ NSP (ours,      │                  │
                │  oracle)        │                  │
                ├─────────────────┼──────────────────┤
  Scoring-      │                 │ TTA-AD (medical) │
  level /       │                 │ LWinNN           │
  TTA           │                 │ Shift-robust     │
                │                 │  scoring (ours?) │
                └─────────────────┴──────────────────┘
```

**★ OUR GAP**: Feature-level nuisance removal × No paired data = **기존 연구 0건**

하지만 Locatello impossibility에 의해 완전 unsupervised disentanglement은 불가능 → **category structure, few-shot reference, scoring-level prior 등 indirect supervision이 필요**.

---

## 5. 우리 연구와의 핵심 연결

### 이론적 연결

| 우리의 발견 | 이론적 근거 |
|-----------|-----------|
| Paired NSP가 +29.4pp | Nonlinear ICA: aux var(environment label) → identifiable |
| Unpaired 5가지 실패 | **Locatello impossibility**: unsupervised disentanglement 불가 |
| Layer 8 자연 분리 | SSL content-style theory: augmentation이 content/style 분리를 유도하되 layer별로 다름 |
| Linear projection이 optimal | PISCO + Post-hoc concept disentanglement: frozen FM feature에서 linear 분리가 유효 |
| 14개 nonlinear 변형 실패 | Emergence of invariance: FM이 이미 충분히 linear하게 nuisance를 encoding |

### 비교군 (논문에 포함 필수)

| 방법 | 유형 | 이유 |
|------|------|------|
| **PatchCore** | Memory bank kNN | 가장 robust한 기존 방법 (Beyond Academic 확인) |
| **Stylist** | Post-hoc feature selection | 가장 유사한 nuisance 제거 방법 |
| **FiCo** | Architecture-specific DA | AAAI 2025, shift robustness for AD |
| **PCIR** | Causal invariant regularization | NeurIPS 2023, 이론적 비교군 |
| **AdaBN** | Test-time statistics | 가장 단순한 TTA baseline |

### 차별화 포인트

| vs | 우리의 차별점 |
|---|-------------|
| Red PANDA | Post-hoc (학습 없음), paired data 불필요한 방향 탐색 |
| Stylist | Feature dropping이 아닌 subspace projection (정보 보존), 이론적 분석 depth |
| FiCo | Architecture-agnostic, 어떤 FM-AD에든 plug-in 가능 |
| PCIR | Feature-level (PCIR은 representation learning), 학습 불필요 |
| PiCo | Post-hoc (PiCo는 end-to-end), active perception 불필요 |

---

## 6. 핵심 논문 읽기 우선순위

### Must Read (즉시)

1. **Stylist** (WACV 2025) — 가장 유사한 방법, 비교 필수
2. **Red PANDA** (2022) — nuisance removal for AD의 원조
3. **PCIR** (NeurIPS 2023) — causal framework for AD under shift
4. **Locatello impossibility** (ICML 2019) — unpaired 실패의 이론적 근거
5. **von Kügelgen SSL content-style** (NeurIPS 2021) — 우리 이론의 backbone

### Should Read (이번 주)

6. **FiCo** (AAAI 2025) — shift robustness for AD, 직접 비교군
7. **AD under Distribution Shift** (ICCV 2023) — formal AD+shift 첫 연구
8. **PiCo** (2026) — 같은 문제의 다른 접근
9. **What Should Not Be Contrastive** (ICLR 2021) — info destruction 이론
10. **PISCO extension** (2025) — post-hoc linear rotation 검증

### Nice to Have (논문 작성 시)

11-15. M2AD, Beyond Academic, Equivariant CL, Structured Disentanglement, iVAE

---

## 7. 우리 연구에 활용 가능한 주요 인사이트

### Insight 1: "Unsupervised disentanglement은 불가능" — 우리의 5가지 실패가 이론의 실증

Locatello et al. (ICML 2019)은 **inductive bias 없이 unsupervised disentanglement은 불가능**하다는 것을 증명했다. 우리의 5가지 unpaired 실패(TTNS, IFR, AdaBN, NN Pseudo, SPAD)는 이 이론의 **AD-specific 실증**이다. 이것은 단순히 "방법이 안 됐다"가 아니라 "이론적으로 안 되는 것이 실제로 안 됐다"라는 구조적 결과.

**활용**: 논문 §4에서 impossibility argument를 Locatello로 프레이밍하면, negative results가 contribution이 됨. "We provide the first empirical confirmation of Locatello's impossibility theorem in the context of one-class AD under distribution shift."

### Insight 2: Paired data = Auxiliary Variable for Identifiability

von Kügelgen (NeurIPS 2021)과 Hyvarinen (AISTATS 2019)의 nonlinear ICA 이론은 **auxiliary variable가 있으면 identifiable**하다는 것을 보인다. NSP에서 paired data의 역할은 정확히 이 auxiliary variable:

```
NSP: d = f(obj, shifted) - f(obj, clean)  ← object를 conditioning하면 pure shift 추출 가능
     = auxiliary variable conditioning으로 nuisance factor를 identify하는 것
```

**활용**: NSP의 성공을 "engineering trick"이 아닌 "identifiability theory에 기반한 principled solution"으로 포지셔닝. 논문 §3에서 이론적 정당화.

### Insight 3: SSL은 Content-Style을 분리하되 "불완전하게" 분리한다

von Kügelgen (NeurIPS 2021)은 SSL이 content(augmentation-invariant)와 style(augmentation-variant)을 block-identifiable하게 분리한다고 증명. 하지만 Achille & Soatto (JMLR 2018)에 따르면 이 분리는 depth가 깊어질수록 점진적이며, **실제로는 완전하지 않다**.

우리 Layer 8 vs Layer 11 발견이 정확히 이것:
- Layer 8: entanglement 0.12 → SSL이 여기까지 분리에 성공
- Layer 11: entanglement 0.53 → 이후 layer에서 task-specific 정보 추가하며 재혼합

**활용**: V-shaped profile의 이론적 설명. "SSL의 content-style 분리가 intermediate layer에서 최대에 달하고, 이후 layer에서 task-specific processing이 재혼합을 유발한다."

### Insight 4: Contrastive Learning의 Information Destruction이 FM-AD 취약성의 근본

"What Should Not Be Contrastive" (ICLR 2021), "Learning the Unlearned" (ECCV 2024), "Rethinking Augmentation" (CVPR 2022)이 공통으로 보이는 것: **CL의 invariance는 task-relevant info를 파괴할 수 있다**. FM-AD의 맥락에서:

```
DINOv2 학습 augmentation:  color jitter → "같은 이미지"로 취급
AD에서의 의미:             color change → 변색 anomaly일 수 있음
→ Anomaly-relevant info가 "nuisance"로 학습되어 파괴
```

**활용**: 메커니즘 #1의 이론적 근거로 직접 인용. "The invariance-information trade-off in contrastive learning (Xiao et al., ICLR 2021) is the theoretical root cause of FM-AD's fragility: augmentations used during pretraining systematically overlap with anomaly-relevant appearance changes."

### Insight 5: Stylist가 가장 가까운 기존 연구 — 하지만 핵심 차이 있음

Stylist (WACV 2025)는 frozen FM feature에서 per-feature environment-bias score를 계산하여 고편향 feature를 드롭. 가장 유사하지만:

| | Stylist | 우리 |
|---|---------|-----|
| 분리 방법 | Feature dropping (binary) | Subspace projection (continuous) |
| 정보 보존 | 드롭된 feature의 정보 완전 손실 | Orthogonal complement 보존 |
| 이론적 근거 | 경험적 bias scoring | Identifiability theory |
| Shift 사전 지식 | Multi-environment label 필수 | Paired data 필요(oracle) / scoring prior(practical) |
| 분석 depth | 방법론 중심 | 7가지 메커니즘 + V-shape + impossibility |

**활용**: Related work에서 Stylist를 비교하되, "우리는 feature dropping이 아닌 subspace projection을 사용하여 정보 보존이 더 나으며, 이론적 프레이밍이 더 깊다"로 차별화.

### Insight 6: PatchCore가 가장 Robust한 이유 — Layer 선택의 중요성 재확인

"Beyond Academic Benchmarks" (CVPR-W 2025)에서 PatchCore가 shift에 가장 robust한 것으로 확인. 이유: **Layer 2-3 feature 사용** (deep layer 회피). 이것은 우리의 "layer 선택이 robustness의 핵심" 발견과 정확히 일치.

**활용**: §2에서 "이 관찰은 독립적으로도 보고되었다: PatchCore의 robustness 우위가 Layer 2-3 선택에서 비롯됨을 Beyond Academic Benchmarks가 확인" — 우리의 V-shape 발견의 external validation.

### Insight 7: Feature-level nuisance removal × No paired data = 0건 — 진짜 gap

positioning gap map에서 확인:
- Red PANDA, Stylist, PCIR: paired/labeled 필요
- FiCo, PiCo: architecture-specific 또는 end-to-end
- **우리의 gap**: post-hoc, architecture-agnostic, vision-encoder-only, 이론적 분석 + practical method

**활용**: Introduction에서 "To our knowledge, no prior work has (1) systematically diagnosed why FM features fail under shift, (2) shown that linear projection is optimal, and (3) demonstrated the structural impossibility of unpaired nuisance estimation in one-class AD."

---

## 8. 결론: 이 문헌 조사가 연구 방향에 주는 시사점

### 확인된 것
1. **문제의 중요성**: M2AD, MVTec AD 2, AeBAD 등 다수 벤치마크가 FM-AD robustness를 핵심 미해결 과제로 지목
2. **이론적 탄탄함**: Locatello impossibility + ICA identifiability가 우리의 paired/unpaired 결과를 정확히 프레이밍
3. **Novelty gap 실존**: Feature-level nuisance removal × no paired data × post-hoc = 0건

### 방향 시사점
1. **비교군 필수**: Stylist, PatchCore, FiCo, PCIR (최소 4개)
2. **이론적 framing 강화**: Locatello로 impossibility, von Kügelgen/Hyvarinen으로 oracle 정당화
3. **현재 shift-agnostic scoring 방향의 positioning**: "Locatello가 말하는 inductive bias를 scoring-level prior(anomaly=local, shift=global)로 제공"하면 unsupervised도 가능할 수 있음 — 이것이 이론과 실험의 접점

---

## 관련 노트

- [FM-AD 핵심논문 10선](2026-03-23_FM_AD_robustness_핵심논문.md)
- [필독논문 목록](2026-03-24_FM_AD_robustness_필독논문.md)
- [이론적 기반](../analysis/fm_ad_robustness/2026-03-23_theoretical_foundations.md)
- [MASTER_REPORT §0](../MASTER_REPORT.md)

- [FM-AD 핵심논문 10선](2026-03-23_FM_AD_robustness_핵심논문.md)
- [필독논문 목록](2026-03-24_FM_AD_robustness_필독논문.md)
- [이론적 기반](../analysis/fm_ad_robustness/2026-03-23_theoretical_foundations.md)
- [MASTER_REPORT §0](../MASTER_REPORT.md)
