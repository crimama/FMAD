# FM-AD Robustness — ICLR 논문 포지셔닝 전략

작성일: 2026-03-23
목적: "Foundation Model AD Robustness 한계 분석 + 해결" 주제의 ICLR 포지셔닝

---

## 1. ICLR이 좋아하는 "통념 도전" 논문 패턴

### 성공 사례 분석

| 논문 | Venue | 도전한 통념 | 성공 요인 |
|------|-------|-----------|----------|
| Can LLMs Generate Novel Ideas? | ICLR 2025 | "LLM이 아이디어를 스케일할 수 있다" | 100+ 연구자 참여 대규모 통제 실험 |
| Closer Look at CLIP Robustness | NeurIPS 2023 | "CLIP이 본질적으로 robust하다" | 83 CLIP + 127 classifier, 10 visual factors |
| Language Modeling is Compression | ICLR 2024 | "스케일링이 항상 좋다" | 우아한 reframing (압축=예측), 숨겨진 비용 폭로 |
| Data Determines Distributional Robustness in CLIP | ICML 2022 | "CLIP의 robustness가 학습 방법 때문" | 실제 원인은 데이터 다양성, 방법론 아님 |

### 공통 성공 패턴
1. **널리 받아들여진 belief을 타겟** — 커뮤니티가 당연시하는 것
2. **대규모 체계적 실험** — 한 실험이 아닌 포괄적 커버리지
3. **통제된 실험 설계** — 변수 분리, 집계가 아닌 개별 분석
4. **실행 가능한 insight** — "X가 틀렸다" + "실제로는 Y가 효과를 만든다"

---

## 2. ICLR AD 논문 프레이밍 분석

### AnomalyCLIP (ICLR 2024)
- **프레이밍**: "VLM은 class semantics에 집중, abnormality에는 아님 → object-agnostic prompt learning으로 해결"
- **유형**: 문제 진단(CLIP의 foreground attend 편향) + targeted fix (학습 가능 prompt)
- **규모**: 17 real-world AD datasets

### MMAD (ICLR 2025)
- **프레이밍**: "MLLM의 산업 AD 능력이 체계적으로 연구되지 않았다"
- **유형**: Benchmark + analysis + enhancement
- **핵심 발견**: GPT-4o도 74.9%에 불과 → 산업 요구 수준에 한참 못 미침
- **수락 이유**: 명확한 evaluation gap 채움 + 대규모(39K QA) + 구체적 한계 노출

### IIPAD (ICLR 2025)
- **프레이밍**: "기존 방법은 카테고리별 고정 prompt → instance-specific prompt generator"
- **유형**: 아키텍처 혁신

### 핵심 관찰
**MMAD가 가장 유사한 선례** — FM 한계를 AD에서 노출하는 benchmark 논문. 그러나 MMAD는 **MLLM** (GPT-4o 등) 대상이지 **vision backbone feature** (CLIP, DINOv2) 대상이 아님. → **vision FM feature의 체계적 robustness 분석은 빈 공간**

---

## 3. Gap Analysis: 무엇이 아직 안 되었나

| 기존 연구 | 커버 범위 | 빠진 것 |
|----------|---------|--------|
| Closer Look at CLIP (NeurIPS 2023) | CLIP robustness for classification | AD-specific 아님 |
| DINOv2 adversarial AD (arXiv 2025) | DINOv2 AD adversarial attack | Natural shift 없음 |
| MMAD (ICLR 2025) | MLLM for AD | Vision backbone feature 아님 |
| Beyond Academic Benchmarks (CVPR-W 2025) | 실환경 제약 | Workshop, FM 중심 아님 |
| AnomalyCLIP (ICLR 2024) | Zero-shot CLIP AD | CLIP이 작동한다고 가정, 실패 분석 없음 |

**명확한 gap**: Vision FM feature (CLIP, DINOv2 등)가 AD에서 natural, non-adversarial distribution shift 하에 언제, 왜 실패하는지 **체계적으로 분석하고 원리적 해결책을 제시한 논문 0편**

---

## 4. 포지셔닝 옵션

### Option A: "Benchmark + Diagnosis" (MMAD 스타일)
**제목 패턴**: "How Robust Are Foundation Model-Based Anomaly Detectors? A Systematic Evaluation"
- **기여**: FM-AD 최초의 comprehensive robustness benchmark (CLIP, DINOv2, 조합)
- **강점**: 명확한 gap 채움 (MMAD가 MLLM을 했듯이, 이 논문은 vision backbone)
- **리스크**: 순수 benchmark는 surprising finding 없으면 incremental로 보일 수 있음

### Option B: "Analysis + Principled Fix" ⭐ 추천
**제목 패턴**: "Rethinking Feature Robustness in Foundation Model-Based Anomaly Detection"
- **기여 1**: FM feature가 AD에서 실패하는 when/why의 체계적 분석
- **기여 2**: 근본 원인 분석에서 도출된 원리적, 경량 해결책
- **강점**: "이해" + "해결" 양쪽 reviewer 모두 만족
- **핵심**: Fix는 **단순하고 해석 가능**해야 함 — 복잡한 fix는 analysis narrative를 약화

### Option C: "Challenging the Assumption" (가장 강한 ICLR fit)
**제목 패턴**: "Are Foundation Model Features All You Need for Anomaly Detection?"
- **핵심 claim**: "FM backbone으로 바꾸면 AD robustness가 해결된다는 가정은 조건 X, Y, Z에서 실패하며, 이는 [메커니즘]에 기인한다"
- **직접적으로 널리 퍼진 belief을 타겟** + 체계적 증거
- **Finding이 genuinely surprising하면 가장 강한 positioning**

---

## 5. 추천 논문 구조 (9-page ICLR)

```
1. Introduction (1.5p)
   - "FM-based AD가 robust하다는 통념" 제시
   - "우리는 이것이 [조건]에서 무너진다는 것을 보인다"
   - Contribution 3줄

2. Related Work (0.5p)
   - FM-based AD methods (brief)
   - AD robustness 연구 (gap 강조)
   - FM robustness analysis (non-AD, 우리가 처음)

3. Systematic Robustness Analysis (3p) ← 핵심 novelty
   - 실험 설계: N개 FM-AD methods × M개 shift types × K개 datasets
   - Failure taxonomy: 어떤 method가 어떤 shift에 어떻게 실패하는가
   - 정량적 증거 + variance/confidence interval

4. Root Cause Analysis (1p)
   - Feature space 시각화 (t-SNE, feature norm dist)
   - 메커니즘 검증 (invariance-info destruction, nuisance entanglement)
   - Layer별/scale별 robustness 차이

5. Principled Fix (1.5p)
   - 진단에서 직접 도출된 해결책
   - 단순하고 해석 가능한 방법
   - Implementation details

6. Experiments: Fix Verification (1.5p)
   - Fix가 identified failure를 해결하는지 확인
   - Nominal 성능 유지 확인
   - Ablation: 각 component의 기여

7. Conclusion (0.5p)
```

**분석:해결 비율**: ~60:40 (분석 중심이되 해결책도 제공)

---

## 6. ICLR Reviewer가 중시하는 것

ICLR 2026 Reviewer Guide에서 직접 인용:
> "Originality does not necessarily require introducing an entirely new method; a work that provides novel insights by evaluating existing methods or demonstrates improved understanding is equally valuable."

> "Submissions bring value when they convincingly demonstrate new, relevant, impactful knowledge — this does not necessarily require state-of-the-art results."

→ **분석 논문이 ICLR에서 명시적으로 환영됨**. SOTA 결과 불필요.

---

## 7. Claim 구체화

### 원안 (사용자 제안)
> "FM 기반 AD 방법들은 통제된 벤치마크에서 near-perfect하지만, distribution shift 및 카테고리 스케일링에서 심각하게 열화. 이는 FM feature의 [구체적 메커니즘]에 기인하며, [제안 방법]으로 해결 가능."

### 구체화 제안
> "Foundation model (DINOv2, CLIP) 기반 AD 방법들은 MVTec AD/VisA에서 99%+ AUROC를 달성하지만, 조명/환경 변화에서 최대 40pp (MVTec AD 2 기준), 카테고리 160+ 스케일링에서 10-20pp 열화된다. 이는 FM feature의 **(1) invariance에 의한 anomaly-relevant 정보 소실**과 **(2) nuisance factor와 semantic 정보의 entanglement**에 기인한다. 우리는 [feature space에서 anomaly-relevant subspace를 분리하는 방법 / test-time feature adaptation / ...]으로 이를 해결한다."

---

## 8. 저자의 강점과의 연결

| 강점 | 활용 방향 |
|------|---------|
| ProxyCore (PatchCore distribution shift) | Robustness 분석에 대한 직접 경험 + domain shift 해결 경험 |
| NF 기반 Continual AD | Feature distribution 모델링 전문성 |
| Memory bank 방법론 | PatchCore 계열의 한계와 개선을 깊이 이해 |
| Distribution shift/robustness | 이 주제의 핵심 도메인과 완벽히 일치 |

---

## 관련 노트

- [2026-03-23_failure_survey.md](2026-03-23_failure_survey.md) — 실패 사례 종합
- [2026-03-23_mechanism_analysis.md](2026-03-23_mechanism_analysis.md) — 메커니즘 분석
- [2026-03-23_solution_survey.md](2026-03-23_solution_survey.md) — 해결 방안 서베이
- [../../ideas/2026-03-23_연구방향_후보.md](../../ideas/2026-03-23_연구방향_후보.md) — 6개 방향 후보
