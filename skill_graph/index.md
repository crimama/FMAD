# Skill Graph Index

> 프로젝트 연구 기록의 키워드 기반 인덱스.
> 키워드 → 문서 링크로 그래프 탐색 가능.
> 새 문서 추가 시 이 파일도 함께 갱신할 것.

---

## 키워드 그래프

```
<!-- ASCII 다이어그램으로 키워드 간 관계를 시각화 -->
<!-- 예시:
  ┌──────────┐      ┌────────────┐
  │ keyword1 │◀────▶│  keyword2  │
  └─────┬────┘      └─────┬──────┘
        │                 │
   ┌────▼────┐     ┌─────▼──────┐
   │ keyword3│     │  keyword4  │
   └─────────┘     └────────────┘
-->
```

---

## 키워드 → 문서 매핑

| 키워드 | 문서 | 카테고리 |
|--------|------|----------|
| <!-- **keyword** | [문서 제목](category/YYYY-MM-DD_name.md) | category --> |

---

## 문서 → 키워드 역매핑

### experiments/

| 문서 | 상태 | 키워드 |
|------|------|--------|
<!-- | [제목](experiments/YYYY-MM-DD_name.md) | 🟢 | `kw1` `kw2` | -->

### papers/
| 문서 | 구분 | 키워드 |
|------|------|--------|
<!-- | [논문명](papers/YYYY-MM-DD_name.md) | competitor | `kw1` `kw2` | -->

### ideas/
| 문서 | 상태 | 키워드 |
|------|------|--------|
<!-- | [아이디어명](ideas/YYYY-MM-DD_name.md) | screened | `kw1` `kw2` | -->

### bugfix/
_(아직 없음)_

---

## 문서 간 연결 (관련 노트)

```
<!-- 문서 간 선행/후속 관계를 흐름도로 표현 -->
<!-- 예시:
experiments/YYYY-MM-DD_hypothesis.md
        │
        ▼ (결과)
experiments/YYYY-MM-DD_result.md

papers/YYYY-MM-DD_related_work.md
        │
        ▼ (inspires)
ideas/YYYY-MM-DD_method.md
        │
        ▼ (tested by)
experiments/YYYY-MM-DD_ablation/
-->
```

---

## 타임라인

| 날짜 | 문서 | 요약 |
|------|------|------|
<!-- | YYYY-MM-DD | [제목](path) | 한 줄 요약 | -->
