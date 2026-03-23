---
name: analyze
description: 분석 노트 생성 + 교훈 승격 관리
allowed-tools:
  - Read
  - Write
  - Glob
---

# /analyze

**사용법**: `/analyze <topic> [promote]`

기존 분석 노트:

```
!find skill_graph/analysis -name '*.md' ! -name '_LESSONS_TEMPLATE.md' 2>/dev/null | sort!
```

## 동작

### `<topic>` → 분석 노트 생성
1. 해당 topic과 관련된 실험 노트 검색:
   - `skill_graph/experiments/` 내 관련 파일 식별
2. `skill_graph/analysis/<topic>/` 디렉토리 생성
3. 관련 실험들의 결과를 종합한 분석 노트 작성:
   - 경로: `skill_graph/analysis/<topic>/YYYY-MM-DD_<topic>.md`
   - 패턴, 트렌드, 공통 교훈 추출
4. 관련 실험 노트의 `## 관련 노트` 섹션에 분석 노트 링크 추가

### `<topic> promote` → 교훈 승격
1. `tasks/lessons.md`에서 해당 topic 관련 교훈 식별
2. `skill_graph/analysis/<topic>/_lessons.md` 생성 또는 업데이트:
   - `_LESSONS_TEMPLATE.md` 형식 적용
   - 실험 출처, 검증 횟수, 적용 조건 명시
3. 원본 `tasks/lessons.md`에 `(→ 승격됨: analysis/<topic>/_lessons.md)` 표시
4. 승격 결과 요약 출력

## 규칙

- 분석은 **최소 2개 이상의 실험 결과**를 종합해야 함
- 승격은 반복 검증된 패턴만 대상 — 단일 실험 결과는 승격 불가
- 기존 `_lessons.md`가 있으면 새로 만들지 말고 업데이트
- 분석 노트와 실험 노트 간 양방향 링크 유지
