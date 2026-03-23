---
name: lessons
description: tasks/lessons.md 조회, 추가, 승격 관리
allowed-tools:
  - Read
  - Write
  - Glob
---

# /lessons

**사용법**: `/lessons [add <title> | promote <topic> | (빈 인자)]`

현재 교훈 목록:

```
!cat tasks/lessons.md 2>/dev/null || echo "(tasks/lessons.md 없음)"!
```

## 동작

### 인자 없음 → 조회
- `tasks/lessons.md` 내용을 읽고 요약 출력
- 등록된 교훈 수, 최근 추가된 항목 하이라이트

### `add <title>` → 교훈 추가
1. `tasks/lessons.md`에 새 교훈 항목 추가:
   ```markdown
   ### [오늘 날짜] <title>
   발생 상황: ...
   잘못한 것: ...
   올바른 방법: ...
   ```
2. 사용자에게 각 필드 내용을 물어보거나, 컨텍스트에서 자동 추론

### `promote <topic>` → 검증된 교훈 승격
1. `tasks/lessons.md`에서 해당 topic 관련 교훈 식별
2. `skill_graph/analysis/<topic>/_lessons.md` 파일 생성 또는 업데이트
   - `_LESSONS_TEMPLATE.md` 참조
3. 승격 완료 후 원본 교훈에 `(→ 승격됨)` 표시

## 규칙

- 교훈은 **구체적**으로 기록 — "주의하자" 같은 모호한 표현 금지
- 승격은 **2회 이상 반복 검증된 패턴**만 대상
- 기존 항목과 중복되면 기존 항목을 보강
