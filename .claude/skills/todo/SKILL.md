---
name: todo
description: tasks/todo.md 조회, 계획 작성, 체크, 완료 관리
allowed-tools:
  - Read
  - Write
---

# /todo

**사용법**: `/todo [plan <desc> | check <n> | done | (빈 인자)]`

현재 상태:

```
!cat tasks/todo.md 2>/dev/null || echo "(tasks/todo.md 없음)"!
```

## 동작

### 인자 없음 → 조회
- `tasks/todo.md` 현재 내용 출력
- 진행률 요약 (완료/전체 체크박스 수)

### `plan <desc>` → 체크리스트 작성
1. `tasks/todo.md`의 `## 현재 작업` 섹션에 작업 제목 설정
2. `## 계획` 섹션에 단계별 체크리스트(`- [ ]`) 작성
3. 3단계 이상이면 Plan Node Default 규칙에 따라 plan mode 권장 안내

### `check <n>` → n번째 항목 체크
1. 계획 섹션의 n번째 `- [ ]`를 `- [x]`로 변경
2. 진행률 업데이트 출력

### `done` → 작업 완료 처리
1. 완료된 체크리스트를 `## 결과` 섹션으로 이동
2. 결과 요약 작성 (변경 파일, 주요 결정 등)
3. 관련 skill_graph 링크 추가

## 규칙

- 기존 내용을 함부로 덮어쓰지 않음 — 추가/수정만
- `done` 시 미완료 항목이 있으면 경고
- 세션 시작 시 이전 todo 상태 확인 권장
