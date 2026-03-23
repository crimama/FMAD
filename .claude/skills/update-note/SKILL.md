---
name: update-note
description: skill_graph/ 템플릿 기반으로 새 노트를 생성합니다
allowed-tools:
  - Read
  - Write
  - Glob
  - Bash
---

# /update-note

**사용법**: `/update-note <category> <short-title>`

사용 가능한 카테고리와 템플릿:

```
!find skill_graph -name '_TEMPLATE.md' -o -name '.gitkeep' | grep -v .gitkeep | sort!
```

## 동작

1. `$ARGUMENTS`에서 `<category>`와 `<short-title>` 파싱
   - 인자 없으면 → 사용 가능한 카테고리 목록 출력 후 종료
   - category만 있으면 → short-title 입력 요청
2. 해당 카테고리의 `_TEMPLATE.md` 읽기
3. 오늘 날짜(YYYY-MM-DD)와 제목을 반영하여 새 파일 생성:
   - 경로: `skill_graph/<category>/YYYY-MM-DD_<short-title>.md`
4. 생성된 파일 경로 출력

## 규칙

- 파일명에 공백 대신 하이픈(`-`) 사용
- 템플릿의 플레이스홀더(`[...]`, `YYYY-MM-DD`)를 실제 값으로 치환
- `## 관련 노트` 섹션: 생성 직후 키워드 기반 자동 연결 시도
  1. 새 노트의 제목/키워드에서 핵심 키워드 추출
  2. skill_graph/ 내 기존 노트와 키워드 겹침 검사
  3. 겹치는 키워드 2개 이상인 노트를 양방향 링크
  4. 관련 노트 없으면 빈 채로 유지
- 이미 같은 이름의 파일이 있으면 경고 후 사용자에게 확인
