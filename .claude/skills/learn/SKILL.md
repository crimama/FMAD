---
name: learn
description: 세션 학습 — 패턴 관찰, 교훈 추출, 지식 승격 관리
allowed-tools:
  - Read
  - Write
  - Edit
  - Glob
  - Grep
---

# /learn

**사용법**: `/learn [observe | extract | promote | status]`

## 개요

세션에서 발견된 패턴을 체계적으로 학습하고 지식 체계로 승격하는 시스템입니다.
현재 프로젝트의 `tasks/lessons.md` → `skill_graph/analysis/` 승격 체계와 연동됩니다.

## 학습 파이프라인

```
세션 활동 (수정, 지적, 에러 해결)
      ↓ 관찰
tasks/lessons.md (즉시 기록)
      ↓ 반복 검증 (2회+)
skill_graph/analysis/{주제}/_lessons.md (승격)
      ↓ 핵심 요약
MEMORY.md (영속 기억)
```

## 동작

### `observe` 또는 인자 없음 → 세션 패턴 관찰
1. 현재 세션의 대화 컨텍스트에서 패턴 식별:
   - 사용자의 수정/지적 패턴
   - 반복된 에러 해결 방법
   - 선호하는 코드 스타일
   - 워크플로우 패턴
2. 발견된 패턴을 `tasks/lessons.md`에 기록 제안
3. 기존 교훈과의 중복 확인

### `extract` → 교훈 추출 및 정리
1. `tasks/lessons.md` 전체 스캔
2. 교훈을 도메인별로 분류:
   - **code-style**: 코드 작성 패턴
   - **debugging**: 디버깅 인사이트
   - **architecture**: 아키텍처 결정
   - **workflow**: 작업 프로세스
   - **testing**: 테스트 패턴
3. 반복 검증 횟수 표시
4. 승격 후보 식별 (2회+ 반복)

### `promote` → 검증된 패턴 승격
1. `tasks/lessons.md`에서 승격 후보 식별:
   - 2회 이상 반복 검증된 패턴
   - 구체적이고 실행 가능한 교훈
2. `skill_graph/analysis/{도메인}/_lessons.md`로 승격:
   - 도메인 디렉토리 자동 생성
   - `_LESSONS_TEMPLATE.md` 형식 적용
   - 출처 세션, 검증 횟수, 적용 조건 명시
3. 원본 `tasks/lessons.md`에 `(→ 승격됨)` 표시
4. MEMORY.md의 Lessons Learned 섹션 갱신 권고

### `status` → 학습 현황 대시보드
```
LEARNING STATUS
===============
Lessons (tasks/lessons.md):     X개
  - code-style:    Y개
  - debugging:     Z개
  - ...
Promoted (skill_graph/analysis/):  N개
  - {도메인}/_lessons.md: M개
Promotion candidates (2회+):    K개

Recent lessons:
  [날짜] 교훈 제목 (검증 N회)
  ...
```

## 패턴 유형별 범위 가이드

| 패턴 유형 | 범위 | 예시 |
|----------|------|------|
| 언어/프레임워크 컨벤션 | 프로젝트 | "React hooks 사용", "Django REST 패턴" |
| 파일 구조 선호 | 프로젝트 | "테스트는 `__tests__/`에", "컴포넌트는 src/components/" |
| 코드 스타일 | 프로젝트 | "함수형 스타일", "dataclass 선호" |
| 보안 실천 | 범용 | "입력 검증 필수", "SQL 살균" |
| 일반 모범 사례 | 범용 | "테스트 먼저 작성", "에러 항상 핸들링" |
| 도구 워크플로우 | 범용 | "수정 전 Grep", "작성 전 Read" |

## 규칙

- 교훈은 **구체적**으로 — "주의하자" 같은 모호한 표현 금지
- 승격은 **2회 이상 반복 검증된 패턴**만
- 기존 교훈과 중복이면 기존 항목 보강
- 추측 금지 — 실제 관찰된 패턴만 기록
- Self-Improvement Loop 원칙 연동
