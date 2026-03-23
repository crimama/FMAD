---
name: verify
description: 코드베이스의 빌드, 타입, 린트, 테스트 상태를 종합 검증합니다
allowed-tools:
  - Bash
  - Read
  - Grep
  - Glob
---

# /verify

**사용법**: `/verify [quick | full | pre-commit | pre-pr]`

## 동작

`$ARGUMENTS`에 따라 검증 범위를 결정합니다.

### `quick` — 빌드 + 타입 체크만
1. 프로젝트 빌드 명령 실행
2. 타입 체커 실행 (TypeScript: `npx tsc --noEmit`, Python: `mypy` 등)

### `full` — 전체 검증 (기본값)
1. **Build Check** — 빌드 명령 실행, 실패 시 에러 보고 후 STOP
2. **Type Check** — 타입 에러를 file:line 형식으로 보고
3. **Lint Check** — 린터 실행, 경고/에러 보고
4. **Test Suite** — 전체 테스트 실행, pass/fail 수 및 커버리지 보고
5. **Debug Log Audit** — 소스 파일에서 console.log/print 검색
6. **Git Status** — 미커밋 변경사항 및 최근 수정 파일 표시

### `pre-commit` — 커밋 관련 검증
- Build + Type + Lint + Debug Log Audit

### `pre-pr` — PR 전 전체 검증 + 보안
- full 검증 + 시크릿 스캔 (.env, API 키, 토큰 패턴 검색)

## 출력 형식

```
VERIFICATION: [PASS/FAIL]

Build:    [OK/FAIL]
Types:    [OK/X errors]
Lint:     [OK/X issues]
Tests:    [X/Y passed, Z% coverage]
Secrets:  [OK/X found]
Logs:     [OK/X console.logs]

Ready for PR: [YES/NO]
```

critical 이슈가 있으면 수정 제안과 함께 나열합니다.

## 규칙

- CLAUDE.md의 Commands 섹션에서 프로젝트별 빌드/테스트 명령을 확인하여 사용
- Commands 섹션이 비어있으면 일반적인 명령 시도 (npm, python, make 등)
- Verification Before Done 원칙의 구체적 실행 도구
- 검증 실패 시 자동으로 수정하지 않음 — 결과 보고만
