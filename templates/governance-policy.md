# AI Tool Access Policy — 거버넌스 정책

<!-- 프로젝트에 맞게 수정하세요 -->
<!-- 이 문서는 목적은 범주 문서를 만드는 것이 아니라, "어디서 자동화 문제"이고 "어디서부터 사람 승인"인지 주는 것이다 -->

## Allowed by default (자동 실행 가능)

- Read, Edit, Write inside approved workspaces
- lint, test, build commands
- approved plugins from managed marketplace

## Requires human approval (사람 검토 후 실행)

- deployment
- customer-facing email send
- scheduled tasks with external side effects
- database writes
- 주기 보고서 자동 생성

## Blocked (실행 금지)

- reading `.env*`
- destructive shell commands
- unapproved MCP servers
- editing outside approved repositories

---

## 고위험 작업 등급표

| 등급 | 예시 | 승인 방식 |
|------|------|----------|
| 낮음 | 파일 읽기, 내부 초안 작성, 테스트 추가 | 자동 실행 가능 |
| 중간 | 고객용 초안 작성, 주기 보고서, 코드 리팩터링 | 사람 검토 후 실행 |
| 높음 | 프로덕션 배포, 고객 발송, 권한 변경, 민감 데이터 조회 | 사람 승인 없이 실행 금지 |

## 도입 우선순위

### 1주차
- `.env` 와 파괴적 명령 차단
- 배포와 발송분 무조건 승인

### 2주차
- 공용 plugin / skill만 허용
- 프로젝트급 CLAUDE.md rules 적용

### 3주차
- 비용 모니터링
- 승인 로그와 handoff 로그 정착
