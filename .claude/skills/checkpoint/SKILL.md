---
name: checkpoint
description: git 기반 작업 체크포인트 생성, 검증, 목록 관리
allowed-tools:
  - Bash
  - Read
  - Write
---

# /checkpoint

**사용법**: `/checkpoint [create <name> | verify <name> | list | clear]`

현재 체크포인트:

```
!cat .claude/checkpoints.log 2>/dev/null || echo "(체크포인트 없음)"!
```

## 동작

### `create <name>` → 체크포인트 생성
1. `/verify quick` 수준의 빠른 상태 확인
2. 현재 변경사항을 git commit (또는 stash) 으로 저장
3. `.claude/checkpoints.log`에 기록:
   ```
   YYYY-MM-DD-HH:MM | <name> | <git-sha-short>
   ```
4. 체크포인트 생성 완료 보고

### `verify <name>` → 체크포인트 비교
1. `.claude/checkpoints.log`에서 해당 체크포인트 조회
2. 현재 상태와 체크포인트 시점 비교:
   - 추가/수정된 파일 수
   - 테스트 통과율 변화
   - 빌드 상태
3. 비교 결과 보고:
   ```
   CHECKPOINT COMPARISON: <name>
   ============================
   Files changed: X
   Tests: +Y passed / -Z failed
   Build: [PASS/FAIL]
   ```

### `list` → 체크포인트 목록
- 이름, 타임스탬프, Git SHA, 현재 브랜치 대비 상태 표시

### `clear` → 오래된 체크포인트 정리
- 최근 5개만 유지, 나머지 삭제

## 워크플로우 예시

```
[시작] → /checkpoint create "feature-start"
   |
[구현] → /checkpoint create "core-done"
   |
[테스트] → /checkpoint verify "core-done"
   |
[리팩토링] → /checkpoint create "refactor-done"
   |
[PR] → /checkpoint verify "feature-start"
```

## 규칙

- `.claude/checkpoints.log`가 없으면 자동 생성
- 체크포인트 이름에 공백 대신 하이픈 사용
- commit 메시지: `checkpoint: <name>`
- 사용자에게 commit 여부 확인 후 진행
