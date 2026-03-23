---
name: orchestrate
description: Multi-agent orchestrator. 복잡한 개발 작업을 Codex CLI 에이전트들에게 병렬 분배하여 수행. 3개 이상 독립 모듈/파일에 걸친 개발 작업 시 사용.
user-invocable: true
argument-hint: "[development requirement description]"
---

# Multi-Agent Orchestrator

개발 요구사항을 분석하고, 독립 작업으로 분해하여, 다수의 Codex CLI 에이전트에 병렬 할당한다.
모든 에이전트와의 통신은 `.orchestrator/sessions/` 내 markdown 파일로 수행한다.

요구사항: $ARGUMENTS

---

아래 단계를 **순서대로** 실행하라. 각 단계에서 Bash 도구로 CLI 명령을 호출한다.

## Phase 1: Init & Plan

### 1-1. 세션 생성

```bash
python -m orchestrator init "$ARGUMENTS"
```

출력에서 세션 경로를 기억한다. 이후 모든 명령에서 이 경로를 `<session>`으로 사용한다.

### 1-2. 코드베이스 분석

요구사항에 관련된 기존 코드를 Read/Grep/Glob으로 탐색한다:
- 수정 대상 파일 식별
- 기존 패턴/인터페이스 파악
- 파일 간 의존성 확인

### 1-3. 작업 분해 & 계획 작성

요구사항을 **독립 실행 가능한 작업 단위**로 분해한다.

분해 원칙:
- 각 작업은 서로 다른 파일을 수정 (파일 충돌 방지)
- 공유 인터페이스가 필요하면 인터페이스 정의를 먼저 agent-00에 할당
- 의존성이 있는 작업은 deps로 표시하여 순차 실행
- 작업당 50-150줄 범위가 적정

plan.md 파일을 Write 도구로 직접 작성한다:
```
<session>/plan.md
```

## Phase 2: Task Assignment

각 작업을 에이전트에 할당한다. 에이전트 ID는 `agent-01`, `agent-02`, ... 형식.

```bash
python -m orchestrator add-task <session> agent-01 \
    --title "작업 제목" \
    --desc "상세 설명. 구현해야 할 내용, 참고할 기존 코드, 기대 결과를 명확히 기술." \
    --files "path/to/file1.py,path/to/file2.py" \
    --criteria "수락기준1|수락기준2|수락기준3" \
    --instruction "AGENTS.md"
```

**desc 작성 가이드:**
- 무엇을 구현/수정해야 하는지 구체적으로
- 참고할 기존 파일/함수 명시
- 예상 인터페이스 (함수 시그니처, 클래스 구조) 포함
- 프로젝트 컨벤션 언급 (AGENTS.md 참조 지시)

모든 에이전트의 task를 추가한 뒤, 사용자에게 계획을 보여주고 확인을 받는다:
- 총 에이전트 수, 각 작업 요약, 예상 파일 변경 범위
- 사용자 승인 후 다음 단계로 진행

## Phase 3: Dispatch

```bash
python -m orchestrator dispatch <session>
```

이 명령은 모든 pending 에이전트를 동시에 실행한다. 각 에이전트는:
- 독립 git worktree에서 작업 (`orch/{agent-id}` 브랜치)
- `codex --approval-mode full-auto` 로 실행
- task.md의 지시사항을 수행

## Phase 4: Monitor

에이전트 실행 후 주기적으로 상태를 확인한다:

```bash
python -m orchestrator status <session>
```

상태별 대응:
- **running**: 대기. 필요시 agent.log를 Read로 확인
- **completed**: Phase 5로 진행
- **failed**: agent.log 분석 → task.md 수정 후 재dispatch
- **timeout**: task 범위 축소 → 재dispatch

모든 에이전트가 완료될 때까지 반복한다.

## Phase 5: Review

각 에이전트의 결과를 검토한다:

1. 변경 사항 확인:
```bash
git diff main...orch/agent-01
```

2. result.md 확인:
```bash
# Read 도구로 읽기
<session>/agents/agent-01/result.md
```

3. 수락 기준 점검 — task.md의 criteria와 대조

문제 발견 시:
- task.md의 Notes 섹션에 수정 지시 추가 (Edit 도구)
- 해당 에이전트만 재dispatch: `python -m orchestrator dispatch <session> agent-01`

## Phase 6: Merge

모든 에이전트가 승인되면 브랜치를 병합한다:

```bash
python -m orchestrator merge <session> --cleanup
```

충돌 발생 시:
1. 충돌 파일 확인
2. 직접 해결하거나 관련 에이전트에 수정 지시
3. 재merge

## Phase 7: Summary

사용자에게 최종 결과를 보고한다:
- 총 에이전트 수, 성공/실패 현황
- 주요 변경 파일 목록
- 각 에이전트의 작업 요약
- 다음 단계 제안 (테스트 실행, 추가 작업 등)

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| ORCH_AGENT_CMD | codex | Agent CLI command |
| ORCH_APPROVAL_MODE | full-auto | Codex approval mode |
| ORCH_AGENT_MODEL | (default) | Codex model override |
| ORCH_POLL_INTERVAL | 5 | Status poll interval (sec) |
| ORCH_AGENT_TIMEOUT | 600 | Agent timeout (sec) |
| ORCH_USE_WORKTREE | true | Use git worktrees |
| ORCH_BASE_BRANCH | main | Base branch for worktrees |

## Communication Protocol

```
.orchestrator/sessions/{session_id}/
├── plan.md                    # [Opus -> All] Master plan
├── meta.md                    # Session metadata
├── agents/
│   └── {agent_id}/
│       ├── task.md            # [Opus -> Codex] Task assignment
│       ├── instruction.md     # [Opus -> Codex] System instructions
│       ├── status.md          # [System] Agent status tracking
│       ├── result.md          # [Codex -> Opus] Deliverable
│       ├── agent.log          # Agent stdout/stderr
│       └── worktree/          # Git worktree (isolated workspace)
└── summary.md                 # [Opus] Final summary
```
