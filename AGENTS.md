# AGENTS.md

This file provides guidance to Codex when working in this repository. It mirrors the existing Claude setup in `CLAUDE.md` and `.claude/skills/`.

## First Read

At the start of each meaningful task, review these files in order:

1. `CLAUDE.md`
2. `tasks/lessons.md`
3. `tasks/todo.md`

Treat `CLAUDE.md` as the primary project playbook unless a direct user instruction overrides it.

## Core Working Rules

- Keep changes minimal, simple, and reproducible.
- Prefer root-cause fixes over patches.
- Any newly added module or feature must be switchable from config with an explicit `enable: true/false` style control where applicable.
- Do not mark work complete without verification evidence.

## Required Workflow

### 1. Task Tracking

Before substantial implementation, update `tasks/todo.md` with:

- `## 현재 작업`
- `## 계획`
- progress checkmarks while working
- `## 결과` after completion

Do not overwrite unrelated existing content.

### 2. Lessons

When the user corrects your approach or a recurring mistake becomes clear, record it in `tasks/lessons.md` using:

```md
### [YYYY-MM-DD] 제목
발생 상황: ...
잘못한 것: ...
올바른 방법: ...
```

If a lesson becomes a repeated, validated pattern, promote it into `skill_graph/analysis/<topic>/_lessons.md`.

### 3. Verification

Do not finish on intent alone. Verify with the strongest practical signal available, such as:

- running the relevant command or experiment entrypoint
- checking logs or generated artifacts
- confirming config wiring and execution path

If full verification is too heavy, state exactly what was and was not verified.

## Agents and Contexts

### Agents (`agents/` directory)
| Agent | Model | Purpose |
|-------|-------|---------|
| planner | opus | Implementation planning |
| code-reviewer | sonnet | Code quality/security review |

### Context Modes (`contexts/` directory)
| Mode | File | Focus |
|------|------|-------|
| dev | `contexts/dev.md` | Implementation — code first |
| research | `contexts/research.md` | Exploration — understand first |
| review | `contexts/review.md` | Quality, security, maintainability |

## Codex Mapping For Existing Claude Skills

Codex cannot auto-register the local `.claude/skills/*` files as native skills, so use them as workflow references:

- `.claude/skills/todo/SKILL.md`: how to maintain `tasks/todo.md`
- `.claude/skills/lessons/SKILL.md`: how to record and promote lessons
- `.claude/skills/update-note/SKILL.md`: create notes under `skill_graph/`
- `.claude/skills/link-notes/SKILL.md`: related-note linking workflow
- `.claude/skills/verify/SKILL.md`: build/type/lint/test verification
- `.claude/skills/checkpoint/SKILL.md`: git-based checkpoint management
- `.claude/skills/compact/SKILL.md`: strategic compaction guide
- `.claude/skills/learn/SKILL.md`: session learning pipeline
