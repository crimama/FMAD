---
name: autoresearch
description: Karpathy-style 자율 실험 루프. AI가 코드를 수정하고, 실험하고, 개선되면 유지/아니면 폐기를 반복. program.md가 연구 방향을 정의.
allowed-tools:
  - Bash
  - Read
  - Edit
  - Write
  - Grep
  - Glob
---

# /autoresearch

**사용법**: `/autoresearch [setup | run | status | results]`

> Andrej Karpathy의 autoresearch 패턴을 일반화한 자율 실험 루프.
> 핵심: **Read → Hypothesize → Modify → Train → Evaluate → Keep/Discard → Repeat**

## 동작

`$ARGUMENTS`에 따라 모드를 결정합니다.

### `setup` — 새 실험 세션 초기화

1. **program.md 확인**: 프로젝트 루트에 `program.md`가 있는지 확인. 없으면 `templates/program.md` 기반으로 사용자와 대화하며 생성
2. **브랜치 생성**: 오늘 날짜 기반 태그 제안 (예: `autoresearch/mar23`), `git checkout -b autoresearch/<tag>`
3. **스코프 파일 읽기**: `program.md`에 명시된 수정 가능 파일과 읽기 전용 파일을 모두 읽어 컨텍스트 파악
4. **기준 데이터 확인**: 학습 데이터, 환경, 의존성이 준비되었는지 확인
5. **results.tsv 초기화**: 헤더 행만 있는 결과 로그 파일 생성
6. **베이스라인 실행**: 현재 코드 그대로 첫 실험 실행하여 기준 지표 기록
7. **확인 후 루프 시작**: 사용자 확인 후 자율 실험 루프 진입

### `run` — 자율 실험 루프 실행 (NEVER STOP)

**무한 반복:**

1. **상태 확인** — 현재 브랜치, 최근 결과, 지금까지의 실험 이력 확인
2. **가설 수립** — `program.md`의 목표와 이전 실험 결과를 바탕으로 다음 실험 아이디어 구상
3. **코드 수정** — `program.md`에서 허용된 파일만 수정. 한 실험에 한 가지 변경(single-claim)
4. **커밋** — 실험 내용을 설명하는 커밋 메시지로 git commit
5. **실험 실행** — `program.md`의 실행 명령어로 학습 실행, 출력을 `run.log`로 리다이렉트
   ```bash
   # 예시 (program.md의 run_command에 따라 달라짐)
   <run_command> > run.log 2>&1
   ```
6. **결과 확인** — 지표 추출 (val_bpb, accuracy, loss 등 program.md에 정의된 metric)
7. **판정 & 기록**:
   - 크래시 시: `tail -n 50 run.log`로 원인 파악, 간단한 수정 시도 (최대 3회). 안 되면 포기
   - 지표 개선 시: **keep** — 브랜치 전진, results.tsv에 기록
   - 지표 동일/악화 시: **discard** — `git reset --hard HEAD~1`, results.tsv에 기록
8. **반복** — 1번으로 돌아감

**절대 멈추지 않는다.** 사용자가 수동으로 중단할 때까지 계속 실험한다. 아이디어가 고갈되면 더 열심히 생각한다 — program.md 재독, 이전 실패 패턴 분석, 더 급진적 변경 시도.

### `status` — 현재 실험 상태 확인

1. 현재 브랜치와 커밋 확인
2. `results.tsv` 읽어서 요약 (총 실험 수, keep/discard/crash 비율, 최고 지표)
3. 최근 5개 실험 결과 표시
4. 현재 최고 성과 vs 베이스라인 비교

### `results` — 전체 결과 분석

1. `results.tsv` 전체 읽기
2. 지표 추이 분석 (개선 곡선)
3. 가장 효과적이었던 변경 Top 5
4. 가장 많이 시도된 카테고리별 성공률
5. `skill_graph/experiments/`에 종합 보고서 생성

## results.tsv 형식

탭 구분 (TSV). 쉼표는 description에서 깨지므로 사용 금지.

```
commit	metric	memory_gb	status	description
a1b2c3d	0.997900	44.0	keep	baseline
b2c3d4e	0.993200	44.2	keep	increase LR to 0.04
c3d4e5f	1.005000	44.0	discard	switch to GeLU activation
d4e5f6g	0.000000	0.0	crash	double model width (OOM)
```

## 핵심 규칙

1. **program.md가 법이다**: 수정 가능 파일, 읽기 전용 파일, 목표 지표, 실행 명령이 모두 여기에 정의됨
2. **한 실험 = 한 변경**: 여러 변경을 한 번에 하지 않는다 (single-claim test)
3. **단순함 우선**: 같은 성능이면 더 단순한 코드를 선택. 복잡도 대비 개선이 미미하면 discard
4. **results.tsv는 git 미추적**: 실험 메타데이터만 기록, 코드 변경은 git으로 추적
5. **타임아웃**: program.md에 정의된 시간 예산의 2배를 초과하면 kill & discard
6. **자율성**: 루프 시작 후 사용자에게 "계속할까요?" 묻지 않는다. 사람이 자고 있을 수 있다
7. **실패도 자산**: crash/discard도 results.tsv에 기록. 왜 실패했는지가 다음 실험의 단서

## program.md 연동

`program.md`가 없으면 `templates/program.md`를 기반으로 사용자와 대화하며 생성합니다.
기존 research 프리셋의 6단계 실험 프로세스와 병행 가능:
- autoresearch는 **탐색 단계**(빠른 반복, 넓은 범위)
- 6단계 프로세스는 **검증 단계**(깊은 분석, 논문 기여)

## tasks/lessons.md 연동

- 반복 실패 패턴 발견 시 `tasks/lessons.md`에 기록
- 검증된 개선 패턴은 `skill_graph/analysis/`로 승격
