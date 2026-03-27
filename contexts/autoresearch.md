# AutoResearch Context

Mode: 자율 실험 루프
Focus: 코드 수정 → 실험 → 평가 → Keep/Discard 무한 반복

> Andrej Karpathy의 autoresearch 패턴.
> "Python이 아닌 program.md로 연구를 프로그래밍한다."

## 행동 지침
- `program.md`가 유일한 법이다. 수정 가능 파일, 지표, 실행 명령이 모두 여기에 정의됨
- **한 실험 = 한 변경** (single-claim test). 여러 변경을 한 번에 하지 않는다
- 결과가 개선되면 keep, 아니면 discard. 단순함이 동일 성능이면 keep
- **절대 멈추지 않는다.** 사용자가 수동으로 중단할 때까지 계속한다
- 아이디어가 고갈되면 program.md를 다시 읽고, 실패 패턴을 분석하고, 더 급진적 변경을 시도한다

## 실험 루프 (The Karpathy Loop)

```
LOOP FOREVER:
  1. 상태 확인 — 현재 브랜치, 최근 결과
  2. 가설 수립 — program.md 목표 + 이전 결과 기반
  3. 코드 수정 — 허용된 파일만, 한 가지 변경
  4. git commit — 실험 설명
  5. 실험 실행 — run_command > run.log 2>&1
  6. 결과 확인 — 지표 추출
  7. 판정:
     - 개선 → keep (브랜치 전진)
     - 동일/악화 → discard (git reset --hard HEAD~1)
     - 크래시 → 간단 수정 시도, 안 되면 포기
  8. results.tsv에 기록
  9. → 1번으로
```

## 선호 도구
- Edit → 코드 수정 (한 번에 한 파일)
- Bash → 실험 실행, git 조작, 로그 확인
- Read → program.md, 결과 파일, 로그 확인
- Grep → 지표 추출, 에러 추적

## 시간 감각
- 5분 학습 = ~12 실험/시간 = ~100 실험/하룻밤
- 타임아웃(시간 예산 2배) 초과 시 kill & discard
- 크래시 수정은 최대 3회 시도 후 포기

## results.tsv 형식
```
commit	metric	memory_gb	status	description
a1b2c3d	0.997900	44.0	keep	baseline
```

## 활성화
`이 세션은 autoresearch 모드로 진행합니다` 또는 contexts/autoresearch.md를 참조하세요.
`/autoresearch setup`으로 새 실험 세션을 초기화하세요.
