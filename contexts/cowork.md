# Cowork Context

Mode: 파일 기반 협업 작업면
Focus: plan.md / handoff.md / outputs/ 구조를 통한 체계적 작업 관리

## 행동 지침
- 작업 시작 전 반드시 `plan.md` 확인 또는 작성
- 모든 산출물은 `outputs/` 폴더에 저장
- 세션 종료 전 `handoff.md` 갱신 필수
- 긴 작업은 반드시 상태 파일을 남긴다 (plan.md, decision-log.md, handoff.md)

## Cowork 파일 구조

```
project/
├── plan.md          # 현재 작업 계획 — 제약, 할 일, 바지 않을 입출력
├── handoff.md       # 인수인계 상태 — 어디까지 했는지, 다음에 볼 파일
├── outputs/         # 최종 산출물 모아두는 곳
├── decision-log.md  # 의사결정 기록 (선택)
└── work-log.md      # 작업 이력 (선택)
```

## 3문항 체크 (작업 전)
1. 지금 비용 대비 행동은 정리, 초안, 검증, 전달, 자동화 중 어디에 가까운가?
2. 이 일은 Cowork처럼 위험한 문서 작업에 가까운가, Claude Code처럼 구현과 검증 작업에 가까운가?
3. 사람이 마지막 승인 없이 바로 나가면 위험한 건인가?

## 멀티에이전트 프로토콜
- **planner**: 제약, 할 일, 바지 않을 입출력 → `plan.md`에 적는다
- **builder**: `plan.md`만 기준으로 작업, 변경 내용 → `implementation-notes.md`에 남김
- **reviewer**: 결과물 + 검증 기준만 읽고 `review-findings.md`에 판별만 적는다
- **human**: `decision-log.md`에 최종 판단을 남긴다

## 활성화
`이 세션은 cowork 모드로 진행합니다` 또는 contexts/cowork.md를 참조하세요.
