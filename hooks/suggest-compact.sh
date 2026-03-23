#!/bin/bash
#
# Strategic Compact Suggestion Hook
# PreToolUse (Edit|Write) 에서 실행
#
# 도구 호출 횟수를 추적하여 논리적 전환 지점에서 /compact 제안
# 환경변수: COMPACT_THRESHOLD (기본: 50)
#

THRESHOLD="${COMPACT_THRESHOLD:-50}"
REMINDER_INTERVAL=25
COUNTER_FILE="/tmp/.claude-compact-counter-$$"

# 부모 프로세스 기준 카운터 (세션 단위)
PPID_COUNTER="/tmp/.claude-compact-counter-${PPID}"

# 카운터 읽기
if [ -f "$PPID_COUNTER" ]; then
    COUNT=$(cat "$PPID_COUNTER")
else
    COUNT=0
fi

# 카운터 증가
COUNT=$((COUNT + 1))
echo "$COUNT" > "$PPID_COUNTER"

# 임계치 도달 시 제안
if [ "$COUNT" -eq "$THRESHOLD" ]; then
    echo "[Strategic Compact] 도구 호출 ${THRESHOLD}회 도달. 현재 작업 단계가 완료되었다면 /compact 실행을 고려하세요." >&2
    echo "  - 리서치 → 구현 전환 시: compact 권장" >&2
    echo "  - 구현 중간: compact 비권장 (컨텍스트 유실)" >&2
elif [ "$COUNT" -gt "$THRESHOLD" ] && [ $(( (COUNT - THRESHOLD) % REMINDER_INTERVAL )) -eq 0 ]; then
    echo "[Strategic Compact] 도구 호출 ${COUNT}회. 장시간 세션입니다. 논리적 전환점에서 /compact를 고려하세요." >&2
fi

# stdin을 그대로 전달 (hooks 파이프라인)
cat
