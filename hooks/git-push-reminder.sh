#!/bin/bash
#
# Git Push Reminder Hook
# PreToolUse (Bash) 에서 실행 — git push 전 리뷰 리마인더
#
# Bash 도구 호출 시 git push 명령이 포함되어 있으면 경고
#

INPUT=$(cat)

# stdin에서 도구 입력 확인
if echo "$INPUT" | grep -q "git push"; then
    echo "[Git Push] push 전 확인: 변경사항을 리뷰했나요? 커밋 메시지가 적절한가요?" >&2
fi

# stdin 전달
echo "$INPUT"
