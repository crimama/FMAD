#!/bin/bash
#
# Lessons Reminder Hook
# Stop 이벤트에서 실행 — 세션 중 교훈 기록 리마인더
#
# 사용자의 수정/지적이 있었을 때 tasks/lessons.md 업데이트를 상기시킨다.
#

# tasks/lessons.md 존재 확인
if [ -f "tasks/lessons.md" ]; then
    # lessons.md가 이번 세션에서 수정되었는지 확인 (git diff)
    if git diff --quiet tasks/lessons.md 2>/dev/null; then
        # 변경 없음 — 리마인더 출력 (비차단)
        echo "[Lessons] 이번 세션에서 수정/지적이 있었다면, tasks/lessons.md에 교훈을 기록하세요." >&2
    fi
fi

# stdin 전달
cat
