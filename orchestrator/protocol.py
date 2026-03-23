"""Markdown-based communication protocol between orchestrator and agents.

File structure per session:
    .orchestrator/sessions/{session_id}/
        plan.md                     # Master plan (Opus writes)
        agents/{agent_id}/
            task.md                 # Task assignment (Opus -> Codex)
            instruction.md          # System instruction for Codex
            status.md               # Live status (updated during execution)
            result.md               # Final deliverable
        summary.md                  # Final aggregated summary
"""

import os
import re
from datetime import datetime
from enum import Enum
from typing import Optional


class AgentStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


def write_plan(session_dir: str, query: str, plan_content: str) -> str:
    """Write the master plan markdown."""
    path = os.path.join(session_dir, "plan.md")
    content = f"""# Development Plan

## Query
{query}

## Created
{datetime.now().isoformat()}

## Plan
{plan_content}
"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(content)
    return path


def write_task(session_dir: str, agent_id: str, task: dict) -> str:
    """Write a task assignment for an agent.

    Args:
        session_dir: Path to session directory.
        agent_id: Agent identifier (e.g., "agent-01").
        task: Dict with keys: title, description, files, acceptance_criteria, dependencies.
    """
    agent_dir = os.path.join(session_dir, "agents", agent_id)
    os.makedirs(agent_dir, exist_ok=True)

    path = os.path.join(agent_dir, "task.md")
    deps = task.get("dependencies", [])
    deps_str = ", ".join(deps) if deps else "None"
    files = task.get("files", [])
    files_str = "\n".join(f"- `{f}`" for f in files) if files else "- (to be determined)"
    criteria = task.get("acceptance_criteria", [])
    criteria_str = "\n".join(f"- [ ] {c}" for c in criteria) if criteria else "- [ ] Task completed"

    content = f"""# Task: {task.get('title', agent_id)}

## Agent
{agent_id}

## Status
{AgentStatus.PENDING.value}

## Description
{task.get('description', '')}

## Target Files
{files_str}

## Dependencies
{deps_str}

## Acceptance Criteria
{criteria_str}

## Notes
(Orchestrator may add coordination notes here)
"""
    with open(path, "w") as f:
        f.write(content)
    return path


def write_instruction(session_dir: str, agent_id: str, instruction: str) -> str:
    """Write system instruction for a Codex agent."""
    agent_dir = os.path.join(session_dir, "agents", agent_id)
    os.makedirs(agent_dir, exist_ok=True)

    path = os.path.join(agent_dir, "instruction.md")
    with open(path, "w") as f:
        f.write(instruction)
    return path


def write_status(session_dir: str, agent_id: str, status: AgentStatus,
                 message: str = "", progress: str = "") -> str:
    """Update agent status file."""
    agent_dir = os.path.join(session_dir, "agents", agent_id)
    os.makedirs(agent_dir, exist_ok=True)

    path = os.path.join(agent_dir, "status.md")
    content = f"""# Agent Status: {agent_id}

## Status
{status.value}

## Updated
{datetime.now().isoformat()}

## Message
{message}

## Progress
{progress}
"""
    with open(path, "w") as f:
        f.write(content)
    return path


def read_status(session_dir: str, agent_id: str) -> dict:
    """Read agent status from markdown."""
    path = os.path.join(session_dir, "agents", agent_id, "status.md")
    if not os.path.exists(path):
        return {"status": AgentStatus.PENDING.value, "message": "", "progress": ""}

    with open(path) as f:
        content = f.read()

    result = {}
    # Parse status
    m = re.search(r"^## Status\n(.+?)$", content, re.MULTILINE)
    result["status"] = m.group(1).strip() if m else AgentStatus.PENDING.value

    m = re.search(r"^## Updated\n(.+?)$", content, re.MULTILINE)
    result["updated"] = m.group(1).strip() if m else ""

    m = re.search(r"^## Message\n(.+?)(?=\n##|\Z)", content, re.MULTILINE | re.DOTALL)
    result["message"] = m.group(1).strip() if m else ""

    m = re.search(r"^## Progress\n(.+?)(?=\n##|\Z)", content, re.MULTILINE | re.DOTALL)
    result["progress"] = m.group(1).strip() if m else ""

    return result


def write_result(session_dir: str, agent_id: str, result_content: str,
                 files_changed: list = None) -> str:
    """Write agent result file."""
    agent_dir = os.path.join(session_dir, "agents", agent_id)
    os.makedirs(agent_dir, exist_ok=True)

    path = os.path.join(agent_dir, "result.md")
    files_str = ""
    if files_changed:
        files_str = "\n".join(f"- `{f}`" for f in files_changed)
    else:
        files_str = "(none reported)"

    content = f"""# Result: {agent_id}

## Completed
{datetime.now().isoformat()}

## Files Changed
{files_str}

## Summary
{result_content}
"""
    with open(path, "w") as f:
        f.write(content)
    return path


def read_result(session_dir: str, agent_id: str) -> Optional[str]:
    """Read agent result if available."""
    path = os.path.join(session_dir, "agents", agent_id, "result.md")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return f.read()


def write_summary(session_dir: str, content: str) -> str:
    """Write final session summary."""
    path = os.path.join(session_dir, "summary.md")
    with open(path, "w") as f:
        f.write(content)
    return path


def list_agents(session_dir: str) -> list:
    """List all agent IDs in a session."""
    agents_dir = os.path.join(session_dir, "agents")
    if not os.path.isdir(agents_dir):
        return []
    return sorted([
        d for d in os.listdir(agents_dir)
        if os.path.isdir(os.path.join(agents_dir, d))
    ])
