"""Session lifecycle management."""

import os
import shutil
from datetime import datetime

from .config import OrchestratorConfig
from .protocol import write_plan


def create_session(query: str, config: OrchestratorConfig = None) -> str:
    """Create a new orchestration session.

    Returns:
        session_dir: Absolute path to the session directory.
    """
    if config is None:
        config = OrchestratorConfig.from_env()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Sanitize query for directory name
    slug = query[:40].strip().replace(" ", "_").replace("/", "-")
    slug = "".join(c for c in slug if c.isalnum() or c in "_-")
    session_id = f"{timestamp}_{slug}"

    session_dir = os.path.abspath(os.path.join(config.workspace_root, session_id))
    os.makedirs(session_dir, exist_ok=True)
    os.makedirs(os.path.join(session_dir, "agents"), exist_ok=True)

    # Write initial plan placeholder
    write_plan(session_dir, query, "(Plan to be filled by orchestrator)")

    # Write session metadata
    meta_path = os.path.join(session_dir, "meta.md")
    with open(meta_path, "w") as f:
        f.write(f"""# Session Metadata

## ID
{session_id}

## Query
{query}

## Created
{datetime.now().isoformat()}

## Config
- agent_command: {config.agent_command}
- approval_mode: {config.agent_approval_mode}
- use_worktree: {config.use_worktree}
- base_branch: {config.base_branch}
- timeout: {config.agent_timeout}s
""")

    return session_dir


def list_sessions(config: OrchestratorConfig = None) -> list:
    """List all sessions."""
    if config is None:
        config = OrchestratorConfig.from_env()

    root = config.workspace_root
    if not os.path.isdir(root):
        return []

    sessions = []
    for name in sorted(os.listdir(root), reverse=True):
        session_dir = os.path.join(root, name)
        if os.path.isdir(session_dir):
            plan_path = os.path.join(session_dir, "plan.md")
            has_plan = os.path.exists(plan_path)
            agents_dir = os.path.join(session_dir, "agents")
            agent_count = len(os.listdir(agents_dir)) if os.path.isdir(agents_dir) else 0
            sessions.append({
                "id": name,
                "path": session_dir,
                "has_plan": has_plan,
                "agent_count": agent_count,
            })
    return sessions


def get_latest_session(config: OrchestratorConfig = None) -> str:
    """Get the most recent session directory path."""
    sessions = list_sessions(config)
    if not sessions:
        raise FileNotFoundError("No sessions found.")
    return sessions[0]["path"]


def clean_session(session_dir: str) -> None:
    """Remove a session and its worktrees."""
    if os.path.isdir(session_dir):
        shutil.rmtree(session_dir)
