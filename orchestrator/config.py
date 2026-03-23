"""Orchestrator configuration."""

import os
from dataclasses import dataclass, field


@dataclass
class OrchestratorConfig:
    """Configuration for the orchestrator system."""

    # Workspace
    workspace_root: str = ".orchestrator/sessions"

    # Agent backend
    agent_command: str = "codex"
    agent_approval_mode: str = "full-auto"
    agent_model: str = ""  # empty = use codex default

    # Monitoring
    poll_interval: int = 5  # seconds
    agent_timeout: int = 600  # seconds (10 min)

    # Git
    use_worktree: bool = True
    base_branch: str = "main"

    # Project context files to pass to agents
    context_files: list = field(default_factory=lambda: [
        "CLAUDE.md",
        "AGENTS.md",
    ])

    @classmethod
    def from_env(cls) -> "OrchestratorConfig":
        """Load config with environment variable overrides."""
        cfg = cls()
        cfg.agent_command = os.environ.get("ORCH_AGENT_CMD", cfg.agent_command)
        cfg.agent_approval_mode = os.environ.get("ORCH_APPROVAL_MODE", cfg.agent_approval_mode)
        cfg.agent_model = os.environ.get("ORCH_AGENT_MODEL", cfg.agent_model)
        cfg.poll_interval = int(os.environ.get("ORCH_POLL_INTERVAL", cfg.poll_interval))
        cfg.agent_timeout = int(os.environ.get("ORCH_AGENT_TIMEOUT", cfg.agent_timeout))
        cfg.use_worktree = os.environ.get("ORCH_USE_WORKTREE", "true").lower() == "true"
        cfg.base_branch = os.environ.get("ORCH_BASE_BRANCH", cfg.base_branch)
        return cfg
