"""Codex agent lifecycle: launch, monitor, terminate."""

import json
import os
import signal
import subprocess
import time
from datetime import datetime
from typing import Optional

from .config import OrchestratorConfig
from .protocol import AgentStatus, read_status, write_status


class AgentProcess:
    """Manages a single Codex agent subprocess."""

    def __init__(self, agent_id: str, session_dir: str, config: OrchestratorConfig):
        self.agent_id = agent_id
        self.session_dir = session_dir
        self.config = config
        self.process: Optional[subprocess.Popen] = None
        self.worktree_path: Optional[str] = None
        self.branch_name: Optional[str] = None
        self.start_time: Optional[float] = None
        self.pid_file = os.path.join(session_dir, "agents", agent_id, ".pid")

    @property
    def agent_dir(self) -> str:
        return os.path.join(self.session_dir, "agents", self.agent_id)

    def setup_worktree(self) -> str:
        """Create a git worktree for isolated agent work."""
        self.branch_name = f"orch/{self.agent_id}"
        self.worktree_path = os.path.join(self.agent_dir, "worktree")

        # Clean up existing worktree if present
        if os.path.exists(self.worktree_path):
            subprocess.run(
                ["git", "worktree", "remove", "--force", self.worktree_path],
                capture_output=True,
            )

        # Delete branch if it exists (from previous run)
        subprocess.run(
            ["git", "branch", "-D", self.branch_name],
            capture_output=True,
        )

        # Create worktree with new branch
        result = subprocess.run(
            ["git", "worktree", "add", "-b", self.branch_name,
             self.worktree_path, self.config.base_branch],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Failed to create worktree: {result.stderr}")

        return self.worktree_path

    def build_command(self) -> list:
        """Build the codex CLI command."""
        task_path = os.path.join(self.agent_dir, "task.md")
        instruction_path = os.path.join(self.agent_dir, "instruction.md")

        # Read task content
        with open(task_path) as f:
            task_content = f.read()

        # Build prompt for codex
        prompt = f"Read and execute the task described below.\n\n{task_content}"

        cmd = [self.config.agent_command, "exec"]

        # Add approval mode
        if self.config.agent_approval_mode == "full-auto":
            cmd.append("--full-auto")

        # Add model if specified
        if self.config.agent_model:
            cmd.extend(["--model", self.config.agent_model])

        # Add the prompt
        cmd.append(prompt)

        return cmd

    def launch(self) -> int:
        """Launch the Codex agent as a subprocess.

        Returns:
            PID of the launched process.
        """
        # Setup worktree if configured
        cwd = os.getcwd()
        if self.config.use_worktree:
            cwd = self.setup_worktree()

        cmd = self.build_command()

        # Update status
        write_status(self.session_dir, self.agent_id, AgentStatus.RUNNING,
                     message=f"Launched at {datetime.now().isoformat()}",
                     progress="Starting...")

        # Log the command
        log_path = os.path.join(self.agent_dir, "agent.log")

        with open(log_path, "w") as log_file:
            log_file.write(f"# Agent Log: {self.agent_id}\n")
            log_file.write(f"# Command: {' '.join(cmd)}\n")
            log_file.write(f"# CWD: {cwd}\n")
            log_file.write(f"# Started: {datetime.now().isoformat()}\n\n")
            log_file.flush()

            self.process = subprocess.Popen(
                cmd,
                cwd=cwd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                preexec_fn=os.setsid,  # New process group for clean termination
            )

        self.start_time = time.time()

        # Save PID
        with open(self.pid_file, "w") as f:
            f.write(str(self.process.pid))

        return self.process.pid

    def poll(self) -> Optional[int]:
        """Check if agent process has finished.

        Returns:
            None if still running, return code if finished.
        """
        if self.process is None:
            # Try to recover from PID file
            if os.path.exists(self.pid_file):
                with open(self.pid_file) as f:
                    pid = int(f.read().strip())
                try:
                    os.kill(pid, 0)  # Check if process exists
                    return None  # Still running
                except ProcessLookupError:
                    return 0  # Process gone, assume completed
            return 0

        return self.process.poll()

    def is_timed_out(self) -> bool:
        """Check if agent has exceeded timeout."""
        if self.start_time is None:
            return False
        return (time.time() - self.start_time) > self.config.agent_timeout

    def terminate(self) -> None:
        """Terminate the agent process."""
        if self.process and self.process.poll() is None:
            try:
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
                self.process.wait(timeout=10)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                try:
                    os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass

        # Clean PID file
        if os.path.exists(self.pid_file):
            os.remove(self.pid_file)

    def cleanup_worktree(self) -> None:
        """Remove the git worktree (keeps branch for merging)."""
        if self.worktree_path and os.path.exists(self.worktree_path):
            subprocess.run(
                ["git", "worktree", "remove", "--force", self.worktree_path],
                capture_output=True,
            )

    def get_changed_files(self) -> list:
        """Get list of files changed by this agent in its worktree branch."""
        if not self.branch_name:
            return []
        result = subprocess.run(
            ["git", "diff", "--name-only", f"{self.config.base_branch}...{self.branch_name}"],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            return []
        return [f.strip() for f in result.stdout.strip().split("\n") if f.strip()]


def recover_agent(agent_id: str, session_dir: str,
                  config: OrchestratorConfig) -> AgentProcess:
    """Recover an agent process from session state."""
    agent = AgentProcess(agent_id, session_dir, config)
    pid_file = os.path.join(session_dir, "agents", agent_id, ".pid")

    if os.path.exists(pid_file):
        with open(pid_file) as f:
            pid = int(f.read().strip())
        # Check if process is still alive
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            pass  # Process already finished

    # Check for worktree
    worktree_path = os.path.join(session_dir, "agents", agent_id, "worktree")
    if os.path.exists(worktree_path):
        agent.worktree_path = worktree_path
        agent.branch_name = f"orch/{agent_id}"

    return agent
