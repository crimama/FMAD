"""Monitor multiple running agents and report status."""

import os
import time
from datetime import datetime
from typing import Optional

from .agent import AgentProcess, recover_agent
from .config import OrchestratorConfig
from .protocol import AgentStatus, list_agents, read_result, read_status, write_status


class AgentMonitor:
    """Monitors all agents in a session."""

    def __init__(self, session_dir: str, config: OrchestratorConfig = None):
        self.session_dir = session_dir
        self.config = config or OrchestratorConfig.from_env()
        self.agents: dict[str, AgentProcess] = {}

    def register(self, agent: AgentProcess) -> None:
        """Register an agent for monitoring."""
        self.agents[agent.agent_id] = agent

    def recover_all(self) -> None:
        """Recover all agents from session state."""
        for agent_id in list_agents(self.session_dir):
            if agent_id not in self.agents:
                agent = recover_agent(agent_id, self.session_dir, self.config)
                self.agents[agent_id] = agent

    def check_agent(self, agent_id: str) -> dict:
        """Check status of a single agent."""
        agent = self.agents.get(agent_id)
        status_data = read_status(self.session_dir, agent_id)

        # If no agent process registered, or agent was never launched, trust status.md
        if agent is None or agent.process is None:
            pid_file = os.path.join(self.session_dir, "agents", agent_id, ".pid")
            if not os.path.exists(pid_file):
                return status_data
            # PID file exists — try to recover
            if agent is None:
                agent = recover_agent(agent_id, self.session_dir, self.config)
                self.agents[agent_id] = agent

        ret = agent.poll()

        if ret is None:
            # Still running
            if agent.is_timed_out():
                agent.terminate()
                write_status(self.session_dir, agent_id, AgentStatus.TIMEOUT,
                             message=f"Timed out after {self.config.agent_timeout}s")
                status_data["status"] = AgentStatus.TIMEOUT.value
            else:
                status_data["status"] = AgentStatus.RUNNING.value
        else:
            # Process finished
            if ret == 0:
                # Check if result file exists
                result = read_result(self.session_dir, agent_id)
                if result or status_data.get("status") == AgentStatus.COMPLETED.value:
                    status_data["status"] = AgentStatus.COMPLETED.value
                else:
                    # Agent finished but no result — check git diff
                    changed = agent.get_changed_files()
                    if changed:
                        write_status(self.session_dir, agent_id, AgentStatus.COMPLETED,
                                     message="Agent completed with changes",
                                     progress=f"Changed files: {', '.join(changed)}")
                        status_data["status"] = AgentStatus.COMPLETED.value
                    else:
                        write_status(self.session_dir, agent_id, AgentStatus.COMPLETED,
                                     message="Agent completed (no result file written)")
                        status_data["status"] = AgentStatus.COMPLETED.value
            else:
                write_status(self.session_dir, agent_id, AgentStatus.FAILED,
                             message=f"Exit code: {ret}")
                status_data["status"] = AgentStatus.FAILED.value

        return status_data

    def check_all(self) -> dict:
        """Check status of all agents."""
        self.recover_all()
        results = {}
        for agent_id in list_agents(self.session_dir):
            results[agent_id] = self.check_agent(agent_id)
        return results

    def is_all_done(self) -> bool:
        """Check if all agents have finished (completed, failed, or timed out)."""
        statuses = self.check_all()
        if not statuses:
            return True
        done_states = {AgentStatus.COMPLETED.value, AgentStatus.FAILED.value,
                       AgentStatus.TIMEOUT.value}
        return all(s.get("status") in done_states for s in statuses.values())

    def wait_all(self, callback=None) -> dict:
        """Block until all agents finish. Optionally call callback on each poll.

        Args:
            callback: Optional function(statuses_dict) called each poll cycle.

        Returns:
            Final status dict for all agents.
        """
        while not self.is_all_done():
            statuses = self.check_all()
            if callback:
                callback(statuses)
            time.sleep(self.config.poll_interval)
        return self.check_all()

    def format_status_table(self) -> str:
        """Format a human-readable status table."""
        statuses = self.check_all()
        if not statuses:
            return "No agents registered."

        lines = ["| Agent | Status | Message |", "|-------|--------|---------|"]
        for agent_id, data in sorted(statuses.items()):
            status = data.get("status", "unknown")
            message = data.get("message", "")[:60]
            lines.append(f"| {agent_id} | {status} | {message} |")

        return "\n".join(lines)
