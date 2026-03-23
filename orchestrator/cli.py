#!/usr/bin/env python3
"""CLI entry point for the orchestrator.

Usage:
    python -m orchestrator init "development query here"
    python -m orchestrator plan <session> --content "plan text"
    python -m orchestrator add-task <session> <agent_id> --title "..." --desc "..."
    python -m orchestrator dispatch <session> [agent_id]
    python -m orchestrator status [session]
    python -m orchestrator merge <session> [agent_id]
    python -m orchestrator clean <session>
    python -m orchestrator sessions
"""

import argparse
import json
import os
import sys

from .agent import AgentProcess
from .config import OrchestratorConfig
from .merge import cleanup_agent_branches, merge_agent_branch, merge_all_agents
from .monitor import AgentMonitor
from .protocol import (
    AgentStatus,
    list_agents,
    read_result,
    read_status,
    write_instruction,
    write_plan,
    write_result,
    write_status,
    write_task,
)
from .session import clean_session, create_session, get_latest_session, list_sessions


def cmd_init(args, config):
    """Initialize a new session."""
    session_dir = create_session(args.query, config)
    print(f"Session created: {session_dir}")
    print(f"Plan template: {os.path.join(session_dir, 'plan.md')}")
    return session_dir


def cmd_plan(args, config):
    """Write or update the session plan."""
    session_dir = resolve_session(args.session, config)

    if args.file:
        with open(args.file) as f:
            content = f.read()
    else:
        content = args.content

    if not content:
        print("Error: Provide --content or --file", file=sys.stderr)
        sys.exit(1)

    path = write_plan(session_dir, "(see plan)", content)
    print(f"Plan written: {path}")


def cmd_add_task(args, config):
    """Add a task for an agent."""
    session_dir = resolve_session(args.session, config)

    task = {
        "title": args.title,
        "description": args.desc or "",
        "files": args.files.split(",") if args.files else [],
        "acceptance_criteria": args.criteria.split("|") if args.criteria else [],
        "dependencies": args.deps.split(",") if args.deps else [],
    }

    path = write_task(session_dir, args.agent_id, task)
    print(f"Task written: {path}")

    # Write instruction if provided
    if args.instruction:
        if os.path.isfile(args.instruction):
            with open(args.instruction) as f:
                instruction = f.read()
        else:
            instruction = args.instruction
        ipath = write_instruction(session_dir, args.agent_id, instruction)
        print(f"Instruction written: {ipath}")


def cmd_dispatch(args, config):
    """Launch agent(s)."""
    session_dir = resolve_session(args.session, config)

    if args.agent_id:
        agent_ids = [args.agent_id]
    else:
        agent_ids = list_agents(session_dir)

    if not agent_ids:
        print("No agents to dispatch. Add tasks first.", file=sys.stderr)
        sys.exit(1)

    monitor = AgentMonitor(session_dir, config)

    for agent_id in agent_ids:
        # Skip already completed agents
        status = read_status(session_dir, agent_id)
        if status.get("status") in (AgentStatus.COMPLETED.value, AgentStatus.RUNNING.value):
            print(f"Skipping {agent_id} (status: {status['status']})")
            continue

        task_path = os.path.join(session_dir, "agents", agent_id, "task.md")
        if not os.path.exists(task_path):
            print(f"Skipping {agent_id} (no task.md)", file=sys.stderr)
            continue

        agent = AgentProcess(agent_id, session_dir, config)
        pid = agent.launch()
        monitor.register(agent)
        print(f"Dispatched {agent_id} (PID: {pid})")

    if args.wait:
        print("\nWaiting for all agents to complete...")

        def on_poll(statuses):
            for aid, s in statuses.items():
                print(f"  {aid}: {s.get('status', '?')}", end="  ")
            print()

        final = monitor.wait_all(callback=on_poll)
        print("\n--- Final Status ---")
        print(monitor.format_status_table())
    else:
        print(f"\nAgents launched. Check status with: python -m orchestrator status {session_dir}")


def cmd_status(args, config):
    """Show status of all agents in a session."""
    session_dir = resolve_session(args.session, config)
    monitor = AgentMonitor(session_dir, config)
    print(monitor.format_status_table())

    # Show results for completed agents
    if args.verbose:
        for agent_id in list_agents(session_dir):
            result = read_result(session_dir, agent_id)
            if result:
                print(f"\n--- Result: {agent_id} ---")
                print(result)


def cmd_merge(args, config):
    """Merge agent branch(es) into base."""
    session_dir = resolve_session(args.session, config)

    if args.agent_id:
        result = merge_agent_branch(args.agent_id, config.base_branch)
        result["agent_id"] = args.agent_id
        results = [result]
    else:
        results = merge_all_agents(session_dir, config)

    for r in results:
        status = "OK" if r["success"] else "CONFLICT"
        print(f"[{status}] {r.get('agent_id', '?')}: {r['message']}")
        if r.get("files_changed"):
            for f in r["files_changed"]:
                print(f"  - {f}")

    if args.cleanup:
        removed = cleanup_agent_branches(session_dir)
        if removed:
            print(f"\nCleaned up branches: {', '.join(removed)}")


def cmd_sessions(args, config):
    """List all sessions."""
    sessions = list_sessions(config)
    if not sessions:
        print("No sessions found.")
        return

    print(f"{'ID':<50} {'Agents':<8} {'Plan'}")
    print("-" * 70)
    for s in sessions:
        plan = "yes" if s["has_plan"] else "no"
        print(f"{s['id']:<50} {s['agent_count']:<8} {plan}")


def cmd_clean(args, config):
    """Clean up a session."""
    session_dir = resolve_session(args.session, config)

    # Clean worktrees first
    for agent_id in list_agents(session_dir):
        branch = f"orch/{agent_id}"
        worktree = os.path.join(session_dir, "agents", agent_id, "worktree")
        if os.path.exists(worktree):
            os.system(f"git worktree remove --force {worktree}")
        os.system(f"git branch -D {branch} 2>/dev/null")

    clean_session(session_dir)
    print(f"Cleaned: {session_dir}")


def resolve_session(session_arg: str, config: OrchestratorConfig) -> str:
    """Resolve session argument to absolute path."""
    if session_arg is None or session_arg == "latest":
        return get_latest_session(config)
    if os.path.isabs(session_arg) and os.path.isdir(session_arg):
        return session_arg
    # Try as session ID
    candidate = os.path.join(config.workspace_root, session_arg)
    if os.path.isdir(candidate):
        return os.path.abspath(candidate)
    # Try as absolute path
    if os.path.isdir(session_arg):
        return os.path.abspath(session_arg)
    raise FileNotFoundError(f"Session not found: {session_arg}")


def main():
    parser = argparse.ArgumentParser(
        prog="orchestrator",
        description="Multi-agent development orchestrator",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # init
    p = sub.add_parser("init", help="Create new session")
    p.add_argument("query", help="Development query / requirement")

    # plan
    p = sub.add_parser("plan", help="Write session plan")
    p.add_argument("session", help="Session dir or 'latest'")
    p.add_argument("--content", help="Plan text")
    p.add_argument("--file", help="Plan from file")

    # add-task
    p = sub.add_parser("add-task", help="Add task for an agent")
    p.add_argument("session", help="Session dir or 'latest'")
    p.add_argument("agent_id", help="Agent identifier (e.g., agent-01)")
    p.add_argument("--title", required=True, help="Task title")
    p.add_argument("--desc", help="Task description")
    p.add_argument("--files", help="Comma-separated target files")
    p.add_argument("--criteria", help="Pipe-separated acceptance criteria")
    p.add_argument("--deps", help="Comma-separated dependency agent IDs")
    p.add_argument("--instruction", help="Instruction text or file path")

    # dispatch
    p = sub.add_parser("dispatch", help="Launch agent(s)")
    p.add_argument("session", help="Session dir or 'latest'")
    p.add_argument("agent_id", nargs="?", help="Specific agent (default: all)")
    p.add_argument("--wait", action="store_true", help="Wait for completion")

    # status
    p = sub.add_parser("status", help="Check agent statuses")
    p.add_argument("session", nargs="?", default="latest", help="Session dir")
    p.add_argument("-v", "--verbose", action="store_true", help="Show results")

    # merge
    p = sub.add_parser("merge", help="Merge agent branches")
    p.add_argument("session", help="Session dir or 'latest'")
    p.add_argument("agent_id", nargs="?", help="Specific agent (default: all)")
    p.add_argument("--cleanup", action="store_true", help="Delete branches after merge")

    # sessions
    sub.add_parser("sessions", help="List all sessions")

    # clean
    p = sub.add_parser("clean", help="Clean up a session")
    p.add_argument("session", help="Session dir or 'latest'")

    args = parser.parse_args()
    config = OrchestratorConfig.from_env()

    commands = {
        "init": cmd_init,
        "plan": cmd_plan,
        "add-task": cmd_add_task,
        "dispatch": cmd_dispatch,
        "status": cmd_status,
        "merge": cmd_merge,
        "sessions": cmd_sessions,
        "clean": cmd_clean,
    }

    try:
        commands[args.command](args, config)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
