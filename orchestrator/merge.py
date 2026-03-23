"""Git merge operations for agent worktree branches."""

import subprocess
from typing import Optional

from .config import OrchestratorConfig
from .protocol import list_agents


def get_agent_branch(agent_id: str) -> str:
    """Get the branch name for an agent."""
    return f"orch/{agent_id}"


def check_conflicts(branch_a: str, branch_b: str) -> bool:
    """Check if two branches would conflict when merged.

    Returns True if conflict detected.
    """
    # Try a merge dry-run
    result = subprocess.run(
        ["git", "merge-tree", "--write-tree", branch_a, branch_b],
        capture_output=True, text=True,
    )
    # merge-tree returns non-zero if there are conflicts
    return result.returncode != 0


def merge_agent_branch(agent_id: str, base_branch: str,
                       merge_message: Optional[str] = None) -> dict:
    """Merge an agent's worktree branch into base branch.

    Returns:
        Dict with 'success', 'message', 'files_changed'.
    """
    branch = get_agent_branch(agent_id)
    if merge_message is None:
        merge_message = f"Merge {agent_id} work from orchestrator"

    # Get changed files before merge
    diff_result = subprocess.run(
        ["git", "diff", "--name-only", f"{base_branch}...{branch}"],
        capture_output=True, text=True,
    )
    files_changed = [f.strip() for f in diff_result.stdout.strip().split("\n") if f.strip()]

    if not files_changed:
        return {"success": True, "message": "No changes to merge", "files_changed": []}

    # Checkout base branch
    subprocess.run(["git", "checkout", base_branch], capture_output=True, text=True)

    # Merge
    result = subprocess.run(
        ["git", "merge", "--no-ff", branch, "-m", merge_message],
        capture_output=True, text=True,
    )

    if result.returncode != 0:
        # Abort on conflict
        subprocess.run(["git", "merge", "--abort"], capture_output=True)
        return {
            "success": False,
            "message": f"Merge conflict: {result.stderr}",
            "files_changed": files_changed,
            "conflict_details": result.stdout,
        }

    return {
        "success": True,
        "message": f"Merged {len(files_changed)} files from {agent_id}",
        "files_changed": files_changed,
    }


def merge_all_agents(session_dir: str, config: OrchestratorConfig) -> list:
    """Merge all agent branches sequentially.

    Returns list of merge results per agent.
    """
    results = []
    agent_ids = list_agents(session_dir)

    for agent_id in agent_ids:
        branch = get_agent_branch(agent_id)

        # Check branch exists
        check = subprocess.run(
            ["git", "rev-parse", "--verify", branch],
            capture_output=True, text=True,
        )
        if check.returncode != 0:
            results.append({
                "agent_id": agent_id,
                "success": False,
                "message": f"Branch {branch} not found",
            })
            continue

        result = merge_agent_branch(agent_id, config.base_branch)
        result["agent_id"] = agent_id
        results.append(result)

        if not result["success"]:
            # Stop on first conflict — orchestrator should resolve
            break

    return results


def cleanup_agent_branches(session_dir: str) -> list:
    """Delete all agent branches after successful merge."""
    removed = []
    for agent_id in list_agents(session_dir):
        branch = get_agent_branch(agent_id)
        result = subprocess.run(
            ["git", "branch", "-d", branch],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            removed.append(branch)
    return removed
