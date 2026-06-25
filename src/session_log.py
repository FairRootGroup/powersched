import datetime
import os
import subprocess
import sys


def _git_info():
    """Return (commit_hash, branch, dirty_flag) strings, or fallback strings on failure."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL
        ).decode().strip()
        dirty = " (dirty)" if status else ""
        return commit, branch, dirty
    except Exception:
        return "unknown", "unknown", ""


def log_invocation(session_root: str) -> None:
    """Append a timestamped invocation record to <session_root>/invocations.log."""
    os.makedirs(session_root, exist_ok=True)
    commit, branch, dirty = _git_info()
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cmd = " ".join(sys.argv)
    line = f"[{ts}] git={commit}{dirty} branch={branch} cmd={cmd}\n"
    log_path = os.path.join(session_root, "invocations.log")
    with open(log_path, "a") as f:
        f.write(line)
