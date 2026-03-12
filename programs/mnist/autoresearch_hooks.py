"""
Autoresearch hooks — auto-loaded by prepare.py via sitecustomize-like mechanism.
Provides experiment status tracking and result logging for the dashboard.
DO NOT MODIFY — this file is infrastructure, not part of the experiment.
"""
import atexit
import json
import subprocess
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


def _get_program_dir():
    """Find programs/mnist/ regardless of working directory."""
    try:
        root = Path(subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip())
        candidate = root / "programs" / "mnist"
        if candidate.exists():
            return candidate
    except Exception:
        pass
    # fallback to cwd
    here = Path.cwd()
    candidate = here / "programs" / "mnist"
    if candidate.exists():
        return candidate
    return here


PROGRAM_DIR = _get_program_dir()
RESULTS_FILE = PROGRAM_DIR / "results.tsv"
RESULTS_HEADER = "commit\tval_loss\tmemory_gb\tstatus\tdescription\n"
STATUS_FILE = PROGRAM_DIR / "status.json"


def start_tracking(time_budget=60):
    """Write status.json for the dashboard. Auto-cleanup on exit."""
    STATUS_FILE.write_text(json.dumps({
        "running": True,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "time_budget": time_budget,
    }))
    atexit.register(lambda: STATUS_FILE.unlink(missing_ok=True))

# Auto-start tracking when imported (via sitecustomize)
start_tracking()


def log_result(val_loss, memory_gb, status="keep", description=""):
    """Append a row to results.tsv."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short=7", "HEAD"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
    except Exception:
        commit = "unknown"
    if not RESULTS_FILE.exists():
        RESULTS_FILE.write_text(RESULTS_HEADER)
    with open(RESULTS_FILE, "a") as f:
        f.write(f"{commit}\t{val_loss:.6f}\t{memory_gb:.1f}\t{status}\t{description}\n")
    print(f"Logged to {RESULTS_FILE}: {commit} val_loss={val_loss:.6f} [{status}]")
