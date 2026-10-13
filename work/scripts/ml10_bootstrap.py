"""stdlib-only path / artifact helpers for the ML-10 action-playbook notebook.

The notebook keeps its own tiny "find or clone the repo" boot step (it has to run
before this module can be imported), then delegates the directory/artifact rules
here so there is a single, tested definition of each one.

No pandas / huggingface_hub import at module load, so this can be unit-tested with
no warehouse and no token. Tested by ``work/tests/test_ml10_workflow.py``.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

# The fork that carries the committed ML-07 / ML-09 receipts this notebook reads.
REPO_URL = "https://github.com/Divyanshu10045/flyrank-assignments"

# Committed INPUT artifacts the notebook only reads. Missing -> hard stop.
REQUIRED_RECEIPTS = (
    "work/outputs/w06_validation_metrics.json",   # ML-09, the model + the L2
    "work/outputs/baseline_metrics.json",         # ML-07, the frozen rule reference
)

# OUTPUT directories this assignment writes into. Created on demand, never inputs.
OUTPUT_DIRS = (
    "work/outputs",
    "work/figures",
)


def _is_repo_root(path: Path) -> bool:
    return (path / "work" / "notebooks").is_dir() and (path / ".git").exists()


def find_repo_root(start=None):
    """Nearest directory at/above ``start`` that looks like the repo root, or None.

    A repo root here means it has both ``work/notebooks/`` and a ``.git`` entry --
    the same test the notebook boot uses, so the two cannot drift.
    """
    here = Path(start).resolve() if start is not None else Path.cwd().resolve()
    for cand in (here, *here.parents):
        if _is_repo_root(cand):
            return cand
    return None


def ensure_dir(path) -> Path:
    """Create a directory (and parents) if missing; return the Path. Outputs only."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_output_dirs(repo):
    """Create every OUTPUT directory under ``repo``; return the created paths.

    Idempotent. Never touches inputs -- a missing output directory is ours to make,
    which is the whole point of separating it from the receipts.
    """
    return [ensure_dir(Path(repo) / rel) for rel in OUTPUT_DIRS]


def missing_receipts(repo, rels=REQUIRED_RECEIPTS):
    """Committed INPUT receipts absent under ``repo`` (returned as relative paths).

    Creating the output directories does not change this: a receipt is a committed
    artifact of an earlier assignment, not something this notebook generates.
    """
    return [rel for rel in rels if not (Path(repo) / rel).is_file()]


def resolve_repo(start=None, in_colab=None, clone_url=REPO_URL, clone=True, runner=subprocess.check_call):
    """Best-effort repo-root resolution shared by the notebook boot and the tests.

    Finds an existing clone first; if none and running in Colab, clones the fork once.
    Raises RuntimeError with an actionable message if the root still cannot be found.
    """
    start_dir = Path(start).resolve() if start is not None else Path.cwd().resolve()
    root = find_repo_root(start_dir)
    if root is None:
        if in_colab is None:
            in_colab = "google.colab" in sys.modules
        if in_colab and clone:
            dest = start_dir / Path(clone_url).name
            if not _is_repo_root(dest):
                runner(["git", "clone", "--depth", "1", clone_url, str(dest)])
            root = find_repo_root(dest)
    if root is None:
        raise RuntimeError(
            "could not find the repo root from %s\n"
            "In Colab, clone it first:  git clone %s\n"
            "then Runtime -> Run all." % (start_dir, clone_url))
    return root
