"""Notebook + lockfile provenance artefacts (O26).

For every run the research agent already registers per-step ``.py``
scripts in the EvidenceStore. That satisfies the "code was executed"
requirement but not the TRIPOD+AI / Nature code availability
expectation of a single runnable notebook plus a pinned environment.

This module adds two thin artefacts:

* ``run.ipynb`` — a minimal, valid Jupyter notebook that concatenates
  every generated script in plan order, each wrapped in a code cell.
  It re-runs top-to-bottom given the same cohort + the same pinned
  environment. No fancy narrative; one section per step.
* ``requirements.lock.txt`` — the output of
  ``pip freeze``-equivalent on the active interpreter, captured at
  run time. We use ``importlib.metadata`` so no subprocess is
  needed and the lockfile is deterministic on a given Python
  version + installed packages.

Both files are registered in the EvidenceStore under stable evidence
ids (``run_notebook`` / ``requirements_lockfile``) so the manuscript
can cite them directly: "Source code, notebook and dependency
lockfile are released as
{evidence:run_notebook} and {evidence:requirements_lockfile}".

Pure stdlib.
"""

from __future__ import annotations

import importlib.metadata as _im
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


# ---------------------------------------------------------------------------
# Lockfile
# ---------------------------------------------------------------------------


def build_requirements_lockfile() -> str:
    """Return a pip-style lockfile reflecting the current interpreter.

    Ordering is case-insensitive by package name so diffs across runs
    are meaningful. The Python interpreter version is written as the
    first comment so downstream tooling can verify before installing.
    """
    rows: List[str] = []
    try:
        dists = list(_im.distributions())
    except Exception:
        dists = []
    for dist in dists:
        try:
            name = dist.metadata["Name"]  # type: ignore[index]
            version = dist.version
        except Exception:
            continue
        if not name or not version:
            continue
        rows.append(f"{name}=={version}")
    rows.sort(key=lambda s: s.lower())
    header = [
        "# easyicu.research_agent — requirements.lock",
        f"# python_version={sys.version.split(' ')[0]}",
        f"# python_implementation={sys.implementation.name}",
        "# generated_by=easyicu.research_agent.repro_artifacts",
    ]
    return "\n".join(header + rows) + "\n"


# ---------------------------------------------------------------------------
# Notebook
# ---------------------------------------------------------------------------


def _code_cell(source: str) -> Dict[str, Any]:
    lines = source.splitlines(keepends=True)
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": lines,
    }


def _md_cell(text: str) -> Dict[str, Any]:
    lines = text.splitlines(keepends=True)
    return {"cell_type": "markdown", "metadata": {}, "source": lines}


@dataclass
class NotebookStep:
    step_id: str
    intent: str
    code: str


def build_notebook(
    *,
    research_question: str,
    cohort_relative_path: str,
    steps: Sequence[NotebookStep],
) -> Dict[str, Any]:
    """Return a valid nbformat-4.5 notebook dict with one cell per step."""
    cells: List[Dict[str, Any]] = [
        _md_cell(
            f"# EasyICU research-agent notebook\n\n"
            f"Research question: **{research_question}**\n\n"
            f"Cohort (relative to this notebook): `{cohort_relative_path}`\n\n"
            "This notebook is auto-generated from the agent's per-step "
            "scripts in plan order. Execute top-to-bottom against the "
            "same cohort parquet to reproduce every registered artefact."
        ),
        _code_cell(
            "import os\n"
            "os.environ.setdefault('COHORT_PARQUET', "
            f"{cohort_relative_path!r})\n"
        ),
    ]
    for step in steps:
        cells.append(
            _md_cell(f"## Step `{step.step_id}` — {step.intent}")
        )
        cells.append(_code_cell(step.code))
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": sys.version.split(" ")[0],
            },
            "easyicu_research_agent": {
                "generated_by": "repro_artifacts",
                "schema_version": "easyicu.notebook/1",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def write_notebook(path: Path, notebook: Dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
    return path


__all__ = [
    "NotebookStep",
    "build_notebook",
    "build_requirements_lockfile",
    "write_notebook",
]
