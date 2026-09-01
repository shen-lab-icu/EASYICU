#!/usr/bin/env python3
"""Run every cheap architecture gate that CI runs, before the commit lands.

Why this exists
---------------
On 2026-08-22 three independent gates were red at ``main`` simultaneously and
had been for four commits:

* ``arch_measure --diff`` — ``acfd465`` grew ``pipeline.py`` 9006 -> 9017 and
  ``_generate_or_resume_plan`` 442 -> 453;
* ``test_plan_phase_decomposition`` — the same function crossed its 450-line cap;
* ``test_trajectory_prompt_compaction`` — ``624b464`` (two days and ~40 commits
  earlier) pushed the widest coder prompt 41,948 -> 42,217 past its 42,000 gate.

None of them are subtle and all three are detected in seconds. They survived
because ``CLAUDE.md``'s development-phase policy — correctly — says not to run
the full suite between small fixes, and nothing cheaper stood in the gap. This
script is that cheaper thing: still far cheaper than the full suite, so there is
no reason to reach ``main`` without it.

Measured 2026-08-30: ~7.5 minutes end to end. The first four gates are
effectively free (arch_measure 1.3s, module-graph ratchet 6.5s, import-linter
0.4s, ruff 0.1s); the size/budget pytest selection dominates at 437s. If you
need a sub-10-second signal mid-edit, run the first four directly and leave the
size/budget gate for pre-push.

Usage::

    python tools/check_architecture_gates.py

Exit code is 0 only when every gate passes. Run it before pushing; it is the
same set ``.github/workflows/research_agent_ci.yml`` runs first, so a green
run here means that job will not fail on its architecture step.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path
from typing import List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]

GATES: Sequence[Tuple[str, List[str]]] = (
    (
        "arch_measure ratchet",
        [
            sys.executable,
            "tools/arch_measure.py",
            "--diff",
            "tools/arch_baselines/execution_phase.json",
        ],
    ),
    (
        "module-graph ratchet",
        [
            sys.executable,
            "tools/research_agent_module_graph.py",
            "--diff",
            "tools/arch_baselines/research_agent_module_graph.json",
        ],
    ),
    (
        "import-linter boundaries",
        ["lint-imports", "--no-cache"],
    ),
    (
        "ruff",
        [sys.executable, "-m", "ruff", "check", "src", "tests"],
    ),
    (
        "size and budget guards",
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-m",
            "",
            "-p",
            "no:randomly",
            "tests/governance/test_arch_measure.py",
            "tests/governance/test_research_agent_module_graph.py",
            "tests/governance/test_static_architecture_policy.py",
            "tests/research_agent/planning/test_plan_phase_decomposition.py",
            "tests/research_agent/planning/test_trajectory_prompt_compaction.py",
            "tests/research_agent/gates/test_execution_phase_contract.py",
        ],
    ),
)


def main() -> int:
    failures: List[str] = []
    for name, argv in GATES:
        print(f"\n=== {name} ===", flush=True)
        started = time.monotonic()
        try:
            completed = subprocess.run(argv, cwd=ROOT, check=False)
        except FileNotFoundError:
            # A missing dev tool is a real failure: reporting it as "passed"
            # would be the exact silent-green failure this script exists to
            # stop. Name the install so the fix is obvious.
            print(
                f"  MISSING TOOL: {argv[0]} is not installed "
                '(pip install -e ".[dev]")',
                file=sys.stderr,
            )
            failures.append(name)
            continue
        elapsed = time.monotonic() - started
        status = "ok" if completed.returncode == 0 else "FAILED"
        print(f"  -> {status} ({elapsed:.1f}s)", flush=True)
        if completed.returncode != 0:
            failures.append(name)

    print()
    if failures:
        print(f"FAIL: {len(failures)} gate(s) red: {', '.join(failures)}")
        return 1
    print(f"OK: all {len(GATES)} architecture gates green.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
