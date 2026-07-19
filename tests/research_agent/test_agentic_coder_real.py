"""Real-machine validation for the altitude-2a agentic coder.

Marked ``needs_real_data`` → skipped by default. Run with::

    EASYICU_DATA_PATH=/path/to/db \
    EASYICU_TEST_COHORT_PARQUET=/path/to/a/cohort.parquet \
    pytest tests/research_agent/test_agentic_coder_real.py --run-real -q

This is the test the offline suite cannot be: it answers the open question the
unit tests deliberately leave open — *can the Codex/Claude CLI sandbox actually
import easyicu and reach the cohort parquet, run the analysis, and self-repair,
or does it silently degrade to "just writes code" (= altitude 1)?*

The proof is step 4: we re-execute the CLI-authored script ourselves against the
real cohort and require exit code 0. If Codex never really ran it, the script it
returns will usually error here.

Requirements (any missing → the test skips with a clear reason):
- ``--run-real`` and an existing ``EASYICU_DATA_PATH`` (the marker gate).
- the ``codex`` CLI installed and signed in (override with
  ``EASYICU_AGENTIC_CODER_BACKEND=claude``).
- ``EASYICU_TEST_COHORT_PARQUET`` pointing at a small prepared cohort parquet
  (one row per ICU stay, the same shape the runtime exports to ``COHORT_PARQUET``).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.needs_real_data


def _backend() -> str:
    return (os.environ.get("EASYICU_AGENTIC_CODER_BACKEND") or "codex").strip().lower()


def _require_cli(ra):
    from easyicu.research_agent.providers.llm import cli_backend_available

    backend = _backend()
    if not cli_backend_available(backend):
        pytest.skip(f"{backend} CLI not installed/on PATH; cannot validate altitude-2a")
    return backend


def _require_cohort_parquet() -> Path:
    raw = os.environ.get("EASYICU_TEST_COHORT_PARQUET", "").strip()
    if not raw:
        pytest.skip(
            "Set EASYICU_TEST_COHORT_PARQUET to a small prepared cohort parquet "
            "(one row per stay) to validate the real agentic-coder loop."
        )
    path = Path(raw)
    if not path.is_file():
        pytest.skip(f"EASYICU_TEST_COHORT_PARQUET does not exist: {path}")
    return path


def _research_context(ra):
    return ra.ResearchContext(
        research_question=(
            "Report the cohort size and the in-hospital mortality rate, and test "
            "the association between the first available SOFA-like severity column "
            "and mortality if such columns exist."
        ),
        cohort=ra.CohortDescriptor(
            cohort_name="real-validation",
            database=os.environ.get("EASYICU_TEST_DB", "unknown"),
            n_stays=0,
            n_patients=0,
        ),
        variables=[],
    )


def _analysis_step(ra):
    return ra.AnalysisStep(
        step_id="01_descriptive_association",
        intent="Describe the cohort and test one severity→mortality association.",
        inputs=["COHORT_PARQUET"],
        expected_outputs=["cohort_size", "mortality_rate", "association_stat"],
        method=None,
    )


def test_agentic_coder_authors_a_runnable_script_against_real_data(ra, monkeypatch):
    backend = _require_cli(ra)
    cohort_parquet = _require_cohort_parquet()

    # The runtime exposes the cohort via COHORT_PARQUET; the agentic coder passes
    # this env through to the CLI sandbox unchanged.
    monkeypatch.setenv("COHORT_PARQUET", str(cohort_parquet))

    from easyicu.research_agent.agentic_coder import AgenticCoderAgent
    from easyicu.research_agent.providers.mocks import MockLLMClient
    from easyicu.research_agent.agents import CoderAgent

    context = _research_context(ra)
    step = _analysis_step(ra)

    # Fallback is a real CoderAgent on the offline mock so a CLI miss is obvious
    # (delegation must be used; we assert that below).
    fallback = CoderAgent(MockLLMClient(context=context))
    agent = AgenticCoderAgent(fallback, backend=backend, timeout=900.0)

    script = agent.run(context=context, step=step)

    # 1. Delegation actually happened (not the mock fallback).
    assert agent.last_delegation_used is True, "CLI delegation did not run"
    # 2. A non-trivial script came back.
    assert script and len(script.strip()) > 40
    # 3. It reads the cohort from the env contract, not inlined data.
    assert "COHORT_PARQUET" in script

    # 4. THE validation: the CLI-authored script actually runs against the real
    #    cohort. This is what distinguishes a faithful altitude-2a loop from a
    #    degraded "just writes code" outcome.
    tmp = cohort_parquet.parent / "_agentic_coder_real_script.py"
    try:
        tmp.write_text(script, encoding="utf-8")
        proc = subprocess.run(
            [sys.executable, str(tmp)],
            capture_output=True,
            text=True,
            timeout=600,
            env={**os.environ, "COHORT_PARQUET": str(cohort_parquet)},
        )
    finally:
        tmp.unlink(missing_ok=True)

    assert proc.returncode == 0, (
        "CLI-authored script failed to run against the real cohort — the sandbox "
        "likely could not import easyicu / reach the data, so the loop degraded "
        f"to altitude-1.\nSTDERR:\n{proc.stderr[-2000:]}"
    )
