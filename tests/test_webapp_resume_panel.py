"""Webapp resume-panel unit tests.

Pins the contract for the per-run audit-relax toggle that the resume
editor in ``easyicu.webapp.research_agent`` exposes to users:

* ``_run_pipeline(audit_relax_probe=True, ...)`` sets
  ``EASYICU_AUDIT_RELAX_PROBE=1`` for the duration of the pipeline call.
* The env var is restored to its prior value after the call returns,
  including when the underlying pipeline raises.
* ``audit_relax_probe=False`` (the default) does not mutate the env at
  all.

These three properties are what make the webapp's "Relax probe-stage
audits" checkbox a clean, scoped ablation toggle rather than a
process-global mode switch that could leak into the next user's run.
"""

from __future__ import annotations

import os
import pandas as pd
import pytest


def _make_handles_stub():
    """A minimal ResearchAgentPipeline stand-in that records the env."""

    class _Pipeline:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def run(self, **kwargs):
            # Capture the env state at call time so the test can assert
            # the var is set during the call.
            self.observed_env = os.environ.get("EASYICU_AUDIT_RELAX_PROBE")
            # If asked to raise, do so to exercise the finally clause.
            prefs = kwargs.get("user_preferences") or {}
            if isinstance(prefs, dict) and prefs.get("_raise"):
                raise RuntimeError("stub failure")
            return self

        run_id = "stub_run"
        workdir = "."
        manuscript_path = ""

    captured: dict = {}

    def _factory(**kw):
        inst = _Pipeline(**kw)
        captured["instance"] = inst
        return inst

    return {"ResearchAgentPipeline": _factory}, captured


def _make_cohort():
    return pd.DataFrame({"stay_id": [1, 2, 3]})


def test_audit_relax_probe_sets_env_during_call():
    from easyicu.webapp.research_agent import _run_pipeline

    os.environ.pop("EASYICU_AUDIT_RELAX_PROBE", None)
    handles, captured = _make_handles_stub()

    _run_pipeline(
        handles=handles,
        cohort=_make_cohort(),
        skill_key=None,
        question="dummy",
        target_outcome=None,
        workdir=".",
        llm=None,
        disable_icu_context=True,
        audit_relax_probe=True,
    )

    assert captured["instance"].observed_env == "1"
    # Restored after the call
    assert os.environ.get("EASYICU_AUDIT_RELAX_PROBE") is None


def test_audit_relax_probe_default_does_not_touch_env():
    from easyicu.webapp.research_agent import _run_pipeline

    os.environ.pop("EASYICU_AUDIT_RELAX_PROBE", None)
    handles, captured = _make_handles_stub()

    _run_pipeline(
        handles=handles,
        cohort=_make_cohort(),
        skill_key=None,
        question="dummy",
        target_outcome=None,
        workdir=".",
        llm=None,
        disable_icu_context=True,
        audit_relax_probe=False,
    )

    assert captured["instance"].observed_env is None
    assert os.environ.get("EASYICU_AUDIT_RELAX_PROBE") is None


def test_audit_relax_probe_restores_prior_value():
    from easyicu.webapp.research_agent import _run_pipeline

    os.environ["EASYICU_AUDIT_RELAX_PROBE"] = "preexisting_value"
    handles, captured = _make_handles_stub()

    _run_pipeline(
        handles=handles,
        cohort=_make_cohort(),
        skill_key=None,
        question="dummy",
        target_outcome=None,
        workdir=".",
        llm=None,
        disable_icu_context=True,
        audit_relax_probe=True,
    )

    # Toggled on during the call
    assert captured["instance"].observed_env == "1"
    # Restored to the prior caller's value (not blanked)
    assert os.environ.get("EASYICU_AUDIT_RELAX_PROBE") == "preexisting_value"
    # Cleanup so we don't pollute the test process
    os.environ.pop("EASYICU_AUDIT_RELAX_PROBE", None)


def test_audit_relax_probe_restores_env_on_exception():
    from easyicu.webapp.research_agent import _run_pipeline

    os.environ.pop("EASYICU_AUDIT_RELAX_PROBE", None)
    handles, _captured = _make_handles_stub()

    with pytest.raises(RuntimeError):
        # Pass the magic key through user_preferences → the stub picks
        # it up via **kwargs and raises so we exercise the finally branch.
        _run_pipeline(
            handles=handles,
            cohort=_make_cohort(),
            skill_key=None,
            question="dummy",
            target_outcome=None,
            workdir=".",
            llm=None,
            disable_icu_context=True,
            audit_relax_probe=True,
            user_preferences={"_raise": True},
        )

    assert os.environ.get("EASYICU_AUDIT_RELAX_PROBE") is None
