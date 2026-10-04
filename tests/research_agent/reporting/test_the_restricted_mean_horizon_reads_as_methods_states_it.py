"""The restricted-mean horizon reads in Results as Methods states it.

The horizon is the endpoint less the landmark.  A landmark that is not a whole
day gives a fractional horizon, which the suite's projected Results sentence
rounded to whole days (a 36-hour landmark's 88.5 read "88", a 30-hour one's
88.75 read "89") while the Methods fact printed it exactly, so the two
sections named different horizons.  The
sentence now prints the horizon as Methods does; a whole-day horizon reads as
before.

Synthetic study and seeded synthetic rows only (renal replacement therapy and
90-day mortality).
"""

from __future__ import annotations

import json

import pytest

from easyicu.research_agent.authority.evidence_store import EvidenceEnforcementMode, EvidenceStore
from easyicu.research_agent.authority.manuscript_method_facts import _design_text
from easyicu.research_agent.contracts.executed_method_design import (
    EXECUTED_METHOD_DESIGN_KEY,
    validate_executed_method_design,
)
from easyicu.research_agent.reporting.manuscript_post import bind_numeric_values
from easyicu.research_agent.reporting.manuscript_projection import project_owner_issued_manuscript_claims
from tests.support.survival_sealed import run_signed_suite, sealed_survival, synthetic_survival_rows

STEP = "primary_survival_suite"
EVIDENCE = "statistic_step_summary_primary_survival_suite"
DRAFT = (
    "## Abstract\n\n**Results:**\n\n**Conclusions:** Independent validation is required.\n\n"
    "## Results\n\n### Survival results\n\n"
)


@pytest.fixture(scope="module")
def suite_summary(tmp_path_factory):
    root = tmp_path_factory.mktemp("suite")
    _context, authority = sealed_survival(root)
    return json.dumps(run_signed_suite(authority, synthetic_survival_rows(), root / "out"))


def _at_landmark(written: str, landmark_hours: float) -> dict:
    """The summary recorded at ``landmark_hours``: follow-up runs to day 90."""

    summary = json.loads(written)
    design, reported = summary[EXECUTED_METHOD_DESIGN_KEY], summary["reportable_survival_results"]
    horizon_days = design["endpoint_horizon_days"] - landmark_hours / 24.0
    design.update(landmark_hours=landmark_hours, rmst_horizon_days=horizon_days)
    reported["landmark_hours"] = landmark_hours
    reported["rmst"]["tau_days_from_landmark"] = horizon_days
    return summary


def _bound_untraced(tmp_path, summary, text) -> list:
    run_dir = tmp_path / "run"
    source = run_dir / "steps" / STEP / "outputs" / "step_summary.json"
    source.parent.mkdir(parents=True)
    source.write_text(json.dumps(summary), encoding="utf-8")
    store = EvidenceStore(run_dir, enforcement_mode=EvidenceEnforcementMode.STRICT)
    store.register_file(
        kind="statistic", description="Signed survival suite summary", source_path=source,
        evidence_id=EVIDENCE, produced_by_step=STEP, producer="runner",
        generation_mode="deterministic_standard",
    )
    store.register_step_summary_numerics(step_id=STEP, evidence_id=EVIDENCE, summary=summary)
    ledger = [{"step_id": STEP, "status": "ok", "evidence_ids": [EVIDENCE]}]
    _, _, untraced = bind_numeric_values(
        store.bind_manuscript(text, per_step_records=ledger), evidence=store, per_step_records=ledger,
    )
    return untraced


@pytest.mark.parametrize(("landmark", "printed"), [(24.0, "89"), (36.0, "88.5"), (30.0, "88.75"), (48.0, "88")])
def test_results_and_methods_name_the_same_horizon(suite_summary, tmp_path, landmark, printed):
    summary = _at_landmark(suite_summary, landmark)

    projected, _repairs = project_owner_issued_manuscript_claims(
        DRAFT,
        per_step_records=[{
            "step_id": STEP, "step_summary": summary, "generation_mode": "deterministic_standard",
            "step_summary_evidence_id": EVIDENCE,
        }],
    )
    methods = _design_text(validate_executed_method_design(summary[EXECUTED_METHOD_DESIGN_KEY]))

    (sentence,) = [
        line for line in projected.split("\n")
        if line.startswith("The unadjusted restricted mean survival time")
    ][:1]
    assert f"at a horizon of {printed} days was " in sentence
    assert f"over the {printed} days after the landmark" in methods
    assert _bound_untraced(tmp_path, summary, sentence) == []
