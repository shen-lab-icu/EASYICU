"""A signed owner's projection target is found in any case.

The quality gate reads abstract labels and required subsections in any case,
so a draft with "**RESULTS:**" or "### Survival Results" has them.  Projection
searched for each target label in its declared case.  It found no target, so
the repair loop asked the Writer to restore a section the draft already had,
and the final draft failed closed for it.  The label now matches in any case.

Synthetic study and seeded synthetic rows only (renal replacement therapy and
90-day mortality).
"""

from __future__ import annotations

import json

import pytest

from easyicu.research_agent.reporting import write_phase
from tests.support.survival_sealed import run_signed_suite, sealed_survival, synthetic_survival_rows

STEP = "primary_survival_suite"
EVIDENCE = f"statistic_step_summary_{STEP}"
TOKEN = "{evidence:" + EVIDENCE + "}"


@pytest.fixture(scope="module")
def records(tmp_path_factory):
    root = tmp_path_factory.mktemp("suite")
    _context, authority = sealed_survival(root)
    summary = json.loads(json.dumps(run_signed_suite(authority, synthetic_survival_rows(), root / "out")))
    return [{
        "step_id": STEP, "status": "ok", "generation_mode": "deterministic_standard",
        "step_summary": summary, "step_summary_evidence_id": EVIDENCE, "evidence_ids": [EVIDENCE],
    }]


@pytest.mark.parametrize(
    ("label", "heading"),
    [("**Results:**", "### Survival results"), ("**RESULTS:**", "### Survival Results")],
    ids=["declared_case", "other_case"],
)
def test_the_owner_targets_are_found_in_any_case(records, label, heading):
    draft = (
        f"## Abstract\n\n{label} The landmark cohort is described below.\n\n"
        f"## Results\n\n### Cohort characteristics\n\nText.\n\n{heading}\n\nText.\n\n"
        "## Discussion\n\nText.\n"
    )

    projected, absent = write_phase._project_and_report_owner_manuscript_claims(draft, records, [])

    assert absent == ()
    abstract, results = projected.split("## Results\n", 1)
    assert TOKEN in abstract
    survival = results.split(heading, 1)[1].split("## Discussion", 1)[0]
    assert TOKEN in survival
    assert TOKEN not in results.split(heading, 1)[0]
