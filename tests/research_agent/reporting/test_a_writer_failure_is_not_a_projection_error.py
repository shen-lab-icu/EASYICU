"""A Writer failure closes the run as not generated, not as a projection error.

When the Writer raises before producing a draft (for example when the model
provider is unreachable), the write phase records the Writer's own failure and
continues with an empty scaffold, so the run ends with the designed
not-generated record. A signed owner that projects its claims into the draft
then found no target in the empty scaffold and raised
``ManuscriptProjectionError``: the run stopped with a generic resume failure,
the Writer's cause was lost, and the paid execution was discarded. Owner
claims are now projected only into a Writer draft; a draft that omits a
projection target still fails closed.

Synthetic study and seeded synthetic rows only (renal replacement therapy and
90-day mortality).
"""

from __future__ import annotations

import json

import pytest

from easyicu.research_agent.reporting import write_phase
from easyicu.research_agent.reporting.manuscript_projection import (
    ManuscriptProjectionError,
    project_owner_issued_manuscript_claims,
)
from tests.support.survival_sealed import run_signed_suite, sealed_survival, synthetic_survival_rows

STEP = "primary_survival_suite"


def _records(tmp_path):
    _context, authority = sealed_survival(tmp_path)
    summary = json.loads(json.dumps(run_signed_suite(authority, synthetic_survival_rows(), tmp_path / "out")))
    return [{
        "step_id": STEP,
        "status": "ok",
        "generation_mode": "deterministic_standard",
        "step_summary": summary,
        "step_summary_evidence_id": f"statistic_step_summary_{STEP}",
        "evidence_ids": [f"statistic_step_summary_{STEP}"],
    }]


def test_the_owner_projects_claims_and_needs_its_targets(tmp_path):
    records = _records(tmp_path)

    with pytest.raises(ManuscriptProjectionError, match="target is absent"):
        project_owner_issued_manuscript_claims("", per_step_records=records)


@pytest.mark.parametrize("scaffold", ["", "\n\n", "   \n"])
def test_no_writer_draft_means_nothing_to_project(tmp_path, scaffold):
    records = _records(tmp_path)
    findings = []

    projected = write_phase._project_and_report_owner_manuscript_claims(scaffold, records, findings)

    assert projected == scaffold
    assert findings == []


def test_a_draft_without_the_target_still_fails_closed(tmp_path):
    records = _records(tmp_path)
    draft = "## Results\n\nThe landmark cohort is described in Table 1.\n"

    with pytest.raises(ManuscriptProjectionError, match="target is absent"):
        write_phase._project_and_report_owner_manuscript_claims(draft, records, [])
