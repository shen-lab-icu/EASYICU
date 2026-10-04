"""A restored Conclusion states every primary estimate.

When the Writer leaves the Conclusion empty, or with only the validation
caveat, the host restores it from the claim tokens of the plan family's
primary Results subsection.  It took the first token.  When the
proportional-hazards test rejects, every interval's adjusted hazard ratio is
primary, so the tenth H1 manuscript concluded with the first interval alone
(days 0 to 7 after the landmark).  The draft stage now hands the run's claims
to the restore, which states every primary estimate there, in Results order;
the PH decision that chose them is a rule outcome and stays in Results.
Without claims the first token still stands for the answer.

Synthetic study and seeded synthetic rows only (renal replacement therapy and
90-day mortality); the crossing-hazard set rejects the PH test.
"""

from __future__ import annotations

import json
import re
from types import SimpleNamespace

import pytest

from easyicu.research_agent.authority.evidence_store import EvidenceEnforcementMode, EvidenceStore
from easyicu.research_agent.orchestration.scientific_runtime import ScientificRuntimeAuthorities
from easyicu.research_agent.reporting import write_phase
from easyicu.research_agent.reporting.manuscript_projection import project_owner_issued_manuscript_claims
from easyicu.research_agent.reporting.manuscript_quality import repair_reader_structure_from_existing_prose
from easyicu.research_agent.reporting.manuscript_result_structure import required_result_subsections
from tests.support.survival_sealed import (
    bound_survival_plan,
    run_signed_suite,
    sealed_survival,
    synthetic_crossing_hazard_rows,
    synthetic_survival_rows,
)

STEP = "primary_survival_suite"
EVIDENCE = "statistic_step_summary_primary_survival_suite"
LEDGER = [{"step_id": STEP, "status": "ok", "evidence_ids": [EVIDENCE]}]
TOKEN = re.compile(r"\{claim:[^{}\s]+\}")
CAVEAT = "Independent validation is required."


def _registered_suite(tmp_path, rows):
    """The suite's plan and summary, and the STRICT store that registered it."""

    context, authority = sealed_survival(tmp_path)
    plan = bound_survival_plan(context, ScientificRuntimeAuthorities(trajectory=None, current_case=authority))
    summary = json.loads(json.dumps(run_signed_suite(authority, rows, tmp_path / "out")))
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
    return plan, summary, store


def _draft(plan, summary) -> str:
    """A Writer draft that concludes with the caveat only, after the suite's projection."""

    subsections = "".join(f"### {heading}\n\n" for heading in required_result_subsections(plan))
    draft = (
        "# Synthetic survival study\n\n## Abstract\n\n"
        "**Background:** Renal replacement therapy is common in the ICU.\n\n"
        "**Methods:** We fitted a prespecified landmark survival suite.\n\n"
        f"**Results:**\n\n**Conclusions:** {CAVEAT}\n\n"
        "## Introduction\n\nThe question is prognostic.\n\n"
        "## Methods\n\n### Statistical analysis\n\nA prespecified landmark survival suite was fitted.\n\n"
        f"## Results\n\n{subsections}"
        "## Discussion\n\nInterpretation stays in the Discussion.\n\n"
        f"## Conclusion\n\n{CAVEAT}\n"
    )
    record = {
        "step_id": STEP, "step_summary": summary,
        "generation_mode": "deterministic_standard", "step_summary_evidence_id": EVIDENCE,
    }
    projected, _repairs = project_owner_issued_manuscript_claims(draft, per_step_records=[record])
    return projected


def _conclusion(text: str) -> str:
    return text.split("## Conclusion\n", 1)[1]


def _abstract_conclusions(text: str) -> str:
    return text.split("**Conclusions:**", 1)[1].split("\n## ", 1)[0]


def _results(text: str) -> str:
    return text.split("## Results\n", 1)[1].split("\n## ", 1)[0]


@pytest.mark.parametrize(
    ("rows", "rejected"), [(synthetic_survival_rows, False), (synthetic_crossing_hazard_rows, True)],
)
def test_the_draft_stage_concludes_with_every_primary_estimate(tmp_path, rows, rejected):
    plan, summary, store = _registered_suite(tmp_path, rows())
    claims = store.authoritative_scientific_claims(LEDGER)
    estimates = [claim for claim in claims if claim.analysis_role == "primary" and claim.rule_outcome is None]
    (ph_rule,) = [claim for claim in claims if claim.rule_outcome is not None]
    assert (len(estimates) > 1) is rejected
    findings = []

    scaffold = write_phase._place_host_claims_and_restore_structure(
        SimpleNamespace(_evidence_enforcement_mode=EvidenceEnforcementMode.STRICT),
        _draft(plan, summary), evidence=store, per_step_records=LEDGER, plan=plan, findings=findings,
    )

    # Every interval when the test rejects; when it does not, the constant
    # estimate without the prespecified secondary intervals.  The test that
    # chose them stays in Results.
    expected = [claim.placeholder for claim in estimates]
    assert TOKEN.findall(_conclusion(scaffold)) == expected
    assert _conclusion(scaffold).rstrip().endswith(CAVEAT)
    assert TOKEN.findall(_abstract_conclusions(scaffold)) == expected
    assert ph_rule.placeholder in _results(scaffold)
    assert "MANUSCRIPT_CONCLUSION_RESTORED" in {
        repair["code"] for finding in findings for repair in (finding.detail or {}).get("repairs", ())
    }


def test_without_the_run_claims_the_first_token_stands_for_the_answer(tmp_path):
    plan, summary, store = _registered_suite(tmp_path, synthetic_crossing_hazard_rows())
    claims = store.authoritative_scientific_claims(LEDGER)
    placed = write_phase._place_host_claims_in_results(
        _draft(plan, summary), claims=claims, plan=plan, findings=[],
    )

    repaired, _repairs = repair_reader_structure_from_existing_prose(placed, analysis_plan=plan)

    first = next(claim for claim in claims if claim.analysis_role == "primary")
    assert TOKEN.findall(_conclusion(repaired)) == [first.placeholder]
