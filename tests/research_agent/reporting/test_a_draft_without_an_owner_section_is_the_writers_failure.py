"""A Writer draft without a signed owner's section is the Writer's failure.

A signed owner projects its claims into named targets: the abstract Results
label and its family's Results subsection.  A Writer draft without one of them
raised ``ManuscriptProjectionError``, the error for an invalid owner contract,
so the run stopped with a generic failure before the repair loop or the final
audits ran.  This happens when the Writer leaves a required subsection empty
and its completed draft is kept for the final audits.  The write phase now
collects the absent targets.  The repair loop asks the Writer to restore those
sections, and a final draft that still lacks one fails closed with that cause.
The owner contract still raises when it is invalid.

Synthetic study and seeded synthetic rows only (renal replacement therapy and
90-day mortality).
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from easyicu.research_agent.reporting import manuscript_sections, write_phase
from easyicu.research_agent.reporting.manuscript_projection import with_absent_target_repairs
from tests.support.survival_sealed import run_signed_suite, sealed_survival, synthetic_survival_rows

STEP = "primary_survival_suite"
EVIDENCE = f"statistic_step_summary_{STEP}"
DRAFT = "## Abstract\n\n**Results:** The landmark cohort is described below.\n\n## Results\n\n### Cohort characteristics\n\nText.\n"


@pytest.fixture(scope="module")
def records(tmp_path_factory):
    root = tmp_path_factory.mktemp("suite")
    _context, authority = sealed_survival(root)
    summary = json.loads(json.dumps(run_signed_suite(authority, synthetic_survival_rows(), root / "out")))
    return [{
        "step_id": STEP, "status": "ok", "generation_mode": "deterministic_standard",
        "step_summary": summary, "step_summary_evidence_id": EVIDENCE, "evidence_ids": [EVIDENCE],
    }]


def test_the_draft_stage_collects_the_sections_the_draft_lacks(records):
    findings = []

    projected, absent = write_phase._project_and_report_owner_manuscript_claims(DRAFT, records, findings)

    assert {(item["target_kind"], item["target_label"]) for item in absent} == {
        ("markdown_heading", "Survival results"),
    }
    assert {item["evidence_id"] for item in absent} == {EVIDENCE}
    # The target the draft has still receives its claims.
    assert "{evidence:" + EVIDENCE + "}" in projected.split("## Results")[0]


def test_each_absent_section_becomes_a_repair_of_its_writer_section():
    absent = [
        {"evidence_id": EVIDENCE, "claim_id": "rmst", "target_kind": "markdown_heading", "target_label": "Survival results"},
        {"evidence_id": EVIDENCE, "claim_id": "hr", "target_kind": "markdown_heading", "target_label": "Survival results"},
        {"evidence_id": EVIDENCE, "claim_id": "hr", "target_kind": "abstract_label", "target_label": "Results"},
    ]

    repairs = with_absent_target_repairs({"results": ("- MANUSCRIPT_X: kept",)}, absent)

    assert repairs["results"][0] == "- MANUSCRIPT_X: kept"
    assert repairs["results"][1:] == (
        f"OWNER_CLAIM_TARGET_ABSENT: restore the '### Survival results' subsection; "
        f"the signed owner of {EVIDENCE} places its result claims there.",
    )
    assert repairs["abstract"] == (
        f"OWNER_CLAIM_TARGET_ABSENT: restore the **Results:** label; "
        f"the signed owner of {EVIDENCE} places its result claims there.",
    )


_ABSENT = ({"evidence_id": EVIDENCE, "claim_id": "rmst", "target_kind": "markdown_heading", "target_label": "Survival results"},)


@pytest.mark.parametrize(("second_draft_absent", "fails_closed"), [((), False), (_ABSENT, True)])
def test_the_repair_loop_asks_for_the_section_and_fails_closed_without_it(
    monkeypatch, second_draft_absent, fails_closed,
):
    drafts = iter([_ABSENT, second_draft_absent])
    repair_requests = []

    def draft_manuscript(*_args, section_repair, **_kwargs):
        repair_requests.append(section_repair)
        return SimpleNamespace(
            scaffold="draft", current_evidence_names=(), writer_error_message=None,
            reader_tables=(), absent_owner_claim_targets=next(drafts),
        )

    monkeypatch.setattr(write_phase, "_draft_manuscript", draft_manuscript)
    monkeypatch.setattr(
        write_phase, "_bind_and_review_manuscript",
        lambda *_args, **_kwargs: SimpleNamespace(bound="bound", primary_result_facts=()),
    )
    monkeypatch.setattr(manuscript_sections, "quality_repair_section_errors", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(write_phase, "baseline_reporting_mentions", lambda *_args, **_kwargs: {})
    findings = []

    write_phase._draft_bind_and_repair_manuscript(
        None, context=None, agent_context=None, evidence=None, findings=findings, literature=None,
        per_step_records=(), plan_result=SimpleNamespace(resume_state=None),
        execute_result=SimpleNamespace(plan=SimpleNamespace(display_labels={})), critic=None,
        role_resolver=None, prompt_version="test", runtime_state=None, run_dir=None, run_id="run",
        run_language="en", writer_probe_mode=False, writer_probe_failed_steps=(),
        emit_progress=lambda *_args, **_kwargs: None,
    )

    assert repair_requests[0] is None
    assert list(repair_requests[1][1]) == ["results"]
    assert "'### Survival results'" in repair_requests[1][1]["results"][0]
    reported = [item for item in findings if (item.detail or {}).get("reason_code") == "writer_draft_lacks_owner_claim_target"]
    assert bool(reported) is fails_closed
    if fails_closed:
        assert reported[0].severity == "error"
