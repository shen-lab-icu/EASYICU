"""The signed survival suite's results reach the manuscript the strict gate admits.

The suite's projection first aimed at subsections a survival plan does not
require.  Once it reached "Survival results", the strict Results grammar still
deleted every projected sentence: in a findings section only host claim
tokens and neutral numeric facts are admitted, and the interval estimates were
interpretive prose whose first sentence carried no citation.  The abstract
Results stayed empty and the Conclusion had no claim to read.

The hazard ratios and the PH decision are now host claims compiled from the
suite's versioned reporting envelope.  Host placement reports them in the
survival results and restores the Conclusion from the primary ones, every
interval when the PH test rejects; the
projection places the primary tokens in the abstract Results with one neutral
restricted-mean sentence.  Every number these sentences print binds to a
value the suite registered.  Synthetic study and seeded synthetic rows only
(renal replacement therapy and 90-day mortality); the second seeded set has
crossing hazards, so the PH test rejects.
"""

from __future__ import annotations

import copy
import json
import re

import pytest

from easyicu.research_agent.authority.evidence_store import (
    EvidenceEnforcementMode,
    EvidenceStore,
)
from easyicu.research_agent.authority.manuscript_claim_policy import (
    expand_scientific_claim_tokens,
    filter_evidence_bound_scaffold,
    place_scientific_claim_tokens_in_results,
)
from easyicu.research_agent.authority.scientific_claims import (
    bind_scientific_claim_drafts,
    derive_scientific_claim_drafts,
)
from easyicu.research_agent.orchestration.scientific_runtime import ScientificRuntimeAuthorities
from easyicu.research_agent.reporting.manuscript_post import bind_numeric_values
from easyicu.research_agent.reporting.manuscript_projection import (
    ManuscriptProjectionError,
    project_owner_issued_manuscript_claims,
)
from easyicu.research_agent.reporting.manuscript_quality import (
    audit_manuscript_quality,
    repair_reader_structure_from_existing_prose,
)
from easyicu.research_agent.reporting.manuscript_result_structure import (
    planned_result_roles,
    required_result_subsections,
    result_section_instruction,
)
from easyicu.research_agent.reporting.manuscript_sections import MANUSCRIPT_SECTION_SPECS
from tests.support.survival_sealed import (
    bound_survival_plan,
    run_signed_suite,
    sealed_survival,
    synthetic_crossing_hazard_rows,
    synthetic_survival_rows,
)

STEP = "primary_survival_suite"
EVIDENCE = "statistic_step_summary_primary_survival_suite"
BLOCKING = {
    "MANUSCRIPT_SUBSECTION_MISSING_OR_EMPTY",
    "MANUSCRIPT_RESULT_SUBSECTION_CALLOUT_ONLY",
    "MANUSCRIPT_ABSTRACT_LABEL_MISSING_OR_EMPTY",
    "MANUSCRIPT_CONCLUSION_WITHOUT_INTERPRETATION",
    "MANUSCRIPT_SECTION_TRUNCATED",
}


def _suite(tmp_path, rows):
    context, authority = sealed_survival(tmp_path)
    plan = bound_survival_plan(context, ScientificRuntimeAuthorities(trajectory=None, current_case=authority))
    # The summary as the evidence store registers it: its JSON bytes.
    summary = json.loads(json.dumps(run_signed_suite(authority, rows, tmp_path / "out")))
    return plan, summary


def _claims(summary):
    return bind_scientific_claim_drafts(
        [draft.model_dump(mode="json") for draft in derive_scientific_claim_drafts(summary)],
        step_id=STEP, evidence_id=EVIDENCE,
    )


def _record(summary) -> dict:
    return {
        "step_id": STEP,
        "step_summary": summary,
        "generation_mode": "deterministic_standard",
        "step_summary_evidence_id": EVIDENCE,
    }


def _strict_store(tmp_path, summary) -> tuple[EvidenceStore, list[dict]]:
    """The summary registered as the host runner registers a signed owner's."""

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
    return store, [{"step_id": STEP, "status": "ok", "evidence_ids": [EVIDENCE]}]


def _writer_draft(plan, summary) -> str:
    """What the strict filter left of a survival Writer's draft without claims.

    The Writer states the landmark cohort; the survival results are empty.
    """

    cohort = (
        f"The cohort comprised {summary['n_landmark_population']} stays {{evidence:{EVIDENCE}}}.\n\n"
    )
    subsections = "".join(
        f"### {heading}\n\n" + (cohort if heading == "Cohort characteristics" else "")
        for heading in required_result_subsections(plan)
    )
    return (
        "# Synthetic survival study\n\n## Abstract\n\n"
        "**Background:** Renal replacement therapy is common in the ICU.\n\n"
        "**Methods:** We fitted a prespecified landmark survival suite.\n\n"
        "**Results:**\n\n**Conclusions:** Independent validation is required.\n\n"
        "## Introduction\n\nThe question is prognostic.\n\n"
        "## Methods\n\n### Study design and cohort\n\nThis was a retrospective cohort study.\n\n"
        "### Variables\n\nRenal replacement therapy was the exposure.\n\n"
        "### Statistical analysis\n\nA prespecified landmark survival suite was fitted.\n\n"
        "### Software and reproducibility\n\nAnalyses used Python.\n\n"
        f"## Results\n\n{subsections}"
        "## Discussion\n\nInterpretation stays in the Discussion.\n\n"
        "## Conclusion\n\nIndependent validation is required.\n"
    )


def _section(text: str, heading: str) -> str:
    body = text.split(f"{heading}\n", 1)[1]
    return re.split(r"\n#{2,3} ", body, maxsplit=1)[0]


def _abstract_results(text: str) -> str:
    return text.split("**Results:**", 1)[1].split("**Conclusions:**", 1)[0]


def _blocking(text: str, plan) -> set[str]:
    audit = audit_manuscript_quality(text, analysis_plan=plan, require_administrative_sections=False)
    return {finding.code for finding in audit.findings} & BLOCKING


@pytest.mark.parametrize(
    ("rows", "rejected"),
    [(synthetic_survival_rows, False), (synthetic_crossing_hazard_rows, True)],
)
def test_the_suite_results_pass_the_strict_gate_into_every_section_they_answer(tmp_path, rows, rejected):
    plan, summary = _suite(tmp_path, rows())
    assert summary["proportional_hazards_status"].startswith("violation_") is rejected
    claims = _claims(summary)
    hazard_ratios = [claim for claim in claims if claim.rule_outcome is None]
    primary = [claim for claim in hazard_ratios if claim.analysis_role == "primary"]
    (ph_rule,) = [claim for claim in claims if claim.rule_outcome is not None]
    intervals = len(summary["reportable_survival_results"]["time_varying_adjusted_association"]["intervals"])
    assert [claim.claim_id for claim in primary] == (
        [f"interval_{position}_adjusted_hazard_ratio" for position in range(1, intervals + 1)]
        if rejected else ["adjusted_hazard_ratio"]
    )
    # The association claims precede the test that chose them.
    assert claims[-1] is ph_rule

    projected, _repairs = project_owner_issued_manuscript_claims(
        _writer_draft(plan, summary), per_step_records=[_record(summary)],
    )
    placed = place_scientific_claim_tokens_in_results(
        projected, claims=claims, planned_step_roles=planned_result_roles(plan),
    )
    repaired, _structure = repair_reader_structure_from_existing_prose(placed.scaffold, claims=claims)
    by_ref = {claim.claim_ref: claim for claim in claims}
    filtered = filter_evidence_bound_scaffold(
        repaired, resolve_claim=by_ref.get, resolve_evidence=lambda ref: ref == EVIDENCE,
    )

    # Nothing the owner or the host placed is deleted by the strict gate.
    assert [
        sentence for sentence in filtered.filtered_sentences
        if "{claim:" in sentence or "restricted mean survival time" in sentence
    ] == []
    assert "restricted mean survival time" in _abstract_results(filtered.scaffold)
    for claim in primary:
        assert claim.placeholder in _abstract_results(filtered.scaffold)
    survival = _section(filtered.scaffold, "### Survival results")
    assert "restricted mean survival time" in survival
    assert all(claim.placeholder in survival for claim in claims)
    # The Conclusion reads every primary association, not the PH rule.
    conclusion = _section(filtered.scaffold, "## Conclusion")
    assert conclusion.split()[: len(primary)] == [claim.placeholder for claim in primary]
    assert ph_rule.placeholder not in conclusion

    expanded = expand_scientific_claim_tokens(filtered.scaffold, resolve_claim=by_ref.get)
    assert expanded.missing_claim_refs == () and expanded.malformed_sentences == ()
    for claim in primary:
        assert claim.render_reader_text(include_estimate=False) in _section(expanded.scaffold, "## Conclusion")
    assert ph_rule.rule_outcome.result_sentence() in _section(expanded.scaffold, "### Survival results")
    assert _blocking(filtered.scaffold, plan) == set()
    assert _blocking(expanded.scaffold, plan) == set()

    # The registered suite issues the same claims, and every number the
    # manuscript now prints binds to a value the suite registered.
    store, ledger = _strict_store(tmp_path, summary)
    assert store.authoritative_scientific_claims(ledger) == claims
    bound = store.bind_manuscript(filtered.scaffold, per_step_records=ledger)
    _, binding, untraced = bind_numeric_values(bound, evidence=store, per_step_records=ledger)
    assert untraced == []
    assert {claim.step_id for claim in binding.values()} == {STEP}


def test_the_writer_is_asked_for_the_sentence_the_suite_projects(tmp_path):
    plan, summary = _suite(tmp_path, synthetic_survival_rows())
    results = next(spec for spec in MANUSCRIPT_SECTION_SPECS if spec.key == "results")
    (rmst,) = [
        claim for claim in summary["reportable_survival_results"]["manuscript_projection"]["claims"]
        if "fragments" in claim
    ]

    # An estimate with its interval: a p value below 0.001 has no display the
    # numeric binder can trace.
    paths = {fragment.get("numeric_path") for fragment in rmst["fragments"]}
    assert {"rmst.ci_low", "rmst.ci_high"} <= paths and "rmst.p_value" not in paths
    for instruction in (result_section_instruction(plan), results.instruction):
        request = instruction.split("reportable_survival_results", 1)[1].split(".", 1)[0]
        assert "confidence interval" in request and "p-value" not in request


def test_an_envelope_signed_before_claims_stays_readable_and_claims_nothing(tmp_path):
    _plan, summary = _suite(tmp_path, synthetic_survival_rows())
    legacy = copy.deepcopy(summary)
    reporting = legacy["reportable_survival_results"]
    reporting["schema_version"] = "easyicu.survival_reporting/1"
    for field in ("exposure", "outcome", "analysis_unit", "landmark_hours", "adjustment_columns",
                  "adjusted_hazard_ratio", "proportional_hazards_test"):
        reporting.pop(field)

    assert derive_scientific_claim_drafts(legacy) == []


def test_a_ph_decision_that_contradicts_its_own_test_is_refused(tmp_path):
    _plan, summary = _suite(tmp_path, synthetic_survival_rows())
    forged = copy.deepcopy(summary)
    forged["reportable_survival_results"]["proportional_hazards_test"]["disposition"] = "assumption_rejected"

    with pytest.raises(ValueError, match="contradicts its own p values"):
        derive_scientific_claim_drafts(forged)


def test_the_projection_refuses_what_the_strict_gate_would_silently_drop(tmp_path):
    plan, summary = _suite(tmp_path, synthetic_survival_rows())
    projection = summary["reportable_survival_results"]["manuscript_projection"]

    unknown = copy.deepcopy(summary)
    unknown["reportable_survival_results"]["manuscript_projection"]["claims"][1][
        "scientific_claim_id"] = "interval_9_adjusted_hazard_ratio"
    with pytest.raises(ManuscriptProjectionError, match="does not compile"):
        project_owner_issued_manuscript_claims(_writer_draft(plan, summary), per_step_records=[_record(unknown)])

    two_sentences = copy.deepcopy(summary)
    fragments = two_sentences["reportable_survival_results"]["manuscript_projection"]["claims"][0]["fragments"]
    fragments.append({"text": " This contrast was retained."})
    with pytest.raises(ManuscriptProjectionError, match="more than one sentence"):
        project_owner_issued_manuscript_claims(_writer_draft(plan, summary), per_step_records=[_record(two_sentences)])
    assert projection["schema_version"] == "easyicu.manuscript_projection/2"
