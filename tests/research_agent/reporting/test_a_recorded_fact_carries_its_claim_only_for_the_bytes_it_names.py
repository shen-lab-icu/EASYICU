"""A host fact stands for the claim it replaces only when the gate can verify it.

The write phase records the host fact sentences beside the bound manuscript.
The claims gate lets a recorded fact carry a claim when the record is bound to
these exact manuscript bytes, the fact cites the claim's evidence at its
current digest, reads verbatim in the Results, and shows every number the
claim states.  Anything else carries nothing, so the gate falls back to the
claim's own sentence.  The study is a synthetic counts-only distribution.
"""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import re

from easyicu.research_agent.authority.evidence_store import EvidenceStore
from easyicu.research_agent.contracts.runtime import ValidationFinding
from easyicu.research_agent.reporting.descriptive_report_facts import (
    compile_counts_only_report_facts,
)
from easyicu.research_agent.reporting.manuscript_gate_state import (
    current_manuscript_completion_state,
    manuscript_result_fact_trace,
)
from easyicu.research_agent.reporting.manuscript_result_facts import (
    RESULT_FACTS_EVIDENCE_ID,
    manuscript_sha256,
    record_result_facts,
    recorded_result_fact_carriage,
    result_facts_payload,
)
from easyicu.research_agent.reporting.readiness import current_validation_findings
from easyicu.research_agent.reporting.write_phase import (
    _result_claim_sufficiency_finding,
)
from easyicu.research_agent.schema import AnalysisPlan

LABELS = {
    "exposure=0": "No Sepsis-3 sepsis",
    "exposure=1": "Sepsis-3 sepsis",
    "outcome": "In-hospital mortality",
}


def _row(level: int, count: int, denominator: int, *, events: bool = False):
    return {
        "level_index": level,
        "level": level,
        "events" if events else "n": count,
        "denominator": denominator,
        "estimate_pct": 100 * count / denominator,
        "interval_method": "none_counts_only",
        "covariance": "none_counts_only",
        "ci_low_pct": None,
        "ci_high_pct": None,
        "confidence_level": None,
        "standard_error_pct": None,
        "cluster_count": None,
    }


def _study(tmp_path: Path):
    summary = {
        "status": "ok",
        "interpretation_class": "exposure_outcome_distribution",
        "analysis_role": "primary",
        "analysis_set": "bound_typed_cohort",
        "interpretation_ceiling": "descriptive_unadjusted_not_causal",
        "adjusted_effect": None,
        "interval_method": "none_counts_only",
        "cohort_n": 3001,
        "exposure": "exposure",
        "outcome": "outcome",
        "descriptive_estimates": {
            "schema_version": "easyicu.exposure_outcome_descriptive_estimates/1",
            "analysis_role": "primary",
            "analysis_set": "bound_typed_cohort",
            "interpretation_ceiling": "descriptive_unadjusted_not_causal",
            "exposure_prevalence": [_row(0, 2470, 3001), _row(1, 531, 3001)],
            "outcome_absolute_risks": [
                _row(0, 247, 2470, events=True),
                _row(1, 71, 531, events=True),
            ],
            "risk_difference": None,
            "dependence": None,
        },
    }
    store = EvidenceStore(tmp_path, enforcement_mode="strict")
    record = store.register_json(
        kind="statistic",
        description="Counts-only distribution",
        payload=summary,
        filename="step_summary.json",
        evidence_id="summary",
        produced_by_step="distribution",
        generation_mode="deterministic_standard",
    )
    store.register_step_summary_numerics(
        step_id="distribution",
        evidence_id=record.evidence_id,
        summary=summary,
    )
    records = [
        {
            "step_id": "distribution",
            "status": "ok",
            "generation_mode": "deterministic_standard",
            "step_summary": summary,
            "step_summary_evidence_id": record.evidence_id,
            "evidence_ids": [record.evidence_id],
        }
    ]
    claims = store.authoritative_scientific_claims(records)
    facts = tuple(
        fact
        for fact in compile_counts_only_report_facts(
            records,
            evidence=store,
            reader_display_labels=LABELS,
            scientific_claims=claims,
        )
        if fact.replaces_claim_ref
    )
    return store, records, claims, facts


def _as_bound(text: str) -> str:
    """The sentence as binding leaves it: footnoted numbers, a resolved citation."""

    footnotes = iter(range(1, 100))
    marked = "".join(
        part
        if part.startswith("“")
        else re.sub(
            r"\d[\d,]*(?:\.\d+)?%?",
            lambda match: f"{match.group()}[^claim_{next(footnotes)}]",
            part,
        )
        for part in re.split(r"(“[^”]*”)", text)
    )
    return f'{marked} [summary](evidence/summary__step_summary.json "sha256=ab12cd34").'


def _manuscript(facts, *, section: str = "Results") -> str:
    sentences = "\n\n".join(_as_bound(fact.text) for fact in facts)
    results = sentences if section == "Results" else "The outcome is described below."
    discussion = sentences if section == "Discussion" else "These are descriptive."
    return (
        "# Mortality by sepsis status\n\n## Results\n\n### Primary outcome\n\n"
        f"{results}\n\n## Discussion\n\n{discussion}\n"
    )


def _carried(tmp_path: Path, store, manuscript: str, claims) -> tuple[dict, dict]:
    carriage = recorded_result_fact_carriage(
        run_dir=tmp_path,
        manuscript_text=manuscript,
        evidence=store,
        claims=claims,
    )
    return dict(carriage.carried), dict(carriage.trace)


def _complete(tmp_path: Path, store, records, manuscript: str) -> bool:
    return current_manuscript_completion_state(
        run_dir=tmp_path,
        manuscript_text=manuscript,
        evidence=store,
        per_step_records=records,
        stop_after_analysis=False,
        writer_probe_mode=False,
        reader_labels=None,
    )["manuscript_result_claims_complete"]


def test_recorded_facts_carry_the_claims_whose_numbers_they_show(tmp_path) -> None:
    store, records, claims, facts = _study(tmp_path)
    manuscript = _manuscript(facts)

    assert not _complete(tmp_path, store, records, manuscript)
    record_result_facts(manuscript, facts, run_dir=tmp_path, evidence=store)

    carried, trace = _carried(tmp_path, store, manuscript, claims)
    assert carried == {claim.claim_ref: index for index, claim in enumerate(claims)}
    assert trace["record"] == "facts_record_read"
    assert [fact["status"] for fact in trace["facts"]] == ["carried", "carried"]
    assert _complete(tmp_path, store, records, manuscript)
    # The claim's 13.370998 percent is shown at two places, as 13.37%.
    assert "(13.37%)" in facts[1].text


def test_a_fact_that_shows_other_numbers_carries_nothing(tmp_path) -> None:
    store, records, claims, facts = _study(tmp_path)
    # The second group's sentence, offered for the first group's claim.
    swapped = (replace(facts[1], replaces_claim_ref=facts[0].replaces_claim_ref),)
    manuscript = _manuscript(swapped)
    record_result_facts(manuscript, swapped, run_dir=tmp_path, evidence=store)

    carried, trace = _carried(tmp_path, store, manuscript, claims)
    assert carried == {}
    [decision] = trace["facts"]
    assert decision["status"] == "fact_numbers_unbound"
    assert {number["reason"] for number in decision["numbers"]} == {
        "fact_number_missing"
    }
    # The write phase applies the same rule, so it reports the claim.
    finding = _result_claim_sufficiency_finding(
        manuscript,
        evidence=store,
        per_step_records=records,
        primary_result_facts=swapped,
        claim_labels={},
    )
    assert finding is not None
    assert set(finding.detail["missing_claim_refs"]) == {
        claim.claim_ref for claim in claims
    }


def test_a_fact_citing_other_evidence_carries_nothing(tmp_path) -> None:
    store, records, claims, facts = _study(tmp_path)
    cited = tuple(replace(fact, evidence_id="other_summary") for fact in facts)
    manuscript = _manuscript(facts)
    record_result_facts(manuscript, cited, run_dir=tmp_path, evidence=store)

    carried, trace = _carried(tmp_path, store, manuscript, claims)
    assert carried == {}
    assert {fact["status"] for fact in trace["facts"]} == {"fact_evidence_mismatch"}


def test_a_fact_from_other_source_bytes_carries_nothing(tmp_path) -> None:
    store, records, claims, facts = _study(tmp_path)
    stale = tuple(replace(fact, source_sha256="b" * 64) for fact in facts)
    manuscript = _manuscript(facts)
    record_result_facts(manuscript, stale, run_dir=tmp_path, evidence=store)

    carried, trace = _carried(tmp_path, store, manuscript, claims)
    assert carried == {}
    assert {fact["status"] for fact in trace["facts"]} == {"fact_source_stale"}


def test_a_record_carries_only_for_the_bytes_it_was_recorded_with(tmp_path) -> None:
    store, records, claims, facts = _study(tmp_path)
    first = _manuscript(facts)
    record_result_facts(first, facts, run_dir=tmp_path, evidence=store)
    second = first.replace("These are descriptive.", "These are descriptive counts.")

    assert _carried(tmp_path, store, second, claims) == (
        {},
        {"record": "facts_record_for_other_manuscript"},
    )
    assert not _complete(tmp_path, store, records, second)
    record_result_facts(second, facts, run_dir=tmp_path, evidence=store)
    assert store.get(RESULT_FACTS_EVIDENCE_ID + "_v2") is not None
    # Either manuscript, restored, still reads its own record.
    for manuscript in (first, second):
        assert len(_carried(tmp_path, store, manuscript, claims)[0]) == 2


def test_a_fact_outside_the_results_carries_nothing(tmp_path) -> None:
    store, records, claims, facts = _study(tmp_path)
    manuscript = _manuscript(facts, section="Discussion")
    record_result_facts(manuscript, facts, run_dir=tmp_path, evidence=store)

    carried, trace = _carried(tmp_path, store, manuscript, claims)
    assert carried == {}
    assert {fact["status"] for fact in trace["facts"]} == {"fact_not_in_results"}


def test_a_fact_for_an_absent_claim_is_recorded_and_ignored(tmp_path) -> None:
    store, records, claims, facts = _study(tmp_path)
    orphan = (replace(facts[0], replaces_claim_ref="distribution.retired_claim"),)
    manuscript = _manuscript(orphan)
    record_result_facts(manuscript, orphan, run_dir=tmp_path, evidence=store)

    carried, trace = _carried(tmp_path, store, manuscript, claims)
    assert carried == {}
    assert trace["facts"] == [
        {
            "fact_index": 0,
            "claim_ref": "distribution.retired_claim",
            "status": "claim_absent",
        }
    ]


def test_an_unverifiable_record_carries_nothing(tmp_path) -> None:
    store, records, claims, facts = _study(tmp_path)
    manuscript = _manuscript(facts)
    digest = manuscript_sha256(manuscript)

    # Another producer's record with the right name is not the host's record.
    store.register_json(
        kind="log",
        description="Not the host record",
        payload={
            **result_facts_payload(facts, manuscript_sha256=digest),
            "note": "copied by another producer",
        },
        filename="facts.json",
        evidence_id=RESULT_FACTS_EVIDENCE_ID,
        producer="writer_agent",
        generation_mode="llm",
        metadata={"source_manuscript_sha256": digest},
    )
    assert _carried(tmp_path, store, manuscript, claims) == (
        {},
        {"record": "facts_record_absent"},
    )

    other_schema = store.register_json(
        kind="log",
        description="A later schema",
        payload={
            **result_facts_payload(facts, manuscript_sha256=digest),
            "schema_version": "easyicu.manuscript_result_facts/2",
        },
        filename="facts_v2.json",
        evidence_id="manuscript_result_facts_json_v7",
        producer="pipeline",
        generation_mode="system",
        metadata={"source_manuscript_sha256": digest},
    )
    carried, trace = _carried(tmp_path, store, manuscript, claims)
    assert (carried, trace["record"]) == ({}, "facts_record_unknown_schema")
    assert trace["evidence_id"] == other_schema.evidence_id

    # A record filed under these bytes whose own content names other bytes.
    store.register_json(
        kind="log",
        description="Facts for another manuscript",
        payload=result_facts_payload(facts, manuscript_sha256="0" * 64),
        filename="facts_other.json",
        evidence_id="manuscript_result_facts_json_v8",
        producer="pipeline",
        generation_mode="system",
        metadata={"source_manuscript_sha256": digest},
    )
    carried, trace = _carried(tmp_path, store, manuscript, claims)
    assert (carried, trace["record"]) == ({}, "facts_record_for_other_manuscript")

    record_result_facts(manuscript, facts, run_dir=tmp_path, evidence=store)
    assert len(_carried(tmp_path, store, manuscript, claims)[0]) == 2
    host_record = store.get(RESULT_FACTS_EVIDENCE_ID + "_v2")
    target = tmp_path / host_record.relative_path
    tampered = json.loads(target.read_text(encoding="utf-8"))
    tampered["facts"][0]["text"] = tampered["facts"][0]["text"].replace("247", "248")
    target.write_text(json.dumps(tampered), encoding="utf-8")
    carried, trace = _carried(tmp_path, store, manuscript, claims)
    assert (carried, trace["record"]) == ({}, "facts_record_unverified")


def test_readiness_retires_a_sufficiency_error_the_recorded_facts_answer(
    tmp_path,
) -> None:
    store, records, claims, facts = _study(tmp_path)
    manuscript = _manuscript(facts)
    plan = AnalysisPlan(
        research_question="How often do patients die, by sepsis?", steps=[]
    )
    earlier = ValidationFinding(
        validator="manuscript_result_sufficiency",
        severity="error",
        message=(
            "Final evidence/numeric filtering removed or failed to bind "
            "host-authorized scientific claim(s) from the Results section."
        ),
        detail={"missing_claim_refs": [claim.claim_ref for claim in claims]},
    )

    def partition():
        active, superseded, _ = current_validation_findings(
            plan=plan,
            per_step_records=records,
            findings=[earlier],
            evidence=store,
            run_dir=tmp_path,
            manuscript_text=manuscript,
            context=None,
        )
        return active, superseded

    assert partition() == ([earlier], [])
    record_result_facts(manuscript, facts, run_dir=tmp_path, evidence=store)
    assert partition() == ([], [earlier])
    assert manuscript_result_fact_trace(
        run_dir=tmp_path,
        manuscript_text=manuscript,
        evidence=store,
        per_step_records=records,
    )["carried"] == {claim.claim_ref: index for index, claim in enumerate(claims)}
