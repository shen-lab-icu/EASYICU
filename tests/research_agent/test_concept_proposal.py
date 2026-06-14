"""Tests for the deterministic gatekeeping of AI-proposed concept extensions."""

from __future__ import annotations

import importlib
import inspect

from easyicu.research_agent.concept_proposal import (
    ConceptProposalDraft,
    DistributionStat,
    gather_candidate_rows,
    propose_concept_selection,
    validate_concept_proposal,
)
from easyicu.research_agent.idea_mining_feasibility_tier import SourceItemIndex


def _index() -> SourceItemIndex:
    return SourceItemIndex(
        [
            {
                "itemid": 50954,
                "label": "Lactate Dehydrogenase (LD)",
                "category": "Chemistry",
                "fluid": "Blood",
                "abbrev": "",
                "param_type": "",
                "table": "hosp/labevents",
            },
            {
                "itemid": 50843,
                "label": "Lactate Dehydrogenase, Ascites",
                "category": "Chemistry",
                "fluid": "Ascites",
                "abbrev": "",
                "param_type": "",
                "table": "hosp/labevents",
            },
            {
                "itemid": 221223,
                "label": "EEG",
                "category": "4-Procedures",
                "fluid": "",
                "abbrev": "EEG",
                "param_type": "Processes",
                "unitname": "None",
                "table": "icu/procedureevents",
            },
            {
                "itemid": 220045,
                "label": "Heart Rate",
                "category": "Routine",
                "fluid": "",
                "abbrev": "HR",
                "param_type": "Numeric",
                "unitname": "bpm",
                "table": "icu/chartevents",
            },
        ]
    )


def _probe_factory(stats):
    def probe(itemids, table):
        return {i: stats[i] for i in itemids if i in stats}

    return probe


def test_rejects_invented_itemid_not_in_catalog():
    draft = ConceptProposalDraft("ldh", (99999999,), target_fluid="Blood")
    res = validate_concept_proposal(draft, source_index=_index())
    assert res.blocked
    assert any(f.gate == "catalog_grounding" for f in res.findings)


def test_specimen_gate_drops_wrong_fluid_itemid():
    # blood LDH concept must drop the ascites itemid.
    draft = ConceptProposalDraft(
        "ldh",
        (50954, 50843),
        unit="IU/L",
        min_value=0,
        max_value=5000,
        target_fluid="Blood",
    )
    probe = _probe_factory(
        {50954: DistributionStat(50954, 5000, 4000, 0.4, 50, 200, 900, ("IU/L",))}
    )
    res = validate_concept_proposal(
        draft, source_index=_index(), distribution_probe=probe
    )
    assert 50843 in res.dropped_itemids
    assert res.resolved_itemids == (50954,)
    assert any(f.gate == "specimen_consistency" for f in res.findings)


def test_role_gate_blocks_process_item_declared_as_measurement():
    # EEG is a procedure/event, not a numeric measurement.
    draft = ConceptProposalDraft("qeeg", (221223,), role="measurement")
    res = validate_concept_proposal(draft, source_index=_index())
    assert res.blocked
    assert any(f.gate == "role_measurability" for f in res.findings)


def test_distribution_bounds_gate_blocks_implausible_median():
    # declared mg/dL bounds but the real median is wildly outside → reject.
    draft = ConceptProposalDraft(
        "hr_bad",
        (220045,),
        role="measurement",
        unit="bpm",
        min_value=0,
        max_value=300,
    )
    probe = _probe_factory(
        {220045: DistributionStat(220045, 9000, 8000, 0.9, 40, 9999, 12000, ("bpm",))}
    )
    res = validate_concept_proposal(
        draft, source_index=_index(), distribution_probe=probe
    )
    assert res.blocked
    assert any(f.gate == "distribution_bounds" for f in res.findings)


def test_multi_unit_in_real_data_blocks():
    draft = ConceptProposalDraft(
        "hr",
        (220045,),
        role="measurement",
        unit="bpm",
        min_value=0,
        max_value=300,
    )
    probe = _probe_factory(
        {220045: DistributionStat(220045, 9000, 8000, 0.9, 50, 80, 150, ("bpm", "x"))}
    )
    res = validate_concept_proposal(
        draft, source_index=_index(), distribution_probe=probe
    )
    assert res.blocked
    assert any(
        f.gate == "unit_consistency" and f.severity == "error" for f in res.findings
    )


def test_clean_proposal_reaches_human_review_never_accepted():
    draft = ConceptProposalDraft(
        "hr",
        (220045,),
        role="measurement",
        unit="bpm",
        min_value=0,
        max_value=300,
    )
    probe = _probe_factory(
        {220045: DistributionStat(220045, 9000, 8000, 0.9, 50, 80, 150, ("bpm",))}
    )
    res = validate_concept_proposal(
        draft, source_index=_index(), distribution_probe=probe
    )
    assert res.status == "needs_human_review"
    assert res.status != "accepted"
    assert res.resolved_itemids == (220045,)


def test_no_probe_cannot_be_approved():
    draft = ConceptProposalDraft(
        "hr",
        (220045,),
        role="measurement",
        unit="bpm",
        min_value=0,
        max_value=300,
    )
    res = validate_concept_proposal(draft, source_index=_index())
    # without real-data validation it stays in review with an explicit warning.
    assert res.status == "needs_human_review"
    assert any(f.gate == "distribution" for f in res.findings)


def test_measurement_without_unit_or_bounds_warns():
    # the LDH run exposed this: LLM returned unit=None/bounds=None, so the
    # plausibility gate silently skipped — must surface a warning.
    draft = ConceptProposalDraft(
        "ldh", (50954,), role="measurement", target_fluid="Blood"
    )
    probe = _probe_factory(
        {50954: DistributionStat(50954, 5000, 4000, 0.2, 100, 230, 2300, ("iu/l",))}
    )
    res = validate_concept_proposal(
        draft, source_index=_index(), distribution_probe=probe
    )
    assert res.status == "needs_human_review"
    metadata = [f for f in res.findings if f.gate == "declared_metadata"]
    assert any("unit" in f.message for f in metadata)
    assert any("min, max" in f.message for f in metadata)


def test_draft_is_quarantined_by_default():
    draft = ConceptProposalDraft("hr", (220045,))
    assert draft.quarantine is True


def test_gather_candidate_rows_returns_catalog_rows_for_concept():
    rows = gather_candidate_rows(_index(), "lactate dehydrogenase")
    ids = {r["itemid"] for r in rows}
    assert 50954 in ids and 50843 in ids


def test_selection_constrains_to_offered_itemids_and_drops_invented():
    rows = gather_candidate_rows(_index(), "lactate dehydrogenase")

    def fake_complete(system, user):
        # model tries to smuggle in an itemid (99999) that was not offered.
        return (
            '{"selected_itemids": [50954, 99999], "role": "measurement", '
            '"unit": "IU/L", "min_value": 0, "max_value": 5000, '
            '"target_fluid": "Blood", "rationale": "serum LDH"}'
        )

    draft = propose_concept_selection("ldh", rows, complete=fake_complete)
    assert draft.candidate_itemids == (50954,)
    assert 99999 not in draft.candidate_itemids
    assert draft.role == "measurement"
    assert draft.target_fluid == "Blood"


def test_selection_parses_fenced_json():
    rows = gather_candidate_rows(_index(), "heart rate")

    def fake_complete(system, user):
        return (
            "```json\n"
            '{"selected_itemids": [220045], "role": "measurement", '
            '"unit": "bpm", "min_value": 0, "max_value": 300, '
            '"target_fluid": null, "rationale": "HR"}\n```'
        )

    draft = propose_concept_selection("hr", rows, complete=fake_complete)
    assert draft.candidate_itemids == (220045,)
    assert draft.min_value == 0 and draft.max_value == 300


def test_module_is_a_leaf_does_not_import_idea_mining():
    src = inspect.getsource(
        importlib.import_module("easyicu.research_agent.concept_proposal")
    )
    assert "import idea_mining" not in src
    assert "from .idea_mining import" not in src
