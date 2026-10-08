"""A manuscript can state the stays each selection criterion excluded, from the source on.

A flow diagram starts at the source database: so many ICU stays, so many
excluded by each criterion, so many analyzed.  The export counted its
selection, but nothing read the counts: the Writer saw neither the source's
stays nor any exclusion before the export.  For a recorded selection the
context now holds the export's own report, and
``export_selection_counts`` reads it into steps that must chain from the
source to the analysis input, the host's first-stay restriction included.
The Writer's population block lists them in ICU stays, and the host
registers them as ``research_context`` claims, one per fact, none for a value
two facts share.  Counts that do not chain give none, and why is kept for
audit, never shown to the Writer.  A cap is stated with the rule that chose
the stays it kept.  Fixtures are synthetic.
"""

from __future__ import annotations

import json
from typing import Any, get_args

from easyicu.research_agent.authority.context_numeric_claims import (
    register_context_numeric_claims,
)
from easyicu.research_agent.authority.evidence_store import EvidenceStore
from easyicu.research_agent.reporting.manuscript_post import bind_numeric_values
from easyicu.research_agent.reporting.population_selection import (
    analyzed_population,
    writer_population_block,
)
from easyicu.research_agent.research_context.export_selection import (
    SelectionCountsUnavailable,
    SelectionStep,
    export_selection_counts,
)
from easyicu.research_agent.schema import (
    AnalysisPlan,
    AnalysisStep,
    CohortDescriptor,
    ConceptDescriptor,
    ResearchContext,
    UserPreferences,
    VariableRole,
)

_REASONS = get_args(SelectionCountsUnavailable)


def _step(
    criterion: str, parameters: dict, before: int, excluded: int, missing: int
) -> dict:
    return {
        "criterion": criterion,
        "parameters": parameters,
        "n_before": before,
        "n_excluded": excluded,
        "n_remaining": before - excluded,
        "n_excluded_missing": missing,
    }


def _report(**overrides: Any) -> dict:
    report = {
        "mode": "adult_first",
        "count_unit": "icu_stay",
        "source_total": 1000,
        "selected": 700,
        "selected_before_cap": 700,
        "selected_before_concept_prefilter": 700,
        "demographic_steps": [
            _step("age", {"age_min": 18}, 1000, 120, 4),
            _step("first_icu_stay", {"first_icu_stay": True}, 880, 130, 0),
            _step("los", {"los_min": 24}, 750, 50, 3),
        ],
        "concept_matches": None,
        "selected_before_icd": 700,
        "max_patients_applied": False,
        "applied_filters": ["demographics"],
        "icd": {"enabled": False, "include_tokens": [], "exclude_tokens": []},
    }
    report.update(overrides)
    return report


def _restriction(before: int, after: int) -> dict:
    return {
        "schema_version": "synthetic",
        "stays_before": before,
        "stays_after": after,
        "non_first_icu_stays_removed": before - after,
    }


def _context(
    report: dict | None = None,
    *,
    n_stays: int = 700,
    restriction: dict | None = None,
    basis: str = "export_contract",
) -> ResearchContext:
    selection: dict[str, Any] = {"basis": basis, "host_applied": []}
    if report is not None:
        selection["export_report"] = report
    return ResearchContext(
        research_question="Describe the outcome.",
        cohort=CohortDescriptor(
            cohort_name="web_study",
            database="miiv",
            n_stays=n_stays,
            id_columns=["stay_id"],
            outcome_columns=["death"],
            provenance=(
                {"first_icu_stay_restriction": restriction} if restriction else {}
            ),
        ),
        variables=[
            ConceptDescriptor(name="stay_id", dtype="object", role=VariableRole.ID),
            ConceptDescriptor(name="death", dtype="int64", role=VariableRole.OUTCOME),
        ],
        target_outcome="death",
        user_preferences=UserPreferences(
            data_constraints=json.dumps({"source_selection": selection})
        ),
    )


def _plan() -> AnalysisPlan:
    return AnalysisPlan(
        research_question="Describe the outcome.",
        steps=[
            AnalysisStep(
                step_id="01_primary",
                intent="Describe the prespecified outcome.",
                inputs=["death"],
                expected_outputs=["table:outcome"],
                method="descriptive_summary",
            )
        ],
        cohort={
            "name": "primary",
            "selection_mode": "all_input_rows",
            "inclusion": [],
            "exclusion": [],
        },
    )


def _block(context: ResearchContext) -> str:
    return writer_population_block(analyzed_population(plan=_plan(), context=context))


def _shape(step: SelectionStep) -> tuple:
    return (
        step.stage,
        step.criterion,
        step.n_before,
        step.n_excluded,
        step.n_remaining,
    )


# Reading ---------------------------------------------------------------


def test_the_counts_chain_from_the_source_to_the_analysis_input() -> None:
    reading = export_selection_counts(
        _context(_report(), restriction=_restriction(700, 700))
    )

    counts = reading.counts
    assert reading.unavailable is None
    assert counts is not None
    assert (counts.unit, counts.source_total, counts.exported) == (
        "icu_stay",
        1000,
        700,
    )
    # The host's restriction removed no stay; it still applied, so it stays.
    assert [_shape(step) for step in counts.steps] == [
        ("export", "age", 1000, 120, 880),
        ("export", "first_icu_stay", 880, 130, 750),
        ("export", "los", 750, 50, 700),
        ("host", "first_icu_stay_restriction", 700, 0, 700),
    ]
    assert counts.steps[0].n_excluded_missing == 4
    assert counts.analysis_input == 700


def test_an_older_report_states_its_demographic_criteria_as_one_step() -> None:
    report = _report(
        source_total=1000,
        selected_before_concept_prefilter=750,
        concept_matches=600,
        selected_before_icd=600,
        icd={"enabled": True, "include_tokens": ["J18"], "exclude_tokens": []},
        selected_before_cap=520,
        selected=500,
        max_patients_applied=True,
    )
    del report["demographic_steps"], report["count_unit"]

    counts = export_selection_counts(_context(report, n_stays=500)).counts

    assert counts is not None
    assert [_shape(step) for step in counts.steps] == [
        ("export", "demographics", 1000, 250, 750),
        ("export", "concept_population", 750, 150, 600),
        ("export", "icd", 600, 80, 520),
        ("export", "cap", 520, 20, 500),
    ]
    assert counts.steps[2].parameters == {"include": ["J18"], "exclude": []}
    assert counts.steps[3].parameters == {"rule": "unrecorded"}


def test_an_export_of_every_stay_has_no_export_step() -> None:
    report = {
        "mode": "all_icu",
        "count_unit": "icu_stay",
        "source_total": 900,
        "selected_before_cap": 900,
        "selected": 900,
        "applied_filters": [],
    }

    counts = export_selection_counts(
        _context(report, n_stays=800, restriction=_restriction(900, 800))
    ).counts

    assert counts is not None
    assert counts.exported == 900
    assert [_shape(step) for step in counts.steps] == [
        ("host", "first_icu_stay_restriction", 900, 100, 800)
    ]


def test_rows_sharing_a_stay_give_no_counts() -> None:
    steps = [
        _step("age", {"age_min": 18}, 1001, 120, 4),
        _step("first_icu_stay", {"first_icu_stay": True}, 881, 130, 0),
        _step("los", {"los_min": 24}, 751, 50, 3),
    ]
    # The criteria counted 701 rows; the selection holds 700 distinct stays.
    reading = export_selection_counts(
        _context(_report(source_total=1001, demographic_steps=steps))
    )
    # A concept population read next would take the extra row as its own.
    conceptual = export_selection_counts(
        _context(
            _report(
                source_total=1001,
                demographic_steps=steps,
                concept_matches=650,
                selected_before_icd=650,
                selected_before_cap=650,
                selected=650,
            ),
            n_stays=650,
        )
    )

    assert reading.counts is None
    assert reading.unavailable == "counts_do_not_chain"
    assert (conceptual.counts, conceptual.unavailable) == (None, "counts_do_not_chain")


def test_counts_that_do_not_add_up_give_none() -> None:
    wrong_total = _report(source_total=1001)
    wrong_excluded = _report()
    wrong_excluded["demographic_steps"][1]["n_excluded"] = 131
    gaining = _report()
    gaining["demographic_steps"][2].update(n_excluded=-10, n_remaining=760)
    boolean = _report()
    boolean["demographic_steps"][0]["n_excluded_missing"] = True
    uncapped_loss = _report(selected=690)

    for report in (wrong_total, wrong_excluded, gaining, uncapped_loss):
        assert export_selection_counts(_context(report)).unavailable == (
            "counts_do_not_chain"
        )
    # A step after the demographic criteria cannot add stays either.
    concept_gain = _report(
        concept_matches=710,
        selected_before_icd=710,
        selected_before_cap=710,
        selected=710,
    )
    assert export_selection_counts(_context(concept_gain, n_stays=710)).unavailable == (
        "counts_do_not_chain"
    )
    # A flag is not a count: the missing stays are then not counted.
    assert (
        export_selection_counts(_context(boolean)).counts.steps[0].n_excluded_missing
        is None
    )


def test_counts_that_do_not_reach_the_analysis_input_give_none() -> None:
    assert export_selection_counts(_context(_report(), n_stays=699)).unavailable == (
        "counts_do_not_reach_the_analysis_input"
    )
    assert (
        export_selection_counts(
            _context(_report(), n_stays=600, restriction=_restriction(710, 600))
        ).unavailable
        == "counts_do_not_reach_the_analysis_input"
    )
    unreadable = _restriction(700, 600) | {"non_first_icu_stays_removed": 99}
    assert (
        export_selection_counts(
            _context(_report(), n_stays=600, restriction=unreadable)
        ).unavailable
        == "host_restriction_unreadable"
    )


def test_only_a_recorded_selection_in_icu_stays_has_counts() -> None:
    cases = {
        "selection_not_recorded": _context(_report(), basis="unrecorded"),
        "export_reports_no_counts": _context(None),
        "export_reports_no_source_total": _context(_report(source_total=None)),
        "count_unit_unsupported": _context(_report(count_unit="patient")),
    }

    for reason, context in cases.items():
        reading = export_selection_counts(context)
        assert (reading.counts, reading.unavailable) == (None, reason)


# The Writer's block and the record ---------------------------------------


def test_the_block_lists_the_stays_each_criterion_excluded() -> None:
    block = _block(_context(_report(), restriction=_restriction(700, 700)))

    assert (
        "- Selection counts, in ICU stays, never patients (cite research_context): "
        "the source database held 1,000; "
        "age under 18 years: 120 excluded (4 of them for want of a value), 880 remain; "
        "not the patient's first ICU stay: 130 excluded, 750 remain; "
        "ICU stay shorter than 24 h: 50 excluded (3 of them for want of a value), "
        "700 remain; the export held 700; "
        "not the patient's first ICU stay (restricted by the host): 0 excluded, "
        "700 remain; the analysis input held 700."
    ) in block
    assert "Export cap" not in block


def test_the_record_carries_the_counts_and_the_cap() -> None:
    report = _report(
        selected=600,
        cap={"max_patients": 600, "rule": "identifier_text_order", "cut": True},
    )

    population = analyzed_population(
        plan=_plan(), context=_context(report, n_stays=600)
    )
    record = population.record()

    assert record["export_cap"] == {
        "max_patients": 600,
        "rule": "identifier_text_order",
        "cut": True,
    }
    assert record["selection_counts"]["steps"][-1] == {
        "stage": "export",
        "criterion": "cap",
        "parameters": {
            "max_patients": 600,
            "rule": "identifier_text_order",
            "cut": True,
        },
        "n_before": 700,
        "n_excluded": 100,
        "n_remaining": 600,
        "n_excluded_missing": None,
    }
    assert record["selection_counts"]["analysis_input"] == 600


def test_a_cap_is_stated_with_the_rule_that_chose_its_stays() -> None:
    cut = _report(
        selected=600,
        cap={"max_patients": 600, "rule": "identifier_text_order", "cut": True},
    )
    unknown = {
        "mode": "all_icu",
        "count_unit": "icu_stay",
        "selected": 500,
        "cap": {"max_patients": 500, "rule": "source_file_order", "cut": None},
    }
    sorted_ids = unknown | {"cap": unknown["cap"] | {"rule": "identifier_order"}}
    seeded = unknown | {"cap": unknown["cap"] | {"rule": "seeded_random_sample"}}
    unrecorded = unknown | {"cap": unknown["cap"] | {"rule": "unrecorded"}}
    short = _report(
        cap={"max_patients": 900, "rule": "identifier_text_order", "cut": False}
    )

    assert (
        "- Export cap: the export kept at most 600 stays, the first by identifier "
        "compared as text; they are not a random sample, and the stays kept may "
        "cluster by hospital or period."
    ) in _block(_context(cut, n_stays=600))
    # A full cap with no count of the source still states its rule.
    unknown_block = _block(_context(unknown, n_stays=500))
    assert "Selection counts" not in unknown_block
    assert (
        "the export kept at most 500 stays, the first in the order the source's "
        "stay table lists them; they are not a random sample"
    ) in unknown_block
    assert "the first by identifier; they are not a random sample" in _block(
        _context(sorted_ids, n_stays=500)
    )
    assert "a random sample with a fixed seed." in _block(_context(seeded, n_stays=500))
    # An order nobody recorded says nothing either way about randomness.
    unrecorded_block = _block(_context(unrecorded, n_stays=500))
    assert (
        "- Export cap: the export kept at most 500 stays, chosen in an order the "
        "export did not record, so they cannot be taken as a random sample."
    ) in unrecorded_block
    assert "not a random sample" not in unrecorded_block
    assert "Export cap" not in _block(_context(short))


def test_no_reason_reaches_the_writer() -> None:
    contexts = [
        _context(_report(), basis="unrecorded"),
        _context(None),
        _context(_report(source_total=None)),
        _context(_report(count_unit="patient")),
        _context(_report(source_total=1001)),
        _context(_report(), n_stays=699),
        _context(_report(), n_stays=600, restriction={"stays_before": "x"}),
    ]

    for context in contexts:
        population = analyzed_population(plan=_plan(), context=context)
        assert population.selection_counts is None
        assert population.selection_counts_unavailable in _REASONS
        written = writer_population_block(population) + json.dumps(population.record())
        assert "Selection counts" not in written
        assert not any(reason in written for reason in _REASONS)


# Claims ------------------------------------------------------------------


def _fields(context: ResearchContext, tmp_path) -> dict[str, float]:
    claims = register_context_numeric_claims(EvidenceStore(tmp_path), context=context)
    return {claim.source_field: claim.canonical for claim in claims}


def test_each_count_is_a_claim_once_and_none_the_context_already_holds(
    tmp_path,
) -> None:
    fields = _fields(_context(_report(), restriction=_restriction(700, 700)), tmp_path)

    assert {
        field: value
        for field, value in fields.items()
        if field.startswith("source_selection.")
    } == {
        "source_selection.source_total": 1000,
        "source_selection.1_age.n_excluded": 120,
        "source_selection.1_age.n_excluded_missing": 4,
        "source_selection.1_age.n_remaining": 880,
        "source_selection.2_first_icu_stay.n_excluded": 130,
        "source_selection.2_first_icu_stay.n_remaining": 750,
        "source_selection.3_los.n_excluded": 50,
        "source_selection.3_los.n_excluded_missing": 3,
    }
    # 700 stays are the context's own count; the host step excluded no one.
    assert fields["cohort.n_stays"] == 700


def test_a_value_two_facts_share_is_a_claim_for_neither(tmp_path) -> None:
    steps = [
        _step("age", {"age_min": 18}, 1000, 120, 0),
        _step("first_icu_stay", {"first_icu_stay": True}, 880, 130, 0),
        _step("los", {"los_min": 24}, 750, 630, 0),
    ]
    report = _report(
        demographic_steps=steps,
        selected_before_concept_prefilter=120,
        selected_before_icd=120,
        selected_before_cap=120,
        selected=120,
    )

    fields = _fields(
        _context(report, n_stays=100, restriction=_restriction(120, 100)), tmp_path
    )
    selection = {f for f in fields if f.startswith("source_selection.")}

    # The age step excluded 120 stays and the stay criterion left 120: two facts.
    assert "source_selection.1_age.n_excluded" not in selection
    assert "source_selection.3_los.n_remaining" not in selection
    assert selection == {
        "source_selection.source_total",
        "source_selection.1_age.n_remaining",
        "source_selection.2_first_icu_stay.n_excluded",
        "source_selection.2_first_icu_stay.n_remaining",
        "source_selection.3_los.n_excluded",
        "source_selection.4_first_icu_stay_restriction.n_excluded",
    }


def test_a_step_that_excluded_no_stay_adds_no_fact(tmp_path) -> None:
    steps = [
        _step("age", {"age_min": 18}, 1000, 120, 0),
        _step("first_icu_stay", {"first_icu_stay": True}, 880, 0, 0),
        _step("los", {"los_min": 24}, 880, 180, 0),
    ]

    fields = _fields(_context(_report(demographic_steps=steps)), tmp_path)

    # It left the stays the age step left: one fact, still a claim.
    assert {
        field: value
        for field, value in fields.items()
        if field.startswith("source_selection.")
    } == {
        "source_selection.source_total": 1000,
        "source_selection.1_age.n_excluded": 120,
        "source_selection.1_age.n_remaining": 880,
        "source_selection.3_los.n_excluded": 180,
    }


def test_a_context_without_counts_registers_what_it_did(tmp_path) -> None:
    fields = _fields(_context(_report(source_total=None)), tmp_path)

    assert set(fields) == {"cohort.n_variables", "cohort.n_stays"}


def test_a_sentence_citing_a_count_binds_to_it_and_the_stays_still_bind(
    tmp_path,
) -> None:
    store = EvidenceStore(tmp_path)
    register_context_numeric_claims(
        store, context=_context(_report(), restriction=_restriction(700, 700))
    )
    manuscript = (
        "Of 1,000 ICU stays in the source database, 120 were excluded by the age "
        "criterion. The analysis included 700 ICU stays."
    )

    bound, binding_map, untraced = bind_numeric_values(manuscript, evidence=store)

    assert untraced == []
    assert "<!-- AMBIGUOUS:" not in bound
    assert sorted(claim.source_field for claim in binding_map.values()) == [
        "cohort.n_stays",
        "source_selection.1_age.n_excluded",
        "source_selection.source_total",
    ]
