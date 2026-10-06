"""An export's selection is its executed contracts and its concept population.

A context's inclusion and exclusion criteria are the contracts the source
export applied.  A context written before the Web caller declared only typed
criteria also carries the study's own cohort wording there, and the same
words, verbatim, in ``data_constraints.cohort``.  Nothing executes prose, so
those words select no row.  The criteria are known to be applied, and to be
the whole selection, only when the host records that the export's contract
states it (``data_constraints.source_selection.basis``); otherwise only the
criteria the host applied itself are.  Fixtures are synthetic; the
populations they name vary so that no rule keys on one condition.
"""

from __future__ import annotations

import json

import pytest

from easyicu.research_agent.research_context.concept_population import (
    ConceptCohortWindow,
    ConceptCohortWindowError,
)
from easyicu.research_agent.research_context.export_selection import (
    AppliedContracts,
    export_applied_selection,
)
from easyicu.research_agent.schema import (
    CohortDescriptor,
    ConceptDescriptor,
    ResearchContext,
    UserPreferences,
    VariableRole,
)


_TYPED_INCLUSION = ("age range: 18 to *", "minimum ICU length of stay: 24 hours")
_TYPED_EXCLUSION = (
    "each patient's later ICU stays: the host keeps only the first ICU stay per "
    "patient across the bound source, before planning",
)
_WORDING = [
    {
        "label": "Adults with septic shock",
        "review": "Adult ICU stays with septic shock receiving vasopressors",
    },
    {
        "label": "Older adults with acute kidney injury",
        "review": "Stays aged 65 or older with KDIGO stage 2 or higher",
        "exclusion_statement": "Excluding stays on chronic dialysis",
    },
    {
        "review": "Patients admitted after cardiac surgery",
        "exclusion_statement": "Excluding stays that ended before 24 hours",
    },
]


def _context(
    *,
    inclusion: tuple[str, ...] = (),
    exclusion: tuple[str, ...] = (),
    constraints: dict | None = None,
) -> ResearchContext:
    return ResearchContext(
        research_question="Describe in-hospital mortality.",
        cohort=CohortDescriptor(
            cohort_name="web_study",
            database="miiv",
            n_stays=100,
            inclusion_criteria=list(inclusion),
            exclusion_criteria=list(exclusion),
            id_columns=["stay_id"],
            outcome_columns=["death"],
        ),
        variables=[
            ConceptDescriptor(name="stay_id", dtype="object", role=VariableRole.ID),
            ConceptDescriptor(name="death", dtype="int64", role=VariableRole.OUTCOME),
        ],
        target_outcome="death",
        user_preferences=UserPreferences(
            data_constraints=json.dumps(constraints) if constraints is not None else None
        ),
    )


def test_typed_criteria_are_the_contracts_the_export_applied() -> None:
    selection = export_applied_selection(
        _context(
            inclusion=_TYPED_INCLUSION,
            exclusion=_TYPED_EXCLUSION,
            constraints={"cohort": {"age_min": 18, "exclude_readmissions": True}},
        )
    )

    assert selection.contracts == AppliedContracts(
        inclusion=_TYPED_INCLUSION, exclusion=_TYPED_EXCLUSION
    )
    assert selection.concept_population is None
    assert selection.selects_rows


@pytest.mark.parametrize("wording", _WORDING)
def test_the_studys_own_wording_is_not_a_contract(wording: dict[str, str]) -> None:
    inclusion = tuple(wording[key] for key in ("label", "review") if key in wording)
    exclusion = (wording["exclusion_statement"],) if "exclusion_statement" in wording else ()

    selection = export_applied_selection(
        _context(
            inclusion=(*inclusion, *_TYPED_INCLUSION),
            exclusion=(*exclusion, *_TYPED_EXCLUSION),
            constraints={"cohort": wording},
        )
    )

    assert selection.contracts == AppliedContracts(
        inclusion=_TYPED_INCLUSION, exclusion=_TYPED_EXCLUSION
    )


@pytest.mark.parametrize("wording", _WORDING)
def test_an_export_whose_criteria_are_only_wording_selects_no_row(
    wording: dict[str, str],
) -> None:
    inclusion = tuple(wording[key] for key in ("label", "review") if key in wording)
    exclusion = (wording["exclusion_statement"],) if "exclusion_statement" in wording else ()

    selection = export_applied_selection(
        _context(inclusion=inclusion, exclusion=exclusion, constraints={"cohort": wording})
    )

    assert selection.contracts == AppliedContracts()
    assert not selection.selects_rows


def test_criteria_from_a_caller_that_records_no_wording_stay_contracts() -> None:
    # The CLI declares its criteria as applied and records no study wording.
    selection = export_applied_selection(_context(inclusion=("Adults with septic shock",)))

    assert selection.contracts.inclusion == ("Adults with septic shock",)
    assert selection.selects_rows


def test_only_the_studys_wording_fields_are_wording() -> None:
    selection = export_applied_selection(
        _context(
            inclusion=("sepsis3",),
            constraints={"cohort": {"preset": "sepsis3", "comparison": "sepsis3"}},
        )
    )

    assert selection.contracts.inclusion == ("sepsis3",)


def test_wording_is_compared_without_surrounding_space() -> None:
    selection = export_applied_selection(
        _context(
            inclusion=("  Adults with septic shock ",),
            constraints={"cohort": {"label": "Adults with septic shock\n"}},
        )
    )

    assert selection.contracts == AppliedContracts()


def test_a_concept_population_is_selected_by_the_export() -> None:
    selection = export_applied_selection(
        _context(
            inclusion=("Adults with sepsis",),
            constraints={
                "cohort": {"label": "Adults with sepsis"},
                "concept_cohort_window": {"definition": "sep3", "window_end_hours": 24},
            },
        )
    )

    assert selection.contracts == AppliedContracts()
    assert selection.concept_population == ConceptCohortWindow(
        definition="sep3", window_end_hours=24.0
    )
    assert selection.selects_rows


def test_an_unreadable_concept_record_is_refused() -> None:
    with pytest.raises(ConceptCohortWindowError):
        export_applied_selection(
            _context(constraints={"concept_cohort_window": {"definition": "sep3"}})
        )


def test_an_export_without_criteria_or_concept_selects_no_row() -> None:
    selection = export_applied_selection(_context())

    assert selection.contracts == AppliedContracts()
    assert selection.concept_population is None
    assert not selection.selects_rows


# The host's record of the selection --------------------------------------


@pytest.mark.parametrize(
    ("record", "basis"),
    [
        ({"basis": "export_contract", "host_applied": []}, "export_contract"),
        ({"basis": "package_declaration", "host_applied": []}, "package_declaration"),
        ({"basis": "unrecorded", "host_applied": []}, "unrecorded"),
        # Written before the basis field.
        ({"recorded": True, "host_applied": []}, "export_contract"),
        ({"recorded": True}, "export_contract"),
        ({"recorded": False, "host_applied": []}, "unrecorded"),
        # A record that names no known basis is not trusted.
        ({"basis": "contract"}, "unrecorded"),
        ({"basis": None, "recorded": True}, "unrecorded"),
        ({"basis": "unrecorded", "recorded": True}, "unrecorded"),
        ({"recorded": "true"}, "unrecorded"),
        ({"recorded": 1}, "unrecorded"),
        (["recorded"], "unrecorded"),
        (None, "unrecorded"),
    ],
    ids=[
        "export contract",
        "package declaration",
        "unrecorded",
        "recorded",
        "recorded, no host list",
        "not recorded",
        "an unknown basis",
        "a null basis",
        "the basis over the old field",
        "a string",
        "a number",
        "a list",
        "a null record",
    ],
)
def test_a_selection_is_whole_and_applied_only_on_the_exports_contract(
    record: object, basis: str
) -> None:
    selection = export_applied_selection(
        _context(
            inclusion=_TYPED_INCLUSION,
            exclusion=_TYPED_EXCLUSION,
            constraints={"source_selection": record},
        )
    )

    recorded = basis == "export_contract"
    assert selection.basis == basis
    assert selection.recorded is recorded
    assert selection.contracts == AppliedContracts(
        inclusion=_TYPED_INCLUSION, exclusion=_TYPED_EXCLUSION
    )
    assert selection.known_applied == (selection.contracts if recorded else AppliedContracts())
    assert selection.unverified == (AppliedContracts() if recorded else selection.contracts)


def test_a_context_without_a_record_has_no_basis() -> None:
    # The CLI and a benchmark write no source_selection record.
    selection = export_applied_selection(
        _context(inclusion=_TYPED_INCLUSION, constraints={"cohort": {"age_min": 18}})
    )

    assert selection.basis is None
    assert not selection.recorded
    assert selection.unverified == AppliedContracts(inclusion=_TYPED_INCLUSION)


def test_the_hosts_own_criteria_are_applied_without_a_record() -> None:
    selection = export_applied_selection(
        _context(
            inclusion=_TYPED_INCLUSION,
            exclusion=_TYPED_EXCLUSION,
            constraints={
                "source_selection": {
                    "recorded": False,
                    # Named as the context states them; other names add nothing.
                    "host_applied": [
                        f"  {_TYPED_EXCLUSION[0]}\n",
                        "a criterion no context field states",
                        7,
                    ],
                }
            },
        )
    )

    assert selection.host_applied == AppliedContracts(exclusion=_TYPED_EXCLUSION)
    assert selection.known_applied == AppliedContracts(exclusion=_TYPED_EXCLUSION)
    assert selection.unverified == AppliedContracts(inclusion=_TYPED_INCLUSION)


def test_a_criterion_the_host_names_is_applied_on_the_side_the_context_states_it() -> None:
    selection = export_applied_selection(
        _context(
            inclusion=("first ICU stay per patient", *_TYPED_INCLUSION),
            constraints={"source_selection": {"host_applied": ["first ICU stay per patient"]}},
        )
    )

    assert selection.known_applied == AppliedContracts(inclusion=("first ICU stay per patient",))
    assert selection.unverified == AppliedContracts(inclusion=_TYPED_INCLUSION)


@pytest.mark.parametrize(
    "listed",
    [_TYPED_EXCLUSION[0], {"exclusion": list(_TYPED_EXCLUSION)}, None],
    ids=["a string", "a mapping", "none"],
)
def test_a_host_list_of_another_shape_names_no_criterion(listed: object) -> None:
    selection = export_applied_selection(
        _context(exclusion=_TYPED_EXCLUSION, constraints={"source_selection": {"host_applied": listed}})
    )

    assert selection.host_applied == AppliedContracts()
    assert selection.unverified == AppliedContracts(exclusion=_TYPED_EXCLUSION)


@pytest.mark.parametrize("wording", _WORDING)
def test_the_studys_wording_named_by_the_host_is_still_wording(wording: dict[str, str]) -> None:
    words = next(iter(wording.values()))

    selection = export_applied_selection(
        _context(
            inclusion=(words,),
            constraints={
                "cohort": wording,
                "source_selection": {"recorded": False, "host_applied": [words]},
            },
        )
    )

    assert selection.contracts == AppliedContracts()
    assert selection.host_applied == AppliedContracts()
