"""The host-verified primary-cohort execution receipt.

The receipt compares two serializations of one locked cohort definition.  These
tests pin the invariant that both sides go through the same owner, so a future
serialization coordinate cannot silently turn every real cohort into a
"mismatch" (which is what killed this path once already).
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import pytest

from easyicu.research_agent.cohort.schema import (
    CohortDefinition,
    ConceptPredicate,
    TimeWindow,
    materialize_locked_analysis_cohort,
)
from easyicu.research_agent.execution import phase as execution_phase
from easyicu.research_agent.execution.phase import (
    _planner_materialized_cohort_execution_receipt,
)
from easyicu.research_agent.intake.materialized_metadata import (
    MaterializedMetadataError,
    materialized_provenance_path,
)
from easyicu.research_agent.planning.cohort_contract import cohort_definition_sha
from easyicu.research_agent.schema import AnalysisPlan

_PREDICATE = ConceptPredicate(
    concept_id="age",
    time_window=TimeWindow(
        anchor="icu_admit",
        start_offset_hours=0,
        end_offset_hours=24,
    ),
    aggregation="first",
    op=">=",
    value=18,
)


def _materialize(tmp_path: Path, definition: CohortDefinition) -> tuple[
    AnalysisPlan, Path, Path
]:
    universe_path = tmp_path / "cohort.parquet"
    pd.DataFrame(
        {"stay_id": [11, 12, 13], "age": [31.0, None, 54.0]}
    ).to_parquet(universe_path, index=False)
    plan = AnalysisPlan(
        research_question="Apply the declared eligibility selection.",
        cohort=definition,
        robustness_specs=[],
        steps=[],
    )
    result = materialize_locked_analysis_cohort(
        run_dir=tmp_path,
        plan=plan,
        universe_path=universe_path,
    )
    assert result["path"] is not None
    return plan, universe_path, Path(result["path"])


def _rewrite_provenance(
    analysis_cohort_path: Path, mutate: Any
) -> None:
    provenance_path = materialized_provenance_path(analysis_cohort_path)
    payload: Dict[str, Any] = json.loads(provenance_path.read_text(encoding="utf-8"))
    mutate(payload)
    provenance_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )


def test_predicate_filtered_cohort_yields_a_receipt(tmp_path: Path) -> None:
    """The default cohort spelling must not read as a plan mismatch."""

    # Regression: ``CohortDefinition.to_dict`` omits a default
    # ``selection_mode`` so legacy authority digests stay stable, while
    # pydantic's ``model_dump`` always emits it.  Comparing those two spellings
    # raised on every predicate-filtered cohort, which killed this receipt for
    # the only branch that then invoked it.
    plan, universe_path, cohort_path = _materialize(
        tmp_path,
        CohortDefinition(name="eligible_stays", inclusion=(_PREDICATE,)),
    )

    receipt = _planner_materialized_cohort_execution_receipt(
        plan=plan,
        universe_path=universe_path,
        analysis_cohort_path=cohort_path,
    )

    assert receipt["raw_universe"]["rows"] == 3
    assert receipt["authoritative_analysis_cohort"]["rows"] == 2
    assert receipt["cohort_definition_sha256"] == cohort_definition_sha(plan.cohort)
    assert [row["predicate_kind"] for row in receipt["ordered_predicate_flow"]] == [
        "universe",
        "inclusion",
    ]


def test_all_input_rows_cohort_yields_a_row_conservation_receipt(
    tmp_path: Path,
) -> None:
    """An explicit all-row selection is a selection, and gets its receipt."""

    plan, universe_path, cohort_path = _materialize(
        tmp_path,
        CohortDefinition(name="every_stay", selection_mode="all_input_rows"),
    )

    receipt = _planner_materialized_cohort_execution_receipt(
        plan=plan,
        universe_path=universe_path,
        analysis_cohort_path=cohort_path,
    )

    assert receipt["raw_universe"]["rows"] == 3
    assert receipt["authoritative_analysis_cohort"]["rows"] == 3
    assert [row["predicate_kind"] for row in receipt["ordered_predicate_flow"]] == [
        "universe"
    ]


def test_execute_phase_asks_for_the_receipt_through_the_selection_owner() -> None:
    """The trigger must use the same helper the cohort layer uses."""

    # ``cohort_definition_has_explicit_selection`` is the single owner of "did
    # this plan make an explicit locked selection".  A local
    # ``inclusion or exclusion`` re-implementation here silently withheld the
    # receipt from every all-row cohort while the cohort layer still
    # materialised one.
    source = inspect.getsource(execution_phase.run_execute_phase)

    assert "primary_cohort_execution_receipt = (" in source
    assert "cohort_definition_has_explicit_selection(" in source
    assert 'getattr(plan.cohort, "inclusion", ())' not in source


@pytest.mark.parametrize(
    ("label", "mutate"),
    [
        (
            "recorded_digest_disagrees_with_the_recorded_definition",
            lambda payload: payload.__setitem__("cohort_sha256", "0" * 64),
        ),
        (
            "recorded_definition_is_not_the_planned_one",
            lambda payload: payload["cohort_definition"].__setitem__(
                "name", "some_other_cohort"
            ),
        ),
        (
            "recorded_definition_gained_an_unplanned_predicate",
            lambda payload: payload["cohort_definition"]["exclusion"].append(
                payload["cohort_definition"]["inclusion"][0]
            ),
        ),
        (
            "recorded_definition_is_missing",
            lambda payload: payload.__setitem__("cohort_definition", None),
        ),
        (
            "recorded_definition_is_unreadable",
            lambda payload: payload.__setitem__(
                "cohort_definition", {"inclusion": "not-a-list"}
            ),
        ),
        (
            "attrition_flow_is_missing",
            lambda payload: payload.__setitem__("cohort_flow", []),
        ),
    ],
)
def test_receipt_fails_closed_on_a_tampered_provenance(
    tmp_path: Path, label: str, mutate: Any
) -> None:
    """Every recorded coordinate the receipt publishes must be checked."""

    plan, universe_path, cohort_path = _materialize(
        tmp_path,
        CohortDefinition(name="eligible_stays", inclusion=(_PREDICATE,)),
    )
    _rewrite_provenance(cohort_path, mutate)

    with pytest.raises(MaterializedMetadataError):
        _planner_materialized_cohort_execution_receipt(
            plan=plan,
            universe_path=universe_path,
            analysis_cohort_path=cohort_path,
        )


def test_receipt_rejects_a_plan_whose_cohort_changed_after_materialisation(
    tmp_path: Path,
) -> None:
    """The active plan, not the recorded one, decides what must be true."""

    plan, universe_path, cohort_path = _materialize(
        tmp_path,
        CohortDefinition(name="eligible_stays", inclusion=(_PREDICATE,)),
    )
    replanned = plan.model_copy(
        update={
            "cohort": CohortDefinition(
                name="eligible_stays",
                inclusion=(_PREDICATE.__class__(**{**_PREDICATE.__dict__, "value": 65}),),
            )
        }
    )

    with pytest.raises(MaterializedMetadataError):
        _planner_materialized_cohort_execution_receipt(
            plan=replanned,
            universe_path=universe_path,
            analysis_cohort_path=cohort_path,
        )
