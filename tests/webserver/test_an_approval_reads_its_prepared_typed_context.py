"""An approval reads the prepared run's typed context by its version.

A plan step that needs patient groups (the static prediction owner splits
rows by patient) is approved only when the data it was planned on holds the
grouping.  The approval read the run's ``research_context.json`` as the
first context version, but a run planned on prepared data seals a versioned
typed context carrying its materialized inputs, which that version forbids.
Every such context read as unreadable, so a hosted prediction plan whose
prepared data held the verified patient grouping stopped at each approval
with ``research_pipeline_patient_grouping_required``.

The approval now reads the context through the versioned parser.  Synthetic
export and plan; no benchmark item.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from easyicu.research_agent.cohort import materializer as cohort_materializer
from easyicu.research_agent.research_context.builder import build_research_context
from easyicu.research_agent.research_context.typed import (
    RESEARCH_CONTEXT_V3_SCHEMA_VERSION,
)
from easyicu.webserver import run_patient_grouping
from tests.support.typed_export import typed_export

_PREDICTION_PLAN = {
    "steps": [
        {
            "step_id": "primary_performance",
            "intent": "Fit and evaluate the prespecified prediction model.",
            "method": "logistic_prediction",
            "scientific_action_id": "prediction.discrimination_calibration",
            "planned_analysis_role": "primary",
            "inputs": ["lact_first", "age", "death", "cohort:analysis_set"],
            "expected_outputs": ["table:prediction_scores", "table:model_performance"],
        }
    ]
}


def _prepared_context(tmp_path: Path, *, grouped: bool):
    paths = cohort_materializer.materialize_to_parquet(
        tmp_path / "materialized",
        data_path=typed_export(tmp_path / "export"),
        database="miiv",
        static_concepts=("age",),
        feature_concepts=("lact",),
        outcome_concepts=("death",),
    )
    context = build_research_context(
        research_question="Build an in-hospital mortality prediction model.",
        cohort=paths["parquet"],
        cohort_name="prepared",
        database="miiv",
        target_outcome="death",
        id_columns=("stay_id",),
        outcome_columns=("death",),
    )
    assert context.schema_version == RESEARCH_CONTEXT_V3_SCHEMA_VERSION
    if not grouped:
        return context
    provenance = {
        **context.cohort.provenance,
        "replacement_row_identity": {
            "output_identity_column": "patient_stay_id",
            "mapping_file_sha256": "e" * 64,
            "patient_group_derivation": {
                "algorithm": "prefix_before_:s",
                "delimiter": ":s",
            },
        },
    }
    cohort = context.cohort.model_copy(
        update={
            "provenance": provenance,
            "id_columns": [*context.cohort.id_columns, "patient_stay_id"],
        }
    )
    return context.model_copy(update={"cohort": cohort})


def _write(tmp_path: Path, context) -> Path:
    path = tmp_path / "run" / "research_context.json"
    path.parent.mkdir(parents=True)
    path.write_text(context.model_dump_json(), encoding="utf-8")
    return path


def test_prepared_data_that_holds_the_grouping_is_approved(tmp_path: Path) -> None:
    path = _write(tmp_path, _prepared_context(tmp_path, grouped=True))

    run_patient_grouping.require_patient_groups_for_approval(
        _PREDICTION_PLAN, context_path=path
    )


def test_prepared_data_without_the_grouping_stops_as_not_held(tmp_path: Path) -> None:
    path = _write(tmp_path, _prepared_context(tmp_path, grouped=False))

    with pytest.raises(run_patient_grouping.ResearchPipelineRunError) as stopped:
        run_patient_grouping.require_patient_groups_for_approval(
            _PREDICTION_PLAN, context_path=path
        )

    assert stopped.value.code == run_patient_grouping.PATIENT_GROUPING_REQUIRED
    assert stopped.value.details["grouping_status"] == "not_in_planned_data"
