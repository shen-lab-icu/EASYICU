from __future__ import annotations

import pytest

from easyicu.research_agent.intake.external_benchmark_authority import (
    compile_external_benchmark_study_authority,
)


def test_external_study_authority_compiles_existing_typed_contracts() -> None:
    authority = compile_external_benchmark_study_authority(
        row={
            "endpoint": {
                "name": "event",
                "kind": "binary",
                "absence_semantics": "no_absent_rows",
                "levels": [0, 1],
            },
            "concept_descriptions": {
                "event": "Documented event during the declared follow-up."
            },
            "time_windows": [
                {
                    "name": "followup",
                    "anchor": "icu_admission",
                    "start_hours": 24,
                    "end_hours": 168,
                }
            ],
            "user_preferences": {
                "covariates": ["age"],
                "covariate_selection": "exact",
                "covariate_rationales": {
                    "age": "Age is a prespecified baseline confounder."
                },
                "covariate_temporal_roles": {"age": "baseline_static"},
            },
        },
        target_outcome="event",
        cohort_columns=["stay_id", "followup_hour", "age", "event"],
        id_columns=["stay_id"],
        time_columns=["followup_hour"],
        outcome_columns=["event"],
    )

    assert authority.endpoint is not None
    assert authority.endpoint.name == "event"
    assert authority.user_preferences is not None
    assert authority.user_preferences["covariate_selection"] == "exact"
    assert authority.time_windows[0]["name"] == "followup"
    assert authority.id_columns == ("stay_id",)
    assert authority.time_columns == ("followup_hour",)
    assert authority.outcome_columns == ("event",)


@pytest.mark.parametrize(
    ("row", "message"),
    [
        (
            {
                "endpoint": {
                    "name": "other",
                    "kind": "binary",
                    "absence_semantics": "no_absent_rows",
                    "levels": [0, 1],
                }
            },
            "ENDPOINT_TARGET_MISMATCH.*must match target_outcome",
        ),
        (
            {
                "user_preferences": {
                    "covariates": ["missing_covariate"],
                    "covariate_selection": "exact",
                }
            },
            "exact covariates must be sealed cohort columns",
        ),
        (
            {"concept_descriptions": {"missing_column": "A description."}},
            "concept descriptions require non-empty sealed cohort columns",
        ),
    ],
)
def test_external_study_authority_fails_closed_before_provider(
    row: dict, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        compile_external_benchmark_study_authority(
            row=row,
            target_outcome="event",
            cohort_columns=["event"],
        )


def test_external_study_authority_reports_stable_owner_reason_code() -> None:
    from easyicu.research_agent.intake.external_benchmark_authority import (
        ExternalBenchmarkAuthorityError,
    )

    with pytest.raises(ExternalBenchmarkAuthorityError) as caught:
        compile_external_benchmark_study_authority(
            row={},
            target_outcome="event",
            cohort_columns=["event"],
            time_columns=["missing_time"],
        )

    assert caught.value.reason_code == "ROLE_COLUMN_MISSING"
    assert caught.value.field == "time_columns"
