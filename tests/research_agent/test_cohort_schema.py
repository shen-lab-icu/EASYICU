"""CTAS cohort time-aggregation schema tests."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest


def _age_predicate(start: float, end: float):
    from easyicu.research_agent.cohort_schema import ConceptPredicate, TimeWindow

    return ConceptPredicate(
        concept_id="age",
        time_window=TimeWindow(
            anchor="icu_admit",
            start_offset_hours=start,
            end_offset_hours=end,
        ),
        aggregation="max",
        op=">=",
        value=18,
    )


def test_concept_predicate_rejects_missing_time_window() -> None:
    from easyicu.research_agent.cohort_schema import ConceptPredicate, CohortSchemaError

    with pytest.raises(CohortSchemaError, match="time_window"):
        ConceptPredicate.from_dict({"concept_id": "age", "aggregation": "max", "op": ">="})


def test_concept_predicate_rejects_missing_aggregation() -> None:
    from easyicu.research_agent.cohort_schema import ConceptPredicate, CohortSchemaError

    with pytest.raises(CohortSchemaError, match="aggregation"):
        ConceptPredicate.from_dict(
            {
                "concept_id": "age",
                "time_window": {
                    "anchor": "icu_admit",
                    "start_offset_hours": 0,
                    "end_offset_hours": 24,
                },
                "op": ">=",
            }
        )


def test_aggregation_op_incompatibility_rejected() -> None:
    from easyicu.research_agent.cohort_schema import (
        CohortSchemaError,
        ConceptPredicate,
        TimeWindow,
    )

    with pytest.raises(CohortSchemaError, match="only supports"):
        ConceptPredicate(
            concept_id="mech_vent",
            time_window=TimeWindow("icu_admit", 0, 24),
            aggregation="any",
            op=">=",
            value=1,
        )


def test_unknown_concept_id_rejected() -> None:
    from easyicu.research_agent.cohort_schema import (
        CohortSchemaError,
        ConceptPredicate,
        TimeWindow,
    )

    with pytest.raises(CohortSchemaError, match="unknown concept_id"):
        ConceptPredicate(
            concept_id="not_a_real_easyicu_concept",
            time_window=TimeWindow("icu_admit", 0, 24),
            aggregation="max",
            op=">=",
            value=1,
        )


def test_time_window_accepts_case_owned_anchor_string() -> None:
    from easyicu.research_agent.cohort_schema import TimeWindow, UNIVERSAL_ANCHORS

    window = TimeWindow("delirium_onset", 0, 24)

    assert window.anchor == "delirium_onset"
    assert "delirium_onset" not in UNIVERSAL_ANCHORS


def test_registered_pattern_expansion_is_deterministic() -> None:
    from easyicu.research_agent.cohort_schema import (
        CohortDefinition,
        expand_named_cohort,
        register_pattern,
        reset_pattern_registry,
    )

    reset_pattern_registry()
    try:
        register_pattern(
            "adult_admission_window",
            CohortDefinition(name="adult", inclusion=(_age_predicate(0, 24),)),
            provenance="test fixture",
        )
        first = expand_named_cohort("adult_admission_window").to_dict()
        second = expand_named_cohort("adult_admission_window").to_dict()
        assert first == second
        assert first["derived_from_named"] == "adult_admission_window"
    finally:
        reset_pattern_registry()


def test_two_registered_patterns_with_different_windows_have_different_hash() -> None:
    from easyicu.research_agent.cohort_schema import (
        CohortDefinition,
        cohort_definition_sha,
        expand_named_cohort,
        register_pattern,
        reset_pattern_registry,
    )

    reset_pattern_registry()
    try:
        register_pattern(
            "adult_first_day",
            CohortDefinition(name="adult_first_day", inclusion=(_age_predicate(0, 24),)),
            provenance="test fixture",
        )
        register_pattern(
            "adult_first_hour",
            CohortDefinition(name="adult_first_hour", inclusion=(_age_predicate(0, 1),)),
            provenance="test fixture",
        )
        assert cohort_definition_sha(
            expand_named_cohort("adult_first_day")
        ) != cohort_definition_sha(expand_named_cohort("adult_first_hour"))
    finally:
        reset_pattern_registry()


def test_unknown_named_pattern_rejected() -> None:
    from easyicu.research_agent.cohort_schema import reset_pattern_registry

    from easyicu.research_agent.schema import AnalysisPlan

    reset_pattern_registry()
    with pytest.raises(ValueError, match="unknown named cohort pattern"):
        AnalysisPlan(
            research_question="Does a predictor associate with mortality?",
            cohort={"from_named": "case_specific_pattern_not_registered"},
            steps=[],
        )


def test_planner_string_cohort_rejected() -> None:
    from easyicu.research_agent.schema import AnalysisPlan

    with pytest.raises(ValueError, match="free-text cohort strings"):
        AnalysisPlan(
            research_question="Does a predictor associate with mortality?",
            cohort="SOFA-2 = 0 patients",
            steps=[],
        )


def test_planner_named_cohort_accepted_and_expanded() -> None:
    from easyicu.research_agent.cohort_schema import (
        CohortDefinition,
        register_pattern,
        reset_pattern_registry,
    )
    from easyicu.research_agent.schema import AnalysisPlan

    reset_pattern_registry()
    try:
        register_pattern(
            "adult_admission_window",
            CohortDefinition(name="adult", inclusion=(_age_predicate(0, 24),)),
            provenance="test fixture",
        )
        plan = AnalysisPlan(
            research_question="Does a predictor associate with mortality?",
            cohort={"from_named": "adult_admission_window"},
            steps=[],
        )
        assert plan.cohort is not None
        assert plan.cohort.derived_from_named == "adult_admission_window"
        assert plan.cohort.inclusion[0].time_window.end_offset_hours == 24
    finally:
        reset_pattern_registry()


def test_robustness_spec_cohort_override_schema_validated() -> None:
    from easyicu.research_agent.cohort_schema import CohortSchemaError
    from easyicu.research_agent.robustness_panel import RobustnessSpec

    with pytest.raises(CohortSchemaError, match="unknown concept_id"):
        RobustnessSpec.from_dict(
            {
                "spec_id": "bad_cohort",
                "axis": "cohort",
                "description": "Invalid concept.",
                "cohort_override": {
                    "name": "bad",
                    "inclusion": [
                        {
                            "concept_id": "not_a_real_easyicu_concept",
                            "time_window": {
                                "anchor": "icu_admit",
                                "start_offset_hours": 0,
                                "end_offset_hours": 24,
                            },
                            "aggregation": "max",
                            "op": ">=",
                            "value": 1,
                        }
                    ],
                    "exclusion": [],
                },
            }
        )


def test_cohort_locked_recorded_in_manifest(ra, tmp_path: Path) -> None:
    from easyicu.research_agent.cohort_schema import (
        CohortDefinition,
        COHORT_LOCK_FILENAME,
        write_locked_cohort_definition,
    )
    from easyicu.research_agent.schema import AnalysisManifest, AnalysisPlan

    plan = AnalysisPlan(
        research_question="Does a predictor associate with mortality?",
        cohort=CohortDefinition(name="primary", inclusion=(_age_predicate(0, 24),)),
        steps=[],
    )
    evidence = ra.EvidenceStore(tmp_path)
    path = write_locked_cohort_definition(
        run_dir=tmp_path,
        plan=plan,
        evidence=evidence,
        prompt_pack_version="test",
        llm_signature="mock",
    )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = AnalysisManifest(
        run_id="r1",
        research_question=plan.research_question,
        started_at=datetime.now(timezone.utc),
        context_path="context.json",
        cohort_locked_path=COHORT_LOCK_FILENAME,
        cohort_locked_sha=digest,
    )
    assert manifest.cohort_locked_path == COHORT_LOCK_FILENAME
    assert manifest.cohort_locked_sha == digest


def test_cohort_lock_reuses_existing_bytes_on_resume(ra, tmp_path: Path) -> None:
    from easyicu.research_agent.cohort_schema import (
        CohortDefinition,
        write_locked_cohort_definition,
    )
    from easyicu.research_agent.schema import AnalysisPlan

    plan = AnalysisPlan(
        research_question="Does a predictor associate with an outcome?",
        cohort=CohortDefinition(name="primary", inclusion=(_age_predicate(0, 24),)),
        steps=[],
    )
    evidence = ra.EvidenceStore(tmp_path)
    path = write_locked_cohort_definition(
        run_dir=tmp_path,
        plan=plan,
        evidence=evidence,
        prompt_pack_version="test",
        llm_signature="mock",
    )
    before = path.read_bytes()

    reused = write_locked_cohort_definition(
        run_dir=tmp_path,
        plan=plan,
        evidence=evidence,
        prompt_pack_version="test",
        llm_signature="mock",
    )

    assert reused == path
    assert path.read_bytes() == before


def test_cohort_lock_resume_rehydrates_only_legacy_timestamp_drift(
    ra,
    tmp_path: Path,
) -> None:
    from easyicu.research_agent.cohort_schema import (
        CohortDefinition,
        write_locked_cohort_definition,
    )
    from easyicu.research_agent.schema import AnalysisPlan

    plan = AnalysisPlan(
        research_question="Does a predictor associate with an outcome?",
        cohort=CohortDefinition(name="primary", inclusion=(_age_predicate(0, 24),)),
        steps=[],
    )
    evidence = ra.EvidenceStore(tmp_path)
    path = write_locked_cohort_definition(
        run_dir=tmp_path,
        plan=plan,
        evidence=evidence,
        prompt_pack_version="test",
        llm_signature="mock",
    )
    anchored = path.read_bytes()
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["locked_at"] = "2099-01-01T00:00:00+00:00"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    write_locked_cohort_definition(
        run_dir=tmp_path,
        plan=plan,
        evidence=evidence,
        prompt_pack_version="test",
        llm_signature="mock",
    )

    assert path.read_bytes() == anchored
    repair = evidence.get("cohort_lock_resume_rehydration")
    assert repair is not None
    assert repair.metadata["llm_signature"] == "mock"


def test_cohort_lock_resume_does_not_rehydrate_scientific_drift(
    ra,
    tmp_path: Path,
) -> None:
    from easyicu.research_agent.cohort_schema import (
        CohortDefinition,
        CohortSchemaError,
        coerce_cohort_definition,
        cohort_definition_sha,
        write_locked_cohort_definition,
    )
    from easyicu.research_agent.schema import AnalysisPlan

    plan = AnalysisPlan(
        research_question="Does a predictor associate with an outcome?",
        cohort=CohortDefinition(name="primary", inclusion=(_age_predicate(0, 24),)),
        steps=[],
    )
    evidence = ra.EvidenceStore(tmp_path)
    path = write_locked_cohort_definition(
        run_dir=tmp_path,
        plan=plan,
        evidence=evidence,
        prompt_pack_version="test",
        llm_signature="mock",
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["cohort"]["inclusion"][0]["value"] = 65
    changed = coerce_cohort_definition(payload["cohort"])
    assert changed is not None
    payload["cohort_sha256"] = cohort_definition_sha(changed)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    with pytest.raises(CohortSchemaError, match="plan-time evidence anchor"):
        write_locked_cohort_definition(
            run_dir=tmp_path,
            plan=plan,
            evidence=evidence,
            prompt_pack_version="test",
            llm_signature="mock",
        )
    assert json.loads(path.read_text(encoding="utf-8"))["cohort"]["inclusion"][0][
        "value"
    ] == 65


def test_assert_cohort_definition_locked_catches_post_lock_mutation(
    ra,
    tmp_path: Path,
) -> None:
    from easyicu.research_agent.cohort_schema import (
        CohortDefinition,
        CohortSchemaError,
        assert_cohort_definition_locked,
        write_locked_cohort_definition,
    )
    from easyicu.research_agent.schema import AnalysisPlan

    plan = AnalysisPlan(
        research_question="Does a predictor associate with mortality?",
        cohort=CohortDefinition(name="primary", inclusion=(_age_predicate(0, 24),)),
        steps=[],
    )
    evidence = ra.EvidenceStore(tmp_path)
    write_locked_cohort_definition(
        run_dir=tmp_path,
        plan=plan,
        evidence=evidence,
        prompt_pack_version="test",
        llm_signature="mock",
    )

    mutated_plan = plan.model_copy(
        update={
            "cohort": CohortDefinition(
                name="primary",
                inclusion=(_age_predicate(0, 1),),
            )
        }
    )
    with pytest.raises(CohortSchemaError, match="changed after plan lock"):
        assert_cohort_definition_locked(run_dir=tmp_path, plan=mutated_plan)


def test_builder_rejects_unknown_aggregation_with_not_implemented() -> None:
    from easyicu.research_agent.cohort_schema import (
        CohortDefinition,
        ConceptPredicate,
        TimeWindow,
        build_cohort,
    )

    definition = CohortDefinition(
        name="unknown_age_aggregation",
        inclusion=(
            ConceptPredicate(
                "age",
                TimeWindow("icu_admit", 0, 24),
                "mode",
                ">=",
                18,
            ),
        ),
    )
    with pytest.raises(NotImplementedError, match="aggregation 'mode'"):
        build_cohort(definition, pd.DataFrame({"age": [21, 17]}))


def test_builder_missing_materialised_column_is_data_error() -> None:
    from easyicu.research_agent.cohort_schema import (
        CohortDataError,
        CohortDefinition,
        build_cohort,
    )

    definition = CohortDefinition(
        name="adult",
        inclusion=(_age_predicate(0, 24),),
    )
    with pytest.raises(CohortDataError, match="missing concept column 'age'"):
        build_cohort(definition, pd.DataFrame({"other_column": [1, 2]}))


def _plan_with_cohort(definition):
    import types

    return types.SimpleNamespace(cohort=definition)


def test_materialize_locked_analysis_cohort_applies_inclusion(tmp_path: Path) -> None:
    """The locked definition must be materialised into a filtered analysis
    cohort parquet — the bridge that enforces 纳排 on the data steps read."""
    from easyicu.research_agent.cohort_schema import (
        CohortDefinition,
        materialize_locked_analysis_cohort,
    )

    universe = pd.DataFrame({"age": [10, 18, 40, 70], "los_icu": [5, 2, 0.5, 3]})
    universe_path = tmp_path / "cohort.parquet"
    universe.to_parquet(universe_path, index=False)

    definition = CohortDefinition(
        name="adult_los1",
        inclusion=(_age_predicate(0, 24),),  # age >= 18 (max)
    )
    result = materialize_locked_analysis_cohort(
        run_dir=tmp_path, plan=_plan_with_cohort(definition), universe_path=universe_path
    )
    assert result["status"] == "applied"
    assert result["n_universe"] == 4
    assert result["n_cohort"] == 3  # drops the age-10 stay
    out = tmp_path / "cohort_analysis.parquet"
    assert out.exists()
    assert len(pd.read_parquet(out)) == 3
    assert (tmp_path / "cohort_analysis_provenance.json").exists()


def test_materialize_no_definition_returns_no_file(tmp_path: Path) -> None:
    from easyicu.research_agent.cohort_schema import (
        CohortDefinition,
        materialize_locked_analysis_cohort,
    )

    universe = pd.DataFrame({"age": [20, 30]})
    universe_path = tmp_path / "cohort.parquet"
    universe.to_parquet(universe_path, index=False)

    result = materialize_locked_analysis_cohort(
        run_dir=tmp_path,
        plan=_plan_with_cohort(CohortDefinition(name="empty")),  # no predicates
        universe_path=universe_path,
    )
    assert result["status"] == "no_definition"
    assert not (tmp_path / "cohort_analysis.parquet").exists()


def test_materialize_missing_column_falls_back_without_breaking(tmp_path: Path) -> None:
    """A predicate the universe cannot satisfy must not break the run: status
    'error', no parquet, caller falls back to the universe."""
    from easyicu.research_agent.cohort_schema import (
        CohortDefinition,
        materialize_locked_analysis_cohort,
    )

    universe = pd.DataFrame({"other_column": [1, 2]})  # no 'age'
    universe_path = tmp_path / "cohort.parquet"
    universe.to_parquet(universe_path, index=False)

    result = materialize_locked_analysis_cohort(
        run_dir=tmp_path,
        plan=_plan_with_cohort(
            CohortDefinition(name="adult", inclusion=(_age_predicate(0, 24),))
        ),
        universe_path=universe_path,
    )
    assert result["status"] == "error"
    assert result["error"]
    assert not (tmp_path / "cohort_analysis.parquet").exists()


def _kdigo_predicate():
    """A predicate referencing the dictionary concept_id `kdigo_aki` — whose
    EasyICU output column is `aki_stage`, so it appears in a wide universe as
    `aki_stage_<agg>`, never as `kdigo_aki`."""
    from easyicu.research_agent.cohort_schema import ConceptPredicate, TimeWindow

    return ConceptPredicate(
        concept_id="kdigo_aki",
        time_window=TimeWindow(
            anchor="icu_admit", start_offset_hours=0.0, end_offset_hours=24.0
        ),
        aggregation="max",
        op=">=",
        value=0,
    )


def test_resolve_predicate_column_bare_and_aggregated_and_alias() -> None:
    from easyicu.research_agent.cohort_schema import _resolve_predicate_column

    cols = ["age", "aki_stage_max", "aki_stage_first", "los_icu", "death"]
    # bare id-level column
    assert _resolve_predicate_column(cols, "age", "first") == "age"
    # dictionary concept_id resolves to its output-column alias + aggregation
    assert _resolve_predicate_column(cols, "kdigo_aki", "max") == "aki_stage_max"
    # wide <concept_id>_<agg> form when the output stem equals the concept id
    assert (
        _resolve_predicate_column(["sofa_resp_max"], "sofa_resp", "max")
        == "sofa_resp_max"
    )
    # genuinely-absent column is not invented (honest failure, not silent skip)
    assert _resolve_predicate_column(cols, "lactate", "max") is None
    # the requested aggregation must exist: only `_first` present, asked `_max`
    assert _resolve_predicate_column(["aki_stage_first"], "kdigo_aki", "max") is None


def test_materialize_resolves_kdigo_alias_to_aki_stage_column(tmp_path: Path) -> None:
    """E3 regression: the locked cohort references concept_id `kdigo_aki`, but the
    universe materialised the concept as `aki_stage_*`. The materializer must
    bridge the concept-id -> output-column gap so the 纳排 is enforced centrally
    (cohort_analysis.parquet written) instead of silently running on the universe."""
    from easyicu.research_agent.cohort_schema import (
        CohortDefinition,
        materialize_locked_analysis_cohort,
    )

    universe = pd.DataFrame(
        {
            "stay_id": [1, 2, 3, 4],
            "age": [20, 65, 40, 17],  # last is a minor -> excluded
            "aki_stage_max": [0, 2, None, 1],  # NaN -> unmeasured, excluded by >=0
        }
    )
    universe_path = tmp_path / "cohort.parquet"
    universe.to_parquet(universe_path, index=False)

    result = materialize_locked_analysis_cohort(
        run_dir=tmp_path,
        plan=_plan_with_cohort(
            CohortDefinition(
                name="primary",
                inclusion=(_age_predicate(0, 24), _kdigo_predicate()),
            )
        ),
        universe_path=universe_path,
    )

    assert result["status"] == "applied"
    assert (tmp_path / "cohort_analysis.parquet").exists()
    # adults (age>=18) with a measured KDIGO stage (aki_stage_max>=0, NaN dropped)
    assert result["n_cohort"] == 2
