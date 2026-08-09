"""CTAS cohort time-aggregation schema tests."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from easyicu.research_agent.planning import cohort_contract


def _age_predicate(start: float, end: float):
    from easyicu.research_agent.cohort.schema import ConceptPredicate, TimeWindow

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
    from easyicu.research_agent.cohort.schema import ConceptPredicate, CohortSchemaError

    with pytest.raises(CohortSchemaError, match="time_window"):
        ConceptPredicate.from_dict(
            {"concept_id": "age", "aggregation": "max", "op": ">="}
        )


def test_concept_predicate_rejects_missing_aggregation() -> None:
    from easyicu.research_agent.cohort.schema import ConceptPredicate, CohortSchemaError

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
    from easyicu.research_agent.cohort.schema import (
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
    from easyicu.research_agent.cohort.schema import (
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
    from easyicu.research_agent.cohort.schema import TimeWindow, UNIVERSAL_ANCHORS

    window = TimeWindow("delirium_onset", 0, 24)

    assert window.anchor == "delirium_onset"
    assert "delirium_onset" not in UNIVERSAL_ANCHORS


def test_registered_pattern_expansion_is_deterministic() -> None:
    from easyicu.research_agent.cohort.schema import (
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
    from easyicu.research_agent.cohort.schema import (
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
            CohortDefinition(
                name="adult_first_day", inclusion=(_age_predicate(0, 24),)
            ),
            provenance="test fixture",
        )
        register_pattern(
            "adult_first_hour",
            CohortDefinition(
                name="adult_first_hour", inclusion=(_age_predicate(0, 1),)
            ),
            provenance="test fixture",
        )
        assert cohort_definition_sha(
            expand_named_cohort("adult_first_day")
        ) != cohort_definition_sha(expand_named_cohort("adult_first_hour"))
    finally:
        reset_pattern_registry()


def test_unknown_named_pattern_rejected() -> None:
    from easyicu.research_agent.cohort.schema import reset_pattern_registry

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
    from easyicu.research_agent.cohort.schema import (
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
    from easyicu.research_agent.cohort.schema import CohortSchemaError
    from easyicu.research_agent.robustness.panel import RobustnessSpec

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
    from easyicu.research_agent.cohort.schema import (
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
    from easyicu.research_agent.cohort.schema import (
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


def test_cohort_lock_resume_rejects_timestamp_drift(
    ra,
    tmp_path: Path,
) -> None:
    from easyicu.research_agent.cohort.schema import (
        CohortDefinition,
        CohortSchemaError,
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
    payload["locked_at"] = "2099-01-01T00:00:00+00:00"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    mutated = path.read_bytes()

    with pytest.raises(CohortSchemaError, match="plan-time evidence anchor"):
        write_locked_cohort_definition(
            run_dir=tmp_path,
            plan=plan,
            evidence=evidence,
            prompt_pack_version="test",
            llm_signature="mock",
        )

    assert path.read_bytes() == mutated
    assert evidence.get("cohort_lock_resume_rehydration") is None


def test_cohort_lock_rejects_precanonical_raw_payload_hash(
    ra,
    tmp_path: Path,
) -> None:
    from easyicu.research_agent.cohort.schema import (
        CohortDefinition,
        CohortSchemaError,
        _load_locked_cohort_definition,
        write_locked_cohort_definition,
    )
    from easyicu.research_agent.schema import AnalysisPlan

    plan = AnalysisPlan(
        research_question="Does a predictor associate with an outcome?",
        cohort=CohortDefinition(name="primary", inclusion=(_age_predicate(0, 24),)),
        steps=[],
    )
    path = write_locked_cohort_definition(
        run_dir=tmp_path,
        plan=plan,
        evidence=ra.EvidenceStore(tmp_path),
        prompt_pack_version="test",
        llm_signature="mock",
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    window = payload["cohort"]["inclusion"][0]["time_window"]
    window["start_offset_hours"] = 0
    window["end_offset_hours"] = 24
    payload["cohort_sha256"] = hashlib.sha256(
        json.dumps(
            payload["cohort"],
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    with pytest.raises(CohortSchemaError, match="lock hash mismatch"):
        _load_locked_cohort_definition(tmp_path)


def test_cohort_lock_resume_does_not_rehydrate_scientific_drift(
    ra,
    tmp_path: Path,
) -> None:
    from easyicu.research_agent.cohort.schema import (
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
    assert (
        json.loads(path.read_text(encoding="utf-8"))["cohort"]["inclusion"][0]["value"]
        == 65
    )


def test_assert_cohort_definition_locked_catches_post_lock_mutation(
    ra,
    tmp_path: Path,
) -> None:
    from easyicu.research_agent.cohort.schema import (
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
    from easyicu.research_agent.cohort.schema import (
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
    from easyicu.research_agent.cohort.schema import (
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
    from easyicu.research_agent.cohort.schema import (
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
        run_dir=tmp_path,
        plan=_plan_with_cohort(definition),
        universe_path=universe_path,
    )
    assert result["status"] == "applied"
    assert result["n_universe"] == 4
    assert result["n_cohort"] == 3  # drops the age-10 stay
    out = tmp_path / "cohort_analysis.parquet"
    assert out.exists()
    assert len(pd.read_parquet(out)) == 3
    assert (tmp_path / "cohort_analysis_provenance.json").exists()
    flow_path = tmp_path / "cohort_analysis_flow.csv"
    assert result["flow_path"] == flow_path
    flow = pd.read_csv(flow_path)
    assert flow[["n_before", "n_excluded", "n_remaining"]].to_dict("records") == [
        {"n_before": 4, "n_excluded": 0, "n_remaining": 4},
        {"n_before": 4, "n_excluded": 1, "n_remaining": 3},
    ]
    provenance = json.loads(
        (tmp_path / "cohort_analysis_provenance.json").read_text(encoding="utf-8")
    )
    assert provenance["cohort_flow"][-1]["n_remaining"] == 3


def test_load_materialized_analysis_cohort_result_reopens_closed_outputs(
    tmp_path: Path,
) -> None:
    from easyicu.research_agent.cohort.schema import (
        CohortDefinition,
        load_materialized_analysis_cohort_result,
        materialize_locked_analysis_cohort,
    )

    universe_path = tmp_path / "cohort.parquet"
    pd.DataFrame({"age": [10, 18, 40, 70]}).to_parquet(universe_path, index=False)
    plan = _plan_with_cohort(
        CohortDefinition(name="adult", inclusion=(_age_predicate(0, 24),))
    )
    materialize_locked_analysis_cohort(
        run_dir=tmp_path,
        plan=plan,
        universe_path=universe_path,
    )

    recovered = load_materialized_analysis_cohort_result(
        run_dir=tmp_path,
        plan=plan,
    )

    assert recovered is not None
    assert recovered["n_universe"] == 4
    assert recovered["n_cohort"] == 3
    assert recovered["path"] == tmp_path / "cohort_analysis.parquet"
    assert recovered["flow_path"] == tmp_path / "cohort_analysis_flow.csv"


def test_load_materialized_analysis_cohort_result_rejects_flow_tampering(
    tmp_path: Path,
) -> None:
    from easyicu.research_agent.cohort.schema import (
        CohortDefinition,
        load_materialized_analysis_cohort_result,
        materialize_locked_analysis_cohort,
    )

    universe_path = tmp_path / "cohort.parquet"
    pd.DataFrame({"age": [10, 18, 40, 70]}).to_parquet(universe_path, index=False)
    plan = _plan_with_cohort(
        CohortDefinition(name="adult", inclusion=(_age_predicate(0, 24),))
    )
    materialize_locked_analysis_cohort(
        run_dir=tmp_path,
        plan=plan,
        universe_path=universe_path,
    )
    flow_path = tmp_path / "cohort_analysis_flow.csv"
    flow = pd.read_csv(flow_path)
    flow.loc[1, "n_excluded"] = 999
    flow.to_csv(flow_path, index=False)

    assert load_materialized_analysis_cohort_result(run_dir=tmp_path, plan=plan) is None


def test_load_materialized_analysis_cohort_result_rejects_parquet_tampering(
    tmp_path: Path,
) -> None:
    """Same-row-count content drift must refuse adoption via the ledger digest."""
    from easyicu.research_agent.cohort.schema import (
        CohortDefinition,
        load_materialized_analysis_cohort_result,
        materialize_locked_analysis_cohort,
    )

    universe_path = tmp_path / "cohort.parquet"
    pd.DataFrame({"age": [10, 18, 40, 70]}).to_parquet(universe_path, index=False)
    plan = _plan_with_cohort(
        CohortDefinition(name="adult", inclusion=(_age_predicate(0, 24),))
    )
    materialize_locked_analysis_cohort(
        run_dir=tmp_path,
        plan=plan,
        universe_path=universe_path,
    )
    provenance = json.loads(
        (tmp_path / "cohort_analysis_provenance.json").read_text(encoding="utf-8")
    )
    assert provenance["cohort_parquet_sha256"]

    cohort_path = tmp_path / "cohort_analysis.parquet"
    tampered = pd.read_parquet(cohort_path)
    tampered.loc[tampered.index[0], "age"] = 999
    tampered.to_parquet(cohort_path, index=False)
    assert len(pd.read_parquet(cohort_path)) == provenance["n_analysis_cohort"]

    assert load_materialized_analysis_cohort_result(run_dir=tmp_path, plan=plan) is None


def test_load_materialized_analysis_cohort_result_rejects_pre_digest_ledger(
    tmp_path: Path,
) -> None:
    """A pre-digest ledger cannot prove the original parquet content."""
    from easyicu.research_agent.cohort.schema import (
        CohortDefinition,
        load_materialized_analysis_cohort_result,
        materialize_locked_analysis_cohort,
    )

    universe_path = tmp_path / "cohort.parquet"
    pd.DataFrame({"age": [10, 18, 40, 70]}).to_parquet(universe_path, index=False)
    plan = _plan_with_cohort(
        CohortDefinition(name="adult", inclusion=(_age_predicate(0, 24),))
    )
    materialize_locked_analysis_cohort(
        run_dir=tmp_path,
        plan=plan,
        universe_path=universe_path,
    )
    provenance_path = tmp_path / "cohort_analysis_provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    del provenance["cohort_parquet_sha256"]
    provenance_path.write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    assert load_materialized_analysis_cohort_result(run_dir=tmp_path, plan=plan) is None


def test_materialize_no_definition_returns_no_file(tmp_path: Path) -> None:
    from easyicu.research_agent.cohort.schema import (
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


def test_materialize_explicit_all_input_rows_preserves_denominator(
    tmp_path: Path,
) -> None:
    from easyicu.research_agent.cohort.schema import (
        CohortDefinition,
        materialize_locked_analysis_cohort,
    )

    universe = pd.DataFrame({"stay_id": [1, 2], "adm": ["A", None]})
    universe_path = tmp_path / "cohort.parquet"
    universe.to_parquet(universe_path, index=False)

    result = materialize_locked_analysis_cohort(
        run_dir=tmp_path,
        plan=_plan_with_cohort(
            CohortDefinition(name="primary", selection_mode="all_input_rows")
        ),
        universe_path=universe_path,
    )

    assert result["status"] == "applied"
    assert result["n_universe"] == 2
    assert result["n_cohort"] == 2
    observed = pd.read_parquet(tmp_path / "cohort_analysis.parquet")
    pd.testing.assert_frame_equal(observed, universe)


def test_all_input_rows_rejects_predicates() -> None:
    from easyicu.research_agent.cohort.schema import (
        CohortDefinition,
        CohortSchemaError,
        validate_cohort_definition,
    )

    with pytest.raises(CohortSchemaError, match="requires empty"):
        validate_cohort_definition(
            CohortDefinition(
                name="primary",
                selection_mode="all_input_rows",
                inclusion=(_age_predicate(0, 24),),
            )
        )


def test_materialize_missing_column_falls_back_without_breaking(tmp_path: Path) -> None:
    """A predicate the universe cannot satisfy must not break the run: status
    'error', no parquet, caller falls back to the universe."""
    from easyicu.research_agent.cohort.schema import (
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
    from easyicu.research_agent.cohort.schema import ConceptPredicate, TimeWindow

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
    from easyicu.research_agent.cohort.schema import _resolve_predicate_column

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


def test_catalog_output_resolution_requires_unique_candidate(monkeypatch) -> None:
    from easyicu.concept.catalog import COMPOSITE_CONCEPT_OUTPUT_SOURCES
    from easyicu.research_agent.cohort.schema import _resolve_predicate_column

    monkeypatch.setitem(
        COMPOSITE_CONCEPT_OUTPUT_SOURCES,
        "synthetic_output_a",
        "synthetic_loader",
    )
    monkeypatch.setitem(
        COMPOSITE_CONCEPT_OUTPUT_SOURCES,
        "synthetic_output_b",
        "synthetic_loader",
    )
    columns = ["synthetic_output_a_max", "synthetic_output_b_max"]

    assert _resolve_predicate_column(columns, "synthetic_loader", "max") is None
    assert (
        _resolve_predicate_column(
            columns,
            "synthetic_loader",
            "max",
            column_bindings={"synthetic_loader": "synthetic_output_b_max"},
        )
        == "synthetic_output_b_max"
    )


def test_materialize_resolves_kdigo_alias_to_aki_stage_column(tmp_path: Path) -> None:
    """E3 regression: the locked cohort references concept_id `kdigo_aki`, but the
    universe materialised the concept as `aki_stage_*`. The materializer must
    bridge the concept-id -> output-column gap so the 纳排 is enforced centrally
    (cohort_analysis.parquet written) instead of silently running on the universe."""
    from easyicu.research_agent.cohort.schema import (
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


def _synthetic_source_predicate(
    *,
    aggregation="any",
    anchor="icu_admit",
    start_offset_hours=0.0,
    end_offset_hours=24.0,
):
    from easyicu.research_agent.cohort.schema import ConceptPredicate, TimeWindow

    return ConceptPredicate(
        concept_id="canonical_signal",
        time_window=TimeWindow(
            anchor=anchor,
            start_offset_hours=start_offset_hours,
            end_offset_hours=end_offset_hours,
        ),
        aggregation=aggregation,
        op="not_missing",
        value=None,
    )


def _synthetic_binding_plan(
    definition,
    inputs,
    *,
    cohort_output="artifact:analysis_cohort",
):
    return SimpleNamespace(
        cohort=definition,
        steps=[
            SimpleNamespace(
                inputs=list(inputs),
                expected_outputs=[
                    "table:cohort_attrition",
                    cohort_output,
                ],
            )
        ],
    )


def _synthetic_binding_context(
    *descriptors,
    primary_exposure="exported_signal_any",
    target_outcome=None,
):
    return SimpleNamespace(
        variables=list(descriptors),
        primary_exposure=primary_exposure,
        target_outcome=target_outcome,
    )


def test_descriptor_window_match_requires_explicit_matching_anchor() -> None:
    from easyicu.research_agent.cohort import schema as cohort_schema

    window = cohort_schema.TimeWindow("hospital_admit", 0, 24)

    assert cohort_schema._descriptor_window_matches_predicate(
        "hospital_admit_0_24h", window
    )
    assert cohort_schema._descriptor_window_matches_predicate(
        "hospital_admission[0,24]h", window
    )
    assert not cohort_schema._descriptor_window_matches_predicate("0_24h", window)
    assert not cohort_schema._descriptor_window_matches_predicate(
        "icu_admit_0_24h", window
    )
    assert not cohort_schema._descriptor_window_matches_predicate(
        "first_24h", cohort_schema.TimeWindow("icu_admit", 0, 24)
    )


def test_materialize_binds_unique_planner_input_by_source_concept(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from easyicu.research_agent.cohort import schema as cohort_schema

    monkeypatch.setattr(
        cohort_contract,
        "_EXTRA_COHORT_CONCEPT_IDS",
        {"canonical_signal"},
    )
    definition = cohort_schema.CohortDefinition(
        name="primary",
        inclusion=(_synthetic_source_predicate(),),
    )
    universe_path = tmp_path / "cohort.parquet"
    pd.DataFrame(
        {
            "stay_id": [1, 2, 3],
            "exported_signal_any": [0.0, None, 2.0],
            "exported_signal_measured": [1, 0, 1],
            "exported_signal_n": [2, 0, 3],
        }
    ).to_parquet(universe_path, index=False)
    context = _synthetic_binding_context(
        SimpleNamespace(
            name="exported_signal_any",
            source_concept="canonical_signal",
            role="other",
            analysis_window="icu_admit_0_24h",
        ),
        SimpleNamespace(
            name="exported_signal_measured",
            source_concept="canonical_signal",
            role="meta",
        ),
        SimpleNamespace(
            name="exported_signal_n",
            source_concept="canonical_signal",
            role="meta",
        ),
    )

    result = cohort_schema.materialize_locked_analysis_cohort(
        run_dir=tmp_path,
        plan=_synthetic_binding_plan(
            definition,
            [
                "stay_id",
                "exported_signal_any",
                "exported_signal_measured",
                "exported_signal_n",
            ],
        ),
        universe_path=universe_path,
        context=context,
    )

    assert result["status"] == "applied"
    assert result["n_cohort"] == 2
    provenance = json.loads(
        (tmp_path / "cohort_analysis_provenance.json").read_text(encoding="utf-8")
    )
    assert provenance["predicate_column_bindings"] == [
        {
            "concept_id": "canonical_signal",
            "column": "exported_signal_any",
            "basis": "planner_declared_context_input_source_concept",
            "predicate_contracts": [
                {
                    "aggregation": "any",
                    "time_window": {
                        "anchor": "icu_admit",
                        "start_offset_hours": 0,
                        "end_offset_hours": 24,
                    },
                }
            ],
        }
    ]


def test_cohort_namespace_product_preserves_planner_column_binding(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """A supported cohort namespace binds like analysis_cohort aliases."""

    from easyicu.research_agent.cohort import schema as cohort_schema

    monkeypatch.setattr(
        cohort_contract,
        "_EXTRA_COHORT_CONCEPT_IDS",
        {"canonical_signal"},
    )
    definition = cohort_schema.CohortDefinition(
        name="primary",
        inclusion=(_synthetic_source_predicate(),),
    )
    universe_path = tmp_path / "cohort.parquet"
    pd.DataFrame(
        {
            "stay_id": [1, 2, 3],
            "exported_signal_any": [0.0, None, 2.0],
        }
    ).to_parquet(universe_path, index=False)
    context = _synthetic_binding_context(
        SimpleNamespace(
            name="exported_signal_any",
            source_concept="canonical_signal",
            role="other",
            analysis_window="icu_admit_0_24h",
        )
    )

    result = cohort_schema.materialize_locked_analysis_cohort(
        run_dir=tmp_path,
        plan=_synthetic_binding_plan(
            definition,
            ["stay_id", "exported_signal_any"],
            cohort_output="cohort:analysis_set",
        ),
        universe_path=universe_path,
        context=context,
    )

    assert result["status"] == "applied"
    assert result["n_cohort"] == 2


def test_materialize_binds_unique_non_primary_qc_input_for_any_missingness(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from easyicu.research_agent.cohort import schema as cohort_schema

    monkeypatch.setattr(
        cohort_contract,
        "_EXTRA_COHORT_CONCEPT_IDS",
        {"canonical_signal"},
    )
    definition = cohort_schema.CohortDefinition(
        name="primary",
        inclusion=(_synthetic_source_predicate(),),
    )
    universe_path = tmp_path / "cohort.parquet"
    pd.DataFrame(
        {
            "stay_id": [1, 2, 3],
            "selected_qc_value_max": [0.0, None, 2.0],
            "unrelated_exposure": [4.0, 5.0, 6.0],
        }
    ).to_parquet(universe_path, index=False)
    context = _synthetic_binding_context(
        SimpleNamespace(
            name="selected_qc_value_max",
            source_concept="canonical_signal",
            role="other",
            analysis_window="icu_admission[0,24]h",
        ),
        SimpleNamespace(
            name="unrelated_exposure",
            source_concept="other_signal",
            role="exposure",
        ),
        primary_exposure="unrelated_exposure",
    )

    result = cohort_schema.materialize_locked_analysis_cohort(
        run_dir=tmp_path,
        plan=_synthetic_binding_plan(
            definition,
            ["stay_id", "selected_qc_value_max", "unrelated_exposure"],
        ),
        universe_path=universe_path,
        context=context,
    )

    assert result["status"] == "applied"
    assert pd.read_parquet(tmp_path / "cohort_analysis.parquet")[
        "stay_id"
    ].tolist() == [1, 3]


def test_materialize_rejects_ambiguous_any_missingness_inputs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from easyicu.research_agent.cohort import schema as cohort_schema

    monkeypatch.setattr(
        cohort_contract,
        "_EXTRA_COHORT_CONCEPT_IDS",
        {"canonical_signal"},
    )
    definition = cohort_schema.CohortDefinition(
        name="primary",
        inclusion=(_synthetic_source_predicate(),),
    )
    universe_path = tmp_path / "cohort.parquet"
    pd.DataFrame(
        {
            "selected_qc_value_first": [0.0, None],
            "selected_qc_value_max": [0.0, 2.0],
        }
    ).to_parquet(universe_path, index=False)
    context = _synthetic_binding_context(
        *(
            SimpleNamespace(
                name=name,
                source_concept="canonical_signal",
                role="other",
                analysis_window="icu_admission[0,24]h",
            )
            for name in ("selected_qc_value_first", "selected_qc_value_max")
        ),
        primary_exposure="selected_qc_value_max",
    )

    result = cohort_schema.materialize_locked_analysis_cohort(
        run_dir=tmp_path,
        plan=_synthetic_binding_plan(
            definition,
            ["selected_qc_value_first", "selected_qc_value_max"],
        ),
        universe_path=universe_path,
        context=context,
    )

    assert result["status"] == "error"
    assert "ambiguous" in result["error"]
    assert not (tmp_path / "cohort_analysis.parquet").exists()


@pytest.mark.parametrize("exact_column", ["canonical_signal", "canonical_signal_any"])
def test_materialize_exact_column_precedes_context_binding(
    tmp_path: Path,
    monkeypatch,
    exact_column: str,
) -> None:
    from easyicu.research_agent.cohort import schema as cohort_schema

    monkeypatch.setattr(
        cohort_contract,
        "_EXTRA_COHORT_CONCEPT_IDS",
        {"canonical_signal"},
    )
    definition = cohort_schema.CohortDefinition(
        name="primary",
        inclusion=(_synthetic_source_predicate(),),
    )
    universe_path = tmp_path / "cohort.parquet"
    pd.DataFrame(
        {
            "stay_id": [1, 2],
            exact_column: [1.0, None],
            "exported_signal_max": [None, 2.0],
        }
    ).to_parquet(universe_path, index=False)
    context = _synthetic_binding_context(
        SimpleNamespace(
            name="exported_signal_max",
            source_concept="canonical_signal",
            role="other",
            analysis_window="icu_admit_0_24h",
        ),
        primary_exposure="exported_signal_max",
    )

    result = cohort_schema.materialize_locked_analysis_cohort(
        run_dir=tmp_path,
        plan=_synthetic_binding_plan(
            definition,
            [exact_column, "exported_signal_max"],
        ),
        universe_path=universe_path,
        context=context,
    )

    assert result["status"] == "applied"
    assert pd.read_parquet(tmp_path / "cohort_analysis.parquet")[
        "stay_id"
    ].tolist() == [1]
    provenance = json.loads(
        (tmp_path / "cohort_analysis_provenance.json").read_text(encoding="utf-8")
    )
    assert provenance["predicate_column_bindings"] == []


def test_materialize_rejects_cross_name_aggregation_or_window_drift(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from easyicu.research_agent.cohort import schema as cohort_schema

    monkeypatch.setattr(
        cohort_contract,
        "_EXTRA_COHORT_CONCEPT_IDS",
        {"canonical_signal"},
    )
    definition = cohort_schema.CohortDefinition(
        name="primary",
        inclusion=(
            _synthetic_source_predicate(
                aggregation="first",
                start_offset_hours=0.0,
                end_offset_hours=6.0,
            ),
        ),
    )
    universe_path = tmp_path / "cohort.parquet"
    pd.DataFrame({"exported_signal_max": [0.0, 1.0]}).to_parquet(
        universe_path,
        index=False,
    )
    context = _synthetic_binding_context(
        SimpleNamespace(
            name="exported_signal_max",
            source_concept="canonical_signal",
            role="other",
            analysis_window="entire_stay",
        ),
        primary_exposure="exported_signal_max",
    )

    result = cohort_schema.materialize_locked_analysis_cohort(
        run_dir=tmp_path,
        plan=_synthetic_binding_plan(definition, ["exported_signal_max"]),
        universe_path=universe_path,
        context=context,
    )

    assert result["status"] == "error"
    assert "proven matching aggregation and time window" in result["error"]
    assert not (tmp_path / "cohort_analysis.parquet").exists()


def test_materialize_rejects_cross_name_window_without_anchor(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from easyicu.research_agent.cohort import schema as cohort_schema

    monkeypatch.setattr(
        cohort_contract,
        "_EXTRA_COHORT_CONCEPT_IDS",
        {"canonical_signal"},
    )
    definition = cohort_schema.CohortDefinition(
        name="primary",
        inclusion=(_synthetic_source_predicate(anchor="hospital_admit"),),
    )
    universe_path = tmp_path / "cohort.parquet"
    pd.DataFrame({"exported_signal_any": [0.0, 1.0]}).to_parquet(
        universe_path,
        index=False,
    )
    context = _synthetic_binding_context(
        SimpleNamespace(
            name="exported_signal_any",
            source_concept="canonical_signal",
            role="other",
            analysis_window="0_24h",
        ),
    )

    result = cohort_schema.materialize_locked_analysis_cohort(
        run_dir=tmp_path,
        plan=_synthetic_binding_plan(definition, ["exported_signal_any"]),
        universe_path=universe_path,
        context=context,
    )

    assert result["status"] == "error"
    assert "proven matching aggregation and time window" in result["error"]
    assert not (tmp_path / "cohort_analysis.parquet").exists()


def test_materialize_rejects_ambiguous_source_concept_bindings(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from easyicu.research_agent.cohort import schema as cohort_schema

    monkeypatch.setattr(
        cohort_contract,
        "_EXTRA_COHORT_CONCEPT_IDS",
        {"canonical_signal"},
    )
    definition = cohort_schema.CohortDefinition(
        name="primary",
        inclusion=(_synthetic_source_predicate(),),
    )
    universe_path = tmp_path / "cohort.parquet"
    pd.DataFrame(
        {
            "exported_signal_any": [0.0, 1.0],
            "other_signal_any": [0.0, 1.0],
        }
    ).to_parquet(universe_path, index=False)
    context = _synthetic_binding_context(
        SimpleNamespace(
            name="exported_signal_any",
            source_concept="canonical_signal",
            role="other",
            analysis_window="icu_admit_0_24h",
        ),
        SimpleNamespace(
            name="other_signal_any",
            source_concept="canonical_signal",
            role="outcome",
            analysis_window="icu_admit_0_24h",
        ),
        target_outcome="other_signal_any",
    )

    result = cohort_schema.materialize_locked_analysis_cohort(
        run_dir=tmp_path,
        plan=_synthetic_binding_plan(
            definition,
            ["exported_signal_any", "other_signal_any"],
        ),
        universe_path=universe_path,
        context=context,
    )

    assert result["status"] == "error"
    assert "binding is ambiguous" in result["error"]
    assert not (tmp_path / "cohort_analysis.parquet").exists()


def test_materialize_does_not_bind_unwindowed_loader_sibling(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from easyicu.research_agent.cohort import schema as cohort_schema

    monkeypatch.setattr(
        cohort_contract,
        "_EXTRA_COHORT_CONCEPT_IDS",
        {"canonical_signal"},
    )
    definition = cohort_schema.CohortDefinition(
        name="primary",
        inclusion=(_synthetic_source_predicate(),),
    )
    universe_path = tmp_path / "cohort.parquet"
    pd.DataFrame(
        {
            "exported_signal_max": [0.0, 1.0],
            "exported_signal_component": [0.0, 1.0],
        }
    ).to_parquet(universe_path, index=False)
    context = _synthetic_binding_context(
        SimpleNamespace(
            name="exported_signal_max",
            source_concept="canonical_signal",
            role="other",
        ),
        SimpleNamespace(
            name="exported_signal_component",
            source_concept="canonical_signal",
            role="other",
        ),
        primary_exposure="exported_signal_max",
    )

    result = cohort_schema.materialize_locked_analysis_cohort(
        run_dir=tmp_path,
        plan=_synthetic_binding_plan(definition, ["exported_signal_component"]),
        universe_path=universe_path,
        context=context,
    )

    assert result["status"] == "error"
    assert "no Planner-declared" in result["error"]
    assert "matching aggregation and time window" in result["error"]
    assert not (tmp_path / "cohort_analysis.parquet").exists()
