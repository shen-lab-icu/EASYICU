"""Adversarial joins between ResearchContext v2 and typed input authority."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import pandas as pd
import pytest

from easyicu.concept.metadata_projection import NumericBounds
from easyicu.research_agent.research_context.builder import build_research_context
from easyicu.research_agent.authority.evidence_store import EvidenceStore
from easyicu.research_agent.intake.materialized_metadata import (
    stage_materialized_cohort_authority,
)
from easyicu.research_agent.intake.materialized_trajectory import (
    StagedTrajectoryBinding,
    stage_materialized_trajectory_authority,
)
from easyicu.research_agent.research_context import typed as research_context_v2_module
from easyicu.research_agent.research_context.typed import (
    ResearchContextV2,
    materialized_research_inputs_from_authority,
)
from easyicu.research_agent.authority.run_input import (
    RUN_INPUT_CAPSULE_FILENAME,
    RunInputIdentityError,
    _validate_v2_context_input_authority,
    build_environment_identity,
    build_scientific_identity,
    seal_run_input_capsule,
)
from tests.research_agent.test_materialized_trajectory_authority import (
    _bundle,
    _implementation_sha,
)

QUESTION = "Is lactate associated with hospital mortality?"


def _scientific_identity(
    *,
    source_cohort_path: Path,
    source_cohort_ref: Any,
    source_trajectory_path: Path,
    source_trajectory_ref: Any,
) -> dict[str, Any]:
    return build_scientific_identity(
        cohort=source_cohort_path,
        question=QUESTION,
        cohort_name="v2_authority_join",
        database="miiv",
        target_outcome="death",
        primary_exposure="lact_max",
        cross_database_validation=None,
        inclusion_criteria=None,
        exclusion_criteria=None,
        id_columns=("stay_id",),
        time_columns=None,
        outcome_columns=("death",),
        time_windows=None,
        concept_descriptions=None,
        user_preferences=None,
        notes=None,
        skill_key=None,
        experiment_spec=None,
        source_files=None,
        disable_icu_context=False,
        materialized_cohort_authority_ref=source_cohort_ref.to_dict(),
        trajectory_path=source_trajectory_path,
        materialized_trajectory_authority_ref=source_trajectory_ref.to_dict(),
    )


def _prepare_typed_run(tmp_path: Path):
    paths, source_cohort, source_trajectory = _bundle(tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    cohort_path = run_dir / "cohort.parquet"
    staged_cohort = stage_materialized_cohort_authority(
        paths["parquet"],
        cohort_path,
        expected_source_authority=source_cohort.reference,
        producer_implementation_sha256=_implementation_sha(),
    )
    assert staged_cohort is not None
    trajectory_path = run_dir / "cohort_trajectory.parquet"
    staged_trajectory = stage_materialized_trajectory_authority(
        paths["trajectory"],
        trajectory_path,
        source_universe_path=paths["parquet"],
        target_universe_path=cohort_path,
        expected_source_authority=source_trajectory.reference,
        expected_target_universe_authority=staged_cohort.reference,
        producer_implementation_sha256=_implementation_sha(),
    )
    context = build_research_context(
        research_question=QUESTION,
        cohort=cohort_path,
        cohort_name="v2_authority_join",
        database="miiv",
        target_outcome="death",
        primary_exposure="lact_max",
        id_columns=("stay_id",),
        outcome_columns=("death",),
        trajectory_binding=StagedTrajectoryBinding(
            path=trajectory_path,
            sha256=staged_trajectory.authority.trajectory_sha256,
            size=staged_trajectory.authority.trajectory_size,
            authority_ref=staged_trajectory.reference,
        ),
    )
    assert isinstance(context, ResearchContextV2)
    identity = _scientific_identity(
        source_cohort_path=paths["parquet"],
        source_cohort_ref=source_cohort.reference,
        source_trajectory_path=paths["trajectory"],
        source_trajectory_ref=source_trajectory.reference,
    )
    return (
        run_dir,
        cohort_path,
        context,
        identity,
        staged_cohort,
        staged_trajectory,
    )


def _register_context(
    *,
    run_dir: Path,
    payload: dict[str, Any],
) -> tuple[Path, EvidenceStore]:
    context_path = run_dir / "research_context.json"
    context_path.write_text(
        json.dumps(payload, sort_keys=True, indent=2),
        encoding="utf-8",
    )
    evidence = EvidenceStore(run_dir)
    evidence.register_file(
        kind="log",
        description="Adversarial ResearchContext v2 authority join.",
        source_path=context_path,
        evidence_id="research_context",
        producer="pipeline",
        generation_mode="system",
    )
    return context_path, evidence


def _increment_cohort_rows(payload: dict[str, Any]) -> None:
    payload["materialized_inputs"]["cohort"]["cohort_rows"] += 1
    payload["cohort"]["n_stays"] += 1


def _change_cohort_database(payload: dict[str, Any]) -> None:
    payload["materialized_inputs"]["cohort"]["source_database"] = "eicu"


def _increment_trajectory_rows(payload: dict[str, Any]) -> None:
    payload["materialized_inputs"]["trajectory"]["trajectory_rows"] += 1


def _change_analysis_plausibility_range(payload: dict[str, Any]) -> None:
    forged_range = {"minimum": 999.0, "maximum": 1000.0}
    payload["materialized_inputs"]["cohort"]["column_bindings"]["age"][
        "analysis_plausibility_range"
    ] = forged_range
    for variable in payload["variables"]:
        if variable["name"] == "age":
            variable["valid_range"] = [999.0, 1000.0]
            break


def _change_trajectory_analysis_plausibility_range(payload: dict[str, Any]) -> None:
    payload["materialized_inputs"]["trajectory"][
        "concept_analysis_plausibility_ranges"
    ]["lact"] = {"minimum": 999.0, "maximum": 1000.0}


def _descriptor(payload: dict[str, Any], name: str) -> dict[str, Any]:
    return next(item for item in payload["variables"] if item["name"] == name)


def _forge_descriptor_source_files(payload: dict[str, Any]) -> None:
    _descriptor(payload, "age")["source_files"] = ["forged.parquet"]


def _forge_descriptor_dtype(payload: dict[str, Any]) -> None:
    _descriptor(payload, "age")["dtype"] = "object"


def _forge_descriptor_observed_domain(payload: dict[str, Any]) -> None:
    _descriptor(payload, "age")["observed_domain"] = {
        "n_unique": 999,
        "is_constant": False,
        "is_binary": False,
        "min": -999.0,
        "max": 999.0,
    }


@pytest.mark.parametrize(
    "tamper",
    (
        _increment_cohort_rows,
        _change_cohort_database,
        _increment_trajectory_rows,
        _change_analysis_plausibility_range,
        _change_trajectory_analysis_plausibility_range,
        _forge_descriptor_source_files,
        _forge_descriptor_dtype,
        _forge_descriptor_observed_domain,
    ),
    ids=(
        "cohort-rows",
        "cohort-source-database",
        "trajectory-rows",
        "analysis-plausibility-range",
        "trajectory-analysis-plausibility-range",
        "descriptor-source-files",
        "descriptor-dtype",
        "descriptor-observed-domain",
    ),
)
def test_fresh_typed_capsule_rejects_redundant_fact_tampering_under_same_refs(
    tmp_path: Path,
    tamper: Callable[[dict[str, Any]], None],
) -> None:
    run_dir, cohort_path, context, identity, _cohort, _trajectory = _prepare_typed_run(
        tmp_path
    )
    payload = context.model_dump(mode="json")
    cohort_ref = dict(payload["materialized_inputs"]["cohort"]["authority_ref"])
    trajectory_ref = dict(payload["materialized_inputs"]["trajectory"]["authority_ref"])
    tamper(payload)
    assert payload["materialized_inputs"]["cohort"]["authority_ref"] == cohort_ref
    assert (
        payload["materialized_inputs"]["trajectory"]["authority_ref"] == trajectory_ref
    )
    context_path, evidence = _register_context(run_dir=run_dir, payload=payload)

    with pytest.raises(RunInputIdentityError):
        seal_run_input_capsule(
            run_dir=run_dir,
            evidence=evidence,
            scientific_identity=identity,
            initial_environment=build_environment_identity(llm_signature="mock"),
            context_path=context_path,
            cohort_path=cohort_path,
            experiment_spec_path=None,
        )

    assert not (run_dir / RUN_INPUT_CAPSULE_FILENAME).exists()


@pytest.mark.parametrize(
    "tamper",
    (
        _increment_cohort_rows,
        _increment_trajectory_rows,
    ),
    ids=("cohort-rows", "trajectory-rows"),
)
def test_resume_join_rejects_redundant_facts_despite_unchanged_authority_refs(
    tmp_path: Path,
    tamper: Callable[[dict[str, Any]], None],
) -> None:
    _run_dir, _cohort_path, context, _identity, cohort, trajectory = _prepare_typed_run(
        tmp_path
    )
    payload = context.model_dump(mode="python")
    tamper(payload)
    tampered_context = ResearchContextV2.model_validate(payload)

    with pytest.raises(
        RunInputIdentityError,
        match="typed input facts do not match staged authority",
    ):
        _validate_v2_context_input_authority(
            tampered_context,
            cohort_path=_cohort_path,
            cohort=cohort,
            trajectory=trajectory,
            allow_v1=True,
            require_current_implementation=False,
        )


def test_resume_preserves_sealed_fallback_range_across_icu_rule_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _run_dir, _cohort_path, context, _identity, cohort, trajectory = _prepare_typed_run(
        tmp_path
    )
    classify = research_context_v2_module.ICU_RULES.classify_variable

    def drifted_classify(name: str, dtype: str, sample: Any):
        hint = classify(name, dtype, sample)
        if name in {"age", "lact"}:
            return replace(hint, valid_range=(-10.0, 200.0))
        return hint

    monkeypatch.setattr(
        research_context_v2_module,
        "ICU_RULES",
        SimpleNamespace(classify_variable=drifted_classify),
    )

    _validate_v2_context_input_authority(
        context,
        cohort_path=_cohort_path,
        cohort=cohort,
        trajectory=trajectory,
        allow_v1=True,
        require_current_implementation=False,
    )
    with pytest.raises(
        RunInputIdentityError,
        match="typed input facts do not match staged authority",
    ):
        _validate_v2_context_input_authority(
            context,
            cohort_path=_cohort_path,
            cohort=cohort,
            trajectory=trajectory,
            allow_v1=True,
            require_current_implementation=True,
        )


def test_fresh_authority_join_reads_the_verified_cohort_snapshot_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _run_dir, cohort_path, context, _identity, cohort, trajectory = _prepare_typed_run(
        tmp_path
    )
    from easyicu.research_agent.authority import run_input as capsule_module

    original = capsule_module.read_verified_materialized_cohort_table
    reads = 0

    def counted_read(*args: Any, **kwargs: Any):
        nonlocal reads
        reads += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(
        capsule_module,
        "read_verified_materialized_cohort_table",
        counted_read,
    )
    _validate_v2_context_input_authority(
        context,
        cohort_path=cohort_path,
        cohort=cohort,
        trajectory=trajectory,
        allow_v1=False,
        require_current_implementation=True,
    )

    assert reads == 1


def test_resume_keeps_explicit_sidecar_range_exact(tmp_path: Path) -> None:
    _run_dir, _cohort_path, context, _identity, cohort, trajectory = _prepare_typed_run(
        tmp_path
    )
    file_binding = next(
        item
        for item in cohort.sidecar.files
        if item.relative_path == cohort.authority.cohort_file
    )
    columns = dict(file_binding.columns)
    age = columns["age"]
    columns["age"] = replace(
        age,
        metadata=replace(
            age.metadata,
            analysis_plausibility_range=NumericBounds(minimum=0.0),
        ),
    )
    explicit_file_binding = replace(file_binding, columns=columns)
    explicit_cohort = replace(
        cohort,
        sidecar=replace(cohort.sidecar, files=(explicit_file_binding,)),
    )
    payload = context.model_dump(mode="python")
    payload["materialized_inputs"] = materialized_research_inputs_from_authority(
        cohort=explicit_cohort,
        trajectory=trajectory,
    ).model_dump(mode="python")
    for variable in payload["variables"]:
        if variable["name"] == "age":
            variable["valid_range"] = None
            break
    explicit_context = ResearchContextV2.model_validate(payload)

    _validate_v2_context_input_authority(
        explicit_context,
        cohort_path=_cohort_path,
        cohort=explicit_cohort,
        trajectory=trajectory,
        allow_v1=True,
        require_current_implementation=False,
    )
    tampered_payload = explicit_context.model_dump(mode="python")
    tampered_payload["materialized_inputs"]["cohort"]["column_bindings"]["age"][
        "analysis_plausibility_range"
    ] = {"minimum": 999.0, "maximum": None}
    tampered_context = ResearchContextV2.model_validate(tampered_payload)

    with pytest.raises(
        RunInputIdentityError,
        match="typed input facts do not match staged authority",
    ):
        _validate_v2_context_input_authority(
            tampered_context,
            cohort_path=_cohort_path,
            cohort=explicit_cohort,
            trajectory=trajectory,
            allow_v1=True,
            require_current_implementation=False,
        )


def test_fresh_typed_capsule_rejects_v1_context(tmp_path: Path) -> None:
    run_dir, cohort_path, _context, identity, _cohort, _trajectory = _prepare_typed_run(
        tmp_path
    )
    legacy_context = build_research_context(
        research_question=QUESTION,
        cohort=pd.read_parquet(cohort_path),
        cohort_name="v2_authority_join",
        database="miiv",
        target_outcome="death",
        primary_exposure="lact_max",
        id_columns=("stay_id",),
        outcome_columns=("death",),
    )
    assert not isinstance(legacy_context, ResearchContextV2)
    payload = legacy_context.model_dump(mode="json")
    context_path, evidence = _register_context(run_dir=run_dir, payload=payload)

    with pytest.raises(RunInputIdentityError, match="require a ResearchContext v2"):
        seal_run_input_capsule(
            run_dir=run_dir,
            evidence=evidence,
            scientific_identity=identity,
            initial_environment=build_environment_identity(llm_signature="mock"),
            context_path=context_path,
            cohort_path=cohort_path,
            experiment_spec_path=None,
        )

    assert not (run_dir / RUN_INPUT_CAPSULE_FILENAME).exists()


@pytest.mark.parametrize(
    ("field", "forged"),
    (
        ("unit", "FORGED"),
        ("valid_range", [999.0, 1000.0]),
        ("source_concept", None),
        ("source_databases", ["forged"]),
        ("derived_from_concepts", ["forged"]),
        ("source_tables", ["forged"]),
        ("item_ids", ["forged"]),
        ("unit_normalization", "forged"),
    ),
)
def test_v2_rejects_legacy_descriptor_physical_fact_tampering(
    tmp_path: Path,
    field: str,
    forged: Any,
) -> None:
    _run_dir, _cohort_path, context, *_rest = _prepare_typed_run(tmp_path)
    payload = context.model_dump(mode="python")
    age = next(item for item in payload["variables"] if item["name"] == "age")
    age[field] = forged

    with pytest.raises(ValueError, match="descriptor physical field"):
        ResearchContextV2.model_validate(payload)


def test_v2_rejects_duplicate_unknown_and_missing_full_variable_rosters(
    tmp_path: Path,
) -> None:
    _run_dir, _cohort_path, context, *_rest = _prepare_typed_run(tmp_path)

    duplicate = context.model_dump(mode="python")
    duplicate["variables"].append(dict(duplicate["variables"][0]))
    with pytest.raises(ValueError, match="variable names must be unique"):
        ResearchContextV2.model_validate(duplicate)

    unknown = context.model_dump(mode="python")
    unknown["variables"][0]["name"] = "forged_unknown"
    with pytest.raises(ValueError, match="absent from the cohort"):
        ResearchContextV2.model_validate(unknown)

    missing = context.model_dump(mode="python")
    missing["variables"].pop()
    with pytest.raises(ValueError, match="does not cover the cohort columns"):
        ResearchContextV2.model_validate(missing)


def test_v2_descriptor_closure_does_not_claim_scientific_role_ownership(
    tmp_path: Path,
) -> None:
    _run_dir, _cohort_path, context, *_rest = _prepare_typed_run(tmp_path)
    payload = context.model_dump(mode="python")
    age = next(item for item in payload["variables"] if item["name"] == "age")
    age["role"] = "other"

    parsed = ResearchContextV2.model_validate(payload)

    assert parsed.variable("age").role.value == "other"


def test_v2_rejects_incomplete_trajectory_range_map(tmp_path: Path) -> None:
    _run_dir, _cohort_path, context, *_rest = _prepare_typed_run(tmp_path)
    payload = context.model_dump(mode="python")
    payload["materialized_inputs"]["trajectory"][
        "concept_analysis_plausibility_ranges"
    ].pop("lact")

    with pytest.raises(ValueError, match="plausibility ranges do not match"):
        ResearchContextV2.model_validate(payload)
