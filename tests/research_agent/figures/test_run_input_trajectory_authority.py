"""Run-input capsule contracts for host-owned trajectory authority."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from pydantic import ValidationError

from easyicu.research_agent.research_context.builder import build_research_context
from easyicu.research_agent.authority.execution_input import (
    ExecutionInputAuthorityState,
)
from easyicu.research_agent.authority.analysis_cohort import (
    ExecutionCohortAuthority,
)
from easyicu.research_agent.authority.evidence_store import EvidenceStore, sha256_of_file
from easyicu.research_agent.intake.materialized_metadata import (
    MaterializedCohortAuthorityRef,
    stage_materialized_cohort_authority,
)
from easyicu.research_agent.intake.materialized_trajectory import (
    MaterializedTrajectoryError,
    MaterializedTrajectoryAuthorityRef,
    StagedTrajectoryBinding,
    VerifiedLegacyTrajectoryCapsuleReceipt,
    load_verified_materialized_trajectory_authority,
    stage_legacy_trajectory_exact,
    stage_materialized_trajectory_authority,
)
from easyicu.research_agent.providers.mocks import MockLLMClient
from easyicu.research_agent.pipeline import ResearchAgentPipeline
from easyicu.research_agent.authority.run_input import (
    RUN_INPUT_CAPSULE_EVIDENCE_ID,
    RUN_INPUT_CAPSULE_FILENAME,
    RUN_INPUT_CAPSULE_SCHEMA_VERSION,
    RUN_INPUT_CAPSULE_SCHEMA_VERSION_V3,
    RunInputCapsuleV2,
    RunInputCapsuleV3,
    RunInputIdentityError,
    _scientific_trajectory_envelope,
    build_environment_identity,
    build_scientific_identity,
    canonical_sha256,
    load_verified_run_input_capsule,
    prepare_existing_resume_input,
    seal_run_input_capsule,
    verify_legacy_trajectory_capsule_receipt,
)
from easyicu.research_agent.research_context.typed import (
    ResearchContextV2,
    parse_research_context_json,
)
from tests.research_agent.figures.test_materialized_trajectory_authority import (
    _bundle,
    _implementation_sha,
)

QUESTION = "Is lactate associated with hospital mortality?"


def _identity(
    *,
    cohort_path: Path,
    cohort_ref=None,
    trajectory_path: Path | None = None,
    trajectory_ref: MaterializedTrajectoryAuthorityRef | None = None,
):
    return build_scientific_identity(
        cohort=cohort_path,
        question=QUESTION,
        cohort_name="typed_trajectory_capsule",
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
        materialized_cohort_authority_ref=(
            cohort_ref.to_dict() if cohort_ref is not None else None
        ),
        trajectory_path=trajectory_path,
        materialized_trajectory_authority_ref=(
            trajectory_ref.to_dict() if trajectory_ref is not None else None
        ),
    )


def _context_and_evidence(
    run_dir: Path,
    cohort_path: Path,
    *,
    trajectory_binding: StagedTrajectoryBinding | None = None,
):
    context = build_research_context(
        research_question=QUESTION,
        cohort=cohort_path,
        cohort_name="typed_trajectory_capsule",
        database="miiv",
        target_outcome="death",
        primary_exposure="lact_max",
        id_columns=("stay_id",),
        outcome_columns=("death",),
        trajectory_binding=trajectory_binding,
    )
    context_path = run_dir / "research_context.json"
    context_path.write_text(context.model_dump_json(indent=2), encoding="utf-8")
    evidence = EvidenceStore(run_dir)
    evidence.register_file(
        kind="log",
        description="Frozen trajectory research context.",
        source_path=context_path,
        evidence_id="research_context",
        producer="pipeline",
        generation_mode="system",
    )
    return context_path, evidence


def _staged_typed_inputs(tmp_path: Path):
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
    identity = _identity(
        cohort_path=paths["parquet"],
        cohort_ref=source_cohort.reference,
        trajectory_path=paths["trajectory"],
        trajectory_ref=source_trajectory.reference,
    )
    context_path, evidence = _context_and_evidence(
        run_dir,
        cohort_path,
        trajectory_binding=StagedTrajectoryBinding(
            path=trajectory_path,
            sha256=staged_trajectory.authority.trajectory_sha256,
            size=staged_trajectory.authority.trajectory_size,
            authority_ref=staged_trajectory.reference,
        ),
    )
    return (
        paths,
        source_cohort,
        source_trajectory,
        run_dir,
        cohort_path,
        staged_cohort,
        trajectory_path,
        staged_trajectory,
        identity,
        context_path,
        evidence,
    )


def test_typed_run_input_capsule_v3_binds_exact_staged_trajectory(tmp_path):
    (
        _paths,
        _source_cohort,
        source_trajectory,
        run_dir,
        cohort_path,
        staged_cohort,
        trajectory_path,
        staged_trajectory,
        identity,
        context_path,
        evidence,
    ) = _staged_typed_inputs(tmp_path)

    capsule = seal_run_input_capsule(
        run_dir=run_dir,
        evidence=evidence,
        scientific_identity=identity,
        initial_environment=build_environment_identity(llm_signature="mock"),
        context_path=context_path,
        cohort_path=cohort_path,
        experiment_spec_path=None,
    )

    assert isinstance(capsule, RunInputCapsuleV3)
    assert capsule.schema_version == RUN_INPUT_CAPSULE_SCHEMA_VERSION_V3
    assert capsule.materialized_trajectory_authority_required is True
    assert capsule.trajectory_relative_path == "cohort_trajectory.parquet"
    assert capsule.trajectory_sha256 == staged_trajectory.authority.trajectory_sha256
    assert (
        capsule.materialized_cohort_authority_ref == staged_cohort.reference.to_dict()
    )
    assert (
        capsule.materialized_trajectory_authority_ref
        == staged_trajectory.reference.to_dict()
    )
    assert (
        staged_trajectory.authority.parent_trajectory_authority
        == source_trajectory.reference
    )
    assert trajectory_path.read_bytes()
    loaded = load_verified_run_input_capsule(
        run_dir=run_dir,
        scientific_identity=identity,
    )
    assert loaded.capsule == capsule
    context = json.loads(context_path.read_text(encoding="utf-8"))
    assert context["schema_version"] == "easyicu.research_context/3"
    trajectory_context = context["materialized_inputs"]["trajectory"]
    assert trajectory_context["authority_ref"] == staged_trajectory.reference.to_dict()
    assert trajectory_context["requested_concepts"] == ["lact"]
    assert trajectory_context["materialized_concepts"] == ["lact"]


def test_v2_trajectory_availability_lists_must_be_unique(tmp_path):
    (
        _paths,
        _source_cohort,
        _source_trajectory,
        _run_dir,
        _cohort_path,
        _staged_cohort,
        _trajectory_path,
        _staged_trajectory,
        _identity_value,
        context_path,
        _evidence,
    ) = _staged_typed_inputs(tmp_path)
    context = parse_research_context_json(
        context_path.read_text(encoding="utf-8")
    )
    payload = context.model_dump(mode="python")
    payload["materialized_inputs"]["trajectory"]["materialized_concepts"] = [
        "lact",
        "lact",
    ]

    with pytest.raises(ValidationError, match="availability states"):
        type(context).model_validate(payload)


def test_fresh_typed_capsule_rejects_legacy_v1_context(tmp_path):
    paths, source_cohort, _source_trajectory = _bundle(tmp_path)
    run_dir = tmp_path / "fresh_typed_v1_context"
    run_dir.mkdir()
    cohort_path = run_dir / "cohort.parquet"
    staged_cohort = stage_materialized_cohort_authority(
        paths["parquet"],
        cohort_path,
        expected_source_authority=source_cohort.reference,
        producer_implementation_sha256=_implementation_sha(),
    )
    assert staged_cohort is not None
    legacy_context = build_research_context(
        research_question=QUESTION,
        cohort=pd.read_parquet(cohort_path),
        cohort_name="typed_trajectory_capsule",
        database="miiv",
        target_outcome="death",
        primary_exposure="lact_max",
        id_columns=("stay_id",),
        outcome_columns=("death",),
    )
    assert not isinstance(legacy_context, ResearchContextV2)
    context_path = run_dir / "research_context.json"
    context_path.write_text(
        legacy_context.model_dump_json(indent=2),
        encoding="utf-8",
    )
    evidence = EvidenceStore(run_dir)
    evidence.register_file(
        kind="log",
        description="Invalid fresh typed V1 context.",
        source_path=context_path,
        evidence_id="research_context",
        producer="pipeline",
        generation_mode="system",
    )
    identity = _identity(
        cohort_path=paths["parquet"],
        cohort_ref=source_cohort.reference,
    )
    cohort_identity = dict(identity["cohort"])
    cohort_identity.pop("trajectory", None)
    identity = {**identity, "cohort": cohort_identity}

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


def test_pipeline_fresh_typed_trajectory_stages_exact_child_before_plan(
    tmp_path,
    monkeypatch,
):
    paths, source_cohort, source_trajectory = _bundle(tmp_path)
    workdir = tmp_path / "agent"
    pipeline = ResearchAgentPipeline(
        workdir=workdir,
        llm=MockLLMClient(),
        enable_cache=False,
        enable_literature=False,
        enable_memory=False,
        enable_latex=False,
    )

    def inspect_staged_inputs(**kwargs):
        run_dir = kwargs["run_dir"]
        staged_cohort_path = run_dir / "cohort.parquet"
        staged_trajectory_path = run_dir / "cohort_trajectory.parquet"
        staged_trajectory = load_verified_materialized_trajectory_authority(
            staged_trajectory_path
        )
        assert staged_trajectory is not None
        assert staged_trajectory_path.read_bytes() == paths["trajectory"].read_bytes()
        assert staged_trajectory.authority.parent_trajectory_authority == (
            source_trajectory.reference
        )
        assert (
            staged_trajectory.authority.bound_universe_file == staged_cohort_path.name
        )
        scientific_identity = kwargs["run_scientific_identity"]
        assert scientific_identity["materialized_cohort_authority_ref"] == (
            source_cohort.reference.to_dict()
        )
        assert scientific_identity["materialized_trajectory_authority_ref"] == (
            source_trajectory.reference.to_dict()
        )
        binding = kwargs["trajectory_binding"]
        assert binding.authority_ref == staged_trajectory.reference
        raise RuntimeError("staged inputs inspected before plan")

    monkeypatch.setattr(pipeline, "_run_plan_phase", inspect_staged_inputs)
    with pytest.raises(RuntimeError, match="staged inputs inspected"):
        pipeline.run(
            question=QUESTION,
            cohort=paths["parquet"],
            cohort_authority_path=(
                paths["parquet"].parent / source_cohort.reference.file
            ),
            cohort_authority_ref=source_cohort.reference,
            trajectory_path=paths["trajectory"],
            trajectory_authority_path=(
                paths["trajectory"].parent / source_trajectory.reference.file
            ),
            trajectory_authority_ref=source_trajectory.reference,
            cohort_name="fresh_typed_trajectory",
            database="miiv",
            target_outcome="death",
            primary_exposure="lact_max",
            stop_after_analysis=True,
        )


def test_pipeline_rejects_raw_trajectory_for_typed_cohort_before_run_write(tmp_path):
    paths, source_cohort, _source_trajectory = _bundle(tmp_path)
    raw_trajectory = tmp_path / "raw_trajectory.parquet"
    raw_trajectory.write_bytes(paths["trajectory"].read_bytes())
    workdir = tmp_path / "agent"
    pipeline = ResearchAgentPipeline(
        workdir=workdir,
        llm=MockLLMClient(),
        enable_cache=False,
        enable_literature=False,
        enable_memory=False,
        enable_latex=False,
    )

    with pytest.raises(MaterializedTrajectoryError, match="sealed trajectory"):
        pipeline.run(
            question=QUESTION,
            cohort=paths["parquet"],
            cohort_authority_path=(
                paths["parquet"].parent / source_cohort.reference.file
            ),
            cohort_authority_ref=source_cohort.reference,
            trajectory_path=raw_trajectory,
            cohort_name="typed_raw_trajectory_rejected",
            database="miiv",
            target_outcome="death",
            primary_exposure="lact_max",
            stop_after_analysis=True,
        )
    assert not list(workdir.glob("run_*"))


def test_prepare_resume_returns_capsule_selected_trajectory_binding(tmp_path):
    (
        _paths,
        _source_cohort,
        _source_trajectory,
        run_dir,
        cohort_path,
        _staged_cohort,
        trajectory_path,
        staged_trajectory,
        identity,
        context_path,
        evidence,
    ) = _staged_typed_inputs(tmp_path)
    environment = build_environment_identity(llm_signature="mock")
    seal_run_input_capsule(
        run_dir=run_dir,
        evidence=evidence,
        scientific_identity=identity,
        initial_environment=environment,
        context_path=context_path,
        cohort_path=cohort_path,
        experiment_spec_path=None,
    )

    prepared = prepare_existing_resume_input(
        run_dir=run_dir,
        resume_state={"per_step_records": []},
        scientific_identity=identity,
        current_environment=environment,
        cohort=cohort_path,
        question=QUESTION,
        resume_from_step_id=None,
        enforcement_mode="soft",
        load_compatible_plan=lambda **_kwargs: (None, None),
    )

    assert prepared.input_verified is True
    assert prepared.trajectory_binding is not None
    assert prepared.trajectory_binding.path == trajectory_path
    assert (
        prepared.trajectory_binding.sha256
        == staged_trajectory.authority.trajectory_sha256
    )
    assert prepared.trajectory_binding.size == trajectory_path.stat().st_size
    assert prepared.trajectory_binding.authority_ref == staged_trajectory.reference


def test_v3_schema_requires_explicit_trajectory_marker_and_reference(tmp_path):
    (
        _paths,
        _source_cohort,
        _source_trajectory,
        run_dir,
        cohort_path,
        _staged_cohort,
        _trajectory_path,
        _staged_trajectory,
        identity,
        context_path,
        evidence,
    ) = _staged_typed_inputs(tmp_path)
    capsule = seal_run_input_capsule(
        run_dir=run_dir,
        evidence=evidence,
        scientific_identity=identity,
        initial_environment=build_environment_identity(llm_signature="mock"),
        context_path=context_path,
        cohort_path=cohort_path,
        experiment_spec_path=None,
    )
    payload = capsule.model_dump(mode="json")

    for missing in (
        "materialized_trajectory_authority_required",
        "materialized_trajectory_authority_ref",
        "trajectory_relative_path",
        "trajectory_sha256",
    ):
        invalid = dict(payload)
        invalid.pop(missing)
        with pytest.raises(ValidationError):
            RunInputCapsuleV3.model_validate(invalid)


@pytest.mark.parametrize("tamper", ("artifact", "selector"))
def test_v3_resume_rejects_trajectory_authority_tamper_before_receipt(
    tmp_path,
    tamper,
):
    (
        _paths,
        _source_cohort,
        source_trajectory,
        run_dir,
        cohort_path,
        _staged_cohort,
        trajectory_path,
        _staged_trajectory,
        identity,
        context_path,
        evidence,
    ) = _staged_typed_inputs(tmp_path)
    seal_run_input_capsule(
        run_dir=run_dir,
        evidence=evidence,
        scientific_identity=identity,
        initial_environment=build_environment_identity(llm_signature="mock"),
        context_path=context_path,
        cohort_path=cohort_path,
        experiment_spec_path=None,
    )
    if tamper == "artifact":
        trajectory_path.write_bytes(trajectory_path.read_bytes() + b"tamper")
    else:
        selector_path = trajectory_path.with_name("cohort_trajectory_provenance.json")
        selector = json.loads(selector_path.read_text(encoding="utf-8"))
        selector["trajectory_authority"][
            "authority"
        ] = source_trajectory.reference.to_dict()
        selector_path.write_text(json.dumps(selector), encoding="utf-8")

    with pytest.raises(RunInputIdentityError):
        prepare_existing_resume_input(
            run_dir=run_dir,
            resume_state={"per_step_records": []},
            scientific_identity=identity,
            current_environment=build_environment_identity(llm_signature="mock"),
            cohort=cohort_path,
            question=QUESTION,
            resume_from_step_id=None,
            enforcement_mode="soft",
            load_compatible_plan=lambda **_kwargs: (None, None),
        )
    assert not list(run_dir.glob("resume_environment_receipt_*.json"))


def test_capsule_seal_rejects_trajectory_identity_presence_mismatch(tmp_path):
    (
        paths,
        source_cohort,
        _source_trajectory,
        run_dir,
        cohort_path,
        _staged_cohort,
        _trajectory_path,
        _staged_trajectory,
        _existing_identity,
        context_path,
        evidence,
    ) = _staged_typed_inputs(tmp_path)
    identity_without_trajectory = _identity(
        cohort_path=paths["parquet"],
        cohort_ref=source_cohort.reference,
    )

    with pytest.raises(
        RunInputIdentityError,
        match="Typed staged trajectory is absent from scientific identity",
    ):
        seal_run_input_capsule(
            run_dir=run_dir,
            evidence=evidence,
            scientific_identity=identity_without_trajectory,
            initial_environment=build_environment_identity(llm_signature="mock"),
            context_path=context_path,
            cohort_path=cohort_path,
            experiment_spec_path=None,
        )


def test_legacy_canonical_sibling_identity_and_v1_capsule_are_stable(tmp_path):
    source_cohort = tmp_path / "legacy.parquet"
    source_trajectory = tmp_path / "legacy_trajectory.parquet"
    pd.DataFrame({"stay_id": [1, 2], "death": [0, 1]}).to_parquet(
        source_cohort, index=False
    )
    pd.DataFrame(
        {
            "stay_id": [1],
            "charttime": [1.0],
            "concept": ["lact"],
            "value_num": [2.0],
            "value_str": ["2.0"],
        }
    ).to_parquet(source_trajectory, index=False)
    implicit = _identity(cohort_path=source_cohort)
    explicit = _identity(
        cohort_path=source_cohort,
        trajectory_path=source_trajectory,
    )
    assert explicit == implicit
    assert "trajectory" not in explicit
    assert explicit["cohort"]["trajectory"]["sha256"]

    run_dir = tmp_path / "legacy_run"
    run_dir.mkdir()
    cohort_path = run_dir / "cohort.parquet"
    cohort_path.write_bytes(source_cohort.read_bytes())
    trajectory_path = stage_legacy_trajectory_exact(
        source_trajectory,
        run_dir / "cohort_trajectory.parquet",
        expected_sha256=implicit["cohort"]["trajectory"]["sha256"],
        expected_size=implicit["cohort"]["trajectory"]["size_bytes"],
    )
    context_path, evidence = _context_and_evidence(run_dir, cohort_path)
    capsule = seal_run_input_capsule(
        run_dir=run_dir,
        evidence=evidence,
        scientific_identity=implicit,
        initial_environment=build_environment_identity(llm_signature="mock"),
        context_path=context_path,
        cohort_path=cohort_path,
        experiment_spec_path=None,
    )

    assert capsule.schema_version == RUN_INPUT_CAPSULE_SCHEMA_VERSION
    assert capsule.trajectory_relative_path == trajectory_path.name
    assert (
        load_verified_run_input_capsule(
            run_dir=run_dir,
            scientific_identity=implicit,
        ).capsule
        == capsule
    )


def test_archived_v2_typed_cohort_with_raw_trajectory_resumes_exactly(tmp_path):
    paths, source_cohort, _source_trajectory = _bundle(tmp_path)
    run_dir = tmp_path / "archived_v2_run"
    run_dir.mkdir()
    cohort_path = run_dir / "cohort.parquet"
    staged_cohort = stage_materialized_cohort_authority(
        paths["parquet"],
        cohort_path,
        expected_source_authority=source_cohort.reference,
        producer_implementation_sha256=_implementation_sha(),
    )
    assert staged_cohort is not None
    identity = _identity(
        cohort_path=paths["parquet"],
        cohort_ref=source_cohort.reference,
        trajectory_path=paths["trajectory"],
    )
    assert "materialized_trajectory_authority_ref" not in identity
    trajectory_identity = identity["cohort"]["trajectory"]
    legacy_trajectory_path = stage_legacy_trajectory_exact(
        paths["trajectory"],
        run_dir / "cohort_trajectory.parquet",
        expected_sha256=trajectory_identity["sha256"],
        expected_size=trajectory_identity["size_bytes"],
    )
    context_path, evidence = _context_and_evidence(run_dir, cohort_path)
    context_record = evidence.get("research_context")
    assert context_record is not None
    environment = build_environment_identity(llm_signature="mock")
    capsule = RunInputCapsuleV2(
        scientific_identity=identity,
        scientific_identity_sha256=canonical_sha256(identity),
        context_sha256=context_record.sha256,
        cohort_sha256=sha256_of_file(cohort_path),
        trajectory_relative_path=legacy_trajectory_path.name,
        trajectory_sha256=sha256_of_file(legacy_trajectory_path),
        initial_environment=environment,
        materialized_cohort_authority_ref=staged_cohort.reference.to_dict(),
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    capsule_path = run_dir / RUN_INPUT_CAPSULE_FILENAME
    capsule_path.write_text(capsule.model_dump_json(indent=2), encoding="utf-8")
    evidence.register_file(
        kind="log",
        description="Archived V2 typed cohort with legacy raw trajectory.",
        source_path=capsule_path,
        evidence_id=RUN_INPUT_CAPSULE_EVIDENCE_ID,
        producer="pipeline",
        generation_mode="system",
    )

    loaded = load_verified_run_input_capsule(
        run_dir=run_dir,
        scientific_identity=identity,
    )
    assert loaded.capsule == capsule
    assert set(capsule.model_dump(mode="json")) == {
        "schema_version",
        "scientific_identity",
        "scientific_identity_sha256",
        "context_evidence_id",
        "context_sha256",
        "context_relative_path",
        "cohort_relative_path",
        "cohort_sha256",
        "trajectory_relative_path",
        "trajectory_sha256",
        "experiment_spec_evidence_id",
        "experiment_spec_sha256",
        "experiment_spec_relative_path",
        "initial_environment",
        "legacy_adopted",
        "created_at",
        "materialized_cohort_authority_required",
        "materialized_cohort_authority_ref",
    }
    prepared = prepare_existing_resume_input(
        run_dir=run_dir,
        resume_state={"per_step_records": []},
        scientific_identity=identity,
        current_environment=environment,
        cohort=cohort_path,
        question=QUESTION,
        resume_from_step_id=None,
        enforcement_mode="soft",
        load_compatible_plan=lambda **_kwargs: (None, None),
    )
    assert prepared.trajectory_binding is not None
    assert prepared.trajectory_binding.path == legacy_trajectory_path
    assert prepared.trajectory_binding.sha256 == sha256_of_file(legacy_trajectory_path)
    assert prepared.trajectory_binding.size == legacy_trajectory_path.stat().st_size
    assert prepared.trajectory_binding.authority_ref is None
    receipt = prepared.trajectory_binding.legacy_capsule_receipt
    assert receipt is not None

    pipeline = ResearchAgentPipeline(
        workdir=tmp_path / "runner_work",
        llm=MockLLMClient(),
        enable_memory=False,
        runner_kind="subprocess",
    )
    runner = pipeline._build_runner(
        run_dir=run_dir,
        cohort_path=cohort_path,
        universe_path=cohort_path,
        universe_is_typed=True,
        universe_authority_ref=staged_cohort.reference,
        trajectory_path=legacy_trajectory_path,
        trajectory_legacy_capsule_receipt=receipt,
    )
    result = runner.run(
        step_id="legacy_v2_smoke",
        code=(
            "import os\n"
            "from pathlib import Path\n"
            "assert Path(os.environ['TRAJECTORY_PARQUET']).name == "
            "'cohort_trajectory.parquet'\n"
        ),
    )
    assert result.returncode == 0

    authority_state = ExecutionInputAuthorityState.bind(
        universe_path=cohort_path,
        analysis_path=run_dir / "cohort_analysis.parquet",
        trajectory_binding=prepared.trajectory_binding,
        run_dir=run_dir,
        legacy_trajectory_verifier=verify_legacy_trajectory_capsule_receipt,
        plan=None,
        context=None,
    )
    assert authority_state.trajectory_authority_sha256 == receipt.capsule_sha256
    assert authority_state.trajectory_integrity_finding(step_id="before") is None
    capsule_path.write_text("{}", encoding="utf-8")
    finding = authority_state.trajectory_integrity_finding(step_id="after")
    assert finding is not None
    assert finding.validator == "execution_input_authority_integrity"


def test_forged_v2_capsule_receipt_without_evidence_authority_is_rejected(tmp_path):
    paths, source_cohort, _source_trajectory = _bundle(tmp_path)
    run_dir = tmp_path / "forged_v2_run"
    run_dir.mkdir()
    cohort_path = run_dir / "cohort.parquet"
    staged_cohort = stage_materialized_cohort_authority(
        paths["parquet"],
        cohort_path,
        expected_source_authority=source_cohort.reference,
        producer_implementation_sha256=_implementation_sha(),
    )
    assert staged_cohort is not None
    identity = _identity(
        cohort_path=paths["parquet"],
        cohort_ref=source_cohort.reference,
        trajectory_path=paths["trajectory"],
    )
    trajectory_identity = identity["cohort"]["trajectory"]
    trajectory_path = stage_legacy_trajectory_exact(
        paths["trajectory"],
        run_dir / "cohort_trajectory.parquet",
        expected_sha256=trajectory_identity["sha256"],
        expected_size=trajectory_identity["size_bytes"],
    )
    capsule = RunInputCapsuleV2(
        scientific_identity=identity,
        scientific_identity_sha256=canonical_sha256(identity),
        context_sha256="1" * 64,
        cohort_sha256=sha256_of_file(cohort_path),
        trajectory_relative_path=trajectory_path.name,
        trajectory_sha256=sha256_of_file(trajectory_path),
        initial_environment=build_environment_identity(llm_signature="mock"),
        materialized_cohort_authority_ref=staged_cohort.reference.to_dict(),
    )
    capsule_path = run_dir / RUN_INPUT_CAPSULE_FILENAME
    capsule_path.write_text(capsule.model_dump_json(indent=2), encoding="utf-8")
    receipt = VerifiedLegacyTrajectoryCapsuleReceipt(
        capsule_sha256=sha256_of_file(capsule_path),
        trajectory_relative_path=trajectory_path.name,
        trajectory_sha256=sha256_of_file(trajectory_path),
        trajectory_size=trajectory_path.stat().st_size,
        universe_authority_sha256=staged_cohort.reference.sha256,
    )
    pipeline = ResearchAgentPipeline(
        workdir=tmp_path / "runner_work",
        llm=MockLLMClient(),
        enable_memory=False,
        runner_kind="subprocess",
    )
    with pytest.raises(RunInputIdentityError, match="evidence authority"):
        pipeline._build_runner(
            run_dir=run_dir,
            cohort_path=cohort_path,
            universe_path=cohort_path,
            universe_is_typed=True,
            universe_authority_ref=staged_cohort.reference,
            trajectory_path=trajectory_path,
            trajectory_legacy_capsule_receipt=receipt,
        )


@pytest.mark.parametrize("invalidity", ("scientific_digest", "cohort_lineage"))
def test_selected_v2_receipt_rechecks_capsule_semantics(tmp_path, invalidity):
    paths, source_cohort, _source_trajectory = _bundle(tmp_path)
    run_dir = tmp_path / f"invalid_{invalidity}"
    run_dir.mkdir()
    cohort_path = run_dir / "cohort.parquet"
    staged_cohort = stage_materialized_cohort_authority(
        paths["parquet"],
        cohort_path,
        expected_source_authority=source_cohort.reference,
        producer_implementation_sha256=_implementation_sha(),
    )
    assert staged_cohort is not None
    identity = _identity(
        cohort_path=paths["parquet"],
        cohort_ref=source_cohort.reference,
        trajectory_path=paths["trajectory"],
    )
    if invalidity == "cohort_lineage":
        identity = json.loads(json.dumps(identity))
        identity["materialized_cohort_authority_ref"]["sha256"] = "f" * 64
    trajectory_identity = identity["cohort"]["trajectory"]
    trajectory_path = stage_legacy_trajectory_exact(
        paths["trajectory"],
        run_dir / "cohort_trajectory.parquet",
        expected_sha256=trajectory_identity["sha256"],
        expected_size=trajectory_identity["size_bytes"],
    )
    identity_sha256 = canonical_sha256(identity)
    if invalidity == "scientific_digest":
        identity_sha256 = "0" * 64
    capsule = RunInputCapsuleV2(
        scientific_identity=identity,
        scientific_identity_sha256=identity_sha256,
        context_sha256="1" * 64,
        cohort_sha256=sha256_of_file(cohort_path),
        trajectory_relative_path=trajectory_path.name,
        trajectory_sha256=sha256_of_file(trajectory_path),
        initial_environment=build_environment_identity(llm_signature="mock"),
        materialized_cohort_authority_ref=staged_cohort.reference.to_dict(),
    )
    capsule_path = run_dir / RUN_INPUT_CAPSULE_FILENAME
    capsule_path.write_text(capsule.model_dump_json(indent=2), encoding="utf-8")
    evidence = EvidenceStore(run_dir)
    evidence.register_file(
        kind="log",
        description="Invalid archived V2 capsule adversarial fixture.",
        source_path=capsule_path,
        evidence_id=RUN_INPUT_CAPSULE_EVIDENCE_ID,
        producer="pipeline",
        generation_mode="system",
    )
    receipt = VerifiedLegacyTrajectoryCapsuleReceipt(
        capsule_sha256=sha256_of_file(capsule_path),
        trajectory_relative_path=trajectory_path.name,
        trajectory_sha256=sha256_of_file(trajectory_path),
        trajectory_size=trajectory_path.stat().st_size,
        universe_authority_sha256=staged_cohort.reference.sha256,
    )
    pipeline = ResearchAgentPipeline(
        workdir=tmp_path / "runner_work",
        llm=MockLLMClient(),
        enable_memory=False,
        runner_kind="subprocess",
    )

    with pytest.raises(RunInputIdentityError):
        pipeline._build_runner(
            run_dir=run_dir,
            cohort_path=cohort_path,
            universe_path=cohort_path,
            universe_is_typed=True,
            universe_authority_ref=staged_cohort.reference,
            trajectory_path=trajectory_path,
            trajectory_legacy_capsule_receipt=receipt,
        )


def test_trajectory_integrity_performs_one_typed_envelope_scan(
    tmp_path,
    monkeypatch,
):
    import easyicu.research_agent.authority.execution_input as execution_input

    trajectory_path = tmp_path / "cohort_trajectory.parquet"
    trajectory_path.write_bytes(b"trajectory")
    trajectory_sha256 = sha256_of_file(trajectory_path)
    universe_ref = MaterializedCohortAuthorityRef(
        file="materialized_authority.json",
        sha256="1" * 64,
        size=1,
    )
    trajectory_ref = MaterializedTrajectoryAuthorityRef(
        file="trajectory_authority.json",
        sha256="2" * 64,
        size=1,
    )
    binding = StagedTrajectoryBinding(
        path=trajectory_path,
        sha256=trajectory_sha256,
        size=trajectory_path.stat().st_size,
        authority_ref=trajectory_ref,
    )
    calls: list[str] = []

    def verify_once(*_args, **_kwargs):
        calls.append("envelope")
        return SimpleNamespace(
            trajectory_sha256=trajectory_sha256,
            trajectory_size=trajectory_path.stat().st_size,
        )

    monkeypatch.setattr(
        execution_input,
        "verify_materialized_trajectory_envelope",
        verify_once,
    )
    monkeypatch.setattr(
        execution_input,
        "_sha256_file",
        lambda _path: pytest.fail("typed envelope must not be hashed twice"),
    )
    state = ExecutionInputAuthorityState(
        universe_path=tmp_path / "cohort.parquet",
        analysis_path=tmp_path / "cohort_analysis.parquet",
        cohort_authority=ExecutionCohortAuthority(
            selected_path=tmp_path / "cohort.parquet",
            universe_path=tmp_path / "cohort.parquet",
            universe_authority=SimpleNamespace(reference=universe_ref),
            analysis_authority=None,
        ),
        trajectory_binding=binding,
        run_dir=tmp_path,
    )

    assert state.trajectory_integrity_finding(step_id="01") is None
    assert calls == ["envelope"]


def test_explicit_non_sibling_trajectory_replaces_ambient_sibling_identity(tmp_path):
    cohort = tmp_path / "cohort.parquet"
    sibling = tmp_path / "cohort_trajectory.parquet"
    selected = tmp_path / "selected.parquet"
    pd.DataFrame({"stay_id": [1]}).to_parquet(cohort, index=False)
    sibling.write_bytes(b"ambient sibling")
    selected.write_bytes(b"selected trajectory")

    identity = _identity(cohort_path=cohort, trajectory_path=selected)

    assert "trajectory" not in identity["cohort"]
    assert (
        identity["trajectory"]["sha256"]
        != build_scientific_identity(
            cohort=cohort,
            question=QUESTION,
            cohort_name="typed_trajectory_capsule",
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
        )["cohort"]["trajectory"]["sha256"]
    )


def test_scientific_trajectory_identity_rejects_leaf_symlink(tmp_path):
    cohort = tmp_path / "cohort.parquet"
    target = tmp_path / "trajectory-target.parquet"
    link = tmp_path / "trajectory-link.parquet"
    pd.DataFrame({"stay_id": [1]}).to_parquet(cohort, index=False)
    target.write_bytes(b"trajectory")
    link.symlink_to(target)

    with pytest.raises(RunInputIdentityError, match="missing or unsafe"):
        _identity(cohort_path=cohort, trajectory_path=link)


def test_scientific_trajectory_envelope_rejects_conflicting_double_binding():
    with pytest.raises(RunInputIdentityError, match="conflicting bytes"):
        _scientific_trajectory_envelope(
            {
                "trajectory": {"sha256": "a" * 64, "size_bytes": 1},
                "cohort": {"trajectory": {"sha256": "b" * 64, "size_bytes": 2}},
            }
        )
