"""Phase-2 Coder Action/Software/Data production-wiring contracts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from easyicu.research_agent.authority.coder_authority import HostCoderAuthority
from easyicu.research_agent.resources import (
    CODER_RESOURCE_PROMPT_LIMIT_BYTES,
    CoderResourceIntegrityError,
    attach_coder_resources,
    build_coder_resource_bundle,
)
from easyicu.research_agent.orchestration.profiles import get_submission_profile


def _bundle(*, profile_ref: str = "npj_dm_framework_v2_dev/20260722"):
    return build_coder_resource_bundle(
        step_id="02_table_one",
        profile_ref=profile_ref,
        analysis_family="association",
        step_role="auxiliary",
        question="Summarise a cohort by exposure group.",
        intent="Build the declared Table One.",
        method="table_one",
        planner_inputs=("cohort:analysis_cohort", "lactate", "death"),
        expected_outputs=("table:table_one",),
        resolved_input_bindings={
            "cohort:analysis_cohort": {
                "evidence_id": "development_execution_cohort",
                "sha256": "1" * 64,
            }
        },
        runtime_import_names=(
            "pandas",
            "numpy",
            "scipy",
            "matplotlib",
            "statsmodels",
            "sklearn",
            "pyarrow",
        ),
        has_table_one_spec=True,
    )


def test_bundle_selects_all_three_resource_kinds_without_provider_calls() -> None:
    first = _bundle()
    second = _bundle()

    assert first == second
    assert first.provider_calls == 0
    assert [receipt.policy.allowed_kinds for receipt in first.selections] == [
        ("action",),
        ("software",),
        ("data",),
    ]
    assert all(receipt.provider_calls == 0 for receipt in first.selections)
    assert first.prompt_bytes <= CODER_RESOURCE_PROMPT_LIMIT_BYTES
    selected_kinds = {
        selected.kind for receipt in first.selections for selected in receipt.selected
    }
    assert selected_kinds == {"action", "software", "data"}


def test_zero_match_is_legal_and_still_receipted() -> None:
    bundle = build_coder_resource_bundle(
        step_id="01_unknown",
        profile_ref="npj_dm_framework_v2_dev/20260722",
        analysis_family="unknown_family",
        step_role="auxiliary",
        question="Unmatched task vocabulary.",
        intent="Unmatched intent vocabulary.",
        method=None,
        planner_inputs=(),
        expected_outputs=(),
        resolved_input_bindings={},
        runtime_import_names=(),
    )

    assert bundle.prompt_projection == ""
    assert bundle.prompt_bytes == 0
    assert all(receipt.selected == () for receipt in bundle.selections)


def test_coder_resources_require_the_additive_profile(ra, tmp_path: Path) -> None:
    profile = get_submission_profile("npj_dm_framework_v2_dev/20260722")
    assert profile.as_pipeline_options()["enable_coder_resources"] is True
    assert profile.as_pipeline_options()["enable_memory"] is False
    assert (
        "enable_coder_resources"
        not in get_submission_profile("npj_dm/20260719").as_pipeline_options()
    )

    with pytest.raises(ValueError, match="require an additive submission profile"):
        ra.ResearchAgentPipeline(
            workdir=tmp_path,
            enable_memory=False,
            enable_coder_resources=True,
        )


def test_persisted_bundle_is_bound_into_host_coder_authority(tmp_path: Path) -> None:
    bundle = _bundle()
    authority, path = attach_coder_resources(
        authority=HostCoderAuthority(), run_dir=tmp_path, bundle=bundle
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    attachment = json.loads(authority.attachments[-1])
    assert payload["profile_ref"] == bundle.profile_ref
    assert attachment["bundle_sha256"] == bundle.sha256
    assert attachment["receipt_path"] == ("resource_selections/coder/02_table_one.json")
    assert attachment["provider_calls"] == 0
    assert attachment["selected_context"] == bundle.prompt_projection


def test_receipt_tamper_and_wrong_profile_fail_closed(tmp_path: Path) -> None:
    bundle = _bundle()
    _, path = attach_coder_resources(
        authority=HostCoderAuthority(), run_dir=tmp_path, bundle=bundle
    )
    path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(CoderResourceIntegrityError, match="changed"):
        attach_coder_resources(
            authority=HostCoderAuthority(), run_dir=tmp_path, bundle=bundle
        )

    path.unlink()
    attach_coder_resources(
        authority=HostCoderAuthority(), run_dir=tmp_path, bundle=bundle
    )
    with pytest.raises(CoderResourceIntegrityError, match="changed"):
        attach_coder_resources(
            authority=HostCoderAuthority(),
            run_dir=tmp_path,
            bundle=_bundle(profile_ref="another_profile/20260722"),
        )


def test_pipeline_production_callsite_persists_and_binds_resources(
    ra, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from easyicu.research_agent.authority.step_capsule import (
        StepAuthorityCapsuleRef,
        load_verified_step_authority_capsule,
        read_verified_content,
    )
    from easyicu.research_agent.execution import phase as execute_phase
    from tests.research_agent.integration import (
        test_trajectory_stability_pipeline_success as fixture,
    )

    fixture._disable_unrelated_audits(monkeypatch)
    coder_authorities: list[str] = []
    original_coder_run = execute_phase.CoderAgent.run

    def observed_coder_run(self, *args, **kwargs):
        coder_authorities.append(kwargs["host_authority"].render())
        return original_coder_run(self, *args, **kwargs)

    monkeypatch.setattr(execute_phase.CoderAgent, "run", observed_coder_run)
    llm = fixture._PlanAndCoderLLM()
    runners: dict[float, object] = {}

    def runner_factory(*, workdir, timeout_seconds, **_kwargs):
        timeout = float(timeout_seconds)
        if timeout not in runners:
            runners[timeout] = fixture._HybridTrajectoryRunner(workdir=Path(workdir))
        return runners[timeout]

    pipeline = ra.ResearchAgentPipeline(
        workdir=tmp_path,
        llm=llm,
        timeout_seconds=17.0,
        standard_executor_timeout_seconds=1_234.0,
        runner_factory=runner_factory,
        enable_literature=False,
        enable_visual_qa=False,
        enable_latex=False,
        enable_llm_concept_audit=False,
        enable_replanning=False,
        enable_deterministic_code_fallback=True,
        enable_deterministic_runner_repair=False,
        max_code_repair_attempts=2,
        submission_profile_name="npj_dm_framework_v2_dev",
        submission_profile_version="20260722",
        enable_coder_resources=True,
    )
    cohort = pd.DataFrame(
        {
            "stay_id": list(range(1, 25)),
            "marker_h0_6": np.linspace(-1.0, 1.0, 24),
            "marker_h6_12": np.linspace(-0.5, 1.5, 24),
            "death": [0, 1] * 12,
        }
    )
    result = pipeline.run(
        question="Assess fixed-window trajectory phenotypes.",
        cohort=cohort,
        cohort_name="coder_resource_production_wiring",
        database="synthetic",
        target_outcome="death",
        stop_after_step_id="01_representation",
        stop_after_analysis=True,
    )

    run_dir = Path(result.workdir)
    resource_path = run_dir / "resource_selections" / "coder" / "01_representation.json"
    assert resource_path.is_file()
    payload = json.loads(resource_path.read_text(encoding="utf-8"))
    assert payload["profile_ref"] == "npj_dm_framework_v2_dev/20260722"
    assert payload["provider_calls"] == 0
    manifest = json.loads(
        (run_dir / "manifest_partial.json").read_text(encoding="utf-8")
    )
    record = next(
        row
        for row in manifest["per_step_records"]
        if row.get("step_id") == "01_representation"
    )
    assert record["coder_resource_provider_calls"] == 0
    assert record["coder_resource_selection_path"] == (
        "resource_selections/coder/01_representation.json"
    )
    assert coder_authorities
    assert '"schema_version":"easyicu.coder_resource_prompt/1"' in (
        coder_authorities[0]
    )
    capsule_ref = StepAuthorityCapsuleRef.model_validate(
        record["step_authority_capsule_ref"]
    )
    verified = load_verified_step_authority_capsule(run_dir, ref=capsule_ref)
    scoped_payload = json.loads(
        read_verified_content(run_dir, verified.capsule.scoped_coder_context)
    )
    attachments = scoped_payload["host_coder_authority"]["attachments"]
    assert any("easyicu.coder_resource_prompt/1" in value for value in attachments)


@pytest.mark.parametrize(
    ("task_id", "family", "question"),
    [
        ("E1", "descriptive", "Sepsis prevalence and mortality association"),
        ("E2", "association", "Peak lactate and mortality association"),
        ("E3", "association", "KDIGO stage gradient for mortality and LOS"),
        ("M1", "association", "Hepatobiliary missingness and mortality"),
        ("M2", "prediction", "First-24h mortality prediction model"),
        ("M3", "phenotyping", "Sepsis subphenotype clustering"),
        ("H1", "time_to_event", "Ventilation and 28-day survival"),
        ("H2", "causal_emulation", "Vasopressor comparative effectiveness"),
        ("H3", "phenotyping", "Longitudinal trajectory clustering"),
    ],
)
def test_canonical9_offline_coder_resource_matrix(
    task_id: str, family: str, question: str
) -> None:
    bundle = build_coder_resource_bundle(
        step_id=f"{task_id.lower()}_analysis",
        profile_ref="npj_dm_framework_v2_dev/20260722",
        analysis_family=family,
        step_role="primary",
        question=question,
        intent=question,
        method=family,
        planner_inputs=("cohort:analysis_cohort", "subject_id"),
        expected_outputs=("table:primary_result",),
        resolved_input_bindings={
            "cohort:analysis_cohort": {
                "evidence_id": "analysis_cohort",
                "sha256": "2" * 64,
            }
        },
        runtime_import_names=(
            "pandas",
            "numpy",
            "scipy",
            "matplotlib",
            "statsmodels",
            "sklearn",
            "pyarrow",
            "lifelines",
        ),
    )

    assert bundle.provider_calls == 0
    assert bundle.prompt_bytes <= CODER_RESOURCE_PROMPT_LIMIT_BYTES
    assert len(bundle.selections) == 3
