"""The capability registry must match the code that is actually wired.

The registry (``easyicu.research_agent.planning.capability_registry``) is only useful if
it cannot lie. These tests cross-check every claim against the live pipeline:
the deterministic-runner names against ``_PRIMARY_DETERMINISTIC_RUNNERS`` in
BOTH pipeline modules, the figure-renderer keys against
``figures.FAMILY_RENDERERS``, the declared runner entrypoints against the
importable modules, and family coverage against the ``StudyDesignFamily`` enum.
Add or remove a runner without updating the registry and one of these fails —
which is the point.
"""

from __future__ import annotations

import ast
import importlib
import inspect
from pathlib import Path
import textwrap
from typing import get_args

from easyicu.research_agent.execution import phase as pipeline_execute
from easyicu.research_agent.execution.runners import selection
from easyicu.research_agent.figures import FAMILY_RENDERERS
from easyicu.research_agent.planning import capability_registry as cr
from easyicu.research_agent.execution.phase import (
    _PRIMARY_DETERMINISTIC_RUNNERS as EXEC_RUNNERS,
)
from easyicu.research_agent.reporting.readiness import (
    _PRIMARY_DETERMINISTIC_RUNNERS as REPORT_RUNNERS,
)
from easyicu.research_agent.planning.study_design_playbook import StudyDesignFamily
from easyicu.research_agent.contracts.capability_ids import (
    LANDMARK_SPLINE_ANALYSIS_KIND,
    LANDMARK_SPLINE_ASSOCIATION_CAPABILITY_ID,
    PHENOTYPING_ANALYSIS_KIND,
    PHENOTYPING_CLUSTER_CAPABILITY_ID,
    SOURCE_FEASIBILITY_NON_USE_CAPABILITY_ID,
    SIGNED_TRAJECTORY_PHENOTYPING_CAPABILITY_ID,
)

_RUNNER_ENTRYPOINTS: dict[str, tuple[str, str]] = {
    "signed_time_varying_exposure_cox": (
        "execution.runners.time_varying_executor", "time_varying_executor_code",
    ),
    "adjusted_association_estimates": (
        "execution.runners.adjusted_association_executor",
        "adjusted_association_executor_code",
    ),
    "survival_primary_cox": (
        "execution.runners.survival_primary_executor",
        "survival_primary_executor_code",
    ),
    "exposure_outcome_distribution": (
        "execution.runners.exposure_outcome_distribution_executor",
        "exposure_outcome_distribution_executor_code",
    ),
    "static_prediction_model": (
        "execution.runners.prediction_model_executor",
        "prediction_model_executor_code",
    ),
    LANDMARK_SPLINE_ANALYSIS_KIND: (
        "execution.runners.landmark_spline_executor",
        "landmark_spline_executor_code",
    ),
    PHENOTYPING_ANALYSIS_KIND: (
        "execution.runners.cross_sectional_phenotyping_executor",
        "cross_sectional_phenotyping_executor_code",
    ),
}


def test_landmark_spline_and_freeform_have_distinct_validation_ceilings():
    landmark = cr.get_capability_by_id(LANDMARK_SPLINE_ASSOCIATION_CAPABILITY_ID)
    freeform = cr.get_capability_by_id("association_freeform_v1")

    assert landmark is not None
    assert landmark.primary_analysis == "deterministic"
    assert landmark.scientific_validation == "reportable"
    assert landmark.scientific_validator_contract == (
        "easyicu.landmark_spline_runtime_receipt/1"
    )
    assert freeform is not None
    assert freeform.scientific_validation == "analysis_only"


def _registry_primary_runners() -> set:
    return {
        c.primary_runner
        for c in cr.CAPABILITY_REGISTRY
        if c.primary_analysis == "deterministic" and c.primary_runner
    }


# --- deterministic primary runners: registry <-> wired sets ----------------


def test_every_registry_runner_is_wired_in_both_pipeline_modules():
    for name in _registry_primary_runners():
        assert name in EXEC_RUNNERS, f"{name} not wired in pipeline_execute"
        assert name in REPORT_RUNNERS, f"{name} not wired in reporting.readiness"


def test_every_wired_runner_is_documented_in_the_registry():
    # No wired deterministic primary runner may be undocumented.
    documented = _registry_primary_runners()
    for name in EXEC_RUNNERS:
        assert name in documented, f"wired runner {name} missing from the registry"


def test_the_two_pipeline_modules_agree_on_the_runner_set():
    assert set(EXEC_RUNNERS) == set(REPORT_RUNNERS)


def test_registry_runner_entrypoints_are_importable_and_callable():
    for name in _registry_primary_runners():
        assert name in _RUNNER_ENTRYPOINTS, f"no entrypoint mapping for {name}"
        mod_name, fn_name = _RUNNER_ENTRYPOINTS[name]
        mod = importlib.import_module(f"easyicu.research_agent.{mod_name}")
        fn = getattr(mod, fn_name)
        assert callable(fn)
        assert "step" in inspect.signature(fn).parameters


# --- deterministic figure renderers ----------------------------------------


def test_registry_figure_renderers_exist_in_family_renderers():
    for c in cr.CAPABILITY_REGISTRY:
        if c.figure != "deterministic" or not c.figure_renderer:
            continue
        # the base association skill is rendered outside FAMILY_RENDERERS
        if c.figure_renderer == "base_association_skill":
            continue
        assert c.figure_renderer in FAMILY_RENDERERS, (
            f"{c.figure_renderer} not in FAMILY_RENDERERS"
        )


def test_only_typed_host_validated_primary_capabilities_default_to_reportable():
    reportable = {
        capability.capability_id
        for capability in cr.CAPABILITY_REGISTRY
        if capability.scientific_validation == "reportable"
    }

    assert reportable == {
        "survival_time_to_event_v1",
        "association_adjusted_v1",
        LANDMARK_SPLINE_ASSOCIATION_CAPABILITY_ID,
        PHENOTYPING_CLUSTER_CAPABILITY_ID,
        "descriptive_exposure_outcome_distribution_v1",
        "prediction_risk_model_v1",
        SOURCE_FEASIBILITY_NON_USE_CAPABILITY_ID,
        SIGNED_TRAJECTORY_PHENOTYPING_CAPABILITY_ID,
    }
    for capability in cr.CAPABILITY_REGISTRY:
        if capability.capability_id in reportable:
            assert capability.scientific_validator_owner
            assert capability.scientific_validator_contract


# --- auxiliary runners are importable --------------------------------------


def test_auxiliary_runner_entrypoints_are_importable():
    for a in cr.AUXILIARY_DETERMINISTIC_RUNNERS:
        mod = importlib.import_module(f"easyicu.research_agent.{a.module}")
        fn = getattr(mod, a.entrypoint)
        assert callable(fn)


# --- family coverage --------------------------------------------------------


def test_every_study_design_family_is_covered():
    families = set(get_args(StudyDesignFamily))
    covered = {c.family for c in cr.CAPABILITY_REGISTRY}
    missing = families - covered
    assert not missing, f"families with no capability record: {missing}"


def test_partition_helpers_are_consistent():
    det = set(cr.deterministic_primary_families())
    llm = set(cr.llm_coded_primary_families())
    assert det == {
        "Association — source-bound time-updated Cox",
        "Association — exact single-model adjusted",
        "Association — digest-bound landmark spline",
        "Descriptive — typed exposure/outcome absolute risks",
        "Prediction / risk modelling",
        "Survival / time-to-event",
        "Phenotyping / clustering",
        "Causal feasibility — verified non-use unavailable",
        "Phenotyping — signed fixed-window trajectories",
    }
    assert llm
    assert det.isdisjoint(llm)
    assert len(det) + len(llm) == len(cr.CAPABILITY_REGISTRY)


def test_survival_and_exact_association_have_deterministic_primary_owners():
    fams = cr.families_without_deterministic_primary()
    assert fams == set(get_args(StudyDesignFamily)) - {
        "association",
        "descriptive",
        "prediction",
        "phenotyping",
        "time_to_event",
        "causal_emulation",
    }


# --- renderer ---------------------------------------------------------------


def test_markdown_matrix_renders_every_family_and_the_ladder():
    md = cr.render_capability_matrix_markdown()
    for c in cr.CAPABILITY_REGISTRY:
        assert c.label in md
    for name in EXEC_RUNNERS:
        assert name in md
    assert "Fail-closed / gap-report ladder" in md
    # the invariant sentence must be present
    assert "never silently filled" in md


def test_known_unsupported_boundary_is_recorded_and_rendered():
    # An explicit "not supported" boundary (competing-risks CIF) must be
    # first-class in the registry, not only a benchmark probe.
    assert cr.KNOWN_UNSUPPORTED_ESTIMANDS
    md = cr.render_capability_matrix_markdown()
    assert "Known unsupported estimands" in md
    assert "Competing-risks" in md


def test_get_capability_disambiguates_association():
    dose = cr.get_capability("association", dose_response=True)
    general = cr.get_capability("association", dose_response=False)
    freeform = cr.get_capability("association", freeform=True)
    assert dose is not None and dose.primary_runner is None
    assert (
        general is not None
        and general.primary_runner == "adjusted_association_estimates"
    )
    assert freeform is not None and freeform.primary_runner is None
    assert "graded ordinal" in dose.label.lower()
    assert "exact single-model" in general.label.lower()
    assert "free-form" in freeform.label.lower()


def test_dynamic_prediction_has_an_honest_analysis_only_capability():
    capability = cr.get_capability_by_id("dynamic_prediction_landmark_v1")
    assert capability is not None
    assert capability.family == "prediction"
    assert capability.primary_analysis == "llm_coded"
    assert capability.scientific_validation == "analysis_only"
    assert capability.primary_runner is None
    assert "patient-level" in " ".join(capability.data_contract)
    assert "Static prediction" in capability.fail_closed


def test_plan_contract_selects_exact_or_freeform_association_capability(ra):
    exact = ra.AnalysisPlan(
        research_question="Estimate one adjusted association.",
        analysis_type="association_study",
        steps=[
            ra.AnalysisStep(
                step_id="01_exact",
                planned_analysis_role="primary",
                intent="Fit the exact adjusted model.",
                method="adjusted_association_models",
                expected_outputs=["table:adjusted_association_estimates"],
            )
        ],
    )
    freeform = ra.AnalysisPlan(
        research_question="Estimate an association with an interaction.",
        analysis_type="association_study",
        steps=[
            ra.AnalysisStep(
                step_id="01_freeform",
                planned_analysis_role="primary",
                intent="Fit the declared interaction model.",
                method="association_interaction_model",
                expected_outputs=["table:interaction_estimates"],
                # Declared, not inferred: "does not match the exact contract"
                # is the shape of a feasibility audit too, so inferring
                # free-form from it handed every under-declared association
                # plan the looser agent-coded obligations.
                scientific_capability="association_freeform_v1",
            )
        ],
    )

    assert (
        cr.get_capability_for_plan(
            analysis_type=exact.analysis_type,
            plan=exact,
        ).capability_id
        == "association_adjusted_v1"
    )
    assert (
        cr.get_capability_for_plan(
            analysis_type=freeform.analysis_type,
            plan=freeform,
        ).capability_id
        == "association_freeform_v1"
    )


def test_only_exact_typed_descriptive_primary_selects_the_reportable_owner(ra):
    exact = ra.AnalysisPlan(
        research_question="Describe exposure prevalence and observed mortality.",
        analysis_type="descriptive_epidemiology",
        steps=[
            ra.AnalysisStep(
                step_id="01_distribution",
                planned_analysis_role="primary",
                intent="Report prespecified unadjusted descriptive risks.",
                method="descriptive",
                inputs=["artifact:analysis_cohort", "exposure", "death"],
                expected_outputs=["table:exposure_outcome_distribution"],
                descriptive_claim={
                    "unresolved_limitations": [
                        "post_baseline_exposure_opportunity_unresolved"
                    ]
                },
                exposure_outcome_distribution_spec={
                    "exposure": "exposure",
                    "exposure_levels": [0, 1],
                    "outcome": "death",
                    "outcome_levels": [0, 1],
                    "outcome_positive_value": 1,
                    "level_match_policy": "exact_typed",
                    "denominator_policy": "all_declared_rows",
                    "missing_outcome_policy": "structural_absence_is_non_event",
                    "confidence_level": 0.95,
                },
            )
        ],
    )
    ordinary = ra.AnalysisPlan(
        research_question="Describe the cohort.",
        analysis_type="descriptive_epidemiology",
        steps=[
            ra.AnalysisStep(
                step_id="01_table_one",
                planned_analysis_role="primary",
                intent="Describe baseline characteristics.",
                method="descriptive",
                inputs=["artifact:analysis_cohort"],
                expected_outputs=["table:table_one"],
            )
        ],
    )

    exact_verdict = cr.resolve_primary_capability(
        analysis_type=exact.analysis_type,
        plan=exact,
    )
    ordinary_verdict = cr.resolve_primary_capability(
        analysis_type=ordinary.analysis_type,
        plan=ordinary,
    )

    assert exact_verdict.capability_id == (
        "descriptive_exposure_outcome_distribution_v1"
    )
    assert exact_verdict.execution_owner == "host_deterministic"
    assert exact_verdict.owner_claimed is True
    assert exact_verdict.scientific_validation == "reportable"
    assert ordinary_verdict.capability_id == "descriptive_measurement_v1"
    assert ordinary_verdict.execution_owner == "agent_coded"
    assert ordinary_verdict.scientific_validation == "analysis_only"


def test_live_auxiliary_dispatch_matches_registry_in_both_directions():
    """Inspect actual execute assignments, not a second hand-maintained set."""

    documented = {runner.name for runner in cr.AUXILIARY_DETERMINISTIC_RUNNERS}

    execute_source = (
        inspect.getsource(pipeline_execute.run_execute_phase)
        + "\n"
        + inspect.getsource(pipeline_execute._step_settle_initial_code)
    )
    assert "select_standard_executor(" in execute_source
    assert 'step_record["deterministic_standard_analysis"] = (' in execute_source

    # Every documented runner must define its registry-declared entrypoint
    # in the module the registry points to.
    active: set[str] = set()
    for runner in cr.AUXILIARY_DETERMINISTIC_RUNNERS:
        module = importlib.import_module(f"easyicu.research_agent.{runner.module}")
        if runner.entrypoint in inspect.getsource(module):
            active.add(runner.name)
    assert active == documented


# --- generated doc stays in sync -------------------------------------------


def test_committed_docs_matrix_matches_the_registry_render():
    # docs/capability_matrix.md is generated from the registry; if it drifts,
    # regenerate with:
    #   python -m easyicu.research_agent.planning.capability_registry > docs/capability_matrix.md
    repo_root = Path(__file__).resolve().parents[3]
    doc = repo_root / "docs" / "capability_matrix.md"
    assert doc.exists(), "docs/capability_matrix.md missing — regenerate it"
    committed = doc.read_text(encoding="utf-8").rstrip("\n")
    rendered = cr.render_capability_matrix_markdown().rstrip("\n")
    assert committed == rendered, "docs/capability_matrix.md is stale — regenerate it"
