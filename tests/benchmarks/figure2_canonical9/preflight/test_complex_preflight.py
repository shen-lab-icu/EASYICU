"""Zero-Provider preflight coverage for M2, H1, H2, and H3.

These are diagnostic fixtures, never paper results.  Structural and reviewed-
code checks always run; the full real-pipeline graph runs only when the exact
CodeRunner isolation capability is available.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

from benchmarks.figure2_canonical9.case_scientific_protocol import (
    ScientificCaseProtocolError,
    default_case_protocol_path,
    load_default_case_protocol,
)
from benchmarks.figure2_canonical9.preflight.fixtures import (
    COMPLEX_CASES,
    H1,
    H2,
    H3,
    M2,
    PreflightCase,
)
from benchmarks.figure2_canonical9.preflight.harness import (
    PreflightRun,
    ScriptedPreflightLLM,
    paper_acceptance_verdict,
    preflight_runtime_manifest,
    provider_transport_spy,
    run_preflight,
)
from easyicu.research_agent.agents.core import PlannerAgent
from easyicu.research_agent.execution.runner import CodeRunner
from easyicu.research_agent.execution.runners.selection import (
    select_standard_executor,
)
from easyicu.research_agent.providers.protocol import LLMMessage
from easyicu.research_agent.schema import CohortDescriptor, ResearchContext

CASES = list(COMPLEX_CASES.values())


def _sealed_authority_block(case: PreflightCase) -> tuple[str, str] | None:
    """Name the sealed authority that forbids a primary result, or return None.

    ``validate_required_primary_result`` demands a
    ``family_primary_result_requirement`` from every ``survival`` /
    ``causal_inference`` plan that declares an exposure and an outcome. That
    requirement is a scientific declaration -- estimator, effect scale,
    population, and for causal families an estimand and comparator -- so it may
    only come from a reviewed source. This resolves that source per case and
    returns the blocker code when there is none, so the classification below
    tracks the sealed protocols rather than a hand-maintained list of ids:
    publish a reviewed H1 protocol and H1 moves to the finalising set by
    itself.
    """

    if case.analysis_type not in {"survival", "causal_inference"}:
        return None
    try:
        default_case_protocol_path(case.task_id)
    except ScientificCaseProtocolError:
        return (
            "SCIENTIFIC_CASE_PROTOCOL_UNKNOWN_TASK",
            "no reviewed case scientific protocol exists for this task",
        )
    protocol = load_default_case_protocol(case.task_id)
    if str(getattr(protocol, "review_status", "")).startswith("ai_development_"):
        return (
            "AI_DEVELOPMENT_REVIEW_ONLY",
            "the development protocol still requires human scientific attestation",
        )
    capture = getattr(protocol, "current_source_capture", None)
    if capture is not None and capture.causal_contrast_authorized is False:
        return (
            capture.reason_code,
            "the sealed capture contract does not authorize a causal contrast",
        )
    return None


AUTHORITY_BLOCKED_CASES = [c for c in CASES if _sealed_authority_block(c) is not None]
FINALISING_CASES = [c for c in CASES if _sealed_authority_block(c) is None]


@pytest.fixture(params=CASES, ids=[case.task_id for case in CASES])
def case(request) -> PreflightCase:
    return request.param


def _reviewed_primary_code(case: PreflightCase) -> tuple[str, ScriptedPreflightLLM]:
    controller = ScriptedPreflightLLM(case)
    code = controller.client.complete(
        [
            LLMMessage(
                role="user",
                content=f"WRITE THE PYTHON CODE {case.primary_step_id}",
            )
        ]
    )
    return code, controller


def test_authority_classification_partitions_every_complex_case() -> None:
    """No case may fall out of both suites.

    The two graph suites are parametrized over disjoint subsets. If
    ``_sealed_authority_block`` ever returned something unexpected for a case,
    that case could silently stop being exercised by either suite while the
    file still reported all-green. This pins the partition instead.
    """

    blocked = {c.task_id for c in AUTHORITY_BLOCKED_CASES}
    finalising = {c.task_id for c in FINALISING_CASES}

    assert blocked | finalising == set(COMPLEX_CASES)
    assert not blocked & finalising
    assert len(AUTHORITY_BLOCKED_CASES) + len(FINALISING_CASES) == len(COMPLEX_CASES)
    # Today's expected split, so a change in either direction is a review event
    # rather than a silent reclassification.
    assert blocked == {"h1_ventilation_survival", "h2_vasopressor_causal"}


def test_complex_registry_is_exact_minimum_family_set() -> None:
    assert set(COMPLEX_CASES) == {
        "m2_mortality_prediction",
        "h1_ventilation_survival",
        "h2_vasopressor_causal",
        "h3_trajectory_clustering",
    }


def test_generated_blueprint_notes_cannot_widen_single_database_contract() -> None:
    """Generic prompt examples must not create a cross-database obligation."""

    authority_context = ResearchContext(
        research_question=M2.question,
        cohort=CohortDescriptor(
            cohort_name="m2_preflight",
            database=M2.database,
            n_stays=160,
            n_patients=80,
        ),
        variables=[],
        target_outcome=M2.target_outcome,
    )
    prompt_context = authority_context.model_copy(
        update={
            "notes": (
                "ANALYSIS BLUEPRINT: generic examples may discuss external "
                "validation and transportability, without requesting either."
            )
        }
    )

    parsed = PlannerAgent(object())._parse(
        M2.build_plan().model_dump_json(),
        prompt_context,
        enforce_article_contract=True,
        article_contract_context=authority_context,
    )

    assert parsed.analysis_type == "prediction_model"


def test_complex_plan_contract_and_typed_inputs(case: PreflightCase) -> None:
    plan = case.build_plan()
    cohort = case.build_cohort(160)

    assert plan.analysis_type == case.analysis_type
    assert [step.step_id for step in plan.steps].count(case.primary_step_id) == 1
    assert (
        next(
            step
            for step in plan.steps
            if step.step_id == case.primary_step_id
        ).planned_analysis_role
        == "primary"
    )
    missing_raw_inputs = {
        input_name
        for step in plan.steps
        for input_name in step.inputs
        if ":" not in input_name and input_name not in cohort.columns
    }
    assert missing_raw_inputs == set()

    table_one = next(
        step for step in plan.steps if step.step_id == case.deterministic_step_id
    )
    selection = select_standard_executor(table_one, plan=plan)
    assert selection is not None
    assert selection.analysis_kind == "grouped_table_one"


def test_complex_guardrails_are_live_and_structurally_covered(
    case: PreflightCase,
) -> None:
    assert len(case.guardrail_checks) == len(case.semantic_guardrails)
    assert sorted(check.guardrail_index for check in case.guardrail_checks) == list(
        range(len(case.semantic_guardrails))
    )
    failures = [
        (check.key, case.semantic_guardrails[check.guardrail_index])
        for check in case.guardrail_checks
        if not check.holds(case)
    ]
    assert failures == []


def test_complex_product_map_covers_live_suite_outputs(case: PreflightCase) -> None:
    mapped = case.product_mapping()
    assert [product for product, _mapping in mapped] == list(case.expected_products)


def test_complex_coder_dependencies_are_importable(case: PreflightCase) -> None:
    missing = [
        package
        for package in case.required_imports
        if importlib.util.find_spec(package) is None
    ]
    assert missing == []


def test_h3_has_independent_deterministic_stability_owner() -> None:
    plan = H3.build_plan()
    stability = next(step for step in plan.steps if step.step_id == "04_stability_freeze")
    selection = select_standard_executor(stability, plan=plan)

    assert selection is not None
    assert selection.analysis_kind == "trajectory_cluster_stability"
    assert "run_trajectory_stability" in selection.code


@pytest.mark.parametrize(
    ("case", "required_files"),
    [
        (
            M2,
            {
                "model_performance_train_test.csv",
                "risk_predictions_test.csv",
                "calibration_curve.csv",
                "decision_curve.csv",
                "split_definition.csv",
            },
        ),
        (
            H1,
            {"cox_summary.csv", "survival_curve.csv", "ph_diagnostics.csv"},
        ),
        (
            H2,
            {
                "causal_effect.csv",
                "covariate_balance.csv",
                "positivity_diagnostics.csv",
                "assignment_model.csv",
            },
        ),
        (
            H3,
            {
                "cluster_assignments.csv",
                "cluster_outcomes.csv",
                "cluster_trajectory_means.csv",
            },
        ),
    ],
    ids=["m2_prediction", "h1_survival", "h2_causal", "h3_trajectory"],
)
def test_reviewed_family_code_executes_with_zero_provider_calls(
    case: PreflightCase,
    required_files: set[str],
    tmp_path: Path,
) -> None:
    """Development smoke for static reviewed code, not a paper-authority run."""

    cohort_path = tmp_path / "cohort.parquet"
    case.build_cohort(160).to_parquet(cohort_path, index=False)
    with provider_transport_spy() as spy:
        code, _controller = _reviewed_primary_code(case)
        runner = CodeRunner(
            workdir=tmp_path / "run",
            cohort_parquet=cohort_path,
            timeout_seconds=60.0,
            network_policy="none",
            # Explicit development-only fallback: the formal graph test below
            # still requires exact non-degraded isolation.
            allow_unsafe_host_fallback=True,
        )
        result = runner.run(step_id=case.primary_step_id, code=code)

    assert spy.calls == 0
    assert result.succeeded, result.stderr
    assert required_files.issubset({path.name for path in result.artefacts})
    summary = json.loads(
        (result.out_dir / "step_summary.json").read_text(encoding="utf-8")
    )
    assert summary.get("status") == "ok" or summary.get("method")
    if case is M2:
        split_frame = pd.read_csv(result.out_dir / "split_definition.csv")
        assert int(split_frame.loc[0, "patient_overlap_n"]) == 0
        metrics = pd.read_csv(result.out_dir / "model_performance_train_test.csv")
        assert {
            "auroc",
            "average_precision",
            "recall",
            "f1",
            "brier",
            "calibration_slope",
        }.issubset(metrics.columns)


def test_production_evaluator_rejects_each_complex_mock_run(
    case: PreflightCase,
    tmp_path: Path,
) -> None:
    _code, controller = _reviewed_primary_code(case)
    run = PreflightRun(
        case=case,
        run_dir=tmp_path,
        run_id="diagnostic-only",
        manifest={},
        llm=controller,
    )
    verdict = paper_acceptance_verdict(run)

    assert verdict.status == "invalid"
    assert verdict.issues


def _run_complex_case(case: PreflightCase, tmp_path_factory) -> PreflightRun:
    manifest = preflight_runtime_manifest()
    if not manifest.integration_ready:
        pytest.skip(manifest.blocked_reason or "integration_not_ready")
    return run_preflight(
        case,
        workdir=tmp_path_factory.mktemp(f"complex_{case.task_id}"),
        n_rows=160,
    )


@pytest.fixture(
    scope="module",
    params=FINALISING_CASES,
    ids=[case.task_id for case in FINALISING_CASES],
)
def complex_graph_run(request, tmp_path_factory) -> PreflightRun:
    return _run_complex_case(request.param, tmp_path_factory)


@pytest.fixture(
    scope="module",
    params=AUTHORITY_BLOCKED_CASES,
    ids=[case.task_id for case in AUTHORITY_BLOCKED_CASES],
)
def authority_blocked_run(request, tmp_path_factory) -> PreflightRun:
    return _run_complex_case(request.param, tmp_path_factory)


def test_complex_real_graph_is_zero_provider_and_finalises(
    complex_graph_run: PreflightRun,
) -> None:
    run = complex_graph_run
    assert run.pipeline_ran is True
    assert run.external_provider_calls == 0
    assert run.raised is None
    assert run.record(run.case.deterministic_step_id)["status"] == "ok"
    assert run.record(run.case.primary_step_id)
    assert run.manifest.get("readiness", {}).get("paper_authorized") is not True
    assert paper_acceptance_verdict(run).status == "invalid"


def test_authority_blocked_case_is_refused_by_name_and_produces_nothing(
    authority_blocked_run: PreflightRun,
) -> None:
    """A case with no reviewed scientific authority must be refused, not run.

    H1 and H2 land here, and their preflight *must* fail -- but a bare red test
    cannot tell "the host correctly refused to invent a scientific contract"
    apart from "something regressed". This asserts the refusal itself, so a
    silent change in either direction breaks CI:

    * H1 now has an AI-reviewed development protocol and deterministic suite,
      but the formal zero-provider preflight remains blocked until human
      scientific attestation is bound.
    * H2 has a protocol, and it records ``causal_contrast_authorized=False`` /
      ``H2_VERIFIED_NON_USE_UNAVAILABLE`` because verified non-use is
      unavailable, listing ``construct_binary_control_arm`` as forbidden.

    The refusal is only trustworthy if nothing was fabricated on the way to it,
    so this also pins zero provider spend and an empty execution record: no
    step ran, no product exists, and nothing is publishable.
    """

    run = authority_blocked_run
    block = _sealed_authority_block(run.case)
    assert block is not None
    blocker_code, _detail = block

    # The pipeline really attempted the case; it was refused, not skipped.
    assert run.pipeline_ran is True
    assert run.raised is not None

    # Refused for the declared reason, naming the contract it could not source.
    assert "family_primary_result_requirement" in run.raised
    assert run.case.analysis_type in run.raised

    # The blocker is the sealed authority's own code, not a test-local string.
    if blocker_code == "AI_DEVELOPMENT_REVIEW_ONLY":
        protocol = load_default_case_protocol(run.case.task_id)
        assert protocol.review_status.startswith("ai_development_")
    elif blocker_code != "SCIENTIFIC_CASE_PROTOCOL_UNKNOWN_TASK":
        protocol = load_default_case_protocol(run.case.task_id)
        assert protocol.current_source_capture.reason_code == blocker_code
        assert protocol.current_source_capture.causal_contrast_authorized is False
    else:
        with pytest.raises(ScientificCaseProtocolError, match=blocker_code):
            default_case_protocol_path(run.case.task_id)

    # Nothing was spent and nothing was fabricated.
    assert run.external_provider_calls == 0
    assert run.step_ids == []
    assert run.record(run.case.primary_step_id) == {}
    assert run.record(run.case.deterministic_step_id) == {}
    assert run.manifest.get("readiness", {}).get("paper_authorized") is not True
    assert paper_acceptance_verdict(run).status == "invalid"
