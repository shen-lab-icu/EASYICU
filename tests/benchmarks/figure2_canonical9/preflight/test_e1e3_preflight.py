"""Increment 1 — distinct E1/E2/E3 semantics, real zero-Provider transport spy,
and an explicit isolation-backend diagnostic.

Two layers of assertion:

* **Env-independent** (always run): the plans are genuinely distinct and each
  suite ``semantic_guardrail`` maps to a structural feature of the plan/cohort;
  the transport spy is a real fail-on-call boundary; the isolation classifier
  distinguishes a nested-sandbox denial from an ordinary exit-71.
* **Real-subprocess integration** (gated on ``integration_ready``): the true
  ``ResearchAgentPipeline`` graph runs offline and must show the deterministic
  Table 1 / Coder division of labour, a ``diagnostic_only`` verdict, zero
  transport calls, and rejection by the production paper-acceptance gate.  Where
  ``sandbox-exec`` is unavailable (a nested sandbox) these are **skipped with a
  structured ``isolation_backend_unavailable`` reason** — never counted as a
  pass and never a silent skip.  The formal gate must genuinely pass in a
  sandbox-permitting environment.

Increment 2 covers the control-flow caps (repair/replan/no-op), real timeout +
watchdog, digest-based stop/resume, and the explicit paper-authority reason.

Run just this batch::

    PYTHONPATH="src:." pytest tests/benchmarks/figure2_canonical9/preflight/ -p no:randomly
"""

from __future__ import annotations

import copy
import json
import os
from dataclasses import replace
from types import SimpleNamespace

import pytest

from benchmarks.figure2_canonical9.preflight import runtime as rt
from benchmarks.figure2_canonical9.preflight.fixtures import (
    E1,
    E1E3_CASES,
    FULFILLMENT_NOT_PRODUCED_OFFLINE,
    FULFILLMENT_PLANNED_ONLY,
    FULFILLMENT_PRODUCED,
    PreflightCase,
    ProductMapping,
)
from benchmarks.figure2_canonical9.preflight import harness
from benchmarks.figure2_canonical9.preflight.harness import (
    ProviderTransportBlocked,
    ScriptedPreflightLLM,
    paper_acceptance_verdict,
    paper_acceptance_status,
    PREFLIGHT_RUNNER_KWARGS,
    preflight_runtime_manifest,
    provider_transport_spy,
    run_preflight,
)

CASES = list(E1E3_CASES.values())

# The distinguishing method signature each task's plan must genuinely contain
# (proves E1/E2/E3 are not one shared skeleton).
_DISTINCT_METHODS = {
    "e1_sepsis3_prevalence_mortality": {"cohort_definition_summary"},
    "e2_lactate_mortality": {
        "within_window_peak_aggregation",
        "missingness_measurement_audit",
    },
    "e3_kdigo_gradient": {"stage_stratified_outcomes", "ordinal_trend_test"},
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def runtime_manifest() -> rt.RuntimeManifest:
    return preflight_runtime_manifest()


@pytest.fixture(params=CASES, ids=[c.task_id for c in CASES])
def case(request) -> PreflightCase:
    return request.param


@pytest.fixture(scope="module", params=CASES, ids=[c.task_id for c in CASES])
def normal_run(request, tmp_path_factory):
    """A real offline subprocess run — gated on an available isolation backend.

    When the isolation backend is unavailable (a nested sandbox) this is a
    structured ``isolation_backend_unavailable`` skip, not a pass and not an
    unexplained skip.  The formal gate genuinely runs where sandbox-exec works.
    """

    manifest = preflight_runtime_manifest()
    if not manifest.integration_ready:
        # Structured, machine-parseable reason (starts with
        # "isolation_backend_unavailable:" / "runtime_incompatible:") — never a
        # pass and never a bare "skipped: env" line.
        pytest.skip(manifest.blocked_reason or "integration_not_ready")
    the_case = request.param
    workdir = tmp_path_factory.mktemp(f"pf_{the_case.task_id}")
    return run_preflight(the_case, workdir=workdir)


def _control_run(case: PreflightCase, tmp_path, **kwargs):
    """Run a control-flow test only with a usable real isolation backend."""

    manifest = preflight_runtime_manifest()
    if not manifest.integration_ready:
        pytest.skip(manifest.blocked_reason or "integration_not_ready")
    return run_preflight(case, workdir=tmp_path, **kwargs)


# ---------------------------------------------------------------------------
# Layer 1a — distinct semantics (env-independent)
# ---------------------------------------------------------------------------


def test_plan_contract_valid(case: PreflightCase) -> None:
    plan = case.build_plan()
    assert plan.analysis_type == "association_study"
    assert len(plan.steps) >= 3

    table_ones = [
        s
        for s in plan.steps
        if s.table_one_spec is not None and s.expected_outputs == ["table:table_one"]
    ]
    assert len(table_ones) == 1, "exactly one deterministic Table 1 step"
    t1 = table_ones[0]
    spec = t1.table_one_spec
    assert spec is not None
    for name in [spec.group_by, *(v.name for v in spec.variables)]:
        assert name in t1.inputs, f"{name} must be an explicit Table 1 input"
    assert t1.step_id == case.deterministic_step_id

    primary = [s for s in plan.steps if s.step_id == case.primary_step_id]
    assert len(primary) == 1
    assert primary[0].planned_analysis_role == "primary"
    assert primary[0].scientific_capability == "association_freeform_v1"
    assert primary[0].expected_outputs == [
        "table:association_model_diagnostics"
    ]
    assert primary[0].model_requirements == []


def test_expected_products_pairwise_distinct() -> None:
    products = {c.task_id: set(c.expected_products) for c in CASES}
    keys = list(products)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            assert (
                products[keys[i]] != products[keys[j]]
            ), f"{keys[i]} and {keys[j]} must have distinct expected products"


def test_every_live_expected_product_has_an_honest_scope_mapping(
    case: PreflightCase,
) -> None:
    """A green offline smoke cannot silently become a formal-output claim."""

    mapped = case.product_mapping()
    assert [product for product, _mapping in mapped] == list(case.expected_products)
    plan_step_ids = {step.step_id for step in case.build_plan().steps}
    for _product, mapping in mapped:
        if mapping.declared_fulfillment == FULFILLMENT_NOT_PRODUCED_OFFLINE:
            assert mapping.step_id is None
            assert mapping.artifact_evidence_prefix is None
        else:
            assert mapping.step_id in plan_step_ids
    produced = [
        (product, mapping)
        for product, mapping in mapped
        if mapping.declared_fulfillment == FULFILLMENT_PRODUCED
    ]
    assert len(produced) == 1
    assert produced[0][0] == "table one"
    assert produced[0][1].artifact_evidence_prefix == "table_step_artifact_"
    assert any(
        mapping.declared_fulfillment == FULFILLMENT_PLANNED_ONLY
        for _product, mapping in mapped
    )


def test_product_map_fails_closed_when_a_live_output_is_unmapped() -> None:
    incomplete = replace(E1, product_map=E1.product_map[:-1])
    with pytest.raises(AssertionError, match="cover each live expected output"):
        incomplete.product_mapping()


def test_product_map_rejects_an_unsourced_produced_claim() -> None:
    with pytest.raises(ValueError, match="producing step and evidence prefix"):
        ProductMapping(0, E1.deterministic_step_id, FULFILLMENT_PRODUCED)


def test_plan_reflects_distinct_task_shape(case: PreflightCase) -> None:
    methods = set(case.plan_methods())
    required = _DISTINCT_METHODS[case.task_id]
    assert required.issubset(methods), (
        f"{case.task_id} plan must carry its distinguishing steps {required}, "
        f"got methods {sorted(methods)}"
    )


def test_guardrail_structural_coverage(case: PreflightCase) -> None:
    guardrails = case.semantic_guardrails
    checks = case.guardrail_checks
    assert guardrails, "case must bind at least one suite guardrail"
    # Full, one-to-one coverage of the live suite guardrails.
    assert sorted(c.guardrail_index for c in checks) == list(range(len(guardrails)))
    for chk in checks:
        assert chk.holds(case), (
            f"{case.task_id} guardrail[{chk.guardrail_index}] "
            f"({chk.key!r}) not structurally honoured: "
            f"{guardrails[chk.guardrail_index]!r}"
        )


def test_llm_offline_classified_is_descriptive_only(case: PreflightCase) -> None:
    # llm_is_mockish is recorded as descriptive colour; it is NOT the security
    # boundary (that is the transport spy).  Here we only assert it classifies
    # the reviewed scripted client as offline.
    llm = ScriptedPreflightLLM(case)
    from easyicu.research_agent.providers.llm import llm_is_mockish

    assert llm_is_mockish(llm.client) is True


def test_preflight_uses_a_reviewed_offline_client_not_a_custom_subclass(
    case: PreflightCase,
) -> None:
    """The controller never crosses the prompt boundary as a custom client."""

    from easyicu.research_agent.providers.llm import llm_is_mockish

    llm = ScriptedPreflightLLM(case)
    assert type(llm.client).__name__ == "PatternScriptedMockLLMClient"
    assert llm_is_mockish(llm.client) is True


def test_preflight_runner_forbids_network_and_host_fallback(tmp_path) -> None:
    llm = ScriptedPreflightLLM(E1)
    pipeline = harness.build_pipeline(E1, workdir=tmp_path, llm=llm.client)
    assert pipeline._runner_kind == "subprocess"
    assert pipeline._runner_network == "none"
    assert pipeline._runner_kwargs == PREFLIGHT_RUNNER_KWARGS


def test_environment_cannot_relax_pinned_host_fallback(tmp_path, monkeypatch) -> None:
    import pandas as pd

    from easyicu.research_agent import CodeRunner

    cohort = tmp_path / "cohort.parquet"
    pd.DataFrame({"stay_id": [1, 2], "death": [0, 1]}).to_parquet(cohort)
    monkeypatch.setenv("EASYICU_ALLOW_UNSAFE_HOST_FALLBACK", "1")

    pinned = CodeRunner(
        workdir=tmp_path / "pinned",
        cohort_parquet=cohort,
        **PREFLIGHT_RUNNER_KWARGS,
    )
    assert pinned.network_policy == "none"
    assert pinned.allow_unsafe_host_fallback is False

    # Control: the environment only wins when the caller leaves the argument
    # unpinned.  This proves the explicit preflight pin is load-bearing.
    unpinned = CodeRunner(
        workdir=tmp_path / "unpinned",
        cohort_parquet=cohort,
        allow_unsafe_host_fallback=None,
    )
    assert unpinned.allow_unsafe_host_fallback is True


def test_unavailable_runtime_blocks_before_pipeline_launch(
    monkeypatch, tmp_path
) -> None:
    unavailable = rt.RuntimeManifest(
        parent={"role": "parent"},
        subprocess={"role": "subprocess"},
        isolation=rt.IsolationCapability(
            backend="macos_sandbox_exec",
            available=False,
            returncode=71,
            detail="sandbox-exec: sandbox_apply: Operation not permitted",
        ),
    )
    launches = 0

    def _runtime(run_dir=None):
        if run_dir is not None:
            rt.write_runtime_manifest(run_dir, unavailable)
        return unavailable

    def _must_not_launch(*_args, **_kwargs):
        nonlocal launches
        launches += 1
        raise AssertionError("pipeline must not launch without usable isolation")

    monkeypatch.setattr(harness, "preflight_runtime_manifest", _runtime)
    monkeypatch.setattr(harness, "build_pipeline", _must_not_launch)
    run = harness.run_preflight(E1, workdir=tmp_path)

    assert launches == 0
    assert run.pipeline_ran is False
    assert run.blocked_reason == unavailable.blocked_reason
    assert run.runtime["integration_ready"] is False
    assert run.llm.total_calls == 0
    assert (tmp_path / "preflight_runtime_manifest.json").is_file()


def test_cli_returns_structured_blocked_exit_code(monkeypatch, capsys) -> None:
    unavailable = rt.RuntimeManifest(
        parent={"role": "parent"},
        subprocess={"role": "subprocess"},
        isolation=rt.IsolationCapability(
            backend="macos_sandbox_exec",
            available=False,
            returncode=71,
            detail="sandbox-exec: sandbox_apply: Operation not permitted",
        ),
    )

    def _runtime(run_dir=None):
        if run_dir is not None:
            rt.write_runtime_manifest(run_dir, unavailable)
        return unavailable

    def _must_not_launch(*_args, **_kwargs):
        raise AssertionError("CLI must not build the pipeline when blocked")

    monkeypatch.setattr(harness, "preflight_runtime_manifest", _runtime)
    monkeypatch.setattr(harness, "build_pipeline", _must_not_launch)
    assert harness._cli("e1") == 2
    output = capsys.readouterr().out
    assert '"blocked": true' in output
    assert "isolation_backend_unavailable" in output


# ---------------------------------------------------------------------------
# Layer 1b — the transport spy is a REAL fail-on-call boundary
# ---------------------------------------------------------------------------


def test_transport_spy_blocks_and_counts_a_real_send() -> None:
    import httpx

    with provider_transport_spy() as spy:
        assert spy.calls == 0
        client = httpx.Client()
        request = httpx.Request("GET", "http://localhost/should-never-fire")
        with pytest.raises(ProviderTransportBlocked):
            client.send(request)
    # The spy counted the attempt (it is not a hardcoded zero).
    assert spy.calls == 1
    assert spy.targets == ["httpx.Client.send"]


def test_transport_spy_restores_httpx_afterwards() -> None:
    import httpx

    original = httpx.Client.send
    with provider_transport_spy():
        assert httpx.Client.send is not original
    assert httpx.Client.send is original


# ---------------------------------------------------------------------------
# Layer 1c — isolation-backend diagnostic (env-independent)
# ---------------------------------------------------------------------------


def _macos_capability(available: bool) -> rt.IsolationCapability:
    return rt.IsolationCapability(
        backend="macos_sandbox_exec",
        available=available,
        returncode=None if available else 71,
        detail=(
            "" if available else "sandbox-exec: sandbox_apply: Operation not permitted"
        ),
    )


def test_runtime_manifest_shape(runtime_manifest: rt.RuntimeManifest) -> None:
    assert runtime_manifest.parent.get("role") == "parent"
    assert runtime_manifest.subprocess.get("role") == "subprocess"
    assert runtime_manifest.isolation.backend
    # The fail-closed verdict is exactly compatible AND isolation-available.
    assert runtime_manifest.integration_ready == (
        runtime_manifest.compatible and runtime_manifest.isolation.available
    )


def test_blocked_reason_is_structured(runtime_manifest: rt.RuntimeManifest) -> None:
    if runtime_manifest.integration_ready:
        assert runtime_manifest.blocked_reason is None
    else:
        assert runtime_manifest.blocked_reason is not None
        assert runtime_manifest.blocked_reason.startswith(
            ("isolation_backend_unavailable:", "runtime_incompatible:")
        )


def test_linux_probe_matches_code_runner_filesystem_fail_closed(
    monkeypatch,
) -> None:
    """A network namespace alone must never authorize generated-code execution."""

    monkeypatch.setattr(rt.sys, "platform", "linux")
    capability = rt.probe_isolation_backend()

    assert capability.backend in {
        "linux_unshare_network_namespace",
        "host_subprocess",
    }
    assert capability.available is False
    assert "filesystem isolation" in capability.detail


def test_macos_probe_exercises_exact_code_runner_boundary(monkeypatch) -> None:
    """The capability probe must not substitute ``/usr/bin/true`` for Python."""

    observed: dict[str, object] = {}

    class _ProbeRunner:
        def __init__(self, **kwargs) -> None:
            observed["init"] = kwargs

        def run(self, **kwargs):
            observed["run"] = kwargs
            return SimpleNamespace(
                succeeded=False,
                effective_isolation="macos_sandbox_exec",
                isolation_degraded=False,
                returncode=71,
                stderr=(
                    "sandbox-exec: execvp() of '/tmp/.venv/bin/python' failed: "
                    "Operation not permitted"
                ),
            )

    monkeypatch.setattr(rt.sys, "platform", "darwin")
    monkeypatch.setattr(rt.shutil, "which", lambda name: "/usr/bin/sandbox-exec")
    monkeypatch.setattr(rt, "CodeRunner", _ProbeRunner)

    capability = rt.probe_isolation_backend()

    assert capability.available is False
    assert capability.returncode == 71
    assert "execvp()" in capability.detail
    init = observed["init"]
    assert isinstance(init, dict)
    assert init["python_executable"] == rt.sys.executable
    assert init["network_policy"] == "none"
    assert init["allow_unsafe_host_fallback"] is False
    assert observed["run"] == {
        "step_id": "isolation_capability_probe",
        "code": "pass\n",
    }


def test_step_isolation_positive_nested_sandbox_denial() -> None:
    record = {
        "returncode": 71,
        "timed_out": False,
        "stderr": "sandbox-exec: sandbox_apply: Operation not permitted",
    }
    reason = rt.step_isolation_unavailable(record, _macos_capability(available=False))
    assert reason is not None
    assert "sandbox_apply" in reason


def test_step_isolation_positive_execvp_denial() -> None:
    record = {
        "returncode": 71,
        "timed_out": False,
        "stderr": (
            "sandbox-exec: execvp() of '/tmp/.venv/bin/python' failed: "
            "Operation not permitted"
        ),
    }
    reason = rt.step_isolation_unavailable(record, _macos_capability(available=False))
    assert reason is not None
    assert "execvp()" in reason


def test_step_isolation_negative_execvp_without_permission_denial() -> None:
    record = {
        "returncode": 71,
        "timed_out": False,
        "stderr": "sandbox-exec: execvp() of '/tmp/missing-python' failed: No such file",
    }
    assert (
        rt.step_isolation_unavailable(record, _macos_capability(available=False))
        is None
    )


def test_step_isolation_negative_plain_exit_71_is_execution_failure() -> None:
    # A generated script that legitimately exits 71 with NO sandbox stderr must
    # be judged an execution failure, never an isolation outage.
    record = {
        "returncode": 71,
        "timed_out": False,
        "stderr": "Traceback (most recent call last):\n  ...\nSystemExit: 71",
    }
    assert (
        rt.step_isolation_unavailable(record, _macos_capability(available=False))
        is None
    )


def test_step_isolation_negative_when_backend_available() -> None:
    # Even with a sandbox-shaped stderr, an *available* backend cannot be the
    # cause — the probe would have let the step run.
    record = {
        "returncode": 71,
        "timed_out": False,
        "stderr": "sandbox-exec: sandbox_apply: Operation not permitted",
    }
    assert (
        rt.step_isolation_unavailable(record, _macos_capability(available=True)) is None
    )


def test_step_isolation_negative_wrong_backend() -> None:
    # Only macos_sandbox_exec can be denied by a nested sandbox.
    cap = rt.IsolationCapability(
        backend="host_subprocess_linux", available=False, returncode=71
    )
    record = {
        "returncode": 71,
        "timed_out": False,
        "stderr": "sandbox-exec: sandbox_apply: Operation not permitted",
    }
    assert rt.step_isolation_unavailable(record, cap) is None


def test_step_isolation_negative_timeout_is_not_isolation() -> None:
    record = {
        "returncode": None,
        "timed_out": True,
        "stderr": "sandbox-exec: sandbox_apply: Operation not permitted",
    }
    assert (
        rt.step_isolation_unavailable(record, _macos_capability(available=False))
        is None
    )


# ---------------------------------------------------------------------------
# Layer 2 — real-subprocess integration (gated on integration_ready)
# ---------------------------------------------------------------------------


def _record_by_analysis(run, analysis: str):
    for r in run.manifest.get("per_step_records", []):
        if r.get("deterministic_standard_analysis") == analysis:
            return r
    return {}


def test_routing_deterministic_table_one_vs_coder_primary(normal_run) -> None:
    the_case = normal_run.case
    table_one = _record_by_analysis(normal_run, "grouped_table_one")
    assert table_one, "deterministic grouped Table 1 record must exist"
    assert table_one.get("generation_mode") == "deterministic_standard"
    assert table_one.get("status") == "ok"

    primary = normal_run.record(the_case.primary_step_id)
    assert primary, f"primary step {the_case.primary_step_id} must exist"
    assert primary.get("generation_mode") == "llm", "primary is agent-owned (Coder)"
    assert primary.get("status") == "ok"


def test_distinct_aux_steps_are_actually_executed(normal_run) -> None:
    # The task-specific auxiliary steps really run (contract_failed offline is
    # fine — the point is the distinct graph is executed, not shared).
    executed = set(normal_run.step_ids)
    the_case = normal_run.case
    aux_ids = {
        s.step_id
        for s in the_case.build_plan().steps
        if s.step_id not in (the_case.deterministic_step_id, the_case.primary_step_id)
    }
    assert aux_ids, "each case must plan >=1 distinct auxiliary step"
    assert aux_ids.issubset(executed), (
        f"{the_case.task_id} distinct aux steps {aux_ids} must be executed; "
        f"ran {sorted(executed)}"
    )


def test_zero_provider_transport_measured(normal_run) -> None:
    # Authoritative zero-Provider evidence: the real httpx transport spy, not a
    # client self-report.
    assert normal_run.external_provider_calls == 0
    assert normal_run.provider_transport_targets == []


def test_final_verdict_is_diagnostic_only(normal_run) -> None:
    assert normal_run.tristate == "diagnostic_only"
    assert normal_run.case.expected_tristate == "diagnostic_only"


def test_runtime_receipt_is_persisted_for_the_actual_run(normal_run) -> None:
    runtime_path = normal_run.run_dir / "preflight_runtime_manifest.json"
    assert runtime_path.is_file()
    assert normal_run.runtime["integration_ready"] is True
    assert normal_run.blocked_reason is None


def test_product_mapping_describes_actual_preflight_scope(normal_run) -> None:
    """Planned nodes run; only declared produced output is asserted as output."""

    for _product, mapping in normal_run.case.product_mapping():
        if mapping.declared_fulfillment == FULFILLMENT_NOT_PRODUCED_OFFLINE:
            assert mapping.step_id is None
            continue
        record = normal_run.record(mapping.step_id or "")
        assert record, f"mapped step {mapping.step_id!r} did not execute"
        if mapping.declared_fulfillment == FULFILLMENT_PRODUCED:
            assert record.get("status") == "ok"
            assert record.get("generation_mode") == "deterministic_standard"
    assert normal_run.tristate == "diagnostic_only"


def test_product_map_resolves_against_real_manifest_and_is_persisted(
    normal_run,
) -> None:
    resolved = normal_run.resolved_product_map()
    assert len(resolved) == len(normal_run.case.expected_products)
    assert all(row["matches"] is True for row in resolved)
    persisted = json.loads(
        (normal_run.run_dir / "preflight_product_map.json").read_text(encoding="utf-8")
    )
    assert persisted["readiness_class"] == "partial_flow_smoke"
    assert persisted["product_map"] == resolved


def test_only_table_one_is_produced_offline(normal_run) -> None:
    assert normal_run.readiness_class == "partial_flow_smoke"
    assert normal_run.produced_suite_outputs() == ["table one"]


def test_dropped_table_one_artifact_downgrades_product_claim(normal_run) -> None:
    produced_mapping = next(
        mapping
        for _product, mapping in normal_run.case.product_mapping()
        if mapping.declared_fulfillment == FULFILLMENT_PRODUCED
    )
    tampered = copy.deepcopy(normal_run)
    record = tampered.record(produced_mapping.step_id or "")
    record["evidence_ids"] = [
        evidence_id
        for evidence_id in record.get("evidence_ids", [])
        if not evidence_id.startswith(produced_mapping.artifact_evidence_prefix or "")
    ]
    assert harness.observe_product_fulfillment(produced_mapping, tampered) == (
        FULFILLMENT_PLANNED_ONLY
    )


def test_network_policy_report_is_no_network_and_non_degraded(normal_run) -> None:
    report = normal_run.network_policy_report()
    assert report["ok"] is True
    assert report["subprocess_steps"]
    for step in report["subprocess_steps"]:
        assert step["requested_network_policy"] == "none"
        assert step["isolation_degraded"] is False


def test_loop_terminates_bounded(normal_run) -> None:
    # No unbounded orchestration: the graph completes in a small bounded number
    # of steps.
    assert 0 < len(normal_run.step_ids) <= 16


def test_paper_acceptance_rejects_mock_run(normal_run) -> None:
    assert paper_acceptance_status(normal_run) == "invalid"
    verdict = paper_acceptance_verdict(normal_run)
    issue_codes = {issue.code for issue in verdict.issues}
    # A single task is rejected for exact Canonical9 coverage, not merely
    # labelled invalid by a mock-only convention.
    assert "TASK_COVERAGE_INVALID" in issue_codes
    assert "EXPECTED_EXECUTION_IDENTITY_MISSING" in issue_codes


def test_single_planner_call(normal_run) -> None:
    # The scripted plan is accepted on the first planner call (no retry storm).
    assert normal_run.llm.plan_calls == 1


# ---------------------------------------------------------------------------
# Layer 2b — real control-flow boundaries (gated on integration_ready)
# ---------------------------------------------------------------------------


def test_runtime_repair_cap_is_consumed_exactly(tmp_path) -> None:
    run = _control_run(
        E1,
        tmp_path,
        fault_step=E1.primary_step_id,
        max_code_repair_attempts=2,
    )

    record = run.record(E1.primary_step_id)
    assert record.get("status") == "execution_failed"
    assert record.get("code_repair_attempts") == 2
    assert record.get("runtime_repair_attempts") == 2
    # One initial code generation plus two repair proposals and two execution
    # repair follow-ups: the cap must bound actual pipeline work, not a counter.
    assert run.llm.code_calls[E1.primary_step_id] == 5
    assert run.external_provider_calls == 0


@pytest.mark.parametrize(
    "fault_code",
    ["import time\ntime.sleep(60)\n", "while True:\n    pass\n"],
    ids=["sleep_timeout", "busy_loop_watchdog"],
)
def test_real_subprocess_timeout_and_watchdog_fail_closed(tmp_path, fault_code) -> None:
    run = _control_run(
        E1,
        tmp_path,
        fault_step=E1.primary_step_id,
        fault_code=fault_code,
        timeout_seconds=0.5,
        max_code_repair_attempts=0,
    )

    record = run.record(E1.primary_step_id)
    assert record.get("status") == "execution_failed"
    assert record.get("timed_out") is True
    assert record.get("returncode") == -1
    assert record.get("code_repair_attempts") == 0
    assert run.external_provider_calls == 0


def test_noop_replan_cap_stops_real_replanner_loop(tmp_path) -> None:
    run = _control_run(
        E1,
        tmp_path,
        enable_replanning=True,
        request_replan_from_primary=True,
        replan_strategy="noop",
        # The current initial-plan shaper adds host-owned figure/audit nodes
        # before the probe.  The static fixture response therefore consumes
        # the probe replan as one substantive normalization; the subsequent
        # primary-requested identical candidate is the no-op whose cap this
        # control test exercises.
        max_consecutive_noop_replans=1,
    )

    finding = next(
        finding
        for finding in run.manifest.get("findings", [])
        if "consecutive no-op revisions" in finding.get("message", "")
    )
    assert finding["detail"]["reason"] == E1.primary_step_id
    assert run.readiness.get("replan_budget_exhausted") is not True
    assert run.external_provider_calls == 0


def test_substantive_replan_cap_stops_real_replanner_loop(tmp_path) -> None:
    run = _control_run(
        E1,
        tmp_path,
        enable_replanning=True,
        request_replan_from_primary=True,
        replan_strategy="substantive",
        max_replans=2,
    )

    finding = next(
        finding
        for finding in run.manifest.get("findings", [])
        if finding.get("validator") == "replan_budget"
    )
    assert finding["detail"] == {
        "replan_budget_exhausted": True,
        "cap": 2,
        "substantive_revisions": 2,
        "reason": E1.primary_step_id,
    }
    assert run.readiness["replan_budget_exhausted"] is True
    assert run.tristate == "diagnostic_only"
    assert run.external_provider_calls == 0


def _evidence_by_id(manifest, evidence_id: str):
    return next(
        record
        for record in manifest.get("evidence", [])
        if record.get("evidence_id") == evidence_id
    )


def test_stop_resume_uses_digests_not_mtime(tmp_path) -> None:
    first = _control_run(
        E1,
        tmp_path,
        stop_after_step_id=E1.deterministic_step_id,
    )
    table_one = first.record(E1.deterministic_step_id)
    evidence_ids = list(table_one.get("evidence_ids", []))
    assert evidence_ids
    original_hashes = {
        evidence_id: _evidence_by_id(first.manifest, evidence_id).get("sha256")
        for evidence_id in evidence_ids
    }

    # Make all persisted timestamps identical.  Resume must still recognize the
    # completed immutable artifacts by content identity, not mtime heuristics.
    fixed_time = 1_700_000_000
    for path in tmp_path.rglob("*"):
        if path.is_file():
            os.utime(path, (fixed_time, fixed_time))

    resumed = _control_run(E1, tmp_path, resume_run_id=first.run_id)
    resumed_table_one = resumed.record(E1.deterministic_step_id)
    assert resumed.run_id == first.run_id
    assert resumed_table_one.get("evidence_ids") == evidence_ids
    assert {
        evidence_id: _evidence_by_id(resumed.manifest, evidence_id).get("sha256")
        for evidence_id in evidence_ids
    } == original_hashes
    assert (
        sum(
            record.get("step_id") == E1.deterministic_step_id
            for record in resumed.manifest.get("per_step_records", [])
        )
        == 1
    )
    assert resumed.llm.plan_calls == 0
    assert resumed.external_provider_calls == 0


def test_tampered_completed_artifact_fails_resume_before_llm_delivery(tmp_path) -> None:
    first = _control_run(
        E1,
        tmp_path,
        stop_after_step_id=E1.deterministic_step_id,
    )
    table_one = first.record(E1.deterministic_step_id)
    artifact_id = next(
        evidence_id
        for evidence_id in table_one.get("evidence_ids", [])
        if evidence_id.startswith("table_step_artifact_")
    )
    artifact = _evidence_by_id(first.manifest, artifact_id)
    (first.run_dir / artifact["relative_path"]).write_text("tampered", encoding="utf-8")

    resumed = _control_run(
        E1,
        tmp_path,
        resume_run_id=first.run_id,
        resume_from_step_id=E1.deterministic_step_id,
    )
    resumed_table_one = next(
        record
        for record in resumed.manifest.get("per_step_records", [])
        if record.get("step_id") == E1.deterministic_step_id
        and record.get("status") == "execution_raised"
    )
    assert str(resumed_table_one.get("error") or "").startswith(
        "EvidenceAuthorityIntegrityError:"
    )
    assert resumed.llm.total_calls == 0
    assert resumed.external_provider_calls == 0


# ---------------------------------------------------------------------------
# Sanity: the case registry is the three E-series tasks
# ---------------------------------------------------------------------------


def test_case_registry_is_e_series() -> None:
    assert set(E1E3_CASES) == {
        "e1_sepsis3_prevalence_mortality",
        "e2_lactate_mortality",
        "e3_kdigo_gradient",
    }
    assert E1.task_id == "e1_sepsis3_prevalence_mortality"
