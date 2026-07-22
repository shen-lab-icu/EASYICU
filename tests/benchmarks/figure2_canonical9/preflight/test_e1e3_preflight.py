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

Increment 1.1 (P1 closure) adds: (P1-A) a per-item formal-output contract map
verified against the real manifest + honest downgrade to a *partial-flow smoke*
(offline only "table one" is produced); (P1-B) the isolation structured block
wired into ``run_preflight``/CLI (no pipeline start when unavailable; per-step
denial converted, not left as ``repair_failed``); (P1-C) a pinned, env-var-proof
no-network runner policy.

Increment 2 adds the control-flow caps (repair/replan/no-op), real timeout +
watchdog, digest-based stop/resume, and the explicit paper-authority reason.

Run just this batch::

    PYTHONPATH="src:." pytest tests/benchmarks/figure2_canonical9/preflight/ -p no:randomly
"""

from __future__ import annotations

from pathlib import Path

import pytest

from benchmarks.figure2_canonical9.preflight import harness as hn
from benchmarks.figure2_canonical9.preflight import runtime as rt
from benchmarks.figure2_canonical9.preflight.fixtures import (
    FULFILLMENT_NOT_PRODUCED_OFFLINE,
    FULFILLMENT_PLANNED_ONLY,
    FULFILLMENT_PRODUCED,
    E1,
    E1E3_CASES,
    PreflightCase,
)
from benchmarks.figure2_canonical9.preflight.harness import (
    PREFLIGHT_RUNNER_KWARGS,
    READINESS_CLASS,
    PreflightRun,
    ProviderTransportBlocked,
    ScriptedPreflightLLM,
    blocking_step_isolation,
    build_pipeline,
    observe_product_fulfillment,
    paper_acceptance_status,
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


def test_expected_products_pairwise_distinct() -> None:
    products = {c.task_id: set(c.expected_products) for c in CASES}
    keys = list(products)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            assert (
                products[keys[i]] != products[keys[j]]
            ), f"{keys[i]} and {keys[j]} must have distinct expected products"


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
    # the scripted client as offline, which the docstring calls out as forgeable.
    llm = ScriptedPreflightLLM(case)
    from easyicu.research_agent.providers.llm import llm_is_mockish

    assert llm_is_mockish(llm) is True


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


def test_step_isolation_positive_nested_sandbox_denial() -> None:
    record = {
        "returncode": 71,
        "timed_out": False,
        "stderr": "sandbox-exec: sandbox_apply: Operation not permitted",
    }
    reason = rt.step_isolation_unavailable(record, _macos_capability(available=False))
    assert reason is not None
    assert "sandbox_apply" in reason


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


def test_loop_terminates_bounded(normal_run) -> None:
    # No unbounded orchestration: the graph completes in a small bounded number
    # of steps.
    assert 0 < len(normal_run.step_ids) <= 16


def test_paper_acceptance_rejects_mock_run(normal_run) -> None:
    assert paper_acceptance_status(normal_run) == "invalid"


def test_single_planner_call(normal_run) -> None:
    # The scripted plan is accepted on the first planner call (no retry storm).
    assert normal_run.llm.plan_calls == 1


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


# ---------------------------------------------------------------------------
# P1-A — formal task output contract (item-by-item, verified, honest downgrade)
# ---------------------------------------------------------------------------


def _fake_run(case: PreflightCase, records) -> PreflightRun:
    return PreflightRun(
        case=case,
        run_dir=Path("/nonexistent-preflight"),
        run_id="synthetic",
        manifest={"per_step_records": records},
        llm=ScriptedPreflightLLM(case),
        pipeline_ran=True,
    )


def test_product_map_covers_suite_outputs(case: PreflightCase) -> None:
    n = len(case.expected_products)
    assert n >= 1
    # Exactly one ProductMapping per live suite expected_output (one-to-one).
    assert sorted(m.output_index for m in case.product_map) == list(range(n))


def test_readiness_class_is_partial_flow_smoke(case: PreflightCase) -> None:
    # Honest naming: NOT E1/E2/E3 readiness.
    assert READINESS_CLASS == "partial_flow_smoke"
    assert _fake_run(case, []).readiness_class == "partial_flow_smoke"


def test_product_map_resolves_against_real_manifest(normal_run) -> None:
    import json

    rows = normal_run.resolved_product_map()
    assert rows, "product map must resolve against the real manifest"
    for row in rows:
        assert row["matches"], f"declared != observed: {row}"
    # A verifiable diagnostic artifact is persisted for the mapping.
    artifact = Path(normal_run.run_dir) / "preflight_product_map.json"
    assert artifact.is_file()
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    assert payload["readiness_class"] == "partial_flow_smoke"
    assert payload["task_id"] == normal_run.case.task_id


def test_only_table_one_is_produced_offline(normal_run) -> None:
    # Anti-overclaim: offline the ONLY genuinely produced formal suite output is
    # "table one".  Figures are not_produced_offline; audits are planned_only.
    assert normal_run.produced_suite_outputs() == ["table one"]


def test_product_map_negative_missing_mapping(case: PreflightCase) -> None:
    # Removing any single mapping breaks the one-to-one suite coverage.
    trimmed = case.product_map[:-1]
    n = len(case.expected_products)
    assert sorted(m.output_index for m in trimmed) != list(range(n))


def test_product_map_negative_dropped_produced_artifact() -> None:
    # If the produced Table 1 artifact evidence is absent, the "produced" mapping
    # resolves to planned_only -> a strict match would fail.
    mapping = next(
        m for m in E1.product_map if m.declared_fulfillment == FULFILLMENT_PRODUCED
    )
    prefix = mapping.artifact_evidence_prefix
    good = _fake_run(
        E1,
        [
            {
                "step_id": mapping.step_id,
                "status": "ok",
                "evidence_ids": [prefix + "abc"],
            }
        ],
    )
    assert observe_product_fulfillment(mapping, good) == FULFILLMENT_PRODUCED
    dropped = _fake_run(
        E1, [{"step_id": mapping.step_id, "status": "ok", "evidence_ids": []}]
    )
    assert observe_product_fulfillment(mapping, dropped) == FULFILLMENT_PLANNED_ONLY


def test_not_produced_offline_mapping_has_no_step() -> None:
    # The sealed publication figure is honestly mapped to no plan step.
    fig = next(
        m
        for m in E1.product_map
        if m.declared_fulfillment == FULFILLMENT_NOT_PRODUCED_OFFLINE
    )
    assert fig.step_id is None


# ---------------------------------------------------------------------------
# P1-B — isolation structured block wired into the real harness/CLI
# ---------------------------------------------------------------------------


def _force_isolation_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(
        rt,
        "probe_isolation_backend",
        lambda: rt.IsolationCapability(
            backend="macos_sandbox_exec",
            available=False,
            returncode=71,
            detail="sandbox-exec: sandbox_apply: Operation not permitted",
        ),
    )


def test_run_preflight_does_not_start_pipeline_when_unavailable(
    tmp_path, monkeypatch
) -> None:
    _force_isolation_unavailable(monkeypatch)
    calls = {"n": 0}

    def _spy_build(*args, **kwargs):
        calls["n"] += 1
        raise AssertionError("pipeline must not be built when isolation is blocked")

    monkeypatch.setattr(hn, "build_pipeline", _spy_build)
    run = run_preflight(E1, workdir=tmp_path)

    assert calls["n"] == 0, "build_pipeline / pipeline.run must be called 0 times"
    assert run.pipeline_ran is False
    assert run.blocked_reason is not None
    assert run.blocked_reason.startswith("isolation_backend_unavailable:")
    assert run.runtime_manifest is not None
    assert run.runtime_manifest.integration_ready is False
    # RuntimeManifest is persisted to the real run_dir even when blocked.
    assert (tmp_path / "preflight_runtime_manifest.json").is_file()


def test_cli_returns_structured_blocked_exit_code(monkeypatch, capsys) -> None:
    _force_isolation_unavailable(monkeypatch)

    def _spy_build(*args, **kwargs):
        raise AssertionError("CLI must not build the pipeline when blocked")

    monkeypatch.setattr(hn, "build_pipeline", _spy_build)
    rc = hn._cli("e1")
    assert rc == 2
    out = capsys.readouterr().out.lower()
    assert '"blocked": true' in out
    assert "isolation_backend_unavailable" in out


def test_blocking_step_isolation_converts_a_denied_step() -> None:
    manifest = {
        "per_step_records": [
            {
                "step_id": "02_table_one",
                "status": "repair_failed",
                "returncode": 71,
                "timed_out": False,
                "stderr": "sandbox-exec: sandbox_apply: Operation not permitted",
            }
        ]
    }
    unavailable = rt.IsolationCapability("macos_sandbox_exec", False, 71, "denied")
    reason = blocking_step_isolation(manifest, unavailable)
    assert reason is not None
    assert reason.startswith("isolation_backend_unavailable:")
    assert "02_table_one" in reason
    # An available backend never reclassifies a step (the step can't be denied).
    available = rt.IsolationCapability("macos_sandbox_exec", True, 0, "")
    assert blocking_step_isolation(manifest, available) is None


def test_blocking_step_isolation_ignores_plain_exit_71() -> None:
    manifest = {
        "per_step_records": [
            {
                "step_id": "02_table_one",
                "status": "repair_failed",
                "returncode": 71,
                "timed_out": False,
                "stderr": "SystemExit: 71",
            }
        ]
    }
    unavailable = rt.IsolationCapability("macos_sandbox_exec", False, 71, "denied")
    assert blocking_step_isolation(manifest, unavailable) is None


# ---------------------------------------------------------------------------
# P1-C — zero-network boundary cannot be relaxed by an environment variable
# ---------------------------------------------------------------------------


def test_preflight_pins_no_network_runner_kwargs() -> None:
    assert PREFLIGHT_RUNNER_KWARGS["network_policy"] == "none"
    assert PREFLIGHT_RUNNER_KWARGS["allow_unsafe_host_fallback"] is False


def test_build_pipeline_forwards_pinned_runner_kwargs(tmp_path) -> None:
    pipe = build_pipeline(E1, workdir=tmp_path, llm=ScriptedPreflightLLM(E1))
    assert pipe._runner_kwargs.get("network_policy") == "none"
    assert pipe._runner_kwargs.get("allow_unsafe_host_fallback") is False


def test_env_var_cannot_relax_host_fallback_when_pinned(tmp_path, monkeypatch) -> None:
    import pandas as pd

    from easyicu.research_agent import CodeRunner

    cohort = tmp_path / "cohort.parquet"
    pd.DataFrame({"stay_id": [1, 2], "death": [0, 1]}).to_parquet(cohort)
    monkeypatch.setenv("EASYICU_ALLOW_UNSAFE_HOST_FALLBACK", "1")

    # Pinned explicit False -> the env var is IGNORED (fail-closed).
    pinned = CodeRunner(
        workdir=tmp_path / "w_pinned", cohort_parquet=cohort, **PREFLIGHT_RUNNER_KWARGS
    )
    assert pinned.allow_unsafe_host_fallback is False
    assert pinned.network_policy == "none"

    # Control: NOT pinned (None) -> the env var DOES relax it, proving the pin
    # is what defeats the environment (this is the boundary we must not leave open).
    unpinned = CodeRunner(
        workdir=tmp_path / "w_unpinned",
        cohort_parquet=cohort,
        allow_unsafe_host_fallback=None,
    )
    assert unpinned.allow_unsafe_host_fallback is True


def test_network_policy_report_all_none_and_non_degraded(normal_run) -> None:
    report = normal_run.network_policy_report()
    assert report["subprocess_steps"], "at least one subprocess step must run"
    assert report["ok"] is True
    for step in report["subprocess_steps"]:
        assert step["requested_network_policy"] == "none"
        assert step["isolation_degraded"] is False
