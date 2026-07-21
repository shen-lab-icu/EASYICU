"""Unit contract for repair_coordination (A2 batch-1 extraction).

These tests pin the EXACT behavioral surface of the four budget-accounting
closures extracted from ``pipeline_execute``: step_record key names and write
order, the neutral provider probe (never a consume), the persisted
``step_llm_repair_classes`` contract that resume replays monotonically, and
the all-or-nothing authorization semantics of the deterministic concept
repair.
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
from types import SimpleNamespace

import pytest

from easyicu.research_agent.execution import phase as pipeline_execute
from easyicu.research_agent.authority.provider_budget import (
    ProviderCallBudgetReceiptError,
    StepProviderCallBudget,
    load_provider_call_budget_state,
)
from easyicu.research_agent.repairs.coordination import (
    RepairAuthorityBinding,
    RepairCoordinator,
    StepRepairBudget,
    authorized_deterministic_concept_repair,
    resume_deterministic_repair_candidate,
)
from easyicu.research_agent.schema import ValidationFinding

STEP_ID = "02_exposure_derivation_and_qc"


def _authority_binding(
    *,
    attempt_id: int = 1,
    repair_class: str = "concept",
    provider_category: str = "concept_repair",
) -> RepairAuthorityBinding:
    return RepairAuthorityBinding(
        step_id=STEP_ID,
        attempt_id=attempt_id,
        repair_class=repair_class,
        provider_category=provider_category,
        before_code_sha256="a" * 64,
        step_spec_sha256="b" * 64,
        resolved_inputs_sha256="c" * 64,
        coder_context_sha256="d" * 64,
        repair_ticket_sha256="e" * 64,
        engine_validator_sha256="f" * 64,
        prompt_pack_version="test-prompts/1",
        run_input_capsule_sha256="1" * 64,
    )


def _budget(tmp_path, *, limit: int = 7):
    return StepProviderCallBudget(
        limit,
        step_id=STEP_ID,
        receipt_path=tmp_path / "receipt.json",
        reserved_final_category="concept_audit",
    )


def _repair_budget(tmp_path, *, limit: int = 7, max_llm: int = 3, initial: int = 0):
    provider = _budget(tmp_path, limit=limit)
    step_record: dict = {}
    budget = StepRepairBudget(
        provider_budget=provider,
        step_record=step_record,
        max_llm_repairs=max_llm,
        initial_llm_repair_attempts=initial,
        provider_receipt_relative_path=".runtime/provider_call_budgets/x.json",
    )
    return provider, step_record, budget


def _complete_attempt(
    provider: StepProviderCallBudget,
    attempt_id: int,
    *,
    category: str,
) -> None:
    provider.consume(category)
    provider.complete_logical_repair_transport(
        attempt_id=attempt_id,
        mode="minimal_patch",
        after_code_sha256=hashlib.sha256(
            f"# repaired {attempt_id}\n".encode("utf-8")
        ).hexdigest(),
    )


def test_resume_preflight_repairs_prior_runtime_failure_before_provider(tmp_path):
    step_dir = tmp_path / "steps" / "cluster_selection"
    (step_dir / "outputs").mkdir(parents=True)
    (step_dir / "run.log").write_text(
        "TypeError: Unsupported JSON value: <class "
        "'sklearn.mixture._gaussian_mixture.GaussianMixture'>",
        encoding="utf-8",
    )
    # A torn summary must not suppress the exact prior runtime repair.
    (step_dir / "outputs" / "step_summary.json").write_text(
        '{"candidate_fit_diagnostics": [', encoding="utf-8"
    )
    code = """fitted_models = {}
fitted_models[model_id] = {"model": model, "labels": labels, "n_clusters": 4}
candidate_diagnostics = []
for record in fitted_models.values():
    candidate_diagnostics.append(record)
step_summary = {"candidate_fit_diagnostics": candidate_diagnostics}
"""

    candidate = resume_deterministic_repair_candidate(
        code=code,
        step_dir=step_dir,
        analysis_family="trajectory_clustering",
    )

    assert candidate is not None
    (repair_id, repaired), source, trigger = candidate
    assert repair_id == "sklearn_runtime_object_diagnostics_v1"
    assert source == "resume_runner_repair_preflight"
    assert trigger["run_log_path"] == str(step_dir / "run.log")
    assert 'if key not in {"model", "labels"}' in repaired


def test_sync_provider_writes_exact_key_set(tmp_path):
    provider, step_record, budget = _repair_budget(tmp_path)
    budget.sync_provider()
    assert list(step_record) == [
        "step_provider_call_budget_scope",
        "step_provider_call_budget",
        "step_provider_call_attempts",
        "step_provider_call_remaining",
        "step_provider_call_budget_exhausted",
        "step_provider_call_categories",
        "step_provider_call_reserved_category",
        "step_provider_call_reservation_released",
        "step_provider_call_receipt_version",
        "step_llm_repair_transport_states",
        "step_provider_call_receipt",
    ]
    assert (
        step_record["step_provider_call_budget_scope"]
        == "coder_generation_repair_concept_audit_and_analyzer"
    )
    # receipt path is only reported once something was actually paid
    assert step_record["step_provider_call_receipt"] is None
    provider.consume("initial_generation")
    budget.sync_provider()
    assert (
        step_record["step_provider_call_receipt"]
        == ".runtime/provider_call_budgets/x.json"
    )


def test_probe_never_consumes_or_touches_receipt(tmp_path):
    provider, step_record, budget = _repair_budget(tmp_path)
    assert budget.provider_available()
    assert provider.used == 0
    assert not (tmp_path / "receipt.json").exists()


def test_probe_refusal_records_unavailable_and_syncs(tmp_path):
    provider, step_record, budget = _repair_budget(tmp_path, limit=1)
    # only the reserved audit slot remains -> non-audit probe is refused
    assert not budget.provider_available()
    assert step_record["step_provider_call_repair_unavailable"] is True
    assert step_record["step_provider_call_budget"] == 1  # sync ran
    assert provider.used == 0  # still no consume


def test_consume_appends_repair_classes_in_order(tmp_path):
    provider, step_record, budget = _repair_budget(tmp_path, max_llm=3)
    assert budget.consume("concept")
    _complete_attempt(provider, 1, category="concept_repair_patch")
    assert budget.consume("runtime")
    assert step_record["step_llm_repair_attempts"] == 2
    assert step_record["step_llm_repair_budget"] == 3
    assert step_record["step_llm_repair_classes"] == ["concept", "runtime"]
    payload = json.loads((tmp_path / "receipt.json").read_text(encoding="utf-8"))
    assert [entry["repair_class"] for entry in payload["logical_repairs"]] == [
        "concept",
        "runtime",
    ]


def test_consume_persists_repair_authority_in_single_provider_receipt(tmp_path):
    _provider, step_record, budget = _repair_budget(tmp_path, max_llm=3)
    binding = _authority_binding()

    assert budget.consume("concept", authority_binding=binding)

    payload = json.loads((tmp_path / "receipt.json").read_text(encoding="utf-8"))
    entry = payload["logical_repairs"][0]
    assert entry["binding"] == binding.payload()
    assert entry["binding_sha256"] == binding.sha256
    assert step_record["step_llm_repair_bindings"] == [binding.sha256]


def test_consume_rejects_authority_for_wrong_attempt_or_class(tmp_path):
    _provider, _step_record, budget = _repair_budget(tmp_path, max_llm=3)

    with pytest.raises(ValueError, match="attempt_id"):
        budget.consume(
            "concept",
            authority_binding=_authority_binding(attempt_id=2),
        )
    with pytest.raises(ValueError, match="class"):
        budget.consume(
            "runtime",
            authority_binding=_authority_binding(repair_class="concept"),
        )


def test_every_pipeline_llm_repair_reservation_is_authority_bound():
    tree = ast.parse(inspect.getsource(pipeline_execute.run_execute_phase))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_consume_llm_repair_budget"
    ]

    assert len(calls) == 7
    for call in calls:
        keywords = {keyword.arg for keyword in call.keywords}
        assert {
            "before_code",
            "repair_ticket",
            "repair_authority",
            "provider_category",
            "failure_status",
        } <= keywords
        assert keywords <= {
            "before_code",
            "repair_ticket",
            "repair_authority",
            "current_repair_authority",
            "provider_category",
            "failure_status",
        }
    reserved_categories = sorted(
        keyword.value.value
        for call in calls
        for keyword in call.keywords
        if keyword.arg == "provider_category"
        and isinstance(keyword.value, ast.Constant)
    )
    coder_categories = sorted(
        keyword.value.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_repair_with_capsule"
        for keyword in node.keywords
        if keyword.arg == "provider_category"
        and isinstance(keyword.value, ast.Constant)
    )
    assert reserved_categories == sorted([*coder_categories, "compatibility_repair"])


def test_runtime_repair_uses_empty_typed_authority_side_channel():
    tree = ast.parse(inspect.getsource(pipeline_execute.run_execute_phase))
    empty_runtime_assignments = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "runtime_repair_authority"
            for target in node.targets
        )
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "RepairPromptAuthority"
        and not node.value.args
        and not node.value.keywords
    ]
    assert len(empty_runtime_assignments) == 1

    for function_name in ("_consume_llm_repair_budget", "_repair_with_capsule"):
        runtime_calls = []
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == function_name
            ):
                continue
            keywords = {keyword.arg: keyword.value for keyword in node.keywords}
            category = keywords.get("provider_category")
            if (
                isinstance(category, ast.Constant)
                and category.value == "runtime_repair"
            ):
                runtime_calls.append(keywords)
        assert len(runtime_calls) == 1
        authority = runtime_calls[0]["repair_authority"]
        assert isinstance(authority, ast.Name)
        assert authority.id == "runtime_repair_authority"


def test_every_pipeline_coder_repair_binds_current_logical_attempt():
    tree = ast.parse(inspect.getsource(pipeline_execute.run_execute_phase))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_repair_with_capsule"
    ]

    assert len(calls) == 6
    for call in calls:
        keywords = {keyword.arg for keyword in call.keywords}
        assert "logical_repair_attempt_id" in keywords


def test_durable_ledger_recovers_attempt_missing_from_step_snapshot(tmp_path):
    provider, _first_record, first = _repair_budget(tmp_path, max_llm=3)
    assert first.consume("concept")

    payload = json.loads((tmp_path / "receipt.json").read_text(encoding="utf-8"))
    restored_provider = StepProviderCallBudget(
        payload["limit"],
        step_id=STEP_ID,
        consumed_categories=tuple(payload["categories"]),
        logical_repair_entries=tuple(payload["logical_repairs"]),
        receipt_path=tmp_path / "receipt.json",
        reserved_final_category="concept_audit",
    )
    resumed_record: dict = {}
    resumed = StepRepairBudget(
        provider_budget=restored_provider,
        step_record=resumed_record,
        max_llm_repairs=3,
        initial_llm_repair_attempts=0,
        initial_repair_classes=(),
        provider_receipt_relative_path=".runtime/provider_call_budgets/x.json",
    )

    assert resumed.llm_repair_attempts == 1
    assert resumed_record["step_llm_repair_classes"] == ["concept"]
    assert resumed.next_attempt_id == 1
    assert resumed.consume("concept")
    assert resumed_record["step_llm_repair_attempts"] == 1
    assert resumed_record["step_llm_repair_classes"] == ["concept"]
    _complete_attempt(restored_provider, 1, category="concept_repair_patch")
    assert resumed.consume("runtime")
    assert resumed_record["step_llm_repair_attempts"] == 2
    assert resumed_record["step_llm_repair_classes"] == ["concept", "runtime"]


def test_sync_reports_receipt_after_logical_reservation_before_provider_call(
    tmp_path,
):
    provider, step_record, budget = _repair_budget(tmp_path)
    assert budget.consume("contract")
    assert provider.used == 0

    budget.sync_provider()

    assert (
        step_record["step_provider_call_receipt"]
        == ".runtime/provider_call_budgets/x.json"
    )


def test_logical_exhaustion_marks_record_and_refuses(tmp_path):
    provider, step_record, budget = _repair_budget(tmp_path, max_llm=1)
    assert budget.consume("concept")
    _complete_attempt(provider, 1, category="concept_repair_patch")
    assert not budget.consume("contract")
    assert step_record["step_llm_repair_budget_exhausted"] is True
    assert step_record["step_llm_repair_classes"] == ["concept"]
    assert budget.llm_repair_attempts == 1


def test_resume_initial_attempts_count_against_allowance(tmp_path):
    provider, step_record, budget = _repair_budget(tmp_path, max_llm=3, initial=3)
    assert not budget.logical_available()
    assert not budget.consume("runtime")
    assert "step_llm_repair_classes" not in step_record  # nothing new appended


def test_authorized_repair_is_all_or_nothing():
    script = "helper_result = {}\nassert isinstance(helper_result, dict)\n"

    calls: list = []

    def approve(payload, **kwargs):
        calls.append(payload)
        return payload

    def deny(payload, **kwargs):
        calls.append(payload)
        return None

    # no matching mechanical repair -> untouched, no authorization attempted
    code, names = authorized_deterministic_concept_repair(
        "x = 1\n", ["unrelated"], authorize=approve, step=None, source="test"
    )
    assert (code, names) == ("x = 1\n", [])

    # a denied authorization rejects the WHOLE candidate even if it matched
    code, names = authorized_deterministic_concept_repair(
        script,
        ["never require `isinstance(helper_result, dict)`"],
        authorize=deny,
        step=None,
        source="test",
    )
    assert code == script
    assert names == []


def test_authorized_repair_applies_host_proven_binary_domain_guard():
    script = """
numeric = {"exposure": frame["exposure"]}
selected = numeric["exposure"]
group = selected.copy().astype(object)
group.loc[selected <= 0] = "No exposure"
group.loc[selected > 0] = "Exposure"
"""
    context = SimpleNamespace(
        primary_exposure="exposure",
        variables=[
            SimpleNamespace(
                name="exposure",
                observed_domain={"is_binary": True, "n_unique": 2},
            )
        ],
    )
    finding = ValidationFinding(
        validator="llm_concept_auditor",
        severity="error",
        message="Untrusted prose is not routing authority.",
        detail={"issue_code": "other", "variables": ["exposure"]},
    )

    code, names = authorized_deterministic_concept_repair(
        script,
        [finding.message],
        repair_findings=[finding],
        authorize=lambda payload, **_: payload,
        step=None,
        source="test",
        context=context,
    )

    assert names == ["observed_binary_primary_exposure_guard_v1"]
    assert "if not bool(selected.dropna().isin([0, 1]).all()):" in code


def test_receipt_on_disk_matches_snapshot_projection(tmp_path):
    provider, step_record, budget = _repair_budget(tmp_path)
    provider.consume("initial_generation")
    provider.consume("concept_repair")
    budget.sync_provider()
    payload = json.loads((tmp_path / "receipt.json").read_text(encoding="utf-8"))
    assert payload["categories"] == step_record["step_provider_call_categories"]
    assert payload["limit"] == step_record["step_provider_call_budget"]


def test_repair_coordinator_keeps_patch_as_default_without_audit_reservation():
    calls: list[str] = []
    provider = StepProviderCallBudget(2, step_id=STEP_ID)
    coordinator = RepairCoordinator(
        provider_budget=provider,
        provider_category="runtime_repair",
        normalize_script=lambda value: value,
        is_executable_script=lambda value: value.startswith("import "),
    )

    result = coordinator.repair(
        code="import os\nvalue = 1\n",
        patch_call=lambda: calls.append("patch") or "not-json",
        full_rewrite_call=lambda _reason: calls.append("rewrite")
        or "import os\nvalue = 2\n",
    )

    assert calls == ["patch", "rewrite"]
    assert result.mode == "full_rewrite"
    assert result.provider_calls == 2
    assert provider.categories == (
        "runtime_repair_patch",
        "runtime_repair_full_rewrite",
    )


def test_patch_response_full_script_requires_authorized_rewrite_transport():
    calls: list[str] = []
    provider = StepProviderCallBudget(2, step_id=STEP_ID)
    coordinator = RepairCoordinator(
        provider_budget=provider,
        provider_category="contract_repair",
        normalize_script=lambda value: value,
        is_executable_script=lambda value: value.startswith("import "),
    )

    result = coordinator.repair(
        code="import os\nvalue = 1\n",
        patch_call=lambda: calls.append("patch") or "import os\nvalue = 2\n",
        full_rewrite_call=lambda _reason: calls.append("rewrite")
        or "import os\nvalue = 3\n",
    )

    assert calls == ["patch", "rewrite"]
    assert result.mode == "full_rewrite"
    assert result.code == "import os\nvalue = 3"
    assert result.provider_calls == 2
    assert provider.categories == (
        "contract_repair_patch",
        "contract_repair_full_rewrite",
    )


def test_repair_coordinator_persists_completed_transport_before_return(tmp_path):
    provider = _budget(tmp_path)
    assert provider.reserve_logical_repair("runtime", max_repairs=2) == 1
    coordinator = RepairCoordinator(
        provider_budget=provider,
        provider_category="runtime_repair",
        normalize_script=lambda value: value,
        is_executable_script=lambda value: value.startswith("import "),
    )

    result = coordinator.repair(
        code="import os\nvalue = 1\n",
        patch_call=lambda: json.dumps(
            {
                "format": "easyicu.code_patch/1",
                "edits": [
                    {
                        "old": "value = 1",
                        "new": "value = 2",
                        "expected_count": 1,
                    }
                ],
            }
        ),
        full_rewrite_call=lambda _reason: "unused",
        logical_repair_attempt_id=1,
    )

    state = load_provider_call_budget_state(
        tmp_path / "receipt.json",
        step_id=STEP_ID,
        expected_reserved_final_category="concept_audit",
    )
    transport = state.logical_repairs[0]["transport"]
    assert transport["state"] == "completed"
    assert transport["mode"] == "minimal_patch"
    assert (
        transport["after_code_sha256"]
        == hashlib.sha256(result.code.encode("utf-8")).hexdigest()
    )


def test_repair_coordinator_persists_failed_transport_before_reraising(tmp_path):
    provider = _budget(tmp_path)
    assert provider.reserve_logical_repair("runtime", max_repairs=2) == 1
    coordinator = RepairCoordinator(
        provider_budget=provider,
        provider_category="runtime_repair",
        normalize_script=lambda value: value,
        is_executable_script=lambda value: value.startswith("import "),
    )

    with pytest.raises(RuntimeError, match="provider unavailable"):
        coordinator.repair(
            code="import os\nvalue = 1\n",
            patch_call=lambda: (_ for _ in ()).throw(
                RuntimeError("provider unavailable")
            ),
            full_rewrite_call=lambda _reason: "unused",
            logical_repair_attempt_id=1,
        )

    state = load_provider_call_budget_state(
        tmp_path / "receipt.json",
        step_id=STEP_ID,
        expected_reserved_final_category="concept_audit",
    )
    assert state.logical_repairs[0]["transport"]["state"] == "failed"
    assert state.logical_repairs[0]["transport"]["error_type"] == "RuntimeError"


def test_repair_preflight_fails_transport_before_any_provider_charge(tmp_path):
    provider = _budget(tmp_path)
    assert provider.reserve_logical_repair("runtime", max_repairs=2) == 1
    coordinator = RepairCoordinator(
        provider_budget=provider,
        provider_category="runtime_repair",
        normalize_script=lambda value: value,
        is_executable_script=lambda value: value.startswith("import "),
    )
    provider_called = False

    def patch_call():
        nonlocal provider_called
        provider_called = True
        return "unused"

    with pytest.raises(RuntimeError, match="prompt transport too large"):
        coordinator.repair(
            code="import os\nvalue = 1\n",
            patch_preflight=lambda: (_ for _ in ()).throw(
                RuntimeError("prompt transport too large")
            ),
            patch_call=patch_call,
            full_rewrite_call=lambda _reason: "unused",
            logical_repair_attempt_id=1,
        )

    state = load_provider_call_budget_state(
        tmp_path / "receipt.json",
        step_id=STEP_ID,
        expected_reserved_final_category="concept_audit",
    )
    assert provider_called is False
    assert provider.categories == ()
    assert state.logical_repairs[0]["transport"]["state"] == "failed"
    assert state.logical_repairs[0]["transport"]["error_type"] == "RuntimeError"


def test_direct_rewrite_preflight_fails_before_any_provider_charge(tmp_path):
    provider = _budget(tmp_path, limit=2)
    assert provider.reserve_logical_repair("runtime", max_repairs=2) == 1
    coordinator = RepairCoordinator(
        provider_budget=provider,
        provider_category="runtime_repair",
        normalize_script=lambda value: value,
        is_executable_script=lambda value: value.startswith("import "),
    )
    provider_called = False

    def rewrite_call(_reason):
        nonlocal provider_called
        provider_called = True
        return "unused"

    with pytest.raises(RuntimeError, match="rewrite prompt too large"):
        coordinator.repair(
            code="import os\nvalue = 1\n",
            patch_call=lambda: "unused",
            full_rewrite_preflight=lambda _reason: (_ for _ in ()).throw(
                RuntimeError("rewrite prompt too large")
            ),
            full_rewrite_call=rewrite_call,
            logical_repair_attempt_id=1,
        )

    state = load_provider_call_budget_state(
        tmp_path / "receipt.json",
        step_id=STEP_ID,
        expected_reserved_final_category="concept_audit",
    )
    assert provider_called is False
    assert provider.categories == ()
    assert state.logical_repairs[0]["transport"]["state"] == "failed"


def test_fallback_rewrite_preflight_preserves_only_paid_patch_call(tmp_path):
    provider = _budget(tmp_path)
    assert provider.reserve_logical_repair("runtime", max_repairs=2) == 1
    coordinator = RepairCoordinator(
        provider_budget=provider,
        provider_category="runtime_repair",
        normalize_script=lambda value: value,
        is_executable_script=lambda value: value.startswith("import "),
    )
    rewrite_called = False

    def rewrite_call(_reason):
        nonlocal rewrite_called
        rewrite_called = True
        return "unused"

    with pytest.raises(RuntimeError, match="rewrite prompt too large"):
        coordinator.repair(
            code="import os\nvalue = 1\n",
            patch_call=lambda: "not a patch",
            full_rewrite_preflight=lambda _reason: (_ for _ in ()).throw(
                RuntimeError("rewrite prompt too large")
            ),
            full_rewrite_call=rewrite_call,
            logical_repair_attempt_id=1,
        )

    state = load_provider_call_budget_state(
        tmp_path / "receipt.json",
        step_id=STEP_ID,
        expected_reserved_final_category="concept_audit",
    )
    assert rewrite_called is False
    assert provider.categories == ("runtime_repair_patch",)
    assert state.logical_repairs[0]["transport"]["state"] == "failed"


def test_repair_coordinator_rejects_mismatched_bound_provider_category(tmp_path):
    provider = _budget(tmp_path)
    binding = _authority_binding(
        repair_class="runtime",
        provider_category="runtime_repair",
    )
    assert (
        provider.reserve_logical_repair(
            "runtime",
            max_repairs=2,
            binding=binding.payload(),
            binding_sha256=binding.sha256,
        )
        == 1
    )
    coordinator = RepairCoordinator(
        provider_budget=provider,
        provider_category="contract_repair",
        normalize_script=lambda value: value,
        is_executable_script=lambda value: value.startswith("import "),
    )
    provider_called = False

    def patch_call():
        nonlocal provider_called
        provider_called = True
        return "unused"

    with pytest.raises(ProviderCallBudgetReceiptError, match="provider category"):
        coordinator.repair(
            code="import os\nvalue = 1\n",
            patch_call=patch_call,
            full_rewrite_call=lambda _reason: "unused",
            logical_repair_attempt_id=1,
        )

    assert provider_called is False
    assert provider.categories == ()
    assert provider.logical_repair_transport_states == ("pending",)


def test_transport_receipt_failure_never_returns_unsealed_repaired_code(
    tmp_path,
    monkeypatch,
):
    provider = _budget(tmp_path)
    assert provider.reserve_logical_repair("runtime", max_repairs=2) == 1
    coordinator = RepairCoordinator(
        provider_budget=provider,
        provider_category="runtime_repair",
        normalize_script=lambda value: value,
        is_executable_script=lambda value: value.startswith("import "),
    )
    original_persist = provider._persist_locked
    persist_calls = 0

    def fail_on_transport_terminal():
        nonlocal persist_calls
        persist_calls += 1
        if persist_calls == 2:
            raise ProviderCallBudgetReceiptError("simulated terminal write failure")
        original_persist()

    monkeypatch.setattr(provider, "_persist_locked", fail_on_transport_terminal)

    with pytest.raises(ProviderCallBudgetReceiptError, match="terminal write failure"):
        coordinator.repair(
            code="import os\nvalue = 1\n",
            patch_call=lambda: json.dumps(
                {
                    "format": "easyicu.code_patch/1",
                    "edits": [
                        {
                            "old": "value = 1",
                            "new": "value = 2",
                            "expected_count": 1,
                        }
                    ],
                }
            ),
            full_rewrite_call=lambda _reason: "unused",
            logical_repair_attempt_id=1,
        )

    assert provider.logical_repair_transport_states == ("pending",)
    assert provider.categories == ("runtime_repair_patch",)
