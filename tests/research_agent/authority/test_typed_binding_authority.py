"""Architecture contract for typed-input binding authority."""

from __future__ import annotations

import ast
import inspect
import os
import re
from pathlib import Path
import subprocess
import sys
import threading
from types import SimpleNamespace

import pytest

from easyicu.research_agent.execution import candidate_loop, phase as execution_phase
from easyicu.research_agent.authority import typed_binding
from easyicu.research_agent.schema import AnalysisPlan, EvidenceRef

_PHASE_TYPED_BINDING_NAMES = {
    "TypedBindingResolver",
    "_EvidenceLineageResolutionError",
    "_assignment_model_authority_context_block",
    "_declared_typed_artifact_paths",
    "_declared_typed_product_paths",
    "_evidence_kind_matches_typed_product",
    "_evidence_record_field",
    "_current_verified_evidence_record",
    "_lineage_failure_product_fields",
    "_normalise_typed_product_name",
    "_registered_source_name",
    "_resolve_typed_artifact_evidence",
    "_resolve_typed_input_evidence",
    "_resolved_typed_input_binding",
    "_resume_typed_input_bindings",
    "_resume_typed_input_bindings_fingerprint",
    "_step_summary_statistic_values",
    "_typed_artifact_name",
    "_typed_input_product",
    "_typed_parent_schema_context_block",
    "_write_host_input_binding_receipts",
    "_write_resolved_inputs_manifest",
}


def _top_level_function_calls(tree: ast.Module) -> dict[str, set[str]]:
    calls: dict[str, set[str]] = {}
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        calls[node.name] = {
            call.func.attr if isinstance(call.func, ast.Attribute) else call.func.id
            for call in ast.walk(node)
            if isinstance(call, ast.Call)
            and isinstance(call.func, (ast.Attribute, ast.Name))
        }
    return calls


def test_execution_phase_uses_typed_binding_objects_with_identity() -> None:
    # 23 private helpers plus two public rules: which producers the host
    # writes receipts for, and how the host-staged ambient trajectory is
    # named in the step's own resolved-inputs record.
    assert len(typed_binding.__all__) == 27
    assert {
        "host_owns_input_binding_receipts",
        "host_authorized_ambient_trajectory_entry",
    } <= set(typed_binding.__all__)
    for name in _PHASE_TYPED_BINDING_NAMES:
        assert getattr(execution_phase, name) is getattr(typed_binding, name)
    assert _PHASE_TYPED_BINDING_NAMES < set(typed_binding.__all__)


def test_standard_executors_receive_host_owned_input_receipts() -> None:
    source = inspect.getsource(candidate_loop._candidate_success_prepare_transition)
    receipt_call = "state.visual_step_summary = _write_host_input_binding_receipts("
    receipt_index = source.index(receipt_call)
    # The guard is the shared owner rather than a hand-written condition; the
    # two spellings of that condition disagreed, and this pre-gate site was the
    # narrower one.  ``test_host_writes_the_receipt_before_the_gate_demands_it``
    # owns the rule itself; what this test still owns is its PLACEMENT.
    guard_index = source.rindex(
        "if host_owns_input_binding_receipts(",
        0,
        receipt_index,
    )
    # "Immediately governs" stated structurally rather than as a character
    # budget: no other branch may open between the guard and the write.  A
    # character distance moved when the guard became a multi-line call and
    # said nothing about what it was meant to protect.
    between = source[
        guard_index + len("if host_owns_input_binding_receipts(") : receipt_index
    ]
    assert not re.search(
        r"\n\s*(if|elif|else|for|while|try)\b", between
    ), "another branch opens between the receipt guard and the receipt write"
    assert receipt_index < source.index("visual_gate = collect_visual_gate_result(")


def test_typed_binding_has_no_orchestration_or_scientific_owner_dependency() -> None:
    tree = ast.parse(inspect.getsource(typed_binding))
    imported_leaves = {
        node.module.rsplit(".", 1)[-1]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    identifiers = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)} | {
        node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
    }

    assert imported_leaves.isdisjoint(
        {"pipeline", "pipeline_execute", "gates", "execution"}
    )
    assert identifiers.isdisjoint(
        {
            "EvidenceStore",
            "LLMConceptAuditor",
            "StepProviderCallBudget",
            "complete_with_provider_budget",
            "consume",
            "promote",
            "register",
            "repair",
            "write_run_checkpoint",
        }
    )


def test_typed_binding_writes_only_its_two_caller_scoped_receipts() -> None:
    calls = _top_level_function_calls(ast.parse(inspect.getsource(typed_binding)))
    writers = {
        name
        for name, function_calls in calls.items()
        if function_calls & {"mkdir", "replace", "write_text", "write_bytes"}
    }
    assert writers == {
        "_write_host_input_binding_receipts",
        "_write_resolved_inputs_manifest",
    }


def test_execute_loop_uses_one_typed_resolver_without_nested_implementation() -> None:
    source = (
        inspect.getsource(execution_phase.run_execute_phase)
        + "\n"
        + inspect.getsource(execution_phase._prepare_execute_phase_authority)
        + "\n"
        + inspect.getsource(execution_phase._execute_step)
        + "\n"
        + inspect.getsource(execution_phase._step_prepare_execution_authority)
        + "\n"
        + inspect.getsource(execution_phase._step_finalize_step)
    )
    tree = ast.parse(source)
    assert "def _evidence_refs_for_names" not in source
    constructor_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "TypedBindingResolver"
    ]
    resolver_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "resolve_names"
    ]
    assert len(constructor_calls) == 1
    assert len(resolver_calls) == 2
    assert (
        sum(
            any(keyword.arg == "consumer_step" for keyword in call.keywords)
            for call in resolver_calls
        )
        == 1
    )


def test_resolver_snapshots_current_records_under_the_shared_lock(tmp_path) -> None:
    record = SimpleNamespace(
        evidence_id="evidence-1",
        kind="table",
        description="current table",
        relative_path="evidence/evidence-1__table.csv",
    )

    class CountingLock:
        active = False
        entries = 0

        def __enter__(self) -> None:
            self.active = True
            self.entries += 1

        def __exit__(self, *_args: object) -> None:
            self.active = False

    lock = CountingLock()

    class GuardedRecords:
        def __iter__(self):
            assert lock.active
            return iter([{"step_id": "producer", "status": "ok"}])

    class Store:
        def get(self, name: str) -> object | None:
            return record if name == "current_table" else None

        def current_verified_records(self, records: object) -> list[object]:
            assert records == [{"step_id": "producer", "status": "ok"}]
            return [record]

    resolver = typed_binding.TypedBindingResolver(
        evidence_store=Store(),
        per_step_records=GuardedRecords(),
        records_lock=lock,
        run_dir=tmp_path,
        authoritative_cohort_path=tmp_path / "cohort.parquet",
    )
    refs, typed_ids, bindings = resolver.resolve_names(
        ["current_table"],
        plan=AnalysisPlan(
            research_question="Lock the resolver snapshot.",
            steps=[],
        ),
    )

    assert [ref.evidence_id for ref in refs] == ["evidence-1"]
    assert typed_ids == []
    assert bindings == {}
    assert lock.entries == 1


def test_resolver_uses_the_current_plan_on_every_call(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    observed_plans: list[AnalysisPlan] = []

    def fail_with_plan(**kwargs: object):
        observed_plans.append(kwargs["plan"])
        return None, {"input": kwargs["input_name"], "reason": "test_failure"}

    monkeypatch.setattr(
        typed_binding,
        "_resolve_typed_input_evidence",
        fail_with_plan,
    )
    resolver = typed_binding.TypedBindingResolver(
        evidence_store=SimpleNamespace(records=lambda: []),
        per_step_records=[],
        records_lock=threading.Lock(),
        run_dir=tmp_path,
        authoritative_cohort_path=tmp_path / "cohort.parquet",
    )
    first = AnalysisPlan(research_question="First plan.", steps=[])
    second = AnalysisPlan(research_question="Replanned question.", steps=[])

    with pytest.raises(typed_binding._EvidenceLineageResolutionError):
        resolver.resolve_names(["artifact:first"], plan=first)
    with pytest.raises(typed_binding._EvidenceLineageResolutionError):
        resolver.resolve_names(["artifact:second"], plan=second)

    assert observed_plans == [first, second]


def test_resolver_allows_only_exact_unpublished_evidence_ids(tmp_path) -> None:
    record = SimpleNamespace(
        evidence_id="evidence-1",
        kind="table",
        description="pending table",
        relative_path="evidence/evidence-1__table.csv",
    )

    class Store:
        def get(self, name: str) -> object | None:
            return record if name in {"evidence-1", "pending_alias"} else None

        def current_verified_records(self, _records: object) -> list[object]:
            return []

    resolver = typed_binding.TypedBindingResolver(
        evidence_store=Store(),
        per_step_records=[],
        records_lock=threading.Lock(),
        run_dir=tmp_path,
        authoritative_cohort_path=tmp_path / "cohort.parquet",
    )
    plan = AnalysisPlan(research_question="Review pending evidence.", steps=[])

    exact_refs, _, _ = resolver.resolve_names(
        ["evidence-1"],
        plan=plan,
        allow_unpublished_direct_ids=True,
    )
    alias_refs, _, _ = resolver.resolve_names(
        ["pending_alias"],
        plan=plan,
        allow_unpublished_direct_ids=True,
    )

    assert [ref.evidence_id for ref in exact_refs] == ["evidence-1"]
    assert alias_refs == []


def test_resolver_aggregates_all_typed_failures(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(**kwargs: object):
        return None, {"input": kwargs["input_name"], "reason": "missing"}

    monkeypatch.setattr(typed_binding, "_resolve_typed_input_evidence", fail)
    resolver = typed_binding.TypedBindingResolver(
        evidence_store=SimpleNamespace(records=lambda: []),
        per_step_records=[],
        records_lock=threading.Lock(),
        run_dir=tmp_path,
        authoritative_cohort_path=tmp_path / "cohort.parquet",
    )

    with pytest.raises(typed_binding._EvidenceLineageResolutionError) as raised:
        resolver.resolve_names(
            ["artifact:first", "table:second"],
            plan=AnalysisPlan(research_question="Aggregate failures.", steps=[]),
        )

    assert raised.value.failures == [
        {"input": "artifact:first", "reason": "missing"},
        {"input": "table:second", "reason": "missing"},
    ]


def test_resolver_preserves_first_seen_order_and_deduplicates_evidence(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record = SimpleNamespace(
        evidence_id="evidence-1",
        kind="table",
        description="current table",
        relative_path="evidence/evidence-1__table.csv",
    )

    def resolve_typed(**_kwargs: object):
        return (
            EvidenceRef(
                evidence_id="evidence-1",
                kind="table",
                description="current table",
                relative_path="evidence/evidence-1__table.csv",
            ),
            None,
        )

    monkeypatch.setattr(
        typed_binding,
        "_resolve_typed_input_evidence",
        resolve_typed,
    )
    monkeypatch.setattr(
        typed_binding,
        "_resolved_typed_input_binding",
        lambda **_kwargs: {"evidence_id": "evidence-1"},
    )
    store = SimpleNamespace(
        records=lambda: [record],
        get=lambda name: record if name == "current_alias" else None,
        current_verified_records=lambda _records: [record],
    )
    resolver = typed_binding.TypedBindingResolver(
        evidence_store=store,
        per_step_records=[{"step_id": "producer", "status": "ok"}],
        records_lock=threading.Lock(),
        run_dir=tmp_path,
        authoritative_cohort_path=tmp_path / "cohort.parquet",
    )

    refs, typed_ids, bindings = resolver.resolve_names(
        ["artifact:analysis_dataset", "current_alias"],
        plan=AnalysisPlan(research_question="Deduplicate evidence.", steps=[]),
    )

    assert [ref.evidence_id for ref in refs] == ["evidence-1"]
    assert typed_ids == ["evidence-1"]
    assert bindings == {"artifact:analysis_dataset": {"evidence_id": "evidence-1"}}


@pytest.mark.parametrize("canonical_first", [True, False])
def test_typed_binding_identity_survives_import_order(canonical_first: bool) -> None:
    canonical = "easyicu.research_agent.authority.typed_binding"
    legacy = "easyicu.research_agent.execution.phase"
    first, second = (canonical, legacy) if canonical_first else (legacy, canonical)
    script = f"""
import importlib
importlib.import_module({first!r})
importlib.import_module({second!r})
canonical = importlib.import_module({canonical!r})
legacy = importlib.import_module({legacy!r})
for name in {_PHASE_TYPED_BINDING_NAMES!r}:
    assert getattr(legacy, name) is getattr(canonical, name), name
"""
    env = dict(os.environ)
    source_root = str(Path(__file__).resolve().parents[3] / "src")
    env["PYTHONPATH"] = source_root + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run([sys.executable, "-c", script], check=True, env=env)
