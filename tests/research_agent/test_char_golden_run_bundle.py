"""Golden normalized run bundle for freeze-safe structural extraction (G5).

The fixture reuses the repository's smallest existing four-step typed-product
pipeline.  It runs locally with a controlled LLM and runner, no network, no
LaTeX, no visual QA, and no concept-auditor call.  The checked-in JSON is the
normalized observable baseline for post-freeze vNext refactors.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import numbers
import re
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping

import numpy as np
import pandas as pd
import pytest

_VOLATILE_FIELD_ALLOWLIST = frozenset(
    {
        "timestamp",
        "created_at",
        "started_at",
        "completed_at",
        "run_id",
        "workdir",
        "absolute_path",
        "duration_seconds",
        "elapsed_seconds",
        "pid",
    }
)
_GOLDEN_PATH = Path(__file__).with_name("fixtures") / "char_golden_run_bundle.json"
_PLAN_STEP_IDS = {
    "01_representation",
    "02_candidates",
    "03_stability",
    "04_characterization",
}
_FINDING_FIELDS = (
    "deterministic_code_findings",
    "stat_findings",
    "clinical_findings",
    "guard_findings",
    "contract_findings",
    "figure_source_findings",
    "llm_concept_findings",
)


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _canonical_table_value(value: Any) -> Any:
    """Normalize tabular scalars without trusting platform-specific float bytes."""

    if value is None or pd.isna(value):
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, numbers.Integral):
        return f"number:{int(value)}"
    if isinstance(value, numbers.Real):
        numeric = float(value)
        if not math.isfinite(numeric):
            return f"number:{numeric}"
        # LAPACK implementations may differ in the final floating-point bits.
        # Six significant digits exceed manuscript reporting precision while
        # excluding BLAS/scikit-learn tail drift from this cross-version oracle.
        return f"number:{numeric:.6g}"
    return str(value)


def _stable_file_sha256(path: Path) -> str:
    """Hash table semantics for data files and exact bytes for everything else."""

    suffix = path.suffix.lower()
    if suffix == ".csv":
        frame = pd.read_csv(path)
    elif suffix == ".parquet":
        frame = pd.read_parquet(path)
    elif suffix == ".feather":
        frame = pd.read_feather(path)
    else:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    payload = {
        "columns": [str(column) for column in frame.columns],
        "rows": [
            [_canonical_table_value(value) for value in row]
            for row in frame.itertuples(index=False, name=None)
        ],
    }
    return _canonical_sha256(payload)


def _evidence_path(run_dir: Path, relative_path: str) -> Path:
    relative = Path(relative_path)
    if relative.parts and relative.parts[0] == "evidence":
        return run_dir / relative
    return run_dir / "evidence" / relative


def _normalize(value: Any) -> Any:
    """Strip only the freeze plan's explicit volatile metadata allowlist."""

    if isinstance(value, Mapping):
        return {
            str(key): _normalize(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            if str(key) not in _VOLATILE_FIELD_ALLOWLIST
        }
    if isinstance(value, tuple):
        return [_normalize(item) for item in value]
    if isinstance(value, list):
        return [_normalize(item) for item in value]
    return value


def _load_typed_pipeline_fixture() -> ModuleType:
    path = Path(__file__).with_name("test_trajectory_stability_pipeline_success.py")
    spec = importlib.util.spec_from_file_location(
        "_easyicu_char_trajectory_fixture",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _step_id_from_output_path(path: Any) -> str | None:
    parts = Path(path).parts
    for index, part in enumerate(parts[:-2]):
        if part == "steps" and parts[index + 1] in _PLAN_STEP_IDS:
            if parts[index + 2] == "outputs":
                return parts[index + 1]
    return None


def _install_authority_order_observer(monkeypatch: pytest.MonkeyPatch):
    """Observe, without replacing, the current validate/seal/register order."""

    from easyicu.research_agent.execution import phase as pipeline_execute
    from easyicu.research_agent.authority.evidence_store import EvidenceStore

    events: list[tuple[str, str]] = []

    def record(step_id: str | None, event: str) -> None:
        if step_id not in _PLAN_STEP_IDS:
            return
        events.append((str(step_id), event))

    original_integrity_audit = pipeline_execute.StepSummaryIntegrityValidator.audit

    def integrity_audit(self, *args, **kwargs):
        step = kwargs.get("step")
        record(getattr(step, "step_id", None), "early_validate")
        return original_integrity_audit(self, *args, **kwargs)

    monkeypatch.setattr(
        pipeline_execute.StepSummaryIntegrityValidator,
        "audit",
        integrity_audit,
    )

    original_sha256 = pipeline_execute.sha256_of_file

    def sha256_of_file(path):
        record(_step_id_from_output_path(path), "seal_hash")
        return original_sha256(path)

    monkeypatch.setattr(pipeline_execute, "sha256_of_file", sha256_of_file)

    original_register_file = EvidenceStore.register_file

    def register_file(self, *args, **kwargs):
        step_id = str(kwargs.get("produced_by_step") or "")
        source_path = kwargs.get("source_path")
        if (
            kwargs.get("publish_aliases") is False
            and _step_id_from_output_path(source_path) == step_id
        ):
            record(step_id, "deferred_history_registration")
        return original_register_file(self, *args, **kwargs)

    monkeypatch.setattr(EvidenceStore, "register_file", register_file)

    original_final_gates = pipeline_execute._evaluate_final_deterministic_gates

    def final_gates(*args, **kwargs):
        step = kwargs.get("step")
        record(getattr(step, "step_id", None), "final_gate")
        return original_final_gates(*args, **kwargs)

    monkeypatch.setattr(
        pipeline_execute,
        "_evaluate_final_deterministic_gates",
        final_gates,
    )

    original_publish = EvidenceStore.publish_step_success_aliases

    def publish_step_success_aliases(self, *args, **kwargs):
        record(str(kwargs.get("step_id") or ""), "current_alias_publish")
        return original_publish(self, *args, **kwargs)

    monkeypatch.setattr(
        EvidenceStore,
        "publish_step_success_aliases",
        publish_step_success_aliases,
    )

    original_numeric = EvidenceStore.register_step_summary_numerics

    def register_step_summary_numerics(self, *args, **kwargs):
        record(str(kwargs.get("step_id") or ""), "numeric_claim_registration")
        return original_numeric(self, *args, **kwargs)

    monkeypatch.setattr(
        EvidenceStore,
        "register_step_summary_numerics",
        register_step_summary_numerics,
    )
    return events


def _run_length_encode_events(events: list[str]) -> list[dict[str, Any]]:
    """Keep exact order and call multiplicity without a noisy repeated list."""

    encoded: list[dict[str, Any]] = []
    for event in events:
        if encoded and encoded[-1]["event"] == event:
            encoded[-1]["count"] += 1
        else:
            encoded.append({"event": event, "count": 1})
    return encoded


def _typed_binding_bundle(run_dir: Path) -> dict[str, Any]:
    bundled: dict[str, Any] = {}
    for path in sorted((run_dir / "resolved_inputs").glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        inputs: dict[str, Any] = {}
        for input_key, raw in sorted((payload.get("inputs") or {}).items()):
            binding = _normalize(dict(raw))
            inputs[str(input_key)] = {
                key: binding.get(key)
                for key in (
                    "evidence_id",
                    "sha256",
                )
                if key in binding
            }
            if "identity_row" in binding:
                inputs[str(input_key)]["identity_row_sha256"] = _canonical_sha256(
                    binding["identity_row"]
                )
            if "product_contract" in binding:
                inputs[str(input_key)]["product_contract_sha256"] = _canonical_sha256(
                    binding["product_contract"]
                )
        bundled[str(payload["step_id"])] = inputs
    return bundled


def _declared_dependency_edges(plan_payload: Mapping[str, Any]) -> list[list[str]]:
    producer_by_product = {
        str(product): str(step["step_id"])
        for step in plan_payload.get("steps") or []
        for product in step.get("expected_outputs") or []
    }
    return sorted(
        [producer_by_product[str(input_name)], str(step["step_id"]), str(input_name)]
        for step in plan_payload.get("steps") or []
        for input_name in step.get("inputs") or []
        if str(input_name) in producer_by_product
    )


def _finding_bundle(
    *,
    manifest: Mapping[str, Any],
    current_records: list[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    owner_seals = {
        str(record.get("step_id") or ""): record.get("result_seal_sha256")
        for record in current_records
    }
    raw_findings: list[Mapping[str, Any]] = [
        finding
        for finding in (manifest.get("findings") or [])
        if isinstance(finding, Mapping)
    ]
    for record in current_records:
        for field in _FINDING_FIELDS:
            raw_findings.extend(
                finding
                for finding in (record.get(field) or [])
                if isinstance(finding, Mapping)
            )
    normalized: dict[str, dict[str, Any]] = {}
    for finding in raw_findings:
        detail = dict(finding.get("detail") or {})
        step_id = str(detail.get("step_id") or "")
        identity = {
            "validator": str(finding.get("validator") or ""),
            "severity": str(finding.get("severity") or ""),
            "reason": str(
                detail.get("reason")
                or detail.get("issue_code")
                or detail.get("issue")
                or finding.get("message")
                or ""
            ),
            "step_id": step_id or None,
            "attempt_id": detail.get("attempt_id"),
            "checkpoint_id": detail.get("checkpoint_id"),
            "finding_detail_has_artifact_digest": any(
                "sha256" in str(key).lower() or "digest" in str(key).lower()
                for key in detail
            ),
            "finding_detail_has_reason_code": any(
                str(key).lower() in {"reason_code", "repair_reason_code", "issue_code"}
                for key in detail
            ),
            # The exact seal digest includes run-local evidence identifiers.
            # Characterize the required owner join, not a temp-path-dependent
            # digest; deterministic table/code bytes are locked separately.
            "owner_result_seal_bound": owner_seals.get(step_id) is not None,
        }
        normalized[_canonical_sha256(identity)] = _normalize(identity)
    return sorted(
        normalized.values(),
        key=lambda item: (
            str(item["step_id"]),
            item["validator"],
            item["severity"],
            item["reason"],
        ),
    )


def _stable_product_shas(
    *,
    run_dir: Path,
    current_records: list[Mapping[str, Any]],
) -> dict[str, str]:
    shas: dict[str, str] = {}
    for record in current_records:
        step_id = str(record.get("step_id") or "")
        if step_id not in _PLAN_STEP_IDS:
            continue
        summary = record.get("step_summary")
        output_files = (
            summary.get("output_files") if isinstance(summary, Mapping) else None
        )
        for product, filename in sorted((output_files or {}).items()):
            path = run_dir / "steps" / step_id / "outputs" / str(filename)
            if path.suffix.lower() not in {".csv", ".parquet", ".feather"}:
                continue
            shas[f"{step_id}:{product}"] = _stable_file_sha256(path)
    return shas


def _audit_log_authority_events(run_dir: Path) -> list[dict[str, Any]]:
    """Return explicit seal/register/alias events, if the audit log has any."""

    path = run_dir / "audit_log.jsonl"
    if not path.is_file():
        return []
    markers = re.compile(
        r"\b(?:seal(?:ed|ing)?|register(?:ed|ing)?|alias(?:es)?|authority)\b",
        flags=re.IGNORECASE,
    )
    events: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        event = str(payload.get("event") or "")
        detail = payload.get("detail") or {}
        detail_keys = " ".join(str(key) for key in detail)
        if markers.search(f"{event} {detail_keys}"):
            events.append(
                {
                    "event": event,
                    "phase": payload.get("phase"),
                    "step_id": payload.get("step_id"),
                }
            )
    return events


def _readiness_bundle(run_dir: Path) -> dict[str, Any]:
    payload = json.loads((run_dir / "run_status.json").read_text(encoding="utf-8"))
    gates = payload.get("gates") or {}
    selected_keys = (
        "execution_complete",
        "required_step_count",
        "completed_step_count",
        "missing_steps",
        "failed_steps",
        "evidence_complete",
        "numeric_verified",
        "analysis_validated",
        "manuscript_generated",
        "manuscript_ready",
        "publication_ready",
        "missing_evidence_count",
        "numeric_error_count",
        "evidence_error_count",
        "analysis_error_count",
        "analysis_errors",
        "blocked_outcome_step_ids",
        "blocked_outcome_not_leaked",
    )
    return {
        "status": payload.get("status"),
        "strict_fail_closed": payload.get("strict_fail_closed"),
        "gates": {key: gates.get(key) for key in selected_keys},
    }


def _build_bundle(*, run_dir: Path, observed_events: list[tuple[str, str]]):
    from easyicu.research_agent.authority.evidence_store import EvidenceStore
    from easyicu.research_agent.authority.runtime_artifacts import (
        current_step_records,
        load_run_artifact_authority,
    )

    authority = load_run_artifact_authority(run_dir)
    assert authority is not None
    ledger = list(authority["per_step_records"])
    partial = json.loads(
        (run_dir / "manifest_partial.json").read_text(encoding="utf-8")
    )
    attempt_history = list(partial["step_attempt_history"])
    assert all(record in attempt_history for record in partial["per_step_records"])
    current = [dict(record) for record in current_step_records(ledger)]
    current.sort(key=lambda record: str(record.get("step_id") or ""))
    plan = json.loads((run_dir / "analysis_plan.json").read_text(encoding="utf-8"))
    store = EvidenceStore(run_dir)
    current_evidence = store.current_verified_records(ledger)
    step_current_evidence = [
        record
        for record in current_evidence
        if str(record.produced_by_step or "") in _PLAN_STEP_IDS
    ]
    current_by_id = {record.evidence_id: record for record in step_current_evidence}
    current_ids = set(current_by_id)
    raw_aliases = {
        str(alias): str(evidence_id)
        for alias, evidence_id in store.aliases().items()
        if evidence_id in current_ids
    }
    # EvidenceStore publishes every record id as a compatibility self-alias.
    # Those ids are content-derived (for example analyzer prose is allowed to
    # vary while its semantic owner stays fixed), so hashing the raw self-alias
    # names makes this characterization oracle environment-sensitive.  The
    # current-evidence bundle above already locks every selected record.  Keep
    # the self-alias invariant as a count and hash only user/product semantic
    # aliases here.
    aliases = {
        alias: {
            "kind": current_by_id[evidence_id].kind,
            "produced_by_step": current_by_id[evidence_id].produced_by_step,
            "description": current_by_id[evidence_id].description,
            "stable_content_sha256": (
                _stable_file_sha256(
                    _evidence_path(
                        run_dir,
                        current_by_id[evidence_id].relative_path,
                    )
                )
                if current_by_id[evidence_id].kind == "table"
                else current_by_id[evidence_id].sha256
                if current_by_id[evidence_id].kind == "code"
                else None
            ),
        }
        for alias, evidence_id in sorted(raw_aliases.items())
        if alias != evidence_id
    }
    claims = store.authoritative_numeric_claims(ledger)
    evidence_authority = sorted(
        [
            {
                "kind": record.kind,
                "produced_by_step": record.produced_by_step,
                "description": record.description,
                "producer": record.producer,
                "generation_mode": record.generation_mode,
                "stable_content_sha256": (
                    _stable_file_sha256(
                        _evidence_path(run_dir, record.relative_path)
                    )
                    if record.kind == "table"
                    else record.sha256
                    if record.kind == "code"
                    else None
                ),
            }
            for record in step_current_evidence
        ],
        key=lambda item: (
            item["produced_by_step"],
            item["kind"],
            item["description"],
        ),
    )
    claim_authority = sorted(
        [
            {
                "evidence_id": claim.evidence_id,
                "step_id": claim.step_id,
                "source_field": claim.source_field,
                "value": claim.value,
                "canonical": claim.canonical,
            }
            for claim in claims
            if claim.step_id in _PLAN_STEP_IDS
        ],
        key=lambda item: (
            item["step_id"],
            item["evidence_id"],
            item["source_field"],
        ),
    )
    finding_authority = _finding_bundle(
        manifest=authority,
        current_records=current,
    )
    event_order = {
        step_id: _run_length_encode_events(
            [event for owner, event in observed_events if owner == step_id]
        )
        for step_id in sorted(_PLAN_STEP_IDS)
    }
    return _normalize(
        {
            "schema": "easyicu.freeze_char_golden/3",
            "volatile_field_allowlist": sorted(_VOLATILE_FIELD_ALLOWLIST),
            "step_statuses": [
                {
                    "step_id": record.get("step_id"),
                    "status": record.get("status"),
                }
                for record in current
            ],
            "dependency_edges": _declared_dependency_edges(plan),
            "dependency_propagation_set": sorted(
                str(record.get("step_id"))
                for record in current
                if str(record.get("status") or "") != "ok"
            ),
            "typed_input_bindings": _typed_binding_bundle(run_dir),
            # The full normalized mappings/sets are hashed rather than
            # truncated.  Any member-level drift changes the golden while the
            # fixture remains compact enough to review in source control.
            "current_evidence": {
                "count": len(evidence_authority),
                "mapping_sha256": _canonical_sha256(evidence_authority),
            },
            "current_aliases": {
                "count": len(aliases),
                "mapping_sha256": _canonical_sha256(aliases),
            },
            "current_self_aliases": {
                "count": sum(
                    alias == evidence_id for alias, evidence_id in raw_aliases.items()
                ),
            },
            "authoritative_numeric_claims": {
                "count": len(claim_authority),
                "set_sha256": _canonical_sha256(claim_authority),
            },
            "validator_findings": {
                "count": len(finding_authority),
                "set_sha256": _canonical_sha256(finding_authority),
                "by_severity": {
                    severity: sum(
                        finding["severity"] == severity for finding in finding_authority
                    )
                    for severity in ("error", "warning", "info")
                },
            },
            "finding_observability": {
                "universal_reason_code": bool(finding_authority)
                and all(
                    finding["finding_detail_has_reason_code"]
                    for finding in finding_authority
                ),
                "universal_artifact_digest_binding": bool(finding_authority)
                and all(
                    finding["finding_detail_has_artifact_digest"]
                    for finding in finding_authority
                ),
                "owner_result_seal_join_used": any(
                    finding["owner_result_seal_bound"]
                    and not finding["finding_detail_has_artifact_digest"]
                    for finding in finding_authority
                ),
            },
            "authority_transition_order": event_order,
            "audit_log_authority_events": _audit_log_authority_events(run_dir),
            "readiness": _readiness_bundle(run_dir),
            "deterministic_product_sha256": _stable_product_shas(
                run_dir=run_dir,
                current_records=current,
            ),
        }
    )


def test_normalizer_removes_only_explicitly_allowed_volatile_fields():
    payload = {
        "timestamp": "volatile",
        "run_id": "volatile",
        "absolute_path": "/volatile/path",
        "duration_seconds": 1.5,
        "sha256": "a" * 64,
        "reason_code": "must_survive",
        "nested": {"pid": 99, "status": "ok"},
    }

    assert _normalize(payload) == {
        "nested": {"status": "ok"},
        "reason_code": "must_survive",
        "sha256": "a" * 64,
    }


def test_table_semantic_digest_ignores_float_tail_but_detects_numeric_drift(
    tmp_path: Path,
):
    baseline = tmp_path / "baseline.csv"
    float_tail = tmp_path / "float_tail.csv"
    material_drift = tmp_path / "material_drift.csv"
    baseline.write_text("id,value\n1,0.1234567890123\n", encoding="utf-8")
    float_tail.write_text("id,value\n1,0.1234567890124\n", encoding="utf-8")
    material_drift.write_text("id,value\n1,0.1234667890123\n", encoding="utf-8")

    assert _stable_file_sha256(baseline) == _stable_file_sha256(float_tail)
    assert _stable_file_sha256(baseline) != _stable_file_sha256(material_drift)


def test_minimal_typed_pipeline_matches_normalized_golden_bundle(
    ra,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    fixture = _load_typed_pipeline_fixture()
    fixture._disable_unrelated_audits(monkeypatch)
    observed_events = _install_authority_order_observer(monkeypatch)
    llm = fixture._PlanAndCoderLLM()
    runners_by_timeout: dict[float, object] = {}

    def runner_factory(*, workdir, timeout_seconds, **_kwargs):
        timeout = float(timeout_seconds)
        runner = runners_by_timeout.get(timeout)
        if runner is None:
            runner = fixture._HybridTrajectoryRunner(workdir=Path(workdir))
            runners_by_timeout[timeout] = runner
        return runner

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
        cohort_name="trajectory_stability_success",
        database="synthetic",
        target_outcome="death",
        stop_after_step_id="04_characterization",
        stop_after_analysis=True,
    )

    actual = _build_bundle(
        run_dir=Path(result.workdir),
        observed_events=observed_events,
    )
    expected = json.loads(_GOLDEN_PATH.read_text(encoding="utf-8"))

    assert actual == expected, json.dumps(actual, indent=2, sort_keys=True)


def test_numeric_authority_failure_prevents_current_alias_publication(
    ra,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """A broken numeric ledger must fail the step before aliases become current."""

    from easyicu.research_agent.authority.evidence_store import EvidenceStore
    from easyicu.research_agent.authority.runtime_artifacts import (
        current_step_records,
        load_run_artifact_authority,
    )

    fixture = _load_typed_pipeline_fixture()
    fixture._disable_unrelated_audits(monkeypatch)
    numeric_calls = 0
    original_numeric_registration = EvidenceStore.register_step_summary_numerics

    def fail_numeric_registration(self, *args, **kwargs):
        nonlocal numeric_calls
        if str(kwargs.get("step_id") or "") != "01_representation":
            return original_numeric_registration(self, *args, **kwargs)
        numeric_calls += 1
        raise OSError("injected numeric authority failure")

    monkeypatch.setattr(
        EvidenceStore,
        "register_step_summary_numerics",
        fail_numeric_registration,
    )
    runners_by_timeout: dict[float, object] = {}

    def runner_factory(*, workdir, timeout_seconds, **_kwargs):
        timeout = float(timeout_seconds)
        runner = runners_by_timeout.get(timeout)
        if runner is None:
            runner = fixture._HybridTrajectoryRunner(workdir=Path(workdir))
            runners_by_timeout[timeout] = runner
        return runner

    pipeline = ra.ResearchAgentPipeline(
        workdir=tmp_path,
        llm=fixture._PlanAndCoderLLM(),
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
        cohort_name="trajectory_stability_failure",
        database="synthetic",
        target_outcome="death",
        stop_after_step_id="01_representation",
        stop_after_analysis=True,
    )

    run_dir = Path(result.workdir)
    authority = load_run_artifact_authority(run_dir)
    assert authority is not None
    current = current_step_records(authority["per_step_records"])
    failed = next(
        record for record in current if record.get("step_id") == "01_representation"
    )
    assert failed["status"] == "contract_failed"
    evidence_findings = [
        finding
        for finding in failed.get("contract_findings", [])
        if finding.get("validator") == "result_evidence_authority"
    ]
    assert len(evidence_findings) == 1
    assert evidence_findings[0]["detail"]["evidence_store_write_suppressed"] is True

    store = EvidenceStore(run_dir)
    unpublished_result_ids = set(failed.get("evidence_ids", [])) - {
        str(failed.get("script_evidence_id") or "")
    }
    assert unpublished_result_ids
    assert not unpublished_result_ids.intersection(store.aliases().values())
    assert numeric_calls == 1
