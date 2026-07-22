"""Reusable zero-Provider graph-level **partial-flow smoke** harness for E1-E3.

This drives the real :class:`ResearchAgentPipeline` graph fully offline and
returns a structured :class:`PreflightRun`.  It is a *partial-flow smoke*, NOT
E1/E2/E3 publication readiness: offline it genuinely produces only the
deterministic Table 1 artifact; the sealed publication figures (and the
data-dependent audit products) are not produced offline.  Each formal suite
``expected_output`` is mapped item-by-item to its plan step and honest
fulfillment level (see :mod:`.fixtures` ``ProductMapping``), and the mapping is
verified against the real run manifest — never by "the three lists differ".

Enforced boundaries:

* **Zero external Provider (measured).** Every run executes inside
  :func:`provider_transport_spy`, which replaces ``httpx.Client.send`` /
  ``httpx.AsyncClient.send`` (the parent-process transport under the OpenAI /
  Anthropic SDKs) with a fail-on-call counter.  ``external_provider_calls == 0``
  is that measured count.  This is the *parent-process* Provider measurement;
  the **subprocess/CLI** no-network boundary is a separate guarantee, proven by
  the runner's pinned ``network_policy="none"`` + ``allow_unsafe_host_fallback
  =False`` (P1-C) recorded per step as ``requested_network_policy`` /
  ``isolation_degraded``.  The forgeable ``llm_is_mockish`` marker is descriptive
  colour only.
* **Isolation fail-closed (wired).** ``run_preflight`` builds and persists a
  :class:`RuntimeManifest`; when ``integration_ready`` is false (e.g. a nested
  macOS sandbox) it returns a unique structured *blocked* outcome and **does not
  start the pipeline**.  A per-step nested-sandbox denial is converted to the
  same structured reason via ``step_isolation_unavailable`` — never left as a
  generic ``repair_failed``.
* **No paper authority.** Diagnostic-only; the production Figure 2 acceptance
  gate rejects every run (asserted).

Run one case directly (development smoke)::

    PYTHONPATH="src:." python -m benchmarks.figure2_canonical9.preflight.harness e2
"""

from __future__ import annotations

import contextlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional


import easyicu.research_agent as ra
from easyicu.research_agent.evaluation_scorecard import compute_tristate
from easyicu.research_agent.providers.llm import llm_is_mockish
from easyicu.research_agent.providers.mocks import (
    _extract_step_id,
    _mock_code_primary_association,
)

from benchmarks.figure2_canonical9.evaluator.acceptance import (
    evaluate_figure2_paper_acceptance,
)
from benchmarks.figure2_canonical9.preflight import runtime as rt
from benchmarks.figure2_canonical9.preflight.fixtures import (
    FULFILLMENT_NOT_PRODUCED_OFFLINE,
    FULFILLMENT_PLANNED_ONLY,
    FULFILLMENT_PRODUCED,
    E1E3_CASES,
    PreflightCase,
    ProductMapping,
)

# P1-C: the preflight PINS a fail-closed no-network runner policy.  Passing an
# explicit ``allow_unsafe_host_fallback=False`` makes the runner ignore the
# ``EASYICU_ALLOW_UNSAFE_HOST_FALLBACK`` env var (runner reads the env only when
# the kwarg is None), so no environment variable can relax the boundary.
PREFLIGHT_RUNNER_KWARGS: Dict[str, Any] = {
    "network_policy": "none",
    "allow_unsafe_host_fallback": False,
}

# This harness is a partial-flow smoke, not readiness.
READINESS_CLASS = "partial_flow_smoke"

_PLAN_ANCHORS = (
    "ICU-AWARE RESEARCH PLAN",
    "RESEARCH PLAN AS JSON",
    "ANALYSISPLAN SCHEMA",
)
_REPLAN_ANCHOR = "REVISE THE ICU-AWARE RESEARCH PLAN"
_CODE_ANCHORS = ("WRITE THE PYTHON CODE", "REPAIR THE PYTHON CODE")

_FAULT_CODE = (
    "# injected preflight coder fault (deterministic, offline)\n"
    "raise RuntimeError('injected preflight coder fault')\n"
)


# ---------------------------------------------------------------------------
# Zero-Provider transport spy (authoritative, parent-process)
# ---------------------------------------------------------------------------


class ProviderTransportBlocked(RuntimeError):
    """Raised if any real network Provider transport is invoked under the spy."""


@dataclass
class TransportSpy:
    """Records attempts to reach the real HTTP transport (must stay at 0)."""

    calls: int = 0
    targets: List[str] = field(default_factory=list)


@contextlib.contextmanager
def provider_transport_spy() -> Iterator[TransportSpy]:
    """Fail-closed spy over the real lowest-layer HTTP transport.

    Every production LLM Provider (OpenAI/Anthropic SDK) sends requests through
    ``httpx``.  We swap ``httpx.Client.send`` / ``httpx.AsyncClient.send`` for a
    counter that RAISES on first use, so any real network Provider call is both
    counted and hard-failed.  A mock client never touches ``httpx``, so the
    counter stays 0 — the authoritative *parent-process* zero-Provider evidence.
    """

    spy = TransportSpy()
    import httpx

    real_sync = httpx.Client.send
    real_async = httpx.AsyncClient.send

    def _blocked_sync(self, *args, **kwargs):  # noqa: ANN001
        spy.calls += 1
        spy.targets.append("httpx.Client.send")
        raise ProviderTransportBlocked(
            "preflight is zero-Provider: httpx.Client.send was invoked"
        )

    async def _blocked_async(self, *args, **kwargs):  # noqa: ANN001
        spy.calls += 1
        spy.targets.append("httpx.AsyncClient.send")
        raise ProviderTransportBlocked(
            "preflight is zero-Provider: httpx.AsyncClient.send was invoked"
        )

    httpx.Client.send = _blocked_sync  # type: ignore[method-assign]
    httpx.AsyncClient.send = _blocked_async  # type: ignore[method-assign]
    try:
        yield spy
    finally:
        httpx.Client.send = real_sync  # type: ignore[method-assign]
        httpx.AsyncClient.send = real_async  # type: ignore[method-assign]


def _last_user(messages) -> str:
    return next(
        (m.content for m in reversed(messages) if m.role == "user"),
        "",
    )


class ScriptedPreflightLLM(ra.MockLLMClient):
    """Deterministic offline client: inject a typed plan + a correct primary.

    Every response is produced locally.  This client's offline nature is recorded
    descriptively via ``llm_is_mockish``; the *authoritative* zero-Provider proof
    is the transport spy in :func:`provider_transport_spy`, not this class.
    """

    name = "scripted-preflight-offline"

    def __init__(
        self,
        case: PreflightCase,
        *,
        fault_step: Optional[str] = None,
    ) -> None:
        super().__init__()
        self._case = case
        self._plan_json = case.build_plan().model_dump_json(indent=2)
        self._fault_step = fault_step
        self.total_calls = 0
        self.plan_calls = 0
        self.code_calls: Dict[str, int] = {}

    def complete(self, messages, **kwargs) -> str:
        self.total_calls += 1
        user = _last_user(messages)
        upper = user.upper()

        is_plan = any(a in upper for a in _PLAN_ANCHORS) and _REPLAN_ANCHOR not in upper
        if is_plan:
            self.plan_calls += 1
            return self._plan_json

        is_code = any(a in upper for a in _CODE_ANCHORS)
        if is_code:
            step_id = _extract_step_id(user) or "step"
            self.code_calls[step_id] = self.code_calls.get(step_id, 0) + 1
            if self._fault_step is not None and step_id == self._fault_step:
                return _FAULT_CODE
            if step_id == self._case.primary_step_id and self.context is not None:
                return _mock_code_primary_association(
                    ctx=self.context,
                    step_id=step_id,
                    outcome=self._case.target_outcome,
                    predictor=self._case.primary_exposure,
                )
        return super().complete(messages, **kwargs)


# ---------------------------------------------------------------------------
# Isolation wiring (P1-B) + product-map / network-policy resolution (P1-A/P1-C)
# ---------------------------------------------------------------------------


def blocking_step_isolation(
    manifest: Dict[str, Any], capability: rt.IsolationCapability
) -> Optional[str]:
    """Convert an actual per-step nested-sandbox denial into the structured block.

    Scans the real per-step records; if any is a nested-sandbox denial (per
    :func:`runtime.step_isolation_unavailable`, which requires the
    ``macos_sandbox_exec`` backend, an unavailable probe, and the persisted
    ``sandbox_apply`` stderr), returns the same ``isolation_backend_unavailable``
    reason so it is never left as a generic ``repair_failed``.
    """

    for rec in manifest.get("per_step_records", []):
        detail = rt.step_isolation_unavailable(rec, capability)
        if detail:
            return (
                "isolation_backend_unavailable: "
                f"{capability.backend} step {rec.get('step_id')} ({detail})"
            )
    return None


def observe_product_fulfillment(mapping: ProductMapping, run: "PreflightRun") -> str:
    """Read the REAL manifest to observe how a mapped suite output was fulfilled."""

    if mapping.step_id is None:
        return FULFILLMENT_NOT_PRODUCED_OFFLINE
    rec = run.record(mapping.step_id)
    if not rec:
        return "absent"
    ok = rec.get("status") == "ok"
    prefix = mapping.artifact_evidence_prefix
    has_artifact = bool(prefix) and any(
        str(e).startswith(prefix) for e in (rec.get("evidence_ids") or [])
    )
    if ok and has_artifact:
        return FULFILLMENT_PRODUCED
    return FULFILLMENT_PLANNED_ONLY


def resolve_product_map(run: "PreflightRun") -> List[Dict[str, Any]]:
    """Per-item resolution of the formal suite output contract against the run."""

    outputs = run.case.expected_products
    resolved: List[Dict[str, Any]] = []
    for mapping in run.case.product_map:
        suite_output = (
            outputs[mapping.output_index]
            if 0 <= mapping.output_index < len(outputs)
            else None
        )
        observed = observe_product_fulfillment(mapping, run)
        resolved.append(
            {
                "output_index": mapping.output_index,
                "suite_output": suite_output,
                "step_id": mapping.step_id,
                "artifact_evidence_prefix": mapping.artifact_evidence_prefix,
                "declared_fulfillment": mapping.declared_fulfillment,
                "observed_fulfillment": observed,
                "matches": observed == mapping.declared_fulfillment,
            }
        )
    return resolved


def network_policy_report(manifest: Dict[str, Any]) -> Dict[str, Any]:
    """P1-C: every subprocess-executing step must be no-network, non-degraded."""

    steps: List[Dict[str, Any]] = []
    for rec in manifest.get("per_step_records", []):
        pol = rec.get("requested_network_policy")
        if pol is None:
            continue  # this step did not spawn a subprocess runner
        steps.append(
            {
                "step_id": rec.get("step_id"),
                "requested_network_policy": pol,
                "effective_isolation": rec.get("effective_isolation"),
                "isolation_degraded": rec.get("isolation_degraded"),
            }
        )
    ok = bool(steps) and all(
        s["requested_network_policy"] == "none" and s["isolation_degraded"] is False
        for s in steps
    )
    return {"ok": ok, "subprocess_steps": steps}


@dataclass
class PreflightRun:
    """Structured graph-level evidence from one offline preflight run."""

    case: PreflightCase
    run_dir: Path
    run_id: str
    manifest: Dict[str, Any]
    llm: ScriptedPreflightLLM
    provider_transport_calls: int = 0
    provider_transport_targets: List[str] = field(default_factory=list)
    raised: Optional[str] = None
    routing: List[Dict[str, Any]] = field(default_factory=list)
    readiness: Dict[str, Any] = field(default_factory=dict)
    runtime_manifest: Optional[rt.RuntimeManifest] = None
    blocked_reason: Optional[str] = None
    pipeline_ran: bool = False

    # -- derived views ----------------------------------------------------
    @property
    def readiness_class(self) -> str:
        return READINESS_CLASS

    @property
    def step_ids(self) -> List[str]:
        return [r.get("step_id") for r in self.manifest.get("per_step_records", [])]

    def record(self, step_id: str) -> Dict[str, Any]:
        for r in self.manifest.get("per_step_records", []):
            if r.get("step_id") == step_id:
                return r
        return {}

    @property
    def tristate(self) -> str:
        return compute_tristate(self.readiness)

    @property
    def external_provider_calls(self) -> int:
        """Authoritative parent-process Provider count (the real httpx spy)."""

        return self.provider_transport_calls

    @property
    def llm_offline_classified(self) -> bool:
        """Descriptive only — ``llm_is_mockish`` is forgeable (package docstring)."""

        return llm_is_mockish(self.llm)

    # -- P1-A product-map -------------------------------------------------
    def resolved_product_map(self) -> List[Dict[str, Any]]:
        return resolve_product_map(self)

    def produced_suite_outputs(self) -> List[str]:
        return [
            row["suite_output"]
            for row in self.resolved_product_map()
            if row["observed_fulfillment"] == FULFILLMENT_PRODUCED
        ]

    # -- P1-C network policy ---------------------------------------------
    def network_policy_report(self) -> Dict[str, Any]:
        return network_policy_report(self.manifest)


def load_manifest(run_dir: Path) -> Dict[str, Any]:
    """Prefer the final manifest; fall back to the partial (stopped) manifest."""

    for name in ("manifest.json", "manifest_partial.json"):
        p = run_dir / name
        if p.is_file():
            return json.loads(p.read_text(encoding="utf-8"))
    return {}


def _routing(manifest: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [
        {
            "step_id": r.get("step_id"),
            "deterministic_standard_analysis": r.get("deterministic_standard_analysis"),
            "generation_mode": r.get("generation_mode"),
            "status": r.get("status"),
        }
        for r in manifest.get("per_step_records", [])
    ]


def build_pipeline(
    case: PreflightCase,
    *,
    workdir: Path,
    llm: ScriptedPreflightLLM,
    timeout_seconds: float = 60.0,
    standard_executor_timeout_seconds: float = 900.0,
    max_code_repair_attempts: int = 2,
    enable_replanning: bool = False,
    max_replans: Optional[int] = None,
) -> ra.ResearchAgentPipeline:
    """Construct an offline pipeline with a PINNED fail-closed no-network runner.

    ``runner_kind='subprocess'`` runs the real host runner (the documented
    offline-diagnosis path); the ``auto`` runner's Docker source-SHA integrity
    gate is a production blocker and is NOT bypassed.  ``runner_kwargs`` pins
    ``network_policy='none'`` and ``allow_unsafe_host_fallback=False`` so no
    environment variable can relax the subprocess no-network boundary (P1-C).
    Tangential Provider-shaped audits are disabled so the run exercises the
    orchestration under test.
    """

    kwargs: Dict[str, Any] = dict(
        workdir=str(workdir),
        llm=llm,
        runner_kind="subprocess",
        runner_kwargs=dict(PREFLIGHT_RUNNER_KWARGS),
        timeout_seconds=timeout_seconds,
        standard_executor_timeout_seconds=standard_executor_timeout_seconds,
        max_code_repair_attempts=max_code_repair_attempts,
        enable_literature=False,
        enable_visual_qa=False,
        enable_latex=False,
        enable_llm_concept_audit=False,
        enable_replanning=enable_replanning,
    )
    if max_replans is not None:
        kwargs["max_replans"] = max_replans
    return ra.ResearchAgentPipeline(**kwargs)


def _blocked_run(
    case: PreflightCase,
    *,
    workdir: Path,
    runtime: rt.RuntimeManifest,
    fault_step: Optional[str],
) -> "PreflightRun":
    """Unique structured blocked outcome — the pipeline is NEVER started."""

    return PreflightRun(
        case=case,
        run_dir=Path(workdir),
        run_id="",
        manifest={},
        llm=ScriptedPreflightLLM(case, fault_step=fault_step),
        runtime_manifest=runtime,
        blocked_reason=runtime.blocked_reason or "integration_not_ready",
        pipeline_ran=False,
    )


def run_preflight(
    case: PreflightCase,
    *,
    workdir: Path,
    n_rows: int = 80,
    fault_step: Optional[str] = None,
    stop_after_step_id: Optional[str] = None,
    resume_run_id: Optional[str] = None,
    resume_from_step_id: Optional[str] = None,
    timeout_seconds: float = 60.0,
    standard_executor_timeout_seconds: float = 900.0,
    max_code_repair_attempts: int = 2,
    enable_replanning: bool = False,
    max_replans: Optional[int] = None,
) -> PreflightRun:
    """Run one offline partial-flow smoke and return structured evidence.

    Fail-closed: the RuntimeManifest is built and persisted first; if the
    isolation backend is unavailable (``integration_ready`` false) the pipeline
    is NOT started and a structured blocked outcome is returned.
    """

    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    runtime = rt.build_runtime_manifest()
    rt.write_runtime_manifest(workdir, runtime)

    if not runtime.integration_ready:
        return _blocked_run(
            case, workdir=workdir, runtime=runtime, fault_step=fault_step
        )

    llm = ScriptedPreflightLLM(case, fault_step=fault_step)
    pipeline = build_pipeline(
        case,
        workdir=workdir,
        llm=llm,
        timeout_seconds=timeout_seconds,
        standard_executor_timeout_seconds=standard_executor_timeout_seconds,
        max_code_repair_attempts=max_code_repair_attempts,
        enable_replanning=enable_replanning,
        max_replans=max_replans,
    )
    run_kwargs: Dict[str, Any] = dict(
        question=case.question,
        cohort=case.build_cohort(n_rows),
        target_outcome=case.target_outcome,
        primary_exposure=case.primary_exposure,
        database=case.database,
        cohort_name=f"{case.task_id}_preflight",
        concept_descriptions=dict(case.concept_descriptions),
    )
    if stop_after_step_id is not None:
        run_kwargs["stop_after_step_id"] = stop_after_step_id
        run_kwargs["stop_after_analysis"] = True
    if resume_run_id is not None:
        run_kwargs["resume_run_id"] = resume_run_id
    if resume_from_step_id is not None:
        run_kwargs["resume_from_step_id"] = resume_from_step_id

    raised: Optional[str] = None
    run_dir = Path(workdir)
    run_id = ""
    with provider_transport_spy() as spy:
        try:
            result = pipeline.run(**run_kwargs)
            run_dir = Path(result.workdir)
            run_id = result.run_id
        except Exception as exc:  # noqa: BLE001 - a raise IS a graph-level outcome
            raised = f"{type(exc).__name__}: {exc}"

    manifest = load_manifest(run_dir)
    rt.write_runtime_manifest(run_dir, runtime)
    # Post-run: convert any per-step nested-sandbox denial into the same block.
    step_block = blocking_step_isolation(manifest, runtime.isolation)

    run = PreflightRun(
        case=case,
        run_dir=run_dir,
        run_id=run_id,
        manifest=manifest,
        llm=llm,
        provider_transport_calls=spy.calls,
        provider_transport_targets=list(spy.targets),
        raised=raised,
        routing=_routing(manifest),
        readiness=manifest.get("readiness", {}),
        runtime_manifest=runtime,
        blocked_reason=step_block,
        pipeline_ran=True,
    )
    _write_product_map_artifact(run)
    return run


def _write_product_map_artifact(run: PreflightRun) -> Path:
    """Persist the resolved per-item product map as a verifiable diagnostic file."""

    path = Path(run.run_dir) / "preflight_product_map.json"
    payload = {
        "task_id": run.case.task_id,
        "readiness_class": run.readiness_class,
        "expected_products": list(run.case.expected_products),
        "product_map": run.resolved_product_map(),
        "produced_suite_outputs": run.produced_suite_outputs(),
        "network_policy": run.network_policy_report(),
    }
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path


def preflight_runtime_manifest(run_dir: Optional[Path] = None) -> rt.RuntimeManifest:
    """Capture the auditable runtime identity + isolation-backend capability.

    Persisted to ``run_dir`` when given.  The integration gate reads
    ``manifest.integration_ready`` (runtime-compatible AND isolation available);
    a nested-sandbox host reports a structured ``blocked_reason`` instead.
    """

    manifest = rt.build_runtime_manifest()
    if run_dir is not None:
        rt.write_runtime_manifest(run_dir, manifest)
    return manifest


def paper_acceptance_status(run: PreflightRun) -> str:
    """Status of the production Figure 2 acceptance gate for a mock run.

    A single diagnostic-only mock run can never satisfy the exact 9-task,
    aware-arm, replay-verified acceptance contract, so this returns ``invalid``.
    """

    results_doc = {
        "items": [run.case.task_id],
        "arms": ["aware"],
        "pending": [],
        "scores": [
            {
                "item_key": run.case.task_id,
                "aware": {
                    "arm": "aware",
                    "workdir": str(run.run_dir),
                    "run_id": run.run_id or "preflight",
                },
            }
        ],
    }
    payload = json.dumps(results_doc, ensure_ascii=False).encode("utf-8")
    results_path = run.run_dir / "preflight_mock_results.json"
    results_path.write_bytes(payload)
    verdict = evaluate_figure2_paper_acceptance(results_path)
    return verdict.status


def _cli(task_key: str) -> int:
    import tempfile

    case = E1E3_CASES.get(task_key)
    if case is None:
        case = next(
            (c for c in E1E3_CASES.values() if c.task_id.startswith(task_key)),
            None,
        )
    if case is None:
        raise SystemExit(f"unknown task key {task_key!r}; try e1/e2/e3")
    tmp = Path(tempfile.mkdtemp(prefix=f"preflight_{task_key}_"))
    run = run_preflight(case, workdir=tmp)

    # Structured blocked outcome: the pipeline was not started.
    if not run.pipeline_ran:
        print(
            json.dumps(
                {
                    "task_id": run.case.task_id,
                    "readiness_class": run.readiness_class,
                    "blocked": True,
                    "pipeline_ran": False,
                    "blocked_reason": run.blocked_reason,
                    "runtime_integration_ready": False,
                },
                indent=2,
                default=str,
            )
        )
        return 2

    print(
        json.dumps(
            {
                "task_id": run.case.task_id,
                "readiness_class": run.readiness_class,
                "pipeline_ran": True,
                "blocked_reason": run.blocked_reason,
                "runtime_integration_ready": True,
                "llm_offline_classified": run.llm_offline_classified,
                "external_provider_calls": run.external_provider_calls,
                "provider_transport_targets": run.provider_transport_targets,
                "network_policy": run.network_policy_report(),
                "expected_products": list(run.case.expected_products),
                "product_map": run.resolved_product_map(),
                "produced_suite_outputs": run.produced_suite_outputs(),
                "step_ids": run.step_ids,
                "tristate": run.tristate,
                "raised": run.raised,
                "paper_acceptance": paper_acceptance_status(run),
            },
            indent=2,
            default=str,
        )
    )
    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(_cli(sys.argv[1] if len(sys.argv) > 1 else "e2"))


__all__ = [
    "PREFLIGHT_RUNNER_KWARGS",
    "READINESS_CLASS",
    "ProviderTransportBlocked",
    "TransportSpy",
    "provider_transport_spy",
    "ScriptedPreflightLLM",
    "PreflightRun",
    "run_preflight",
    "build_pipeline",
    "preflight_runtime_manifest",
    "blocking_step_isolation",
    "observe_product_fulfillment",
    "resolve_product_map",
    "network_policy_report",
    "load_manifest",
    "paper_acceptance_status",
]
