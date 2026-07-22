"""Reusable zero-Provider graph-level preflight harness for E1-E3.

This module drives the real :class:`ResearchAgentPipeline` graph fully offline
and returns a structured :class:`PreflightRun` summary.  It never calls an
external Provider, never reads patient data, and never grants paper authority
(see the package docstring for the enforced boundaries).

The zero-Provider guarantee is **not** taken on the client's word.  Every run is
wrapped in :func:`provider_transport_spy`, which replaces the real lowest-layer
HTTP transport (``httpx.Client.send`` / ``httpx.AsyncClient.send`` — the path all
production OpenAI/Anthropic SDK calls funnel through) with a counter that raises
on first use.  ``PreflightRun.external_provider_calls`` is that spy's count, so
``== 0`` is authoritative and independent of the (forgeable)
``__easyicu_mock_client__`` / class-name markers.  ``llm_is_mockish`` is recorded
only as *descriptive* colour, never as the security boundary.

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
    E1E3_CASES,
    PreflightCase,
)

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
# Zero-Provider transport spy (authoritative)
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
    counter stays 0 — that zero is the authoritative zero-Provider evidence.

    Only ``send`` is patched (not client construction), so incidental
    construction of an ``httpx.Client`` that never sends is untouched; the moment
    a request would leave the process it is blocked.
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

    * The planner request returns the fixture's typed ``AnalysisPlan`` (so the
      deterministic grouped-Table-1 executor is eligible).
    * The primary-association code request returns the mock's battle-tested
      logistic-regression script *bound to the fixture's exact exposure* (so the
      agent-owned primary does not depend on the mock's role-inference heuristic).
    * Optional fault injection returns runtime-raising code for a target step on
      both its initial write and every repair, to exercise the repair cap.

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
        # code write/repair calls keyed by resolved step_id
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
                # Reuse the mock's tested primary script, bound to the exact
                # fixture exposure so the agent-owned primary is deterministic.
                return _mock_code_primary_association(
                    ctx=self.context,
                    step_id=step_id,
                    outcome=self._case.target_outcome,
                    predictor=self._case.primary_exposure,
                )
        return super().complete(messages, **kwargs)


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

    # -- derived views ----------------------------------------------------
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
        """Authoritative: the real httpx transport spy count, not a self-report."""

        return self.provider_transport_calls

    @property
    def llm_offline_classified(self) -> bool:
        """Descriptive only — ``llm_is_mockish`` is forgeable (package docstring)."""

        return llm_is_mockish(self.llm)


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
    """Construct an offline pipeline.

    ``runner_kind='subprocess'`` runs the real host runner (the documented
    offline-diagnosis path).  The ``auto`` runner's Docker source-SHA integrity
    gate is a *production* blocker and is deliberately NOT bypassed or weakened;
    the subprocess runner is only for offline diagnosis.  Tangential
    Provider-shaped audits (literature, visual QA, LaTeX, LLM concept audit) are
    disabled so the run exercises the orchestration under test, not unrelated
    quality validators.
    """

    kwargs: Dict[str, Any] = dict(
        workdir=str(workdir),
        llm=llm,
        runner_kind="subprocess",
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
    """Run one offline graph-level preflight and return structured evidence.

    The whole pipeline run executes inside :func:`provider_transport_spy`, so the
    returned ``provider_transport_calls`` is a real transport-layer measurement.
    """

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
    return PreflightRun(
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
    )


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
    We write a minimal one-item results doc that references the run and evaluate
    it through the real production gate — nothing here grants paper authority.
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
    manifest = preflight_runtime_manifest(tmp)
    run = run_preflight(case, workdir=tmp)
    print(
        json.dumps(
            {
                "task_id": run.case.task_id,
                "runtime_integration_ready": manifest.integration_ready,
                "runtime_blocked_reason": manifest.blocked_reason,
                "llm_offline_classified": run.llm_offline_classified,
                "external_provider_calls": run.external_provider_calls,
                "provider_transport_targets": run.provider_transport_targets,
                "total_llm_calls": run.llm.total_calls,
                "plan_calls": run.llm.plan_calls,
                "expected_products": list(run.case.expected_products),
                "step_ids": run.step_ids,
                "routing": run.routing,
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
    "ProviderTransportBlocked",
    "TransportSpy",
    "provider_transport_spy",
    "ScriptedPreflightLLM",
    "PreflightRun",
    "run_preflight",
    "build_pipeline",
    "preflight_runtime_manifest",
    "load_manifest",
    "paper_acceptance_status",
]
