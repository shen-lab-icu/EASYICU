"""Reusable zero-Provider partial-flow smoke harness for E1-E3.

This module drives the real :class:`ResearchAgentPipeline` graph fully offline
and returns a structured :class:`PreflightRun` summary. It never calls an
external Provider, never reads patient data, and never grants paper authority.
It is deliberately a *partial-flow smoke*, not E1/E2/E3 readiness: formal
outputs are resolved against the run manifest, and only the deterministic Table
1 artifact is actually produced offline.

The zero-Provider guarantee is **not** taken on the client's word.  Every run is
wrapped in :func:`provider_transport_spy`, which replaces the real lowest-layer
HTTP transport (``httpx.Client.send`` / ``httpx.AsyncClient.send`` — the path all
production OpenAI/Anthropic SDK calls funnel through) with a counter that raises
on first use. ``PreflightRun.external_provider_calls`` is that spy's count, so
``== 0`` is authoritative for the parent process and independent of the (forgeable)
``__easyicu_mock_client__`` / class-name markers.  ``llm_is_mockish`` is recorded
only as *descriptive* colour, never as the security boundary. Generated-code
subprocesses have their own explicit ``network_policy="none"`` and
``allow_unsafe_host_fallback=False`` boundary, verified from step records.

Run one case directly (development smoke)::

    PYTHONPATH="src:." python -m benchmarks.figure2_canonical9.preflight.harness e2
"""

from __future__ import annotations

import contextlib
import json
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional


import easyicu.research_agent as ra
from easyicu.research_agent.evaluation_scorecard import compute_tristate
from easyicu.research_agent.providers.llm import llm_is_mockish
from easyicu.research_agent.providers.mocks import (
    _mock_code_primary_association,
    _mock_code_prediction_model,
    _mock_code_trajectory_clustering,
    PatternScriptedMockLLMClient,
)
from easyicu.research_agent.schema import CohortDescriptor, ResearchContext

from benchmarks.figure2_canonical9.evaluator.acceptance import (
    evaluate_figure2_paper_acceptance,
)
from benchmarks.figure2_canonical9.preflight import runtime as rt
from benchmarks.figure2_canonical9.preflight.fixtures import (
    PREFLIGHT_CASES,
    FULFILLMENT_NOT_PRODUCED_OFFLINE,
    FULFILLMENT_PLANNED_ONLY,
    FULFILLMENT_PRODUCED,
    PreflightCase,
    ProductMapping,
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


def _survival_preflight_code(case: PreflightCase) -> str:
    """Reviewed local H1 script: real PHReg + auditable diagnostic artifacts."""

    template = r"""
    from __future__ import annotations
    import json
    import os
    from pathlib import Path

    import numpy as np
    import pandas as pd
    from scipy.stats import norm
    from statsmodels.duration.hazard_regression import PHReg

    out_dir = Path(os.environ["STEP_OUT_DIR"])
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(os.environ["COHORT_PARQUET"])
    columns = ["vent_24h_any", "followup_days", "event_28d", "age", "sofa2"]
    model_df = df[columns].replace([np.inf, -np.inf], np.nan).dropna().copy()
    model_df["event_28d"] = model_df["event_28d"].astype(int)
    if len(model_df) < 40 or model_df["event_28d"].nunique() != 2:
        raise RuntimeError("survival preflight requires both event classes and >=40 rows")

    exog = model_df[["vent_24h_any", "age", "sofa2"]].astype(float)
    fit = PHReg(
        model_df["followup_days"].astype(float),
        exog,
        status=model_df["event_28d"],
        ties="efron",
    ).fit(disp=0)
    names = list(exog.columns)
    rows = []
    for name, beta, se, p_value in zip(names, fit.params, fit.bse, fit.pvalues):
        rows.append({
            "contrast_id": name,
            "effect_measure": "hazard_ratio",
            "estimate": float(np.exp(beta)),
            "lower_ci": float(np.exp(beta - 1.959963984540054 * se)),
            "upper_ci": float(np.exp(beta + 1.959963984540054 * se)),
            "p_value": float(p_value),
            "n": int(len(model_df)),
            "event_n": int(model_df["event_28d"].sum()),
        })
    pd.DataFrame(rows).to_csv(out_dir / "cox_summary.csv", index=False)

    curve_rows = []
    for group, group_df in model_df.groupby("vent_24h_any", observed=True):
        survival = 1.0
        for time in sorted(group_df.loc[group_df["event_28d"] == 1, "followup_days"].unique()):
            at_risk = int((group_df["followup_days"] >= time).sum())
            events = int(
                (
                    (group_df["followup_days"] == time)
                    & (group_df["event_28d"] == 1)
                ).sum()
            )
            if at_risk:
                survival *= 1.0 - events / at_risk
            curve_rows.append({
                "vent_24h_any": int(group),
                "time_days": float(time),
                "survival_probability": float(survival),
                "at_risk": at_risk,
                "events": events,
            })
    pd.DataFrame(curve_rows).to_csv(out_dir / "survival_curve.csv", index=False)

    schoenfeld = np.asarray(fit.schoenfeld_residuals)
    event_times = model_df["followup_days"].to_numpy(dtype=float)
    valid = (
        np.isfinite(schoenfeld[:, 0])
        & np.isfinite(event_times)
        & (model_df["event_28d"].to_numpy(dtype=int) == 1)
    )
    if int(valid.sum()) >= 3:
        correlation = float(np.corrcoef(schoenfeld[valid, 0], np.log1p(event_times[valid]))[0, 1])
        z_value = correlation * np.sqrt(max(int(valid.sum()) - 2, 1)) / np.sqrt(
            max(1.0 - correlation**2, 1e-12)
        )
        p_value = float(2.0 * norm.sf(abs(z_value)))
    else:
        correlation = float("nan")
        p_value = float("nan")
    pd.DataFrame([{
        "term": "vent_24h_any",
        "diagnostic": "schoenfeld_time_correlation_screen",
        "correlation": correlation,
        "p_value": p_value,
        "diagnostic_only": True,
    }]).to_csv(out_dir / "ph_diagnostics.csv", index=False)

    primary = next(row for row in rows if row["contrast_id"] == "vent_24h_any")
    summary = {
        "method": "cox_proportional_hazards",
        "status": "ok",
        "primary_estimand": "hazard ratio from fixed 24-hour landmark",
        "hazard_ratio": primary["estimate"],
        "output_files": {
            "table:cox_summary": "cox_summary.csv",
            "table:survival_curve": "survival_curve.csv",
            "table:ph_diagnostics": "ph_diagnostics.csv",
            "statistic:hazard_ratio": "cox_summary.csv",
        },
    }
    (out_dir / "step_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary))
    """
    return textwrap.dedent(template).strip() + "\n"


def _causal_preflight_code(case: PreflightCase) -> str:
    """Reviewed local H2 script: propensity, IPTW, balance, and positivity."""

    template = r"""
    from __future__ import annotations
    import json
    import os
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import statsmodels.api as sm

    out_dir = Path(os.environ["STEP_OUT_DIR"])
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(os.environ["COHORT_PARQUET"])
    exposure = "vasopressor"
    outcome = "death"
    covariates = ["age", "sofa2", "lact", "map"]
    model_df = df[[exposure, outcome, *covariates]].replace(
        [np.inf, -np.inf], np.nan
    ).dropna().copy()
    model_df[exposure] = model_df[exposure].astype(int)
    model_df[outcome] = model_df[outcome].astype(int)
    if len(model_df) < 60 or model_df[exposure].nunique() != 2:
        raise RuntimeError("causal preflight requires exposure overlap and >=60 rows")

    design = sm.add_constant(model_df[covariates].astype(float), has_constant="add")
    assignment_fit = sm.Logit(model_df[exposure], design).fit(disp=0, maxiter=200)
    propensity = np.asarray(assignment_fit.predict(design), dtype=float)
    propensity = np.clip(propensity, 1e-4, 1.0 - 1e-4)
    treated_probability = float(model_df[exposure].mean())
    treatment = model_df[exposure].to_numpy(dtype=int)
    weights = np.where(
        treatment == 1,
        treated_probability / propensity,
        (1.0 - treated_probability) / (1.0 - propensity),
    )
    observed = model_df[outcome].to_numpy(dtype=float)
    treated_risk = float(np.average(observed[treatment == 1], weights=weights[treatment == 1]))
    control_risk = float(np.average(observed[treatment == 0], weights=weights[treatment == 0]))
    risk_difference = treated_risk - control_risk

    effect = pd.DataFrame([{
        "contrast_id": "early_vasopressor_ate",
        "estimand": "stabilized_iptw_ate_risk_difference",
        "effect_measure": "risk_difference",
        "estimate": risk_difference,
        "treated_risk": treated_risk,
        "control_risk": control_risk,
        "n": int(len(model_df)),
    }])
    effect.to_csv(out_dir / "causal_effect.csv", index=False)

    def smd(values, group, sample_weight):
        treated = group == 1
        mean_t = np.average(values[treated], weights=sample_weight[treated])
        mean_c = np.average(values[~treated], weights=sample_weight[~treated])
        var_t = np.average((values[treated] - mean_t) ** 2, weights=sample_weight[treated])
        var_c = np.average((values[~treated] - mean_c) ** 2, weights=sample_weight[~treated])
        return float((mean_t - mean_c) / np.sqrt(max((var_t + var_c) / 2.0, 1e-12)))

    balance_rows = []
    unit_weights = np.ones(len(model_df), dtype=float)
    for covariate in covariates:
        values = model_df[covariate].to_numpy(dtype=float)
        balance_rows.append({
            "covariate": covariate,
            "smd_before": smd(values, treatment, unit_weights),
            "smd_after": smd(values, treatment, weights),
        })
    pd.DataFrame(balance_rows).to_csv(
        out_dir / "covariate_balance.csv", index=False
    )

    positivity = pd.DataFrame([{
        "propensity_min": float(propensity.min()),
        "propensity_max": float(propensity.max()),
        "fraction_below_0_05": float(np.mean(propensity < 0.05)),
        "fraction_above_0_95": float(np.mean(propensity > 0.95)),
        "effective_sample_size": float(weights.sum() ** 2 / np.sum(weights**2)),
    }])
    positivity.to_csv(out_dir / "positivity_diagnostics.csv", index=False)

    assignment = model_df[[exposure, *covariates]].copy()
    assignment["propensity_score"] = propensity
    assignment["stabilized_weight"] = weights
    assignment.to_csv(out_dir / "assignment_model.csv", index=False)

    summary = {
        "method": "causal_emulation",
        "status": "ok",
        "primary_estimand": "stabilized IPTW ATE-style 28-day risk difference",
        "causal_claim_scope": "observational target-trial assumptions required",
        "assignment_models": [{
            "model_id": "preflight_propensity",
            "fit_status": "fitted",
            "n": int(len(model_df)),
            "propensity_score_column": "propensity_score",
        }],
        "output_files": {
            "table:causal_effect": "causal_effect.csv",
            "table:covariate_balance": "covariate_balance.csv",
            "table:positivity_diagnostics": "positivity_diagnostics.csv",
            "artifact:assignment_model": "assignment_model.csv",
            "statistic:risk_difference": "causal_effect.csv",
        },
    }
    (out_dir / "step_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary))
    """
    return textwrap.dedent(template).strip() + "\n"


def _primary_preflight_code(
    case: PreflightCase,
    *,
    context: ResearchContext,
) -> str:
    """Choose a reviewed local Coder response by typed analysis family."""

    if case.primary_code_kind == "association":
        return _mock_code_primary_association(
            ctx=context,
            step_id=case.primary_step_id,
            outcome=case.target_outcome,
            predictor=case.primary_exposure,
            adjust=["age"],
            typed_model_contract=True,
        )
    if case.primary_code_kind == "prediction":
        return _mock_code_prediction_model(
            ctx=context,
            step_id=case.primary_step_id,
            outcome=case.target_outcome,
        )
    if case.primary_code_kind == "survival":
        return _survival_preflight_code(case)
    if case.primary_code_kind == "causal":
        return _causal_preflight_code(case)
    if case.primary_code_kind == "trajectory":
        return _mock_code_trajectory_clustering(
            ctx=context,
            step_id=case.primary_step_id,
            outcome=case.target_outcome,
        )
    raise ValueError(
        f"unsupported preflight primary_code_kind={case.primary_code_kind!r}"
    )

# Both values must be passed to CodeRunner explicitly.  ``runner_network`` is
# a DockerRunner option; it does not configure the subprocess runner used by
# this diagnostic preflight.  The explicit False also prevents an ambient
# EASYICU_ALLOW_UNSAFE_HOST_FALLBACK environment variable from weakening the
# boundary (CodeRunner only reads that variable when the kwarg is None).
PREFLIGHT_RUNNER_KWARGS: Dict[str, Any] = {
    "network_policy": "none",
    "allow_unsafe_host_fallback": False,
}
READINESS_CLASS = "partial_flow_smoke"


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


class ScriptedPreflightLLM:
    """Build a reviewed static offline client plus local run instrumentation.

    The object handed to :class:`ResearchAgentPipeline` is the exact built-in
    :class:`PatternScriptedMockLLMClient`, rather than this controller or a
    benchmark-defined subclass.  That matters after the provider hardening:
    arbitrary subclasses and wrappers cannot obtain prompt-delivery authority
    merely by inheriting from a mock.  The factory registers the exact reviewed
    built-in type at construction and binds its dispatch code before delivery.

    * The planner request returns the fixture's typed ``AnalysisPlan`` (so the
      deterministic grouped-Table-1 executor is eligible).
    * The primary-association code request returns the mock's battle-tested
      logistic-regression script *bound to the fixture's exact exposure* (so the
      agent-owned primary does not depend on the mock's role-inference heuristic).
    * Optional fault injection returns runtime-raising code for a target step on
      both its initial write and every repair, to exercise the repair cap.

    Every response is produced locally.  Offline classification is descriptive;
    the authoritative zero-Provider proof is the transport spy in
    :func:`provider_transport_spy`.
    """

    def __init__(
        self,
        case: PreflightCase,
        *,
        fault_step: Optional[str] = None,
        fault_code: str = _FAULT_CODE,
        request_replan_from_primary: bool = False,
        replan_strategy: str = "noop",
    ) -> None:
        self._case = case
        self._fault_step = fault_step
        if replan_strategy not in {"noop", "substantive"}:
            raise ValueError(f"unsupported replan strategy {replan_strategy!r}")
        plan_json = case.build_plan().model_dump_json(indent=2)
        context = ResearchContext(
            research_question=case.question,
            cohort=CohortDescriptor(
                cohort_name=f"{case.task_id}_preflight",
                database=case.database,
                n_stays=0,
                n_patients=0,
            ),
            variables=[],
            target_outcome=case.target_outcome,
            primary_exposure=case.primary_exposure,
        )
        primary_code = _primary_preflight_code(case, context=context)
        if request_replan_from_primary and case.primary_code_kind == "association":
            primary_code = primary_code.replace(
                '"method": "logistic_regression",',
                '"method": "logistic_regression",\n'
                '        "replan_requested": True,',
            )
        # Re-serve the same reviewed script on a bounded repair request.  The
        # harness is testing the pipeline's repair/fail-stop control flow, not
        # asking a mock to invent progressively different science.
        primary_responses: List[str] = [primary_code] * 16
        if fault_step == case.primary_step_id:
            primary_responses = [fault_code] * 16
        replan_responses = _replan_responses(case, strategy=replan_strategy)
        rules: List[tuple[str, List[str]]] = [
            # Rules are generic -> specific; PatternScriptedMock gives the
            # later matching rule priority.
            ("WRITE THE PYTHON CODE", ["MOCK RESPONSE — preflight auxiliary"] * 32),
            (case.primary_step_id, primary_responses),
        ]
        if fault_step is not None and fault_step != case.primary_step_id:
            rules.append((fault_step, [fault_code] * 16))
        # Complex Planner prompts can contain benchmark step identifiers in
        # their case-specific context.  This exact role anchor must therefore
        # outrank code step-id rules, while remaining narrower than text merely
        # embedding a parsed plan.
        rules.append(
            ("Produce an ICU-AWARE RESEARCH PLAN as JSON", [plan_json] * 8)
        )
        # This must come after the step-id rules: a Replanner prompt embeds the
        # current plan (and therefore every step id), but it must receive a
        # plan JSON rather than a code-fault response.
        rules.append((_REPLAN_ANCHOR, replan_responses))
        self.client = PatternScriptedMockLLMClient(
            rules,
            default="MOCK RESPONSE — preflight default",
        )

    @property
    def total_calls(self) -> int:
        return len(self.client.calls)

    @property
    def plan_calls(self) -> int:
        return sum(
            1
            for messages, _kwargs in self.client.calls
            if any(anchor in _last_user(messages).upper() for anchor in _PLAN_ANCHORS)
            and _REPLAN_ANCHOR not in _last_user(messages).upper()
        )

    @property
    def code_calls(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for messages, _kwargs in self.client.calls:
            prompt = _last_user(messages)
            if not any(anchor in prompt.upper() for anchor in _CODE_ANCHORS):
                continue
            for step in self._case.build_plan().steps:
                if step.step_id in prompt:
                    counts[step.step_id] = counts.get(step.step_id, 0) + 1
                    break
        return counts


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
    runtime: Dict[str, Any] = field(default_factory=dict)
    blocked_reason: Optional[str] = None
    pipeline_ran: bool = False

    # -- derived views ----------------------------------------------------
    @property
    def readiness_class(self) -> str:
        """This offline harness can exercise a flow, never grant readiness."""

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
        """Authoritative: the real httpx transport spy count, not a self-report."""

        return self.provider_transport_calls

    @property
    def llm_offline_classified(self) -> bool:
        """Descriptive only; transport spying supplies the zero-Provider proof."""

        return llm_is_mockish(self.llm.client)

    def resolved_product_map(self) -> List[Dict[str, Any]]:
        """Resolve formal outputs against this run's persisted evidence."""

        return resolve_product_map(self)

    def produced_suite_outputs(self) -> List[str]:
        return [
            row["suite_output"]
            for row in self.resolved_product_map()
            if row["observed_fulfillment"] == FULFILLMENT_PRODUCED
        ]

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


def observe_product_fulfillment(mapping: ProductMapping, run: PreflightRun) -> str:
    """Read a product's honest fulfillment from the real manifest/evidence."""

    if mapping.step_id is None:
        return FULFILLMENT_NOT_PRODUCED_OFFLINE
    record = run.record(mapping.step_id)
    if not record:
        return "absent"
    prefix = mapping.artifact_evidence_prefix
    has_artifact = bool(prefix) and any(
        str(evidence_id).startswith(prefix)
        for evidence_id in record.get("evidence_ids", [])
    )
    if record.get("status") == "ok" and has_artifact:
        return FULFILLMENT_PRODUCED
    return FULFILLMENT_PLANNED_ONLY


def resolve_product_map(run: PreflightRun) -> List[Dict[str, Any]]:
    """Bind every live suite output to its observed offline fulfillment."""

    resolved: List[Dict[str, Any]] = []
    for suite_output, mapping in run.case.product_mapping():
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
    """Report whether every recorded subprocess step stayed no-network/strict."""

    subprocess_steps = [
        {
            "step_id": record.get("step_id"),
            "requested_network_policy": record.get("requested_network_policy"),
            "effective_isolation": record.get("effective_isolation"),
            "isolation_degraded": record.get("isolation_degraded"),
        }
        for record in manifest.get("per_step_records", [])
        if record.get("requested_network_policy") is not None
    ]
    return {
        "ok": bool(subprocess_steps)
        and all(
            step["requested_network_policy"] == "none"
            and step["isolation_degraded"] is False
            for step in subprocess_steps
        ),
        "subprocess_steps": subprocess_steps,
    }


def _write_product_map_artifact(run: PreflightRun) -> Path:
    """Persist the resolved, diagnostic-only formal-output ledger for a run."""

    path = run.run_dir / "preflight_product_map.json"
    path.write_text(
        json.dumps(
            {
                "task_id": run.case.task_id,
                "readiness_class": run.readiness_class,
                "expected_products": list(run.case.expected_products),
                "product_map": run.resolved_product_map(),
                "produced_suite_outputs": run.produced_suite_outputs(),
                "network_policy": run.network_policy_report(),
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    return path


def _replan_responses(case: PreflightCase, *, strategy: str) -> List[str]:
    """Build closed static candidates for real replan-cap control tests."""

    initial = case.build_plan()
    if strategy == "noop":
        return [initial.model_dump_json(indent=2)] * 16
    primary_index = next(
        index
        for index, step in enumerate(initial.steps)
        if step.step_id == case.primary_step_id
    )
    # The first non-primary fixture step deliberately contract-fails offline,
    # so it is not an immutable completed record when the second replan runs.
    # Altering it gives the total-cap test a real second substantive candidate.
    future_index = next(
        (index for index in range(len(initial.steps)) if index != primary_index),
        primary_index,
    )
    first_steps = list(initial.steps)
    first_steps[primary_index] = first_steps[primary_index].model_copy(
        update={"intent": first_steps[primary_index].intent + " [replan-A]"}
    )
    first = initial.model_copy(
        update={"revision": initial.revision + 1, "steps": first_steps}
    )
    second_steps = list(first.steps)
    second_steps[future_index] = second_steps[future_index].model_copy(
        update={"intent": second_steps[future_index].intent + " [replan-B]"}
    )
    second = first.model_copy(
        update={"revision": first.revision + 1, "steps": second_steps}
    )
    return [
        first.model_dump_json(indent=2),
        second.model_dump_json(indent=2),
        second.model_dump_json(indent=2),
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
    max_consecutive_noop_replans: Optional[int] = None,
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
        # ``runner_network`` applies to DockerRunner.  The real subprocess
        # preflight boundary is explicitly pinned in runner_kwargs.
        runner_network="none",
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
    if max_consecutive_noop_replans is not None:
        kwargs["max_consecutive_noop_replans"] = max_consecutive_noop_replans
    return ra.ResearchAgentPipeline(**kwargs)


def run_preflight(
    case: PreflightCase,
    *,
    workdir: Path,
    n_rows: int = 80,
    fault_step: Optional[str] = None,
    fault_code: str = _FAULT_CODE,
    request_replan_from_primary: bool = False,
    replan_strategy: str = "noop",
    stop_after_step_id: Optional[str] = None,
    resume_run_id: Optional[str] = None,
    resume_from_step_id: Optional[str] = None,
    timeout_seconds: float = 60.0,
    standard_executor_timeout_seconds: float = 900.0,
    max_code_repair_attempts: int = 2,
    enable_replanning: bool = False,
    max_replans: Optional[int] = None,
    max_consecutive_noop_replans: Optional[int] = None,
) -> PreflightRun:
    """Run one offline graph-level preflight and return structured evidence.

    The whole pipeline run executes inside :func:`provider_transport_spy`, so the
    returned ``provider_transport_calls`` is a real transport-layer measurement.
    """

    runtime_manifest = preflight_runtime_manifest(workdir)
    llm = ScriptedPreflightLLM(
        case,
        fault_step=fault_step,
        fault_code=fault_code,
        request_replan_from_primary=request_replan_from_primary,
        replan_strategy=replan_strategy,
    )
    if not runtime_manifest.integration_ready:
        return PreflightRun(
            case=case,
            run_dir=Path(workdir),
            run_id="",
            manifest={},
            llm=llm,
            runtime=runtime_manifest.as_dict(),
            blocked_reason=runtime_manifest.blocked_reason,
            pipeline_ran=False,
        )
    pipeline = build_pipeline(
        case,
        workdir=workdir,
        llm=llm.client,
        timeout_seconds=timeout_seconds,
        standard_executor_timeout_seconds=standard_executor_timeout_seconds,
        max_code_repair_attempts=max_code_repair_attempts,
        enable_replanning=enable_replanning,
        max_replans=max_replans,
        max_consecutive_noop_replans=max_consecutive_noop_replans,
    )
    run_kwargs: Dict[str, Any] = dict(
        question=case.question,
        cohort=case.build_cohort(n_rows),
        target_outcome=case.target_outcome,
        database=case.database,
        cohort_name=f"{case.task_id}_preflight",
        concept_descriptions=dict(case.concept_descriptions),
    )
    if case.primary_exposure:
        run_kwargs["primary_exposure"] = case.primary_exposure
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
    # The initial probe prevents a known-bad host from launching generated
    # code.  Probe again after a run so an isolation backend that becomes
    # unavailable while a nested session starts is reported explicitly rather
    # than leaking through as a generic repair failure.
    final_runtime = preflight_runtime_manifest(run_dir)
    isolation_reason = next(
        (
            rt.step_isolation_unavailable(record, final_runtime.isolation)
            for record in manifest.get("per_step_records", [])
            if rt.step_isolation_unavailable(record, final_runtime.isolation)
        ),
        None,
    )
    blocked_reason = final_runtime.blocked_reason if isolation_reason else None
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
        runtime=final_runtime.as_dict(),
        blocked_reason=blocked_reason,
        pipeline_ran=True,
    )
    _write_product_map_artifact(run)
    return run


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

    return paper_acceptance_verdict(run).status


def paper_acceptance_verdict(run: PreflightRun):
    """Return the production gate's typed rejection evidence for a mock run."""

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
    return evaluate_figure2_paper_acceptance(results_path)


def _cli(task_key: str) -> int:
    import tempfile

    case = PREFLIGHT_CASES.get(task_key)
    if case is None:
        case = next(
            (c for c in PREFLIGHT_CASES.values() if c.task_id.startswith(task_key)),
            None,
        )
    if case is None:
        raise SystemExit(
            f"unknown task key {task_key!r}; try e1/e2/e3/m2/h1/h2/h3"
        )
    tmp = Path(tempfile.mkdtemp(prefix=f"preflight_{task_key}_"))
    run = run_preflight(case, workdir=tmp)
    if not run.pipeline_ran:
        print(
            json.dumps(
                {
                    "task_id": run.case.task_id,
                    "readiness_class": run.readiness_class,
                    "blocked": True,
                    "pipeline_ran": False,
                    "runtime_integration_ready": False,
                    "runtime_blocked_reason": run.blocked_reason,
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
                "runtime_integration_ready": run.runtime.get("integration_ready"),
                "runtime_blocked_reason": run.blocked_reason,
                "llm_offline_classified": run.llm_offline_classified,
                "external_provider_calls": run.external_provider_calls,
                "provider_transport_targets": run.provider_transport_targets,
                "total_llm_calls": run.llm.total_calls,
                "plan_calls": run.llm.plan_calls,
                "expected_products": list(run.case.expected_products),
                "product_map": run.resolved_product_map(),
                "produced_suite_outputs": run.produced_suite_outputs(),
                "network_policy": run.network_policy_report(),
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
    "observe_product_fulfillment",
    "resolve_product_map",
    "network_policy_report",
    "load_manifest",
    "paper_acceptance_verdict",
    "paper_acceptance_status",
]
