"""Pure metric-comparison rules shared by replication and audits."""

from __future__ import annotations

import json
import math
from easyicu.concept.metadata_sidecar import ColumnMetadataBinding
from easyicu.databases.research_capabilities import ResearchCapabilityProfile
from ..contracts.prediction_validation import ExternalValidationError, FrozenLogisticModel
from ..methods.sensitivity import compute_meta_analysis

from types import MappingProxyType
from typing import Any, Mapping, Optional

#: The structured EasyICU metric each paper-claim metric is compared with.  A
#: ratio is compared only on its own scale: a paper's risk ratio with a
#: reported risk ratio (``primary_rr``), never with an odds ratio.
CLAIM_METRIC_KEYS: Mapping[str, str] = MappingProxyType(
    {
        "or": "primary_or",
        "hr": "primary_or",
        "rr": "primary_rr",
        "auroc": "auroc",
        "auc": "auroc",
        "brier_score": "brier_score",
        "p_value": "primary_pvalue",
        "p": "primary_pvalue",
        "n": "n_stays",
    }
)


def claim_metric_key(metric: Optional[str]) -> Optional[str]:
    """The EasyICU metric a paper claim's metric is compared with, if any."""

    return None if metric is None else CLAIM_METRIC_KEYS.get(metric.lower())


def compare_metric_values(
    *,
    metric: Optional[str],
    paper_value: Optional[float],
    paper_direction: Optional[str],
    easyicu_value: Any,
) -> tuple[str, str]:
    """Classify paper vs EasyICU alignment for one claim."""

    if paper_value is None:
        if paper_direction and easyicu_value is not None:
            easy_dir = "positive" if float(easyicu_value) > 1 else "negative"
            if paper_direction == easy_dir:
                return (
                    "directionally_aligned",
                    "Direction matched but no exact paper scalar was available.",
                )
        return (
            "not_comparable",
            "Paper claim did not expose a comparable numeric value.",
        )
    if easyicu_value is None:
        return (
            "not_comparable",
            "EasyICU run did not emit a comparable structured metric.",
        )
    try:
        easy = float(easyicu_value)
    except Exception:
        return (
            "not_comparable",
            "EasyICU metric could not be interpreted numerically.",
        )

    metric_name = (metric or "").lower()
    if metric_name in {"p_value", "p"}:
        paper_sig = paper_value < 0.05
        easy_sig = easy < 0.05
        if paper_sig == easy_sig:
            return "directionally_aligned", "Significance state matched."
        return "not_aligned", "Significance state did not match."

    if metric_name in {"or", "hr", "rr"}:
        if (paper_value > 1 and easy > 1) or (paper_value < 1 and easy < 1):
            delta = abs(paper_value - easy) / max(abs(paper_value), 1e-6)
            if delta <= 0.25:
                return "aligned", "Effect direction and magnitude were close."
            return (
                "directionally_aligned",
                "Effect direction matched but magnitude differed.",
            )
        return "not_aligned", "Effect direction did not match."

    if metric_name in {"auroc", "auc", "brier_score", "outcome_rate", "n"}:
        delta = abs(paper_value - easy)
        tolerance = (
            0.03
            if metric_name in {"auroc", "auc", "brier_score", "outcome_rate"}
            else max(5.0, 0.05 * max(abs(paper_value), 1.0))
        )
        if delta <= tolerance:
            return "aligned", "Numeric value was within tolerance."
        return (
            "not_aligned",
            "Numeric value differed beyond the comparison tolerance.",
        )

    delta = abs(paper_value - easy)
    if delta <= max(0.05, 0.2 * max(abs(paper_value), 1.0)):
        return "aligned", "Generic numeric tolerance matched."
    return "not_aligned", "Generic numeric tolerance did not match."


__all__ = ["CLAIM_METRIC_KEYS", "claim_metric_key", "compare_metric_values", "pool_database_estimates"]


def require_transport_capabilities(profile: ResearchCapabilityProfile, model: FrozenLogisticModel) -> None:
    """A physical column cannot override an absent/degraded dictionary recipe."""
    if not profile.source_registered:
        raise ExternalValidationError("validation_database_unavailable", database=profile.database)
    missing = [p.concept for p in model.predictors if not profile.concept(p.concept).mapped]
    if missing:
        raise ExternalValidationError("predictor_unavailable", database=profile.database, concepts=missing)
    if not profile.concept(model.outcome_concept).mapped:
        raise ExternalValidationError("outcome_unavailable", database=profile.database, concept=model.outcome_concept)


def transport_concept_meaning(binding: ColumnMetadataBinding, profile: ResearchCapabilityProfile) -> str:
    """Compare meaning, units, aggregation, window and representation, not names.

    Source-table and local column names deliberately do not enter a meaning:
    harmonized concepts can come from different tables in different databases.
    """
    metadata = binding.metadata
    concept = metadata.source_concept
    extra = None
    if concept == "death":
        extra = profile.mortality.meaning if profile.mortality else "unavailable"
    elif concept in {"age", "weight"}:
        representation = getattr(profile, concept)
        extra = {
            "kind": representation.kind, "top_code": representation.top_code,
            "bin_edges": representation.bin_edges, "bin_values": representation.bin_values,
            "exact_through": representation.exact_through,
        }
    return json.dumps({
        "concept": concept, "role": metadata.role.value, "unit": metadata.canonical_unit,
        "aggregation": metadata.aggregation,
        "window": binding.derivation_window.to_dict() if binding.derivation_window else None,
        "representation_transform": binding.representation_transform, "source_meaning": extra,
    }, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def pool_database_estimates(
    rows: list[dict], *, reference_estimand: str, hartung_knapp: bool = False,
) -> dict:
    """Pool verified aggregate results with the reference's exact estimand.

    Only the sealing owner may set gate_passed=True, after checking the run's
    own gates and evidence. A design identity binds exposure/contrast, outcome
    definition, population and adjustment set. Changed definitions and db6a
    substitutions stay excluded even if their display labels happen to agree.
    Confirmation to execute a substituted design is not confirmation to pool it.
    SEs are never reconstructed from rounded intervals or guessed from N.
    """
    supported = {"odds_ratio", "hazard_ratio", "risk_ratio", "risk_difference", "mean_difference"}
    candidates = [row for row in rows if row.get("estimand_sha256") == reference_estimand
                  and row.get("gate_passed") is True and not row.get("substitution_codes")
                  and not row.get("definition_codes") and row.get("unit") in supported]
    anchor = candidates[0] if candidates else None
    result = dict(status="unavailable", claim_ceiling="pooled_observational_estimate",
                  scope="analysis_level", included=[], excluded=[], leave_one_out=[])
    if not 1 <= len(rows) <= 6 or len({row.get("database") for row in rows}) != len(rows):
        return {**result, "reason_code": "database_sources_not_independent"}
    for row in rows:
        reason = None
        if row.get("gate_passed") is not True:
            reason = "result_gate_failed"
        elif (not reference_estimand or row.get("estimand_sha256") != reference_estimand
              or row.get("substitution_codes") or row.get("definition_codes")):
            reason = "changed_estimand"
        elif (anchor is None or row.get("unit") != anchor["unit"]
              or row.get("estimand_label") != anchor.get("estimand_label")):
            reason = "effect_measure_changed"
        elif row.get("standard_error") is None:
            reason = "standard_error_unavailable"
        else:
            try:
                error = float(row["standard_error"])
                value = float(row["estimate"])
                low, high = float(row["ci_low"]), float(row["ci_high"])
                if not math.isfinite(error) or error <= 0 or isinstance(row["standard_error"], bool):
                    reason = "standard_error_invalid"
                elif (not all(map(math.isfinite, (value, low, high))) or not low <= value <= high
                      or (row["unit"].endswith("ratio") and low <= 0)):
                    reason = "estimate_invalid"
            except (ValueError, TypeError, KeyError):
                reason = "estimate_invalid"
        if reason:
            result["excluded"].append(dict(database=row.get("database"), reason_code=reason))
        else:
            result["included"].append(dict(row, estimate=value, ci_low=low, ci_high=high, standard_error=error))
    included = result["included"]
    if (any(not row.get("source_sha256") for row in included)
            or len({row.get("source_sha256") for row in included}) != len(included)):
        return {**result, "reason_code": "database_sources_not_independent"}
    if len(included) < 2:
        return {**result, "reason_code": "at_least_two_eligible_databases_required"}
    ratio = anchor["unit"].endswith("ratio")

    def calculate(subset, model="random", tau_method="reml"):
        computed = compute_meta_analysis(
            [math.log(row["estimate"]) if ratio else row["estimate"] for row in subset],
            [row["standard_error"] for row in subset], model=model,
            tau_method=tau_method, hartung_knapp=hartung_knapp)
        # SE, tau² and Q stay on the declared analysis scale; only effect and
        # interval coordinates are back-transformed for display.
        if ratio:
            for key in ("estimate", "ci_low", "ci_high", "prediction_low", "prediction_high"):
                if computed[key] is not None:
                    computed[key] = math.exp(computed[key])
        return computed

    try:
        result.update(fixed=calculate(included, model="fixed"), random=calculate(included),
                      dl_sensitivity=calculate(included, tau_method="dl"))
        for row in included:
            subset = [other for other in included if other["database"] != row["database"]]
            result["leave_one_out"].append(dict(omitted_database=row["database"], random=calculate(subset))
                if len(subset) >= 2 else dict(omitted_database=row["database"], random=None,
                                              reason_code="at_least_two_eligible_databases_required"))
    except (ValueError, OverflowError, FloatingPointError):
        return {**result, "status": "unavailable", "reason_code": "meta_analysis_numerical_failure"}
    return {**result, "status": "pooled", "unit": anchor["unit"], "analysis_scale": "log" if ratio else "identity",
            "reference_estimand": reference_estimand,
            "assumptions": "Independent databases; same exposure, outcome definition, population, adjustment set and effect measure. Observational association; pooling does not remove confounding."}


def read_replication_estimate(run_dir, database: str) -> dict:
    """Admit a native replication's current approved, sealed aggregate table.

    Only a named primary adjusted-association product is supported. No summary,
    narrative, filesystem discovery or interval-derived SE can supply numbers.
    """
    import csv
    from pathlib import Path

    from ..authority.evidence_store import sha256_of_file
    from ..authority.plan_review import PlanReviewAuthority
    from ..authority.plan_scope import plan_signature
    from ..authority.run_input import load_verified_run_input_capsule
    from ..authority.runtime_artifacts import load_run_artifact_authority, verified_run_evidence_path
    from ..authority.typed_binding import _resolve_typed_input_evidence
    from ..canonical_json import canonical_sha256
    from ..orchestration.human_review_checkpoint import HumanReviewCheckpoint
    from ..schema import AnalysisPlan, ResearchContext

    run = Path(run_dir)

    def read(name):
        path = run / name
        if path.is_symlink() or not path.is_file() or path.stat().st_size > 16 * 1024 * 1024:
            raise ValueError("replication_authority_unavailable")
        return json.loads(path.read_bytes())

    gates = read("run_status.json").get("gates", {})
    if not all(gates.get(key) is True for key in ("execution_complete", "evidence_complete", "numeric_verified", "analysis_validated")):
        raise ValueError("result_gate_failed")
    checkpoint = HumanReviewCheckpoint.model_validate(read("human_review_checkpoint.json"))
    if checkpoint.state != "completed" or not checkpoint.execution_start_receipt or not checkpoint.requests:
        raise ValueError("replication_approval_unavailable")
    plan = AnalysisPlan.model_validate(checkpoint.plan_handoff["plan"])
    decisions = {row["review_id"]: row for row in checkpoint.approved_decisions}
    if set(decisions) != {request.review_id for request in checkpoint.requests}:
        raise ValueError("replication_approval_changed")
    for request in checkpoint.requests:
        decision = decisions[request.review_id]
        authority = PlanReviewAuthority.model_validate(request.payload["plan_review_authority"])
        if (decision["decision"] != "approved" or decision["authority_sha256"] != request.authority_sha256
                or authority.plan_sha256 != canonical_sha256(plan.model_dump(mode="json"))):
            raise ValueError("replication_approval_changed")
    if sha256_of_file(run / "run_input_capsule.json") != checkpoint.run_input_capsule_sha256:
        raise ValueError("replication_input_changed")
    capsule = read("run_input_capsule.json")
    verified = load_verified_run_input_capsule(run_dir=run, scientific_identity=capsule["scientific_identity"])
    context = ResearchContext.model_validate_json(verified.context_evidence_path.read_bytes())
    if context.cohort.database != database:
        raise ValueError("replication_database_changed")
    from easyicu.databases.research_plan_requirements import plan_capability_decision

    decision = plan_capability_decision(context, plan)
    if decision is None or decision.status == "refusal":
        raise ValueError("replication_capability_unassessed")
    product = pooling_primary_product(plan)
    ledger = load_run_artifact_authority(run)
    if ledger is None:
        raise ValueError("replication_result_unsealed")
    ref, failure = _resolve_typed_input_evidence(input_name=product, plan=plan,
        evidence_records=list(verified.evidence_records.values()), per_step_records=ledger["per_step_records"], run_dir=run)
    if ref is None or failure:
        raise ValueError("replication_result_unsealed")
    record = verified.evidence_records[ref.evidence_id]
    path = verified_run_evidence_path(run, record)
    if path is None or path.stat().st_size > 128 * 1024:
        raise ValueError("replication_result_unsealed")
    with path.open(newline="", encoding="utf-8") as stream:
        rows = []
        for row in csv.DictReader(stream):
            rows.append(row)
            if len(rows) > 64:
                raise ValueError("replication_primary_unavailable")
    chosen = pooling_primary_row(plan, product, rows)
    identity = canonical_sha256(dict(plan=plan_signature(plan), context=pooling_context_coordinates(plan, context)))
    return dict(database=database, source_sha256=verified.capsule.cohort_sha256, estimand_sha256=identity,
        estimand_label="adjusted_survival" if "survival" in product else "adjusted_association", unit=chosen["effect_scale"], gate_passed=True,
        estimate=float(chosen["estimate"]), ci_low=float(chosen["ci_low"]), ci_high=float(chosen["ci_high"]),
        standard_error=float(chosen["standard_error"]) if chosen.get("standard_error") else None,
        substitution_codes=[row.reason_code for row in decision.substitutions],
        definition_codes=["binned_covariate:" + key for key in sorted(
            context.cohort.provenance.get("database_capability_binding", {}).get("binned_covariates", {}))],
        evidence_id=ref.evidence_id, evidence_sha256=record["sha256"])


def pooling_primary_product(plan) -> str:
    supported = {"table:adjusted_association_estimates", "table:survival_effect_estimates"}
    products = [product for step in plan.steps if step.planned_analysis_role == "primary"
                for product in step.expected_outputs if product in supported]
    if len(products) != 1:
        raise ValueError("replication_primary_unavailable")
    return products[0]


def pooling_primary_row(plan, product: str, rows: list[dict]) -> dict:
    """Normalize only two governed primary table schemas, never a summary."""
    from ..contracts.association_execution import headline_association_row

    if pooling_primary_product(plan) != product:
        raise ValueError("replication_primary_unavailable")
    step = next(step for step in plan.steps if step.planned_analysis_role == "primary" and product in step.expected_outputs)
    if product == "table:survival_effect_estimates":
        required = step.family_primary_result_requirement
        if (required is None or required.analysis_family != "survival"
                or len(rows) != 1 or rows[0].get("exposure_source") != required.exposure_source
                or rows[0].get("outcome") != required.outcome or rows[0].get("effect_scale") != "hazard_ratio"
                or required.effect_scale != "hazard_ratio"):
            raise ValueError("replication_primary_unavailable")
        return {**rows[0], "estimate": rows[0]["hazard_ratio"], "n": rows[0]["n_analysis_rows"]}
    if len(step.model_requirements) != 1:
        raise ValueError("replication_primary_unavailable")
    required = step.model_requirements[0]
    chosen = headline_association_row(rows)
    if (chosen is None or chosen.get("fit_status") != "fitted" or chosen.get("requirement_id") != required.requirement_id
            or chosen.get("outcome") != required.outcome or chosen.get("exposure") != required.exposure_source):
        raise ValueError("replication_primary_unavailable")
    return chosen


def pooling_context_coordinates(plan, context) -> dict:
    """Only declared science, excluding observed distributions and source IDs."""
    from easyicu.databases.research_plan_requirements import _constraints
    from ..contracts.exposure_group_rules import EXPOSURE_GROUP_LABELS_KEY

    names = {context.primary_exposure, context.target_outcome}
    for step in plan.steps:
        if step.planned_analysis_role != "primary":
            continue
        requirements = list(step.model_requirements)
        if step.family_primary_result_requirement:
            requirements.append(step.family_primary_result_requirement)
        for required in requirements:
            names.update([required.exposure_source, required.outcome, *(required.covariates or [])])
    fields = ("name", "source_concept", "unit", "aggregation_default", "analysis_window", "analysis_window_role",
              "clinical_definition", "is_ordinal", "ordinal_levels", "derived_from_concepts", "temporal_resolution")
    definitions = {}
    for name in sorted(names - {None}):
        variable = context.variable(name)
        definitions[name] = {key: variable.model_dump(mode="json")[key] for key in fields} if variable else None
    constraints = _constraints(context)
    return dict(exposure=context.primary_exposure, outcome=context.target_outcome,
        endpoint=context.endpoint.model_dump(mode="json") if context.endpoint else None,
        time_windows=[row.model_dump(mode="json") for row in context.time_windows],
        temporal_constraints=[row.model_dump(mode="json") for row in context.temporal_constraints],
        population={key: constraints.get(key) for key in ("cohort", "population_spec", "analysis_design", "analysis_timing", "exposure_groupings")},
        exposure_groups=context.cohort.provenance.get(EXPOSURE_GROUP_LABELS_KEY), definitions=definitions,
        inclusion=context.cohort.inclusion_criteria, exclusion=context.cohort.exclusion_criteria)
