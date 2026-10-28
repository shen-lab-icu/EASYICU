"""Host projection of sealed primary products, plus comparison evidence."""
from __future__ import annotations

import csv
from pathlib import Path

from pydantic import Field, model_validator

from easyicu.research_agent.authority.evidence_store import EvidenceStore
from easyicu.research_agent.authority.runtime_artifacts import load_run_artifact_authority, verified_run_evidence_path
from easyicu.research_agent.authority.typed_binding import _resolve_typed_input_evidence
from easyicu.research_agent.canonical_json import canonical_sha256
from easyicu.research_agent.methods.sensitivity import describe_estimate_heterogeneity
from easyicu.research_agent.replication.metrics import pool_database_estimates, pooling_primary_product, pooling_primary_row
from easyicu.research_agent.replication.report import render_database_meta_analysis
from .research_append_source import _approved_source, contained, read_json
from .research_followup_source import load_completed_result
from .research_multi_contracts import Code, Contract, Database, Digest, MultiDatabaseError
from .research_multi_plan import estimand_identity


class SealedEstimate(Contract):
    database: Database
    run_id: Code
    source_sha256: Digest
    plan_sha256: Digest
    estimand_sha256: Digest
    estimand_label: Code
    unit: Code
    population_n: int = Field(gt=0)
    estimate: float = Field(allow_inf_nan=False)
    ci_low: float = Field(allow_inf_nan=False)
    ci_high: float = Field(allow_inf_nan=False)
    # Coefficient SE: log scale for ratios; original scale for differences.
    # Legacy products without an SE or their complete gates remain displayable
    # but cannot acquire pooling authority from their confidence intervals.
    standard_error: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    gate_passed: bool = False
    evidence_id: Code
    evidence_sha256: Digest
    substitution_codes: tuple[Code, ...] = Field(default=(), max_length=64)
    definition_codes: tuple[Code, ...] = Field(default=(), max_length=64)

    @model_validator(mode="after")
    def interval(self):
        if not self.ci_low <= self.estimate <= self.ci_high:
            raise ValueError("multi_database_estimate_invalid")
        if self.unit in {"odds_ratio", "hazard_ratio", "risk_ratio"} and self.ci_low <= 0:
            raise ValueError("multi_database_estimate_invalid")
        return self


def sealed_primary_estimate(wrapper: Path, target):
    """Read only a verified aggregate table produced by the approved primary.

    Unsupported result formats are explicit stops; never parse a narrative or
    take a first/largest coefficient as the study's primary effect.
    """
    review = target.review
    source = load_completed_result(wrapper, run_id=review.run_id, study_id=review.study_context_id)
    run = contained(wrapper, "pipeline/" + review.run_id)
    _, plan, _, verified = _approved_source(run)
    if canonical_sha256(plan.model_dump(mode="json")) != review.plan_sha256:
        raise MultiDatabaseError("multi_database_result_changed")
    product = pooling_primary_product(plan)
    ledger = load_run_artifact_authority(run)
    ref, failure = _resolve_typed_input_evidence(input_name=product, plan=plan,
        evidence_records=list(verified.evidence_records.values()), per_step_records=ledger["per_step_records"], run_dir=run)
    if ref is None or failure:
        raise MultiDatabaseError("multi_database_result_unsealed")
    record = verified.evidence_records[ref.evidence_id]
    path = verified_run_evidence_path(run, record)
    if path is None or path.stat().st_size > 128 * 1024:
        raise MultiDatabaseError("multi_database_result_unsealed")
    with path.open(newline="", encoding="utf-8") as stream:
        rows = []
        for row in csv.DictReader(stream):
            rows.append(row)
            if len(rows) > 64:
                raise MultiDatabaseError("multi_database_primary_estimate_unavailable")
    chosen = pooling_primary_row(plan, product, rows)
    gates = read_json(run / "run_status.json").get("gates", {})
    gate_passed = all(gates.get(key) is True for key in
                      ("execution_complete", "evidence_complete", "numeric_verified", "analysis_validated"))
    raw_se = chosen.get("standard_error")
    return SealedEstimate(database=target.target.database, run_id=source.run_id,
        source_sha256=target.source_sha256, plan_sha256=review.plan_sha256,
        estimand_sha256=estimand_identity(plan, source.context),
        estimand_label="adjusted_survival" if "survival" in product else "adjusted_association",
        unit=chosen["effect_scale"], population_n=int(chosen["n"]),
        estimate=float(chosen["estimate"]), ci_low=float(chosen["ci_low"]), ci_high=float(chosen["ci_high"]),
        standard_error=float(raw_se) if raw_se not in (None, "") else None, gate_passed=gate_passed,
        evidence_id=ref.evidence_id, evidence_sha256=record["sha256"],
        substitution_codes=tuple(row.reason_code for row in target.decision.substitutions),
        definition_codes=getattr(target, "definition_codes", ())).model_dump(mode="json")


def combined_answer(batch, *, hartung_knapp: bool = False):
    record, card, targets = batch.load()
    state = batch.state()
    rows, estimates = [], []
    reference = None
    try:
        raw = getattr(batch.owner, "reference_result", lambda _: None)(record["design"])
        if raw is not None:
            original = SealedEstimate.model_validate(raw)
            reference = {"state": "completed", "primary": original.model_dump(mode="json")}
            if original.database not in {row.target.database for row in targets}:
                estimates.append(original)
    except (ValueError, OSError, KeyError) as exc:
        reference = {"state": "unavailable", "reason_code": getattr(exc, "code", "multi_database_result_changed")}
    for target in targets:
        status = state["targets"][target.target.database]
        row = {"database": target.target.database, "run_id": target.review.run_id if target.review else None,
               "state": status["state"], "reason_code": status["reason_code"]}
        if status["state"] == "completed":
            try:
                fresh = batch.owner.result(target)
                if fresh != status["result"]:
                    raise MultiDatabaseError("multi_database_result_changed")
                estimate = SealedEstimate.model_validate(fresh)
                if (estimate.database != target.target.database or estimate.run_id != target.review.run_id
                        or estimate.source_sha256 != target.source_sha256 or estimate.plan_sha256 != target.review.plan_sha256):
                    raise MultiDatabaseError("multi_database_result_changed")
                row["primary"] = estimate.model_dump(mode="json")
                estimates.append(estimate)
            except (ValueError, OSError, KeyError) as exc:
                row.update(primary_state="unavailable", reason_code=getattr(exc, "code", "multi_database_result_changed"))
        rows.append(row)
    comparison = {"reason_code": "multi_database_comparison_insufficient"}
    meta_analysis, meta_report = _combined_meta_analysis(batch, record, card, targets, rows, reference, hartung_knapp=hartung_knapp)
    if len(estimates) >= 2:
        compatible = len({(row.estimand_sha256, row.unit, row.estimand_label) for row in estimates}) == 1
        distinct = (len({row.source_sha256 for row in estimates}) == len(estimates)
                    and len({row.database for row in estimates}) == len(estimates) and len(estimates) <= 6)
        if not compatible or not distinct:
            comparison = {"reason_code": "multi_database_estimates_incompatible"}
        else:
            unit = estimates[0].unit
            null = {"odds_ratio": 1.0, "risk_ratio": 1.0, "hazard_ratio": 1.0,
                    "risk_difference": 0.0, "mean_difference": 0.0, "beta": 0.0}.get(unit)
            if null is None:
                return dict(schema_version="easyicu.multi_database_answer/1", batch_id=card.batch_id,
                    scope="analysis_level", card_sha256=card.digest, reference=reference, targets=rows,
                    meta_analysis=meta_analysis, meta_analysis_report=meta_report,
                    heterogeneity={"reason_code": "multi_database_estimates_incompatible"})
            calculation = describe_estimate_heterogeneity([row.estimate for row in estimates], null_value=null)
            payload = dict(card_sha256=card.digest, inputs=[row.model_dump(mode="json") for row in estimates], result=calculation)
            digest = canonical_sha256(payload)
            evidence = EvidenceStore(batch.root / "comparison")
            evidence_id = "heterogeneity_" + digest[:24]
            prior = evidence.get(evidence_id)
            if prior and verified_run_evidence_path(batch.root / "comparison", prior) is None:
                raise MultiDatabaseError("multi_database_result_changed")
            item = prior or evidence.register_json(kind="log", description="Analysis-level database heterogeneity",
                evidence_id=evidence_id, filename=evidence_id + ".json", payload=payload)
            comparison = {**calculation, "evidence_id": item.evidence_id, "evidence_sha256": item.sha256}
    return dict(schema_version="easyicu.multi_database_answer/1", batch_id=card.batch_id,
                scope="analysis_level", card_sha256=card.digest, reference=reference, targets=rows, heterogeneity=comparison,
                meta_analysis=meta_analysis, meta_analysis_report=meta_report)


def _combined_meta_analysis(batch, record, card, targets, rows, reference, *, hartung_knapp: bool = False):
    """Seal the calculation, report and forest against this card and inputs."""
    inputs = []
    if reference and "primary" in reference and reference["primary"]["database"] not in {row["database"] for row in rows}:
        inputs.append(reference["primary"])
    for target, row in zip(targets, rows):
        value = dict(row.get("primary") or {}, database=row["database"], source_sha256=target.source_sha256)
        value["gate_passed"] = (value.get("gate_passed") is True and row["state"] == "completed"
                                and target.decision.status != "refusal" and target.review is not None)
        value["substitution_codes"] = [item.reason_code for item in target.decision.substitutions]
        value["definition_codes"] = list(target.definition_codes)
        inputs.append(value)
    design = record["design"]
    if "plan" in design and "context" in design:
        from easyicu.research_agent.schema import AnalysisPlan, ResearchContext

        identity = estimand_identity(AnalysisPlan.model_validate(design["plan"]), ResearchContext.model_validate(design["context"]))
    else:
        # Protocol implementations may supply their source identity directly;
        # never infer the reference from a majority of returned estimates.
        identity = design.get("estimand_sha256", "")
    calculation = pool_database_estimates(inputs, reference_estimand=identity, hartung_knapp=hartung_knapp)
    rendered = render_database_meta_analysis(calculation)
    payload = dict(card_sha256=card.digest, inputs=inputs, result=calculation, report=rendered)
    evidence = EvidenceStore(batch.root / "comparison")
    evidence_id = "meta_analysis_" + canonical_sha256(payload)[:24]
    prior = evidence.get(evidence_id)
    if prior and verified_run_evidence_path(batch.root / "comparison", prior) is None:
        raise MultiDatabaseError("multi_database_result_changed")
    item = prior or evidence.register_json(kind="log", description="Governed per-database meta-analysis",
        evidence_id=evidence_id, filename=evidence_id + ".json", payload=payload)
    for name, content in (("report.md", rendered["markdown"]), ("forest.svg", rendered["forest_svg"])):
        if content:
            artifact_id = evidence_id + ("_forest" if name.endswith("svg") else "_report")
            old = evidence.get(artifact_id)
            if old and verified_run_evidence_path(batch.root / "comparison", old) is None:
                raise MultiDatabaseError("multi_database_result_changed")
            artifact = old or evidence.register_text(kind="figure" if name.endswith("svg") else "log",
                description="Aggregate meta-analysis " + name, evidence_id=artifact_id,
                filename=artifact_id + (".svg" if name.endswith("svg") else ".md"), text=content)
            rendered["forest_evidence_id" if name.endswith("svg") else "report_evidence_id"] = artifact.evidence_id
    return {**calculation, "evidence_id": item.evidence_id, "evidence_sha256": item.sha256}, rendered
