"""Host projection of sealed primary products, plus comparison evidence."""
from __future__ import annotations

import csv
from pathlib import Path

from pydantic import Field, model_validator

from easyicu.research_agent.authority.evidence_store import EvidenceStore
from easyicu.research_agent.authority.runtime_artifacts import load_run_artifact_authority, verified_run_evidence_path
from easyicu.research_agent.authority.typed_binding import _resolve_typed_input_evidence
from easyicu.research_agent.canonical_json import canonical_sha256
from easyicu.research_agent.execution.output_files import _headline_association_row
from easyicu.research_agent.methods.sensitivity import describe_estimate_heterogeneity
from .research_append_source import _approved_source, contained
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
    evidence_id: Code
    evidence_sha256: Digest
    substitution_codes: tuple[Code, ...] = Field(default=(), max_length=64)

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
    primary = [step for step in plan.steps if step.planned_analysis_role == "primary"
               and "table:adjusted_association_estimates" in step.expected_outputs]
    if len(primary) != 1:
        raise MultiDatabaseError("multi_database_primary_estimate_unavailable")
    ledger = load_run_artifact_authority(run)
    ref, failure = _resolve_typed_input_evidence(input_name="table:adjusted_association_estimates", plan=plan,
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
    chosen = _headline_association_row(rows)
    requirements = primary[0].model_requirements
    if (chosen is None or chosen.get("fit_status") != "fitted" or len(requirements) != 1
            or chosen.get("requirement_id") != requirements[0].requirement_id
            or chosen.get("outcome") != requirements[0].outcome
            or chosen.get("exposure") != requirements[0].exposure_source):
        raise MultiDatabaseError("multi_database_primary_estimate_unavailable")
    return SealedEstimate(database=target.target.database, run_id=source.run_id,
        source_sha256=target.source_sha256, plan_sha256=review.plan_sha256,
        estimand_sha256=estimand_identity(plan), estimand_label="adjusted_association",
        unit=chosen["effect_scale"], population_n=int(chosen["n"]),
        estimate=float(chosen["estimate"]), ci_low=float(chosen["ci_low"]), ci_high=float(chosen["ci_high"]),
        evidence_id=ref.evidence_id, evidence_sha256=record["sha256"],
        substitution_codes=tuple(row.reason_code for row in target.decision.substitutions)).model_dump(mode="json")


def combined_answer(batch):
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
                scope="analysis_level", card_sha256=card.digest, reference=reference, targets=rows, heterogeneity=comparison)
