"""Small synthetic, approved baseline run for append lifecycle checks."""
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from easyicu.research_agent.authority.baseline_association_runtime import build_baseline_authority
from easyicu.research_agent.authority.evidence_store import EvidenceStore, sha256_of_file
from easyicu.research_agent.authority.run_input import RunInputCapsule
from easyicu.research_agent.authority.plan_scope import _serializable_plan_scientific_scope_signature
from easyicu.research_agent.canonical_json import canonical_sha256
from easyicu.research_agent.contracts.exposure_group_rules import EXPOSURE_GROUP_LABELS_KEY
from easyicu.research_agent.execution.runners.baseline_association_executor import run_baseline_step
from easyicu.research_agent.orchestration.human_review_checkpoint import HumanReviewCheckpoint, write_checkpoint
from easyicu.research_agent.orchestration.workflow import (
    HumanReviewDecision, _human_review_decision_record, human_review_requests_for_plan,
)
from easyicu.research_agent.schema import AnalysisStep, CohortDescriptor, ConceptDescriptor, ResearchContext, UserPreferences
from easyicu.research_agent.planning.scientific_review import PlanScientificReview
from easyicu.research_agent.schema import AnalysisPlan


def source_plan():
    return AnalysisPlan(
        research_question="Synthetic marker and synthetic outcome association",
        steps=[AnalysisStep(step_id="original", intent="Estimate the prespecified association",
                            method="adjusted_association_models", inputs=["marker", "outcome"],
                            planned_analysis_role="primary", expected_outputs=["table:original"])],
    )


def approvable_review(plan):
    return PlanScientificReview(
        status="analysis_only", approval_allowed=True, top_journal_candidate=False,
        score=90, dimension_scores={}, findings=[], context_sha256="a" * 64,
        plan_sha256=canonical_sha256(plan.model_dump(mode="json")),
        literature_sha256="b" * 64, figure_strategy_sha256="c" * 64,
        generated_at="2026-10-23T00:00:00Z",
    )


def addition(method="complete_case_sensitivity", *, action=None):
    return AnalysisStep(
        step_id="added", intent="Add the requested synthetic check", method=method,
        scientific_action_id=action, inputs=["marker", "outcome", "table:original"],
        expected_outputs=["table:added"], planned_analysis_role="sensitivity",
    )


def tier_plan(tier):
    from easyicu.research_agent.planning.design_selection import ResearchDesignSelection

    candidates = [dict(
        design_id=name, analysis_type="association_study", estimand="Synthetic marker association",
        time_zero="Synthetic baseline", observation_window="Synthetic follow-up period",
        primary_method="Logistic regression" if name == "main" else "Descriptive cross-tabulation",
        required_variables=["marker", "outcome"],
        assumptions=["Marker precedes outcome"], novelty_positioning="Synthetic design comparison",
        figure_role="Display the synthetic estimate", supports="Observational association",
        cannot_prove="A causal effect", disposition=disposition,
        decision_reason="The prespecified synthetic design determines this choice",
    ) for name, disposition in (("main", "selected"), ("alternative", "rejected"))]
    return source_plan().model_copy(update={"analysis_type": "association_study", "design_selection": ResearchDesignSelection(
        execution_tier=tier, candidates=candidates,
    )})


def json_file(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def completed_source(tmp_path):
    wrapper = tmp_path / "study" / "run_source"
    run = wrapper / "pipeline" / "source"
    run.mkdir(parents=True)
    random = np.random.default_rng(35)
    size = 400
    marker = random.normal(size=size)
    frame = pd.DataFrame(dict(marker=marker, marker_group=(marker > 0).astype(int),
                              adjuster=random.normal(size=size), stratum=random.integers(0, 2, size),
                              outcome=random.binomial(1, 0.25 + 0.12 * (marker > 0), size)))
    frame.to_parquet(run / "cohort.parquet", index=False)
    terms = [dict(name="marker_group", role="exposure", coding="binary", transform="treatment_contrast",
                  levels=["0", "1"], reference_level="0"),
             dict(name="adjuster", role="covariate", coding="continuous", transform="identity"),
             dict(name="stratum", role="covariate", coding="binary", transform="treatment_contrast",
                  levels=["0", "1"], reference_level="0")]
    requirement = dict(requirement_id="main", outcome="outcome", outcome_type="binary",
                       method_family="statsmodels_logit_mle", exposure_source="marker_group", analysis_role="primary",
                       analysis_set="source_aware", covariates=["adjuster", "stratum"], model_terms=terms,
                       exposure_levels=["0", "1"], exposure_reference_level="0", primary_contrast_level="1")
    steps = [AnalysisStep(step_id=name, intent="Audit synthetic inputs", method=name,
                          expected_outputs=[product]) for name, product in (
        ("cohort_definition", "artifact:analysis_cohort"), ("table_one", "table:baseline"),
        ("measurement_audit", "table:measurement_audit"))]
    steps += [AnalysisStep(step_id="adjusted_association", intent="Estimate the synthetic association",
                            method="adjusted_association_models", planned_analysis_role="primary",
                            scientific_action_id="association.adjusted_association", model_requirements=[requirement],
                            inputs=["artifact:analysis_cohort", "marker_group", "outcome", "adjuster", "stratum"],
                            expected_outputs=["table:adjusted_association_estimates"]),
              AnalysisStep(step_id="absolute_risk_context", intent="Describe synthetic risks",
                           method="signed_baseline_absolute_risk", expected_outputs=["table:absolute_risk_context"])]
    plan = tier_plan("first_result").model_copy(update={"steps": steps})
    baseline = build_baseline_authority(protocol_content_sha256="b" * 64, exposure_column="marker_group",
                                        outcome_column="outcome", exposure_levels=("0", "1"),
                                        exposure_reference_level="0", primary_contrast_level="1",
                                        required_adjustment_columns=("adjuster", "stratum"), exposure_encoding="treatment_contrast")
    baseline = baseline.seal_adjustment_roster(plan)
    plan = baseline.bind_plan(plan)
    context = ResearchContext(
        research_question=plan.research_question,
        cohort=CohortDescriptor(cohort_name="synthetic", database="synthetic", n_stays=size, provenance={
            EXPOSURE_GROUP_LABELS_KEY: [{"variable": "marker_group", "parameters": {
                "concept": "marker", "window": None, "scale": "ordinal", "unmeasured": "exclude",
                "source_columns": {"value": "marker"}, "groups": [
                    {"id": "g1", "rule": {"summary": "value", "op": "<", "value": 0.0}},
                    {"id": "g2", "rule": "otherwise"},
                ],
            }}],
        }),
        variables=[ConceptDescriptor(name=name, role="other", dtype=str(dtype)) for name, dtype in frame.dtypes.items()],
        primary_exposure="marker_group", target_outcome="outcome", user_preferences=UserPreferences(),
    )
    evidence = EvidenceStore(run)
    context_path = run / "research_context.json"
    context_path.write_text(context.model_dump_json())
    evidence.register_file(kind="log", description="Synthetic source context", source_path=context_path, evidence_id="research_context")
    identity = {"cohort": {"sha256": sha256_of_file(run / "cohort.parquet"), "size_bytes": (run / "cohort.parquet").stat().st_size}}
    capsule = RunInputCapsule(scientific_identity=identity, scientific_identity_sha256=canonical_sha256(identity),
                              context_sha256=sha256_of_file(context_path), cohort_sha256=sha256_of_file(run / "cohort.parquet"),
                              initial_environment={})
    capsule_path = run / "run_input_capsule.json"
    capsule_path.write_text(capsule.model_dump_json())
    evidence.register_file(kind="log", description="Synthetic sealed input", source_path=capsule_path, evidence_id="run_input_capsule")
    records = []
    primary_table = None
    for step in plan.steps:
        out = run / "steps" / step.step_id
        out.mkdir(parents=True)
        if step.step_id == "cohort_definition":
            path = run / "cohort.parquet"
            summary = {"output_files": {step.expected_outputs[0]: path.name}}
        elif step.step_id in {"adjusted_association", "absolute_risk_context"}:
            summary = run_baseline_step(step_id=step.step_id, plan=plan, authority=baseline,
                                        runtime_projection_sha256="c" * 64, frame=frame, output_dir=out,
                                        primary_table=primary_table)
            path = out / next(iter(summary["output_files"].values()))
            if step.step_id == "adjusted_association":
                primary_table = pd.read_csv(path)
        else:
            path = out / (step.step_id + ".csv")
            pd.DataFrame([{"variable": "marker_group", "n": size, "missing": 0}]).to_csv(path, index=False)
            summary = {"output_files": {step.expected_outputs[0]: path.name}}
        rec = evidence.register_file(kind="table", description=step.intent, source_path=path, produced_by_step=step.step_id)
        records.append(dict(step_id=step.step_id, status="ok", evidence_ids=[rec.evidence_id], step_summary=summary,
                            analysis_request={"step": step.model_dump(mode="json")},
                            plan_scientific_signature=_serializable_plan_scientific_scope_signature(plan)))
    requests = human_review_requests_for_plan(findings=[], plan=plan, evidence=evidence, require_plan_review=True)
    checkpoint = HumanReviewCheckpoint.create(
        run_id="source", pipeline_config_sha256="d" * 64, environment_identity={}, llm_signature_sha256="e" * 64,
        run_input_capsule_sha256=sha256_of_file(capsule_path), capability_activation_sha256="f" * 64,
        runtime_capabilities=(), runtime_bundle=None, requests=requests,
        plan_handoff={"plan": plan.model_dump(mode="json")}, execution_coordinates={},
    )
    decisions = [HumanReviewDecision(review_id=r.review_id, authority_sha256=r.authority_sha256, decision="approved",
                                     reviewer="synthetic", decided_at=datetime.now(timezone.utc).isoformat()) for r in requests]
    wire = [d.model_dump(mode="json") for d in decisions]
    checkpoint = checkpoint.approved(decisions=wire, decision_records=[_human_review_decision_record(
        request=r, decision=d, reviewer_identity="synthetic") for r, d in zip(requests, decisions)], decision_sha256=canonical_sha256(wire))
    checkpoint = checkpoint.execution_started().transitioned("completed")
    write_checkpoint(run / "human_review_checkpoint.json", checkpoint)
    json_file(run / "manifest.json", {"run_id": "source", "per_step_records": records})
    json_file(run / "run_status.json", {"gates": {"execution_complete": True, "analysis_validated": True}})
    json_file(wrapper / "scientific_plan_review.json", approvable_review(plan).model_dump(mode="json"))
    return wrapper, run, plan, context, baseline
