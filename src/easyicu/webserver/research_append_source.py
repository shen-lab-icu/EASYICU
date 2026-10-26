"""Read-only admission of a completed run's sealed inputs and approval.

Completion retains a minimal append seal before discarding provider recovery.
The seal contains no provider credentials and grants no execution approval.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from easyicu.research_agent.authority.baseline_association_runtime import BaselineAssociationRuntimeAuthority
from easyicu.research_agent.authority.evidence_store import sha256_of_file
from easyicu.research_agent.authority.plan_review import PlanReviewAuthority
from easyicu.research_agent.authority.run_input import load_verified_run_input_capsule
from easyicu.research_agent.authority.runtime_artifacts import load_run_artifact_authority
from easyicu.research_agent.authority.typed_binding import _resolve_typed_input_evidence
from easyicu.research_agent.canonical_json import canonical_sha256
from easyicu.research_agent.orchestration.human_review_checkpoint import HumanReviewCheckpoint
from easyicu.research_agent.planning.scientific_review import PlanScientificReview
from easyicu.research_agent.schema import AnalysisPlan, ResearchContext

SEAL_NAME = "result_append_source.json"


def read_json(path: Path):
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 16 * 1024 * 1024:
        raise ValueError("Missing or unsafe append authority file.")
    return json.loads(path.read_bytes())


def contained(root: Path, relative: str) -> Path:
    path = Path(relative)
    if path.is_absolute() or not path.parts or any(part in {"..", "."} for part in path.parts):
        raise ValueError("Append input escapes its run.")
    current = root
    for part in path.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError("Append input is a symbolic link.")
    current.resolve(strict=True).relative_to(root.resolve(strict=True))
    return current


def _approved_source(run_dir: Path):
    # Completed approvals do not expire when a later, independently approved
    # run cites them. Still verify every checkpoint digest and approved request.
    checkpoint = HumanReviewCheckpoint.model_validate(read_json(run_dir / "human_review_checkpoint.json"))
    if checkpoint.state != "completed" or not checkpoint.execution_start_receipt or not checkpoint.requests:
        raise ValueError("An append requires a completed, approved source run.")
    decisions = {row["review_id"]: row for row in checkpoint.approved_decisions}
    if set(decisions) != {request.review_id for request in checkpoint.requests}:
        raise ValueError("Source approval is incomplete.")
    plans = []
    for request in checkpoint.requests:
        decision = decisions[request.review_id]
        if decision["decision"] != "approved" or decision["authority_sha256"] != request.authority_sha256:
            raise ValueError("Source approval differs from the reviewed request.")
        packet = PlanReviewAuthority.model_validate(request.payload["plan_review_authority"])
        plans.append(AnalysisPlan.model_validate(packet.model_dump(mode="json")["plan_payload"]))
    if any(plan != plans[0] for plan in plans) or plans[0] != AnalysisPlan.model_validate(checkpoint.plan_handoff["plan"]):
        raise ValueError("Source approved plans disagree.")
    capsule_path = run_dir / "run_input_capsule.json"
    if sha256_of_file(capsule_path) != checkpoint.run_input_capsule_sha256:
        raise ValueError("Source input capsule changed.")
    capsule = read_json(capsule_path)
    verified = load_verified_run_input_capsule(run_dir=run_dir, scientific_identity=capsule["scientific_identity"])
    context = ResearchContext.model_validate_json(verified.context_evidence_path.read_bytes())
    return checkpoint, plans[0], context, verified


@dataclass(frozen=True)
class AppendSource:
    wrapper: Path
    run_dir: Path
    seal: dict[str, Any]
    plan: AnalysisPlan
    context: ResearchContext
    baseline: BaselineAssociationRuntimeAuthority
    review: PlanScientificReview

    @property
    def digest(self):
        return self.seal["source_run_sha256"]

    def verify(self):
        fresh = load_append_source(self.wrapper)
        if fresh.digest != self.digest:
            raise ValueError("The source run changed after the append was offered.")
        return fresh

    def earlier_export_notice(self, *, changed=False):
        prefix = "The registered source has changed. " if changed else ""
        return prefix + f"This addition uses the earlier export sealed by run {self.seal['run_id']}."


def seal_completed_source(*, wrapper: Path, run_dir: Path, study, baseline, prepared_package_binding):
    checkpoint, plan, context, verified = _approved_source(run_dir)
    status = read_json(run_dir / "run_status.json")
    gates = status.get("gates", {})
    if gates.get("execution_complete") is not True or gates.get("analysis_validated") is not True:
        raise ValueError("Source analysis has not passed its execution checks.")
    baseline = BaselineAssociationRuntimeAuthority.model_validate_json(json.dumps(baseline))
    baseline = baseline.seal_adjustment_roster(plan)
    baseline.validate_plan(plan)
    review = PlanScientificReview.model_validate(read_json(wrapper / "scientific_plan_review.json"))
    if not review.approval_allowed or review.plan_sha256 != canonical_sha256(plan.model_dump(mode="json")):
        raise ValueError("The source scientific review is not bound to its plan.")
    ledger = load_run_artifact_authority(run_dir)
    if ledger is None:
        raise ValueError("Source execution ledger is absent.")
    records = verified.evidence_records
    products = {}
    for step in plan.steps:
        for key in step.expected_outputs:
            ref, failure = _resolve_typed_input_evidence(
                input_name=key, plan=plan, evidence_records=list(records.values()),
                per_step_records=ledger["per_step_records"], run_dir=run_dir,
            )
            if failure is None and ref is not None:
                products[key] = records[ref.evidence_id]
    if "artifact:analysis_cohort" not in products:
        raise ValueError("Source analysis cohort evidence is absent.")
    files = {name: sha256_of_file(contained(run_dir, name)) for name in (
        "human_review_checkpoint.json", "run_input_capsule.json", "run_status.json", "manifest.json",
        verified.capsule.cohort_relative_path,
    )}
    body = dict(
        schema_version="easyicu.result_append_source/1", study_id=study["id"], run_id=checkpoint.run_id,
        source_plan=plan.model_dump(mode="json"), source_context=context.model_dump(mode="json"),
        baseline=baseline.model_dump(mode="json"), review=review.model_dump(mode="json"),
        data_source=dict(study.get("data_source") or {}), prepared_package_binding=prepared_package_binding,
        products=products, files=files,
    )
    body["source_run_sha256"] = canonical_sha256(body)
    path = wrapper / SEAL_NAME
    if path.exists():
        if read_json(path) != body:
            raise ValueError("A completed source seal is immutable.")
    else:
        from easyicu.webserver.agent_review_recovery import _write

        _write(path, body)
    return load_append_source(wrapper)


def load_append_source(wrapper: Path) -> AppendSource:
    wrapper = Path(wrapper)
    if wrapper.is_symlink() or wrapper.parent.is_symlink():
        raise ValueError("Unsafe source wrapper.")
    seal = read_json(wrapper / SEAL_NAME)
    if seal.get("schema_version") != "easyicu.result_append_source/1" or canonical_sha256(
        {key: value for key, value in seal.items() if key != "source_run_sha256"}
    ) != seal.get("source_run_sha256"):
        raise ValueError("Source append seal changed.")
    run_id = seal["run_id"]
    if Path(run_id).name != run_id:
        raise ValueError("Unsafe source run id.")
    run_dir = contained(wrapper, "pipeline/" + run_id)
    for relative, digest in seal["files"].items():
        if sha256_of_file(contained(run_dir, relative)) != digest:
            raise ValueError("A source input or result changed.")
    _, plan, context, verified = _approved_source(run_dir)
    if plan.model_dump(mode="json") != seal["source_plan"] or context.model_dump(mode="json") != seal["source_context"]:
        raise ValueError("Source scientific coordinates changed.")
    from easyicu.research_agent.authority.runtime_artifacts import verified_run_evidence_path

    for record in seal["products"].values():
        if verified.evidence_records.get(record["evidence_id"]) != record or verified_run_evidence_path(run_dir, record) is None:
            raise ValueError("Source product evidence changed.")
    baseline = BaselineAssociationRuntimeAuthority.model_validate_json(json.dumps(seal["baseline"]))
    baseline.validate_plan(plan)
    return AppendSource(wrapper, run_dir, seal, plan, context, baseline, PlanScientificReview.model_validate(seal["review"]))


def retain_completed_append_source(*, wrapper_dir, projection):
    """Completion hook: retain source authority, never reopen a finished run."""
    from easyicu.webserver.agent_review_recovery import WebReviewRecoveryError, load_recovery_seed

    try:
        seed = load_recovery_seed(wrapper_dir)
    except WebReviewRecoveryError:
        return
    if seed is None:
        return
    baseline = seed.pipeline_config.get("current_case_scientific_runtime_authority")
    if not baseline or baseline.get("authority_kind") != "baseline_adjusted_association":
        return
    run_dir = Path(wrapper_dir) / "pipeline" / str(projection["run_id"])
    try:
        seal_completed_source(wrapper=Path(wrapper_dir), run_dir=run_dir, study=seed.study,
                              baseline=baseline, prepared_package_binding=seed.prepared_package_binding)
    except (ValueError, OSError, KeyError):
        # Failed or unsupported sources remain available through their owner.
        return


def registered_source_changed(source: AppendSource, study) -> bool:
    """Compare the registry and extraction binding; execution still uses copies."""
    from easyicu.webserver import dataio, sources

    old = source.seal["data_source"]
    current = study.get("data_source") or {}
    if any(old.get(key) != current.get(key) for key in ("path", "database", "source_id")):
        return True
    if not old.get("path") or not source.seal["prepared_package_binding"]:
        return True
    try:
        sources.resolve_registered_export(current["path"])
        dataio.validate_research_pipeline_source(current["path"], database=current.get("database"),
                                                expected_binding=source.seal["prepared_package_binding"])
    except (ValueError, OSError, sources.RegisteredExportSelectionError):
        return True
    return False
