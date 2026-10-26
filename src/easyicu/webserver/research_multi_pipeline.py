"""Concrete adapter to the normal study handoff, planner and review resume."""
from __future__ import annotations

from pathlib import Path

from easyicu.databases.research_approval import REVIEW_FIELD, DatabaseCapabilityReview
from easyicu.research_agent.canonical_json import canonical_sha256
from easyicu.research_agent.schema import AnalysisPlan, ResearchContext
from . import agent_pipeline_runs, agent_runs, dataio, settings, study_contexts
from .agent_review_recovery import _write
from .demo_source_storage import egress_source_database, public_demo_egress_scope
from .pi_copilot.contracts import plan_approval_allowed
from .pi_copilot.run_authority import research_pipeline_workspace
from .research_append_source import read_json
from .research_followup_contracts import CompletedResult
from .research_multi_contracts import DatabaseTarget, MultiDatabaseError, PreparedTarget, failure_code
from .research_multi_plan import compile_same_study, require_same_design, target_decision
from .research_multi_review import read_review
from .research_multi_sources import egress_grant, select_source
from .research_run_submission import provider_environment_for_agent_run


def source_result(design):
    return CompletedResult(design["run_id"], design["source_sha256"],
        AnalysisPlan.model_validate(design["plan"]), ResearchContext.model_validate(design["context"]), frozenset(), ())


class PipelineDatabaseOwner:
    def __init__(self, service, request, root):
        self.service, self.request, self.root = service, request, Path(root)

    def _provider(self, source):
        self.service.host_action_session(self.request.session_id, project_id=self.request.project_id,
                                         study_context_id=self.request.study_context_id)
        record = self.service._scoped_record(self.request.session_id, project_id=self.request.project_id)
        if not record.external_llm_opt_in or record.agent_mode != "research":
            raise MultiDatabaseError("multi_database_opt_in_required")
        selected = record.research_provider
        account_environment = None
        if selected.credential_source == "codex_user_auth":
            from . import codex_account_sessions

            account_environment = codex_account_sessions.environment_for_binding(
                str(selected.account_session_sha256 or ""), model=selected.model)
        environment = provider_environment_for_agent_run(credential_source=selected.credential_source,
            engine="research_agent_pipeline", run_type="full", external_llm_opt_in=True,
            llm_provider=selected.provider, account_environment=account_environment)
        with public_demo_egress_scope(source["path"]):
            provider = agent_runs.resolve_agent_provider_config(run_type="full", llm_provider=selected.provider,
                external_llm_opt_in=True, ai_enabled=bool(settings.load_settings().get("ai_enabled")),
                environ=environment, database=egress_source_database(source["path"], source["database"]))
        digest = canonical_sha256({"provider": provider, "selection": selected.model_dump(mode="json")})
        handling = (environment or {}).get("EASYICU_PROVIDER_DATA_HANDLING", "unspecified")
        return provider, environment, selected.credential_source, digest, handling

    def _plan(self, study, source, *, job, receipt):
        if receipt.exists():
            return read_json(receipt)
        marker = receipt.with_suffix(".started.json")
        if marker.exists():
            raise MultiDatabaseError("multi_database_preparation_interrupted")
        if job is None:
            raise MultiDatabaseError("multi_database_job_required")
        provider, environment, credential_source, _, _ = self._provider(source)
        from .research_multi_extraction import prepare_source

        study, source = prepare_source(study, source, job)
        provider, environment, credential_source, _, _ = self._provider(source)
        dataio.validate_research_pipeline_source(source["path"], database=source["database"])
        root = research_pipeline_workspace().project_root(study["id"])
        prepared_study = []
        runner = agent_pipeline_runs.make_research_pipeline_run_runner(
            export_path=source["path"], study_context=study, project_root=str(root),
            provider=provider, provider_environment=environment, credential_source=credential_source,
            budget_mode="full_reviewed", on_prepared_study=prepared_study.append)
        study = prepared_study[0] if prepared_study else study
        study_contexts.handoff_context(study["id"], active_job_id=job.id, current_stage="analyze",
                                      last_route="agent", expected_revision=study["revision"])
        _write(marker, {"study_id": study["id"], "job_id": job.id})
        try:
            result = runner(job)
        finally:
            study_contexts.clear_active_job_if(study["id"], job.id, current_stage="review", last_route="agent")
        pending = agent_pipeline_runs.pending_review(result.get("run_id"))
        if not pending or not plan_approval_allowed(pending) or pending.get("budget_mode") != "full_reviewed":
            raise MultiDatabaseError((result.get("gate") or {}).get("reason") or "multi_database_plan_unavailable")
        locator = {"run_id": result["run_id"], "wrapper": result["project_dir"], "study_id": study["id"]}
        _write(receipt, locator)
        return locator

    def design(self, request, job):
        study = study_contexts.get_context(request.study_context_id)
        if study is None or study.get("active_job_id"):
            raise MultiDatabaseError("study_context_active_job_conflict")
        if request.source_run_id:
            row = self.service._research_run_row(request.project_id, request.source_run_id)
            if row["study_id"] != study["id"]:
                raise MultiDatabaseError("multi_database_study_mismatch")
            locator = dict(wrapper=row["project_dir"], run_id=request.source_run_id, study_id=study["id"])
        else:
            locator = self._plan(study, study["data_source"], job=job, receipt=self.root / "design_run.json")
            study = study_contexts.get_context(study["id"])
        binding, context, plan, checkpoint, _ = read_review(locator["wrapper"], locator["run_id"], study)
        if checkpoint.state == "completed":
            from .research_followup_source import load_completed_result

            source = load_completed_result(Path(locator["wrapper"]), run_id=locator["run_id"], study_id=study["id"])
            digest = source.source_sha256
        elif checkpoint.state == "pending":
            digest = canonical_sha256(binding.model_dump(mode="json"))
        else:
            raise MultiDatabaseError("multi_database_design_review_required")
        return {**locator, "source_sha256": digest, "source_state": checkpoint.state,
                "binding": binding.model_dump(mode="json"), "study": study,
                "context": context.model_dump(mode="json"), "plan": plan.model_dump(mode="json")}

    def verify_design(self, design):
        study = study_contexts.get_context(design["study_id"])
        if study is None:
            raise MultiDatabaseError("multi_database_design_changed")
        binding, _, _, checkpoint, _ = read_review(design["wrapper"], design["run_id"], study)
        own_approval = self.root / "approval.json"
        advanced_here = (design["source_state"] == "pending" and own_approval.exists()
            and bool(checkpoint.approved_decisions) and all(row.get("note") == "multi_database_combined_approval"
                and row.get("reviewer") == read_json(own_approval)["reviewer"] for row in checkpoint.approved_decisions))
        if binding.model_dump(mode="json") != design["binding"] or (checkpoint.state != design["source_state"] and not advanced_here):
            raise MultiDatabaseError("multi_database_design_changed")
        if design["source_state"] == "completed":
            from .research_followup_source import load_completed_result

            current = load_completed_result(Path(design["wrapper"]), run_id=design["run_id"], study_id=study["id"])
            if current.source_sha256 != design["source_sha256"]:
                raise MultiDatabaseError("multi_database_design_changed")

    def reference_target(self, design):
        """An unexecuted source plan joins the same approval; a result does not."""
        if design["source_state"] != "pending":
            return None
        from . import sources
        from .research_multi_contracts import ReviewedRun

        selected = design["study"]["data_source"]
        identity = sources.registered_source_identity(selected["path"])
        if identity is None:
            raise MultiDatabaseError("multi_database_source_unavailable")
        target = DatabaseTarget(database=selected["database"], source_id=identity.registration_id)
        source, digest = select_source(target)
        _, _, _, provider_sha, handling = self._provider(source)
        grant = egress_grant(source, digest, provider_sha256=provider_sha, data_handling=handling)
        decision = target_decision(source_result(design), target.database)
        if decision.status == "refusal":
            raise MultiDatabaseError("study_capability_refused")
        _write(self.root / (target.database + ".run.json"),
               {key: design[key] for key in ("wrapper", "run_id", "study_id")})
        return PreparedTarget(target=target, decision=decision, source_sha256=digest,
                              review=ReviewedRun.model_validate(design["binding"]), egress=grant)

    def _child_study(self, design, source, batch_id):
        identity = "multi_" + canonical_sha256([batch_id, source["id"]])[:32]
        existing = study_contexts.get_context(identity)
        if existing is not None:
            return existing
        parent = design["study"]
        patch = {key: parent[key] for key in study_contexts._SCIENTIFIC_CONFIGURATION_FIELDS if key in parent}
        # Source-bound confirmations and trial approvals cannot migrate.
        for key in ("cohort_eligibility_authority", "literature_authority", "target_trial_design", "mentioned_objects"):
            patch.pop(key, None)
        patch.update(id=identity, title=parent.get("title", "") or "Multi database study",
                     data_source={key: source[key] for key in ("path", "database", "label") if key in source})
        result = study_contexts.upsert_context(patch, active=False, _server_skill_attachments_write=True,
            _server_concept_selection_authority_write=True, _server_question_wording_write=True,
            _server_design_variance_basis_write=True)
        return result

    def prepare(self, design, target, *, batch_id, job):
        original = source_result(design)
        decision = target_decision(original, target.database)
        digest = None
        try:
            if job is not None and job.is_cancel_requested():
                raise MultiDatabaseError("multi_database_cancelled")
            source, digest = select_source(target)
            target = DatabaseTarget(database=target.database, source_id=source["id"])
            if decision.status == "refusal":
                return PreparedTarget(target=target, decision=decision, source_sha256=digest, stop_code="study_capability_refused")
            _, _, _, provider_sha, handling = self._provider(source)
            grant = egress_grant(source, digest, provider_sha256=provider_sha, data_handling=handling)
            child = self._child_study(design, source, batch_id)
            locator = self._plan(child, source, job=job, receipt=self.root / (target.database + ".run.json"))
            child = study_contexts.get_context(child["id"])
            binding, context, actual, _, run = read_review(locator["wrapper"], locator["run_id"], child)
            proposed, expected, decision = compile_same_study(original, context)
            require_same_design(actual, expected)
            if (context.target_outcome != proposed.target_outcome or context.endpoint != proposed.endpoint
                    or context.primary_exposure != proposed.primary_exposure):
                raise MultiDatabaseError("multi_database_design_changed")
            review = read_json(run / "scientific_plan_review.json")
            companion = DatabaseCapabilityReview.model_validate(review["facts"][REVIEW_FIELD])
            if companion.decision != decision or companion.unapplied_substitutions:
                raise MultiDatabaseError("capability_substitution_review_mismatch")
            bands = proposed.cohort.provenance.get("database_capability_binding", {}).get("binned_covariates", {})
            return PreparedTarget(target=target, decision=decision, source_sha256=digest, review=binding, egress=grant,
                                  definition_codes=tuple("binned_covariate:" + name for name in sorted(bands)))
        except Exception as exc:
            code = failure_code(exc, "multi_database_plan_unavailable")
            return PreparedTarget(target=target, decision=decision, source_sha256=digest, stop_code=code)

    def verify(self, design, target):
        source, digest = select_source(target.target)
        _, _, _, provider_sha, handling = self._provider(source)
        if digest != target.source_sha256 or egress_grant(source, digest, provider_sha256=provider_sha,
                                                        data_handling=handling) != target.egress:
            raise MultiDatabaseError("multi_database_egress_changed")
        locator = read_json(self.root / (target.target.database + ".run.json"))
        study = study_contexts.get_context(target.review.study_context_id)
        binding, _, _, checkpoint, _ = read_review(locator["wrapper"], target.review.run_id, study)
        if checkpoint.state != "pending" and not (self.root / "approval.json").exists():
            raise MultiDatabaseError("multi_database_review_changed")
        if binding != target.review or target_decision(source_result(design), target.target.database) != target.decision:
            raise MultiDatabaseError("multi_database_review_changed")

    def execute(self, target, *, reviewer, job, recovering):
        source, _ = select_source(target.target)
        _, environment, _, _, _ = self._provider(source)
        locator = read_json(self.root / (target.target.database + ".run.json"))
        study = study_contexts.get_context(target.review.study_context_id)
        _, _, _, checkpoint, _ = read_review(locator["wrapper"], target.review.run_id, study)
        if checkpoint.state == "completed":
            return {}
        if checkpoint.state not in {"pending", "approved_pending_execution"}:
            # Existing single-run recovery refuses replaying an interrupted fit.
            raise MultiDatabaseError("multi_database_execution_interrupted")
        if checkpoint.state == "pending":
            pending = agent_pipeline_runs.pending_review(target.review.run_id)
            if not pending or not plan_approval_allowed(pending):
                raise MultiDatabaseError("multi_database_review_changed")
        study_contexts.handoff_context(study["id"], active_job_id=job.id, current_stage="analyze",
                                      last_route="agent", expected_revision=study["revision"])
        try:
            with public_demo_egress_scope(study["data_source"]["path"]):
                result = agent_pipeline_runs.resume_research_pipeline(run_id=target.review.run_id, study_context_id=study["id"],
                    decision="approved", reviewer=reviewer, note="multi_database_combined_approval", job=job,
                    current_study_context=study, provider_environment=environment,
                    require_recorded_decision=checkpoint.state != "pending")
            try:
                from .research_followup_source import load_completed_result

                load_completed_result(Path(locator["wrapper"]), run_id=target.review.run_id, study_id=study["id"])
            except (ValueError, OSError, KeyError) as exc:
                code = (result.get("gate") or {}).get("reason")
                raise MultiDatabaseError(code or failure_code(exc, "multi_database_execution_failed")) from exc
            return result
        finally:
            study_contexts.clear_active_job_if(study["id"], job.id, current_stage="review", last_route="agent")

    def result(self, target):
        from .research_multi_results import sealed_primary_estimate

        locator = read_json(self.root / (target.target.database + ".run.json"))
        return sealed_primary_estimate(Path(locator["wrapper"]), target)

    def reference_result(self, design):
        if design["source_state"] != "completed":
            return None
        from types import SimpleNamespace
        from .research_multi_contracts import ReviewedRun
        from .research_multi_results import sealed_primary_estimate

        self.verify_design(design)
        original = source_result(design)
        # This is a read of an already-approved result, with no egress grant.
        reference = SimpleNamespace(target=DatabaseTarget(database=original.context.cohort.database),
            source_sha256=design["binding"]["input_sha256"],
            review=ReviewedRun.model_validate(design["binding"]),
            decision=target_decision(original, original.context.cohort.database))
        return sealed_primary_estimate(Path(design["wrapper"]), reference)
