"""Conversation adapter for quick proposals, explicit clicks and study handoff."""

from __future__ import annotations

from uuid import uuid4

from pydantic import ValidationError

from easyicu.research_agent.providers.client_trust import source_requires_clearance
from easyicu.webserver import provider_adapter, sources, study_contexts
from easyicu.webserver.demo_source_storage import public_demo_egress_scope
from easyicu.webserver.host_action_contracts import host_action_id
from easyicu.webserver.quick_answer import QuickApproval, QuickStop, execute_quick_answer
from easyicu.webserver.quick_answer.contracts import QuickProposal, StudyHandoff
from easyicu.webserver.quick_answer.preparation import prepare_registered_question
from easyicu.webserver.study_question_wording import normalize_question_text

from .contracts import PiCopilotError, PiToolResult, ToolExecutionContext, utc_now
from .quick_answer_store import QuickAnswerStore, QuickQuestionInput, error, public_attachment, source_record, stored_source


def submit_quick_question(context: ToolExecutionContext, params) -> dict:
    """The only model tool: intent in, host proposal out. Never execute here."""
    try:
        reading = QuickQuestionInput.model_validate(params).reading
    except ValidationError as exc:
        raise error("quick_reading_invalid") from exc
    if context.quick_answer_submit is None:
        raise error("quick_conversation_required")
    attachment = context.quick_answer_submit(context, reading)
    if isinstance(attachment, QuickProposal):
        return PiToolResult(
            status="ok", code=attachment.reason,
            summary=("Re-read the original question once and submit the corrected QuickReading. "
                     "Use registered concept ids that preserve the question's meaning and scope. "
                     "If you cannot resolve it, the host will offer the full study."),
            owner="easyicu.webserver.quick_answer",
            details={"reading_issues": [issue.model_dump(mode="json") for issue in attachment.reading_issues]},
            authority=context.session.binding.model_dump(mode="json"),
        ).model_dump(mode="json")
    if attachment is None:
        return PiToolResult(
            status="ok", code="quick_full_study_continues", summary="",
            owner="easyicu.webserver.quick_answer", details={},
            authority=context.session.binding.model_dump(mode="json"),
        ).model_dump(mode="json")
    proposal = attachment["proposal"]
    return PiToolResult(
        status="blocked" if proposal["status"] == "stopped" else "ok",
        code=proposal["reason"], summary="", owner="easyicu.webserver.quick_answer",
        details={"quick_answer": attachment},
        authority=context.session.binding.model_dump(mode="json"),
    ).model_dump(mode="json")


class QuickAnswerService:
    """Uses the session service's scope, provider, replay and study authorities."""

    def _quick_store(self) -> QuickAnswerStore:
        return QuickAnswerStore(self.store_path.with_suffix(".quick-answers"))

    def _quick_provider(self, record, database):
        if not source_requires_clearance(database):
            return None
        environment = self.provider_store.research_agent_environment(
            external_llm_opt_in=record.external_llm_opt_in,
            environ=self.gateway.environ, include_file=False,
        )
        environment["OPENAI_MODEL"] = record.conversation_provider.model
        try:
            client, _ = provider_adapter.build_research_agent_provider_client(
                {"provider": record.research_provider.provider, "external": True}, environ=environment,
            )
        except provider_adapter.ProviderAdapterError as exc:
            raise error("quick_qualifying_provider_required") from exc
        return client

    def _quick_study(self, record):
        # Shared Copilot clearance also protects read/reload and follows the
        # current provider declaration, including revocation since the card.
        from .tools import require_copilot_data_clearance

        require_copilot_data_clearance(ToolExecutionContext(
            session=record, data_handling=self._data_handling_for_session(record),
            note_credentialed_history=lambda: self._note_credentialed_history(record.session_id),
        ))
        if record.agent_mode != "research" or record.data_source_authorization.status in {"pending", "selection_in_progress"}:
            raise error("pi_session_data_source_confirmation_required")
        if self._stale_details(record).get("stale"):
            raise error("pi_session_authority_stale")
        study = study_contexts.get_context(record.binding.study_context_id) or {}
        if not study.get("data_source", {}).get("path") or study.get("active_job_id"):
            raise error("quick_bound_source_unavailable")
        return study

    def _transition_quick_question(self, context: ToolExecutionContext) -> str:
        """The saved question the data-source transition may offer as a quick answer.

        Only a study question that is, word for word, one message the
        researcher typed on the active branch: the step-one save may keep a
        part of a message, join several, or carry a confirmed model wording,
        and none of those is their question as asked. No text is classified
        here; the conversation's model reads the question and the quick-answer
        owner qualifies it. Empty means the transition offers nothing.
        """

        record = context.session
        if record.agent_mode != "research" or record.data_source_authorization.status in {"pending", "selection_in_progress"}:
            return ""
        study = study_contexts.get_context(record.binding.study_context_id) or {}
        question = str(study.get("question") or "")
        path = str((study.get("data_source") or {}).get("path") or "")
        if not question or not path or study.get("active_job_id"):
            return ""
        # A quick answer reads only a registered prepared export; a raw folder
        # chosen for the study would always be stopped, so none is offered.
        if sources.registered_source_identity(path) is None:
            return ""
        typed = {
            normalize_question_text(str(turn.get("text") or "")) for turn in context.user_turns
            if turn.get("message_origin") == "typed" and not turn.get("superseded")
        }
        return question if normalize_question_text(question) in typed else ""

    def _quick_turn_source(self, context):
        context.assert_authority_fresh()
        record = self._scoped_record(context.session.session_id, project_id=context.session.project_id)
        if record.binding != context.session.binding:
            raise error("pi_session_authority_stale")
        return record, self._quick_study(record)["data_source"]

    def _submit_quick_answer(self, context, reading, *, message_job_id):
        # The reading restates the turn's bound question: the typed message,
        # or the transition's saved question. A turn without one offers none.
        if not context.quick_question:
            raise error("quick_question_unbound")
        store = self._quick_store()
        with store.lock(context.session.session_id):
            if context.quick_reading_repair_finished:
                raise error("quick_reading_repair_exhausted")
            repair = context.quick_reading_to_repair
            # A repair that reaches the owner consumes the slot even if the
            # owner stops. Schema validation and a decline happen before here.
            context.quick_reading_repair_finished = repair is not None
            record, source = self._quick_turn_source(context)
            if repair is not None:
                registered = sources.registered_source_identity(source["path"])
                if registered is None or registered.to_dict() != repair[1]:
                    raise error("quick_source_changed")
            identity = "qa_" + uuid4().hex
            directory = store.path(record.session_id, identity).parent
            try:
                with public_demo_egress_scope(source["path"]):
                    proposal, prepared = prepare_registered_question(
                        reading, question=context.quick_question, export_path=source["path"],
                        database=source["database"], output_dir=directory / "prepared",
                        provider=self._quick_provider(record, source["database"]),
                    )
            except QuickStop as exc:
                raise error(exc.code) from exc
            record, current_source = self._quick_turn_source(context)
            registered = sources.registered_source_identity(current_source["path"])
            bound_source = proposal.expand.source
            if (current_source != source or registered is None or registered.to_dict() != bound_source
                    or (repair is not None and repair[1] != bound_source)):
                raise error("quick_source_changed")
            if proposal.reading_issues and repair is None:
                context.quick_reading_to_repair = (reading, bound_source)
                return proposal
            # An unresolved reading after repair carries its own explanation
            # and expand action, including on the data-source transition.
            if context.quick_question_offered and proposal.status == "full_study" and not proposal.reading_issues:
                return None
            row = {
                "quick_answer_id": identity, "session_id": record.session_id,
                "project_id": record.project_id, "message_job_id": message_job_id,
                "binding": record.binding.model_dump(mode="json"),
                "proposal": proposal.model_dump(mode="json"), "prepared": source_record(prepared),
                "answer": None, "created_at": utc_now(),
            }
            store.write(row)
        return public_attachment(row)

    def _finish_quick_reading_repair(self, context, *, message_job_id):
        # A re-read can decline or fail schema validation before reaching the
        # preparation owner. Give the original unresolved reading its handoff.
        repair = context.quick_reading_to_repair
        if repair is not None and not context.quick_reading_repair_finished:
            try:
                self._submit_quick_answer(context, repair[0], message_job_id=message_job_id)
            except PiCopilotError:
                # Source confirmation already has its host acknowledgement.
                # A direct question needs the typed stop from the message job.
                if not context.quick_question_offered:
                    raise

    def _quick_row(self, record, quick_answer_id):
        study = self._quick_study(record)
        row = self._quick_store().read(record.session_id, record.project_id, quick_answer_id)
        active = self.replay_store.snapshot(session_id=record.session_id, project_id=record.project_id, limit=400)
        # The card computes its own pinned reading on its pinned prepared
        # source (checked below and again at execution), so a later revision
        # of the same study -- such as the assistant saving the study in its
        # automatic follow-up turn -- does not change its answer. It is stale
        # only in another study or once its turn left the active conversation.
        if (row["binding"].get("study_context_id") != record.binding.study_context_id
                or row["message_job_id"] not in {turn["job_id"] for turn in active["turns"]}):
            raise error("quick_card_stale")
        identity = sources.registered_source_identity(study["data_source"]["path"])
        if identity is None or identity.to_dict() != row["proposal"]["expand"]["source"]:
            raise error("quick_source_changed")
        return study, row

    def approve_quick_answer(self, session_id, *, project_id, quick_answer_id, card_sha256, accept_substitutions=False):
        record = self._scoped_record(session_id, project_id=project_id)
        store = self._quick_store()
        with store.lock(session_id):
            study, row = self._quick_row(record, quick_answer_id)
            proposal = QuickProposal.model_validate(row["proposal"])
            if proposal.card is None or card_sha256 != proposal.card.sha256:
                raise error("quick_approval_mismatch")
            approval = QuickApproval(card_sha256=card_sha256, accept_substitutions=accept_substitutions)
            if row.get("approval") and row["approval"] != approval.model_dump(mode="json"):
                raise error("quick_approval_mismatch")
            if not row.get("answer"):
                with public_demo_egress_scope(study["data_source"]["path"]):
                    answer = execute_quick_answer(
                        proposal, approval=approval, source=stored_source(row["prepared"]),
                        provider=self._quick_provider(record, study["data_source"]["database"]),
                        run_dir=store.path(session_id, quick_answer_id).parent / "execution",
                    )
                if answer.reason in {"quick_approval_mismatch", "quick_substitution_approval_required", "quick_qualifying_provider_required"}:
                    raise error(answer.reason)
                row.update(answer=answer.model_dump(mode="json"), approval=approval.model_dump(mode="json"))
                store.write(row)
            self._record_quick_action(record, "approve_quick_answer", quick_answer_id)
            return {"ok": True, "quick_answer": store.public(row)}

    def expand_quick_answer(self, session_id, *, project_id, quick_answer_id):
        record = self._scoped_record(session_id, project_id=project_id)
        store = self._quick_store()
        with store.lock(session_id):
            study, row = self._quick_row(record, quick_answer_id)
            handoff = StudyHandoff.model_validate((row.get("answer") or row["proposal"])["expand"])
            updated = study_contexts.upsert_context(
                {"id": study["id"], "question": handoff.reading.question,
                 "quick_answer_handoff": handoff.model_dump(mode="json")},
                expected_revision=study["revision"], require_revision=True, active=False,
                lifecycle_write=False, _server_quick_answer_handoff_write=True,
            )
            record.binding = self._binding_for_context(updated, run_id=self._latest_run_id(updated["id"]))
            self._save_record(record)
            row.update(handoff_study_context_id=updated["id"], binding=record.binding.model_dump(mode="json"))
            store.write(row)
            self._record_quick_action(record, "expand_into_full_study", quick_answer_id)
            return {"ok": True, "study_context_id": updated["id"], "study_context_revision": updated["revision"],
                    "expand": handoff.model_dump(mode="json"), "quick_answer": store.public(row)}

    def _record_quick_action(self, record, action, quick_answer_id):
        self.replay_store.record_host_action(
            session_id=record.session_id, project_id=record.project_id,
            action_id=host_action_id(record.session_id, action, quick_answer_id),
            action_code=action, action_key=quick_answer_id, status="done",
        )
        record.conversation_updated_at = utc_now()
        self._save_record(record)

    def _quick_attachments(self, record, turns):
        return self._quick_store().attachments(record.session_id, record.project_id, {
            value for turn in turns for value in (turn["job_id"], turn.get("action_key")) if value
        })
