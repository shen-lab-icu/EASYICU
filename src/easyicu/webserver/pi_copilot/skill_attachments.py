"""Conversation attachment admission, draft editing and governed turn loading."""

from __future__ import annotations

import json

from easyicu.extensions.contracts import ExtensionRegistryError
from easyicu.extensions.skill_attachments import attachment_payload, parse_attachments
from easyicu.research_agent.method_skills import METHOD_SKILLS
from easyicu.extensions.method_attachments import builtin_attachment, method_constraints
from easyicu.webserver import capabilities, study_contexts

from .contracts import PiCopilotError, PiSessionRecord


def _error(code: str) -> PiCopilotError:
    return PiCopilotError(code, code, status_code=409)


def validate_attachments(value: object, *, activation, registry, research: bool = True) -> list:
    """Admit the complete set against the frozen session; never omit a reference."""
    try:
        refs = parse_attachments(value)
        resolved = []
        frozen = {item.name: item for item in activation.skills}
        for ref in refs:
            if ref.source == "builtin":
                method_constraints([ref])
                if not research or not capabilities.capability_settings()["science_skills_enabled"]:
                    raise _error("skill_attachment_disabled")
                resolved.append(builtin_attachment(ref.id))
            else:
                skill = frozen.get(ref.id)
                if skill is None:
                    known = {row["name"] for row in registry.list_public()["skills"]}
                    raise _error("skill_attachment_disabled" if ref.id in known else "skill_attachment_unknown")
                if skill.disable_model_invocation or "conversation" not in skill.stages:
                    raise _error("skill_attachment_disabled")
                if skill.digest != ref.digest:
                    raise _error("skill_attachment_stale")
                registry.load_skill(name=ref.id, digest=ref.digest)
                resolved.append(ref.model_dump(mode="json"))
        method_constraints(resolved)
        return parse_attachments(resolved)
    except ExtensionRegistryError as exc:
        raise _error(exc.code) from exc


def load_attached_users(record, refs, registry) -> list[dict]:
    """Use _load_skill's digest-checked registry owner with an explicit user source.

    The source disambiguates installed names from shell-only built-ins such
    as web-prototype; their instructions cannot replace a user attachment.
    """
    loaded = []
    for ref in refs:
        if ref.source == "user":
            frozen = next((skill for skill in record.extension_activation.skills if skill.name == ref.id), None)
            if frozen is None or frozen.digest != ref.digest:
                raise _error("skill_attachment_stale")
            try:
                skill = registry.load_skill(name=ref.id, digest=frozen.digest)
            except ExtensionRegistryError as exc:
                raise _error(exc.code) from exc
            loaded.append({
                "skill": ref.id, "digest": frozen.digest, "instructions": skill["instructions"],
                "authority": "user_advisory_only",
            })
    if len(json.dumps(loaded, ensure_ascii=False).encode()) > 64_000:
        raise _error("skill_attachment_context_too_large")
    return loaded


class SkillAttachmentService:
    """Small service adapter; persistence and session locking remain with their owners."""

    def attachable_skills(self, session_id: str, *, project_id: str) -> dict:
        record = self._scoped_record(session_id, project_id=project_id)
        enabled = record.agent_mode == "research" and capabilities.capability_settings()["science_skills_enabled"]
        items = [{
            **builtin_attachment(spec.skill_id), "title": spec.title, "title_zh": spec.title_zh,
            "category": spec.category, "enabled": enabled,
        } for spec in METHOD_SKILLS]
        items.extend({
            "source": "user", "id": skill.name, "digest": skill.digest,
            "title": skill.name, "title_zh": "", "category": "user",
            "enabled": "conversation" in skill.stages and not skill.disable_model_invocation,
        } for skill in record.extension_activation.skills)
        return {"items": items, "max_attached_skills": 8}

    def set_draft_skills(self, session_id: str, *, project_id: str, attached_skills: object) -> dict:
        with self._lock:
            record = self._scoped_record(session_id, project_id=project_id)
            if session_id in self._busy_sessions or session_id in self._dispatching_followups:
                raise _error("pi_session_busy")
            record.draft_attached_skills = validate_attachments(
                attached_skills, activation=record.extension_activation, registry=self.extension_registry,
                research=record.agent_mode == "research",
            )
            self._save_record(record)
            return {"attached_skills": attachment_payload(record.draft_attached_skills)}

    def project_default_skills(self, *, project_id: str, attached_skills: object = None) -> dict:
        self._assert_project_initialized(project_id)
        if attached_skills is not None:
            refs = validate_attachments(
                attached_skills, activation=self.extension_registry.snapshot(), registry=self.extension_registry,
            )
            self.project_store.set_default_skills(project_id, refs)
        binding = self.project_store.binding(project_id)
        return {"attached_skills": attachment_payload(binding.default_skills)}

    def _freeze_default_skills(self, project_id: str, activation, agent_mode: str) -> list:
        binding = self.project_store.binding(project_id)
        return validate_attachments(
            binding.default_skills if binding else [], activation=activation,
            registry=self.extension_registry, research=agent_mode == "research",
        )

    def _prepare_attached_skills(self, record: PiSessionRecord, supplied: object, regenerate_target) -> tuple[list, list]:
        if regenerate_target is not None:
            value = self.replay_store.attachments_for_turn(
                session_id=record.session_id, project_id=record.project_id,
                turn_index=int(regenerate_target.get("turn_index") or 0),
            )
        else:
            value = record.draft_attached_skills if supplied is None else supplied
        refs = validate_attachments(
            value, activation=record.extension_activation, registry=self.extension_registry,
            research=record.agent_mode == "research",
        )
        return refs, load_attached_users(record, refs, self.extension_registry)

    def _apply_turn_skills(self, record: PiSessionRecord, refs: list) -> PiSessionRecord:
        # Built-in attachments are durable study setup. User instructions load
        # only for this turn; their identities remain in its receipts/provenance.
        if refs and record.agent_mode == "research":
            current = study_contexts.get_context(record.binding.study_context_id)
            if current is None:
                raise _error("study_context_not_found")
            retained = [row for row in current.get("attached_skills", []) if row["source"] == "builtin"]
            combined = {(row["source"], row["id"]): row for row in retained}
            combined.update({(ref.source, ref.id): ref.model_dump(mode="json", exclude_none=True) for ref in refs})
            try:
                selected = attachment_payload(list(combined.values()))
                method_constraints(selected)
                saved = study_contexts.upsert_context(
                    {"id": current["id"], "attached_skills": selected},
                    expected_revision=current["revision"], lifecycle_write=False,
                    _server_skill_attachments_write=True,
                )
            except (ExtensionRegistryError, study_contexts.StudyContextError) as exc:
                raise _error(getattr(exc, "code", None) or exc.detail["error"]) from exc
            record.binding = self._binding_for_context(saved, run_id=record.binding.run_id)
        record.last_attached_skills = refs
        record.draft_attached_skills = []
        self._save_record(record)
        return record
