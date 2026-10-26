"""Conversation-scoped mention picker, admission and existing-owner adapters."""

from __future__ import annotations

import hashlib
import json

from easyicu.extensions.contracts import ExtensionRegistryError
from easyicu.extensions.skill_attachments import (
    ConceptMention, DatabaseMention, FileMention, SkillMention,
    mention_payload, parse_mentions,
)
from easyicu.webserver import sources, study_contexts
from easyicu.webserver.mention_objects import (
    DatabaseObject, ResolvedMention, SkillObject,
    concept_object, file_id, file_object, object_digest, require_mentioned_file, resolved_payload,
)

from .contracts import PiCopilotError, ToolExecutionContext


def _error(code: str) -> PiCopilotError:
    return PiCopilotError(code, code, status_code=409)


class MentionService:
    """The sk2 service uses these adapters inside its one turn lifecycle."""

    def search_mentions(self, session_id: str, *, project_id: str, kind: str,
                        query: str = "", limit: int = 20) -> dict:
        record = self._scoped_record(session_id, project_id=project_id)
        if kind not in {"database", "concept", "file", "skill"} or not isinstance(query, str) or len(query) > 120:
            raise _error("mention_search_invalid")
        if type(limit) is not int or not 1 <= limit <= 40:
            raise _error("mention_search_invalid")
        if kind == "concept":
            from easyicu.concept.catalog import CONCEPT_DICTIONARY, HIDDEN_DICTIONARY_CONCEPTS
            from easyicu.resources import load_dictionary
            from easyicu.webserver.catalog_metadata import catalog_definitions, concept_audience, variant_primary
            dictionary = load_dictionary(include_sofa2=True)
            definitions = catalog_definitions()
            # Bookkeeping columns and loader entry points answer only their
            # exact name, as in the data dictionary; a folded variant follows
            # the entry it belongs to.
            names = sorted(
                (name for name in set(dictionary.keys()) | set(CONCEPT_DICTIONARY)
                 if name == query.strip() or (
                     name not in HIDDEN_DICTIONARY_CONCEPTS
                     and concept_audience(name, definitions.get(name, {})) != "internal")),
                key=lambda name: (variant_primary(name) is not None, name),
            )
            rows = [{"kind": kind, "id": name, "label": (
                CONCEPT_DICTIONARY[name][0] if name in CONCEPT_DICTIONARY else dictionary.get(name).description or name
            ), "_terms": (CONCEPT_DICTIONARY.get(name) or ("", ""))[1]} for name in names]
        elif kind == "database":
            from easyicu.databases.profiles import iter_database_profiles
            rows = [{"kind": kind, "id": profile.key, "source": "database", "label": profile.display_name}
                    for profile in iter_database_profiles(public_only=True)]
            rows.extend({"kind": kind, "id": row["id"], "source": "registered_source", "label": row["database"]}
                        for row in sources.load_registry().get("sources", []) if row.get("ok") and row.get("id") and row.get("database"))
        elif kind == "file":
            from .workspace import _relative_file
            rows = []
            for row in self.workspace.list_files(project_id):
                try:
                    _relative_file(row["file"], model_read=True)
                except PiCopilotError:
                    continue
                rows.append({"kind": kind, "id": file_id(project_id, row["file"]), "label": row["file"]})
        else:
            rows = [{"kind": kind, "id": row["id"], "source": row["source"], "label": row["title"]}
                    for row in self.attachable_skills(record.session_id, project_id=project_id)["items"] if row["enabled"]]
        # A concept also answers to its Chinese name, which the client shows.
        terms = query.casefold().split()
        matches = [row for row in rows
                   if all(term in f"{row['id']} {row['label']} {row.get('_terms', '')}".casefold() for term in terms)]
        return {"items": [{**{key: value for key, value in row.items() if key != "_terms"}, "label": row["label"][:240]}
                          for row in matches[:limit]],
                "has_more": len(matches) > limit, "limit": limit, "max_mentions": 16}

    def resolve_mention(self, session_id: str, *, project_id: str, kind: str, id: str,
                        source: str | None = None, role: str = "candidate", digest: str | None = None) -> dict:
        record = self._scoped_record(session_id, project_id=project_id)
        return self._resolve_mention(record, kind=kind, id=id, source=source, role=role, digest=digest)

    def _resolve_mention(self, record, *, kind, id, source=None, role="candidate", digest=None) -> dict:
        from pydantic import TypeAdapter, ValidationError
        from easyicu.databases.profiles import get_database_profile
        from easyicu.extensions.skill_attachments import ObjectId

        try:
            TypeAdapter(ObjectId).validate_python(id)
            if (kind != "concept" and role != "candidate") or (kind in {"concept", "file"} and source is not None):
                raise _error("mention_invalid")
            if kind == "concept":
                obj = concept_object(id)
                ref = ConceptMention(id=id, role=role, digest=object_digest(obj))
            elif kind == "database":
                scope = source or "database"
                if scope == "registered_source":
                    row = self._mention_source(id)
                    # Hash the registration snapshot, never inspect its export.
                    identity = hashlib.sha256(json.dumps(row, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()
                    profile = get_database_profile(row["database"])
                    obj = DatabaseObject(database=profile.key, label=profile.display_name, source_id=id, identity_sha256=identity,
                                         path_sha256=hashlib.sha256(str(row["path"]).encode()).hexdigest())
                else:
                    profile = get_database_profile(id)
                    if profile.key != id or not profile.is_public:
                        raise _error("mention_unknown")
                    obj = DatabaseObject(database=profile.key, label=profile.display_name)
                ref = DatabaseMention(id=id, source=scope, digest=object_digest(obj))
            elif kind == "file":
                row = next((row for row in self.workspace.list_files(record.project_id)
                            if file_id(record.project_id, row["file"]) == id), None)
                if row is None:
                    raise _error("mention_unknown")
                material = self.workspace.read_file(record.project_id, row["file"])
                obj = file_object(record.project_id, material)
                ref = FileMention(id=id, digest=object_digest(obj))
            elif kind == "skill":
                from .skill_attachments import validate_attachments
                rows = self.attachable_skills(record.session_id, project_id=record.project_id)["items"]
                row = next((row for row in rows if row["id"] == id and row["source"] == source), None)
                if row is None:
                    raise _error("mention_unknown")
                attachment = {key: row[key] for key in ("source", "id", "version", "digest") if key in row}
                admitted = validate_attachments([attachment], activation=record.extension_activation,
                                                registry=self.extension_registry, research=record.agent_mode == "research")[0]
                obj = SkillObject(attachment=admitted)
                ref = SkillMention(attachment=admitted)
            else:
                raise _error("mention_invalid")
            expected = ref.attachment.digest if ref.kind == "skill" else ref.digest
            if digest is not None and digest != expected:
                raise _error("mention_stale")
            return resolved_payload([ResolvedMention(reference=ref, object=obj)])[0]
        except ExtensionRegistryError as exc:
            raise _error(exc.code) from exc
        except KeyError as exc:
            raise _error("mention_unknown") from exc
        except ValidationError as exc:
            raise _error("mention_invalid") from exc

    @staticmethod
    def _mention_source(source_id):
        rows = [row for row in sources.load_registry().get("sources", []) if row.get("ok") and row.get("id") == source_id]
        if len(rows) != 1:
            raise _error("mention_unknown")
        return rows[0]

    def _validate_mentions(self, record, value) -> list[dict]:
        from .skill_attachments import validate_attachments
        from .tools import require_copilot_data_clearance
        from easyicu.research_agent.providers.client_trust import ProviderConfigurationError, require_data_handling

        try:
            refs = parse_mentions(value)
            rows = []
            for ref in refs:
                if ref.kind == "skill":
                    admitted = validate_attachments([ref.attachment], activation=record.extension_activation,
                                                    registry=self.extension_registry, research=record.agent_mode == "research")[0]
                    rows.append(resolved_payload([ResolvedMention(reference=SkillMention(attachment=admitted), object=SkillObject(attachment=admitted))])[0])
                else:
                    rows.append(self._resolve_mention(record, kind=ref.kind, id=ref.id, digest=ref.digest,
                                                      source=getattr(ref, "source", None), role=getattr(ref, "role", "candidate")))
            # Admission is complete before a source, study or draft is changed.
            for row in rows:
                obj = row["object"]
                if obj["kind"] == "database":
                    if obj.get("source_id"):
                        require_copilot_data_clearance(ToolExecutionContext(session=record, data_handling=self._data_handling_for_session(record)), {"source_id": obj["source_id"]})
                    else:
                        require_data_handling(obj["database"], self._data_handling_for_session(record))
            return resolved_payload(rows)
        except ExtensionRegistryError as exc:
            raise _error(exc.code) from exc
        except ProviderConfigurationError as exc:
            raise PiCopilotError(exc.issue, exc.issue, status_code=403) from exc

    def set_draft_mentions(self, session_id: str, *, project_id: str, mentions: object) -> dict:
        with self._lock:
            record = self._scoped_record(session_id, project_id=project_id)
            if session_id in self._busy_sessions or session_id in self._dispatching_followups:
                raise _error("pi_session_busy")
            rows = self._validate_mentions(record, mentions)
            record.draft_mentions = parse_mentions([row["reference"] for row in rows])
            self._save_record(record)
            return {"mentions": mention_payload(record.draft_mentions), "mentioned_objects": rows}

    def _prepare_turn_mentions(self, record, supplied, regenerate_target) -> list[dict]:
        if regenerate_target is not None:
            value = self.replay_store.mentions_for_turn(session_id=record.session_id, project_id=record.project_id,
                                                       turn_index=int(regenerate_target.get("turn_index") or 0))
        else:
            value = record.draft_mentions if supplied is None else supplied
        return self._validate_mentions(record, value)

    def _load_mention_context(self, record, rows):
        from .projections import MAX_TEXT_CHARS, ensure_safe_projection

        context = []
        for row in rows:
            supplied = dict(row)
            if row["reference"]["kind"] == "file":
                material = self.workspace.read_file(record.project_id, row["object"]["file"])
                try:
                    require_mentioned_file(record.project_id, [row["reference"]], material)
                except ExtensionRegistryError as exc:
                    raise _error(exc.code) from exc
                if len(material["text"]) > MAX_TEXT_CHARS:
                    material = {**material, "text": material["text"][:MAX_TEXT_CHARS], "truncated": True}
                supplied["content"] = ensure_safe_projection(material)
            context.append(supplied)
        if len(json.dumps(context, ensure_ascii=False).encode()) > 64_000:
            raise _error("mention_context_too_large")
        return context

    def _mention_source_for_turn(self, rows):
        for item in rows:
            obj = item["object"]
            if obj["kind"] == "database" and obj.get("source_id"):
                return self._mention_source(obj["source_id"])
        return None

    def _confirm_mentioned_source(self, record, saved):
        from .contracts import PiSessionDataSourceAuthorization, utc_now

        record.data_source_authorization = PiSessionDataSourceAuthorization(
            status="confirmed", reason=None, confirmation_mode="select_registered_source",
            extraction_scope="reuse_prepared_full", source=self._session_source_reference(saved), confirmed_at=utc_now(),
        )

    @staticmethod
    def _merge_mentioned_objects(current, rows):
        # A turn explicitly replaces a single role/database, retaining other
        # exact identities. It cannot silently displace two pins in one turn.
        from easyicu.extensions.skill_attachments import mention_identity
        previous = current.get("mentioned_objects") or []
        refs = parse_mentions([row["reference"] for row in rows])
        replaced = {mention_identity(ref) for ref in refs}
        roles = {ref.role for ref in refs if ref.kind == "concept" and ref.role in {"exposure", "outcome"}}
        kept = [row for row in previous
                if mention_identity(parse_mentions([row["reference"]])[0]) not in replaced
                and not (row["reference"]["kind"] == "database" and any(ref.kind == "database" for ref in refs))
                and not (row["reference"]["kind"] == "concept" and row["reference"]["role"] in roles)]
        return resolved_payload([*kept, *rows])

    def _preview_turn_mentions(self, record, rows):
        if not rows or record.agent_mode != "research":
            return
        current = study_contexts.get_context(record.binding.study_context_id)
        if current is None:
            raise _error("study_context_not_found")
        if current.get("active_job_id"):
            raise _error("study_context_active_job_conflict")
        patch = {"mentioned_objects": self._merge_mentioned_objects(current, rows)}
        selected = self._mention_source_for_turn(rows)
        if selected:
            patch["data_source"] = {key: selected[key] for key in ("path", "database", "label") if key in selected}
        try:
            study_contexts.validate_context_update(patch, current_context=current, _server_skill_attachments_write=True)
        except study_contexts.StudyContextError as exc:
            raise _error(exc.detail["error"]) from exc
