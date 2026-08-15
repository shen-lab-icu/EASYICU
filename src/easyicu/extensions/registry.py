"""Content-addressed registry for user-installed Skills and MCP servers."""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
import re
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Mapping, Sequence

import yaml

from easyicu.outbound_url_security import (
    OutboundUrlSecurityError,
    validate_outbound_http_endpoint,
)

from .contracts import (
    ExtensionActivationSnapshot,
    ExtensionRegistryError,
    McpServerActivation,
    NAME_PATTERN,
    SkillActivation,
    TOOL_PATTERN,
)

REGISTRY_SCHEMA = "easyicu.extension-registry/1"
MAX_REGISTRY_BYTES = 512 * 1024
MAX_SKILL_BYTES = 12_000
MAX_ACTIVE_SKILLS = 8
MAX_ACTIVE_SKILL_BYTES = 48_000
MAX_WRITING_ADVISORY_BYTES = 8_000
_SKILL_FRONTMATTER_KEYS = frozenset(
    {"name", "description", "disable-model-invocation"}
)
_MODEL_VISIBLE_SECRET_OR_HOST_PATH = re.compile(
    r"(?:file://|/(?:Users|home|private|tmp|var|etc|opt|Volumes)/|[A-Za-z]:\\|"
    r"\bBearer\s+[A-Za-z0-9._~+/=-]{8,}|\bsk-[A-Za-z0-9_-]{8,}|"
    r"\b(?:api[_-]?key|password|secret|token)\s*[:=]\s*\S+|"
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----)",
    flags=re.IGNORECASE,
)
_THREAD_LOCKS: Dict[str, threading.RLock] = {}
_THREAD_LOCKS_GUARD = threading.Lock()


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _clean_name(value: Any, *, code: str) -> str:
    name = str(value or "").strip()
    if not re.fullmatch(NAME_PATTERN, name):
        raise ExtensionRegistryError(
            code,
            "Extension names must use lowercase letters, digits, and single hyphens.",
            details={"name": name[:80]},
        )
    return name


def _normalise_stages(values: Iterable[Any]) -> tuple[str, ...]:
    stages = tuple(dict.fromkeys(str(value or "").strip() for value in values))
    if not stages or any(value not in {"conversation", "writing"} for value in stages):
        raise ExtensionRegistryError(
            "extension_skill_stage_invalid",
            "Choose conversation, writing, or both as the Skill activation stage.",
        )
    return stages


def _normalise_tools(values: Iterable[Any]) -> tuple[str, ...]:
    tools = tuple(dict.fromkeys(str(value or "").strip() for value in values))
    if not tools or len(tools) > 32:
        raise ExtensionRegistryError(
            "extension_mcp_tool_allowlist_required",
            "An MCP server needs between 1 and 32 explicitly allowed tools.",
        )
    invalid = [name for name in tools if not re.fullmatch(TOOL_PATTERN, name)]
    if invalid:
        raise ExtensionRegistryError(
            "extension_mcp_tool_name_invalid",
            "An MCP allowlist entry has an invalid tool name.",
            details={"tool": invalid[0][:160]},
        )
    return tools


def parse_skill_markdown(skill_markdown: str) -> Dict[str, Any]:
    """Parse the supported SKILL.md contract without resolving local paths."""

    text = str(skill_markdown or "").replace("\r\n", "\n").replace("\r", "\n")
    if not text.strip():
        raise ExtensionRegistryError(
            "extension_skill_content_required", "Paste or upload a SKILL.md file."
        )
    encoded = text.encode("utf-8")
    if len(encoded) > MAX_SKILL_BYTES:
        raise ExtensionRegistryError(
            "extension_skill_too_large",
            "The Skill exceeds the 12 KB reviewed-instruction limit.",
            details={"bytes": len(encoded), "max_bytes": MAX_SKILL_BYTES},
        )
    match = re.match(r"\A---\n(.*?)\n---\n(.*)\Z", text, flags=re.DOTALL)
    if match is None:
        raise ExtensionRegistryError(
            "extension_skill_frontmatter_required",
            "SKILL.md must start with YAML frontmatter containing name and description.",
        )
    try:
        metadata = yaml.safe_load(match.group(1))
    except yaml.YAMLError as exc:
        raise ExtensionRegistryError(
            "extension_skill_frontmatter_invalid", "SKILL.md frontmatter is invalid YAML."
        ) from exc
    if not isinstance(metadata, Mapping):
        raise ExtensionRegistryError(
            "extension_skill_frontmatter_invalid", "SKILL.md frontmatter must be an object."
        )
    unknown = sorted(set(metadata) - _SKILL_FRONTMATTER_KEYS)
    if unknown:
        raise ExtensionRegistryError(
            "extension_skill_frontmatter_key_unknown",
            "SKILL.md contains an unsupported frontmatter key.",
            details={"keys": unknown},
        )
    name = _clean_name(metadata.get("name"), code="extension_skill_name_invalid")
    description = re.sub(r"\s+", " ", str(metadata.get("description") or "")).strip()
    if not description or len(description) > 1024:
        raise ExtensionRegistryError(
            "extension_skill_description_invalid",
            "Skill description must contain between 1 and 1024 characters.",
        )
    disable = metadata.get("disable-model-invocation", False)
    if not isinstance(disable, bool):
        raise ExtensionRegistryError(
            "extension_skill_invocation_policy_invalid",
            "disable-model-invocation must be true or false when present.",
        )
    body = match.group(2).strip()
    if not body:
        raise ExtensionRegistryError(
            "extension_skill_instructions_required", "SKILL.md needs instruction content."
        )
    if _MODEL_VISIBLE_SECRET_OR_HOST_PATH.search(text):
        raise ExtensionRegistryError(
            "extension_skill_sensitive_content_rejected",
            "SKILL.md must not contain credentials or absolute host filesystem paths.",
        )
    canonical = text.rstrip() + "\n"
    return {
        "name": name,
        "description": description,
        "disable_model_invocation": disable,
        "instructions": body,
        "canonical_markdown": canonical,
        "digest": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "size_bytes": len(canonical.encode("utf-8")),
    }


class ExtensionRegistry:
    """Own extension persistence, validation, snapshots, and exact Skill loads."""

    def __init__(self, root: Path | str | None = None) -> None:
        configured = root or os.getenv("EASYICU_EXTENSION_HOME")
        self.root = (
            Path(configured).expanduser()
            if configured
            else Path.home() / ".easyicu" / "extensions"
        ).absolute()
        self.state_path = self.root / "registry.json"
        self.lock_path = self.root / ".registry.lock"
        self.skill_objects = self.root / "objects" / "skills"

    @property
    def _thread_lock(self) -> threading.RLock:
        key = str(self.root)
        with _THREAD_LOCKS_GUARD:
            return _THREAD_LOCKS.setdefault(key, threading.RLock())

    def _ensure_root(self) -> None:
        self.skill_objects.mkdir(parents=True, exist_ok=True, mode=0o700)
        with contextlib.suppress(OSError):
            os.chmod(self.root, 0o700)

    @contextlib.contextmanager
    def _locked(self) -> Iterator[None]:
        self._ensure_root()
        with self._thread_lock:
            with self.lock_path.open("a+b") as handle:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    @staticmethod
    def _empty_state() -> Dict[str, Any]:
        return {
            "schema_version": REGISTRY_SCHEMA,
            "revision": 0,
            "skills": [],
            "mcp_servers": [],
        }

    def _read_state_unlocked(self) -> Dict[str, Any]:
        try:
            size = self.state_path.stat().st_size
            if size > MAX_REGISTRY_BYTES:
                raise ExtensionRegistryError(
                    "extension_registry_too_large",
                    "The extension registry exceeds its bounded contract.",
                )
            raw = json.loads(self.state_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return self._empty_state()
        except json.JSONDecodeError as exc:
            raise ExtensionRegistryError(
                "extension_registry_invalid", "The extension registry is invalid JSON."
            ) from exc
        if not isinstance(raw, dict) or raw.get("schema_version") != REGISTRY_SCHEMA:
            raise ExtensionRegistryError(
                "extension_registry_invalid", "The extension registry schema is invalid."
            )
        if not isinstance(raw.get("skills"), list) or not isinstance(
            raw.get("mcp_servers"), list
        ):
            raise ExtensionRegistryError(
                "extension_registry_invalid", "The extension registry shape is invalid."
            )
        return raw

    def _write_state_unlocked(self, state: Mapping[str, Any]) -> None:
        payload = json.dumps(
            dict(state), sort_keys=True, ensure_ascii=False, indent=2
        ).encode("utf-8")
        if len(payload) > MAX_REGISTRY_BYTES:
            raise ExtensionRegistryError(
                "extension_registry_too_large",
                "The extension registry exceeds its bounded contract.",
            )
        fd, temporary = tempfile.mkstemp(prefix=".registry-", suffix=".json", dir=self.root)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(temporary, 0o600)
            os.replace(temporary, self.state_path)
        finally:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(temporary)

    def _write_skill_object_unlocked(self, parsed: Mapping[str, Any]) -> None:
        target = self.skill_objects / f"{parsed['digest']}.md"
        content = str(parsed["canonical_markdown"]).encode("utf-8")
        if target.exists():
            if target.read_bytes() != content:
                raise ExtensionRegistryError(
                    "extension_skill_object_digest_collision",
                    "A stored Skill object does not match its content digest.",
                )
            return
        fd, temporary = tempfile.mkstemp(prefix=".skill-", suffix=".md", dir=self.skill_objects)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(temporary, 0o600)
            os.replace(temporary, target)
        finally:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(temporary)

    @staticmethod
    def _replace_named(rows: list[Dict[str, Any]], row: Dict[str, Any]) -> None:
        rows[:] = [item for item in rows if str(item.get("name")) != row["name"]]
        rows.append(row)
        rows.sort(key=lambda item: str(item.get("name") or ""))

    def install_skill(
        self,
        skill_markdown: str,
        *,
        stages: Sequence[str] = ("conversation",),
        enabled: bool = True,
    ) -> Dict[str, Any]:
        parsed = parse_skill_markdown(skill_markdown)
        clean_stages = _normalise_stages(stages)
        if parsed["disable_model_invocation"] and enabled:
            raise ExtensionRegistryError(
                "extension_skill_manual_invocation_unsupported",
                "This SKILL.md disables model invocation. Install it disabled, or remove that flag after review before enabling it in EasyICU.",
            )
        now = _utc_now()
        with self._locked():
            state = self._read_state_unlocked()
            existing = next(
                (
                    row
                    for row in state["skills"]
                    if str(row.get("name")) == parsed["name"]
                ),
                None,
            )
            self._write_skill_object_unlocked(parsed)
            row = {
                "name": parsed["name"],
                "description": parsed["description"],
                "digest": parsed["digest"],
                "size_bytes": parsed["size_bytes"],
                "stages": list(clean_stages),
                "disable_model_invocation": parsed["disable_model_invocation"],
                "enabled": bool(enabled),
                "installed_at": (existing or {}).get("installed_at") or now,
                "updated_at": now,
            }
            self._replace_named(state["skills"], row)
            state["revision"] = int(state.get("revision") or 0) + 1
            self._validate_active_limits(state)
            self._write_state_unlocked(state)
        return self._public_skill(row)

    def install_mcp_server(
        self,
        *,
        name: str,
        url: str,
        allowed_tools: Sequence[str],
        enabled: bool = False,
    ) -> Dict[str, Any]:
        clean_name = _clean_name(name, code="extension_mcp_name_invalid")
        clean_tools = _normalise_tools(allowed_tools)
        try:
            clean_url = validate_outbound_http_endpoint(url)
        except OutboundUrlSecurityError as exc:
            raise ExtensionRegistryError(
                "extension_mcp_url_rejected",
                "The MCP endpoint violates the outbound network policy.",
                details={"reason": exc.reason},
            ) from exc
        now = _utc_now()
        with self._locked():
            state = self._read_state_unlocked()
            existing = next(
                (
                    row
                    for row in state["mcp_servers"]
                    if str(row.get("name")) == clean_name
                ),
                None,
            )
            row = {
                "name": clean_name,
                "url": clean_url,
                "transport": "streamable-http",
                "allowed_tools": list(clean_tools),
                "enabled": bool(enabled),
                "installed_at": (existing or {}).get("installed_at") or now,
                "updated_at": now,
                "authentication": "none",
                "data_scope": "external_metadata_only",
            }
            self._replace_named(state["mcp_servers"], row)
            state["revision"] = int(state.get("revision") or 0) + 1
            self._write_state_unlocked(state)
        return self._public_mcp(row)

    def set_enabled(self, *, kind: str, name: str, enabled: bool) -> Dict[str, Any]:
        collection = self._collection_name(kind)
        clean_name = _clean_name(name, code="extension_name_invalid")
        with self._locked():
            state = self._read_state_unlocked()
            row = next(
                (item for item in state[collection] if item.get("name") == clean_name),
                None,
            )
            if row is None:
                raise ExtensionRegistryError(
                    "extension_not_found", "The requested extension is not installed."
                )
            if (
                collection == "skills"
                and bool(enabled)
                and bool(row.get("disable_model_invocation"))
            ):
                raise ExtensionRegistryError(
                    "extension_skill_manual_invocation_unsupported",
                    "This Skill disables model invocation and cannot be enabled in the current EasyICU host flow.",
                )
            row["enabled"] = bool(enabled)
            row["updated_at"] = _utc_now()
            state["revision"] = int(state.get("revision") or 0) + 1
            self._validate_active_limits(state)
            self._write_state_unlocked(state)
            return self._public_skill(row) if collection == "skills" else self._public_mcp(row)

    def remove(self, *, kind: str, name: str) -> Dict[str, Any]:
        collection = self._collection_name(kind)
        clean_name = _clean_name(name, code="extension_name_invalid")
        with self._locked():
            state = self._read_state_unlocked()
            before = len(state[collection])
            state[collection] = [
                item for item in state[collection] if item.get("name") != clean_name
            ]
            if len(state[collection]) == before:
                raise ExtensionRegistryError(
                    "extension_not_found", "The requested extension is not installed."
                )
            state["revision"] = int(state.get("revision") or 0) + 1
            self._write_state_unlocked(state)
        return {"ok": True, "kind": kind, "name": clean_name, "removed": True}

    @staticmethod
    def _collection_name(kind: str) -> str:
        clean = str(kind or "").strip().lower()
        if clean == "skill":
            return "skills"
        if clean == "mcp":
            return "mcp_servers"
        raise ExtensionRegistryError(
            "extension_kind_invalid", "Extension kind must be skill or mcp."
        )

    @staticmethod
    def _validate_active_limits(state: Mapping[str, Any]) -> None:
        active = [row for row in state.get("skills", []) if row.get("enabled")]
        total = sum(int(row.get("size_bytes") or 0) for row in active)
        if len(active) > MAX_ACTIVE_SKILLS or total > MAX_ACTIVE_SKILL_BYTES:
            raise ExtensionRegistryError(
                "extension_skill_activation_limit",
                "The active Skill set exceeds its count or instruction-size limit.",
                details={
                    "active_skills": len(active),
                    "max_active_skills": MAX_ACTIVE_SKILLS,
                    "active_bytes": total,
                    "max_active_bytes": MAX_ACTIVE_SKILL_BYTES,
                },
            )

    @staticmethod
    def _public_skill(row: Mapping[str, Any]) -> Dict[str, Any]:
        return {
            key: row.get(key)
            for key in (
                "name",
                "description",
                "digest",
                "size_bytes",
                "stages",
                "disable_model_invocation",
                "enabled",
                "installed_at",
                "updated_at",
            )
        }

    @staticmethod
    def _public_mcp(row: Mapping[str, Any]) -> Dict[str, Any]:
        return {
            key: row.get(key)
            for key in (
                "name",
                "url",
                "transport",
                "allowed_tools",
                "enabled",
                "installed_at",
                "updated_at",
                "authentication",
                "data_scope",
            )
        }

    def list_public(self) -> Dict[str, Any]:
        with self._locked():
            state = self._read_state_unlocked()
            snapshot = self._snapshot_from_state(state)
            return {
                "schema_version": REGISTRY_SCHEMA,
                "revision": int(state.get("revision") or 0),
                "activation_sha256": snapshot.activation_sha256,
                "skills": [self._public_skill(row) for row in state["skills"]],
                "mcp_servers": [
                    self._public_mcp(row) for row in state["mcp_servers"]
                ],
                "policy": {
                    "skill_stages": ["conversation", "writing"],
                    "mcp_transport": "streamable-http",
                    "mcp_authentication": "none",
                    "mcp_tool_policy": "explicit_read_only_allowlist",
                    "snapshot_timing": "new_session_or_run",
                },
            }

    def snapshot(self) -> ExtensionActivationSnapshot:
        with self._locked():
            return self._snapshot_from_state(self._read_state_unlocked())

    def _snapshot_from_state(
        self, state: Mapping[str, Any]
    ) -> ExtensionActivationSnapshot:
        self._validate_active_limits(state)
        skills = [
            SkillActivation(
                name=row["name"],
                description=row["description"],
                digest=row["digest"],
                stages=tuple(row["stages"]),
                disable_model_invocation=bool(row.get("disable_model_invocation")),
            )
            for row in state.get("skills", [])
            if row.get("enabled")
        ]
        mcp_servers = [
            McpServerActivation(
                name=row["name"],
                url=row["url"],
                transport="streamable-http",
                allowed_tools=tuple(row["allowed_tools"]),
            )
            for row in state.get("mcp_servers", [])
            if row.get("enabled")
        ]
        return ExtensionActivationSnapshot.build(
            revision=int(state.get("revision") or 0),
            skills=skills,
            mcp_servers=mcp_servers,
        )

    def load_skill(
        self, *, name: str, digest: str | None = None
    ) -> Dict[str, Any]:
        clean_name = _clean_name(name, code="extension_skill_name_invalid")
        clean_digest = str(digest or "").strip()
        if not re.fullmatch(r"[a-f0-9]{64}", clean_digest):
            raise ExtensionRegistryError(
                "extension_skill_digest_required",
                "An exact frozen Skill digest is required.",
            )
        target = self.skill_objects / f"{clean_digest}.md"
        try:
            content = target.read_text(encoding="utf-8")
        except FileNotFoundError as exc:
            raise ExtensionRegistryError(
                "extension_skill_object_not_found",
                "The frozen Skill object is no longer available.",
            ) from exc
        parsed = parse_skill_markdown(content)
        if parsed["digest"] != clean_digest or parsed["name"] != clean_name:
            raise ExtensionRegistryError(
                "extension_skill_object_mismatch",
                "The frozen Skill object does not match its activation receipt.",
            )
        return {
            key: parsed[key]
            for key in (
                "name",
                "description",
                "digest",
                "disable_model_invocation",
                "instructions",
            )
        }

    def pipeline_activation(
        self, snapshot: ExtensionActivationSnapshot | None = None
    ) -> Dict[str, Any]:
        """Compile exact writing advisory text into a run-hashed plain payload."""

        frozen = snapshot or self.snapshot()
        blocks: list[str] = []
        active_descriptors: list[Dict[str, Any]] = []
        for skill in frozen.skills:
            descriptor = skill.model_dump(mode="json")
            active_descriptors.append(descriptor)
            if "writing" not in skill.stages:
                continue
            loaded = self.load_skill(name=skill.name, digest=skill.digest)
            blocks.append(
                f"### Skill: {skill.name} (sha256:{skill.digest})\n"
                + str(loaded["instructions"])
            )
        advisory = "\n\n".join(blocks)
        if len(advisory.encode("utf-8")) > MAX_WRITING_ADVISORY_BYTES:
            raise ExtensionRegistryError(
                "extension_writing_advisory_too_large",
                "Active writing Skills exceed the run prompt advisory limit.",
                details={"max_bytes": MAX_WRITING_ADVISORY_BYTES},
            )
        receipt = {
            "schema_version": "easyicu.user-extension-run-receipt/1",
            "activation_sha256": frozen.activation_sha256,
            "revision": frozen.revision,
            "skills": active_descriptors,
            "mcp_servers": [
                {
                    "name": item.name,
                    "transport": item.transport,
                    "allowed_tools": list(item.allowed_tools),
                    "endpoint_sha256": hashlib.sha256(
                        item.url.encode("utf-8")
                    ).hexdigest(),
                }
                for item in frozen.mcp_servers
            ],
        }
        return {
            "schema_version": "easyicu.pipeline-extension-activation/1",
            "activation_sha256": frozen.activation_sha256,
            "revision": frozen.revision,
            "writing_advisory": advisory,
            "receipt": receipt,
        }


__all__ = [
    "ExtensionRegistry",
    "MAX_SKILL_BYTES",
    "REGISTRY_SCHEMA",
    "parse_skill_markdown",
]
