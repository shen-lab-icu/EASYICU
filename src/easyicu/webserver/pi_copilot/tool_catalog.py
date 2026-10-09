"""Single cross-language policy roster for Pi Copilot host tools."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence


_SCHEMA_VERSION = "easyicu.pi-tool-catalog/3"
# ``_arguments`` is a prose note in the file, addressed to whoever edits it.
_ROOT_FIELDS = frozenset({"schema_version", "_arguments", "tools"})
_ARGUMENT_FIELDS = frozenset({"model", "host", "required"})
_TOOL_FIELDS = frozenset(
    {
        "name",
        "surface",
        "policy_group",
        "execution_mode",
        "host_mutating",
        "data_source_required",
        "arguments",
        "model_schema",
    }
)


@dataclass(frozen=True)
class ToolArguments:
    """The one declaration of what a host tool accepts.

    ``model`` is what the language model may send, and must equal the TypeBox
    property names declared for this tool in ``node_app/src/main.mjs``, or the
    properties of the entry's generated ``model_schema``; a contract test
    fails the build when the two drift. ``host`` are keys only
    the host injects and the model never sees — ``easyicu_run``'s
    ``llm_provider`` is one, which is why the JavaScript schema deliberately
    omits it. ``required`` must be present and non-empty.

    Before this existed a tool's identity was split across three files in two
    languages, with the argument names written out twice and no gate between
    them, kept aligned by hand across 42 tools.
    """

    model: tuple[str, ...]
    host: tuple[str, ...]
    required: tuple[str, ...]

    @property
    def allowed(self) -> frozenset[str]:
        """Every key the host will accept: what the model may send, plus its own."""
        return frozenset(self.model) | frozenset(self.host)


@dataclass(frozen=True)
class ToolCatalogEntry:
    name: str
    surface: str
    policy_group: str
    execution_mode: str
    host_mutating: bool
    data_source_required: bool
    arguments: ToolArguments
    # The JSON schema main.mjs gives the model, when the host generates it
    # from its own models instead of main.mjs declaring it in TypeBox.
    model_schema: Mapping[str, object] | None = None


def _parse_arguments(raw: object) -> ToolArguments:
    """Read one tool's argument declaration, fail closed on anything unexpected."""

    if not isinstance(raw, Mapping) or set(raw) != _ARGUMENT_FIELDS:
        raise RuntimeError("pi_tool_catalog_arguments_invalid")
    parsed: dict[str, tuple[str, ...]] = {}
    for field in ("model", "host", "required"):
        values = raw.get(field)
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            raise RuntimeError("pi_tool_catalog_arguments_invalid")
        names = [str(value) for value in values]
        if any(not value.strip() for value in names) or len(set(names)) != len(names):
            raise RuntimeError("pi_tool_catalog_arguments_invalid")
        parsed[field] = tuple(names)
    if set(parsed["model"]) & set(parsed["host"]):
        raise RuntimeError("pi_tool_catalog_arguments_overlap")
    accepted = set(parsed["model"]) | set(parsed["host"])
    if not set(parsed["required"]) <= accepted:
        raise RuntimeError("pi_tool_catalog_arguments_required_unknown")
    return ToolArguments(**parsed)


def _parse_model_schema(
    raw: object, arguments: ToolArguments
) -> Mapping[str, object] | None:
    """Read a generated parameter schema; it must offer exactly ``model``."""

    if raw is None:
        return None
    if (
        not isinstance(raw, Mapping)
        or raw.get("type") != "object"
        or raw.get("additionalProperties") is not False
        or not isinstance(raw.get("properties"), Mapping)
    ):
        raise RuntimeError("pi_tool_catalog_model_schema_invalid")
    if set(raw["properties"]) != set(arguments.model):
        raise RuntimeError("pi_tool_catalog_model_schema_arguments_mismatch")
    required = raw.get("required", [])
    if (
        not isinstance(required, Sequence)
        or isinstance(required, (str, bytes))
        or not set(required) <= set(arguments.model)
    ):
        raise RuntimeError("pi_tool_catalog_model_schema_invalid")
    return raw


def load_tool_catalog(path: Path | None = None) -> tuple[ToolCatalogEntry, ...]:
    """Load the committed catalog and reject drift or extension fail closed."""

    catalog_path = path or Path(__file__).with_name("tool_catalog.json")
    try:
        payload = json.loads(catalog_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("pi_tool_catalog_unreadable") from exc
    if not isinstance(payload, Mapping) or set(payload) != _ROOT_FIELDS:
        raise RuntimeError("pi_tool_catalog_root_invalid")
    if payload.get("schema_version") != _SCHEMA_VERSION:
        raise RuntimeError("pi_tool_catalog_schema_unsupported")
    rows = payload.get("tools")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise RuntimeError("pi_tool_catalog_tools_invalid")
    entries: list[ToolCatalogEntry] = []
    names: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != _TOOL_FIELDS:
            raise RuntimeError("pi_tool_catalog_entry_invalid")
        name = row.get("name")
        surface = row.get("surface")
        policy_group = row.get("policy_group")
        execution_mode = row.get("execution_mode")
        if not isinstance(name, str) or not name.startswith("easyicu_") or name in names:
            raise RuntimeError("pi_tool_catalog_name_invalid")
        if surface not in {"research", "workspace"}:
            raise RuntimeError("pi_tool_catalog_surface_invalid")
        if policy_group not in {"read", "control", "workspace"}:
            raise RuntimeError("pi_tool_catalog_policy_group_invalid")
        if (surface == "workspace") != (policy_group == "workspace"):
            raise RuntimeError("pi_tool_catalog_surface_policy_mismatch")
        if execution_mode not in {"parallel", "sequential"}:
            raise RuntimeError("pi_tool_catalog_execution_mode_invalid")
        if not isinstance(row.get("host_mutating"), bool) or not isinstance(
            row.get("data_source_required"), bool
        ):
            raise RuntimeError("pi_tool_catalog_boolean_invalid")
        arguments = _parse_arguments(row.get("arguments"))
        model_schema = _parse_model_schema(row.get("model_schema"), arguments)
        names.add(name)
        entries.append(
            ToolCatalogEntry(
                name=name,
                surface=surface,
                policy_group=policy_group,
                execution_mode=execution_mode,
                host_mutating=row["host_mutating"],
                data_source_required=row["data_source_required"],
                arguments=arguments,
                model_schema=model_schema,
            )
        )
    if not entries:
        raise RuntimeError("pi_tool_catalog_empty")
    return tuple(entries)


TOOL_CATALOG = load_tool_catalog()
READ_TOOLS = frozenset(row.name for row in TOOL_CATALOG if row.policy_group == "read")
CONTROL_TOOLS = frozenset(
    row.name for row in TOOL_CATALOG if row.policy_group == "control"
)
WORKSPACE_TOOLS = frozenset(
    row.name for row in TOOL_CATALOG if row.policy_group == "workspace"
)
ALLOWED_TOOLS = READ_TOOLS | CONTROL_TOOLS | WORKSPACE_TOOLS
MUTATING_HOST_TOOLS = frozenset(row.name for row in TOOL_CATALOG if row.host_mutating)
DATA_SOURCE_REQUIRED_TOOLS = frozenset(
    row.name for row in TOOL_CATALOG if row.data_source_required
)
RESEARCH_TOOL_NAMES = tuple(
    row.name for row in TOOL_CATALOG if row.surface == "research"
)
ALL_TOOL_NAMES = tuple(row.name for row in TOOL_CATALOG)
TOOL_ARGUMENTS = {row.name: row.arguments for row in TOOL_CATALOG}
TOOL_MODEL_SCHEMAS = {
    row.name: row.model_schema for row in TOOL_CATALOG if row.model_schema is not None
}


__all__ = [
    "ALLOWED_TOOLS",
    "ALL_TOOL_NAMES",
    "CONTROL_TOOLS",
    "DATA_SOURCE_REQUIRED_TOOLS",
    "MUTATING_HOST_TOOLS",
    "READ_TOOLS",
    "RESEARCH_TOOL_NAMES",
    "TOOL_ARGUMENTS",
    "TOOL_CATALOG",
    "TOOL_MODEL_SCHEMAS",
    "ToolArguments",
    "ToolCatalogEntry",
    "WORKSPACE_TOOLS",
    "load_tool_catalog",
]
