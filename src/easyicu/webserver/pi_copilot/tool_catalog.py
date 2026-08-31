"""Single cross-language policy roster for Pi Copilot host tools."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence


_SCHEMA_VERSION = "easyicu.pi-tool-catalog/1"
_ROOT_FIELDS = frozenset({"schema_version", "tools"})
_TOOL_FIELDS = frozenset(
    {
        "name",
        "surface",
        "policy_group",
        "execution_mode",
        "host_mutating",
        "data_source_required",
    }
)


@dataclass(frozen=True)
class ToolCatalogEntry:
    name: str
    surface: str
    policy_group: str
    execution_mode: str
    host_mutating: bool
    data_source_required: bool


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
        names.add(name)
        entries.append(
            ToolCatalogEntry(
                name=name,
                surface=surface,
                policy_group=policy_group,
                execution_mode=execution_mode,
                host_mutating=row["host_mutating"],
                data_source_required=row["data_source_required"],
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


__all__ = [
    "ALLOWED_TOOLS",
    "ALL_TOOL_NAMES",
    "CONTROL_TOOLS",
    "DATA_SOURCE_REQUIRED_TOOLS",
    "MUTATING_HOST_TOOLS",
    "READ_TOOLS",
    "RESEARCH_TOOL_NAMES",
    "TOOL_CATALOG",
    "ToolCatalogEntry",
    "WORKSPACE_TOOLS",
    "load_tool_catalog",
]
