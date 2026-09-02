from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from easyicu.webserver.pi_copilot import tools
from easyicu.webserver.pi_copilot.tool_catalog import (
    ALLOWED_TOOLS,
    ALL_TOOL_NAMES,
    DATA_SOURCE_REQUIRED_TOOLS,
    MUTATING_HOST_TOOLS,
    TOOL_ARGUMENTS,
    TOOL_CATALOG,
    load_tool_catalog,
)


def test_catalog_is_the_ordered_policy_authority_for_python_dispatch() -> None:
    assert tuple(row.name for row in TOOL_CATALOG) == ALL_TOOL_NAMES
    assert frozenset(tools._DISPATCH) == ALLOWED_TOOLS
    assert MUTATING_HOST_TOOLS == frozenset(
        row.name for row in TOOL_CATALOG if row.host_mutating
    )
    assert DATA_SOURCE_REQUIRED_TOOLS == frozenset(
        row.name for row in TOOL_CATALOG if row.data_source_required
    )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda payload: payload.update(extra=True),
        lambda payload: payload["tools"].append(dict(payload["tools"][0])),
        lambda payload: payload["tools"][0].update(execution_mode="hidden"),
        # An argument declaration is part of a tool's identity, so the same
        # fail-closed rules apply to it.
        lambda payload: payload["tools"][0].pop("arguments"),
        lambda payload: payload["tools"][0].update(arguments={"model": ["a"]}),
        lambda payload: payload["tools"][0].update(
            arguments={"model": ["a"], "host": ["a"], "required": []}
        ),
        lambda payload: payload["tools"][0].update(
            arguments={"model": ["a"], "host": [], "required": ["b"]}
        ),
        lambda payload: payload["tools"][0].update(
            arguments={"model": ["a", "a"], "host": [], "required": []}
        ),
        lambda payload: payload["tools"][0].update(
            arguments={"model": "a", "host": [], "required": []}
        ),
    ],
)
def test_catalog_rejects_extension_duplicate_and_unknown_policy(
    tmp_path: Path,
    mutation,
) -> None:
    payload = {
        "schema_version": "easyicu.pi-tool-catalog/2",
        "_arguments": ["note for whoever edits this file"],
        "tools": [
            {
                "name": "easyicu_example",
                "surface": "research",
                "policy_group": "read",
                "execution_mode": "parallel",
                "host_mutating": False,
                "data_source_required": False,
                "arguments": {"model": ["a"], "host": [], "required": ["a"]},
            }
        ],
    }
    mutation(payload)
    path = tmp_path / "tool_catalog.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError):
        load_tool_catalog(path)


# ---------------------------------------------------------------------------
# The catalog owns the argument names; both language sides derive from it
# ---------------------------------------------------------------------------
#
# A host tool's identity used to be split across three files in two languages:
# this catalog held the policy, node_app/src/main.mjs restated the argument
# names as a TypeBox schema, and tools.py restated them again as a
# `_require_args(allowed=...)` tuple. 42 tools, kept aligned by hand, with the
# contract test locking only the name set — so the one real drift
# (`easyicu_run`'s host-only `llm_provider`) was invisible.


_MAIN_MJS = (
    Path(tools.__file__).with_name("node_app") / "src" / "main.mjs"
)


def _typebox_properties(source: str, tool_name: str) -> list[str]:
    """Top-level property names of one hostTool's `parameters` schema."""

    marker = f'name: "{tool_name}"'
    start = source.index(marker)
    tail = source[start:]
    params_at = tail.index("parameters:") + len("parameters:")
    rest = tail[params_at:].lstrip()
    if rest.startswith("empty"):
        return []
    assert rest.startswith("Type.Object("), (
        f"{tool_name} declares parameters in an unexpected shape"
    )

    def _balanced_end(text: str, open_at: int) -> int:
        depth = 0
        for index in range(open_at, len(text)):
            if text[index] in "({[":
                depth += 1
            elif text[index] in ")}]":
                depth -= 1
                if depth == 0:
                    return index
        raise AssertionError(f"unbalanced schema for {tool_name}")

    body_open = rest.index("(")
    body = rest[body_open + 1 : _balanced_end(rest, body_open)]
    object_open = body.index("{")
    inner = body[object_open + 1 : _balanced_end(body, object_open)]

    names: list[str] = []
    depth = 0
    buffer = ""
    for char in inner:
        if char in "({[":
            depth += 1
        elif char in ")}]":
            depth -= 1
        if char == "," and depth == 0:
            buffer = ""
            continue
        buffer += char
        if char == ":" and depth == 0:
            names.append(buffer[:-1].strip().strip('"'))
            buffer = ""
    return names


def test_every_catalogued_tool_declares_the_javascript_argument_names() -> None:
    source = _MAIN_MJS.read_text(encoding="utf-8")
    drift = {}
    for row in TOOL_CATALOG:
        declared = sorted(row.arguments.model)
        in_javascript = sorted(_typebox_properties(source, row.name))
        if declared != in_javascript:
            drift[row.name] = {"catalog": declared, "main.mjs": in_javascript}
    assert drift == {}, (
        "the catalog's `arguments.model` is the single declaration of what the "
        f"model may send; main.mjs has drifted from it: {drift}"
    )


def test_host_only_arguments_are_not_offered_to_the_model() -> None:
    """`host` keys must be absent from the JavaScript schema, by construction."""

    source = _MAIN_MJS.read_text(encoding="utf-8")
    for row in TOOL_CATALOG:
        if not row.arguments.host:
            continue
        in_javascript = set(_typebox_properties(source, row.name))
        assert not (set(row.arguments.host) & in_javascript), (
            f"{row.name} exposes a host-only argument to the model"
        )
    # easyicu_run is the reason this distinction exists: the host, not the
    # model, chooses the verified provider configuration.
    assert TOOL_ARGUMENTS["easyicu_run"].host == ("llm_provider",)


def test_handlers_no_longer_restate_their_own_argument_lists() -> None:
    """Ownership regression: the gate runs once, in execute_tool."""

    source = Path(tools.__file__).read_text(encoding="utf-8")
    # The one legitimate call is inside _require_catalog_args, which reads the
    # catalog. What must not come back is a handler passing its own literal.
    calls = re.findall(r"_require_args\(\s*params,\s*allowed=[\(\[\{\"']", source)
    assert calls == [], (
        "argument checking belongs to the catalog via _require_catalog_args; a "
        "handler that restates `allowed=` reintroduces the drift this replaced"
    )
    assert "_require_catalog_args(tool_name, arguments)" in source
