from __future__ import annotations

import json
from pathlib import Path

import pytest

from easyicu.webserver.pi_copilot import tools
from easyicu.webserver.pi_copilot.tool_catalog import (
    ALLOWED_TOOLS,
    ALL_TOOL_NAMES,
    DATA_SOURCE_REQUIRED_TOOLS,
    MUTATING_HOST_TOOLS,
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
    ],
)
def test_catalog_rejects_extension_duplicate_and_unknown_policy(
    tmp_path: Path,
    mutation,
) -> None:
    payload = {
        "schema_version": "easyicu.pi-tool-catalog/1",
        "tools": [
            {
                "name": "easyicu_example",
                "surface": "research",
                "policy_group": "read",
                "execution_mode": "parallel",
                "host_mutating": False,
                "data_source_required": False,
            }
        ],
    }
    mutation(payload)
    path = tmp_path / "tool_catalog.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError):
        load_tool_catalog(path)
