from __future__ import annotations

import pytest

from easyicu.research_agent.providers.strict_json_schema import (
    StrictJsonSchemaError,
    assert_closed_json_schema,
)


@pytest.mark.parametrize(
    "keyword",
    ("allOf", "anyOf", "oneOf", "prefixItems", "enum"),
)
def test_strict_schema_rejects_empty_structural_arrays(keyword: str) -> None:
    schema = {"type": "string", keyword: []}

    with pytest.raises(
        StrictJsonSchemaError,
        match=rf"keyword {keyword} must be non-empty",
    ):
        assert_closed_json_schema(schema)
