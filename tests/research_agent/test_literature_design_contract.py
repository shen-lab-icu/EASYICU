from __future__ import annotations

import pytest
from pydantic import ValidationError

from easyicu.research_agent.planning.literature_contract import (
    LiteratureDesignBinding,
)
from easyicu.research_agent.schema import (
    LiteratureDesignBinding as SchemaLiteratureDesignBinding,
)


def test_schema_facade_exports_the_planning_owned_contract() -> None:
    assert SchemaLiteratureDesignBinding is LiteratureDesignBinding


def test_literature_design_binding_rejects_duplicate_elements() -> None:
    with pytest.raises(ValidationError, match="design_elements must be unique"):
        LiteratureDesignBinding(
            citation_key="method_2025",
            design_elements=["estimand", "estimand"],
            application="Use the prespecified estimand definition.",
        )


def test_literature_design_binding_rejects_unstable_keys_and_extra_fields() -> None:
    with pytest.raises(ValidationError):
        LiteratureDesignBinding(
            citation_key="not a stable key",
            design_elements=["population"],
            application="Use the reported eligibility boundary.",
            invented_quote="unsupported",
        )
