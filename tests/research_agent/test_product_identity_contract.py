from __future__ import annotations

from easyicu.research_agent.contracts.declared_product import (
    typed_product as compatibility_typed_product,
)
from easyicu.research_agent.contracts.product_identity import (
    CANONICAL_TYPED_PRODUCT_TOKEN_PATTERN,
    canonical_product_kind,
    is_canonical_typed_product_token,
    normalize_product_token,
    typed_product,
)


def test_declared_product_facade_reexports_identity_owner() -> None:
    assert compatibility_typed_product is typed_product


def test_typed_product_canonicalizes_only_representation_aliases() -> None:
    assert typed_product("cohort:Analysis Set.parquet") == (
        "dataset",
        "analysis_set",
    )
    assert typed_product("plot:Adjusted OR.svg") == ("figure", "adjusted_or")
    assert typed_product("metric:Mortality Rate") == (
        "statistic",
        "mortality_rate",
    )
    assert typed_product("no-separator") is None


def test_product_token_and_kind_contracts_are_closed() -> None:
    assert normalize_product_token("  ICU stay / outcome ") == "icu_stay_outcome"
    assert canonical_product_kind("heatmap") == "figure"
    assert canonical_product_kind("report") == "report"


def test_canonical_typed_product_wire_grammar_is_strict_and_shared() -> None:
    assert CANONICAL_TYPED_PRODUCT_TOKEN_PATTERN.startswith("^")
    assert is_canonical_typed_product_token("table:primary_result") is True
    assert is_canonical_typed_product_token("table:Primary Result") is False
    assert is_canonical_typed_product_token("primary_result.csv") is False
    assert is_canonical_typed_product_token(7) is False
