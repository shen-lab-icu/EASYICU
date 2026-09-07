"""Figure-family routing has one owner, and the renderers declare what they own.

Six renderers already shared one clean interface. Choosing between them did
not: selection was split across four mechanisms in two modules — a dict in
`pipeline`, an if-chain beside it, the same dict restated 1,900 lines further
down, and a third table in `publication_bundles`. Ownership was signalled by
returning `None`, which is how figure policy came to live in an 8,000-line
pipeline module.

These tests pin the routing the four mechanisms produced, so the collapse into
one registry is a refactor and not a behaviour change.
"""

from __future__ import annotations

import pytest

from easyicu.research_agent.reporting.figure_bundle_registry import (
    FIGURE_BUNDLES,
    FigureBundleRegistry,
    FigureBundleRoute,
)


# Exactly the mapping the three key tables encoded before the collapse.
_FAMILY_TO_KEY = {
    "association": "association",
    "dose_response": "association",
    "prediction": "prediction",
    "prediction_model": "prediction",
    "survival": "survival",
    "survival_analysis": "survival",
    "cohort_definition": "cohort",
    "cohort_definition_sensitivity": "sensitivity",
    "sensitivity_analysis": "sensitivity",
    "missingness": "missingness",
    "measurement": "missingness",
    "data_quality": "missingness",
    "absolute_risk_context": "absolute_risk",
    "phenotyping": "phenotype",
    "clustering": "phenotype",
    "descriptive": "descriptive",
    "table_one": "descriptive",
    "baseline": "descriptive",
}
_METHOD_TO_KEY = {
    "ordinal_exposure_derivation_and_quality_control": "ordered_distribution",
    "exposure_distribution_and_missingness_audit": "distribution_availability",
    "cohort_definition_sensitivity": "sensitivity",
    "missingness": "missingness",
    "missingness_audit": "missingness",
    "missingness_measurement_audit": "missingness",
}


def _pipeline():
    import easyicu.research_agent.pipeline as pipeline

    return pipeline


@pytest.mark.parametrize("family", ["association", "sensitivity_analysis"])
def test_prior_output_renderers_have_lower_layer_owners_without_pipeline_providers(family, monkeypatch):
    key = FIGURE_BUNDLES._owner_of_family(family)
    route = FIGURE_BUNDLES._route(key)
    monkeypatch.setattr(route, "provider", None)
    monkeypatch.setattr(route, "renderer", None)
    renderer = FIGURE_BUNDLES.renderer_for_analysis_family(family)
    assert renderer.__module__.startswith("easyicu.research_agent.figures.")


def test_pipeline_does_not_implement_prior_output_rendering():
    import ast
    from pathlib import Path

    pipeline = _pipeline()
    definitions = {
        node.name for node in ast.parse(Path(pipeline.__file__).read_text()).body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert not definitions.intersection({
        "_render_association_publication_bundle_from_prior_outputs",
        "_render_sensitivity_publication_bundle_from_prior_outputs",
        "_resolve_upstream_analysis_method", "_planned_primary_association_contract",
    })


@pytest.mark.parametrize(("family", "key"), sorted(_FAMILY_TO_KEY.items()))
def test_every_analysis_family_routes_where_it_used_to(family, key) -> None:
    _pipeline()
    renderer = FIGURE_BUNDLES.renderer_for_analysis_family(family)
    assert renderer is not None, f"{family} lost its renderer"
    assert key.split("_")[0] in renderer.__name__


@pytest.mark.parametrize(("method", "key"), sorted(_METHOD_TO_KEY.items()))
def test_every_controlled_method_routes_where_it_used_to(method, key) -> None:
    _pipeline()
    renderer = FIGURE_BUNDLES.renderer_for_method(method)
    assert renderer is not None, f"{method} lost its renderer"
    assert key.split("_")[0] in renderer.__name__


def test_an_undeclared_family_or_method_owns_nothing() -> None:
    assert FIGURE_BUNDLES.renderer_for_analysis_family("not_a_family") is None
    assert FIGURE_BUNDLES.renderer_for_analysis_family(None) is None
    assert FIGURE_BUNDLES.renderer_for_method("") is None
    assert FIGURE_BUNDLES.renderer_for_figure_data_family("not_a_contract") is None


def test_the_only_figure_data_contract_is_the_ordered_distribution_one() -> None:
    renderer = FIGURE_BUNDLES.renderer_for_figure_data_family(
        "ordered_category_distribution"
    )
    assert renderer is not None
    assert "ordered_distribution" in renderer.__name__


def test_the_cohort_family_fallback_chains_are_the_declared_ones() -> None:
    _pipeline()
    chain = [
        renderer.__name__
        for renderer in FIGURE_BUNDLES.fallback_renderers_for_family(
            "cohort_definition_sensitivity"
        )
    ]
    assert chain == [
        "_render_sensitivity_publication_bundle_from_prior_outputs",
        "_render_cohort_overlap_publication_bundle_from_prior_outputs",
        "_render_cohort_flow_publication_bundle_from_prior_outputs",
    ]
    # A bare cohort_definition renders the exact closed product its parent
    # declared; it must never probe the overlap renderer first and let a schema
    # coincidence choose a different scientific display.
    assert [
        renderer.__name__
        for renderer in FIGURE_BUNDLES.fallback_renderers_for_family("cohort_definition")
    ] == ["_render_cohort_flow_publication_bundle_from_prior_outputs"]


def test_a_family_without_a_declared_chain_falls_back_to_its_own_renderer() -> None:
    _pipeline()
    chain = FIGURE_BUNDLES.fallback_renderers_for_family("survival")
    assert len(chain) == 1
    assert "survival" in chain[0].__name__


@pytest.mark.parametrize(
    "field",
    ["analysis_families", "methods", "figure_data_families"],
)
def test_a_declared_key_may_only_be_owned_once(field) -> None:
    """Two routes claiming one key used to resolve by dict order, not ownership.

    That is exactly how `missingness` was briefly claimed by both the
    sensitivity and the missingness route while still appearing to work.
    """
    registry = FigureBundleRegistry()
    registry.declare(FigureBundleRoute(key="a", **{field: ("shared",)}))
    with pytest.raises(ValueError, match="already owned"):
        registry.declare(FigureBundleRoute(key="b", **{field: ("shared",)}))


def test_a_route_key_may_only_be_declared_once() -> None:
    registry = FigureBundleRegistry()
    registry.declare(FigureBundleRoute(key="a"))
    with pytest.raises(ValueError, match="already declared"):
        registry.declare(FigureBundleRoute(key="a"))


def test_binding_or_chaining_an_undeclared_route_is_refused() -> None:
    registry = FigureBundleRegistry()
    with pytest.raises(KeyError):
        registry.bind("absent", lambda **_: None)
    with pytest.raises(KeyError):
        registry.declare_family_fallback("f", ("absent",))


def test_the_pipeline_helpers_read_the_registry() -> None:
    pipeline = _pipeline()
    for family in _FAMILY_TO_KEY:
        assert pipeline._renderer_for_upstream_family(
            family
        ) is FIGURE_BUNDLES.renderer_for_analysis_family(family)
    for method in _METHOD_TO_KEY:
        assert pipeline._renderer_for_upstream_method(
            method
        ) is FIGURE_BUNDLES.renderer_for_method(method)
