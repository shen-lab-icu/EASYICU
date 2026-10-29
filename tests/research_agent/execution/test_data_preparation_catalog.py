"""Preparation components bind executable owners, not extraction prose."""

import importlib

from easyicu.research_agent.contracts.method_kernels import CURATED_METHOD_KERNELS
from easyicu.research_agent.planning.analysis_method_suite import get_suite


def test_every_preparation_component_resolves_a_real_callable_kernel():
    suite = get_suite("data_preparation")
    assert suite is not None
    assert {item.key for item in suite.methods} == {
        "outlier_rules",
        "window_aggregation",
        "unit_harmonisation",
        "missing_data",
        "scaling_coding",
    }
    kernels = {item.module: item for item in CURATED_METHOD_KERNELS}
    for component in suite.methods:
        assert component.implementation == "deterministic"
        assert component.kernel_modules
        assert component.reporting_items
        for name in component.kernel_modules:
            kernel = kernels[name]
            module = importlib.import_module(kernel.import_path)
            for entrypoint in kernel.entrypoints:
                assert callable(getattr(module, entrypoint))


def test_public_catalog_names_the_family_and_honest_source_boundary():
    from easyicu.research_agent.method_skills import _method_component_catalog

    items, _ = _method_component_catalog(enabled=True)
    assert len(items) == 60
    assert {
        "association.proportional_odds", "association.free_days", "association.count_outcome",
        "prediction.external_validation",
        "cross_database.meta_analysis",
        "association.mixed_effects", "association.gee",
        "phenotyping.mixed_effects_growth",
    }.issubset({row["id"] for row in items})
    preparation = [row for row in items if row["method_family"] == "data_preparation"]
    assert len(preparation) == 5
    assert {item["category_zh"] for item in preparation} == {"数据预处理"}
    assert all(item["claim_ceiling"] == "analysis_only" for item in preparation)
    assert (
        "raw"
        in next(row for row in preparation if row["method_key"] == "outlier_rules")[
            "description"
        ].lower()
    )
