"""Tests for the L2 data-foundation catalog + coverage judgement."""
from __future__ import annotations

import pandas as pd

from easyicu.research_agent.data_catalog import (
    AvailableCatalog,
    CatalogConcept,
    assess_coverage,
    build_available_catalog,
)


def _catalog(*ids: str) -> AvailableCatalog:
    return AvailableCatalog(
        source="mem",
        concepts=[CatalogConcept(concept_id=i, category="labs") for i in ids],
    )


def test_coverage_sufficient_when_all_present():
    cat = _catalog("lact", "sofa2", "death")
    rep = assess_coverage(["lact", "sofa2", "death"], cat)
    assert rep.sufficient
    assert rep.missing == []
    assert set(rep.available) == {"lact", "sofa2", "death"}


def test_coverage_flags_missing_with_reextract_advice():
    cat = _catalog("lact", "death")
    rep = assess_coverage(["lact", "death", "troponin"], cat)
    assert not rep.sufficient
    assert rep.missing == ["troponin"]
    assert rep.advice and "troponin" in rep.advice[0]
    assert "re-extract" in rep.advice[0].lower()


def test_coverage_conservative_alias_resolves_unique_variant():
    # `sep3` exported only as `sep3_sofa2` -> unique suffix resolves.
    cat = _catalog("sep3_sofa2", "death")
    rep = assess_coverage(["sep3", "death"], cat)
    assert rep.sufficient
    assert rep.resolved["sep3"] == "sep3_sofa2"


def test_coverage_ambiguous_alias_does_not_resolve():
    # `los` -> los_icu / los_hosp is ambiguous and must NOT silently resolve.
    cat = _catalog("los_icu", "los_hosp", "death")
    rep = assess_coverage(["los", "death"], cat)
    assert "los" in rep.missing


def test_extra_available_marks_provided_cohort_columns_present():
    # A pre-filtered cohort parquet carries its own columns; mark them present.
    cat = _catalog("death")
    rep = assess_coverage(["custom_score", "death"], cat, extra_available=["custom_score"])
    assert rep.sufficient
    assert rep.resolved["custom_score"] == "custom_score"


def test_build_available_catalog_from_export_dir(tmp_path):
    # index_export_package indexes concept columns, skipping id/time columns.
    df = pd.DataFrame(
        {"stay_id": [1, 2], "charttime": [0, 1], "lact": [1.0, 2.0], "sofa2": [3, 4]}
    )
    df.to_parquet(tmp_path / "labs.parquet", index=False)
    cat = build_available_catalog(tmp_path)
    ids = set(cat.ids())
    assert {"lact", "sofa2"} <= ids
    assert "stay_id" not in ids and "charttime" not in ids


def test_render_for_prompt_groups_by_category_and_lists_ids():
    cat = AvailableCatalog(
        source="mem",
        concepts=[
            CatalogConcept("lact", description="lactate", category="blood gas"),
            CatalogConcept("sofa2", description="SOFA-2", category="scores"),
        ],
    )
    rendered = cat.render_for_prompt()
    assert "[blood gas]" in rendered and "[scores]" in rendered
    assert "lact" in rendered and "lactate" in rendered
