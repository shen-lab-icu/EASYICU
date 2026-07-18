"""Tests for the L2 data-foundation catalog + coverage judgement."""

from __future__ import annotations

import json

import pandas as pd

from easyicu.research_agent.data_catalog import (
    AvailableCatalog,
    CatalogConcept,
    _methodology_tag,
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
    rep = assess_coverage(
        ["custom_score", "death"], cat, extra_available=["custom_score"]
    )
    assert rep.sufficient
    assert rep.resolved["custom_score"] == "custom_score"


def test_build_available_catalog_from_export_dir(tmp_path):
    # index_export_package indexes concept columns, skipping id/time columns.
    df = pd.DataFrame(
        {"stay_id": [1, 2], "charttime": [0, 1], "lact": [1.0, 2.0], "sofa2": [3, 4]}
    )
    df.to_parquet(tmp_path / "labs.parquet", index=False)
    (tmp_path / "_manifest.json").write_text(
        json.dumps(
            {
                "database": "miiv",
                "format": "parquet",
                "concept_selection": {"modules": {"labs": ["lact", "sofa2"]}},
                "files": [
                    {
                        "file": "labs.parquet",
                        "module": "labs",
                        "concepts": 2,
                        "concept_ids": ["lact", "sofa2"],
                        "rows": 2,
                    }
                ],
                "feature_definitions": {"included": False},
            }
        ),
        encoding="utf-8",
    )
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


def test_catalog_annotates_methodology_for_hazard_concepts():
    cat = AvailableCatalog(
        source="mem",
        concepts=[
            CatalogConcept(
                "death_icu",
                description="ICU mortality",
                category="outcome",
                methodology=_methodology_tag("death_icu", "outcome"),
            ),
            CatalogConcept(
                "norepi",
                description="norepinephrine",
                category="medications",
                methodology=_methodology_tag("norepi", "medications"),
            ),
            CatalogConcept(
                "sofa",
                description="SOFA score",
                category="outcome",
                methodology=_methodology_tag("sofa", "outcome"),
            ),
            CatalogConcept(
                "age",
                description="age",
                category="demographics",
                methodology=_methodology_tag("age", "demographics"),
            ),
        ],
    )
    rendered = cat.render_for_prompt()
    # The true endpoint gets a leakage caution; a vasopressor gets the
    # treatment caution; a plain demographic gets none.
    assert "leakage" in rendered
    assert "confounder vs mediator" in rendered
    # A SOFA score sits in the dictionary's "outcome" category but is NOT a study
    # endpoint -> it must NOT be mislabelled as a leakage outcome; it is a
    # derived score instead.
    sofa_line = next(line for line in rendered.splitlines() if "- sofa " in line)
    assert "leakage" not in sofa_line
    assert "derived" in sofa_line
    # The legend appears only because at least one concept carries a tag.
    assert "methodological cautions" in rendered


def test_build_available_catalog_populates_methodology(tmp_path, monkeypatch):
    import pandas as pd

    monkeypatch.setattr(
        "easyicu.research_agent.data_catalog._concept_dict_meta",
        lambda: {
            "norepi": {
                "description": "norepinephrine exposure",
                "category": "medications",
            }
        },
    )

    # Native manifests authorize only the selected concept (and its declared
    # companion columns).  Keep the fixture physically consistent with that
    # contract instead of relying on the legacy generic ``value`` column.
    pd.DataFrame({"stay_id": [1], "norepi": [1.0]}).to_parquet(
        tmp_path / "norepi.parquet"
    )
    (tmp_path / "_manifest.json").write_text(
        json.dumps(
            {
                "database": "miiv",
                "format": "parquet",
                "concept_selection": {"modules": {"vasopressors": ["norepi"]}},
                "files": [
                    {
                        "file": "norepi.parquet",
                        "module": "vasopressors",
                        "concepts": 1,
                        "concept_ids": ["norepi"],
                        "rows": 1,
                    }
                ],
                "feature_definitions": {"included": False},
            }
        ),
        encoding="utf-8",
    )
    cat = build_available_catalog(tmp_path)
    by_id = {c.concept_id: c for c in cat.concepts}
    # norepi is a medication -> treatment caution attached from the dictionary.
    assert by_id["norepi"].category == "medications"
    assert "treatment" in by_id["norepi"].methodology
