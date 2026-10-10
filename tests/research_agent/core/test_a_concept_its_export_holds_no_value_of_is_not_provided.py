"""A concept its export lists but holds no value of is not provided.

A native export states, per concept, whether its producer wrote any value
(``produced_all_null``) or the source cannot hold it and the export wrote a
placeholder (``structurally_unavailable_placeholder``).  The column exists, but
a materialized event status reads its missing rows as "did not occur", so a
study analysing it would report an absence the source never recorded.  The
source catalog does not offer such a concept, so an acquisition that needs it
as an outcome or a covariate stops and names it.  Synthetic exports only.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from easyicu.research_agent.acquisition.catalog import build_available_catalog
from easyicu.research_agent.acquisition.foundation import acquire_universe_for_question
from easyicu.research_agent.intake.export_package import (
    LEGACY_MANIFEST,
    NATIVE_MANIFEST,
    concepts_without_values,
)
from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient
from tests.support.native_outcome_export import native_outcome, untyped_native_export


def _stated(root: Path, statuses: dict[str, dict[str, str]]) -> Path:
    """Add each file's per-concept availability, as the native exporter writes it."""

    manifest = json.loads((root / NATIVE_MANIFEST).read_text(encoding="utf-8"))
    for entry in manifest["files"]:
        entry["concept_status"] = {
            concept: {
                "availability": availability,
                "non_null": 0 if availability != "available" else 4,
                "excluded_out_of_bounds": 0,
            }
            for concept, availability in statuses.get(entry["module"], {}).items()
        }
    (root / NATIVE_MANIFEST).write_text(json.dumps(manifest), encoding="utf-8")
    return root


def _export(tmp_path: Path, *, stated: bool = True) -> Path:
    root = untyped_native_export(
        tmp_path / "export",
        outcome=native_outcome(death=[0, 1, 0, 1], circ_failure=[None, None, None, None]),
        outcome_concepts=["death", "circ_failure"],
    )
    if not stated:
        return root
    return _stated(
        root,
        {
            "demographics": {"age": "available"},
            "outcome": {"death": "available", "circ_failure": "produced_all_null"},
        },
    )


def test_the_export_names_the_concepts_it_holds_no_value_of(tmp_path: Path) -> None:
    root = tmp_path / "manifest_only"
    root.mkdir()
    (root / NATIVE_MANIFEST).write_text(
        json.dumps(
            {
                "database": "miiv",
                "files": [
                    {"module": "chemistry", "concept_status": {
                        "lact": {"availability": "available"},
                        "tri": {"availability": "produced_all_null"},
                        "pct": {"availability": "structurally_unavailable_placeholder"},
                        "alb": {"availability": "produced_all_null"},
                    }},
                    # A concept another file holds a value of has a value.
                    {"module": "renal", "concept_status": {
                        "alb": {"availability": "available"},
                    }},
                ],
            }
        ),
        encoding="utf-8",
    )
    assert concepts_without_values(root) == ("pct", "tri")

    legacy = tmp_path / "legacy"
    legacy.mkdir()
    (legacy / LEGACY_MANIFEST).write_text(json.dumps({"database": "miiv"}), encoding="utf-8")
    # A legacy manifest states no availability: its rows are the only evidence.
    assert concepts_without_values(legacy) == ()


def test_the_source_catalog_does_not_offer_a_concept_without_values(tmp_path: Path) -> None:
    assert "circ_failure" not in build_available_catalog(_export(tmp_path)).ids()
    assert "death" in build_available_catalog(tmp_path / "export").ids()
    # The same column, with no statement against it, is offered.
    unstated = tmp_path / "unstated"
    unstated.mkdir()
    assert "circ_failure" in build_available_catalog(
        _export(unstated, stated=False)
    ).ids()


@pytest.mark.parametrize(
    ("arguments", "reason"),
    [
        pytest.param(
            {"target_outcome": "circ_failure", "outcome_concepts": ["circ_failure"]},
            "outcome_concept_unavailable",
            id="outcome",
        ),
        pytest.param(
            {
                "target_outcome": "death",
                "outcome_concepts": ["death"],
                "required_feature_concepts": ["circ_failure"],
            },
            "required_concepts_unavailable",
            id="covariate",
        ),
    ],
)
def test_an_event_concept_without_values_enters_no_study(
    tmp_path: Path, arguments: dict, reason: str
) -> None:
    llm = ScriptedMockLLMClient([])
    result = acquire_universe_for_question(
        export_dir=_export(tmp_path),
        question="Does circulatory failure in the first day relate to death?",
        llm=llm,
        output_dir=tmp_path / "universe",
        concept_selection_authority="host_exact",
        **arguments,
    )

    assert result.blocked
    assert result.blocked_reason_code == reason
    assert tuple(result.missing_concepts) == ("circ_failure",)
    assert llm.calls == []
