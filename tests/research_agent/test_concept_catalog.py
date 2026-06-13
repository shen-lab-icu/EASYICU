"""Tests for the dictionary-driven concept catalog (idea-mining layers 1/2/3)."""

from __future__ import annotations

import json

from easyicu.research_agent.concept_catalog import (
    DERIVED_CONCEPT_HINTS,
    ConceptCatalog,
    load_concept_catalog,
)


def test_aliases_map_concept_keys_to_literature_phrasing() -> None:
    cat = load_concept_catalog()
    # description "norepinephrine rate" -> short alias "norepinephrine"
    assert any("norepinephrine" == a.lower() for a in cat.concept_aliases["norepi_rate"])
    # "vasopressin rate" -> "vasopressin"
    assert any("vasopressin" == a.lower() for a in cat.concept_aliases["adh_rate"])
    # "Positive End Expiratory Pressure"
    assert any(
        "positive end expiratory pressure" == a.lower()
        for a in cat.concept_aliases["peep"]
    )


def test_mortality_description_expands_to_mortality_alias() -> None:
    cat = load_concept_catalog()
    aliases = [a.lower() for a in cat.concept_aliases["death"]]
    assert "mortality" in aliases
    assert "hospital mortality" in aliases
    assert "in-hospital mortality" in aliases
    assert "intensive-care unit mortality" in aliases
    assert "intensive care unit mortality" in aliases


def test_binary_outcomes_are_declared_known_0_1() -> None:
    cat = load_concept_catalog()
    for key in ("death", "susp_inf", "sep3"):
        assert cat.outcome_determinability[key]["status"] == "known_0_1"


def test_intervention_logical_concepts_are_not_default_outcomes() -> None:
    cat = load_concept_catalog()
    for key in ("rrt", "vaso_ind", "vent_ind", "mech_vent"):
        assert key not in cat.outcome_determinability


def test_administration_exposure_concepts_are_not_default_outcomes() -> None:
    cat = load_concept_catalog()
    for key in ("norepi60", "epi60", "dopa60"):
        assert key not in cat.outcome_determinability


def test_ordinal_scores_are_determinable_but_not_binary() -> None:
    cat = load_concept_catalog()
    # SOFA / qSOFA / components are 0-N scales: not 0/1 outcomes, but the
    # present/NA coding trap does not apply, so they are determinable outcomes
    # (must leave the dry run executable, not gated out as "unknown").
    for key in ("sofa", "qsofa", "sofa_cardio", "mews", "news"):
        assert cat.outcome_determinability[key]["status"] == "non_binary_determinable"


def test_continuous_outcomes_are_determinable_but_not_binary() -> None:
    cat = load_concept_catalog()
    for key in ("los_icu", "los_hosp"):
        assert cat.outcome_determinability[key]["status"] == "non_binary_determinable"


def test_treatment_exposure_concepts_stay_undeterminable() -> None:
    # Using a treatment/exposure as an outcome is genuinely ambiguous; the
    # conservative block (no determinability spec -> "unknown" gate) is kept.
    cat = load_concept_catalog()
    for key in ("rrt", "vaso_ind", "norepi60"):
        assert key not in cat.outcome_determinability


def test_colliding_literature_aliases_are_suppressed(tmp_path) -> None:
    concept_dict = {
        "death_icu": {
            "class_name": "lgl_cncpt",
            "description": "ICU mortality",
            "category": "outcome",
        },
        "death_hosp": {
            "class_name": "lgl_cncpt",
            "description": "hospital mortality",
            "category": "outcome",
        },
    }
    dict_path = tmp_path / "concept-dict.json"
    dict_path.write_text(json.dumps(concept_dict), encoding="utf-8")

    cat = load_concept_catalog(dict_paths=[dict_path])

    lower_aliases = {
        key: {alias.lower() for alias in aliases}
        for key, aliases in cat.concept_aliases.items()
    }
    assert "mortality" not in lower_aliases["death_icu"]
    assert "mortality" not in lower_aliases["death_hosp"]
    assert "hospital mortality" in lower_aliases["death_hosp"]


def test_derived_concepts_absent_from_dicts_get_hint_metadata() -> None:
    # aki etc. live in code, not the JSON dicts; the hint table must supply them.
    cat = load_concept_catalog(restrict_to=["aki", "aki_stage_rrt", "norepi_rate", "death"])
    assert "acute kidney injury" in [a.lower() for a in cat.concept_aliases["aki"]]
    assert cat.outcome_determinability["aki"]["status"] == "known_0_1"
    assert cat.outcome_determinability["aki_stage_rrt"]["status"] == "known_0_1"


def test_restrict_to_limits_available_concepts() -> None:
    cat = load_concept_catalog(restrict_to=["norepi_rate", "death", "not_a_real_concept"])
    assert set(cat.available_concepts) == {"norepi_rate", "death", "not_a_real_concept"}
    # unknown concept yields no alias / outcome, but does not crash.
    assert "not_a_real_concept" not in cat.outcome_determinability


def test_extra_aliases_merge_on_top_of_derived() -> None:
    cat = load_concept_catalog(
        restrict_to=["norepi_rate"],
        extra_aliases={"norepi_rate": ["noradrenaline"]},
    )
    aliases = [a.lower() for a in cat.concept_aliases["norepi_rate"]]
    assert "noradrenaline" in aliases
    assert "norepinephrine" in aliases  # derived one preserved


def test_catalog_aliases_let_resolver_bind_literature_phrases() -> None:
    # Integration with the idea_mining resolver: the derived aliases must
    # actually let an LLM-written phrase resolve to the concept key.
    from easyicu.research_agent.idea_mining import (
        _build_concept_lookup,
        _resolve_concept,
    )

    cat = load_concept_catalog(
        restrict_to=["norepi_rate", "adh_rate", "rrt", "aki", "death", "peep"]
    )
    lookup = _build_concept_lookup(
        cat.available_concepts, concept_aliases=cat.concept_aliases
    )
    assert _resolve_concept("early norepinephrine initiation", lookup) == "norepi_rate"
    assert _resolve_concept("vasopressin administration", lookup) == "adh_rate"
    assert _resolve_concept("renal replacement therapy", lookup) == "rrt"
    # the resolver canonicalizes aki -> kdigo_aki; the outcome table registers
    # both forms so determinability lookups still hit (see test below).
    assert _resolve_concept("acute kidney injury", lookup) == "kdigo_aki"


def test_aki_outcome_determinability_registered_under_canonical_key() -> None:
    cat = load_concept_catalog(restrict_to=["aki"])
    # both the manifest key and the resolver's canonical form resolve to a spec
    assert cat.outcome_determinability["aki"]["status"] == "known_0_1"
    assert cat.outcome_determinability["kdigo_aki"]["status"] == "known_0_1"


def test_hint_table_is_nonempty_and_well_formed() -> None:
    assert DERIVED_CONCEPT_HINTS
    for key, (aliases, is_binary) in DERIVED_CONCEPT_HINTS.items():
        assert isinstance(aliases, list) and aliases, f"{key} has no aliases"
        assert isinstance(is_binary, bool)


def test_catalog_is_frozen_dataclass() -> None:
    cat = load_concept_catalog(restrict_to=["death"])
    assert isinstance(cat, ConceptCatalog)
    import dataclasses

    assert dataclasses.is_dataclass(cat)


def test_resolver_prefers_most_specific_concept_over_incidental_token() -> None:
    # With the FULL dictionary, both vent_ind ("ventilation durations") and
    # kdigo_aki ("acute kidney injury") are candidates for an outcome phrased
    # "ventilation-induced acute kidney injury". The resolver must bind it to AKI
    # (3-token semantic match), NOT vent_ind (incidental 1-token "ventilation").
    # Guards against regressing _resolve_concept back to first-subset-hit.
    from easyicu.research_agent.idea_mining import (
        _build_concept_lookup,
        _resolve_concept,
    )

    cat = load_concept_catalog()
    lookup = _build_concept_lookup(
        cat.available_concepts, concept_aliases=cat.concept_aliases
    )
    assert (
        _resolve_concept("ventilation-induced acute kidney injury", lookup)
        == "kdigo_aki"
    )


def _ambiguous_idea(**overrides):
    from easyicu.research_agent.idea_mining import LiteratureIdeaCandidate

    base = dict(
        source_snapshot_id="snap1",
        citation_key="cite1",
        source_adapter_level="user_supplied_excerpt",
        population="adults with septic shock",
        # the phrase contains TWO concept words: norepinephrine (the real
        # exposure) and lactate (a subgroup qualifier). Without a core concept
        # the resolver can bind to either.
        exposure_or_predictor=(
            "early norepinephrine within 1 h in patients with lactate <= 3 mmol/L"
        ),
        outcome="intensive-care unit mortality",
        rationale="trial sequential analysis indicated more data are needed",
        source_quote="more data are needed",
    )
    base.update(overrides)
    return LiteratureIdeaCandidate(**base)


def test_core_concept_disambiguates_exposure_from_qualifier() -> None:
    # exposure_core_concept must steer resolution to norepinephrine, NOT the
    # incidental "lactate" qualifier sharing the phrase.
    from easyicu.research_agent.idea_mining import (
        map_literature_idea_to_executable_candidate,
    )

    cat = load_concept_catalog()
    idea = _ambiguous_idea(
        exposure_core_concept="norepinephrine",
        outcome_core_concept="ICU mortality",
    )
    exe = map_literature_idea_to_executable_candidate(
        idea,
        available_concepts=cat.available_concepts,
        concept_aliases=cat.concept_aliases,
        outcome_determinability=cat.outcome_determinability,
    )
    assert exe.resolved_predictor_concept in {
        "norepi_rate",
        "norepi_dur",
        "norepi60",
        "norepi_equiv",
    }
    assert exe.resolved_predictor_concept != "lact"
    assert exe.resolved_outcome_concept == "death"


def test_missing_core_concept_falls_back_to_full_phrase() -> None:
    # back-compat: when the model omits core fields, mapping must still run
    # (resolution may be ambiguous, but must not crash or regress the schema).
    from easyicu.research_agent.idea_mining import (
        map_literature_idea_to_executable_candidate,
    )

    cat = load_concept_catalog()
    idea = _ambiguous_idea()  # no core concepts supplied
    exe = map_literature_idea_to_executable_candidate(
        idea,
        available_concepts=cat.available_concepts,
        concept_aliases=cat.concept_aliases,
        outcome_determinability=cat.outcome_determinability,
    )
    assert exe is not None  # resolves without error regardless of ambiguity


def _full_lookup():
    from easyicu.research_agent.idea_mining import _build_concept_lookup

    cat = load_concept_catalog()
    return _build_concept_lookup(
        cat.available_concepts, concept_aliases=cat.concept_aliases
    )


def test_uk_us_spelling_and_term_synonyms_resolve() -> None:
    # The dictionary only carries US spellings; literature (esp. ICM/Lancet)
    # uses UK spellings and equivalent terms that must still bind.
    from easyicu.research_agent.idea_mining import _resolve_concept

    lookup = _full_lookup()
    assert _resolve_concept("noradrenaline", lookup) == "norepi_rate"
    assert _resolve_concept("adrenaline", lookup) == "epi_rate"
    assert _resolve_concept("antidiuretic hormone", lookup) == "adh_rate"
    assert _resolve_concept("dialysis", lookup) == "rrt"
    assert _resolve_concept("CRRT", lookup) == "rrt"


def test_synonym_group_fixes_pf_ratio_misresolution() -> None:
    # "P/F ratio" previously mis-resolved to inr_pt via the bare "ratio" token;
    # the oxygenation synonym group binds it to the Horowitz-index concept pafi.
    from easyicu.research_agent.idea_mining import _resolve_concept

    lookup = _full_lookup()
    assert _resolve_concept("P/F ratio", lookup) == "pafi"
    assert _resolve_concept("PaO2/FiO2", lookup) == "pafi"


def test_parenthetical_brand_and_chemical_names_become_aliases() -> None:
    from easyicu.research_agent.idea_mining import _resolve_concept

    lookup = _full_lookup()
    assert _resolve_concept("Lasix", lookup) == "furosemide"
    assert _resolve_concept("acetylsalicylic acid", lookup) == "aspirin"


def test_bracket_extraction_filters_dosing_route_and_formula_noise() -> None:
    from easyicu.research_agent.concept_catalog import _bracket_aliases

    assert _bracket_aliases("human albumin IV (5%, 20%, 25%) - colloid") == []
    assert _bracket_aliases("serum anion gap (Na - Cl - HCO3)") == []
    assert _bracket_aliases("aspirin (any route)") == []
    # a genuine brand/chemical name is still kept
    assert "Lasix" in _bracket_aliases("furosemide (Lasix) administration")
