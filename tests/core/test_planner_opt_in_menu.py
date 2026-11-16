"""Opt-in export modules have one owner, and the Planner's dictionary fallback
lists only what a default export carries (thirteenth reseal, 2026-11-16)."""

from easyicu.api import extraction as api
from easyicu.concept.catalog import CONCEPT_GROUPS_INTERNAL, OPT_IN_MODULE_CONCEPTS, OPT_IN_ONLY_CONCEPTS
from easyicu.concept.native_events import NATIVE_EVENT_MODULES
from easyicu.research_agent.agents.planner import _format_concept_id_allowlist


def test_every_opt_in_module_comes_from_the_catalog_roster():
    opt_in = set(api.EXTRACT_MODULES) - set(api.EXTRACT_MODULE_ORDER) - set(NATIVE_EVENT_MODULES)
    assert opt_in == set(OPT_IN_MODULE_CONCEPTS)
    assert all(api.EXTRACT_MODULES[m] == list(c) for m, c in OPT_IN_MODULE_CONCEPTS.items())
    assert not set(OPT_IN_MODULE_CONCEPTS) & set(api.EXTRACT_MODULE_ORDER)


def test_opt_in_only_concepts_are_exactly_those_no_default_module_exports():
    defaults = {c for concepts in CONCEPT_GROUPS_INTERNAL.values() for c in concepts}
    assert not OPT_IN_ONLY_CONCEPTS & defaults
    # map_invasive is published by vitals and by haemo_native: it stays on the menu.
    assert "map_invasive" in defaults and "map_invasive" not in OPT_IN_ONLY_CONCEPTS
    assert {"sbp_invasive", "crystalloid_input_ml", "il6", "weight_admit_strict"} <= OPT_IN_ONLY_CONCEPTS


def test_the_fallback_menu_omits_opt_in_only_concepts():
    menu = _format_concept_id_allowlist([])
    assert "a default export carries" in menu
    for name in ("sbp_invasive", "crystalloid_input_ml", "il6"):
        assert f"`{name}`" not in menu
    for name in ("hgb", "map", "map_invasive"):
        assert f"`{name}`" in menu


def test_an_export_that_carries_an_opt_in_concept_still_lists_it():
    menu = _format_concept_id_allowlist(["hgb", "il6"])
    assert "bind against this run's sealed export" in menu
    assert "`il6`" in menu and "`hgb`" in menu
