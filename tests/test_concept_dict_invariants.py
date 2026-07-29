"""Structural invariants on concept-dict.json.

These guard defect classes that produce *plausible numbers* rather than errors, so
nothing else in the pipeline notices them: no exception, no empty table, no QC trip.
Each test below corresponds to a real defect found on 2026-07-17.
"""
import json
from pathlib import Path

import pytest

DICT_PATH = Path(__file__).resolve().parents[1] / "src" / "easyicu" / "data" / "concept-dict.json"


@pytest.fixture(scope="module")
def concept_dict():
    with open(DICT_PATH, encoding="utf-8") as fh:
        return json.load(fh)


def test_rec_cncpt_aggregate_arity_matches_components(concept_dict):
    """`aggregate` is POSITIONAL: entry i applies to concepts[i].

    A short list is not a validation error anywhere in the engine -- it silently
    leaves the trailing components without an entry (concept/__init__.py
    _build_sub_aggregate: `if i < len(aggregate_spec)`), which
    _normalise_aggregators turns into 'auto', which resolves to *median* for a
    numeric column. The result is an index built from an extreme numerator over a
    median denominator: neither the worst case nor the central estimate, and
    directionally biased.

    Found 2026-07-17 in 10 concepts (shock_index, modified_shock_index,
    diastolic_shock_index, nlr, plr, bun_creatinine_ratio, oxygenation_index,
    anion_gap, egfr, susp_inf). Upstream ricu's idiom is one entry per component:
    `pafi {concepts: [po2, fio2], aggregate: [min, max]}`.
    """
    violations = []
    for name, block in concept_dict.items():
        if not isinstance(block, dict):
            continue
        components = block.get("concepts")
        aggregate = block.get("aggregate")
        if isinstance(components, list) and isinstance(aggregate, list):
            if len(aggregate) != len(components):
                violations.append(
                    f"{name}: {len(components)} components {components} "
                    f"but {len(aggregate)} aggregate entries {aggregate}"
                )
    assert not violations, (
        "aggregate must have exactly one entry per concepts entry (it is positional); "
        "a short list silently falls back to median for the trailing components:\n  "
        + "\n  ".join(violations)
    )


# Scale families that are NOT interchangeable. Declaring units from two different
# families on one concept means two numerically different scales reach one column.
# Deliberately narrow: only families whose confusion is a known, silent, large-factor
# killer. Case/separator variants ("IU/L" vs "U/l" vs "E/l", "mcg/kg/min" vs
# "mcgkgmin", "mm/hr" vs "mm/uur") are pure aliases and are not modelled here.
INCOMPATIBLE_SCALE_FAMILIES = {
    "mass_per_dl": {"mg/dl", "g/dl", "mcg/dl", "ug/dl", "µg/dl"},
    "molar": {"mmol/l", "umol/l", "µmol/l", "nmol/l"},
    "mass_per_l": {"g/l", "mg/l"},
    "pressure_mmhg": {"mmhg", "mm hg"},
    "pressure_kpa": {"kpa"},
}


def _families_present(units):
    lowered = {str(u).strip().lower() for u in units}
    return {
        family
        for family, members in INCOMPATIBLE_SCALE_FAMILIES.items()
        if lowered & members
    }


def _has_conversion_callback(block):
    for sources in (block.get("sources") or {}).values():
        for source in sources if isinstance(sources, list) else [sources]:
            callback = (source or {}).get("callback") or ""
            if "convert_unit" in callback or "binary_op" in callback:
                return True
    return False


def test_incompatible_unit_mix_has_a_conversion(concept_dict):
    """If a concept declares units from two incompatible scale families, some
    source must actually convert.

    Declaring ["mg/dL", "mmol/l"] does not make a concept legitimately bi-unit --
    it records a column that was never harmonised, and nothing downstream notices
    because both scales sit inside the min/max bounds. On 2026-07-17 this exact
    pattern hid a 10x error (total_protein: AUMC g/L pooled with g/dL) and an 88.6x
    error (trig: AUMC mmol/L pooled with mg/dL -- while its three lipid siblings
    cholesterol/hdl/ldl all carried x38.67 on the same database).

    This is deliberately weaker than "units must be equivalent": a concept may
    legitimately draw kPa from one database and mmHg from another *provided the
    diverging source converts* (etco2 does exactly this, via convert_unit on AUMC).
    What is forbidden is declaring the mix and converting nothing.
    """
    violations = []
    for name, block in concept_dict.items():
        if not isinstance(block, dict):
            continue
        units = block.get("unit")
        if not isinstance(units, list) or len(units) < 2:
            continue
        families = _families_present(units)
        if len(families) > 1 and not _has_conversion_callback(block):
            violations.append(f"{name}: {units} spans {sorted(families)} with no conversion callback")
    assert not violations, (
        "these concepts pool incompatible scales with nothing converting them:\n  "
        + "\n  ".join(violations)
    )


def test_ambiguous_cross_database_mappings_are_not_exported(concept_dict):
    """Semantic uncertainty must become explicit unavailability, not plausible data."""
    assert "aumc" not in concept_dict["adh_rate"]["sources"]
    assert "aumc" not in concept_dict["d_dimer"]["sources"]
    assert "hirid" not in concept_dict["neut"]["sources"]
    assert "hirid" not in concept_dict["lymph"]["sources"]

    for dataset in ("miiv", "mimic", "mimic_demo"):
        source = concept_dict["d_dimer"]["sources"][dataset][0]
        assert source["callback"] == "convert_unit(set_val(NA), 'ng/mL', 'FEU')"


def test_aumc_hba1c_scales_are_harmonised_before_pooling(concept_dict):
    sources = concept_dict["hba1c"]["sources"]["aumc"]
    by_itemid = {source["ids"]: source for source in sources}

    assert set(by_itemid) == {11812, 16166}
    assert by_itemid[11812]["callback"] == (
        "convert_unit(binary_op(`*`, 1), '%', 'Geen|%')"
    )
    callback = by_itemid[16166]["callback"]
    assert "binary_op(`*`, 0.09148)" in callback
    assert "binary_op(`+`, 2.152)" in callback
    assert concept_dict["hba1c"]["unit"] == "%"
    assert concept_dict["hba1c"]["min"] == 2
    assert concept_dict["hba1c"]["max"] == 25


def test_vasopressor_durations_are_non_negative_hours(concept_dict):
    """Export one explicit duration contract across all databases.

    Source systems occasionally contain end times before start times.  Without
    a lower bound those records survive as plausible numeric exposures, while
    the missing unit leaves downstream clients unable to compare databases.
    """
    for name in ("dobu_dur", "dopa_dur", "epi_dur", "norepi_dur"):
        assert concept_dict[name]["unit"] == "hours"
        assert concept_dict[name]["min"] == 0


def test_declared_component_concepts_exist(concept_dict):
    """Every rec_cncpt component must resolve to a concept that exists.

    Guards the other half of the 2026-07-17 plr fix: the callback was changed to
    consume `wbc`, and if the dict had not gained the matching component the merge
    would have raised at extraction time -- but only for the databases that got
    that far.
    """
    missing = []
    for name, block in concept_dict.items():
        if not isinstance(block, dict):
            continue
        components = block.get("concepts") or []
        # `concepts` may be a bare string for a single-component rec_cncpt --
        # iterating it directly would walk its characters.
        if isinstance(components, str):
            components = [components]
        for component in components:
            if component not in concept_dict:
                missing.append(f"{name} -> {component}")
    assert not missing, "component concepts referenced but not defined:\n  " + "\n  ".join(missing)
