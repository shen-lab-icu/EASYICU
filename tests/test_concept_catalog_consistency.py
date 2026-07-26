from __future__ import annotations

import json
import re
from pathlib import Path

from easyicu.datasource import (
    AUMC_NUMERICITEMS_ITEMIDS,
    HIRID_OBSERVATIONS_VARIABLEIDS,
    MIIV_CHARTEVENTS_ITEMIDS,
    MIIV_LABEVENTS_ITEMIDS,
    MIMIC_DEMO_CHARTEVENTS_ITEMIDS,
    MIMIC_DEMO_LABEVENTS_ITEMIDS,
)
from easyicu.resources import load_dictionary
from easyicu.concept.catalog import (
    COMPOSITE_CONCEPT_OUTPUT_SOURCES,
    CONCEPT_DB_COVERAGE,
    CONCEPT_DESCRIPTIONS,
    CONCEPT_DICTIONARY,
    CONCEPT_GROUP_NAMES,
    CONCEPT_GROUPS_INTERNAL,
    HIDDEN_DICTIONARY_CONCEPTS,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "src" / "easyicu" / "data"
BENCHMARK_DIR = REPO_ROOT / "benchmark"
STATIC_DATA_CATALOG_JS = (
    REPO_ROOT / "src" / "easyicu" / "webserver" / "static" / "js" / "data-catalog.js"
)
SOURCE_FIELD_KEYS = {
    "amount_var",
    "auom_var",
    "aux_time",
    "dir_var",
    "dur_var",
    "end_var",
    "grp_var",
    "id_var",
    "index_var",
    "rate_var",
    "stop_var",
    "sub_var",
    "unit_var",
    "val_var",
    "value_var",
    "weight_var",
}


def _load_json(filename: str) -> dict | list:
    return json.loads((DATA_DIR / filename).read_text())


def _data_source_tables() -> dict[str, dict]:
    data_sources = _load_json("data-sources.json")
    return {source["name"]: source["tables"] for source in data_sources}


def _miiv_source_catalog() -> set[tuple[str, int]]:
    payload = json.loads((BENCHMARK_DIR / "source_item_catalog_miiv.json").read_text())
    catalog: set[tuple[str, int]] = set()
    for item in payload.get("items", []):
        itemid = item.get("itemid")
        table = str(item.get("table", ""))
        if "/" in table:
            table = table.split("/", 1)[1]
        if isinstance(itemid, int) and table:
            catalog.add((table, itemid))
    return catalog


def _aumc_numericitems_prefilter_ids() -> set[int]:
    return set(AUMC_NUMERICITEMS_ITEMIDS)


def _source_ids(source_def: dict) -> set[int]:
    ids = source_def.get("ids")
    if isinstance(ids, int):
        return {ids}
    if isinstance(ids, list):
        return {item for item in ids if isinstance(item, int)}
    return set()


def _dictionary_source_ids(dataset: str, table: str, sub_var: str) -> set[int]:
    source_ids: set[int] = set()
    for filename in ("concept-dict.json", "sofa2-dict.json"):
        dictionary = _load_json(filename)
        for concept_def in dictionary.values():
            sources = concept_def.get("sources")
            if not isinstance(sources, dict):
                continue
            for source_def in sources.get(dataset, []):
                if (
                    source_def.get("table") == table
                    and source_def.get("sub_var") == sub_var
                ):
                    source_ids.update(_source_ids(source_def))
    return source_ids


def _concept_source_ids(concept: str, dataset: str, table: str, sub_var: str) -> set[int]:
    source_ids: set[int] = set()
    for filename in ("concept-dict.json", "sofa2-dict.json"):
        dictionary = _load_json(filename)
        concept_def = dictionary.get(concept)
        if not isinstance(concept_def, dict):
            continue
        sources = concept_def.get("sources")
        if not isinstance(sources, dict):
            continue
        for source_def in sources.get(dataset, []):
            if (
                source_def.get("table") == table
                and source_def.get("sub_var") == sub_var
            ):
                source_ids.update(_source_ids(source_def))
    return source_ids


def test_web_catalog_groups_are_unique_and_complete() -> None:
    grouped = [concept for concepts in CONCEPT_GROUPS_INTERNAL.values() for concept in concepts]

    assert len(CONCEPT_GROUPS_INTERNAL) == 19
    # 2026-07-17: +vent_mode, +vent_breath_seq (harmonised ventilator-mode concepts,
    # grouped under 'ventilator'). 277 -> 279.
    # 2026-07-17: +cvp into the 'vitals' group (was only in concept-dict.json /
    # pulled by a separate cvp_extraction; central venous pressure is a measured
    # vital and now extracts with the vitals module). 280 -> 281.
    assert len(CONCEPT_DICTIONARY) == 281
    assert set(CONCEPT_GROUP_NAMES) >= set(CONCEPT_GROUPS_INTERNAL)
    assert len(grouped) == len(set(grouped))
    assert set(grouped) == set(CONCEPT_DICTIONARY)


def _parse_js_object_block(source: str, var: str) -> dict[str, str]:
    """Extract ``const <var> = { key: <value>, ... };`` -> {key: raw_value_text}.

    Independent of the generator so a generator bug can't hide drift. Assumes the
    block bodies contain no nested ``{}`` (true for groupConcepts/dict/cov/desc).
    """
    marker = f"const {var} = {{"
    start = source.index(marker) + len(marker)
    depth = 1
    i = start
    while i < len(source) and depth:
        if source[i] == "{":
            depth += 1
        elif source[i] == "}":
            depth -= 1
        i += 1
    body = source[start : i - 1]
    out: dict[str, str] = {}
    for key, value in re.findall(r"(\w+)\s*:\s*(\[[^\]]*\]|\d+)", body):
        out[key] = value
    return out


def _js_list_items(raw: str) -> list[str]:
    return re.findall(r"""['"]([^'"]*)['"]""", raw)


def test_static_data_catalog_js_matches_python_catalog() -> None:
    """The static bootstrap (data-catalog.js) must carry the SAME feature set as
    the Python SSOT that /api/catalog serves. Regenerate with
    ``python tools/generate_static_catalog.py`` when this fails."""
    js = STATIC_DATA_CATALOG_JS.read_text(encoding="utf-8")

    web_group_concepts = {
        module: _js_list_items(raw)
        for module, raw in _parse_js_object_block(js, "groupConcepts").items()
    }
    web_dict = set(_parse_js_object_block(js, "dict"))
    web_cov = set(_parse_js_object_block(js, "cov"))
    web_desc = set(_parse_js_object_block(js, "desc"))

    hint = "run: python tools/generate_static_catalog.py"

    # groupConcepts: exact module -> ordered-members parity with the catalog.
    assert web_group_concepts == {
        module: list(concepts)
        for module, concepts in CONCEPT_GROUPS_INTERNAL.items()
    }, hint

    # dict / cov / desc: same key sets as their Python sources.
    assert web_dict == set(CONCEPT_DICTIONARY), hint
    assert web_cov == set(CONCEPT_DB_COVERAGE), hint
    assert web_desc == set(CONCEPT_DESCRIPTIONS), hint

    # web internal consistency: every grouped concept has a label; no orphan label.
    grouped = [c for members in web_group_concepts.values() for c in members]
    assert set(grouped) == web_dict, hint
    assert len(grouped) == len(set(grouped)), "duplicate concept in web groupConcepts"


def test_extract_database_modules_use_the_shared_catalog() -> None:
    from easyicu.api import EXTRACT_MODULE_ORDER, EXTRACT_MODULES

    assert EXTRACT_MODULES == {
        module: list(concepts)
        for module, concepts in CONCEPT_GROUPS_INTERNAL.items()
    }
    assert set(EXTRACT_MODULE_ORDER) == set(CONCEPT_GROUPS_INTERNAL)
    assert len(EXTRACT_MODULE_ORDER) == len(CONCEPT_GROUPS_INTERNAL) == 19


def test_web_catalog_aligns_with_merged_extraction_dictionary() -> None:
    dictionary = load_dictionary(include_sofa2=True)
    dict_concepts = set(dictionary.keys())
    web_concepts = set(CONCEPT_DICTIONARY)

    unresolved_web_concepts = web_concepts - dict_concepts - set(COMPOSITE_CONCEPT_OUTPUT_SOURCES)
    hidden_dict_concepts = dict_concepts - web_concepts

    assert unresolved_web_concepts == set()
    assert hidden_dict_concepts <= HIDDEN_DICTIONARY_CONCEPTS


def test_composite_output_sources_are_valid() -> None:
    dictionary = load_dictionary(include_sofa2=True)
    dict_concepts = set(dictionary.keys())
    special_sources = {"circ_failure_loader", "comorbidity_loader",
                       "outcomes_loader", "microbiology_loader"}

    for output_concept, source_concept in COMPOSITE_CONCEPT_OUTPUT_SOURCES.items():
        assert output_concept in CONCEPT_DICTIONARY
        assert source_concept in dict_concepts or source_concept in special_sources


def test_dictionary_source_tables_and_columns_exist() -> None:
    """Pin source-table wiring without needing local ICU data extracts."""
    data_source_tables = _data_source_tables()
    problems: list[str] = []

    for filename in ("concept-dict.json", "sofa2-dict.json"):
        dictionary = _load_json(filename)
        for concept_name, concept_def in dictionary.items():
            sources = concept_def.get("sources")
            if not isinstance(sources, dict):
                continue

            for database, source_defs in sources.items():
                database_tables = data_source_tables.get(database)
                if database_tables is None:
                    problems.append(f"{filename}:{concept_name}:{database}: missing database")
                    continue

                for source_index, source_def in enumerate(source_defs):
                    table_name = source_def.get("table")
                    if not table_name:
                        continue
                    table_def = database_tables.get(table_name)
                    if table_def is None:
                        problems.append(
                            f"{filename}:{concept_name}:{database}[{source_index}] "
                            f"missing table {table_name}"
                        )
                        continue

                    columns = {column.lower() for column in table_def.get("cols", {})}
                    for field_key in SOURCE_FIELD_KEYS:
                        field_name = source_def.get(field_key)
                        if (
                            isinstance(field_name, str)
                            and field_name.lower() not in columns
                        ):
                            problems.append(
                                f"{filename}:{concept_name}:{database}[{source_index}] "
                                f"{table_name}.{field_name} from {field_key}"
                            )

    assert problems == []


def test_aumc_ventilator_dictionary_items_are_prefiltered() -> None:
    dictionary = _load_json("concept-dict.json")
    prefilter_ids = _aumc_numericitems_prefilter_ids()

    fio2_ids = set(dictionary["fio2"]["sources"]["aumc"][0]["ids"])
    peep_ids = set(dictionary["peep"]["sources"]["aumc"][0]["ids"])

    assert {6699, 12279, 12282} <= fio2_ids
    assert {6694, 8862, 8879, 12284} <= peep_ids
    assert fio2_ids <= prefilter_ids
    assert peep_ids <= prefilter_ids


def test_all_aumc_numericitems_dictionary_sources_are_prefiltered() -> None:
    assert (
        _dictionary_source_ids("aumc", "numericitems", "itemid")
        == _aumc_numericitems_prefilter_ids()
    )


def test_global_itemid_prefilters_follow_dictionary_sources() -> None:
    expected_prefilters = [
        (
            "miiv",
            "chartevents",
            "itemid",
            set(MIIV_CHARTEVENTS_ITEMIDS),
        ),
        (
            "miiv",
            "labevents",
            "itemid",
            set(MIIV_LABEVENTS_ITEMIDS),
        ),
        (
            "mimic_demo",
            "chartevents",
            "itemid",
            set(MIMIC_DEMO_CHARTEVENTS_ITEMIDS),
        ),
        (
            "mimic_demo",
            "labevents",
            "itemid",
            set(MIMIC_DEMO_LABEVENTS_ITEMIDS),
        ),
        (
            "hirid",
            "observations",
            "variableid",
            set(HIRID_OBSERVATIONS_VARIABLEIDS),
        ),
    ]

    for dataset, table, sub_var, prefilter_ids in expected_prefilters:
        assert _dictionary_source_ids(dataset, table, sub_var) == prefilter_ids


def test_miiv_integer_itemids_exist_in_packaged_source_catalog() -> None:
    catalog = _miiv_source_catalog()
    missing: list[tuple[str, str, str, int]] = []

    for filename in ("concept-dict.json", "sofa2-dict.json"):
        dictionary = _load_json(filename)
        for concept_name, concept_def in dictionary.items():
            sources = concept_def.get("sources")
            if not isinstance(sources, dict):
                continue
            for source_def in sources.get("miiv", []):
                table_name = source_def.get("table")
                ids = source_def.get("ids")
                if isinstance(ids, int):
                    source_ids = [ids]
                elif isinstance(ids, list):
                    source_ids = [item for item in ids if isinstance(item, int)]
                else:
                    continue
                for itemid in source_ids:
                    if (table_name, itemid) not in catalog:
                        missing.append((filename, concept_name, str(table_name), itemid))

    assert missing == []


def test_mimic_demo_sources_mirror_mimic_sources() -> None:
    missing: list[tuple[str, str]] = []
    different: list[tuple[str, str]] = []
    allowed_missing = {"samp"}
    allowed_different = {"samp"}

    for filename in ("concept-dict.json", "sofa2-dict.json"):
        dictionary = _load_json(filename)
        for concept_name, concept_def in dictionary.items():
            sources = concept_def.get("sources")
            if not isinstance(sources, dict):
                continue
            if (
                "mimic" in sources
                and "mimic_demo" not in sources
                and concept_name not in allowed_missing
            ):
                missing.append((filename, concept_name))
            elif (
                "mimic" in sources
                and "mimic_demo" in sources
                and sources["mimic"] != sources["mimic_demo"]
                and concept_name not in allowed_different
            ):
                different.append((filename, concept_name))

    assert missing == []
    assert different == []


def test_ventilator_itemids_keep_measured_and_set_semantics_separate() -> None:
    concept = _load_json("concept-dict.json")

    tidal_vol_miiv = set(concept["tidal_vol"]["sources"]["miiv"][0]["ids"])
    tidal_vol_mimic = set(concept["tidal_vol"]["sources"]["mimic"][0]["ids"])
    tidal_vol_set_miiv = set(concept["tidal_vol_set"]["sources"]["miiv"][0]["ids"])
    tidal_vol_set_mimic = set(concept["tidal_vol_set"]["sources"]["mimic"][0]["ids"])
    resp_miiv = set(concept["resp"]["sources"]["miiv"][0]["ids"])
    resp_mimic = set(concept["resp"]["sources"]["mimic"][0]["ids"])
    resp_mimic_demo = set(concept["resp"]["sources"]["mimic_demo"][0]["ids"])
    vent_rate_miiv = set(concept["vent_rate"]["sources"]["miiv"][0]["ids"])
    vent_rate_mimic = set(concept["vent_rate"]["sources"]["mimic"][0]["ids"])
    vent_rate_mimic_demo = set(concept["vent_rate"]["sources"]["mimic_demo"][0]["ids"])
    minute_vol_mimic = set(concept["minute_vol"]["sources"]["mimic"][0]["ids"])

    assert 224684 not in tidal_vol_miiv
    assert 224684 not in tidal_vol_mimic
    assert 224684 in tidal_vol_set_miiv
    assert 224684 in tidal_vol_set_mimic

    assert 224688 not in resp_miiv
    assert 224688 not in resp_mimic
    assert 224688 not in resp_mimic_demo
    assert 619 not in resp_mimic
    assert 619 not in resp_mimic_demo
    assert 224688 in vent_rate_miiv
    assert 224688 in vent_rate_mimic
    assert 224688 in vent_rate_mimic_demo
    assert 619 in vent_rate_mimic
    assert 619 in vent_rate_mimic_demo

    assert {224422, 224689, 224690}.isdisjoint(vent_rate_miiv)
    assert 224422 not in vent_rate_mimic
    assert {224688, 224690}.isdisjoint(minute_vol_mimic)
    assert 224687 in minute_vol_mimic
    assert {614, 615, 653, 1884, 3603, 8113}.issubset(resp_mimic)
    assert {219, 613, 619, 1635}.isdisjoint(resp_mimic)
    assert {614, 615, 653, 1884, 3603, 8113}.issubset(resp_mimic_demo)
    assert {219, 613, 619, 1635}.isdisjoint(resp_mimic_demo)


def test_tidal_volume_semantic_coverage_converts_liter_only_source() -> None:
    concept = _load_json("concept-dict.json")
    measured_mimic_ids = {
        source_id
        for source in concept["tidal_vol"]["sources"]["mimic"]
        if source.get("table") == "chartevents" and "callback" not in source
        for source_id in source["ids"]
    }
    converted_mimic_ids = {
        source_id
        for source in concept["tidal_vol"]["sources"]["mimic"]
        if source.get("table") == "chartevents" and "convert_unit" in source.get("callback", "")
        for source_id in source["ids"]
    }

    assert {65, 654, 684, 2094, 2311, 2400, 2402, 2408, 2420, 2534, 2566}.issubset(
        measured_mimic_ids
    )
    assert {2998, 3003, 3004, 3045, 3050, 3083, 3086, 5593, 6289, 224421}.issubset(
        measured_mimic_ids
    )
    assert 652 in converted_mimic_ids

    excluded = {
        639,  # sigh tidal volume, V-mA unit in local data
        224684,  # set tidal volume
        224743,  # Vd/Vt ratio
        3688,  # valueuom=in in local data
        3689,  # valueuom=in in local data
        6933,  # mixed L/ml-like values without reliable unit metadata
        6935,  # mixed L/ml-like values without reliable unit metadata
    }
    all_mimic_tidal_ids = {
        source_id
        for source in concept["tidal_vol"]["sources"]["mimic"]
        if source.get("table") == "chartevents"
        for source_id in source["ids"]
    }
    assert excluded.isdisjoint(all_mimic_tidal_ids)

    assert concept["tidal_vol"]["sources"]["mimic"] == concept["tidal_vol"]["sources"]["mimic_demo"]

    eicu_ids = set(concept["tidal_vol"]["sources"]["eicu"][0]["ids"])
    assert {"Tidal Volume, Delivered", "Vt Spontaneous (mL)"}.issubset(eicu_ids)
    assert "Adult Con Setting Spont Exp Vt" not in eicu_ids

    miiv_ids = set(concept["tidal_vol"]["sources"]["miiv"][0]["ids"])
    assert 224421 in miiv_ids


def test_aumc_tidal_volume_liter_sources_are_converted_to_ml() -> None:
    concept = _load_json("concept-dict.json")
    sources = concept["tidal_vol"]["sources"]["aumc"]

    mapped_ids = {
        item_id
        for source in sources
        for item_id in _source_ids(source)
    }
    assert 12373 in mapped_ids
    assert {8871, 9669}.issubset(mapped_ids)

    liter_sources = [
        source
        for source in sources
        if {8871, 9669} & _source_ids(source)
    ]
    assert liter_sources
    for source in liter_sources:
        callback = source.get("callback", "")
        assert "convert_unit" in callback
        assert "1000" in callback


def test_minute_volume_semantic_coverage_keeps_observed_values_not_alarms() -> None:
    concept = _load_json("concept-dict.json")

    aumc_ids = set(concept["minute_vol"]["sources"]["aumc"][0]["ids"])
    assert {8870, 8875, 9663, 9668}.issubset(aumc_ids)
    assert 9619 not in aumc_ids  # Mv Lekkage

    eicu_ids = set(concept["minute_vol"]["sources"]["eicu"][0]["ids"])
    assert "Minute Volume, Spontaneous" in eicu_ids
    assert "Minute Ventilation Set(L/min)" not in eicu_ids
    assert "Minute Volume Leak" not in eicu_ids

    measured_mimic_ids = {448, 450, 650, 1883, 3049, 3259, 6078, 6932, 6934}
    excluded_mimic_ids = {449, 2012}
    for dataset in ("mimic", "mimic_demo"):
        mimic_ids = set(concept["minute_vol"]["sources"][dataset][0]["ids"])
        assert measured_mimic_ids.issubset(mimic_ids)
        assert excluded_mimic_ids.isdisjoint(mimic_ids)


def test_sic_lab_fio2_is_mapped_as_percent_not_ventilator_setting() -> None:
    concept = _load_json("concept-dict.json")
    sic_sources = concept["fio2"]["sources"]["sic"]

    lab_source = next(
        source
        for source in sic_sources
        if source.get("table") == "laboratory" and source.get("ids") == 684
    )
    assert lab_source["sub_var"] == "LaboratoryID"
    assert "percent_as_numeric" in lab_source.get("callback", "")


def test_miiv_bilirubin_keeps_blood_total_not_body_fluid_variants() -> None:
    concept = _load_json("concept-dict.json")
    labevent_ids = {
        source_id
        for source in concept["bili"]["sources"]["miiv"]
        if source.get("table") == "labevents"
        for source_id in _source_ids(source)
    }
    chartevent_ids = {
        source_id
        for source in concept["bili"]["sources"]["miiv"]
        if source.get("table") == "chartevents"
        for source_id in _source_ids(source)
    }

    assert {50885, 53089}.issubset(labevent_ids)
    assert 225690 in chartevent_ids
    assert {50838, 51028, 51049, 51568, 51783, 51812, 51932}.isdisjoint(labevent_ids)


def test_module_coverage_lab_extensions_are_semantically_narrow() -> None:
    assert _concept_source_ids("hbco", "miiv", "labevents", "itemid") == {50805}
    assert _concept_source_ids("hbco", "mimic", "labevents", "itemid") == {50805}
    assert _concept_source_ids("hbco", "mimic_demo", "labevents", "itemid") == {50805}

    assert _concept_source_ids("tco2", "sic", "laboratory", "LaboratoryID") == {670}
    assert 716 not in _concept_source_ids("tco2", "sic", "data_float_h", "DataID")

    assert _concept_source_ids("hba1c", "sic", "laboratory", "LaboratoryID") == {214}
    assert 474 not in _concept_source_ids("hba1c", "sic", "laboratory", "LaboratoryID")

    assert _concept_source_ids("ammonia", "hirid", "observations", "variableid") == {24000568}
    assert _concept_source_ids("amylase", "hirid", "observations", "variableid") == {24000427}
    assert 24000587 not in _concept_source_ids("amylase", "hirid", "observations", "variableid")
    assert _concept_source_ids("ferritin", "hirid", "observations", "variableid") == {24000678}
    assert _concept_source_ids("lipase", "hirid", "observations", "variableid") == {24000555}


def test_vent_end_excludes_o2_delivery_device_itemid() -> None:
    concept = _load_json("concept-dict.json")

    for dataset in ("miiv", "mimic", "mimic_demo"):
        chartevent_ids = set(
            source_id
            for source in concept["vent_end"]["sources"][dataset]
            if source.get("table") == "chartevents"
            for source_id in source["ids"]
        )
        assert 226732 not in chartevent_ids


def test_urine_output_excludes_enteral_residuals_and_keeps_perioperative_urine() -> None:
    concept = _load_json("concept-dict.json")
    eicu_regexes = [
        source.get("regex", "")
        for source in concept["urine"]["sources"]["eicu"]
        if source.get("table") == "intakeoutput"
        and source.get("regex")
    ]
    assert any(re.search(pattern, "Urine Output-Foley", re.I) for pattern in eicu_regexes)
    assert any(re.search(pattern, "Suprapubic Urine Output", re.I) for pattern in eicu_regexes)
    assert any(re.search(pattern, "Urine, void:", re.I) for pattern in eicu_regexes)
    assert not any(re.search(pattern, "Mixed Urine/Stool Volume", re.I) for pattern in eicu_regexes)

    for dataset in ("miiv", "mimic", "mimic_demo"):
        output_sources = [
            source
            for source in concept["urine"]["sources"][dataset]
            if source.get("table") == "outputevents"
            and source.get("sub_var") == "itemid"
        ]
        urine_ids = set().union(*(set(source["ids"]) for source in output_sources))

        assert 227510 not in urine_ids
        assert 227511 not in urine_ids
        # 2026-07-17 REVERSED. This previously asserted that OR/PACU urine
        # (226627/226631) must be KEPT -- a deliberate call, made to stop the
        # perioperative window reading as anuric. Three lines of evidence overturned it:
        #   1. Official mimic-code measurement/urine_output.sql uses
        #      226559/226560/226561/226584/226563/226564/226565/226567/226557/226558
        #      + 227488/227489 -- and NOT 226627/226631.
        #   2. The raw data shows why: OR Urine is 1.16 rows/stay (median 400 mL,
        #      p95 2,250) and PACU Urine 1.05 rows/stay (median 625 mL, p95 3,980),
        #      versus Foley's 54.33 rows/stay at a median of 80 mL. They are single
        #      BULK totals for an entire perioperative period, not hourly measurements.
        #   3. `urine` is an hourly concept feeding kdigo_uo's mL/kg/h rate. Dropping a
        #      3,980 mL bulk onto one timestamp invents a massive one-hour diuresis and
        #      still leaves the surrounding OR hours empty -- it does not fix the false
        #      anuria it was added for, it adds a second artefact on top of it.
        assert {226627, 226631}.isdisjoint(urine_ids)
        assert {43348, 43365, 43372, 43638, 227489}.isdisjoint(urine_ids)

    for dataset in ("mimic", "mimic_demo"):
        output_sources = [
            source
            for source in concept["urine"]["sources"][dataset]
            if source.get("table") == "outputevents"
            and source.get("sub_var") == "itemid"
        ]
        urine_ids = set().union(*(set(source["ids"]) for source in output_sources))
        assert {
            42068,
            42111,
            42119,
            42366,
            42676,
            43966,
            44325,
            44506,
            44706,
            44824,
            44911,
            45804,
            45991,
            46578,
            46658,
        }.issubset(urine_ids)


def test_mimic_legacy_vital_channels_keep_semantic_boundaries() -> None:
    concept = _load_json("concept-dict.json")

    for dataset in ("mimic", "mimic_demo"):
        map_ids = set(concept["map"]["sources"][dataset][0]["ids"])
        assert {
            438,
            1321,
            2309,
            2353,
            2369,
            2544,
            2770,
            2974,
            3067,
            5680,
            5804,
            6399,
            6579,
            6605,
        }.issubset(map_ids)
        assert {672, 1199, 2522}.isdisjoint(map_ids)

        spo2_ids = set(concept["spo2"]["sources"][dataset][0]["ids"])
        o2sat_ids = set(concept["o2sat"]["sources"][dataset][0]["ids"])
        assert 6719 in spo2_ids
        assert 6719 in o2sat_ids
        assert 220227 not in spo2_ids
        assert 220227 in o2sat_ids
        assert {226860, 226861, 226862, 226863, 226865}.isdisjoint(spo2_ids)
        assert {226860, 226861, 226862, 226863, 226865}.isdisjoint(o2sat_ids)


def test_rrt_uses_active_treatment_evidence_not_access_placement() -> None:
    concept = _load_json("concept-dict.json")
    sofa2 = _load_json("sofa2-dict.json")

    for dictionary in (concept, sofa2):
        miiv_sources = dictionary["rrt"]["sources"]["miiv"]
        procedure_ids = set(
            source_id
            for source in miiv_sources
            if source.get("table") == "procedureevents"
            for source_id in source["ids"]
        )
        chartevent_ids = set(
            source_id
            for source in miiv_sources
            if source.get("table") == "chartevents"
            for source_id in source["ids"]
        )
        active_monitoring_ids = {
            224144,
            224149,
            224150,
            224151,
            224152,
            224153,
            224154,
            224191,
            225806,
            225807,
            225810,
            225976,
            225977,
            226457,
            226499,
            228005,
            228006,
        }
        miiv_only_active_setting_ids = {229247, 229248, 230083, 230084, 230085, 230177}

        assert 224270 not in procedure_ids
        # 2026-07-17 REVERSED for 225436 (CRRT Filter Change). Official mimic-code
        # treatment/rrt.sql does list 225436, but its CASE marks it `dialysis_active = 0`
        # -- the same treatment it gives 224270 (Dialysis Catheter), which this test
        # already excludes on the line above. A filter change is a procedural event, not
        # an active treatment session. `rrt` is a boolean with no active/inactive axis,
        # so dropping the id is the faithful mapping of `dialysis_active = 0`.
        # This test's own AUMC half already applies exactly this rule -- it asserts
        # "Filter CVVH wisselen" (Dutch for 'change CVVH filter') must NOT match. So the
        # old MIIV assertion contradicted both the official code and this test's own
        # stated principle ("active treatment evidence, not access placement").
        # Cost of the change, measured: of the 240 MIMIC-IV stays with a filter change,
        # 239 already carry a core dialysis id -- exactly 1 stay is affected.
        assert 225436 not in procedure_ids
        assert {225441, 225802, 225803, 225805, 225809, 225955}.issubset(
            procedure_ids
        )
        assert 227290 in chartevent_ids
        assert active_monitoring_ids.issubset(chartevent_ids)
        assert miiv_only_active_setting_ids.issubset(chartevent_ids)
        assert {224135, 225126, 225128, 225954}.isdisjoint(chartevent_ids)

        aumc_numeric_ids = set(
            source_id
            for source in dictionary["rrt"]["sources"]["aumc"]
            if source.get("table") == "numericitems"
            for source_id in source["ids"]
        )
        assert {8805, 7666, 7667, 7668, 10736, 12444, 6684, 8806, 8808, 12091}.issubset(
            aumc_numeric_ids
        )
        aumc_procedure_regex = next(
            source["regex"]
            for source in dictionary["rrt"]["sources"]["aumc"]
            if source.get("table") == "procedureorderitems"
        )
        assert re.search(aumc_procedure_regex, "CVVH starten", re.I)
        assert re.search(aumc_procedure_regex, "CVVH stoppen", re.I)
        assert not re.search(aumc_procedure_regex, "CVVH-lab. afnemen", re.I)
        assert not re.search(aumc_procedure_regex, "Filter CVVH wisselen", re.I)
        assert not re.search(aumc_procedure_regex, "Resetten CVVH", re.I)
        assert not re.search(aumc_procedure_regex, "Citraat-CVVH urine 24 uur", re.I)

        mimic_procedure_ids = set(
            source_id
            for source in dictionary["rrt"]["sources"]["mimic"]
            if source.get("table") == "procedureevents_mv"
            for source_id in source["ids"]
        )
        # 225436 excluded here for the same reason as the miiv block above
        # (official rrt.sql marks CRRT Filter Change dialysis_active = 0).
        assert 225436 not in mimic_procedure_ids
        assert {225441, 225802, 225803, 225805, 225809, 225955}.issubset(
            mimic_procedure_ids
        )
        mimic_chartevent_ids = set(
            source_id
            for source in dictionary["rrt"]["sources"]["mimic"]
            if source.get("table") == "chartevents"
            for source_id in source["ids"]
        )
        assert active_monitoring_ids.issubset(mimic_chartevent_ids)
        assert {224135, 225126, 225128, 225954}.isdisjoint(mimic_chartevent_ids)
        mimic_output_ids = set(
            source_id
            for source in dictionary["rrt"]["sources"]["mimic"]
            if source.get("table") == "outputevents"
            for source_id in source["ids"]
        )
        assert {40613, 42388, 43052, 41527, 40910, 43115}.issubset(mimic_output_ids)
        assert {44085, 42972, 41034, 46232, 46713}.isdisjoint(mimic_output_ids)

        eicu_intakeoutput_sources = [
            source
            for source in dictionary["rrt"]["sources"].get("eicu", [])
            if source.get("table") == "intakeoutput"
        ]
        cellpath_regex = next(
            source["regex"]
            for source in eicu_intakeoutput_sources
            if source.get("sub_var") == "cellpath"
        )
        celllabel_regex = next(
            source["regex"]
            for source in eicu_intakeoutput_sources
            if source.get("sub_var") == "celllabel"
        )
        assert re.search(cellpath_regex, "flowsheet|I&O|Dialysis (ml)|Out", re.I)
        assert re.search(cellpath_regex, "flowsheet|I&O|Output (ml)|HemodialysisOut", re.I)
        assert not re.search(cellpath_regex, "CL Flush: Non-Tunneled dialysis IJ L", re.I)
        assert re.search(celllabel_regex, "CRRT - UF removed", re.I)
        assert re.search(celllabel_regex, "CRRT Actual Pt Fluid Removed", re.I)
        assert not re.search(celllabel_regex, "CRRT Flush", re.I)
        assert not re.search(celllabel_regex, "Volume calcium chloride infusion (CRRT with citrate anticoag)", re.I)


def test_miiv_top_level_mechanism_concepts_stay_aligned_with_sofa2() -> None:
    concept = _load_json("concept-dict.json")
    sofa2 = _load_json("sofa2-dict.json")

    concept_ecmo_ids = set(concept["ecmo"]["sources"]["miiv"][0]["ids"])
    sofa2_ecmo_ids = set(sofa2["ecmo"]["sources"]["miiv"][0]["ids"])
    assert sofa2_ecmo_ids.issubset(concept_ecmo_ids)

    concept_ecmo_indication_ids = set(
        concept["ecmo_indication"]["sources"]["miiv"][0]["ids"]
    )
    assert concept_ecmo_indication_ids == {229268}
    assert concept["ecmo_indication"]["sources"]["miiv"][0].get("val_var") == "value"

    concept_mcs_chartevent_ids = set(
        source_id
        for source in concept["mech_circ_support"]["sources"]["miiv"]
        if source.get("table") == "chartevents"
        for source_id in source["ids"]
    )
    sofa2_mcs_chartevent_ids = set(sofa2["mech_circ_support"]["sources"]["miiv"][0]["ids"])
    assert sofa2_mcs_chartevent_ids.issubset(concept_mcs_chartevent_ids)


def test_mimiciii_mechanical_support_excludes_mimiciv_only_items() -> None:
    """These two labels exist in MIMIC-IV but not MIMIC-III d_items."""

    mimiciv_only = {228866, 229254}
    for filename in ("concept-dict.json", "sofa2-dict.json"):
        payload = _load_json(filename)
        sources = payload["mech_circ_support"]["sources"]
        miiv_ids = set(sources["miiv"][0]["ids"])
        mimic_ids = set(sources["mimic"][0]["ids"])
        mimic_demo_ids = set(sources["mimic_demo"][0]["ids"])

        assert mimiciv_only.issubset(miiv_ids)
        assert mimiciv_only.isdisjoint(mimic_ids)
        assert mimic_demo_ids == mimic_ids


def test_cross_source_mechanism_concepts_do_not_use_ambiguous_indication_sources() -> None:
    concept = _load_json("concept-dict.json")

    for dataset in ("eicu", "eicu_demo"):
        ecmo_nurse_sources = [
            source
            for source in concept["ecmo"]["sources"][dataset]
            if source.get("table") == "nursecharting"
        ]
        assert ecmo_nurse_sources
        assert all(
            source.get("sub_var") == "nursingchartvalue"
            for source in ecmo_nurse_sources
        )

        ecmo_indication_sources = concept["ecmo_indication"]["sources"][dataset]
        assert all(
            source.get("table") != "nursecharting"
            for source in ecmo_indication_sources
        )

    eicu_mcs_regex = concept["mech_circ_support"]["sources"]["eicu"][0]["regex"]
    assert "Tandem" in eicu_mcs_regex
    assert "ventricular assist" in eicu_mcs_regex

    aumc_mcs_regex = concept["mech_circ_support"]["sources"]["aumc"][0]["regex"]
    assert "VAD" in aumc_mcs_regex
    assert "ventricular assist" in aumc_mcs_regex

    eicu_rrt_sources = concept["rrt"]["sources"]["eicu"]
    eicu_rrt_treatment_regex = eicu_rrt_sources[0]["regex"]
    assert re.search(eicu_rrt_treatment_regex, "renal|dialysis|hemodialysis", re.I)
    assert not re.search(
        eicu_rrt_treatment_regex,
        "renal|dialysis|insertion of venous catheter for hemodialysis",
        re.I,
    )
    assert any(
        source.get("table") == "intakeoutput"
        and source.get("sub_var") == "cellpath"
        and source.get("regex") == "(Dialysis \\(ml\\)|Hemodialysis)"
        for source in eicu_rrt_sources
    )
    assert "ultrafiltration" in eicu_rrt_treatment_regex
    assert "renal replacement" in eicu_rrt_treatment_regex
