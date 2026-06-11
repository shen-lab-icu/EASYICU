from __future__ import annotations

import json
from pathlib import Path

from easyicu.resources import load_dictionary
from easyicu.webapp.concept_catalog import (
    COMPOSITE_CONCEPT_OUTPUT_SOURCES,
    CONCEPT_DICTIONARY,
    CONCEPT_GROUPS_INTERNAL,
    HIDDEN_DICTIONARY_CONCEPTS,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "src" / "easyicu" / "data"
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


def test_web_catalog_groups_are_unique_and_complete() -> None:
    grouped = [concept for concepts in CONCEPT_GROUPS_INTERNAL.values() for concept in concepts]

    assert len(grouped) == len(set(grouped))
    assert set(grouped) == set(CONCEPT_DICTIONARY)


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
    special_sources = {"circ_failure_loader"}

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
