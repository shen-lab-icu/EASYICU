"""Compare a thesis projection against the frozen L-COMMON reducer.

Run from the checkout with PYTHONPATH=src. Both sides use the same raw input
sample. Example command forms are documented in this directory's README.
The program writes one JSON object of aggregates to stdout. Exit status is
0 for agreement, 1 for a mismatch, and 2 if the comparison could not run.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager, redirect_stderr, redirect_stdout
import hashlib
from importlib.machinery import SourceFileLoader
from importlib.util import module_from_spec, spec_from_file_location
from io import TextIOBase
import json
import logging
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from uuid import uuid4

import numpy as np
import pandas as pd

from easyicu.concept.catalog import THESIS_SUPPORT_MODULES
from easyicu.concept.support_exports import select_support_cohort
from easyicu.concept.thesis_support import load_thesis_support, support_input_files, thesis_support_manifest
from easyicu.config import DATABASE_ID_CONFIG
from easyicu.datasource import storage_source
from easyicu.runtime import duckdb_spill
from easyicu.storage_binding import binding_unit
from easyicu.storage_layout import parquet_relation

_FROZEN_SOURCE_DIR = Path(
    "/home/zhuhb/workspace/claude-ws/_wt/ys351lcommon/frozen_copies/L-COMMON/src/phd_common"
)
_FROZEN_DIGESTS = {
    "sources": "35f160b2d8105af7ed76a887ceb247117676b255b64a9f7a7537fc18f4a1bdc6",
    "new_sources": "d189e0ae1deefb19547332f269e72330a1eda54c95f23f55a69c0d119443258f",
}


def _quote(path):
    return "'" + str(path).replace("'", "''") + "'"


class _ReadOnlyFrozenLoader(SourceFileLoader):
    def get_code(self, fullname):
        path = self.get_filename(fullname)
        source = self.get_data(path)
        name = fullname.rsplit(".", 1)[1]
        if hashlib.sha256(source).hexdigest() != _FROZEN_DIGESTS[name]:
            raise ValueError("unexpected frozen source digest")
        return self.source_to_code(source, path)


@contextmanager
def _frozen_module(source_dir, name):
    """Execute a named frozen file without importing the rest of its checkout."""
    if Path(source_dir) != _FROZEN_SOURCE_DIR or name not in _FROZEN_DIGESTS:
        raise ValueError("unexpected frozen source path")
    path = _FROZEN_SOURCE_DIR / f"{name}.py"
    if path.resolve() != path:
        raise ValueError("frozen source path redirects")
    package = ModuleType(f"_thesis_frozen_{uuid4().hex}")
    package.__path__ = []  # Do not traverse the frozen checkout on imports.
    duck = ModuleType(f"{package.__name__}.duck")
    duck.q = _quote
    duck.copy_to_parquet = None  # The PEEP SQL builder only uses q.
    release = ModuleType(f"{package.__name__}.release")
    release.Release = object
    panel = ModuleType(f"{package.__name__}.panel")
    panel.HOURLY_COLS = panel.SOFA_COLS = ()  # Unused by sic_sources_sql.
    qualified_name = f"{package.__name__}.{name}"
    loader = _ReadOnlyFrozenLoader(qualified_name, str(path))
    spec = spec_from_file_location(qualified_name, path, loader=loader)
    module = module_from_spec(spec)
    installed = {item.__name__: item for item in (package, duck, release, panel, module)}
    try:
        sys.modules.update(installed)
        loader.exec_module(module)
        yield module
    finally:
        for name in installed:
            sys.modules.pop(name, None)


class _FrozenSQL:
    """Bind frozen path literals to the corresponding native input relations."""

    def __init__(self, con):
        self.con = con
        self.paths = {}

    def bind(self, path, relation):
        self.paths[f"read_parquet({_quote(path)})"] = relation
        return path

    def execute(self, sql):
        for call, relation in self.paths.items():
            sql = sql.replace(call, relation)
        if "read_parquet('/__thesis_frozen__/" in sql:
            raise ValueError("unbound frozen SQL input")
        return self.con.execute(sql)


def _peep_reference(sql, rel, source, database, relations):
    # Unrelated gas channels are bypassed with typed empty relations. Keep
    # the frozen setting SQL, cohort join, item filters and median intact.
    inputs = {
        "aumc": ("admissions.parquet", "numericitems/*.parquet"),
        "hirid": ("general.parquet", "observations/*.parquet"),
        "miiv": ("icustays.parquet", "chartevents/*.parquet"),
        "sic": ("cases.parquet", "data_float_h/*.parquet"),
    }
    for path, relation in zip(inputs[database], relations, strict=True):
        sql.bind(rel.raw(database) / path, relation)
    if database == "sic":
        sql.bind(rel.raw(database) / "laboratory/*.parquet", """(
            SELECT NULL::BIGINT AS CaseID, NULL::BIGINT AS "Offset", NULL::BIGINT AS LaboratoryID,
                   NULL::BIGINT AS LaboratoryType, NULL::DOUBLE AS LaboratoryValue
            WHERE FALSE
        )""")
        build = source.sic_sources_sql
    else:
        sql.bind(rel.raw(database) / "labevents/*.parquet", """(
            SELECT NULL::BIGINT AS itemid, NULL::VARCHAR AS value, NULL::BIGINT AS specimen_id,
                   NULL::TIMESTAMP AS charttime, NULL::BIGINT AS hadm_id, NULL::DOUBLE AS valuenum
            WHERE FALSE
        )""")
        sql.bind(rel.corrected["aumc_pao2_view"], """(
            SELECT NULL::BIGINT AS stay_id, NULL::INTEGER AS charttime, NULL::DOUBLE AS po2 WHERE FALSE
        )""")
        build = source.BUILDERS[database]
    cohort = sql.bind(rel.common_dir / "selected", "(SELECT stay_id, TRUE AS in_base, database FROM selected)")
    query = build(rel, cohort)
    return sql.execute(f"""
        SELECT stay_id, icu_hour AS charttime, peep_set AS observed_peep_set
        FROM ({query}) WHERE peep_set IS NOT NULL AND icu_hour IS NOT NULL
    """).df()


def _reference(source, family, stay, ids, frozen_dir, tables):
    virtual = Path("/__thesis_frozen__")
    rel = SimpleNamespace(
        raw=lambda db: virtual / "raw" / db,
        common_dir=virtual,
        corrected={"aumc_pao2_view": virtual / "unrelated_pao2"},
    )
    files = [support_input_files(source, table, stay, ids) for table in tables]
    if any(not paths for paths in files):
        raise ValueError("no reference input files for these stays")
    relations = [parquet_relation(paths, ordered=False) for paths in files]
    with duckdb_spill.connection() as con:
        con.register("selected", pd.DataFrame({
            "stay_id": pd.Series(ids, dtype="int64"), "database": family,
        }))
        name = "new_sources" if family == "sic" else "sources"
        with _frozen_module(frozen_dir, name) as frozen:
            return _peep_reference(_FrozenSQL(con), rel, frozen, family, relations)


def _rounded(count):
    return None if count < 10 else 10 * ((count + 5) // 10)


def _encoded(frame):
    """Canonical values with column names, nulls, and duplicate row multiplicity."""
    rows = []
    for values in frame.itertuples(index=False, name=None):
        row = tuple(
            "null" if pd.isna(value) else json.dumps(
                value.item() if isinstance(value, np.generic) else value, allow_nan=False,
            )
            for value in values
        )
        rows.append(row)
    return rows


def _summary(rows):
    return {
        "rows": _rounded(len(rows)),
        "stays": _rounded(len({row[0] for row in rows})),
    }


def compare_rows(actual, expected, *, key_columns):
    """Compare all columns, matching rows by key and counting duplicate copies."""
    columns = actual.columns.tolist()
    if columns != expected.columns.tolist() or columns[0] != "stay_id":
        raise ValueError("incompatible projection columns")
    keys = [columns.index(column) for column in key_columns]
    left, right = _encoded(actual), _encoded(expected)
    left_only, right_only = Counter(left) - Counter(right), Counter(right) - Counter(left)
    unmatched = defaultdict(lambda: ([], []))
    for side, rows in enumerate((left_only, right_only)):
        for row in rows.elements():
            unmatched[tuple(row[key] for key in keys)][side].append(row)

    cells = Counter()
    different_rows = 0
    for left_group, right_group in unmatched.values():
        different_rows += max(len(left_group), len(right_group))
        for a, b in zip(sorted(left_group), sorted(right_group)):
            cells.update(column for column, x, y in zip(columns, a, b, strict=True) if x != y)
        cells.update({column: abs(len(left_group) - len(right_group)) for column in columns})
    return {
        "status": "agree" if not different_rows else "different",
        "module": _summary(left),
        "frozen": _summary(right),
        "mismatching_rows": _rounded(different_rows),
        "mismatching_cells": {column: _rounded(cells[column]) for column in columns},
    }


def run_equivalence(*, database, db_root, stay_ids, module, frozen_src_dir=_FROZEN_SOURCE_DIR):
    """Run both implementations against the same tables and selected stay IDs."""
    stay = DATABASE_ID_CONFIG[database]["id_col"]
    family = database.removesuffix("_demo")
    with binding_unit():
        actual = load_thesis_support(module=module, database=database, data_path=db_root, patient_ids=stay_ids)
        receipt = thesis_support_manifest(module, database, actual.attrs["available_fields"])
        if receipt["status"] != "projected":
            return {"status": receipt["status"]}
        ids = select_support_cohort(database, db_root, stay_ids)[stay]
        expected = _reference(
            storage_source(database, db_root), family, stay, ids, frozen_src_dir, receipt["input_tables"],
        )
    actual = actual.rename(columns={stay: "stay_id"})
    expected = expected.astype(actual.dtypes)
    return compare_rows(actual, expected, key_columns=["stay_id", "charttime"])


class _DiscardOutput(TextIOBase):
    def write(self, message):
        return len(message)


class _PrivateArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        print('{"status": "invalid_arguments"}')
        raise SystemExit(2)


def main(argv=None):
    parser = _PrivateArgumentParser(description=__doc__)
    parser.add_argument("--database", choices=("aumc", "eicu", "hirid", "miiv", "mimic", "sic"), required=True)
    parser.add_argument("--db-root", type=Path, required=True)
    parser.add_argument("--stay-ids", type=int, nargs="+", required=True)
    parser.add_argument("--module", choices=THESIS_SUPPORT_MODULES, required=True)
    parser.add_argument("--frozen-src-dir", type=Path, default=_FROZEN_SOURCE_DIR,
                        help="approved directory containing frozen sources.py and new_sources.py")
    args = parser.parse_args(argv)
    # A backend exception can contain a raw SQL value or path. Keep its
    # diagnostic private and emit a fixed status code at this boundary.
    with _DiscardOutput() as sink, redirect_stdout(sink), redirect_stderr(sink):
        prior_level = logging.root.manager.disable
        logging.disable(logging.CRITICAL)
        try:
            result = run_equivalence(**vars(args))
        except Exception as error:
            result = {"status": "comparison_failed", "error_type": type(error).__name__}
        finally:
            logging.disable(prior_level)
    print(json.dumps(result, sort_keys=True))
    return {"agree": 0, "different": 1}.get(result["status"], 2)


if __name__ == "__main__":
    sys.exit(main())
