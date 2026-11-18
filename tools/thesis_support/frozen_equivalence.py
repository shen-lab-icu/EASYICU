"""Compare a thesis projection against the frozen L-COMMON reducer.

Run from the checkout with PYTHONPATH=src. Example command forms and the
staged authority table names are documented in this directory's README.
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

_COHORT_SHA256 = "d8e529cdc8034c990c2ee97e3f0f9b2cf8866087b1c975f295a7d15ae628f9a1"


def _quote(path):
    return "'" + str(path).replace("'", "''") + "'"


class _ReadOnlyFrozenLoader(SourceFileLoader):
    def get_code(self, fullname):
        path = self.get_filename(fullname)
        source = self.get_data(path)
        if fullname.endswith(".cohort") and hashlib.sha256(source).hexdigest() != _COHORT_SHA256:
            raise ValueError("unexpected frozen cohort digest")
        return self.source_to_code(source, path)


@contextmanager
def _frozen_modules(source_dir, names):
    """Execute only the named frozen sources, with inert import dependencies."""
    source_dir = Path(source_dir)
    paths = {name: source_dir / f"{name}.py" for name in names}
    package = ModuleType(f"_thesis_frozen_{uuid4().hex}")
    package.__path__ = []  # Do not traverse the frozen checkout on imports.
    duck = ModuleType(f"{package.__name__}.duck")
    duck.q = _quote
    release = ModuleType(f"{package.__name__}.release")
    release.Release = object
    installed = {module.__name__: module for module in (package, duck, release)}
    modules = {}
    try:
        sys.modules.update(installed)
        for name, path in paths.items():
            qualified_name = f"{package.__name__}.{name}"
            loader = _ReadOnlyFrozenLoader(qualified_name, str(path))
            spec = spec_from_file_location(qualified_name, path, loader=loader)
            module = module_from_spec(spec)
            installed[module.__name__] = module
            sys.modules[module.__name__] = module
            loader.exec_module(module)
            modules[name] = module
        yield modules
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
    }
    for path, relation in zip(inputs[database], relations, strict=True):
        sql.bind(rel.raw(database) / path, relation)
    sql.bind(rel.raw(database) / "labevents/*.parquet", """(
        SELECT NULL::BIGINT AS itemid, NULL::VARCHAR AS value, NULL::BIGINT AS specimen_id,
               NULL::TIMESTAMP AS charttime, NULL::BIGINT AS hadm_id, NULL::DOUBLE AS valuenum
        WHERE FALSE
    )""")
    sql.bind(rel.corrected["aumc_pao2_view"], """(
        SELECT NULL::BIGINT AS stay_id, NULL::INTEGER AS charttime, NULL::DOUBLE AS po2 WHERE FALSE
    )""")
    cohort = sql.bind(rel.common_dir / "selected", "(SELECT stay_id, TRUE AS in_base, database FROM selected)")
    query = source.BUILDERS[database](rel, cohort)
    return sql.execute(f"""
        SELECT stay_id, icu_hour AS charttime, peep_set AS observed_peep_set
        FROM ({query}) WHERE peep_set IS NOT NULL AND icu_hour IS NOT NULL
    """).df()


def _cohort_reference(sql, rel, cohort, database, module, relation):
    if module == "rrt_time":
        if database == "miiv":
            sql.bind(rel.corrected["miiv_rrt_time"], relation)
        elif database == "mimic3cv":
            sql.bind(rel.common_dir / "_inputs/mimic3cv_rrt_hours.parquet", relation)
        else:
            sql.bind(rel.analysis_hourly(database), relation)
            cohort.stage_hourly_view(sql, rel, database, ["rrt"], "hc")
        query = cohort.rrt_hours_sql(rel, database)
        return sql.execute(f"""
            SELECT q.stay_id, q.icu_hour AS charttime, TRUE AS rrt_active
            FROM ({query}) q SEMI JOIN selected USING (stay_id)
        """).df()

    # build_cohort_sql constructs the original CTEs. Run just the requested
    # input projection, without its unrelated cohort, RRT or outcome outputs.
    if database == "hirid":
        sql.bind(rel.derived_shared / "flat/hirid_vent_anchor.parquet", relation)
        hours = "SELECT stay_id, NULL::INTEGER AS icu_hour, NULL::BOOLEAN AS imv FROM selected WHERE FALSE"
    else:
        hours = f"SELECT stay_id, icu_hour, imv FROM {relation} SEMI JOIN selected USING (stay_id)"
    sql.bind(rel.analysis_hourly(database), f"""(
        SELECT *, NULL AS patient_id, NULL AS age, NULL AS sex, NULL AS weight,
               NULL AS icu_los_h, NULL AS rrt FROM ({hours})
    )""")
    query = cohort.build_cohort_sql(sql, rel, database)
    if database == "hirid":
        anchor_and_span = query.split("    anchor AS (", 1)[1].split("    rrt AS (", 1)[0].rstrip().removesuffix(",")
        return sql.execute(f"""
            WITH anchor AS ({anchor_and_span}
            SELECT ep.stay_id, t0_hour AS charttime, ep_end_hour AS imv_recorded_end_hour
            FROM ep SEMI JOIN selected USING (stay_id)
        """).df()
    hourly = query.split("        ih AS (", 1)[1].split("        g AS (", 1)[0].rstrip().removesuffix(",")
    return sql.execute(f"WITH ih AS ({hourly} SELECT stay_id, h AS charttime, TRUE AS imv FROM ih").df()


def _reference(source, family, stay, ids, module, frozen_dir, receipt):
    database = "mimic3cv" if family == "mimic" else family
    virtual = Path("/__thesis_frozen__")
    rel = SimpleNamespace(
        raw=lambda db: virtual / "raw" / db,
        common_dir=virtual,
        derived_shared=virtual / "derived",
        analysis_hourly=lambda db: virtual / "hourly" / db,
        corrected={
            "aumc_pao2_view": virtual / "unrelated_pao2",
            "miiv_rrt_time": virtual / "rrt",
        },
        protected_test={"policy": "exclude", "miiv_allowed_keys": virtual / "allowed"},
        params={"cohort": {"episode_gap_tolerance_h": 0, "min_age": 0, "primary_min_imv_h": 0}},
    )
    input_key = stay if module == "observed_peep_set" else "stay_id"
    files = [support_input_files(source, table, input_key, ids) for table in receipt["input_tables"]]
    if any(not paths for paths in files):
        raise ValueError("no reference input files for these stays")
    relations = [parquet_relation(paths, ordered=False) for paths in files]
    with duckdb_spill.connection() as con:
        con.register("selected", pd.DataFrame({
            "stay_id": pd.Series(ids, dtype="int64"), "database": database,
        }))
        sql = _FrozenSQL(con)
        sql.bind(virtual / "allowed", "(SELECT stay_id FROM selected)")
        if module == "observed_peep_set":
            with _frozen_modules(frozen_dir, ("sources",)) as modules:
                return _peep_reference(sql, rel, modules["sources"], database, relations)
        with _frozen_modules(frozen_dir, ("readers", "cohort")) as modules:
            return _cohort_reference(sql, rel, modules["cohort"], database, module, relations[0])


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


def _summary(columns, rows):
    digest = hashlib.sha256()
    digest.update((json.dumps(columns) + "\n").encode())
    for row in sorted(rows):
        digest.update(("[" + ",".join(row) + "]\n").encode())
    return {
        "rows": _rounded(len(rows)),
        "stays": _rounded(len({row[0] for row in rows})),
        "sha256": digest.hexdigest(),
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
        "module": _summary(columns, left),
        "frozen": _summary(columns, right),
        "mismatching_rows": _rounded(different_rows),
        "mismatching_cells": {column: _rounded(cells[column]) for column in columns},
    }


def run_equivalence(*, database, db_root, stay_ids, module, frozen_src_dir):
    """Run both implementations against the same tables and selected stay IDs."""
    stay = DATABASE_ID_CONFIG[database]["id_col"]
    family = database.removesuffix("_demo")
    with binding_unit():
        actual = load_thesis_support(module=module, database=database, data_path=db_root, patient_ids=stay_ids)
        receipt = thesis_support_manifest(module, database, actual.attrs["available_fields"])
        if receipt["status"] != "projected":
            return {"status": receipt["status"]}
        ids = select_support_cohort(database, db_root, stay_ids)[stay]
        expected = _reference(storage_source(database, db_root), family, stay, ids, module, frozen_src_dir, receipt)
    actual = actual.rename(columns={stay: "stay_id"})
    # The frozen SQL projects only the database's fields. Include the same
    # typed-null columns as the native module before computing either digest.
    expected = expected.reindex(columns=actual.columns).astype(actual.dtypes)
    keys = ["stay_id"] if receipt["charttime_role"] == "first_recorded_hour" else ["stay_id", "charttime"]
    return compare_rows(actual, expected, key_columns=keys)


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
    parser.add_argument("--frozen-src-dir", type=Path, required=True, help="directory containing frozen cohort.py")
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
