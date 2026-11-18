"""Compare a thesis projection against the frozen L-COMMON reducer.

Run from the checkout with PYTHONPATH=src. Both sides use the same raw input
sample or the same explicitly pinned D0 module files. Example command
forms are documented in this directory's README.
The program writes one JSON object of aggregates to stdout. Exit status is
0 for agreement, 1 for a mismatch, and 2 if the comparison could not run.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
from contextlib import contextmanager, redirect_stderr, redirect_stdout
import hashlib
from importlib.machinery import SourceFileLoader
from importlib.util import module_from_spec, spec_from_file_location
from io import TextIOBase
import json
import logging
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from uuid import uuid4

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from easyicu.concept.catalog import FIRST_IN_HOSP_FROZEN, THESIS_SUPPORT_MODULES
from easyicu.concept.support_exports import select_support_cohort
from easyicu.concept.thesis_support import (
    FrozenReleaseParameter,
    PINNED_SUPPORT_MODULES,
    PinnedInputError,
    PinnedModuleInput,
    load_thesis_support,
    open_pinned_modules,
    pinned_stay_ids,
    read_frozen_release_parameter,
    read_hourly_contract,
    read_pinned_columns,
    support_input_files,
    thesis_support_manifest,
    validate_frozen_release_parameter,
    validate_pinned_inputs,
)
from easyicu.config import DATABASE_ID_CONFIG
from easyicu.datasource import storage_source
from easyicu.runtime import duckdb_spill
from easyicu.storage_binding import StorageSourceChanged, bind_files, binding_unit
from easyicu.storage_layout import StorageIntegrityError, parquet_relation

_FROZEN_SOURCE_DIR = Path(
    "/home/zhuhb/workspace/claude-ws/_wt/ys351lcommon/frozen_copies/L-COMMON/src/phd_common"
)
_FROZEN_DIGESTS = {
    "sources": "35f160b2d8105af7ed76a887ceb247117676b255b64a9f7a7537fc18f4a1bdc6",
    "new_sources": "d189e0ae1deefb19547332f269e72330a1eda54c95f23f55a69c0d119443258f",
}
_PINNED_REFERENCES = {
    "hourly_core": (
        Path("/home/zhuhb/workspace/claude-ws/_wt/rfrunL/L-HC/frozen/hc/src/data_processing/hourly_core.py"),
        "5b073a7c9f899929ea4d823b6eb5d3b1ef2ac597e89f40afe70b59ef2858c759",
    ),
    "cohort": (
        _FROZEN_SOURCE_DIR / "cohort.py",
        "d8e529cdc8034c990c2ee97e3f0f9b2cf8866087b1c975f295a7d15ae628f9a1",
    ),
    "build_flat_tables": (
        _FROZEN_SOURCE_DIR.parents[1] / "scripts" / "build_flat_tables.py",
        "44b0f250132f90f7790e70092f02a4908b7cb25cb42340f92ba58afce860c990",
    ),
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
    if name not in _FROZEN_DIGESTS:
        raise ValueError("unexpected frozen source path")
    path = Path(os.path.abspath(source_dir)) / f"{name}.py"
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


def _reference(source, family, stay, ids, frozen_dir, tables, *, empty):
    files = [support_input_files(source, table, stay, ids) for table in tables]
    if any(paths is None for paths in files):
        raise ValueError("reference input table unavailable")
    if ids == [] or any(not paths for paths in files):
        return empty

    virtual = Path("/__thesis_frozen__")
    rel = SimpleNamespace(
        raw=lambda db: virtual / "raw" / db,
        common_dir=virtual,
        corrected={"aumc_pao2_view": virtual / "unrelated_pao2"},
    )
    relations = [parquet_relation(paths, ordered=False) for paths in files]
    with duckdb_spill.connection() as con:
        if ids is None:
            # The first input is the raw stay table for each frozen PEEP builder.
            con.execute(f"""
                CREATE TEMP VIEW selected AS
                SELECT DISTINCT {stay} AS stay_id, {_quote(family)} AS database
                FROM {relations[0]}
            """)
        else:
            con.register("selected", pd.DataFrame({
                "stay_id": pd.Series(ids, dtype="int64"), "database": family,
            }))
        name = "new_sources" if family == "sic" else "sources"
        with _frozen_module(frozen_dir, name) as frozen:
            return _peep_reference(_FrozenSQL(con), rel, frozen, family, relations)


def _verify_first_in_hosp_reference_binding(table, path, expected_sha256):
    try:
        bind_files(table, (path,), sha256=True)
        # An explicitly bound digest may be cached even when the stat is unchanged.
        with path.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
    except (OSError, StorageSourceChanged):
        raise StorageSourceChanged("storage_source_changed", table) from None
    if digest != expected_sha256:
        raise StorageSourceChanged("storage_source_changed", table)


def _first_in_hosp_reference(source, family, stay, ids, receipt):
    # The predicate text below is copied from build_c565v6.py:99-107.
    expression = {
        "aumc": "admissioncount = 1",
        "sic": "COALESCE(OffsetAfterFirstAdmission, 0) = 0",
    }[family]
    table = receipt["source_table"]
    try:
        files = support_input_files(source, table, stay, None) or ()
        path = files[0].resolve(strict=True) if len(files) == 1 else None
    except (OSError, RuntimeError, StorageIntegrityError):
        raise StorageSourceChanged("storage_source_changed", table) from None
    if path != Path(receipt["source_realpath"]):
        raise StorageSourceChanged("storage_source_changed", table)
    _verify_first_in_hosp_reference_binding(table, path, receipt["source_sha256"])
    relation = parquet_relation((path,), ordered=False)
    with duckdb_spill.connection() as con:
        con.execute(f"""
            CREATE TEMP TABLE ordk AS
            SELECT {stay} stay_id, {expression} first_in_hosp
            FROM {relation}
        """)
        if ids is not None:
            con.register("hc", pd.DataFrame({"stay_id": pd.Series(ids, dtype="int64")}))
            con.execute("DELETE FROM ordk WHERE stay_id NOT IN (SELECT stay_id FROM hc)")
        rows = con.execute("SELECT stay_id, first_in_hosp AS first_in_hosp_frozen FROM ordk").df()
    _verify_first_in_hosp_reference_binding(table, path, receipt["source_sha256"])
    return rows


def _pinned_reference_ast(name):
    path, expected = _PINNED_REFERENCES[name]
    if path.resolve() != path:
        raise ValueError("frozen source path redirects")
    source = path.read_bytes()
    if hashlib.sha256(source).hexdigest() != expected:
        raise ValueError("unexpected frozen source digest")
    return path, ast.parse(source, filename=str(path))


def _reference_functions(reference, names, **bindings):
    """Execute the original selected definitions and constants from an exact source file."""
    path, tree = reference
    nodes = []
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            nodes.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in names:
            nodes.append(node)
        elif isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id in names for target in node.targets
        ):
            nodes.append(node)
    # Path and source bytes have already matched the frozen allowlist.
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), bindings)  # noqa: S102
    if set(names) - bindings.keys():
        raise ValueError("frozen reference definition missing")
    return SimpleNamespace(**{name: bindings[name] for name in names})


def _reference_sql(reference, fragment, **bindings):
    """Evaluate a single frozen SQL f-string, preserving its SQL text."""
    path, tree = reference
    (query,) = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.JoinedStr) and any(
            isinstance(part, ast.Constant) and fragment in part.value for part in node.values
        )
    ]
    return eval(compile(ast.Expression(query), str(path), "eval"), bindings)  # noqa: S307


def _pinned_rrt_reference(con, database, opened, spine, hourly, cohort, contract):
    renal = read_pinned_columns(opened["renal"], stays=spine["stay_id"].unique())
    renal["stay_id"] = hourly._normalise_stay_id(renal["stay_id"], context="renal")
    prepared, _ = hourly._prepare_dynamic(
        renal, {"rrt": contract["dynamic"]["renal"]["rrt"]}, module="renal",
    )
    # The frozen spine's exact left join drops the same rows as its explicit range filter.
    hc = spine.merge(prepared, on=hourly.KEY, how="left", validate="one_to_one", sort=False)
    con.register("hc", hc)
    query = cohort.rrt_hours_sql(SimpleNamespace(corrected={}), database)
    return con.execute(f"""
        SELECT stay_id, icu_hour AS charttime, TRUE AS rrt_active FROM ({query})
    """).df()


def _pinned_hirid_reference(con, opened, spine, ids, cohort_reference):
    entry, handle = opened["respiratory"]
    path = Path(entry.path)

    def input_stream(requested):
        if requested != path:
            raise ValueError("unbound frozen respiratory input")
        handle.seek(0)
        return handle

    def read_table(requested, *, columns, filters):
        if set(columns) != set(entry.allowed_columns):
            raise ValueError("unapproved frozen respiratory projection")
        return pq.read_table(input_stream(requested), columns=columns, filters=filters)

    flat = _reference_functions(
        _pinned_reference_ast("build_flat_tables"),
        ("VENT_ANCHOR_RULE", "VENT_ANCHOR_INCLUDES_NIV", "TIME_KEY", "_normalised_key_frame",
         "_assert_unique_keys", "_read_cols", "build_vent_anchor"),
        pd=pd, np=np, print=lambda *args: None, pq=SimpleNamespace(
            read_schema=lambda requested: pq.read_schema(input_stream(requested)),
            read_table=read_table,
        ),
    )
    anchors, _ = flat.build_vent_anchor(path.parent, "hirid", ids)
    con.register("anchor_input", anchors)
    con.register("stays", spine[["stay_id", "icu_los_h"]].drop_duplicates())
    sql = _FrozenSQL(con)
    virtual = Path("/__thesis_frozen__")
    sql.bind(virtual / "flat" / "hirid_vent_anchor.parquet", "anchor_input")
    anchor = _reference_sql(
        cohort_reference, "greatest(imv_t0_h, 0) AS anchor_t0_h",
        q=_quote, rel=SimpleNamespace(derived_shared=virtual), db="hirid",
    )
    episode = _reference_sql(cohort_reference, "CAST(floor(a.anchor_t0_h) AS INTEGER) AS t0_hour")
    return sql.execute(f"""
        WITH anchor AS ({anchor}), {episode} projected AS (
            SELECT e.stay_id, e.t0_hour AS charttime, TRUE AS imv,
                   e.ep_end_hour AS imv_recorded_end_hour
            FROM ep e JOIN stays s USING (stay_id)
            WHERE e.t0_hour < s.icu_los_h
        )
        SELECT stay_id, charttime, imv, imv_recorded_end_hour FROM projected
    """).df()


def _pinned_hourly_imv_reference(con, opened, spine, hourly, cohort_reference, gap):
    respiratory = read_pinned_columns(opened["respiratory"], stays=spine["stay_id"].unique())
    respiratory = respiratory.dropna(subset=["mech_vent"])
    respiratory["stay_id"] = hourly._normalise_stay_id(respiratory["stay_id"], context="respiratory")
    mapping = read_hourly_contract()["dynamic"]["respiratory"]
    # Restrict the frozen binning kernel to the requested ICU spine and apply
    # its derived-feature case rule before ranking modes in the same hour.
    hour = np.floor(pd.to_numeric(respiratory["charttime"], errors="coerce"))
    last = respiratory["stay_id"].map(spine.groupby("stay_id")["icu_hour"].max())
    on_spine = hour.between(spine["icu_hour"].min(), last)
    modes = respiratory["mech_vent"].astype("string").str.lower()
    repeated = respiratory.assign(icu_hour=hour).duplicated(hourly.KEY, keep=False)
    # A singleton unknown mode cannot contribute to IMV. Keep unknown modes
    # in collisions so the frozen rank validation rejects them.
    keep = hour.isna() | (on_spine & (repeated | modes.isin(["invasive", "noninvasive"])))
    prepared, _ = hourly._prepare_dynamic(
        respiratory.loc[keep].assign(mech_vent=modes), {"mech_vent": mapping["mech_vent"]}, module="respiratory",
    )
    hc = spine.merge(prepared, on=hourly.KEY, how="left", validate="one_to_one", sort=False)
    # hourly_core.py:385,392 derives IMV from the normalized, binned vent type.
    vent_type = hc["vent_type"].astype("string").str.lower()
    con.register("hc", hc[["stay_id", "icu_hour"]].assign(imv=vent_type.eq("invasive").fillna(False)))
    episode = _reference_sql(cohort_reference, "ih AS (SELECT stay_id, icu_hour AS h", gap=gap)
    return con.execute(f"""
        WITH {episode.rstrip().removesuffix(",")}
        SELECT stay_id, t0_hour AS charttime, TRUE AS imv, ep_end_hour AS imv_recorded_end_hour
        FROM ep
    """).df()


def _pinned_reference(module, database, pinned_inputs, stay_ids, empty, frozen_release_parameter=None):
    if stay_ids == set():
        return empty
    hourly = _reference_functions(
        _pinned_reference_ast("hourly_core"),
        ("KEY", "_normalise_stay_id", "_assert_unique", "build_hourly_spine", "_prepare_dynamic"),
        np=np, pd=pd,
    )
    cohort_reference = _pinned_reference_ast("cohort")
    parameter = (
        read_frozen_release_parameter(frozen_release_parameter) if frozen_release_parameter is not None else None
    )
    with open_pinned_modules(module, database, pinned_inputs) as opened, duckdb_spill.connection() as con:
        outcome = read_pinned_columns(opened["outcome"], stays=stay_ids)
        spine, _ = hourly.build_hourly_spine(outcome, stay_ids=stay_ids)
        if module == "rrt_time":
            cohort = _reference_functions(cohort_reference, ("rrt_hours_sql",))
            return _pinned_rrt_reference(con, database, opened, spine, hourly, cohort, read_hourly_contract())
        if parameter is not None:
            return _pinned_hourly_imv_reference(con, opened, spine, hourly, cohort_reference, parameter["value"])
        return _pinned_hirid_reference(con, opened, spine, stay_ids, cohort_reference)


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


def run_equivalence(
    *, database, module, db_root=None, stay_ids=None, frozen_src_dir=_FROZEN_SOURCE_DIR,
    pinned_inputs=None, frozen_release_parameter=None,
):
    """Compare the selected stays, defaulting to all raw stays outside MIIV."""
    family = database.removesuffix("_demo")
    if module in PINNED_SUPPORT_MODULES:
        database = family
    stay = DATABASE_ID_CONFIG[database]["id_col"]
    with binding_unit():
        options = {} if pinned_inputs is None else {"pinned_inputs": tuple(pinned_inputs)}
        if frozen_release_parameter is not None:
            options["frozen_release_parameter"] = frozen_release_parameter
        if module in PINNED_SUPPORT_MODULES:
            stay_ids = pinned_stay_ids(database, stay_ids)
        actual = load_thesis_support(
            module=module, database=database, data_path=db_root, patient_ids=stay_ids, **options,
        )
        receipt = actual.attrs.get("thesis_support") or thesis_support_manifest(
            module, database, actual.attrs["available_fields"],
        )
        if receipt["status"] != "projected":
            if module in PINNED_SUPPORT_MODULES:
                return {key: receipt[key] for key in ("status", "reason", "missing_inputs") if key in receipt}
            return {"status": receipt["status"]}
        if module in PINNED_SUPPORT_MODULES:
            actual = actual.rename(columns={stay: "stay_id"})
            try:
                expected = _pinned_reference(
                    module, database, options["pinned_inputs"], stay_ids, actual.iloc[:0], frozen_release_parameter,
                )
            except PinnedInputError:
                raise
            except (Exception, SystemExit) as error:
                return {"status": "comparison_failed", "error_type": type(error).__name__}
            return compare_rows(actual, expected.astype(actual.dtypes), key_columns=["stay_id", "charttime"])
        selection = select_support_cohort(database, db_root, stay_ids)
        ids = None if selection is None else selection[stay]
        actual = actual.rename(columns={stay: "stay_id"})
        source = storage_source(database, db_root)
        keys = ["stay_id"]
        if module == FIRST_IN_HOSP_FROZEN:
            expected = _first_in_hosp_reference(source, family, stay, ids, receipt)
        else:
            keys.append("charttime")
            expected = _reference(
                source, family, stay, ids, frozen_src_dir, receipt["input_tables"],
                empty=actual.iloc[:0],
            )
    expected = expected.astype(actual.dtypes)
    return compare_rows(actual, expected, key_columns=keys)


class _DiscardOutput(TextIOBase):
    def write(self, message):
        return len(message)


class _PrivateArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        print('{"status": "invalid_arguments"}')
        raise SystemExit(2)


def _read_cohort_file(path, database):
    if path.suffix.lower() in {".parquet", ".pq"}:
        import pyarrow as pa
        import pyarrow.parquet as pq

        with pq.ParquetFile(path) as parquet:
            schema = parquet.schema_arrow
            native = DATABASE_ID_CONFIG[database]["id_col"]
            column = native if native in schema.names else "stay_id"
            if column not in schema.names or not pa.types.is_integer(schema.field(column).type):
                raise ValueError("cohort Parquet requires an integer stay ID column")
            values = parquet.read(columns=[column]).column(column)
            if values.null_count:
                raise ValueError("cohort stay IDs must be non-null")
            ids = values.to_pylist()
    else:
        with path.open(encoding="utf-8") as cohort:
            ids = [int(line) for line in cohort if line.strip()]
    if any(not -(2**63) <= value < 2**63 for value in ids):
        raise ValueError("cohort stay IDs must fit a signed 64-bit integer")
    return ids


def main(argv=None):
    parser = _PrivateArgumentParser(description=__doc__)
    parser.add_argument("--database", choices=("aumc", "eicu", "hirid", "miiv", "mimic", "sic"), required=True)
    parser.add_argument("--db-root", type=Path)
    parser.add_argument(
        "--pinned-input", action="append", nargs=4, metavar=("MODULE", "PATH", "SHA256", "COLUMNS"),
        help="D0 file, full expected SHA-256 and comma-separated projected columns; repeat per file",
    )
    parser.add_argument(
        "--frozen-release-parameter", nargs=2, metavar=("PATH", "SHA256"),
        help="S1345 receipt and full SHA-256 for the frozen IMV episode gap",
    )
    parser.add_argument("--common-version", help="version in the S1345 receipt (default: v0.2)")
    cohort = parser.add_mutually_exclusive_group()
    cohort.add_argument("--stay-ids", type=int, nargs="+", help="explicit synthetic stay IDs")
    cohort.add_argument("--stay-ids-file", type=Path, help="approved private Parquet or text cohort file")
    parser.add_argument("--module", choices=THESIS_SUPPORT_MODULES, required=True)
    parser.add_argument("--frozen-src-dir", type=Path, default=_FROZEN_SOURCE_DIR,
                        help="directory containing SHA-256-pinned frozen sources.py and new_sources.py")
    args = parser.parse_args(argv)
    if args.module in PINNED_SUPPORT_MODULES:
        if args.db_root is not None:
            parser.error("pinned support does not use a raw database root")
    elif args.db_root is None or args.pinned_input or args.frozen_release_parameter:
        parser.error("raw support requires --db-root")
    if args.common_version is not None and args.frozen_release_parameter is None:
        parser.error("a common version requires a frozen release parameter")
    if args.module == FIRST_IN_HOSP_FROZEN and args.stay_ids is not None:
        parser.error("first_in_hosp_frozen requires --stay-ids-file for explicit cohorts")
    if (args.database == "miiv" and args.module != FIRST_IN_HOSP_FROZEN
            and args.stay_ids is None and args.stay_ids_file is None):
        parser.error("MIIV requires an explicit stay cohort")
    # A backend exception can contain a raw SQL value or path. Keep its
    # diagnostic private and emit a fixed status code at this boundary.
    with _DiscardOutput() as sink, redirect_stdout(sink), redirect_stderr(sink):
        prior_level = logging.root.manager.disable
        logging.disable(logging.CRITICAL)
        try:
            options = vars(args)
            cohort_file = options.pop("stay_ids_file")
            pinned = options.pop("pinned_input")
            parameter = options.pop("frozen_release_parameter")
            version = options.pop("common_version")
            if parameter is not None:
                path, digest = parameter
                version_arg = {} if version is None else {"common_version": version}
                options["frozen_release_parameter"] = FrozenReleaseParameter(Path(path), digest, **version_arg)
                validate_frozen_release_parameter(args.module, args.database, options["frozen_release_parameter"])
            if pinned is not None:
                options["pinned_inputs"] = tuple(
                    PinnedModuleInput(module, Path(path), digest, tuple(columns.split(",")))
                    for module, path, digest, columns in pinned
                )
                validate_pinned_inputs(args.module, args.database, options["pinned_inputs"])
            if cohort_file is not None:
                options["stay_ids"] = _read_cohort_file(cohort_file, args.database)
            result = run_equivalence(**options)
        except PinnedInputError as error:
            result = {"status": "comparison_failed", "reason": error.reason_code}
        except Exception as error:
            result = {"status": "comparison_failed", "error_type": type(error).__name__}
        finally:
            logging.disable(prior_level)
    print(json.dumps(result, sort_keys=True))
    return {"agree": 0, "different": 1}.get(result["status"], 2)


if __name__ == "__main__":
    sys.exit(main())
