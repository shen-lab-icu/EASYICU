"""Typed value failures pass every broad catch on the value path.

Codex delta2 D2-AGG-CATCH: ``_load_recursive_concept`` wrapped its interval
step in ``except Exception`` and kept the unaggregated result when
``change_interval`` raised ``IntervalAggregationError``; other value-path
fallbacks (stay-ID alignment, the MIMIC-IV ICU-patient filter, the admitted
fluid balance, the medication bundle report, the legacy ConceptLoader) turned
a ``StorageIntegrityError`` into "returned the input", "empty" or "partial".
Each now re-raises :data:`easyicu.value_failures.TYPED_VALUE_FAILURES` first.
All rows are invented.
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import easyicu
from easyicu.io import ts_utils
from easyicu.io.ts_utils import IntervalAggregationError, TimeAxisUnitError
from easyicu.resolver_errors import (
    ResolverBusyError,
    ResolverCallClosed,
    ResolverJoinTimeout,
    ResolverPoolExhausted,
)
from easyicu.storage_layout import StorageIntegrityError, TableScanError
from easyicu.utils.exact_sum import ExactSumOverflowError
from easyicu.value_failures import TYPED_VALUE_FAILURES
from tests.support.urine_raw_source import _load, _source

pytestmark = pytest.mark.filterwarnings("ignore")

_ORIGIN = pd.Timestamp("2180-01-01 00:00")


def _vanished(table="icustays"):
    return TableScanError("table_scan_files_missing", table, "listed output absent")


def _miiv(root):
    stays = pd.DataFrame({"stay_id": [1], "subject_id": [1], "hadm_id": [1],
                          "intime": [_ORIGIN], "outtime": [_ORIGIN + pd.Timedelta(days=2)]})
    stamps = _ORIGIN + pd.to_timedelta([2, 2, 5, 5], unit="h")
    chart = pd.DataFrame({"stay_id": 1, "subject_id": 1, "hadm_id": 1, "charttime": stamps,
                          "itemid": [220045, 220179, 220045, 220179],
                          "valuenum": [90.0, 120.0, 100.0, 110.0], "value": ["x"] * 4,
                          "valueuom": ["bpm", "mmHg", "bpm", "mmHg"]})
    return _source(root, "miiv", icustays=stays, chartevents=chart)


def test_the_shared_tuple_names_every_type():
    # lead-r13 delta4: the exact-sum refusal (D4-EXACT-CATCH) and the undeclared
    # time-axis unit (D4-AXIS-CATCH) are value failures too; a broad catch must
    # not turn them into an empty or unaggregated result. lead-r13 delta18s:
    # a busy shared resolver (B18-REENTRY-DEADLOCK) is refused, not swallowed;
    # delta27s: so are a copied context outliving its call and an exhausted
    # resolver pool.
    assert set(TYPED_VALUE_FAILURES) == {
        IntervalAggregationError, StorageIntegrityError, ExactSumOverflowError,
        TimeAxisUnitError, ResolverBusyError, ResolverCallClosed, ResolverPoolExhausted,
        ResolverJoinTimeout,  # delta35t D35-JOIN-DEADLINE
    }


_BIG = 1.7976931348623157e308


def _overflow():
    return ExactSumOverflowError(3)


def test_recursive_interval_step_does_not_keep_the_result_on_an_exact_sum_refusal(tmp_path, monkeypatch):
    source = _miiv(tmp_path / "miiv")
    real = ts_utils.change_interval

    def failing(*args, **kwargs):
        if sys._getframe(1).f_code.co_name == "_load_recursive_concept":
            raise _overflow()
        return real(*args, **kwargs)

    monkeypatch.setattr(ts_utils, "change_interval", failing)
    with pytest.raises(ExactSumOverflowError):
        _load(source, ["shock_index"])


def test_admitted_fluid_balance_is_not_empty_when_its_sum_is_refused():
    from easyicu.concept import callbacks as concept_callbacks
    from easyicu.table import ICUTable

    class _Inputs:
        config = SimpleNamespace(name="miiv")

        def load_table(self, *args, **kwargs):
            # Two finite amounts in one hour whose exact total is beyond float64.
            return pd.DataFrame({"stay_id": [1, 1], "starttime": [1.2, 1.7], "amount": [_BIG, _BIG],
                                 "amountuom": ["mL", "mL"], "statusdescription": ["FinishedRunning"] * 2})

    urine = ICUTable(data=pd.DataFrame({"stay_id": [1], "charttime": [1.0], "urine": [50.0]}),
                     id_columns=["stay_id"], index_column="charttime", value_column="urine")
    ctx = SimpleNamespace(data_source=_Inputs(), patient_ids=None, interval=None,
                          resolver=None, kwargs={}, database="miiv", concept_name="fluid_balance_admitted")
    with pytest.raises(ExactSumOverflowError):
        concept_callbacks._callback_fluid_balance_admitted({"urine": urine}, ctx)


def test_recursive_interval_step_does_not_keep_the_unaggregated_result(tmp_path, monkeypatch):
    baseline = _load(_miiv(tmp_path / "ok"), ["shock_index"])["shock_index"]
    assert len(baseline.data) == 2  # the real path works on these rows
    source = _miiv(tmp_path / "miiv")  # a fresh root: no cached concept result

    real = ts_utils.change_interval

    def failing(*args, **kwargs):
        if sys._getframe(1).f_code.co_name == "_load_recursive_concept":
            raise IntervalAggregationError(
                group_columns=["stay_id"], aggregations={"shock_index": "median"},
                cause=ValueError("injected"),
            )
        return real(*args, **kwargs)

    monkeypatch.setattr(ts_utils, "change_interval", failing)
    with pytest.raises(IntervalAggregationError):
        _load(source, ["shock_index"])


def _listed_but_gone_icustays(root):
    source = _miiv(root)
    (root / "conversion_manifest.json").write_text(json.dumps({"tables": [
        {"table": "icustays", "row_count": 1, "outputs": [{"relative_path": "icustays.parquet"}]},
    ]}))
    (root / "icustays.parquet").unlink()
    return source


@pytest.mark.parametrize("primary", ["subject_id", "stay_id"])
def test_alignment_with_relative_hours_does_not_hide_a_vanished_stay_table(tmp_path, primary):
    from easyicu.concept import ConceptResolver
    from easyicu.resources import load_dictionary

    source = _listed_but_gone_icustays(tmp_path / "miiv")
    frame = pd.DataFrame({primary: [1, 1], "charttime": [2.0, 5.0], "hr": [90.0, 100.0]})
    resolver = ConceptResolver(load_dictionary(include_sofa2=True))
    # Before: a numeric index returned the frame unchanged (or skipped the
    # admission merge) whatever the stay-table load raised.
    with pytest.raises(StorageIntegrityError) as caught:
        resolver._align_time_to_admission(frame, source, [primary], "charttime")
    assert caught.value.reason == "table_scan_files_missing"


class _VanishedSource:
    config = SimpleNamespace(name="miiv")

    def load_table(self, *args, **kwargs):
        raise _vanished()


def test_miiv_icu_patient_filters_do_not_return_the_unfiltered_input(monkeypatch):
    from easyicu import callbacks as legacy
    from easyicu.concept import callbacks as concept_callbacks
    from easyicu.datasource import ICUDataSource
    from easyicu.table import ICUTable

    monkeypatch.setattr(ICUDataSource, "get_instance", lambda *_: _VanishedSource(), raising=False)
    patients = pd.DataFrame({"subject_id": [1], "anchor_age": [60]})
    with pytest.raises(StorageIntegrityError):
        legacy.miiv_icu_patients_filter(patients, database="miiv")

    table = ICUTable(data=patients, id_columns=["subject_id"], index_column=None, value_column="anchor_age")
    ctx = SimpleNamespace(database="miiv")
    with pytest.raises(StorageIntegrityError):
        concept_callbacks._callback_miiv_icu_patients_filter({"age": table}, ctx)


def test_admitted_fluid_balance_is_not_empty_when_its_input_table_vanished():
    from easyicu.concept import callbacks as concept_callbacks
    from easyicu.table import ICUTable

    urine = ICUTable(data=pd.DataFrame({"stay_id": [1], "charttime": [1.0], "urine": [50.0]}),
                     id_columns=["stay_id"], index_column="charttime", value_column="urine")
    ctx = SimpleNamespace(data_source=_VanishedSource(), patient_ids=None, interval=None,
                          resolver=None, kwargs={}, database="miiv", concept_name="fluid_balance_admitted")
    with pytest.raises(StorageIntegrityError):
        concept_callbacks._callback_fluid_balance_admitted({"urine": urine}, ctx)


def test_medication_bundle_does_not_report_a_vanished_table_as_one_failed_concept():
    from easyicu.api.medications import load_medications_impl

    def load(**_):
        raise _vanished("inputevents")

    with pytest.raises(StorageIntegrityError):
        load_medications_impl(
            load_concepts_fn=load, validate_concepts_fn=lambda concepts, _v: list(concepts)[:2],
            allow_partial=True,
        )


# --- structural guard -------------------------------------------------------

_VALUE_PATH = ("concept/", "callbacks.py", "io/ts_utils.py", "load_concepts.py", "scores/",
               "table/", "datasource.py", "api/", "patient_filter.py")
# The exact-sum entry points raise ExactSumOverflowError (lead-r13 delta4
# D4-EXACT-CATCH): a try around one is an aggregating try.
_EXACT_SUMS = {"exact_groupby_sum", "exact_groupby_agg", "exact_terms_to_values",
               "exact_group_totals", "apply_exact_columns"}
_LOADS = {"load_table", "_safe_load_table", "_load_one_concept", "_load_single_concept",
          "change_interval", "load_concepts", "load_concepts_fn", "resolve_target_files",
          "_align_time_to_admission", "load_id_tbl", "_load_recursive_concept",
          "resolve_table_files"} | _EXACT_SUMS
# A try that aggregates or loads a concept must also pass IntervalAggregationError
# and ExactSumOverflowError: only the shared tuple is accepted.
_AGGREGATING = {"change_interval", "_load_one_concept", "_load_single_concept", "load_concepts",
                "load_concepts_fn", "_load_recursive_concept"} | _EXACT_SUMS
# The module runner records a failed module in its manifest (never writes it).
_RECORDING_BOUNDARIES = {("api/extraction.py", "_run_module_extraction")}


def _names(handler):
    if handler.type is None:
        return ["<bare>"]
    elts = handler.type.elts if isinstance(handler.type, ast.Tuple) else [handler.type]
    return [ast.unparse(e) for e in elts]


def _always_raises(handler):
    nodes = [x for stmt in handler.body for x in ast.walk(stmt)]
    escapes = any(isinstance(x, (ast.Return, ast.Continue, ast.Pass, ast.Break)) for x in nodes)
    return isinstance(handler.body[-1], ast.Raise) and not escapes


def test_no_broad_catch_around_a_load_or_aggregation_drops_the_typed_failures():
    root = Path(easyicu.__file__).parent
    offenders = []
    for path in sorted(root.rglob("*.py")):
        rel = path.relative_to(root).as_posix()
        if not rel.startswith(_VALUE_PATH):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
        for node in ast.walk(tree):
            if not isinstance(node, ast.Try):
                continue
            called = {
                x.func.attr if isinstance(x.func, ast.Attribute) else getattr(x.func, "id", "")
                for stmt in node.body for x in ast.walk(stmt) if isinstance(x, ast.Call)
            }
            if not called & _LOADS:
                continue
            owner = node
            while owner in parents and not isinstance(owner, (ast.FunctionDef, ast.AsyncFunctionDef)):
                owner = parents[owner]
            if (rel, getattr(owner, "name", "")) in _RECORDING_BOUNDARIES:
                continue
            accepted = ("TYPED_VALUE_FAILURES",) if called & _AGGREGATING else (
                "TYPED_VALUE_FAILURES", "StorageIntegrityError")
            typed = False
            for handler in node.handlers:
                names = _names(handler)
                typed = typed or any(n.endswith(accepted) for n in names)
                broad = any(n in ("Exception", "BaseException", "<bare>") for n in names)
                if broad and not typed and not _always_raises(handler):
                    offenders.append(f"{rel}:{handler.lineno}")
    assert offenders == []
