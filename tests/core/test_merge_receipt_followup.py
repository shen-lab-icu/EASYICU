"""r13mergeidx follow-up (delta22m): contract axis, row-bearing axis check,
committed-attempt receipts and typed missing-sidecar failures.  Synthetic rows only."""

from __future__ import annotations

import json

import pandas as pd
import pytest

import easyicu
from easyicu.api import extraction as api
from easyicu.concept import ConceptResolver
from easyicu.concept.merge_time_index import RELATIVE_HOURS, record_empty_index_fill
from easyicu.table import ICUTable

HOUR = pd.Timedelta(hours=1)


def _aligned(name, rows):
    frame = pd.DataFrame(rows, columns=["icustay_id", "charttime", name]).astype(
        {"icustay_id": "int64", "charttime": "float64", name: "float64"})
    return ICUTable(frame, id_columns=["icustay_id"], index_column="charttime", value_column=name,
                    time_basis=RELATIVE_HOURS, interval=HOUR)


def _declared_empty(name):
    return _aligned(name, [])


def _raw_empty(name):
    frame = pd.DataFrame({"icustay_id": pd.Series(dtype="int64"),
                          "charttime": pd.Series(dtype="datetime64[ns]"),
                          name: pd.Series(dtype="float64"), "valueuom": pd.Series(dtype="object")})
    return ICUTable(frame, id_columns=["icustay_id"], index_column=None, value_column=name,
                    time_columns=["charttime"])


def _merge(tables):
    return ConceptResolver({})._merge_tables(tables, relative_axis=True)


def _schema(frame):
    return {c: str(t) for c, t in frame.dtypes.items()}



def _default_schema(names):
    return _schema(_merge({n: _aligned(n, [(1, 0.0, 1.0)]) for n in names}))


@pytest.mark.parametrize("combo", [
    ("raw", "raw"), ("declared", "raw"), ("raw", "declared"), ("declared", "declared"),
    ("raw", "raw", "raw"),
], ids=lambda c: "_".join(c))
def test_an_all_empty_batch_has_the_default_schema(combo):
    names = [f"c{i}" for i in range(len(combo))]
    tables = {n: (_raw_empty(n) if kind == "raw" else _declared_empty(n)) for n, kind in zip(names, combo)}
    merged = _merge(tables)
    assert merged.empty
    assert not merged.columns.duplicated().any()
    assert _schema(merged) == _default_schema(names)


def test_all_empty_without_a_relative_contract_is_left_unchanged():
    tables = {"a": _raw_empty("a"), "b": _raw_empty("b")}
    merged = ConceptResolver({})._merge_tables(tables)
    assert "valueuom" not in merged.columns or merged.empty


@pytest.mark.parametrize("first", [True, False], ids=["populated_no_index_first", "populated_no_index_last"])
def test_a_populated_table_without_the_index_is_refused_in_either_order(first):
    id_only = ICUTable(pd.DataFrame({"icustay_id": [1], "b": [5.0]}), id_columns=["icustay_id"],
                       index_column=None, value_column="b")
    tables = {"b": id_only, "a": _aligned("a", [(1, 0.0, 1.0)])} if first else \
        {"a": _aligned("a", [(1, 0.0, 1.0)]), "b": id_only}
    with pytest.raises(ValueError, match="identical index column"):
        _merge(tables)


def test_id_only_merges_are_unchanged():
    a = ICUTable(pd.DataFrame({"icustay_id": [1, 2], "a": [1.0, 2.0]}), id_columns=["icustay_id"],
                 index_column=None, value_column="a")
    b = ICUTable(pd.DataFrame({"icustay_id": [2], "b": [3.0]}), id_columns=["icustay_id"],
                 index_column=None, value_column="b")
    merged = _merge({"a": a, "b": b}).sort_values("icustay_id").reset_index(drop=True)
    assert merged["b"].tolist()[1] == 3.0 and pd.isna(merged["b"].tolist()[0])


# ── receipts: only the committed attempt counts ─────────────────────────
@pytest.mark.parametrize("second_fills", [[], ["kept_signal"]])
def test_a_failed_attempts_fills_are_dropped_before_the_retry(tmp_path, monkeypatch, second_fills):
    calls = []

    def fake_load_concepts(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            record_empty_index_fill("stale_signal")
            raise MemoryError("injected")
        for name in second_fills:
            record_empty_index_fill(name)
        ids = [1, 2]
        return pd.DataFrame({"stay_id": ids, "charttime": [0.0, 0.0], "test_signal": [1.0, 2.0]})

    monkeypatch.setattr(easyicu, "load_concepts", fake_load_concepts)
    api._run_module_extraction(
        "test_module", ["test_signal"], "miiv", str(tmp_path), {"stay_id": [1, 2]}, None,
        str(tmp_path), stream_output_batches=False,
    )
    manifest = json.loads((tmp_path / "_manifest.json").read_text())
    assert len(calls) == 2
    assert manifest["merge_empty_index_filled"] == [{"batch_id": None, "concept": c} for c in second_fills]


# ── a missing isolated-batch sidecar is a typed stream failure ──────────
class _Run:
    def __init__(self, *, target, args, daemon):
        self.target, self.args, self.exitcode = target, args, None

    def start(self):
        try:
            self.target(*self.args)
            self.exitcode = 0
        except BaseException:
            self.exitcode = 1

    def join(self):
        return None


@pytest.mark.parametrize("kind", ["bounds", "batch_receipt"])
def test_a_missing_sidecar_is_a_typed_stream_failure(tmp_path, monkeypatch, kind):
    real = api._write_isolated_stream_batch

    def drop_sidecar(module, concepts, load_kwargs, patient_ids, destination):
        real(module, concepts, load_kwargs, patient_ids, destination)
        sidecar = (api._stream_batch_bounds_sidecar if kind == "bounds" else api._stream_batch_fill_sidecar)
        sidecar(destination).unlink()

    class Context:
        @staticmethod
        def Process(*, target, args, daemon):
            return _Run(target=drop_sidecar, args=args, daemon=daemon)

    monkeypatch.setattr(easyicu, "load_concepts", lambda **kw: pd.DataFrame({
        "stay_id": kw["patient_ids"]["stay_id"], "charttime": 0.0, "test_signal": 1.0}))
    monkeypatch.setattr(api, "_extract_worker_env_setup", lambda _path: None)
    monkeypatch.setattr(api, "_get_extraction_mp_context", lambda _mp: Context())
    monkeypatch.setattr(api, "_ISOLATED_STREAM_BATCH_TARGETS", frozenset({("eicu", "test_module")}))
    monkeypatch.setattr(api, "_cgroup_oom_kills", lambda: None)
    monkeypatch.setattr(api, "_host_reports_oom_kills", lambda: False)
    api._run_module_extraction(
        "test_module", ["test_signal"], "eicu", str(tmp_path), {"stay_id": list(range(1, 5))}, 2,
        str(tmp_path), stream_output_batches=True,
    )
    manifest = json.loads((tmp_path / "_manifest.json").read_text())
    failure = manifest["stream_failure"]
    assert failure["exception_type"] == "StreamBatchReceiptMissing"
    assert failure["failure_class"] == "batch_receipt_missing"
    assert failure["sidecar_kind"] == kind
    assert (failure["batch_index"], failure["start_offset"], failure["batch_size"]) == (1, 0, 2)
    assert manifest["errors"]


SOFA2_GROUP = ["sofa2", "sofa2_cns_ascertainment", "sofa2_cns_delirium_tx_ascertainment",
               "sofa2_cns_proxy_sensitivity"]


@pytest.mark.parametrize("raw_first", [True, False])
def test_an_all_empty_sofa2_group_batch_has_the_default_schema(raw_first):
    kinds = ["raw", "declared", "raw", "declared"] if raw_first else ["declared", "raw", "declared", "raw"]
    tables = {n: (_raw_empty(n) if k == "raw" else _declared_empty(n)) for n, k in zip(SOFA2_GROUP, kinds)}
    merged = _merge(tables)
    assert merged.empty and _schema(merged) == _default_schema(SOFA2_GROUP)


def test_renal_urine_chem_published_at_batch_one_matches_the_default(tmp_path, monkeypatch):
    """Batch 1 makes stays without urine rows all-empty batches."""
    import pyarrow.parquet as pq

    from easyicu.api.concepts import clear_global_loader
    from tests.support.urine_chemistry_source import urine_module_source

    monkeypatch.setenv("EASYICU_NATIVE_V2_REAL_ROOT", "")
    source = tmp_path / "source"
    urine_module_source(source, "mimic")
    outs = {}
    try:
        for label, batch in (("default", None), ("batch1", 1)):
            out = tmp_path / label
            kwargs = {"batch_size": batch, "stream_output_batches": True} if batch else {}
            result = api.extract_database(
                "mimic", data_path=source, output_dir=out, modules=["outcome", "renal_urine_chem"],
                native_export_v2=True, resource_budget_mb=2048, verbose=False, **kwargs,
            )
            assert not {n: i["errors"] for n, i in result["modules"].items() if i.get("errors")}
            outs[label] = out
            clear_global_loader()
        a, b = (outs[k] / "renal_urine_chem.parquet" for k in ("default", "batch1"))
        assert pq.read_schema(a).equals(pq.read_schema(b), check_metadata=False)
        assert a.read_bytes() == b.read_bytes()
        statuses = [
            {f["module"]: f["concept_status"] for f in json.loads((outs[k] / "_manifest.json").read_text())["files"]}
            for k in ("default", "batch1")
        ]
        assert statuses[0] == statuses[1]
    finally:
        clear_global_loader()
