"""A failure to count the bounds drops fails the read; it never reads as an absent source.

Codex delta7x B-AUDIT-CATCH: the counting runs inside the single-concept
source read, whose ``(KeyError, FileNotFoundError, ValueError)`` handler marks
a table unavailable and moves on. An accounting error of one of those types
(a missing hand-back file, say) silently dropped a healthy source. Every
accounting failure is now ``SourceBoundsAuditError`` (in
``TYPED_VALUE_FAILURES``), re-raised before any source-unavailable handler.
Each path gets an injected failure; none may skip the source or publish.
Synthetic rows only.
"""

from __future__ import annotations

import contextlib

import pytest

from easyicu import source_bounds
from easyicu.concept.aggregation_route import force_raw_aggregation
from easyicu.source_bounds import SourceBoundsAuditError, collect_bounds_drops
from easyicu.value_failures import TYPED_VALUE_FAILURES
from tests.core.test_bounds_drops_batch_invariant import _STAYS, _eicu, _sic
from tests.support.urine_raw_source import _load

pytestmark = pytest.mark.filterwarnings("ignore")


def _broken_digest(_stays):
    raise FileNotFoundError("injected: accounting store missing")


# The same requests take these routes in test_bounds_drops_batch_invariant
# (bucketed single, long-table batch, raw, wide-table batch, wide single column).
@pytest.mark.parametrize(("database", "names", "table", "kwargs"), [
    ("sic", ["temp"], "data_float_h", {}),
    ("sic", ["temp", "spo2"], "data_float_h", {}),
    ("sic", ["temp"], "data_float_h", {"raw": True}),
    ("eicu", ["pap_sys", "pap_mean"], "vitalperiodic", {}),
    ("eicu", ["pap_sys"], "vitalperiodic", {"aggregate": "min"}),
])
def test_an_accounting_failure_fails_the_read(tmp_path, monkeypatch, database, names, table, kwargs):
    source = _sic(tmp_path / "sic") if database == "sic" else _eicu(tmp_path / "eicu")
    raw = kwargs.pop("raw", False)
    monkeypatch.setattr(source_bounds, "stay_set_digest", _broken_digest)
    forced = force_raw_aggregation() if raw else contextlib.nullcontext()
    with forced, collect_bounds_drops():
        with pytest.raises(SourceBoundsAuditError) as caught:
            _load(source, names, **kwargs)
    assert isinstance(caught.value.__cause__, FileNotFoundError)
    assert f"{database}.{table}:" in caught.value.stage
    assert SourceBoundsAuditError in TYPED_VALUE_FAILURES


def _stream(tmp_path, *, isolated=False):
    from easyicu.api import extraction
    from easyicu.api.concepts import clear_global_loader

    source = _sic(tmp_path / "sic")
    out = tmp_path / ("isolated" if isolated else "in_process")
    out.mkdir()
    clear_global_loader()
    try:
        return out, extraction._stream_module_batches_to_parquet(
            "vitals", ["temp", "spo2"],
            dict(
                data_path=str(source.base_path), database="sic", concepts=["temp", "spo2"],
                verbose=False, merge=True, concept_workers=1, _defer_empty_columns_to_arrow=True,
            ),
            {"CaseID": list(range(1, _STAYS + 1))}, 2, str(out),
        )
    finally:
        clear_global_loader()


def test_a_failed_batch_commit_of_the_counts_fails_the_module(tmp_path, monkeypatch):
    from easyicu.api import extraction

    def broken_totals(_records):
        raise OSError("injected: ledger counts")

    monkeypatch.setattr(extraction, "bounds_totals", broken_totals)
    with pytest.raises(SourceBoundsAuditError) as caught:
        _stream(tmp_path)
    assert "stream_batch_commit" in caught.value.stage
    assert not (tmp_path / "in_process" / "vitals.parquet").exists()


def test_a_missing_isolated_hand_back_fails_the_module(tmp_path, monkeypatch):
    from pathlib import Path

    from easyicu.api import extraction

    monkeypatch.setattr(extraction, "_requires_isolated_stream_batch", lambda *args: True)
    # The parent looks for the child's counts where the child did not write them.
    monkeypatch.setattr(
        extraction, "_stream_batch_bounds_sidecar",
        lambda part: Path(part).with_name(f".{Path(part).name}.elsewhere.json"),
    )
    with pytest.raises(SourceBoundsAuditError) as caught:
        _stream(tmp_path, isolated=True)
    assert "isolated_batch_hand_back" in caught.value.stage
    assert isinstance(caught.value.__cause__, FileNotFoundError)
    assert not (tmp_path / "isolated" / "vitals.parquet").exists()
