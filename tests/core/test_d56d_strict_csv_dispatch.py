"""Owner-selected MIMIC CSV keeps strict semantics through the public reader.

Each regression fails at 01af932fe. Fixtures contain synthetic records only.
"""

import gzip

import pytest

from easyicu.config import load_src_cfg
from easyicu.datasource import FilterOp, FilterSpec, ICUDataSource
from easyicu.storage_layout import TableScanError
from tests.core.test_d25d_csv_projection import HEADER, _row


def _source(root, suffix, *, bad_unselected=False):
    rows = [_row(i) for i in range(1, 21)]
    rows[10:] = [row.replace(",220045,", ",999999,") for row in rows[10:]]
    if bad_unselected:
        # CGID is declared numeric but absent from the minimal projection.
        # Put the bad token outside the requested item set as well.
        rows[-1] = rows[-1].replace(",17,", ",invalid,")
    data = "\n".join([HEADER, *rows]) + "\n"
    path = root / ("CHARTEVENTS" + suffix)
    if suffix.endswith(".gz"):
        with gzip.open(path, "wt") as stream:
            stream.write(data)
    else:
        path.write_text(data)
    return ICUDataSource(load_src_cfg("mimic"), base_path=root, enable_cache=False)


def _public_read(source):
    return source.load_table(
        "chartevents", columns=["icustay_id", "itemid", "value"],
        filters=[FilterSpec("itemid", FilterOp.IN, [220045])],
    ).dataframe()


@pytest.mark.parametrize("suffix", [".csv", ".csv.gz"])
def test_public_reader_pushes_item_filter_into_the_csv_read(tmp_path, monkeypatch, suffix):
    source = _source(tmp_path, suffix)
    raw = []
    read = source._load_raw_frame

    def observe(*args, **kwargs):
        frame = read(*args, **kwargs)
        raw.append((len(frame), bool(frame["itemid"].eq(220045).all())))
        return frame

    monkeypatch.setattr(source, "_load_raw_frame", observe)
    frame = _public_read(source)
    # Public load_table also filters afterwards; inspect the real raw result
    # so that post-filtering cannot mask a lost storage predicate.
    selected_count = len(frame)
    assert raw == [(10, True)] and selected_count == 10


@pytest.mark.parametrize("suffix", [".csv", ".csv.gz"])
def test_public_reader_keeps_numeric_looking_value_as_text(tmp_path, suffix):
    frame = _public_read(_source(tmp_path, suffix))
    all_text = bool(frame["value"].map(lambda value: isinstance(value, str)).all())
    assert all_text
    assert set(frame["value"]) == {"60", "61", "62", "63", "64"}


@pytest.mark.parametrize("suffix", [".csv", ".csv.gz"])
def test_public_reader_refuses_bad_unselected_declared_column(tmp_path, suffix):
    source = _source(tmp_path, suffix, bad_unselected=True)
    with pytest.raises(TableScanError) as error:
        _public_read(source)
    assert error.value.reason == "csv_rows_rejected"
    assert "full-schema validation scan" in str(error.value)
