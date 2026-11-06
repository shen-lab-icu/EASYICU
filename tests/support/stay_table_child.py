"""Child-process side of the stay-table run-binding tests (synthetic eICU rows)."""

from __future__ import annotations

from pathlib import Path


def load_under_binding(snapshot, root: str, names, stays, results) -> None:
    """Load ``names`` for ``stays`` in a unit opened from ``snapshot``; report the outcome."""
    from easyicu.concept.stay_tables import StayTableSourceChanged
    from easyicu.storage_binding import binding_unit
    from easyicu.config import load_src_cfg
    from easyicu.datasource import ICUDataSource
    from tests.support.urine_raw_source import _load

    source = ICUDataSource(load_src_cfg("eicu"), base_path=Path(root), enable_cache=False)
    try:
        with binding_unit(snapshot):
            _load(source, list(names), patient_ids={"patientunitstayid": list(stays)})
        results.put("ok")
    except StayTableSourceChanged as error:
        results.put(error.reason)
    except BaseException as error:  # noqa: BLE001 - reported to the parent
        results.put(type(error).__name__)


def load_twice_in_own_unit(root: str, names, stays, loaded, go, results, tag, drop=False) -> None:
    """One top-level unit: load, (drop its payload), signal, wait, load again; report."""
    from easyicu.concept.stay_tables import StayTableSourceChanged, drop_stay_table_frames
    from easyicu.storage_binding import binding_unit
    from easyicu.config import load_src_cfg
    from easyicu.datasource import ICUDataSource
    from tests.support.urine_raw_source import _load

    source = ICUDataSource(load_src_cfg("eicu"), base_path=Path(root), enable_cache=False)
    ids = {"patientunitstayid": list(stays)}
    try:
        with binding_unit():
            try:
                _load(source, list(names), patient_ids=ids)
                if drop:
                    drop_stay_table_frames()
            finally:
                loaded.set()
            go.wait(timeout=300)
            _load(source, list(names), patient_ids=ids)
        results.put((tag, "ok"))
    except StayTableSourceChanged as error:
        results.put((tag, error.reason))
    except BaseException as error:  # noqa: BLE001 - reported to the parent
        results.put((tag, type(error).__name__))
