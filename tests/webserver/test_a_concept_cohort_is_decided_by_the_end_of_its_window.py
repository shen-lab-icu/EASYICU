"""A concept-derived cohort is decided by the end of the window it states.

Data Extraction admits a stay to a concept-derived population (Sepsis-3, AKI,
ventilation, vasopressor, respiratory support) on a positive concept row timed
at or before the end of the cohort's observation window, in hours after ICU
admission.  The window decides who enters and is never a scoring window: SOFA
and SOFA-2 keep their own.  The export manifest records the rule, and a
registered export made under the earlier rule is reused only where its rows
cannot differ.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import pytest

from easyicu.webserver import dataio, primary_cohort
from easyicu.webserver.pi_copilot.extraction_handoff import (
    compile_registered_export_handoff,
)

# preset -> (the concepts its matcher loads, one positive column)
_PRESETS = {
    "sepsis3": (["sep3_sofa2"], "sep3_sofa2"),
    "aki": (["aki"], "aki"),
    "ventilation": (["mech_vent", "vent_ind"], "mech_vent"),
    "vasopressor": (["vaso_ind"], "vaso_ind"),
    "respiratory": (["adv_resp", "mech_vent", "vent_ind", "pafi", "safi"], "adv_resp"),
}


class _Api:
    """The loader surface the matcher reads, returning one fixed payload."""

    def __init__(self, payload: Any) -> None:
        self.payload = payload
        self.calls: list[dict[str, Any]] = []

    def load_concepts(self, concepts: list[str], **kwargs: Any) -> Any:
        self.calls.append({"concepts": list(concepts), "kwargs": kwargs})
        return self.payload


def _match(
    payload: Any,
    preset: str = "sepsis3",
    window_hours: int = 24,
    sepsis_kwargs: Optional[dict[str, Any]] = None,
) -> tuple[set[Any], _Api]:
    api = _Api(payload)
    ids = dataio._match_concept_derived_cohort_ids(
        api, "/unused", "miiv", "stay_id", {1, 2, 3, 4, 5}, preset, window_hours, sepsis_kwargs
    )
    return ids, api


@pytest.mark.parametrize("preset", sorted(_PRESETS))
def test_a_stay_enters_only_on_a_positive_row_by_the_end_of_its_window(preset: str) -> None:
    concepts, positive = _PRESETS[preset]
    rows = pd.DataFrame(
        {
            "stay_id": [1, 2, 3, 4, 4, 5, 5],
            "charttime": [30.0, 24.0, -6.0, 3.0, 40.0, 2.0, 50.0],
            positive: [1, 1, 1, 0, 0, 1, 1],
        }
    )

    ids, api = _match(rows, preset)

    # 1 turns positive only after hour 24; 2 at hour 24 itself; 3 before
    # admission, as far as the loader reads; 4 never; 5 within and after.
    assert ids == {2, 3, 5}
    assert api.calls[0]["concepts"] == concepts


def test_threshold_and_rate_signals_count_only_within_the_window() -> None:
    respiratory = pd.DataFrame(
        {"stay_id": [1, 2, 3], "charttime": [40.0, 3.0, 5.0], "pafi": [250.0, 280.0, 420.0]}
    )
    vasopressor = pd.DataFrame(
        {"stay_id": [1, 2], "charttime": [30.0, 12.0], "norepi_rate": [0.1, 0.05]}
    )

    assert _match(respiratory, "respiratory")[0] == {2}
    assert _match(vasopressor, "vasopressor")[0] == {2}


@pytest.mark.parametrize("time_column", ["datetime", "observationoffset"])
def test_the_kdigo_bundle_time_column_bounds_the_window(time_column: str) -> None:
    rows = pd.DataFrame({"stay_id": [1, 2], time_column: [30.0, 6.0], "aki": [True, True]})

    assert _match(rows, "aki")[0] == {2}


def test_a_timedelta_row_time_is_read_in_hours_after_admission() -> None:
    rows = pd.DataFrame(
        {
            "stay_id": [1, 2],
            "charttime": pd.to_timedelta([25, 23], unit="h"),
            "sep3_sofa2": [True, True],
        }
    )

    assert _match(rows)[0] == {2}


@pytest.mark.parametrize("row_time", ["missing", "absolute"])
def test_a_payload_without_admission_hours_fails_closed(row_time: str) -> None:
    rows = pd.DataFrame({"stay_id": [1], "sep3_sofa2": [True]})
    if row_time == "absolute":
        rows["charttime"] = pd.to_datetime(["2150-01-01 06:00"])

    with pytest.raises(dataio.ExportCohortError) as excinfo:
        _match(rows)

    assert excinfo.value.error == "concept_cohort_row_time_unavailable"
    assert excinfo.value.detail["concepts"] == ["sep3_sofa2"]


def test_the_window_is_not_passed_as_a_scoring_window() -> None:
    rows = pd.DataFrame({"stay_id": [1], "charttime": [1.0], "sep3_sofa2": [True]})

    _, api = _match(rows, "sepsis3", 72, {"si_lwr": "48h"})

    assert "win_length" not in api.calls[0]["kwargs"]
    assert api.calls[0]["kwargs"]["si_lwr"] == "48h"


class _Job:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def emit(self, payload: dict[str, Any]) -> None:
        self.events.append(payload)


def _patch_extraction(
    monkeypatch: pytest.MonkeyPatch, loaded: list[dict[str, Any]], matcher_payload: Any
) -> None:
    import easyicu.api as api_module
    import easyicu.patient_filter as patient_filter_module
    from easyicu.resources import load_dictionary

    dictionary = load_dictionary(include_sofa2=True)

    class FakePatientFilter:
        def __init__(self, database: str, data_path: str, verbose: bool = False) -> None:
            pass

        def filter(self, **kwargs: Any) -> pd.DataFrame:
            return pd.DataFrame({"patient_id": [1, 2, 3]})

    @contextmanager
    def fake_keep_cache(**_: Any):
        yield None

    def fake_load_concepts(concepts: list[str], **kwargs: Any) -> pd.DataFrame:
        loaded.append({"concepts": list(concepts), "kwargs": kwargs})
        if len(loaded) == 1:
            return matcher_payload
        ids = (kwargs.get("patient_ids") or {}).get("stay_id", [])
        payload: dict[str, Any] = {"stay_id": ids}
        for concept in concepts:
            definition = dictionary.get(concept)
            if definition is not None and definition.class_name == "lgl_cncpt":
                payload[concept] = [index % 2 for index in range(len(ids))]
            else:
                payload[concept] = [65.0] * len(ids)
        return pd.DataFrame(payload)

    monkeypatch.setattr(patient_filter_module, "PatientFilter", FakePatientFilter)
    monkeypatch.setattr(api_module, "keep_cache", fake_keep_cache)
    monkeypatch.setattr(api_module, "load_concepts", fake_load_concepts)


def test_an_export_records_its_cohort_rule_and_keeps_the_scores_own_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    loaded: list[dict[str, Any]] = []
    matcher = pd.DataFrame(
        {"stay_id": [1, 2, 3], "charttime": [80.0, 72.0, 5.0], "sep3_sofa2": [True, True, False]}
    )
    _patch_extraction(monkeypatch, loaded, matcher)
    runner = dataio.make_export_runner(
        data_path=str(tmp_path),
        database="miiv",
        modules=["demographics"],
        export_format="csv",
        out_dir=str(tmp_path / "out"),
        cohort={"preset": "sepsis3", "observation_window_hours": 72},
    )

    runner(_Job())
    manifest = json.loads((tmp_path / "out" / "_manifest.json").read_text(encoding="utf-8"))
    readme = (tmp_path / "out" / "README.md").read_text(encoding="utf-8")

    assert all("win_length" not in call["kwargs"] for call in loaded)
    assert loaded[1]["kwargs"]["patient_ids"] == {"stay_id": [2]}
    assert manifest["cohort_execution"] == {
        "schema_version": dataio.EXPORT_COHORT_EXECUTION_SCHEMA,
        "concept_cohort_window": {
            "definition": "sepsis3",
            "positive_rows": primary_cohort.CONCEPT_POSITIVE_ROWS,
            "window_end_hours": 72,
        },
        "score_window_hours": 24,
    }
    assert "a stay enters on a positive `sepsis3` row at or before hour `72`" in readme
    assert "SOFA and SOFA-2 keep their own `24 h` worst-value window" in readme
    assert dataio.export_cohort_execution_current(manifest)


def test_a_cohort_without_a_concept_population_records_only_the_score_window() -> None:
    record = dataio.export_cohort_execution(
        {"preset": "adult_first", "observation_window_hours": 48}
    )

    assert record["concept_cohort_window"] is None
    assert record["score_window_hours"] == 24


def test_the_cohort_scope_states_the_positive_row_rule_for_a_concept_population() -> None:
    concept = primary_cohort.normalize_primary_cohort_scope(
        {"preset": "ventilation", "observation_window_hours": 24}
    )
    plain = primary_cohort.normalize_primary_cohort_scope(
        {"preset": "adult_first", "observation_window_hours": 24}
    )

    assert concept.to_dict()["phenotype_window"] == {
        "definition": "ventilation",
        "observation_window_hours": 24,
        "positive_rows": primary_cohort.CONCEPT_POSITIVE_ROWS,
    }
    assert plain.to_dict()["phenotype_window"] == {}


def _registered_handoff(
    tmp_path: Path, cohort: dict[str, Any], execution: Optional[dict[str, Any]] = None
) -> Any:
    raw = tmp_path / "raw"
    raw.mkdir()
    export = tmp_path / "export"
    export.mkdir()
    manifest: dict[str, Any] = {
        "schema_version": "easyicu_native_export_v2",
        "database": "miiv",
        "data_path": str(raw),
        "format": "csv",
        "cohort_contract": dataio.normalize_export_cohort_contract(cohort),
        "files": [{"file": "demographics.csv", "module": "demographics"}],
    }
    if execution is not None:
        manifest["cohort_execution"] = execution
    (export / "_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    study = {
        "data_source": {"path": str(export), "database": "miiv"},
        "cohort": dict(cohort),
        "modules": ["demographics"],
        "export_format": "csv",
    }
    return compile_registered_export_handoff(
        study, {"id": "src", "path": str(export), "database": "miiv", "ok": True}
    )


@pytest.mark.parametrize(
    ("cohort", "reusable"),
    [
        # A concept cohort chosen under the earlier rule.
        ({"preset": "sepsis3", "observation_window_hours": 24}, False),
        # Scores computed over a 720 h window.
        ({"preset": "adult_first", "observation_window_hours": 720}, False),
        # Neither change can reach these rows.
        ({"preset": "adult_first", "observation_window_hours": 24}, True),
    ],
)
def test_an_export_made_under_the_earlier_rule_is_reused_only_where_its_rows_agree(
    tmp_path: Path, cohort: dict[str, Any], reusable: bool
) -> None:
    handoff = _registered_handoff(tmp_path, cohort)

    assert handoff.reusable is reusable
    outdated = "registered_export_cohort_execution_outdated" in handoff.mismatch_codes
    assert outdated is not reusable


def test_an_export_that_records_the_current_rule_is_reused(tmp_path: Path) -> None:
    cohort = {"preset": "sepsis3", "observation_window_hours": 24}

    handoff = _registered_handoff(tmp_path, cohort, dataio.export_cohort_execution(cohort))

    assert handoff.reusable is True
    assert handoff.mismatch_codes == ()
