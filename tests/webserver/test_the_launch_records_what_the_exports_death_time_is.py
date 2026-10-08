"""A launch records what the bound export's death time is, as its producer labels it.

The native export labels the death time it issues beside the death status by how
its source records it (``easyicu.utils.death_time_semantics``), in the outcome
file's time-axis audit.  Nothing read the label, so a prediction's risk set could
not tell a death time recorded to the hour from a date or a proxy.  The launch
now reads it once, from the manifest it binds, into
``data_constraints.event_time_semantics``; planning reads only that record.  An
export that labels none is recorded as labelling none.  Fixtures are generic.
"""

from __future__ import annotations

import json
from pathlib import Path

from easyicu.webserver.agent_pipeline_runs import _research_user_preferences
from easyicu.webserver.research_launch_scientific import (
    bound_export_event_time_semantics,
)
from tests.webserver.copilot.research_workflow_fixtures import complete_study


def _export(path: Path, *audits: dict | None) -> str:
    path.mkdir(parents=True, exist_ok=True)
    files = [
        {
            "file": f"module_{index}.parquet",
            "module": "outcome" if audit else "vitals",
            **({"time_axis_audit": audit} if audit is not None else {}),
        }
        for index, audit in enumerate(audits)
    ]
    (path / "_manifest.json").write_text(json.dumps({"files": files}), encoding="utf-8")
    return str(path)


def _death(label: str) -> dict:
    return {
        "policy": "stay_level_at_icu_admission_with_event_time_companions",
        "event_time_companion": "death_time",
        "event_time_semantics": label,
        "event_rows": 3,
        "timed_event_rows": 2,
    }


def test_the_launch_reads_the_label_the_export_writes(tmp_path: Path) -> None:
    export = _export(
        tmp_path / "miiv",
        {"policy": "longitudinal"},  # a module's audit without an event time
        _death("recorded_deathtime"),
    )

    assert bound_export_event_time_semantics(export) == {
        "death_time": "recorded_deathtime"
    }


def test_an_export_labelling_none_or_two_ways_records_no_label(tmp_path: Path) -> None:
    unlabelled = _export(tmp_path / "old", None, {"policy": "longitudinal"})
    disagreeing = _export(
        tmp_path / "two",
        _death("recorded_deathtime"),
        _death("structurally_unavailable"),
    )

    assert bound_export_event_time_semantics(unlabelled) == {}
    assert bound_export_event_time_semantics(disagreeing) == {}
    assert bound_export_event_time_semantics(str(tmp_path / "missing")) == {}
    assert bound_export_event_time_semantics(None) == {}


def test_planning_reads_the_record_the_launch_writes() -> None:
    study = complete_study()

    stated = json.loads(
        _research_user_preferences(
            study,
            event_time_semantics={
                "death_time": "recorded_dateofdeath_for_72h_post_icu_discharge_death_proxy"
            },
        )["data_constraints"]
    )
    labelling_none = json.loads(
        _research_user_preferences(study, event_time_semantics={})["data_constraints"]
    )

    assert stated["event_time_semantics"] == {
        "death_time": "recorded_dateofdeath_for_72h_post_icu_discharge_death_proxy"
    }
    # Recorded as labelling nothing: a context older than the record has no key.
    assert labelling_none["event_time_semantics"] == {}
    assert "event_time_semantics" not in json.loads(
        _research_user_preferences(study)["data_constraints"]
    )


def test_the_launch_passes_the_bound_exports_label_to_planning() -> None:
    import inspect

    from easyicu.webserver import agent_pipeline_runs

    source = inspect.getsource(agent_pipeline_runs)
    call = source.index("preferences = _research_user_preferences(")
    assert (
        "event_time_semantics=bound_export_event_time_semantics(export_path)"
        in source[
            call : source.index(")", source.index("event_time_semantics=", call)) + 1
        ]
    )
