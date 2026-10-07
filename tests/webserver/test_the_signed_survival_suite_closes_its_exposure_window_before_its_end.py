"""The signed survival suite closes its exposure window before its end.

The suite counted an exposure first recorded after the prevalence cutoff and
at or before the exposure window's end as incident.  The window ends by the
landmark, and every other landmark window is closed at its start and open at
its end: hourly data name each hour by its start, so a first record at the
window's end lies in the hour that begins there, after the window.  Such a
record counted the exposure in the exposed group.  It is now outside the
window, so the stay is excluded like any exposure first recorded after it,
in neither group; a record before the end stays incident.  The flow note and
the Methods say "at or after" and "before" the window's end.

Synthetic study and seeded synthetic rows only (renal replacement therapy and
90-day mortality).
"""

from __future__ import annotations

import json

import pandas as pd

from easyicu.research_agent.authority.manuscript_method_facts import _design_text
from easyicu.research_agent.contracts.executed_method_design import (
    EXECUTED_METHOD_DESIGN_KEY,
    validate_executed_method_design,
)
from easyicu.research_agent.contracts.manuscript_tables import (
    MANUSCRIPT_TABLES_KEY,
    validate_manuscript_table_declarations,
)
from tests.support.survival_sealed import (
    run_signed_suite,
    sealed_survival,
    synthetic_survival_rows,
)


def test_a_first_record_at_the_window_end_is_not_incident(tmp_path):
    _context, authority = sealed_survival(tmp_path)
    end = float(authority.exposure_window_hours[1])
    onset_column = authority.exposure_onset_column
    rows = synthetic_survival_rows()
    late = rows.index[rows["rrt"].eq(1) & rows[onset_column].eq(18.0)]
    at_end, before_end = late[: len(late) // 2], late[len(late) // 2 :]
    rows.loc[at_end, onset_column] = end
    rows.loc[before_end, onset_column] = end - 0.5

    summary = json.loads(
        json.dumps(run_signed_suite(authority, rows, tmp_path / "out"))
    )

    analysis = pd.read_parquet(tmp_path / "out" / summary["analysis_cohort_file"])
    onset = analysis[onset_column]
    exposed = analysis[authority.derived_exposure_column].eq(1)
    assert not onset.eq(end).any()
    assert len(at_end) and int(onset[exposed].eq(end - 0.5).sum()) == len(before_end)
    cutoff = float(authority.prevalent_exposure_cutoff_hours)
    excluded = rows["rrt"].eq(1) & (
        rows[onset_column].le(cutoff) | rows[onset_column].ge(end)
    )
    assert summary["n_landmark_population"] == len(rows) - int(excluded.sum())
    flow = next(
        table
        for table in validate_manuscript_table_declarations(
            summary[MANUSCRIPT_TABLES_KEY]
        )
        if table.product == authority.risk_set_product
    )
    assert flow.notes[0].endswith(
        f"at or before hour {cutoff:g} or at or after hour {end:g}."
    )
    design = validate_executed_method_design(summary[EXECUTED_METHOD_DESIGN_KEY])
    assert f"before hour {end:g} formed the exposed group;" in _design_text(design)
    assert f"by hour {end:g} formed" not in _design_text(design)
