"""sep3_sofa2 reads unscored SOFA-2 grid hours as the published total does.

The SOFA-2 concept leaves a gap-filled hour without any owner row NA; the
published ``sofa2_score`` total normal-imputes it to 0 (ricu ``sofa_score``:
fill_gaps, slide, rowSums(na.rm = TRUE)).  The streamed export derives
sep3_sofa2 from the published total, the one-shot path from the concept, so
both must read such an hour as 0 or the onset depends on the execution path
(lead-r13: MIIV onsets 0 vs 17 and 10 vs 15 hours between 14 and 64 GiB).
"""

from types import SimpleNamespace

import pandas as pd
import pytest

from easyicu.concept.callbacks import ConceptCallbackContext, _callback_sep3_sofa2
from easyicu.table import ICUTable


def _context(database):
    data_source = SimpleNamespace(config=SimpleNamespace(name=database)) if database else None
    return ConceptCallbackContext(
        concept_name="sep3_sofa2",
        target=None,
        interval=None,
        resolver=None,
        data_source=data_source,
        patient_ids=None,
        kwargs={},
    )


def _tables(sofa2_values):
    hours = [-20.0, *range(-19, 0), *range(0, 20)]
    sofa2 = pd.DataFrame(
        {
            "stay_id": 1,
            "charttime": [float(h) for h in hours],
            # Hour -20 owns a zero score; -19..-1 are gap-filled and unscored;
            # the score is 3 from admission and 8 from hour 17.
            "sofa2": [0.0, *sofa2_values, *([3.0] * 17), *([8.0] * 3)],
        }
    )
    susp = pd.DataFrame({"stay_id": [1], "charttime": [42.0], "susp_inf": [True]})
    return {
        "sofa2": ICUTable(sofa2, id_columns=["stay_id"], index_column="charttime", value_column="sofa2"),
        "susp_inf": ICUTable(susp, id_columns=["stay_id"], index_column="charttime", value_column="susp_inf"),
    }


def _onset(tables, database="miiv"):
    result = _callback_sep3_sofa2(tables, _context(database)).data
    return result.loc[result["sep3_sofa2"].astype(bool), "charttime"].tolist()


@pytest.mark.parametrize("database", ["miiv", "eicu", "aumc", "hirid", "mimic", None])
def test_unscored_grid_hours_read_as_the_published_zero(database):
    unscored = _onset(_tables([float("nan")] * 19), database)
    published = _onset(_tables([0.0] * 19), database)
    # The 48 h look-back from SI at hour 42 starts at hour -6: the zero hours
    # before admission make admission (3 points) the first >= 2 rise.
    assert published == [0.0]
    assert unscored == published


def test_database_without_structural_sofa2_total_keeps_it_unscored():
    tables = _tables([float("nan")] * 19)
    tables["sofa2"].data["sofa2"] = float("nan")
    assert _onset(tables, "sic") == []
