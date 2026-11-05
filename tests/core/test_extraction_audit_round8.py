"""Round-8 extraction audit (2026-11-05): lead items on 1510135f0."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from easyicu.concept.eicu_source_qc import arterial_gases


class _Source:
    """Only the two optional eICU channels the specimen selector reads."""

    def __init__(self, customlab, lab):
        self.tables = {"customlab": customlab, "lab": lab}

    def load_table(self, table, columns=None, filters=(), verbose=False):
        frame = self.tables[table]
        for spec in filters:
            frame = frame.loc[frame[spec.column].isin(spec.value)]
        return SimpleNamespace(data=frame if columns is None else frame[list(columns)])


@pytest.mark.parametrize("analyte", ["paO2", "paCO2", "pH", "O2 Sat (%)"])
@pytest.mark.parametrize("text", ["Unknown", "Unspecified", "", "Other"])
def test_r8_f018_recorded_unknown_sample_is_excluded_before_fallbacks(analyte, text):
    # R7-F018: an explicit Unknown sample was treated as "no sample record" and
    # admitted by the label (paO2/paCO2) or same-panel (pH, SaO2) fallback.
    # Offset 0 records an unknown sample, offset 60 records none, offset 120 is arterial.
    frame = pd.DataFrame({"patientunitstayid": 1, "labresultoffset": [0, 60, 120],
                          "labname": analyte, "labresult": [7.3, 7.31, 7.32]})
    customlab = pd.DataFrame({"patientunitstayid": 1, "labotheroffset": [0, 120],
                              "labothername": "Sample Type",
                              "labothervaluetext": [text, "Arterial"], "labotherresult": ""})
    lab = pd.DataFrame({"patientunitstayid": 1, "labresultoffset": [0, 60, 120],
                        "labname": "paO2", "labresult": [80., 80., 80.]})
    out = arterial_gases(frame, _Source(customlab, lab), analytes=(analyte.casefold(),))
    fallback = "label_defined_arterial" if analyte in {"paO2", "paCO2"} else "co_measured_arterial"
    assert dict(zip(out.labresultoffset, out.specimen_status)) == {60: fallback, 120: "explicit_arterial"}
    assert out.attrs["eicu_specimen_audit"]["excluded_recorded_unknown"] == 1
    assert out.attrs["eicu_specimen_audit"]["excluded_unknown"] == 0


@pytest.mark.parametrize("gap_seconds, breaks", [(12 * 3600, False), (12 * 3600 + 1, True)])
def test_r8_exact_twelve_hour_record_gap_does_not_break_on_float_noise(gap_seconds, breaks):
    # MIMIC-III/MIIV urine ledgers: a gap of exactly 12 h computed from float
    # hours can be 12.000000000000002 and was treated as "> 12 h" (break).
    from easyicu.scores.urine_output import collection_intervals

    first = 7 * 3600 + 13  # (7h00m13s, 19h00m13s) differ by 12.000000000000002 h
    ends = np.array([first, first + 1800, first + 1800 + gap_seconds]) / 3600
    assert bool(ends[2] - ends[1] > 12) is (gap_seconds >= 12 * 3600)
    _, _, _, valid, segment = collection_intervals(ends, np.array([50., 50., 50.]),
                                                   record_gaps=np.r_[0., np.diff(ends)])
    assert bool(valid[2]) is not breaks
    assert bool(segment[2] == 2) is breaks
