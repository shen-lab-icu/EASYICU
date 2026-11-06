"""A mixed text/number GCS column aggregates the same whatever leads the batch.

eICU tgcs ('GCS Total', 'Score (Glasgow Coma Scale)') carries text such as
'Unable to score due to medication' beside numbers; AUMC vgcs carries 6735
'Geïntubeerd'. A batch whose first rows were text kept the column as text and
took an hour's first row; a batch led by numbers took the median (lead-r13
sweep: eICU tgcs and AUMC vgcs, batch 50 vs default). The declared value kind
now decides. Every record here is synthetic.
"""

import pandas as pd
import pytest

from easyicu.concept import ConceptDictionary, ConceptResolver
from easyicu.config import load_src_cfg
from easyicu.datasource import ICUDataSource
from tests.support.blood_gas_sources import DICTIONARY, _source, _write

TARGET = 9
PEERS = [1, 2, 3]
MINUTE = 60_000
UNSCORABLE = "Unable to score due to medication"


def _load(source, concept, id_column, batch):
    frame = ConceptResolver(ConceptDictionary.from_json(DICTIONARY)).load_concepts(
        [concept], source, merge=True, interval=pd.Timedelta(hours=1),
        patient_ids={id_column: batch}, concept_workers=1, verbose=False,
    )
    rows = frame.loc[frame[id_column].eq(TARGET), ["charttime", concept]]
    return rows.sort_values("charttime").reset_index(drop=True)


def _eicu(tmp_path, peer_value):
    stays = [*PEERS, TARGET]
    _write(pd.DataFrame({
        "patientunitstayid": stays, "patienthealthsystemstayid": [10 + s for s in stays],
        "uniquepid": [f"p{s}" for s in stays], "age": ["60"] * 4, "gender": ["Male"] * 4,
        "hospitalid": [1] * 4, "hospitaladmitoffset": [0] * 4, "hospitaldischargeoffset": [72 * 60] * 4,
        "unitdischargeoffset": [48 * 60] * 4, "unitdischargestatus": ["Alive"] * 4,
        "hospitaldischargestatus": ["Alive"] * 4, "admissionweight": [80.0] * 4,
    }), tmp_path / "patient.parquet")
    # The target stay: GCS total 13 and 14 in hour 2, and one unscorable hour 3.
    rows = [(TARGET, 125, "13"), (TARGET, 130, "14"), (TARGET, 185, UNSCORABLE)]
    # Peers lead the batch with 40 hourly rows each, all text or all numbers.
    rows += [(stay, minute, peer_value) for stay in PEERS for minute in range(5, 40 * 60, 60)]
    chart = pd.DataFrame(rows, columns=["patientunitstayid", "nursingchartoffset", "nursingchartvalue"])
    chart["nursingchartentryoffset"] = chart["nursingchartoffset"]
    chart["nursingchartcelltypecat"] = "Scores"
    chart["nursingchartcelltypevallabel"] = "Glasgow coma score"
    chart["nursingchartcelltypevalname"] = "GCS Total"
    chart["nursingchartid"] = range(len(chart))
    _write(chart, tmp_path / "nursecharting" / "part.parquet")
    return ICUDataSource(load_src_cfg("eicu"), base_path=tmp_path, enable_cache=False)


def _aumc(tmp_path, peer_value):
    stays = [*PEERS, TARGET]
    rows = [(TARGET, 13066, "Verwarde taal", 125), (TARGET, 16640, "Georiënteerd", 130)]
    peer_item = 6735 if peer_value == "Geïntubeerd" else 13066
    rows += [(stay, peer_item, peer_value, minute) for stay in PEERS for minute in range(5, 40 * 60, 60)]
    items = pd.DataFrame(rows, columns=["admissionid", "itemid", "value", "minute"])
    items["measuredat"] = items.pop("minute") * MINUTE
    items["item"] = "verbal"
    items["valueid"] = 1
    items["registeredat"] = items["measuredat"]
    _write(items, tmp_path / "listitems.parquet")
    _write(pd.DataFrame({
        "admissionid": stays, "patientid": [100 + s for s in stays], "admittedat": [0] * 4,
        "dischargedat": [72 * 60 * MINUTE] * 4, "admissioncount": [1] * 4, "location": ["IC"] * 4,
        "dateofdeath": pd.array([None] * 4, dtype="Int64"),
    }), tmp_path / "admissions.parquet")
    return _source(tmp_path, "aumc")


CASES = {
    "eicu_tgcs": (_eicu, "tgcs", "patientunitstayid", UNSCORABLE, "15", [13.5]),
    "aumc_vgcs": (_aumc, "vgcs", "admissionid", "Geïntubeerd", "Verwarde taal", [4.5]),
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_hour_is_the_same_alone_and_behind_text_or_numeric_peers(tmp_path, case):
    build, concept, id_column, text_peer, number_peer, expected = CASES[case]
    results = {}
    for lead, peer_value in (("text", text_peer), ("number", number_peer)):
        source = build(tmp_path / lead, peer_value)
        results[f"{lead}_alone"] = _load(source, concept, id_column, [TARGET])
        results[f"{lead}_batch"] = _load(source, concept, id_column, [*PEERS, TARGET])

    reference = results["number_alone"]
    assert reference[concept].tolist() == expected
    for name, frame in results.items():
        pd.testing.assert_frame_equal(frame, reference, obj=name)
