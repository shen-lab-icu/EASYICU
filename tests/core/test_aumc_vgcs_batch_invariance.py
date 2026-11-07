"""AUMC GCS verbal does not depend on which stays share a patient batch.

vgcs maps list items 6735, 13066 and 16640. Item 6735 also records
'Geïntubeerd', which no verbal score maps, so the concatenated vgcs column is
text-and-number. The resolver used to coerce it to numbers only when most of
the batch's first 100 values were numeric. A batch led by intubated stays kept
the column as text, and the hour of another stay with two verbal answers
became the first answer instead of their median (lead-r13 sweep: AUMC
neurological vgcs, batch 50 vs default). Every record here is synthetic.
"""

import pandas as pd
import pytest

from easyicu.concept import ConceptDictionary, ConceptResolver
from tests.support.blood_gas_sources import DICTIONARY, _source, _write

MINUTE = 60_000
STAYS = [1, 2, 3, 4]


@pytest.fixture
def aumc(tmp_path):
    # Stay 1 answers verbal 4 (13066) and 5 (16640) in hour 2.
    rows = [(1, 13066, "Verwarde taal", 125), (1, 16640, "Georiënteerd", 130)]
    # Stays 2-4 are intubated for 60 hours: 6735 'Geïntubeerd' every hour.
    rows += [(stay, 6735, "Geïntubeerd", minute) for stay in (2, 3, 4) for minute in range(5, 3600, 60)]
    items = pd.DataFrame(rows, columns=["admissionid", "itemid", "value", "minute"])
    items["measuredat"] = items.pop("minute") * MINUTE
    items["item"] = "verbal"
    items["valueid"] = 1
    items["registeredat"] = items["measuredat"]
    _write(items, tmp_path / "listitems.parquet")
    _write(
        pd.DataFrame({
            "admissionid": STAYS, "patientid": [11, 12, 13, 14], "admittedat": [0] * 4,
            "dischargedat": [72 * 60 * MINUTE] * 4, "admissioncount": [1] * 4, "location": ["IC"] * 4,
            "dateofdeath": pd.array([None] * 4, dtype="Int64"),
        }),
        tmp_path / "admissions.parquet",
    )
    return _source(tmp_path, "aumc")


def _stay_one(source, batch):
    frame = ConceptResolver(ConceptDictionary.from_json(DICTIONARY)).load_concepts(
        ["vgcs"], source, merge=True, interval=pd.Timedelta(hours=1),
        patient_ids={"admissionid": batch}, concept_workers=1, verbose=False,
    )
    rows = frame.loc[frame["admissionid"].eq(1), ["charttime", "vgcs"]]
    return rows.sort_values("charttime").reset_index(drop=True)


def test_vgcs_hour_is_the_same_alone_and_with_intubated_batch_peers(aumc):
    alone = _stay_one(aumc, [1])
    together = _stay_one(aumc, STAYS)

    pd.testing.assert_frame_equal(alone, together)
    assert alone["vgcs"].tolist() == [4.5]
