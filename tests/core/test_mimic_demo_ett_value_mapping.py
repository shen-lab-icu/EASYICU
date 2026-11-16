"""The MIMIC-III demo reads the full MIMIC-III value->itemid mapping (ricu demo parity, 2026-11-16).

The demo entry once listed only MetaVision 223900, so CareVue verbal "1.0 ET/Trach"
rows (itemid 723) never reached ett_gcs on the demo: every differing GCS cell of
the demo parity traced to it.
"""

from easyicu.datasource import VALUE_TO_ITEMID_MAPPING


def test_the_demo_uses_the_full_mimic_iii_mapping():
    assert VALUE_TO_ITEMID_MAPPING["mimic_demo"] == VALUE_TO_ITEMID_MAPPING["mimic"]
    values = VALUE_TO_ITEMID_MAPPING["mimic_demo"]["chartevents"]["value"]
    assert 723 in values["1.0 ET/Trach"] and 723 in values["No Response-ETT"]
