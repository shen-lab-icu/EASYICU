"""eICU RASS: EasyICU's declared extension beyond ricu's mapping is pinned.

ricu 0.6.3 maps eICU ``rass`` to nursecharting valname 'Sedation Score' only.
EasyICU keeps that source (restricted to rows paired with 'Sedation Scale' =
'RASS') and adds two 'Other Vital Signs and Infusions' labels, 'RASS' and
'SEDATION SCORE' (valname 'Value'), whose values are all integers in -5..+4
(2026-11-03 extraction audit 2). Through eICU ett_gcs (RASS <= -3) these labels
open sedation windows ricu does not; on the eICU demo they account for 170 of
the 176 GCS cells that still differ from ricu. The supervisor kept them as a
declared deviation (r13gcs); any change to this set must be deliberate.
"""

from __future__ import annotations

import json

from tests.support.eicu_source import DICTIONARY

RICU_VALNAMES = {"Sedation Score"}
EXTENSION_LABELS = {"RASS", "SEDATION SCORE"}


def _eicu_rass_sources():
    return json.loads(DICTIONARY.read_text())["rass"]["sources"]["eicu"]


def test_eicu_rass_sources_are_ricus_valname_plus_two_declared_labels():
    valnames, labels, other = set(), set(), []
    for source in _eicu_rass_sources():
        assert source["table"] == "nursecharting"
        ids = source["ids"] if isinstance(source["ids"], list) else [source["ids"]]
        if source["sub_var"] == "nursingchartcelltypevalname":
            valnames.update(ids)
        elif source["sub_var"] == "nursingchartcelltypevallabel":
            labels.update(ids)
        else:
            other.append(source["sub_var"])
    assert valnames == RICU_VALNAMES
    assert labels == EXTENSION_LABELS
    assert other == []


def test_the_ricu_valname_keeps_its_rass_scale_restriction():
    by_valname = [s for s in _eicu_rass_sources() if s["sub_var"] == "nursingchartcelltypevalname"]
    assert [s.get("callback") for s in by_valname] == ["eicu_rass_scale_only"]
