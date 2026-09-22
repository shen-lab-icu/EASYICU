"""A folder scan names every packaged database, demo exports included."""

from __future__ import annotations

from easyicu.databases.profiles import iter_database_profiles
from easyicu.webserver import dataio


def test_every_packaged_database_reads_as_a_named_database() -> None:
    # The scan reports Web aliases (mimic -> miii, sic -> sicdb) and, for a
    # demo export, the key its manifest declares (eicu_demo).
    aliases = {"mimic": "miii", "sic": "sicdb"}
    for profile in iter_database_profiles():
        assert dataio._database_label(aliases.get(profile.key, profile.key)) != "Unknown", profile.key


def test_demo_exports_read_as_their_profile_name() -> None:
    assert dataio._database_label("eicu_demo") == "eICU Demo"
    assert dataio._database_label("mimic_demo") == "MIMIC-III Demo"
    # The surface labels of the full databases are unchanged.
    assert dataio._database_label("eicu") == "eICU-CRD"
    assert dataio._database_label("miiv") == "MIMIC-IV"
    assert dataio._database_label("not-a-database") == "Unknown"
