"""SICdb d_references rows that name discharge and follow-up types.

The labels and codes are those of the two releases (1.0.6 and 1.0.8
d_references, ReferenceName DischargeType / DischargeState /
EstimatedSurvivalObservationTime). 1.0.8 keeps the 1.0.6 rows and adds
seven German discharge types.
"""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

_COMMON = [
    (2026, "Survived", "DischargeType"),
    (2028, "Deceased", "DischargeType"),
    (2202, "lebend", "DischargeState"),
    (2212, "Unknown", "DischargeState"),
    (2215, "verstorben", "DischargeState"),
    (3076, "6 Months", "EstimatedSurvivalObservationTime"),
    (3077, "1 Year", "EstimatedSurvivalObservationTime"),
]
_ADDED_108 = [
    (3129, "Entlassung aus dem Krankenhaus", "DischargeType"),
    (3130, "Sterbefall", "DischargeType"),
    (3131, "Transfer in ein anderes Krankenhaus", "DischargeType"),
    (3132, "Entlassung gegen Revers", "DischargeType"),
    (3133, "Abschluss eines Aufenthaltes im halbstationären Bereich oder krankenhausinterne "
           "Verlegung in den halbstationären Bereich", "DischargeType"),
    # Truncated in the published table; kept as published.
    (3134, "Krankenhausinterne Verlegung vom Bereich der Rehabilitation und vom "
           "ausschließlichen Bereich der Pflege in den allgemeinen stati", "DischargeType"),
]


def sic_references(release: str = "1.0.6") -> pd.DataFrame:
    rows = _COMMON + (_ADDED_108 if release == "1.0.8" else [])
    return pd.DataFrame(rows, columns=["ReferenceGlobalID", "ReferenceValue", "ReferenceName"])


def sic_raw_tables(cases: pd.DataFrame, release: str = "1.0.6"):
    """A stand-in for ``outcomes._raw_table`` serving ``cases`` and d_references."""
    def read(_database, _path, table="cases"):
        return sic_references(release) if table == "d_references" else cases
    return read


def with_sic_references(load_table, release: str = "1.0.6"):
    """Wrap a ``load_table`` so that d_references is served from the fixture."""
    def load(table_name, *args, **kwargs):
        if table_name == "d_references":
            return SimpleNamespace(data=sic_references(release))
        return load_table(table_name, *args, **kwargs)
    return load


def write_sic_references(root, release: str = "1.0.6") -> None:
    sic_references(release).to_parquet(f"{root}/d_references.parquet", index=False)


class SicReferenceSource:
    """A data source that holds only d_references (every other table is absent)."""

    def __init__(self, release: str = "1.0.6") -> None:
        self.release = release

    def load_table(self, table_name, columns=None, filters=None, verbose=False):
        if table_name != "d_references":
            raise KeyError(table_name)
        return SimpleNamespace(data=sic_references(self.release))
