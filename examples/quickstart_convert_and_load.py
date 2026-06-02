"""Quickstart: convert a raw ICU database, then extract standardized concepts.

This is the shortest correct path for a new **Python API** user. It exists
because the #1 onboarding mistake is calling ``load_concepts(...)`` on a raw
download. Every extraction API in EasyICU expects a **prepared (converted)**
directory — so the flow is always:

    raw download  ──DataConverter──▶  prepared dir  ──load_concepts──▶  features

Run it::

    # 1. install
    pip install -e ".[all]"

    # 2. point these at your own raw database directory and edit IDs, then:
    python examples/quickstart_convert_and_load.py

Notes
-----
* ``database`` is one of: ``miiv`` (MIMIC-IV), ``mimic`` (MIMIC-III),
  ``eicu``, ``aumc`` (AmsterdamUMCdb), ``hirid``, ``sic`` (SICdb).
* Conversion only needs to happen **once** per database. On reruns you can
  skip Step 1 and pass the already-prepared directory straight to ``data_path``.
* ``data_path`` always means the *prepared* directory (which, for the
  in-place converter, is the same directory you converted).
"""

from __future__ import annotations

from pathlib import Path

from easyicu import load_concepts
from easyicu.data_converter import DataConverter

# --- Edit these three for your environment -------------------------------
DATABASE = "miiv"
RAW_DATA_PATH = Path("/path/to/mimic-iv-raw")  # your original download
PATIENT_IDS = [30000123, 30000456]  # set to None to load the whole cohort
# -------------------------------------------------------------------------


def main() -> None:
    # Step 1 — convert raw CSV / CSV.GZ / tar.gz into the prepared Parquet
    # layout. Safe to skip on reruns once the directory is already prepared.
    converter = DataConverter(str(RAW_DATA_PATH), database=DATABASE)
    converter.convert_all()

    # Step 2 — extract standardized concepts from the *prepared* directory.
    # `data_path` is the converted directory, never the raw download.
    vitals = load_concepts(
        concepts=["hr", "map", "resp", "spo2"],
        database=DATABASE,
        data_path=str(RAW_DATA_PATH),
        patient_ids=PATIENT_IDS,
        interval="1h",
        aggregate="mean",
        verbose=True,
    )

    print(vitals.head())

    # Step 3 (optional) — persist the feature table for downstream analysis.
    out = Path(f"{DATABASE}_vitals_1h.parquet")
    vitals.to_parquet(out, index=False)
    print(f"wrote {out.resolve()}")


if __name__ == "__main__":
    main()
