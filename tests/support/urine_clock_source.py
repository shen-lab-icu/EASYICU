"""Physical urine sources with quarter-hour records in all six databases."""

import numpy as np
import pandas as pd


def urine_clock_source(root, database):
    """Tiny physical sources with urine records at quarter hours, never patient data."""
    if database in {"miiv", "mimic", "eicu"}:
        from tests.support.renal_export import write_renal_source

        write_renal_source(root, "miiv" if database == "mimic" else database, 2, hours=48)
        table = "intakeoutput" if database == "eicu" else "outputevents"
        path = root / f"{table}.parquet"
        frame = pd.read_parquet(path).iloc[:1]
        frame = pd.concat([frame] * 25, ignore_index=True)
        hours = np.arange(25.) + .25
        if database == "eicu":
            frame["intakeoutputoffset"] = hours * 60
            frame["intakeoutputentryoffset"] = hours * 60
            frame["cellvaluenumeric"] = 25.
        else:
            frame["charttime"] = pd.Timestamp("2200-01-01") + pd.to_timedelta(hours, unit="h")
            frame["value"] = 25.
        frame.to_parquet(path, index=False)
        if database != "eicu":
            for table in ("chartevents", "labevents"):
                path = root / f"{table}.parquet"
                frame = pd.read_parquet(path)
                # Raw text values are present in both MIMIC schemas.
                frame["value"] = frame.valuenum.astype(str)
                frame.to_parquet(path, index=False)
            path = root / "admissions.parquet"
            frame = pd.read_parquet(path)
            frame["deathtime"] = pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns]")
            frame.to_parquet(path, index=False)
            pd.DataFrame({"subject_id": [3, 4], "dod": pd.Timestamp("2201-01-01"),
                          "anchor_year": 2200, "anchor_year_group": "2017 - 2019",
                          "anchor_age": 65, "gender": "M", "dob": pd.Timestamp("2135-01-01")}).to_parquet(
                root / "patients.parquet", index=False,
            )
        else:
            path = root / "patient.parquet"
            frame = pd.read_parquet(path).assign(unitvisitnumber=1)
            frame.to_parquet(path, index=False)
        if database == "mimic":
            for path in root.glob("*.parquet"):
                frame = pd.read_parquet(path).rename(columns={"stay_id": "icustay_id"})
                frame.to_parquet(path, index=False)
    else:
        from easyicu.synthetic_sources import generate_source

        generate_source(root, database, stays=3)
        if database == "sic":
            # No native case labels; retain their physical source schema.
            pd.DataFrame({col: pd.Series(dtype="int64") for col in ("CaseID", "FieldID", "RefID")}).to_parquet(
                root / "data_ref.parquet", index=False,
            )
        table, item, code, clock, shift = {
            "hirid": ("observations", "variableid", 10020000, "datetime", pd.Timedelta(minutes=15)),
            "aumc": ("numericitems", "itemid", 8794, "measuredat", 900000),
            "sic": ("data_float_h", "DataID", 725, "Offset", 900),
        }[database]
        for path in root.glob(f"{table}*/**/*.parquet"):
            frame = pd.read_parquet(path)
            frame.loc[frame[item].eq(code), clock] += shift
            frame.to_parquet(path, index=False)
        path = root / f"{table}.parquet"
        if path.exists():
            frame = pd.read_parquet(path)
            frame.loc[frame[item].eq(code), clock] += shift
            frame.to_parquet(path, index=False)
