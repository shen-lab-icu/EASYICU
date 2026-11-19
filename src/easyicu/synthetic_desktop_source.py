"""Generate the invented eICU-shaped archive owned by desktop M1 acceptance."""

from __future__ import annotations

import csv
import hashlib
import io
import random
import zipfile
from functools import lru_cache
from pathlib import Path

from .config import load_src_cfg
from .official_demo_sources import DemoSourcePaths, DemoSourceSpec

DATABASE = "eicu_demo"
PROVENANCE_DATABASE = "synthetic_eicu_demo"
QUESTION = "In the synthetic eICU cohort, is age associated with hospital mortality after adjustment for sex?"
ANALYSIS_INPUTS = {"question": QUESTION, "primary_exposure": "age", "target_outcome": "death", "covariates": ("sex",)}
FIRST_STAY_COORDINATE_FILENAME = "synthetic_first_stay.parquet"
SEED = 731
STAYS = 128


def _patient_rows():
    rng = random.Random(SEED)
    # Every age/sex cell has invented deaths and survivors. Shuffle the cells
    # independently of the stay IDs and acquisition order.
    cells = [(age, sex, i < (1 if age < 59 else 2)) for age in range(35, 83, 3)
             for sex in ("Female", "Male") for i in range(4)]
    rng.shuffle(cells)
    for stay, (age, sex, death) in enumerate(cells, 1):
        yield {
            "patientunitstayid": stay, "patienthealthsystemstayid": stay,
            "uniquepid": f"invented-{stay}", "gender": sex, "age": age,
            "hospitalid": 1, "wardid": 1, "unittype": "MICU",
            "admissionheight": 170, "admissionweight": 70, "dischargeweight": 70,
            "hospitaladmitoffset": -60, "hospitaldischargeoffset": 5760,
            "unitdischargeoffset": 2880, "unitvisitnumber": 1, "unitstaytype": "admit",
            "unitdischargestatus": "Alive", "hospitaldischargestatus": "Expired" if death else "Alive",
            "unitadmitsource": "Emergency Department", "hospitaladmitsource": "Emergency Department",
            "unitadmittime24": "01:00:00", "unitdischargetime24": "01:00:00",
            "hospitaladmittime24": "00:00:00", "hospitaldischargetime24": "00:00:00",
            "hospitaldischargeyear": 2100,
        }


@lru_cache(maxsize=1)
def _archive_bytes() -> bytes:
    payload = io.BytesIO()
    tables = load_src_cfg(DATABASE).tables
    with zipfile.ZipFile(payload, "w") as archive:
        for name, table in sorted(tables.items()):
            columns = list(table.columns or {})
            if not columns:
                continue
            stream = io.StringIO(newline="")
            writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
            writer.writeheader()
            if name == "patient":
                writer.writerows(_patient_rows())
            elif name == "vitalperiodic":
                writer.writerows({
                    "vitalperiodicid": stay, "patientunitstayid": stay, "observationoffset": 60,
                    "heartrate": 80, "systemicmean": 75, "sao2": 97, "temperature": 37,
                } for stay in range(1, STAYS + 1))
            entry = zipfile.ZipInfo(f"{name}.csv")
            entry.create_system = 3
            archive.writestr(entry, stream.getvalue().encode("utf-8"))
    return payload.getvalue()


def source_spec() -> DemoSourceSpec:
    archive = _archive_bytes()
    return DemoSourceSpec(
        id="synthetic_eicu_m1", database=DATABASE, version="synthetic-v1",
        title="Synthetic eICU desktop acceptance", description="Generated from code with invented values.",
        scope_summary="Synthetic desktop first-result acceptance", patients=STAYS, icu_stays=str(STAYS),
        archive_filename="synthetic-eicu-m1.zip", size_label=f"{len(archive)} bytes", size_bytes=len(archive),
        max_download_bytes=len(archive), max_uncompressed_bytes=len(archive),
        archive_sha256=hashlib.sha256(archive).hexdigest(),
        download_url="", landing_page="", citation_url="", license_url="", attribution="EasyICU fixtures",
    )


def write_archive(paths: DemoSourcePaths) -> None:
    paths.root.mkdir(parents=True, exist_ok=False)
    with paths.archive.open("xb") as archive:
        archive.write(_archive_bytes())


def first_stay_materializer_options(work_dir: Path, archive_sha256: str) -> dict:
    """Derive the normal first-stay coordinate from the generated identity table."""
    import pandas as pd

    from .research_agent.acquisition.first_icu_stay import FirstIcuStayBinding, derive_first_icu_stay
    from .research_agent.canonical_json import sha256_file

    if archive_sha256 != source_spec().archive_sha256:
        raise ValueError("synthetic archive identity changed")
    derived = derive_first_icu_stay(
        pd.DataFrame(_patient_rows()), stay_column="patientunitstayid", patient_column="uniquepid",
        order_column="unitvisitnumber", first_order_value=1, identity_table_name="synthetic_patient",
    )
    coordinate = work_dir / FIRST_STAY_COORDINATE_FILENAME
    with coordinate.open("xb") as output:
        derived.frame.to_parquet(output, index=False)
    return FirstIcuStayBinding(
        coordinate, sha256_file(coordinate), {**derived.receipt, "archive_sha256": archive_sha256},
    ).materializer_kwargs()
