"""Tiny MIMIC-III/IV parquet sources exercising source column renaming and ICU clocks."""

from tests.support.bucket_storage import write_fixture_parquet

from tests.support.bucket_storage import synthetic_data_source

import pandas as pd

from easyicu.concept import ConceptResolver
from easyicu.config import load_src_cfg
from easyicu.datasource import ICUDataSource
from easyicu.resources import load_dictionary


def mimic_synthetic_source(root, database, *, bucketed=False, **tables):
    """Tiny parquet fixture exercising real source column renaming and ICU clocks."""
    id_col = "stay_id" if database == "miiv" else "icustay_id"
    start = pd.Timestamp("2150-01-01")
    write_fixture_parquet(pd.DataFrame({id_col: [1, 2, 3], "subject_id": [11, 12, 13], "hadm_id": [21, 22, 23],
                  "intime": start, "outtime": start + pd.Timedelta(days=3), "los": 3.}), root / "icustays.parquet", index=False)
    for table, frame in tables.items():
        path = root / f"{table}.parquet"
        if bucketed and table != "d_items":
            path = root / f"{table}_bucket" / "bucket_id=0" / "part.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
        write_fixture_parquet(frame, path, index=False)
    return ICUDataSource(load_src_cfg(database), base_path=root, enable_cache=False)


def mimic_synthetic_load(source, names, *, batch=True):
    resolver = ConceptResolver(load_dictionary(include_sofa2=True))
    id_col = "stay_id" if source.config.name == "miiv" else "icustay_id"
    result = {}
    for group in ([names] if batch else [[name] for name in names]):
        result.update(resolver.load_concepts(
            group, source, merge=False, r_compatible=False, interval=pd.Timedelta(hours=1),
            patient_ids={id_col: [1, 2, 3]}, concept_workers=1, verbose=False,
        ))
    return result
