"""Synthetic HiRID schemas whose clinical rows are replaced by test literals."""


from tests.support.bucket_storage import write_fixture_parquet
from tests.support.bucket_storage import declare_single_bucket_fixtures

import pandas as pd


def hirid_source_root(tmp_path, table, rows, layout="flat", preserve_other=False):
    """A complete tiny schema with all clinical rows replaced by test literals."""
    from easyicu.synthetic_sources import generate_source

    root = tmp_path / "hirid"
    generate_source(root, "hirid", stays=3)
    path = root / table / "1.parquet"
    original = pd.read_parquet(path)
    schema_row = original.iloc[[0]]
    frame = pd.concat([schema_row] * len(rows), ignore_index=True)
    for key in rows[0]:
        frame[key] = [row[key] for row in rows]
    origins = pd.read_parquet(root / "general.parquet").set_index("patientid").admissiontime
    time = "givenat" if table == "pharma" else "datetime"
    frame[time] = pd.to_datetime(frame.patientid.map(origins)) + pd.to_timedelta(frame[time], unit="h")
    if preserve_other:
        item_col = "pharmaid" if table == "pharma" else "variableid"
        original = original.loc[~original[item_col].isin(frame[item_col])]
        frame = pd.concat([original, frame], ignore_index=True)
    if layout == "bucket":
        path.unlink()
        path.parent.rmdir()
        path = root / f"{table}_bucket/bucket_id=0/1.parquet"
        path.parent.mkdir(parents=True)
    write_fixture_parquet(frame, path, index=False)
    declare_single_bucket_fixtures(root)
    return root


def hirid_load(root, names, **kwargs):
    from easyicu.api import concepts as api

    try:
        result = api.load_concepts(concepts=names, database="hirid", data_path=str(root), merge=False,
                                   verbose=False, concept_workers=1, parallel_workers=1, **kwargs)
        result = {names[0]: result} if isinstance(result, pd.DataFrame) else result
        return {name: table if isinstance(table, pd.DataFrame) else table.data
                for name, table in result.items()}
    finally:
        api.clear_global_loader()


def hirid_native(root, names, output, monkeypatch, module):
    """Publish only the requested synthetic concepts through native-v2."""
    from easyicu.api import extraction
    from easyicu.api.concepts import clear_global_loader

    monkeypatch.setitem(extraction.EXTRACT_MODULES, module, names)
    monkeypatch.setitem(extraction.EXTRACT_MODULES, "outcome", ["los_icu"])
    try:
        result = extraction.extract_database(
            "hirid", data_path=str(root), output_dir=str(output),
            modules=[module, "outcome"], patient_ids={"patientid": [1, 2, 3]},
            native_export_v2=True, verbose=False, resource_budget_mb=1024,
        )
        assert not {key: state["errors"] for key, state in result["modules"].items() if state.get("errors")}
        return pd.read_parquet(output / f"{module}.parquet")
    finally:
        clear_global_loader()
