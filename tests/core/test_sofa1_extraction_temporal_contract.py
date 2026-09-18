"""SOFA module publication must retain the aggregate callback's windowed organs."""

from __future__ import annotations

import json

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import easyicu
from easyicu.api import extraction as api
from easyicu.concept import ConceptResolver
from easyicu.concept.schema import ConceptDefinition, ConceptDictionary, ConceptSource
from easyicu.config import DataSourceConfig
from easyicu.scores.sepsis import sep3
from easyicu.table import ICUTable

COMPONENTS = list(api._SOFA1_COMPONENT_NAMES)
REQUESTED = ["sofa", *COMPONENTS]


@pytest.fixture
def score_loader():
    """Replace only source IO; retain real resolver, callback and output merge."""
    frames = []
    for stay in (1, 2, 3, 4):
        frame = pd.DataFrame(
            {
                "stay_id": stay,
                "charttime": range(28),
                **{c: [0.0] * 28 for c in COMPONENTS},
            }
        )
        if stay == 1:
            frame.loc[0, "sofa_cns"] = 4.0
            frame.loc[2, "sofa_cns"] = 2.0
        elif stay == 2:
            frame.loc[2, "sofa_resp"] = 2.0
        elif stay == 3:
            frame.loc[0, "sofa_cns"] = 3.0
            frame.loc[1, "sofa_resp"] = 2.0
            frame.loc[25, "sofa_liver"] = 4.0
        else:
            frame[COMPONENTS] = float("nan")
        # A missing organ remains missing; do not invent source observations.
        frame["sofa_renal"] = float("nan")
        frames.append(frame)
    observations = pd.concat(frames, ignore_index=True)
    definitions = {
        c: ConceptDefinition(
            name=c, sources={"unit": [ConceptSource(table=c, value_var="value")]}
        )
        for c in COMPONENTS
    }
    definitions["sofa"] = ConceptDefinition(
        name="sofa", sources={}, sub_concepts=COMPONENTS, callback="sofa_score"
    )

    class Source:
        def __init__(self, id_col):
            self.id_col = id_col
            self.config = DataSourceConfig(
                name="unit",
                tables={
                    c: {
                        "defaults": {
                            "id_var": id_col,
                            "index_var": "charttime",
                            "val_var": "value",
                        }
                    }
                    for c in COMPONENTS
                },
            )

        def load_table(self, table_name, columns=None, filters=None, verbose=False):
            frame = observations[["stay_id", "charttime", table_name]].rename(
                columns={table_name: "value", "stay_id": self.id_col}
            )
            return ICUTable(
                frame.copy(),
                id_columns=[self.id_col],
                index_column="charttime",
                value_column="value",
            )

        def clear_cache(self):
            pass

    def load(**kwargs):
        # Fresh resolver models an isolated extraction worker, not cached test values.
        resolver = ConceptResolver(ConceptDictionary(definitions))
        id_col = {"eicu": "patientunitstayid", "aumc": "admissionid"}.get(
            kwargs.get("database"), "stay_id"
        )
        return resolver.load_concepts(
            kwargs["concepts"],
            Source(id_col),
            merge=kwargs.get("merge", True),
            interval=pd.Timedelta(hours=1),
            keep_components=kwargs.get("keep_components", False),
            r_compatible=kwargs.get("r_compatible", True),
            verbose=False,
            concept_workers=1,
        )

    return load


def test_real_resolver_distinguishes_point_and_rolled_components(score_loader):
    separate = score_loader(concepts=REQUESTED)
    projected = score_loader(concepts=["sofa"], keep_components=True)
    rolled = score_loader(concepts=["sofa"], keep_components=True, r_compatible=False)
    assert "sofa_cns" not in projected
    assert (
        separate.loc[
            (separate.stay_id == 1) & (separate.charttime == 1), "sofa_cns"
        ].item()
        == 0
    )
    assert (
        rolled.loc[(rolled.stay_id == 1) & (rolled.charttime == 1), "sofa_cns"].item()
        == 4
    )
    assert (
        rolled.loc[(rolled.stay_id == 3) & (rolled.charttime == 1), "sofa"].item() == 5
    )


@pytest.mark.parametrize("streamed", [False, True])
@pytest.mark.parametrize("force_duckdb", [False, True])
def test_sofa_module_to_native_preserves_callback_windows(
    score_loader,
    monkeypatch,
    tmp_path,
    streamed,
    force_duckdb,
):
    monkeypatch.setattr(easyicu, "load_concepts", score_loader)
    monkeypatch.setattr(
        api,
        "EXTRACT_MODULES",
        {"sofa1_score": REQUESTED, "sepsis3_sofa1": ["sep3_sofa1"]},
    )
    if force_duckdb:
        monkeypatch.setattr(
            api, "_native_export_pandas_fallback_is_bounded", lambda _size: False
        )
    api._run_module_extraction(
        module_name="sofa1_score",
        concepts=REQUESTED,
        database="miiv",
        data_path="/synthetic-only",
        patient_ids_filter={"stay_id": [1, 2, 3, 4]},
        batch_size=4,
        output_dir=str(tmp_path),
        stream_output_batches=streamed,
    )
    staging = pd.read_parquet(tmp_path / "sofa1_score.parquet")
    assert json.loads((tmp_path / "_manifest.json").read_text())["errors"] == []
    (tmp_path / "_manifest.json").rename(tmp_path / "sofa1_score.manifest.json")
    reference = score_loader(
        concepts=["sofa"], keep_components=True, r_compatible=False
    )
    columns = ["stay_id", "charttime", *COMPONENTS, "sofa"]
    pd.testing.assert_frame_equal(
        staging[columns].sort_values(["stay_id", "charttime"]).reset_index(drop=True),
        reference[columns].sort_values(["stay_id", "charttime"]).reset_index(drop=True),
        check_dtype=False,
    )
    # Exercise both native duplicate-key backends with the same rolled states.
    # Duplicating a producer row must neither reroll it nor change its score.
    table = pq.read_table(tmp_path / "sofa1_score.parquet")
    pq.write_table(
        pa.concat_tables([table, table.slice(0, 1)]), tmp_path / "sofa1_score.parquet"
    )
    api._publish_native_export_v2(
        database="miiv",
        data_path="/synthetic-only",
        output_dir=str(tmp_path),
        modules=["sofa1_score"],
        max_patients=None,
        result={"modules": {"sofa1_score": {"errors": []}}},
    )
    published = pd.read_parquet(tmp_path / "sofa1_score.parquet")
    assert (
        pq.read_schema(tmp_path / "sofa1_score.parquet").metadata[
            api._SOFA1_TIME_BASIS_KEY
        ]
        == api._SOFA1_TIME_BASIS
    )
    manifest = json.loads((tmp_path / "_manifest.json").read_text())
    backend = manifest["files"][0]["row_grain_audit"]["publication_backend"]
    assert ("duckdb" in backend) == force_duckdb
    actual = published.set_index(["stay_id", "charttime"])
    assert actual.loc[(1, 24), "sofa"] == 4
    assert actual.loc[(1, 25), "sofa"] == 2
    assert actual.loc[(1, 27), "sofa"] == 0
    assert actual.loc[(3, 1), "sofa"] == 5
    assert actual.loc[(3, 24), "sofa"] == 5
    assert actual.loc[(3, 25), "sofa"] == 6
    assert actual.loc[(3, 26), "sofa"] == 4
    assert actual["sofa_renal"].isna().all()
    assert 4 not in actual.index.get_level_values("stay_id")
    assert actual["sofa"].equals(
        actual[COMPONENTS].sum(axis=1).astype(actual["sofa"].dtype)
    )

    # An observed high -> low -> rise is not a new SOFA rise while the original
    # high remains in the 24h window. A separate genuine baseline -> rise is.
    dependency = api._consolidate_special_score_dependency(
        staging,
        score_name="sofa",
        id_col="stay_id",
        time_col="charttime",
        database="miiv",
    )
    suspicion = pd.DataFrame(
        {"stay_id": [1, 2], "charttime": [0.0, 0.0], "susp_inf": [True, True]}
    )
    events = sep3(dependency, suspicion, id_cols=["stay_id"], index_col="charttime")
    assert events.stay_id.tolist() == [2]
    assert events.charttime.tolist() == [2.0]

    # The real streamed Sepsis reader must accept the published time basis.
    suspicion.to_parquet(tmp_path / "sepsis_shared.parquet", index=False)
    output = tmp_path / "special"
    output.mkdir()
    api._stream_special_extraction_batches(
        ["sepsis3_sofa1"],
        "miiv",
        "/synthetic-only",
        {"stay_id": [1, 2]},
        2,
        str(output),
        use_sofa2=False,
        published_output_dir=str(tmp_path),
    )
    consumed = pd.read_parquet(output / "sep3_sofa1.parquet")
    assert consumed.stay_id.tolist() == [2]
    assert consumed.charttime.tolist() == [2.0]


def test_legacy_point_components_are_refused_without_mutation(score_loader, tmp_path):
    """An old file cannot acquire rolled authority just by passing arithmetic."""
    import hashlib

    legacy = score_loader(concepts=REQUESTED)
    path = tmp_path / "sofa1_score.parquet"
    legacy.to_parquet(path, index=False)
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="component time basis is unverified"):
        api._publish_native_export_v2(
            database="miiv",
            data_path="/synthetic-only",
            output_dir=str(tmp_path),
            modules=["sofa1_score"],
            max_patients=None,
            result={"modules": {"sofa1_score": {"errors": []}}},
        )
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before

    assert not (tmp_path / "_manifest.json").exists()
    pd.DataFrame({"stay_id": [1], "charttime": [0.0], "susp_inf": [True]}).to_parquet(
        tmp_path / "sepsis_shared.parquet",
        index=False,
    )
    output = tmp_path / "special"
    output.mkdir()
    with pytest.raises(ValueError, match="component time basis is unverified"):
        api._stream_special_extraction_batches(
            ["sepsis3_sofa1"],
            "miiv",
            "/synthetic-only",
            {"stay_id": [1]},
            1,
            str(output),
            use_sofa2=False,
            published_output_dir=str(tmp_path),
        )
    assert not (output / "sep3_sofa1.parquet").exists()
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


@pytest.mark.parametrize(
    "database,id_col", [("eicu", "patientunitstayid"), ("aumc", "admissionid")]
)
def test_native_identifier_normalization_keeps_rolled_companions(
    score_loader,
    tmp_path,
    database,
    id_col,
):
    # Disabling the display projection must not lose the native stay identity.
    loaded = api._load_module_concepts(
        score_loader,
        module_name="sofa1_score",
        load_kwargs={"concepts": REQUESTED, "database": database, "merge": True},
    )
    assert id_col in loaded
    assert "stay_id" not in loaded
    table = api._module_arrow_table(loaded, REQUESTED, pa, module="sofa1_score")
    pq.write_table(table, tmp_path / "sofa1_score.parquet")
    api._publish_native_export_v2(
        database=database,
        data_path="/synthetic-only",
        output_dir=str(tmp_path),
        modules=["sofa1_score"],
        max_patients=None,
        result={"modules": {"sofa1_score": {"errors": []}}},
    )
    published = pd.read_parquet(tmp_path / "sofa1_score.parquet")
    assert id_col not in published
    assert (
        published.loc[
            (published.stay_id == 1) & (published.charttime == 1), "sofa_cns"
        ].item()
        == 4
    )
