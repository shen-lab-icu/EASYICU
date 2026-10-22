"""SOFA-2 exports retain the score owner's rolling organs and receipts."""

from __future__ import annotations

import copy
import json

import pandas as pd
import pytest

import easyicu
from easyicu.api import extraction as api
from easyicu.concept import ConceptResolver
from easyicu.concept.schema import ConceptSource
from easyicu.config import DataSourceConfig
from easyicu.resources import load_dictionary
from easyicu.scores.sofa2_aggregate import SOFA2_COMPONENT_NAMES
from easyicu.table import ICUTable


COMPONENTS = list(SOFA2_COMPONENT_NAMES)
REQUESTED = list(api.EXTRACT_MODULES["sofa2_score"])


@pytest.fixture
def score_loader():
    """Replace source IO only; execute the shipped scorer and output merge."""
    dictionary = copy.deepcopy(load_dictionary(include_sofa2=True))
    frames = []
    for stay in (11, 12):
        frame = pd.DataFrame(
            {
                "stay_id": stay,
                "charttime": range(28),
                "pafi": 500.0,
                "adv_resp": True,
                "plt": 200.0,
                "bili": 1.0,
                "map": 75.0,
                "gcs": 15.0,
                "delirium_tx_proxy": False,
                "crea": 1.0,
            }
        )
        if stay == 11:
            # The 24h worst CNS (3) and respiratory (2) states overlap,
            # although the abnormal physiology is observed at different hours.
            frame.loc[0, "gcs"] = 6.0
            frame.loc[1, "pafi"] = 200.0
        else:
            # Missing renal evidence contributes zero. Liver state remains
            # available by SOFA-2 LOCF when respiratory failure starts on day 2.
            frame["crea"] = float("nan")
            frame["bili"] = float("nan")
            frame.loc[0, "bili"] = 8.0
            frame.loc[25, "pafi"] = 200.0
        frames.append(frame)
    observations = pd.concat(frames, ignore_index=True)
    for name in observations.columns.difference(["stay_id", "charttime"]):
        dictionary[name].sources["fixture"] = [
            ConceptSource(table="events", value_var=name, index_var="charttime")
        ]

    class Source:
        base_path = None

        def __init__(self, id_col, patient_ids):
            self.id_col = id_col
            self.patient_ids = patient_ids
            self.config = DataSourceConfig(
                name="fixture",
                tables={
                    "events": {
                        "defaults": {
                            "id_var": id_col,
                            "index_var": "charttime",
                            "val_var": "value",
                        }
                    }
                },
            )

        def load_table(self, table_name, columns=None, filters=None, verbose=False):
            frame = observations.rename(columns={"stay_id": self.id_col}).copy()
            if self.patient_ids:
                frame = frame[frame[self.id_col].isin(self.patient_ids[self.id_col])]
            if columns:
                frame = frame[
                    list(
                        dict.fromkeys(
                            [
                                self.id_col,
                                "charttime",
                                *(c for c in columns if c in frame),
                            ]
                        )
                    )
                ]
            values = [c for c in frame if c not in {self.id_col, "charttime"}]
            return ICUTable(
                frame,
                id_columns=[self.id_col],
                index_column="charttime",
                value_column=values[0] if len(values) == 1 else None,
            )

        def clear_cache(self):
            pass

    def load(**kwargs):
        patient_ids = kwargs.get("patient_ids")
        if patient_ids:
            patient_ids = {"stay_id": next(iter(patient_ids.values()))}
        result = ConceptResolver(dictionary).load_concepts(
            kwargs["concepts"],
            Source("stay_id", patient_ids),
            merge=kwargs.get("merge", True),
            interval=pd.Timedelta(hours=1),
            keep_components=kwargs.get("keep_components", False),
            r_compatible=kwargs.get("r_compatible", True),
            verbose=False,
            concept_workers=1,
        )
        # Model the native eICU module boundary without borrowing its raw IO.
        if kwargs.get("database") == "eicu":
            result = result.rename(columns={"stay_id": "patientunitstayid"})
        return result

    return load


@pytest.mark.parametrize("streamed", [False, True])
@pytest.mark.parametrize("force_duckdb", [False, True])
def test_sofa2_module_preserves_rolling_and_locf_states(
    score_loader,
    monkeypatch,
    tmp_path,
    streamed,
    force_duckdb,
):
    monkeypatch.setattr(easyicu, "load_concepts", score_loader)
    # Keep synthetic IO in this process; the production child uses this same
    # module loader, while a spawned interpreter would lose the fixture.
    monkeypatch.setattr(api, "_requires_isolated_stream_batch", lambda *_args: False)
    if force_duckdb:
        monkeypatch.setattr(
            api, "_native_export_pandas_fallback_is_bounded", lambda _size: False
        )
    reference = score_loader(
        concepts=[c for c in REQUESTED if c not in COMPONENTS],
        keep_components=True,
        r_compatible=False,
        database="eicu",
    ).rename(columns={"patientunitstayid": "stay_id"})
    expected = reference.set_index(["stay_id", "charttime"])
    receipts = [
        f"{c}_{suffix}"
        for c in ["sofa2", *COMPONENTS]
        for suffix in ("observed", "available")
    ]
    expected[receipts] = expected[receipts].astype("boolean")
    assert expected.loc[(11, 1), "sofa2"] == 5
    assert expected.loc[(12, 25), "sofa2"] == 5
    assert expected.loc[(12, 25), "sofa2_n_available_components"] == 5

    api._run_module_extraction(
        module_name="sofa2_score",
        concepts=REQUESTED,
        database="eicu",
        data_path="/synthetic-only",
        patient_ids_filter={"patientunitstayid": [11, 12]},
        batch_size=1,
        output_dir=str(tmp_path),
        stream_output_batches=streamed,
    )
    assert json.loads((tmp_path / "_manifest.json").read_text())["errors"] == []
    (tmp_path / "_manifest.json").rename(tmp_path / "sofa2_score.manifest.json")
    staging = pd.read_parquet(tmp_path / "sofa2_score.parquet")
    assert staging.groupby("patientunitstayid").sofa2.max().to_dict() == {
        11: 5.0,
        12: 5.0,
    }

    api._publish_native_export_v2(
        database="eicu",
        data_path="/synthetic-only",
        output_dir=str(tmp_path),
        modules=["sofa2_score"],
        max_patients=None,
        result={"modules": {"sofa2_score": {"errors": []}}},
    )
    actual = pd.read_parquet(tmp_path / "sofa2_score.parquet")
    assert actual.groupby("stay_id").sofa2.max().to_dict() == {11: 5.0, 12: 5.0}
    assert (
        staging.loc[
            (staging.patientunitstayid == 11) & (staging.charttime == 1), "sofa2_cns"
        ].item()
        == 3
    )
    assert (
        staging.loc[
            (staging.patientunitstayid == 12) & (staging.charttime == 25),
            "sofa2_liver_available",
        ].item()
        == 1
    )
    columns = [
        *REQUESTED,
        *receipts,
    ]
    pd.testing.assert_frame_equal(
        actual.set_index(["stay_id", "charttime"])[columns].sort_index(),
        expected[columns].sort_index(),
        check_dtype=False,
    )
    assert "patientunitstayid" not in actual


@pytest.mark.parametrize("omitted", ["sofa2", "sofa2_renal"])
def test_sofa2_module_refuses_incomplete_rolled_output(score_loader, omitted):
    def incomplete(**kwargs):
        return score_loader(**kwargs).drop(columns=[omitted])

    with pytest.raises(
        ValueError, match="SOFA-2 aggregate callback omitted rolled components"
    ):
        api._load_module_concepts(
            incomplete,
            module_name="sofa2_score",
            load_kwargs={"concepts": REQUESTED},
        )


def test_component_only_projection_keeps_point_states(score_loader):
    kwargs = {"concepts": ["sofa2_cns"]}
    actual = api._load_module_concepts(
        score_loader,
        module_name="sofa2_score",
        load_kwargs=kwargs,
    )
    pd.testing.assert_frame_equal(actual, score_loader(**kwargs))
    assert (
        actual.loc[(actual.stay_id == 11) & (actual.charttime == 1), "sofa2_cns"].item()
        == 0
    )
