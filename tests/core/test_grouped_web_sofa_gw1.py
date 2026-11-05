"""Grouped and web SOFA modules share the registered rolling score owner."""

from __future__ import annotations

import copy
import json
from contextlib import nullcontext
from types import SimpleNamespace

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import easyicu
import easyicu.api as public_api
from easyicu.api import extraction
from easyicu.concept import ConceptResolver
from easyicu.concept.callbacks import ConceptCallbackContext, _callback_sofa2_score
from easyicu.concept.schema import ConceptDictionary, ConceptSource
from easyicu.config import DataSourceConfig
from easyicu.resources import load_dictionary
from easyicu.table import ICUTable
from easyicu.webserver import dataio


class Job:
    def emit(self, _event):
        pass


@pytest.fixture
def score_source(monkeypatch, tmp_path):
    """Invent point organs; run the shipped resolver, score and output merge."""
    shipped = load_dictionary(include_sofa2=True)
    definitions = {name: copy.deepcopy(shipped[name]) for name in ("sofa", "sofa2")}
    frames = {}
    for score in ("sofa", "sofa2"):
        for organ in ("resp", "coag", "liver", "cardio", "cns", "renal"):
            name = f"{score}_{organ}"
            definition = copy.deepcopy(shipped[name])
            definition.class_name = "num_cncpt"
            definition.callback = None
            definition.sub_concepts = []
            definition.depends_on = []
            definition.sources = {"fixture": [
                ConceptSource(table=name, value_var="value", index_var="charttime"),
            ]}
            definitions[name] = definition
            # CNS injury precedes respiratory injury. Point CNS has recovered
            # at hour 1, while the rolling owner must retain its worst state.
            frames[name] = pd.DataFrame({
                "stay_id": [11, 11, 11, 12, 12],
                "charttime": [0.0, 1.0, 26.0, 0.0, 26.0],
                "value": [4.0 if organ == "cns" else 0.0,
                          2.0 if organ == "resp" else 0.0,
                          0.0, 0.0, 0.0],
            })
    # The full SOFA-2 catalog carries sensitivity/ascertainment outputs too.
    # They stay independent of the six rolling organs and must survive.
    for name in set(extraction.EXTRACT_MODULES["sofa2_score"]) - set(definitions):
        definition = copy.deepcopy(shipped[name])
        definition.class_name = "chr_cncpt" if "ascertainment" in name else "num_cncpt"
        definition.callback = None
        definition.sub_concepts = []
        definition.depends_on = []
        definition.sources = {"fixture": [
            ConceptSource(table=name, value_var="value", index_var="charttime"),
        ]}
        definitions[name] = definition
        frames[name] = pd.DataFrame({
            "stay_id": [11, 12], "charttime": [1.0, 1.0],
            "value": ["unknown", "unknown"] if "ascertainment" in name else [1.0, 0.0],
        })
    dictionary = ConceptDictionary(definitions)

    class Source:
        base_path = None
        config = DataSourceConfig(name="fixture", tables={
            name: {"defaults": {"id_var": "stay_id", "index_var": "charttime", "val_var": "value"}}
            for name in frames
        })

        def __init__(self, ids):
            self.ids = next(iter(ids.values())) if isinstance(ids, dict) else ids

        def load_table(self, table_name, columns=None, filters=None, verbose=False):
            frame = frames[table_name]
            if self.ids is not None:
                frame = frame[frame.stay_id.isin(self.ids)]
            return ICUTable(frame.copy(), ["stay_id"], "charttime", "value")

        def clear_cache(self):
            pass

    calls = []

    def load(concepts, **kwargs):
        calls.append((list(concepts), kwargs))
        if concepts == ["los_icu"]:
            return pd.DataFrame({"patientunitstayid": [11, 12], "los_icu": [2.0, 2.0]})
        result = ConceptResolver(dictionary).load_concepts(
            concepts, Source(kwargs.get("patient_ids")), interval=pd.Timedelta(hours=1),
            keep_components=kwargs.get("keep_components", False),
            r_compatible=kwargs.get("r_compatible", True),
            merge=kwargs.get("merge", True), verbose=False, concept_workers=1,
        )
        return result.rename(columns={"stay_id": "patientunitstayid"})

    monkeypatch.setenv("EASYICU_DISABLE_PROVIDER_ENV_FILE", "1")
    monkeypatch.setattr(easyicu, "load_concepts", load)
    monkeypatch.setattr(public_api, "load_concepts", load)
    monkeypatch.setattr(public_api, "keep_cache", lambda **_kwargs: nullcontext())
    monkeypatch.setattr(public_api, "get_all_patient_ids", lambda *_args, **_kwargs: ([11, 12], "patientunitstayid"))
    return load, calls


def _web_export(tmp_path, module, concepts=None):
    output = tmp_path / "web"
    dataio.make_export_runner(
        str(tmp_path), "eicu", modules=[module],
        concepts={module: concepts} if concepts is not None else None,
        export_format="parquet", out_dir=str(output), include_feature_definitions=False,
    )(Job())
    return pd.read_parquet(output / f"{module}.parquet")


@pytest.mark.parametrize("module,score", [("sofa1_score", "sofa"), ("sofa2_score", "sofa2")])
def test_web_and_grouped_sofa_preserve_the_same_rolling_organs(score_source, tmp_path, module, score):
    requested = extraction.EXTRACT_MODULES[module]
    grouped = tmp_path / "grouped"
    grouped.mkdir()
    extraction._run_module_extraction(
        module_name=module, concepts=requested, database="eicu", data_path=str(tmp_path),
        patient_ids_filter={"patientunitstayid": [11, 12]}, batch_size=2,
        output_dir=str(grouped), use_sofa2=score == "sofa2",
    )
    assert json.loads((grouped / "_manifest.json").read_text())["errors"] == []
    (grouped / "_manifest.json").rename(grouped / f"{module}.manifest.json")
    extraction._publish_native_export_v2(
        database="eicu", data_path=str(tmp_path), output_dir=str(grouped),
        modules=[module], max_patients=None, result={"modules": {module: {"errors": []}}},
    )
    expected = pd.read_parquet(grouped / f"{module}.parquet").set_index(["stay_id", "charttime"])
    actual = _web_export(tmp_path, module).rename(
        columns={"patientunitstayid": "stay_id"},
    ).set_index(["stay_id", "charttime"])
    assert len(expected) == 54  # Both callback grids span hours 0 through 26.
    assert len(actual) == len(expected)
    pd.testing.assert_index_equal(actual.index.sort_values(), expected.index.sort_values())
    # Explicit clinical oracle in addition to path parity: no recovered point
    # state can replace a valid trailing worst state or outlive its window.
    assert expected.loc[(11, 1.0), f"{score}_cns"] == 4.0
    assert expected.loc[(11, 26.0), f"{score}_cns"] == 0.0
    assert actual.loc[(11, 1.0), f"{score}_cns"] == 4.0
    assert expected.loc[(11, 25.0), score] == 0.0
    assert actual.loc[(11, 25.0), score] == 0.0
    numeric = {name: "float64" for name in requested if pd.api.types.is_numeric_dtype(expected[name])}
    pd.testing.assert_frame_equal(
        actual[requested].astype(numeric).sort_index(), expected[requested].astype(numeric).sort_index(),
        check_dtype=False,
    )
    assert not actual.index.duplicated().any()
    if score == "sofa2":
        flag = "sofa2_all_components_missing"
        for output, frame in ((grouped, expected), (tmp_path / "web", actual)):
            assert pd.api.types.is_bool_dtype(frame[flag])
            assert not frame[flag].isna().any()
            pd.testing.assert_series_equal(
                frame[flag], frame[list(extraction.SOFA2_COMPONENT_NAMES)].isna().all(axis=1),
                check_names=False, check_dtype=False,
            )
            manifest = json.loads((output / "_manifest.json").read_text())
            # Only native-v2 and web publish the typed metadata sidecar.
            if "column_metadata" in manifest:
                metadata = json.loads((output / manifest["column_metadata"]["file"]).read_text())
                binding = next(item for item in metadata["files"] if item["module"] == module)["columns"][flag]
                assert binding["metadata"]["source_concept"] == "sofa2"
                assert binding["metadata"]["role"] == "measurement_status"
                assert "all six SOFA-2 component values are missing" in binding["metadata"]["description"]
                assert flag in manifest["files"][0]["column_metadata_columns"]
        pd.testing.assert_series_equal(actual[flag].sort_index(), expected[flag].sort_index(), check_dtype=False)


@pytest.mark.parametrize("module,component", [("sofa1_score", "sofa_cns"), ("sofa2_score", "sofa2_cns")])
def test_component_only_web_projection_keeps_point_states(score_source, tmp_path, module, component):
    load, _ = score_source
    expected = load([component], merge=True)
    actual = _web_export(tmp_path, module, [component])
    pd.testing.assert_frame_equal(actual, expected, check_dtype=False)
    assert actual.loc[actual.patientunitstayid.eq(11) & actual.charttime.eq(1.0), component].item() == 0.0


@pytest.mark.parametrize("module,omitted", [("sofa1_score", "sofa_renal"), ("sofa2_score", "sofa2_renal")])
def test_web_refuses_a_total_without_its_rolling_companions(score_source, monkeypatch, tmp_path, module, omitted):
    load, _ = score_source

    def incomplete(concepts, **kwargs):
        return load(concepts, **kwargs).drop(columns=[omitted], errors="ignore")

    monkeypatch.setattr(public_api, "load_concepts", incomplete)
    with pytest.raises(ValueError, match="aggregate callback omitted rolled components"):
        _web_export(tmp_path, module)
    assert not (tmp_path / "web" / f"{module}.parquet").exists()
    assert not (tmp_path / "web" / "_manifest.json").exists()


@pytest.mark.parametrize("backend", ["arrow", "pandas", "duckdb"])
def test_native_publication_preserves_research_normal_imputation(score_source, monkeypatch, tmp_path, backend):
    requested = extraction.EXTRACT_MODULES["sofa2_score"]
    extraction._run_module_extraction(
        module_name="sofa2_score", concepts=requested, database="eicu", data_path=str(tmp_path),
        patient_ids_filter={"patientunitstayid": [11, 12]}, batch_size=2,
        output_dir=str(tmp_path), use_sofa2=True,
    )
    # A real assessment with six missing domains still owns a normal-imputed
    # zero. Exercise the total's domain witness independently of organ values.
    tables = {
        name: ICUTable(pd.DataFrame({
            "patientunitstayid": [13], "charttime": [0.0], name: [float("nan")],
            f"{name}_observed": [0], f"{name}_available": [0],
        }), ["patientunitstayid"], "charttime", name)
        for name in extraction.SOFA2_COMPONENT_NAMES
    }
    assessed = _callback_sofa2_score(tables, ConceptCallbackContext(
        "sofa2", None, pd.Timedelta(hours=1), None,
        SimpleNamespace(config=SimpleNamespace(name="eicu")), None, {"keep_components": True},
    )).data
    staging = pd.concat([pd.read_parquet(tmp_path / "sofa2_score.parquet"), assessed], ignore_index=True)
    staging.to_parquet(tmp_path / "sofa2_score.parquet", index=False)
    owner = staging.set_index(["patientunitstayid", "charttime"])
    # Hour 25 is a filled grid cell beyond all evidence windows. It is not
    # an assessed patient with missing domains, for whom zero is correct.
    assert pd.isna(owner.loc[(11, 25.0), "sofa2"])
    assert owner.loc[(11, 26.0), "sofa2"] == 0.0
    assert owner.loc[(13, 0.0), "sofa2"] == 0.0
    assert owner.loc[(11, 25.0), list(extraction.SOFA2_COMPONENT_NAMES)].isna().all()
    if backend != "arrow":
        table = pq.read_table(tmp_path / "sofa2_score.parquet")
        pq.write_table(pa.concat_tables([table, table]), tmp_path / "sofa2_score.parquet")
    if backend == "duckdb":
        monkeypatch.setattr(extraction, "_native_export_pandas_fallback_is_bounded", lambda _size: False)
    (tmp_path / "_manifest.json").rename(tmp_path / "sofa2_score.manifest.json")
    extraction._publish_native_export_v2(
        database="eicu", data_path=str(tmp_path), output_dir=str(tmp_path),
        modules=["sofa2_score"], max_patients=None,
        result={"modules": {"sofa2_score": {"errors": []}}},
    )
    manifest = json.loads((tmp_path / "_manifest.json").read_text())
    assert manifest["files"][0]["row_grain_audit"]["publication_backend"] == {
        "arrow": "pyarrow_record_batches",
        "pandas": "pandas_bounded_row_grain_fallback",
        "duckdb": "duckdb_bounded_spillable_row_grain_consolidation",
    }[backend]
    published = pd.read_parquet(tmp_path / "sofa2_score.parquet").set_index(["stay_id", "charttime"])
    assert published.loc[(11, 25.0), "sofa2"] == 0.0
    assert published.loc[(11, 26.0), "sofa2"] == 0.0
    assert published.loc[(13, 0.0), "sofa2"] == 0.0
    # Compare as float64: pandas 3 may publish nullable Float64 on one side.
    pd.testing.assert_series_equal(published.sofa2.sort_index().astype("float64"),
                                   owner.sofa2.fillna(0).sort_index().astype("float64"), check_names=False)
    flag = "sofa2_all_components_missing"
    assert published[flag].notna().all()
    assert published.loc[(11, 25.0), flag]
    assert published.loc[(13, 0.0), flag]
    assert not published.loc[(11, 26.0), flag]  # Six present zero scores.
    assert published.columns[-1] == flag
    metadata = json.loads((tmp_path / manifest["column_metadata"]["file"]).read_text())
    assert flag in metadata["files"][0]["columns"]
    assert flag in manifest["files"][0]["column_metadata_columns"]


@pytest.mark.parametrize("backend", ["pandas", "arrow"])
@pytest.mark.parametrize("component", ["sofa2_resp", "sofa2_renal"])
def test_component_evidence_still_allows_a_missing_total_to_be_rebuilt(backend, component):
    components = list(extraction.SOFA2_COMPONENT_NAMES)
    frame = pd.DataFrame({
        "stay_id": [11], "charttime": [0.0], "sofa2": [float("nan")],
        "sofa2_observed": [False], "sofa2_available": [False],
        **{name: [4.0 if name == component else float("nan")] for name in components},
        **{f"{name}_{suffix}": [name == component]
           for name in components for suffix in ("observed", "available")},
    })
    options = dict(module="sofa2_score", requested_concepts=["sofa2", *components], database="eicu")
    if backend == "arrow":
        result = extraction._recompute_native_sofa2_aggregate_arrow(pa.Table.from_pandas(frame), **options).to_pandas()
    else:
        result = extraction._recompute_native_sofa2_aggregate_frame(frame, **options)
    assert result.sofa2.item() == 4.0
    assert not result.sofa2_available.item()


@pytest.mark.parametrize("backend", ["pandas", "arrow", "arrow_nan"])
@pytest.mark.parametrize("database", ["eicu", "sic", "sicdb"])
def test_sofa2_missing_flag_preserves_every_existing_column(backend, database):
    """Frozen pre-gw1 totals/receipts; zero and missing totals cannot identify gaps."""
    components = list(extraction.SOFA2_COMPONENT_NAMES)
    frame = pd.DataFrame({
        "stay_id": [11] * 5, "charttime": [0., 1., 2., 3., 4.],
        "sofa2": [float("nan"), 0., float("nan"), 9., 2.],
        "sofa2_observed": pd.Series([False] * 5, dtype="boolean"),
        "sofa2_available": pd.Series([False] * 5, dtype="boolean"),
        **{name: [float("nan"), 0. if name == components[0] else float("nan"),
                  4. if name == components[-1] else float("nan"), float("nan"),
                  2. if name == "sofa2_cns" else float("nan")]
           for name in components},
        **{f"{name}_{suffix}": pd.Series(
            [pd.NA, name == components[0], name == components[-1], False, False], dtype="boolean",
        ) for name in components for suffix in ("observed", "available")},
        "sofa2_cns_delirium_tx_ascertainment": pd.Series(["unknown"] * 5, dtype="string"),
    })
    expected = frame.copy(deep=True)
    expected["sofa2"] = [0., 0., 4., 0., 0.] if database == "eicu" else float("nan")
    options = dict(module="sofa2_score", requested_concepts=["sofa2", *components], database=database)
    if backend.startswith("arrow"):
        table = pa.Table.from_pandas(frame)
        if backend == "arrow_nan":
            for name in components:
                table = table.set_column(
                    table.column_names.index(name), name,
                    pa.array(frame[name].tolist(), type=pa.float64(), from_pandas=False),
                )
        result = extraction._recompute_native_sofa2_aggregate_arrow(table, **options).to_pandas()
    else:
        result = extraction._recompute_native_sofa2_aggregate_frame(frame, **options)
    pd.testing.assert_frame_equal(result[list(expected)], expected)
    assert list(result) == [*expected, "sofa2_all_components_missing"]
    assert result.sofa2_all_components_missing.tolist() == [True, False, False, True, False]
    assert pd.api.types.is_bool_dtype(result.sofa2_all_components_missing)


@pytest.mark.parametrize("backend", ["pandas", "duckdb"])
def test_native_flag_is_rebuilt_after_component_consolidation(monkeypatch, tmp_path, backend):
    components = list(extraction.SOFA2_COMPONENT_NAMES)
    requested = ["sofa2", *components]
    monkeypatch.setattr(extraction, "EXTRACT_MODULES", {"sofa2_score": requested})
    frame = pd.DataFrame({
        "stay_id": [11, 11], "charttime": [0., 0.], "sofa2": [0., 0.],
        "sofa2_observed": [False, False], "sofa2_available": [False, False],
        **{name: [None, 0. if name == "sofa2_resp" else None] for name in components},
        **{f"{name}_{suffix}": [False, name == "sofa2_resp"]
           for name in components for suffix in ("observed", "available")},
        "sofa2_all_components_missing": [True, False],
    })
    frame.to_parquet(tmp_path / "sofa2_score.parquet", index=False)
    if backend == "duckdb":
        monkeypatch.setattr(extraction, "_native_export_pandas_fallback_is_bounded", lambda _size: False)
    extraction._publish_native_export_v2(
        database="eicu", data_path=str(tmp_path), output_dir=str(tmp_path),
        modules=["sofa2_score"], max_patients=None,
        result={"modules": {"sofa2_score": {"errors": []}}},
    )
    published = pd.read_parquet(tmp_path / "sofa2_score.parquet")
    assert len(published) == 1
    assert published.sofa2.item() == 0.
    assert published.sofa2_resp.item() == 0.
    assert published.sofa2_all_components_missing.tolist() == [False]
