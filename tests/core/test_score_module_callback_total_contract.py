"""Catalog score exports preserve independently loaded owner totals.

Only source IO is synthetic. The public concept dispatcher, resolver, registered
callbacks, module writer, Sepsis dependency reader and native publisher run for
real. In particular, a sum of published point components is not the oracle.
"""

from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from easyicu.api import concepts as concept_api
from easyicu.api import extraction as api
from easyicu.concept import ConceptResolver
from easyicu.concept.catalog import CONCEPT_GROUPS_INTERNAL
from easyicu.concept.schema import ConceptSource
from easyicu.config import DataSourceConfig
from easyicu.resources import load_dictionary
from easyicu.scores import comorbidity
from easyicu.table import ICUTable


SCORE_TOTALS = {
    "sofa1_score": ("sofa",),
    "sofa2_score": ("sofa2",),
    "sepsis3_sofa1": ("sep3_sofa1",),
    "sepsis3_sofa2": ("sep3_sofa2",),
    "vitals": ("shock_index", "modified_shock_index", "diastolic_shock_index"),
    "neurological": ("gcs",),
    "renal": (
        "aki_reference", "aki_stage_reference", "aki_severe_reference",
        "aki_stage_reference_smoothed_6h", "aki_source_native",
        "aki_stage_source_native", "aki_severe_source_native",
        "aki_stage_source_native_smoothed",
    ),
    "circulatory": ("circ_event", "circ_failure"),
    # Derive this list from the catalog so a new score cannot silently escape.
    "other_scores": tuple(CONCEPT_GROUPS_INTERNAL["other_scores"]),
}

# These are source-supplied admission summaries, not executable score callbacks.
# There is consequently no callback-alone oracle; fabricating one would test a
# replacement implementation. Their physical types remain covered by native-v2
# schema tests. The module still loads them alongside the executable scores.
SKIPPED_TOTALS = {
    "apache_iv": "Source-native first-day APACHE IVa; no scoring callback.",
    "apache_iv_pred_hosp_mort": "Source-native APACHE prediction; no callback.",
    "saps3": "Source-native SICdb admission SAPS-3; no scoring callback.",
}

# Explicit dispositions for every catalog module outside the score sweep.
NON_SCORE_MODULES = {
    "sepsis_shared": "Infection/culture evidence, consumed by the Sepsis cases.",
    "respiratory": "Physiological ratios and support indicators, not score totals.",
    "ventilator": "Measured settings and derived pressures, not score totals.",
    "blood_gas": "Measured blood gases; no derived aggregate.",
    "chemistry": "Measured chemistry and derived concentrations, not scores.",
    "hematology": "Measured counts and cell ratios, not scores.",
    "vasopressors": "Drug rates, duration windows and equivalence, not scores.",
    "medications": "Medication exposure/rates; no score total.",
    "demographics": "Stay-level demographics/BMI, not a longitudinal score.",
    "outcome": "Endpoint/follow-up definitions, not severity score totals.",
}


def _cases():
    assert set(SCORE_TOTALS).isdisjoint(NON_SCORE_MODULES)
    assert set(SCORE_TOTALS) | set(NON_SCORE_MODULES) == set(CONCEPT_GROUPS_INTERNAL)
    for module, totals in SCORE_TOTALS.items():
        assert set(totals).issubset(CONCEPT_GROUPS_INTERNAL[module])
        for total in totals:
            reason = SKIPPED_TOTALS.get(total)
            yield pytest.param(
                module, total, id=f"{module}-{total}",
                marks=pytest.mark.skip(reason=reason) if reason else (),
            )


@pytest.fixture
def synthetic_loader(monkeypatch, tmp_path):
    """Bind the public API to in-memory source tables, never a data directory."""
    dictionary = copy.deepcopy(load_dictionary(include_sofa2=True))
    frames = []
    for stay in (11, 12):
        frame = pd.DataFrame({
            "stay_id": stay, "charttime": range(30),
            "hr": 80.0, "sbp": 120.0, "dbp": 60.0, "map": 75.0,
            "resp": 18.0, "temp": 37.0, "o2sat": 98.0,
            "pafi": 500.0, "adv_resp": True, "plt": 200.0, "bili": 1.0,
            "egcs": 4.0, "mgcs": 6.0, "vgcs": 5.0,
            "crea": 1.0, "urine": 70.0, "weight": 70.0,
            "rrt": False, "crrt_mode_input": False, "lact": 1.0,
            "wbc": 8.0, "pco2": 40.0, "bnd": 0.0,
            "delirium_tx_proxy": False, "abx": True, "samp": True,
        })
        if stay == 11:
            # Verbal=2 avoids MIIV's verbal=1 sedation-imputation proxy, so
            # the initial CNS state overlaps the later respiratory insult.
            frame.loc[0, ["egcs", "mgcs", "vgcs"]] = [2.0, 3.0, 2.0]
            frame.loc[1, "pafi"] = 200.0
            frame.loc[2, ["hr", "sbp", "map", "resp", "temp", "lact"]] = [
                130.0, 85.0, 55.0, 30.0, 39.0, 3.0,
            ]
            # Asynchronous gaps exercise LOCF, not arithmetic on raw inputs.
            frame.loc[3:29, ["hr", "sbp", "resp", "temp"]] = float("nan")
            frame.loc[7:10, ["egcs", "vgcs"]] = float("nan")
            frame.loc[12:18, "urine"] = 0.0
            frame.loc[20:29, "crea"] = 2.5
            frame.loc[26:29, "rrt"] = True
        else:
            # Missing-as-normal differs from complete-case scoring. Persistent
            # liver state and a day-two respiratory insult exercise SOFA-2 LOCF.
            frame["crea"] = float("nan")
            frame["bili"] = float("nan")
            frame.loc[0, "bili"] = 8.0
            frame.loc[25, "pafi"] = 200.0
        frames.append(frame)
    observations = pd.concat(frames, ignore_index=True)
    # Distinct observations in one source hour test the resolver's aggregation
    # as well as the native writer's duplicate-key handling.
    subhour = observations.loc[observations.charttime == 2].copy()
    subhour["charttime"] += 0.5
    subhour.loc[subhour.stay_id == 11, ["hr", "sbp", "resp", "temp"]] = [
        60.0, 140.0, 10.0, 36.5,
    ]
    observations = pd.concat([observations, subhour], ignore_index=True)
    # No absent concept may fall through to an actual source mapping.
    for definition in dictionary.values():
        definition.sources["miiv"] = []
    for name in observations.columns.difference(["stay_id", "charttime"]):
        dictionary[name].sources["miiv"] = [
            ConceptSource(table=name, value_var="value", index_var="charttime")
        ]

    class Source:
        base_path = None
        config = DataSourceConfig(
            name="miiv", tables={
                name: {"defaults": {
                    "id_var": "stay_id", "index_var": "charttime", "val_var": "value",
                }}
                for name in observations.columns if name not in {"stay_id", "charttime"}
            },
        )

        def __init__(self, patients):
            self.ids = next(iter(patients.values())) if isinstance(patients, dict) else patients

        def load_table(self, table_name, columns=None, filters=None, verbose=False):
            assert table_name in observations, "Synthetic fixture forbids raw IO"
            frame = observations[["stay_id", "charttime", table_name]].rename(
                columns={table_name: "value"}
            ).dropna(subset=["value"])
            if self.ids is not None:
                frame = frame[frame.stay_id.isin(self.ids)]
            return ICUTable(
                frame.copy(), id_columns=["stay_id"], index_column="charttime",
                value_column="value",
            )

        def clear_cache(self):
            pass

    class Loader:
        database = "miiv"
        data_path = tmp_path / "synthetic-source"

        def load_concepts(self, concepts, **kwargs):
            result = ConceptResolver(dictionary).load_concepts(
                concepts, Source(kwargs.get("patient_ids")), interval=pd.Timedelta(hours=1),
                keep_components=kwargs.get("keep_components", False),
                merge=kwargs.get("merge", True),
                r_compatible=kwargs.get("r_compatible", True),
                verbose=False, concept_workers=1,
            )
            patients = kwargs.get("patient_ids")
            ids = next(iter(patients.values())) if isinstance(patients, dict) else patients
            if ids is not None:
                if isinstance(result, pd.DataFrame) and "stay_id" in result:
                    result = result[result.stay_id.isin(ids)].copy()
                elif isinstance(result, dict):
                    result = {
                        name: ICUTable(
                            table.data[table.data.stay_id.isin(ids)].copy(),
                            id_columns=table.id_columns, index_column=table.index_column,
                            value_column=table.value_column,
                        )
                        for name, table in result.items()
                    }
            return result

        def clear_cache(self):
            pass

    monkeypatch.setattr(concept_api, "_get_global_loader", lambda **_kwargs: Loader())
    # Comorbidity has its own raw-table adapter. Replace its IO, retain its
    # diagnosis matching, weighting and admission-to-stay projection.
    diagnosis_tables = {
        "diagnoses_icd": pd.DataFrame({
            "hadm_id": [101, 102], "icd_code": ["I50", "E119"], "icd_version": [10, 10],
        }),
        "icustays": pd.DataFrame({"hadm_id": [101, 102], "stay_id": [11, 12]}),
    }
    monkeypatch.setattr(comorbidity, "_build_datasource", lambda *_args: SimpleNamespace())
    monkeypatch.setattr(
        comorbidity, "_table_df", lambda _source, name: diagnosis_tables[name].copy(),
    )
    monkeypatch.setattr(api, "_requires_isolated_stream_batch", lambda *_args: False)
    return concept_api.load_concepts


@pytest.mark.parametrize("module,total", list(_cases()))
@pytest.mark.parametrize("streamed,backend", [
    (False, "pandas"), (True, "arrow"), (True, "duckdb"),
])
def test_every_score_module_exports_its_callback_total(
    synthetic_loader, monkeypatch, tmp_path, module, total, streamed, backend,
):
    requested = list(CONCEPT_GROUPS_INTERNAL[module])
    load_kwargs = {
        "database": "miiv", "data_path": str(tmp_path / "synthetic-source"),
        "patient_ids": {"stay_id": [11, 12]}, "verbose": False,
        "parallel_workers": 1, "concept_workers": 1,
    }
    # SOFA-1 Sepsis publishes sep3 under its public module alias.
    owner_concept = "sep3" if total == "sep3_sofa1" else total
    reference = synthetic_loader(
        concepts=[owner_concept], keep_components=True, r_compatible=False, **load_kwargs,
    ).rename(columns={owner_concept: total})
    assert total in reference, f"{total}: callback-alone oracle is missing"
    assert reference[total].notna().any(), f"{total}: oracle must not be vacuous"
    stay_level = "charttime" not in reference

    if module.startswith("sepsis3_"):
        score_module = module.replace("sepsis3_", "") + "_score"
        for dependency in (score_module, "sepsis_shared"):
            api._run_module_extraction(
                module_name=dependency, concepts=list(CONCEPT_GROUPS_INTERNAL[dependency]),
                patient_ids_filter={"stay_id": [11, 12]}, batch_size=1 if streamed else 2,
                output_dir=str(tmp_path), stream_output_batches=streamed,
                database="miiv", data_path=load_kwargs["data_path"],
            )
            manifest = tmp_path / "_manifest.json"
            assert json.loads(manifest.read_text())["errors"] == []
            manifest.rename(tmp_path / f"{dependency}.manifest.json")
        special_dir = tmp_path / "special"
        special_dir.mkdir()
        api._stream_special_extraction_batches(
            [module], "miiv", load_kwargs["data_path"], {"stay_id": [11, 12]},
            1, str(special_dir), use_sofa2=True, published_output_dir=str(tmp_path),
        )
        (special_dir / f"{total}.parquet").rename(tmp_path / f"{module}.parquet")
        special_manifest = json.loads((special_dir / "_manifest.json").read_text())
        assert special_manifest["errors"] == []
        (tmp_path / f"{module}.manifest.json").write_text(json.dumps(special_manifest))
    else:
        api._run_module_extraction(
            module_name=module, concepts=requested, database="miiv",
            data_path=load_kwargs["data_path"], patient_ids_filter={"stay_id": [11, 12]},
            batch_size=1 if streamed else 2, output_dir=str(tmp_path),
            stream_output_batches=streamed,
        )
        manifest = tmp_path / "_manifest.json"
        assert json.loads(manifest.read_text())["errors"] == []
        manifest.rename(tmp_path / f"{module}.manifest.json")

    # An identical producer duplicate exercises consolidation without inventing
    # a different callback definition for conflicting within-hour observations.
    artifact = tmp_path / f"{module}.parquet"
    table = pq.read_table(artifact)
    if backend != "arrow":
        pq.write_table(pa.concat_tables([table, table.slice(0, 1)]), artifact)
    if backend == "duckdb":
        monkeypatch.setattr(api, "_native_export_pandas_fallback_is_bounded", lambda _size: False)
    api._publish_native_export_v2(
        database="miiv", data_path=load_kwargs["data_path"], output_dir=str(tmp_path),
        modules=[module], max_patients=None, result={"modules": {module: {"errors": []}}},
    )
    native_manifest = json.loads((tmp_path / "_manifest.json").read_text())
    assert native_manifest["files"][0]["row_grain_audit"]["publication_backend"] == {
        "pandas": "pandas_bounded_row_grain_fallback",
        "arrow": "pyarrow_record_batches",
        "duckdb": "duckdb_bounded_spillable_row_grain_consolidation",
    }[backend]
    actual = pd.read_parquet(tmp_path / f"{module}.parquet")
    keys = ["stay_id", "charttime"]
    assert not actual.duplicated(keys).any()
    published = actual.set_index(keys)[total].sort_index()
    if stay_level:
        # Diagnosis indices are stay attributes broadcast by the public API
        # onto the other scores' time axis; every copy must equal the owner.
        expected = pd.Series(
            published.index.get_level_values("stay_id").map(reference.set_index("stay_id")[total]),
            index=published.index, name=total,
        )
    else:
        expected = reference.set_index(keys)[total].sort_index()
        assert not expected.index.duplicated().any()
    # Other module fields may add keys where this score is unavailable. Keep
    # nulls in the comparison, so a publisher cannot invent a normal zero there.
    index = expected.index.union(published.index)
    pd.testing.assert_series_equal(
        published.reindex(index), expected.reindex(index), check_dtype=False,
        check_exact=False, rtol=1e-6, atol=1e-6,
    )
