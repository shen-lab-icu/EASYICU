"""Windowed quantities use synthetic concept owners, never hosted examples."""

from __future__ import annotations

import re
import json
from pathlib import Path

import pandas as pd
import pytest

from easyicu.concept.schema import ConceptDefinition
from easyicu.research_agent.acquisition import catalog
from easyicu.research_agent.contracts.cohort_summary import primary_descriptive_summary_scope
from easyicu.research_agent.research_context.builder import build_research_context
from easyicu.research_agent.schema import AggregationRule, TimeWindow, VariableRole
from easyicu.webserver import research_launch_scientific as launch


WINDOWED_SUMMARY_MESSAGES = {
    "windowed_summary_statistic_unresolved": (
        "The question names a measurement but not which summary to report, such as its highest, "
        "lowest or mean value over a time window. State it, for example the highest lactate in "
        "the first 24 hours after ICU admission, then generate the plan again."
    ),
    "windowed_summary_statistic_unsupported": (
        "EasyICU can summarize a measurement over a time window only as its highest, lowest, "
        "mean, first or last value. Choose one of these, then generate the plan again."
    ),
    "windowed_summary_window_required": (
        "A summary of a measurement needs a time window, for example the first 24 hours after "
        "ICU admission. State the window in the question, then generate the plan again."
    ),
    "windowed_summary_window_unresolved": (
        "EasyICU could not read a single time window for this measurement from the question. "
        "State one window, for example the first 24 hours after ICU admission, then generate "
        "the plan again."
    ),
    "windowed_summary_window_unsupported": (
        "EasyICU can summarize a measurement only over a window that starts at ICU admission, "
        "such as the first 24 hours. Change the window, then generate the plan again."
    ),
    "windowed_summary_concept_not_continuous": (
        "This variable is not a continuous measurement, so a highest, lowest or mean value "
        "over a time window does not apply. Ask about it another way, for example whether it "
        "occurred, then generate the plan again."
    ),
    "windowed_summary_materialization_window_mismatch": (
        "The prepared data covers a different time window from the one the question states, "
        "so the summary would not match the question. No analysis was run. Prepare the data "
        "again for the stated window."
    ),
}


@pytest.fixture
def owners(monkeypatch):
    from easyicu.concept import loader
    from easyicu.webserver import study_intent
    from easyicu.research_agent.research_context import builder

    definitions = {
        "signal": ConceptDefinition.from_name_and_payload("signal", {
            "unit": "mmol/L", "description": "Synthetic measured concentration",
        }),
        "ward_group": ConceptDefinition.from_name_and_payload("ward_group", {
            "class": "fct_cncpt", "target": "id_tbl", "description": "Synthetic ward category",
        }),
        "episode_duration": ConceptDefinition.from_name_and_payload("episode_duration", {
            "unit": "hours", "target": "id_tbl", "description": "Synthetic whole episode duration",
        }),
        "episode_duration_max": ConceptDefinition.from_name_and_payload("episode_duration_max", {
            "unit": "hours", "target": "id_tbl", "description": "Synthetic owner-defined episode attribute",
        }),
    }
    menu = catalog.AvailableCatalog(source="synthetic", concepts=[
        catalog.CatalogConcept(concept_id=name, description=definition.description or "")
        for name, definition in definitions.items()
    ])
    monkeypatch.setattr(loader, "load_dictionary", lambda: definitions)
    monkeypatch.setattr(catalog, "build_database_capability_catalog", lambda db: menu)
    monkeypatch.setattr(builder, "_concept_info_for_wide_column", lambda col:
                        ({"name": col, "description": definitions[col].description}, col)
                        if col in definitions else (None, None))
    intent = study_intent.deterministic_intent

    def read(question):
        result = intent(question)
        result["slots"]["exposure"] = {
            "value": "signal", "provenance": "keyword", "evidence": "signal",
        }
        return result

    monkeypatch.setattr(study_intent, "deterministic_intent", read)
    monkeypatch.setattr(study_intent, "named_study_concepts", lambda q: [(("signal",), "signal")])
    monkeypatch.setattr(study_intent, "catalog_named_concepts", lambda q: ["signal", "ward_group"])
    monkeypatch.setattr(study_intent, "_concept_name_pattern", lambda name: rf"\b{re.escape(name)}\b")
    return definitions


@pytest.mark.parametrize("statistic", ["max", "min", "mean", "first", "last"])
def test_the_named_window_summary_binds_without_an_exposure_slot(owners, statistic):
    word = {"max": "maximum", "min": "minimum"}.get(statistic, statistic)
    result = launch._metadata_only_planning_coordinates(
        question=f"Describe {word} signal within the first 24 hours by ward_group.",
        database="synthetic",
    )
    assert result["target_outcome"] == "signal"
    assert result["endpoint"].kind == "continuous"
    assert result["endpoint"].name == f"signal_{statistic}"
    assert result["target_outcome_aggregation"] == statistic
    assert result["target_outcome_window"] == {
        "anchor": "icu_admission", "start_hours": 0.0, "end_hours": 24.0,
    }
    assert all(row["column"] != "signal" for row in result["coordinate_readings"]
               if row["role"] == "outcome")


def _planning_context(statistic, *, host_window=True):
    word = {"max": "maximum", "min": "minimum"}.get(statistic, statistic)
    question = f"Describe {word} signal within the first 24 hours by ward_group."
    result = launch._metadata_only_planning_coordinates(
        question=question, database="synthetic",
    )
    assert result["endpoint"] is not None
    frame = pd.DataFrame({
        "stay_id": pd.Series(dtype="int64"), "signal": pd.Series(dtype="float64"),
        f"signal_{statistic}": pd.Series(dtype="float64"), "ward_group": pd.Series(dtype="string"),
    })
    frame.attrs["easyicu_planning_authority"] = {
        "kind": "metadata_only_planning_catalog", "patient_rows_read": False,
    }
    context = build_research_context(
        research_question=question,
        cohort=frame, cohort_name="synthetic", database="synthetic",
        id_columns=["stay_id"],
        target_outcome=result["endpoint"].name, endpoint=result["endpoint"],
        time_windows=[TimeWindow(name="bound", anchor="icu_admission", start_hours=0, end_hours=24)],
        user_preferences={"data_constraints": json.dumps({"materialization_window": {
            "role": "outer_observation_window", "anchor": "icu_admission", "hours": 24,
        }})} if host_window else None,
    )
    return context


@pytest.mark.parametrize("statistic", ["max", "min", "mean", "first", "last"])
def test_the_planning_descriptor_names_the_exact_window_and_reduction(owners, statistic):
    context = _planning_context(statistic)
    column = f"signal_{statistic}"
    quantity = context.variable(column)
    assert quantity.role == VariableRole.OUTCOME
    assert quantity.source_concept == "signal"
    assert quantity.unit == "mmol/L"
    assert quantity.unit_normalization == f"window_numeric_{statistic}"
    assert quantity.allowed_aggregations == [AggregationRule.NONE]
    assert quantity.aggregation_default == AggregationRule.NONE
    assert quantity.analysis_window == "icu_admission[0,24]h"
    assert quantity.analysis_window_role == "outer_observation_window"
    assert context.variable("signal").role != VariableRole.OUTCOME
    assert primary_descriptive_summary_scope(
        declared_columns=["stay_id", "ward_group", column],
        variables={v.name: v for v in context.variables}, planning_context=context,
    ) == ("ward_group", column)


def test_a_row_free_proposal_can_bind_the_window_stated_in_its_question(owners):
    quantity = _planning_context("max", host_window=False).variable("signal_max")
    assert quantity.analysis_window == "icu_admission[0,24]h"


def test_a_row_free_proposal_does_not_bind_a_synthesized_default_window(owners):
    from easyicu.research_agent.schema import EndpointSpec

    frame = pd.DataFrame({"signal_max": pd.Series(dtype="float64")})
    frame.attrs["easyicu_planning_authority"] = {"kind": "metadata_only_planning_catalog", "patient_rows_read": False}
    with pytest.raises(ValueError, match="windowed_summary_window_required") as error:
        build_research_context(
            research_question="Describe the synthetic selected quantity.", cohort=frame,
            cohort_name="synthetic", database="synthetic", target_outcome="signal_max",
            endpoint=EndpointSpec(name="signal_max", kind="continuous", absence_semantics="absent_row_is_unmeasured"),
        )
    assert str(error.value).startswith(
        "A windowed summary outcome needs a time window anchored at ICU admission, "
        "and neither the host materialization window nor the question gives one."
    )


@pytest.mark.parametrize("phrase", [
    "max signal on the first ICU day", "min signal on the first-day", "首日最高 signal",
    "第一天最低 signal", "首24小时平均 signal", "最初24小时最高 signal", "前24小时末次 signal",
])
def test_common_statistic_and_first_day_spellings_reach_the_same_window(owners, phrase):
    result = launch._metadata_only_planning_coordinates(question=f"Describe {phrase} by ward_group.", database="synthetic")
    assert result["endpoint"] is not None
    assert result["target_outcome_window"]["end_hours"] == 24.0


@pytest.mark.parametrize("question,code", [
    ("Describe maximum signal by ward_group.", "windowed_summary_window_required"),
    ("Describe median signal in the first 24 hours by ward_group.", "windowed_summary_statistic_unsupported"),
    ("Describe sum signal in the first 24 hours by ward_group.", "windowed_summary_statistic_unsupported"),
    ("Describe variance signal in the first 24 hours by ward_group.", "windowed_summary_statistic_unresolved"),
    ("Describe maximum and minimum signal in the first 24 hours by ward_group.", "windowed_summary_statistic_unresolved"),
    ("Describe signal maximum and minimum in the first 24 hours by ward_group.", "windowed_summary_statistic_unresolved"),
    ("Describe maximum signal in the first 12 hours and first 24 hours by ward_group.", "windowed_summary_window_unresolved"),
    ("Describe maximum signal within 24 hours after hospital admission by ward_group.", "windowed_summary_window_unsupported"),
    ("Describe maximum signal in the first day after hospital admission by ward_group.", "windowed_summary_window_unsupported"),
    ("Describe maximum signal in the first 24 hours after hospital admission by ward_group.", "windowed_summary_window_unsupported"),
    ("Describe 住院首日最高 signal by ward_group.", "windowed_summary_window_unsupported"),
    ("Describe 入ICU前24小时最高 signal by ward_group.", "windowed_summary_window_required"),
    ("Describe maximum signal in the first 0 hours by ward_group.", "windowed_summary_window_unsupported"),
    ("Describe maximum signal in the first 24 hours after suspected infection onset by ward_group.", "windowed_summary_window_required"),
])
def test_an_unbound_summary_stops_with_its_reason(owners, question, code):
    with pytest.raises(launch.ResearchPipelineRunError) as error:
        launch._metadata_only_planning_coordinates(question=question, database="synthetic")
    assert error.value.code == code
    assert error.value.details == {"concept": "signal"}
    assert str(error.value) == WINDOWED_SUMMARY_MESSAGES[code]


def test_numeric_storage_does_not_make_a_factor_a_continuous_outcome(owners):
    owners["signal"].class_name = "fct_cncpt"
    with pytest.raises(launch.ResearchPipelineRunError) as error:
        launch._metadata_only_planning_coordinates(
            question="Describe maximum signal in the first 24 hours by ward_group.", database="synthetic",
        )
    assert error.value.code == "windowed_summary_concept_not_continuous"
    assert str(error.value) == WINDOWED_SUMMARY_MESSAGES[error.value.code]


def test_the_raw_measurement_never_substitutes_for_the_requested_quantity(owners):
    context = _planning_context("max")
    with pytest.raises(ValueError, match="typed numeric continuous outcome: signal"):
        primary_descriptive_summary_scope(
            declared_columns=["stay_id", "ward_group", "signal"],
            variables={v.name: v for v in context.variables}, planning_context=context,
        )


def _typed_export(root, definitions, *, observed=True, two_quantities=False):
    from easyicu.concept.export_metadata import build_export_file_metadata_binding
    from easyicu.concept.metadata_sidecar import (
        EXPORT_PHYSICAL_SCOPE, ColumnMetadataSidecar, write_content_addressed_sidecar,
    )
    from easyicu.research_agent.intake import export_package as intake

    root.mkdir()
    rows = pd.DataFrame({
        "stay_id": [1, 1, 1, 1, 1, 2, 2, 2, 3, 4, 4, 4, 5, 6, 6],
        "charttime": [-1., 0., 1., 5., 24., 2., 4., 7., 24., 0., 6., 12., 30., 1., 2.],
        "signal": [90., 4., 2., 8., 99., 1., 3., 7., 50., 6., 10., 5., 8., None, 12.],
    })
    if not observed:
        rows["charttime"] += 100.
    if two_quantities:
        rows["signal"] = rows["stay_id"].map(lambda stay: 2.0 if stay % 2 else 8.0)
    rows = rows.sample(frac=1, random_state=17).reset_index(drop=True)
    modules = {
        "demographics": (pd.DataFrame({"stay_id": range(1, 7), "ward_group": ["A"] * 3 + ["B"] * 3}), ["ward_group"]),
        "labs": (rows, ["signal"]),
    }
    bindings, files = [], []
    for module, (frame, concepts) in modules.items():
        name = f"{module}.parquet"
        frame.to_parquet(root / name, index=False)
        binding = build_export_file_metadata_binding(
            relative_path=name, module=module, frame=frame, concept_ids=concepts,
            database="miiv", database_class_prefixes=(), dictionary=definitions,
        )
        bindings.append(binding)
        files.append({"file": name, "module": module, "concept_ids": concepts,
                      "concepts": len(concepts), "rows": len(frame), "column_metadata_columns": concepts})
    ref = write_content_addressed_sidecar(root, ColumnMetadataSidecar(
        source_database="miiv", source_database_class_prefixes=(), scope=EXPORT_PHYSICAL_SCOPE,
        files=tuple(bindings),
    ))
    (root / intake.NATIVE_MANIFEST).write_text(json.dumps({
        "schema_version": intake.NATIVE_MANIFEST_SCHEMA_V2, "database": "miiv", "format": "parquet",
        "time_window_authority": {
            "required": True, "source": "outcome.los_icu", "bounded_stays": 6,
        },
        "concept_selection": {"mode": "explicit", "modules": {name: concepts for name, (_, concepts) in modules.items()}},
        "files": files, "feature_definitions": {"included": False}, "column_metadata": ref.to_dict(),
    }))
    return root


@pytest.mark.parametrize("statistic", ["max", "min", "mean", "first", "last"])
def test_two_observed_quantities_keep_their_continuous_scale(owners, tmp_path, statistic):
    from easyicu.research_agent.cohort.materializer import materialize_to_parquet

    paths = materialize_to_parquet(
        tmp_path / "materialized", data_path=_typed_export(tmp_path / "export", owners, two_quantities=True),
        database="miiv", static_concepts=["ward_group"], feature_concepts=["signal"], outcome_concepts=[],
        last_value_concepts=["signal"],
    )
    column = f"signal_{statistic}"
    context = build_research_context(
        research_question=_planning_context(statistic).research_question, cohort=paths["parquet"],
        cohort_name="synthetic", database="miiv", target_outcome=column, endpoint=_planning_context(statistic).endpoint,
    )
    assert context.variable(column).observed_domain["levels"] == [2.0, 8.0]
    assert primary_descriptive_summary_scope(
        declared_columns=["ward_group", column], variables={v.name: v for v in context.variables},
        planning_context=context,
    ) == ("ward_group", column)


@pytest.mark.parametrize("change", [
    {"role": VariableRole.LAB}, {"unit_normalization": None}, {"analysis_window": None},
    {"analysis_window_role": None}, {"source_concept": None}, {"unit": None},
    {"unit": "category"}, {"unit": "boolean"}, {"dtype": "object"},
    {"source_concept": "ward_group", "unit": "category"}, {"is_ordinal": True},
])
def test_two_observed_values_never_supply_missing_quantity_authority(owners, change):
    from easyicu.research_agent.contracts.table_one_semantics import table_one_variable_kind

    context = _planning_context("max")
    variables = {v.name: v for v in context.variables}
    variables["signal_max"] = variables["signal_max"].model_copy(update={
        "observed_domain": {"levels": [2., 8.]}, **change,
    })
    assert table_one_variable_kind(variables["signal_max"], [2., 8.]) != "continuous"
    with pytest.raises(ValueError, match="typed numeric continuous outcome"):
        primary_descriptive_summary_scope(
            declared_columns=["ward_group", "signal_max"], variables=variables,
            grouping="ward_group", planning_context=context,
        )


def test_a_declared_closed_domain_is_never_overridden_by_a_numeric_transform(owners):
    context = _planning_context("max")
    variables = {v.name: v for v in context.variables}
    variables["signal_max"] = variables["signal_max"].model_copy(update={"ordinal_levels": [2, 8]})
    with pytest.raises(ValueError, match="typed numeric continuous outcome"):
        primary_descriptive_summary_scope(
            declared_columns=["ward_group", "signal_max"], variables=variables,
            grouping="ward_group", planning_context=context,
        )


@pytest.mark.parametrize("statistic,expected", [
    ("max", [8., 7., None, 10., None, 12.]),
    ("min", [2., 1., None, 5., None, 12.]),
    ("mean", [14/3, 11/3, None, 7., None, 12.]),
    ("first", [4., 1., None, 6., None, 12.]),
    ("last", [8., 7., None, 5., None, 12.]),
])
def test_the_host_materializes_and_summarizes_the_selected_quantity_by_group(
    owners, monkeypatch, tmp_path, statistic, expected,
):
    from easyicu.research_agent.cohort.materializer import materialize_to_parquet
    from easyicu.research_agent.execution.runners.cohort_summary_executor import run_cohort_summary_from_env
    from easyicu.research_agent.contracts.cohort_summary import descriptive_summary_result_receipt_valid

    paths = materialize_to_parquet(
        tmp_path / "materialized", data_path=_typed_export(tmp_path / "export", owners), database="miiv",
        static_concepts=["ward_group"], feature_concepts=["signal"], outcome_concepts=[],
        last_value_concepts=["signal"] if statistic == "last" else [],
    )
    column = f"signal_{statistic}"
    frame = pd.read_parquet(paths["parquet"])
    if statistic != "last":
        assert "signal_last" not in frame.columns
    pd.testing.assert_series_equal(frame[column], pd.Series(expected, name=column), check_dtype=False)
    planning = _planning_context(statistic)
    context = build_research_context(
        research_question=planning.research_question, cohort=paths["parquet"],
        cohort_name="synthetic", database="miiv", id_columns=["stay_id"],
        target_outcome=column, endpoint=planning.endpoint,
    )
    quantity = context.variable(column)
    for field in ("unit", "source_concept", "unit_normalization", "analysis_window", "analysis_window_role"):
        assert getattr(quantity, field) == getattr(planning.variable(column), field)
    (tmp_path / "research_context.json").write_text(context.model_dump_json())
    monkeypatch.setenv("EASYICU_RUN_DIR", str(tmp_path))
    monkeypatch.setenv("STEP_OUT_DIR", str(tmp_path / "out"))
    result = run_cohort_summary_from_env(
        declared_columns=["stay_id", "ward_group", column], typed_cohort_input="artifact:analysis_cohort",
        frame=frame, cohort_path=paths["parquet"], primary_summary=True,
    )
    assert descriptive_summary_result_receipt_valid(result)
    assert result["cohort_n"] == 6
    assert result["declared_columns"] == [column]
    table = pd.read_csv(tmp_path / "out/cohort_summary.csv")
    assert set(table["variable"]) == {column, "__cohort__"}
    assert set(table["group"]) == {"Overall", "A", "B"}
    overall = table[(table["row_role"] == "overall") & (table["variable"] == column)].set_index("statistic")
    assert overall.loc["missing_n", "value"] == 2
    assert overall.loc["nonmissing_n", "value"] == 4
    for group in ("A", "B"):
        missing = table[(table["group"] == group) & (table["statistic"] == "missing_n")].iloc[0]
        assert missing["value"] == 1 and missing["denominator"] == 3


def test_a_window_with_no_measurements_retains_quantities_but_stops_analysis(owners, tmp_path, monkeypatch):
    from easyicu.research_agent.cohort.materializer import materialize_to_parquet
    from easyicu.research_agent.execution.runners.cohort_summary_executor import run_cohort_summary_from_env
    from easyicu.research_agent.methods.descriptive_inputs import RequestedConceptNoDataError

    paths = materialize_to_parquet(
        tmp_path / "materialized", data_path=_typed_export(tmp_path / "export", owners, observed=False),
        database="miiv", static_concepts=["ward_group"], feature_concepts=["signal"], outcome_concepts=[],
        last_value_concepts=["signal"],
    )
    frame = pd.read_parquet(paths["parquet"])
    assert len(frame) == 6
    for statistic in ("max", "min", "mean", "first", "last"):
        assert frame[f"signal_{statistic}"].isna().all()
    assert frame["signal_n"].eq(0).all()
    assert frame["signal_measured"].eq(0).all()
    assert frame.groupby("ward_group").size().to_dict() == {"A": 3, "B": 3}
    assert json.loads(paths["provenance"].read_text())["last_value_concepts"] == ["signal"]
    proposal = _planning_context("max")
    context = build_research_context(
        research_question=proposal.research_question, cohort=paths["parquet"], cohort_name="synthetic", database="miiv",
        target_outcome="signal_max", endpoint=proposal.endpoint,
    )
    (tmp_path / "research_context.json").write_text(context.model_dump_json())
    monkeypatch.setenv("EASYICU_RUN_DIR", str(tmp_path))
    monkeypatch.setenv("STEP_OUT_DIR", str(tmp_path / "out"))
    with pytest.raises(RequestedConceptNoDataError) as stopped:
        run_cohort_summary_from_env(
            declared_columns=["stay_id", "ward_group", "signal_max"], typed_cohort_input="artifact:analysis_cohort",
            frame=frame, cohort_path=paths["parquet"], primary_summary=True,
        )
    assert stopped.value.reason_code == "requested_concept_no_data"
    assert stopped.value.columns == ("signal_max",)
    assert not (tmp_path / "out/cohort_summary.csv").exists()
    assert not (tmp_path / "out/step_summary.json").exists()


def test_an_optional_last_summary_must_name_a_materialized_feature(owners, tmp_path):
    from easyicu.research_agent.cohort.materializer import materialize_to_parquet, MaterializedMetadataError

    with pytest.raises(MaterializedMetadataError, match="windowed_summary_last_concept_unavailable") as error:
        materialize_to_parquet(
            tmp_path / "materialized", data_path=_typed_export(tmp_path / "export", owners), database="miiv",
            static_concepts=["ward_group"], feature_concepts=["signal"], outcome_concepts=[],
            last_value_concepts=["ward_group"],
        )
    assert str(error.value).startswith(
        "A last-value summary was requested for a concept that is not among the extracted "
        "time-varying features."
    )


def test_acquisition_requests_the_optional_last_reduction(owners, tmp_path):
    from easyicu.research_agent.acquisition.foundation import acquire_universe_for_question
    from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient

    result = acquire_universe_for_question(
        export_dir=_typed_export(tmp_path / "export", owners), output_dir=tmp_path / "materialized",
        question="Describe last signal in the first 24 hours.",
        llm=ScriptedMockLLMClient([json.dumps({"selected_concepts": ["signal"]})]),
        target_outcome="signal", target_outcome_aggregation="last",
        static_concepts=["ward_group"], required_feature_concepts=["signal"], outcome_concepts=[],
        require_outcome=False, emit_trajectory=False,
    )
    assert not result.blocked
    assert result.endpoint.name == result.analysis_columns["signal"] == "signal_last"
    frame = pd.read_parquet(result.universe_path)
    assert frame["signal_last"].tolist() == pytest.approx([8., 7., float("nan"), 5., float("nan"), 12.], nan_ok=True)


@pytest.mark.parametrize("statistic,reason", [
    ("max", "windowed_summary_column_unavailable"), ("median", "windowed_summary_statistic_unsupported"),
])
def test_an_absent_requested_summary_never_falls_back_to_a_raw_column(owners, monkeypatch, tmp_path, statistic, reason):
    from easyicu.research_agent.acquisition import foundation
    from easyicu.research_agent.cohort import materializer
    from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient

    monkeypatch.setattr(foundation, "build_available_catalog", lambda path: catalog.build_database_capability_catalog("synthetic"))
    parquet = tmp_path / "synthetic.parquet"
    pd.DataFrame({"stay_id": [1, 2], "signal": [123., 456.]}).to_parquet(parquet)
    provenance = tmp_path / "synthetic.json"
    provenance.write_text(json.dumps({"columns": ["stay_id", "signal"]}))
    monkeypatch.setattr(materializer, "materialize_to_parquet", lambda **kwargs:
                        {"parquet": parquet, "provenance": provenance})
    with pytest.raises(materializer.MaterializedMetadataError, match=reason) as error:
        foundation.acquire_universe_for_question(
            export_dir=tmp_path, output_dir=tmp_path, question="Describe maximum signal within the first 24 hours.",
            llm=ScriptedMockLLMClient([json.dumps({"selected_concepts": ["signal"]})]), target_outcome="signal",
            target_outcome_aggregation=statistic, outcome_concepts=[], require_outcome=False,
            required_feature_concepts=["signal"], emit_trajectory=False,
        )
    expected = (
        "The summary column is not in the prepared data: signal_max" if statistic == "max"
        else "The requested summary statistic is not one the host materializes (max, min, mean, first, last)."
    )
    assert str(error.value).startswith(expected)


def _compile_summary(context):
    from easyicu.research_agent.planning.progressive_compiler import compile_progressive_plan
    from easyicu.research_agent.planning.progressive_contract import ProgressivePlanSkeleton

    column = context.target_outcome
    skeleton = ProgressivePlanSkeleton.model_validate({
        "analysis_type": "descriptive_epidemiology",
        "cohort": {"name": "synthetic", "selection_mode": "all_input_rows"},
        "display_labels": [{"key": name, "value": name} for name in ("signal", column, "ward_group")],
        "steps": [
            {"step_id": "cohort", "module_id": "cohort_definition", "planned_analysis_role": "auxiliary",
             "objective": "Account for the declared synthetic stay denominator.", "raw_inputs": ["stay_id"]},
            {"step_id": "summary", "module_id": "custom_analysis", "planned_analysis_role": "primary",
             "scientific_action_id": "descriptive.descriptive_summary", "custom_method": "empirical_summary",
             "objective": "Describe the selected synthetic quantity within each ward.", "depends_on": ["cohort"],
             "raw_inputs": ["stay_id", "ward_group", "signal"],
             "product_inputs": [{"producer_step_id": "cohort", "product_id": "artifact:analysis_cohort"}],
             "outputs": [{"product_id": product, "semantic_role": "custom"}
                         for product in ("table:cohort_summary", "table:distribution_summary")]},
        ],
        "rationale": "Describe the requested synthetic quantity with its original denominator.",
    })
    plan, _ = compile_progressive_plan(skeleton=skeleton, context=context)
    return plan


@pytest.mark.parametrize("statistic", ["max", "min", "mean", "first", "last"])
def test_compilation_gives_the_primary_the_host_bound_quantity(owners, statistic):
    from easyicu.research_agent.planning.capability_registry import assess_scientific_capability
    from easyicu.research_agent.planning.scientific_review import build_plan_scientific_review

    context = _planning_context(statistic)
    column = context.target_outcome
    plan = _compile_summary(context)
    step = next(step for step in plan.steps if step.planned_analysis_role == "primary")
    assert column in step.inputs
    assert "signal" not in step.inputs
    assert primary_descriptive_summary_scope(
        declared_columns=[name for name in step.inputs if ":" not in name],
        variables={v.name: v for v in context.variables}, planning_context=context,
    ) == ("ward_group", column)
    assessment = assess_scientific_capability(analysis_type=plan.analysis_type, plan=plan, context=context)
    assert assessment.input_contract_resolved
    assert assessment.claim_ceiling == "reportable"
    review = build_plan_scientific_review(context=context, plan=plan, require_reportable_capability=True)
    assert "SCIENTIFIC_CAPABILITY_NOT_REPORTABLE" not in {f.code for f in review.findings}


@pytest.mark.parametrize("changes", [
    {"role": VariableRole.LAB}, {"analysis_window": None},
    {"unit_normalization": None}, {"source_concept": "another_signal"},
])
def test_compilation_cannot_promote_an_unbound_measurement(owners, changes):
    context = _planning_context("max")
    context = context.model_copy(update={"variables": [
        item.model_copy(update=changes) if item.name == context.target_outcome else item for item in context.variables
    ]})
    plan = _compile_summary(context)
    step = next(step for step in plan.steps if step.planned_analysis_role == "primary")
    assert "signal_max" not in step.inputs
    with pytest.raises(ValueError, match="typed numeric continuous outcome: signal"):
        primary_descriptive_summary_scope(
            declared_columns=[name for name in step.inputs if ":" not in name],
            variables={v.name: v for v in context.variables}, planning_context=context,
        )


@pytest.mark.parametrize("outcome", ["episode_duration", "episode_duration_max"])
@pytest.mark.parametrize("statistic", ["", "mean ", "maximum ", "minimum "])
def test_a_whole_episode_quantity_keeps_its_original_outcome_binding(owners, outcome, statistic):
    coordinates = launch._metadata_only_planning_coordinates(
        question=f"Describe {statistic}{outcome} by ward_group.", database="synthetic", configured_outcome=outcome,
    )
    assert coordinates["endpoint"].name == outcome
    assert coordinates["target_outcome_aggregation"] is None
    assert coordinates["target_outcome_window"] is None
    frame = pd.DataFrame({name: pd.Series(dtype="float64") for name in ("ward_group", outcome)})
    frame.attrs["easyicu_planning_authority"] = {"kind": "metadata_only_planning_catalog", "patient_rows_read": False}
    context = build_research_context(
        research_question=f"Describe {outcome} by ward_group.", cohort=frame, cohort_name="synthetic",
        database="synthetic", target_outcome=outcome, endpoint=coordinates["endpoint"],
    )
    quantity = context.variable(outcome)
    assert quantity.role == VariableRole.OUTCOME
    assert quantity.analysis_window is None
    assert quantity.unit_normalization is None
@pytest.mark.parametrize("code,message", WINDOWED_SUMMARY_MESSAGES.items())
def test_every_summary_refusal_reaches_submission_with_the_reviewed_sentence(code, message):
    from easyicu.webserver.research_pipeline_run_errors import WINDOWED_SUMMARY_STOP_MESSAGES
    from easyicu.webserver.research_run_submission import ResearchRunSubmissionError

    error = launch.ResearchPipelineRunError(code, WINDOWED_SUMMARY_STOP_MESSAGES[code])
    assert str(error) == message
    # The submission boundary forwards launch errors verbatim; its code-only
    # fallback must give the same next step to an adapter missing the message.
    for detail in ({"error": code, "message": str(error)}, {"error": code}):
        submitted = ResearchRunSubmissionError(detail)
        assert submitted.code == code
        assert submitted.summary == message


def test_no_lead_copy_placeholder_remains_in_source():
    source = Path(__file__).resolve().parents[2] / "src"
    placeholders = [
        str(path.relative_to(source))
        for path in source.rglob("*")
        if path.suffix in {".py", ".js", ".mjs", ".json", ".md", ".txt"}
        and "meta_generalization" not in path.parts
        and b"TODO(lead copy)" in path.read_bytes()
    ]
    assert not placeholders, placeholders


