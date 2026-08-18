"""Tests for the L2 data-foundation agent (concept selection + acquisition)."""

from __future__ import annotations

import json

import pytest

import easyicu.research_agent.acquisition.foundation as df_mod
from easyicu.research_agent.acquisition.catalog import (
    AvailableCatalog,
    CatalogConcept,
    assess_coverage,
)
from easyicu.research_agent.acquisition.foundation import (
    DataFoundationAgent,
    _extract_json,
    acquire_universe_for_question,
)
from easyicu.research_agent.acquisition.patient_grouping import (
    PatientGroupingBinding,
)
from easyicu.research_agent.intake.materialized_metadata import (
    MaterializedMetadataError,
)
from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient


def _stub(response: str) -> ScriptedMockLLMClient:
    return ScriptedMockLLMClient([response])


def _catalog(*ids: str) -> AvailableCatalog:
    return AvailableCatalog(
        source="mem", concepts=[CatalogConcept(concept_id=i) for i in ids]
    )


def test_extract_json_handles_fenced_and_bare():
    assert _extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert _extract_json('prose {"a": 2} trailing') == {"a": 2}
    assert _extract_json("not json at all") is None


def test_agent_selects_concepts_and_reports_coverage():
    llm = _stub(
        '{"selected_concepts": ["sep3_sofa2", "sofa2", "death", "made_up"], '
        '"inclusion_exclusion": ["adults", "ICU LoS >= 1 day"], '
        '"rationale": "sepsis exposure + severity"}'
    )
    cat = _catalog("sep3_sofa2", "sofa2", "death", "age", "los_icu")
    sel = DataFoundationAgent(llm).select_concepts(
        question="sepsis-3 vs mortality", catalog=cat, target_outcome="death"
    )
    assert "sep3_sofa2" in sel.selected_concepts
    assert sel.inclusion_exclusion == ["adults", "ICU LoS >= 1 day"]
    # coverage flags the hallucinated concept, but not the real ones
    assert sel.coverage is not None
    assert sel.coverage.missing == ["made_up"]
    assert "sofa2" in sel.coverage.available


def test_agent_empty_or_garbage_response_is_safe():
    sel = DataFoundationAgent(_stub("garbage")).select_concepts(
        question="q", catalog=_catalog("death")
    )
    assert sel.selected_concepts == []
    assert sel.coverage is not None
    assert sel.selection_succeeded is False
    assert "parseable JSON object" in sel.selection_error
    assert sel.to_dict()["selection_succeeded"] is False


def test_acquire_blocks_unparseable_selection_before_materialization(monkeypatch):
    called = {"materialize": False}

    monkeypatch.setattr(
        df_mod,
        "build_available_catalog",
        lambda _d: _catalog("death", "age"),
    )
    import easyicu.research_agent.cohort.materializer as cm

    def _fake_materialize(**_kwargs):
        called["materialize"] = True
        return {"parquet": "u.parquet", "provenance": "u.json"}

    monkeypatch.setattr(cm, "materialize_to_parquet", _fake_materialize)

    result = acquire_universe_for_question(
        export_dir="/nonexistent",
        question="q",
        llm=_stub("garbage"),
        output_dir="/tmp/x",
        target_outcome="death",
        outcome_concepts=["death"],
        static_concepts=["age"],
    )

    assert result.blocked is True
    assert result.selection.selection_succeeded is False
    assert "selection failed" in result.note.lower()
    assert called["materialize"] is False


def test_acquire_blocks_when_outcome_missing(monkeypatch):
    # If the outcome concept itself is not in the data, hard-block with advice.
    called = {"materialize": False}

    def _fake_materialize(**kwargs):
        called["materialize"] = True
        return {"parquet": "x.parquet", "provenance": "x.json"}

    monkeypatch.setattr(
        df_mod, "build_available_catalog", lambda _d: _catalog("lact", "sofa2")
    )
    # patch the lazily-imported materializer symbol
    import easyicu.research_agent.cohort.materializer as cm

    monkeypatch.setattr(cm, "materialize_to_parquet", _fake_materialize)

    res = acquire_universe_for_question(
        export_dir="/nonexistent",
        question="q",
        llm=_stub('{"selected_concepts": ["lact"]}'),
        output_dir="/tmp/x",
        target_outcome="death",
        outcome_concepts=["death"],
    )
    assert res.blocked
    assert not called["materialize"]
    assert "outcome" in res.note.lower()


def test_coverage_of_an_empty_request_reads_the_same_as_full_coverage():
    """`sufficient` measures the request, so naming nothing always passes.

    This pins the trap rather than the wish: ``missing`` is derived from
    ``requested``, so an empty request and a fully covered one are the same
    shape. Any caller reading this property as "the data can answer the
    question" is reading something it does not measure.
    """
    empty = assess_coverage([], _catalog("lact"))
    covered = assess_coverage(["lact"], _catalog("lact"))

    assert empty.sufficient is True
    assert empty.missing == []
    assert (empty.sufficient, empty.missing) == (covered.sufficient, covered.missing)


def test_acquire_blocks_when_an_outcome_is_required_but_none_is_named(monkeypatch):
    """A caller that required an outcome may not be satisfied by silence.

    ``require_outcome=True`` asserts this study has an outcome. Because an
    empty ``outcome_concepts`` reads ``sufficient`` (see the test above),
    checking coverage alone would let it through and materialise a cohort with
    no outcome column while reporting success.
    """
    called = {"materialize": False}

    def _fake_materialize(**_kwargs):
        called["materialize"] = True
        return {"parquet": "x.parquet", "provenance": "x.json"}

    monkeypatch.setattr(
        df_mod, "build_available_catalog", lambda _d: _catalog("lact", "sofa2")
    )
    import easyicu.research_agent.cohort.materializer as cm

    monkeypatch.setattr(cm, "materialize_to_parquet", _fake_materialize)

    res = acquire_universe_for_question(
        export_dir="/nonexistent",
        question="q",
        llm=_stub('{"selected_concepts": ["lact"]}'),
        output_dir="/tmp/x",
        target_outcome="death",
        outcome_concepts=[],
        require_outcome=True,
    )

    assert res.blocked
    assert not called["materialize"]
    assert "named no outcome concept" in res.note


def test_acquire_proceeds_on_available_subset_when_outcome_present(monkeypatch):
    captured = {}

    def _fake_materialize(**kwargs):
        captured.update(kwargs)
        return {"parquet": "u.parquet", "provenance": "u.json"}

    monkeypatch.setattr(
        df_mod,
        "build_available_catalog",
        lambda _d: _catalog("sofa2", "lact", "death", "age", "sex", "los_icu"),
    )
    import easyicu.research_agent.cohort.materializer as cm

    monkeypatch.setattr(cm, "materialize_to_parquet", _fake_materialize)

    res = acquire_universe_for_question(
        export_dir="/nonexistent",
        question="q",
        llm=_stub('{"selected_concepts": ["sofa2", "lact", "made_up", "death"]}'),
        output_dir="/tmp/x",
        target_outcome="death",
        outcome_concepts=["death"],
        static_concepts=["age", "sex", "los_icu"],
    )
    assert not res.blocked
    # outcome/demographics are passed via dedicated args, not feature_concepts;
    # the hallucinated concept is dropped; only available features remain.
    assert set(captured["feature_concepts"]) == {"sofa2", "lact"}
    assert res.coverage.missing == ["made_up"]
    assert "re-extract" in res.note.lower()


def test_host_exact_acquisition_does_not_widen_the_configured_universe(
    monkeypatch,
):
    captured = {}
    llm = _stub(
        '{"selected_concepts": ["age", "icu_readmission", "sep3_sofa2", "death"]}'
    )
    monkeypatch.setattr(
        df_mod,
        "build_available_catalog",
        lambda _d: _catalog(
            "age", "sex", "icu_readmission", "sep3_sofa2", "death"
        ),
    )
    import easyicu.research_agent.cohort.materializer as cm

    monkeypatch.setattr(
        cm,
        "materialize_to_parquet",
        lambda **kwargs: captured.update(kwargs)
        or {"parquet": "u.parquet", "provenance": "u.json"},
    )

    result = acquire_universe_for_question(
        export_dir="/nonexistent",
        question="How common is the configured phenotype?",
        llm=llm,
        output_dir="/tmp/x",
        target_outcome="death",
        outcome_concepts=["death"],
        required_feature_concepts=["sep3_sofa2"],
        static_concepts=["age", "sex"],
        concept_selection_authority="host_exact",
    )

    assert result.blocked is False
    assert llm.calls == []
    assert result.selection.selection_authority == "host_exact"
    assert result.selection.selected_concepts == [
        "age",
        "sex",
        "sep3_sofa2",
        "death",
    ]
    assert captured["feature_concepts"] == ["sep3_sofa2"]
    assert "icu_readmission" not in captured["feature_concepts"]
    assert result.selection_usage is None
    assert result.selection_model is None


def test_legacy_acquisition_declares_sparse_event_features_before_materialization(
    monkeypatch,
):
    captured = {}
    catalog = AvailableCatalog(
        source="legacy",
        concepts=[
            CatalogConcept(
                concept_id="sep3_sofa2",
                file_name="sepsis3_sofa2.parquet",
                column_role="event_status",
            ),
            CatalogConcept(
                concept_id="death",
                file_name="outcome.parquet",
                column_role="event_status",
            ),
            CatalogConcept(concept_id="age", file_name="demographics.parquet"),
        ],
    )
    monkeypatch.setattr(df_mod, "build_available_catalog", lambda _d: catalog)
    import easyicu.research_agent.cohort.materializer as cm

    monkeypatch.setattr(
        cm,
        "materialize_to_parquet",
        lambda **kwargs: captured.update(kwargs)
        or {"parquet": "u.parquet", "provenance": "u.json"},
    )

    result = acquire_universe_for_question(
        export_dir="/nonexistent",
        question="q",
        llm=_stub('{"selected_concepts": ["sep3_sofa2", "death"]}'),
        output_dir="/tmp/x",
        target_outcome="death",
        outcome_concepts=["death"],
        required_feature_concepts=["sep3_sofa2"],
        static_concepts=["age"],
    )

    assert result.blocked is False
    assert captured["feature_concepts"] == ["sep3_sofa2"]
    assert captured["outcome_concepts"] == ["death"]
    assert captured["positive_only_event_concepts"] == ["sep3_sofa2"]


@pytest.mark.parametrize("typed_metadata", [False, True])
def test_acquisition_binds_event_endpoint_and_exposure_to_materialized_columns(
    monkeypatch, tmp_path, typed_metadata
):
    catalog = AvailableCatalog(
        source="legacy",
        concepts=[
            CatalogConcept(
                concept_id="sep3_sofa2",
                file_name="sepsis3_sofa2.parquet",
                column_role="event_status",
                typed_metadata=typed_metadata,
            ),
            CatalogConcept(
                concept_id="death",
                file_name="outcome.parquet",
                column_role="event_status",
                typed_metadata=typed_metadata,
            ),
            CatalogConcept(
                concept_id="age",
                file_name="demographics.parquet",
                typed_metadata=typed_metadata,
            ),
        ],
    )
    monkeypatch.setattr(df_mod, "build_available_catalog", lambda _d: catalog)
    import easyicu.research_agent.cohort.materializer as cm

    captured = {}

    def materialize(**kwargs):
        captured.update(kwargs)
        parquet = tmp_path / "universe.parquet"
        provenance = tmp_path / "universe_provenance.json"
        parquet.write_bytes(b"legacy-parquet-placeholder")
        provenance.write_text(
            json.dumps(
                {
                    "columns": [
                        "stay_id",
                        "age",
                        "sep3_sofa2_max",
                        "death",
                    ]
                }
            ),
            encoding="utf-8",
        )
        return {"parquet": str(parquet), "provenance": str(provenance)}

    monkeypatch.setattr(cm, "materialize_to_parquet", materialize)

    result = acquire_universe_for_question(
        export_dir="/nonexistent",
        question="q",
        llm=_stub('{"selected_concepts": ["sep3_sofa2", "death"]}'),
        output_dir=tmp_path,
        target_outcome="death",
        primary_exposure_concept="sep3_sofa2",
        outcome_concepts=["death"],
        required_feature_concepts=["sep3_sofa2"],
        static_concepts=["age"],
    )

    assert result.analysis_columns == {
        "age": "age",
        "death": "death",
        "sep3_sofa2": "sep3_sofa2_max",
    }
    assert captured["positive_only_event_concepts"] == (
        [] if typed_metadata else ["sep3_sofa2"]
    )
    assert result.endpoint is not None
    assert result.endpoint.model_dump(mode="json") == {
        "name": "death",
        "kind": "binary",
        "absence_semantics": "no_absent_rows",
        "levels": [0, 1],
        "event_column": None,
        "time_column": None,
        "time_origin": None,
        "censoring_rule": None,
    }


def test_acquire_limits_agent_catalog_to_configured_modules(monkeypatch):
    captured = {}
    catalog = AvailableCatalog(
        source="mem",
        concepts=[
            CatalogConcept(
                concept_id="age",
                file_name="demographics.parquet",
            ),
            CatalogConcept(
                concept_id="death",
                file_name="outcome.parquet",
            ),
            CatalogConcept(
                concept_id="lact",
                file_name="blood_gas.parquet",
            ),
        ],
    )

    monkeypatch.setattr(df_mod, "build_available_catalog", lambda _d: catalog)
    original_select = DataFoundationAgent.select_concepts

    def capture_catalog(self, *, question, catalog, target_outcome):
        captured["catalog_ids"] = catalog.ids()
        return original_select(
            self,
            question=question,
            catalog=catalog,
            target_outcome=target_outcome,
        )

    monkeypatch.setattr(DataFoundationAgent, "select_concepts", capture_catalog)
    import easyicu.research_agent.cohort.materializer as cm

    monkeypatch.setattr(
        cm,
        "materialize_to_parquet",
        lambda **_kwargs: {"parquet": "u.parquet", "provenance": "u.json"},
    )

    result = acquire_universe_for_question(
        export_dir="/nonexistent",
        question="q",
        llm=_stub('{"selected_concepts": ["death"]}'),
        output_dir="/tmp/x",
        target_outcome="death",
        outcome_concepts=["death"],
        static_concepts=["age"],
        allowed_modules=["demographics", "outcome"],
    )

    assert result.blocked is False
    assert captured["catalog_ids"] == ["age", "death"]


def test_outcome_free_acquisition_materialises_required_trajectory_concept(
    monkeypatch,
):
    captured = {}

    monkeypatch.setattr(
        df_mod,
        "build_available_catalog",
        lambda _d: _catalog("sofa2", "age"),
    )
    import easyicu.research_agent.cohort.materializer as cm

    monkeypatch.setattr(
        cm,
        "materialize_to_parquet",
        lambda **kwargs: captured.update(kwargs)
        or {"parquet": "u.parquet", "provenance": "u.json"},
    )

    result = acquire_universe_for_question(
        export_dir="/nonexistent",
        question="Are SOFA-2 trajectories reproducible?",
        llm=_stub('{"selected_concepts": []}'),
        output_dir="/tmp/x",
        target_outcome=None,
        outcome_concepts=(),
        required_feature_concepts=("sofa2",),
        require_outcome=False,
    )

    assert result.blocked is False
    assert captured["feature_concepts"] == ["sofa2"]
    assert captured["outcome_concepts"] == []
    assert captured["trajectory_concepts"] == ["sofa2"]


def test_required_trajectory_concept_missing_blocks_before_materialisation(
    monkeypatch,
):
    called = {"materialize": False}
    monkeypatch.setattr(
        df_mod,
        "build_available_catalog",
        lambda _d: _catalog("age"),
    )
    import easyicu.research_agent.cohort.materializer as cm

    def _fake_materialize(**_kwargs):
        called["materialize"] = True
        return {"parquet": "u.parquet", "provenance": "u.json"}

    monkeypatch.setattr(cm, "materialize_to_parquet", _fake_materialize)

    result = acquire_universe_for_question(
        export_dir="/nonexistent",
        question="Are SOFA-2 trajectories reproducible?",
        llm=_stub('{"selected_concepts": []}'),
        output_dir="/tmp/x",
        target_outcome=None,
        outcome_concepts=(),
        required_feature_concepts=("sofa2",),
        require_outcome=False,
    )

    assert result.blocked is True
    assert called["materialize"] is False
    assert "sofa2" in result.note


def test_acquire_preserves_legacy_trajectory_without_typed_loader(
    monkeypatch, tmp_path
):
    """A legacy export trajectory is path/provenance bound, not typed-authority bound."""

    universe = tmp_path / "universe.parquet"
    provenance = tmp_path / "universe_provenance.json"
    trajectory = tmp_path / "universe_trajectory.parquet"
    trajectory_provenance = tmp_path / "universe_trajectory_provenance.json"

    monkeypatch.setattr(
        df_mod,
        "build_available_catalog",
        lambda _d: _catalog("sofa2", "death", "age"),
    )
    monkeypatch.setattr(
        df_mod, "load_verified_materialized_cohort_authority", lambda _path: None
    )

    def _typed_loader_must_not_run(*_args, **_kwargs):
        raise AssertionError("legacy trajectory was sent through the typed loader")

    monkeypatch.setattr(
        df_mod,
        "load_verified_materialized_trajectory_authority",
        _typed_loader_must_not_run,
    )
    import easyicu.research_agent.cohort.materializer as cm

    monkeypatch.setattr(
        cm,
        "materialize_to_parquet",
        lambda **_kwargs: {
            "parquet": universe,
            "provenance": provenance,
            "trajectory": trajectory,
            "trajectory_provenance": trajectory_provenance,
        },
    )

    result = acquire_universe_for_question(
        export_dir=tmp_path,
        question="q",
        llm=_stub('{"selected_concepts": ["sofa2", "death"]}'),
        output_dir=tmp_path,
        target_outcome="death",
        outcome_concepts=["death"],
        static_concepts=["age"],
    )

    assert result.blocked is False
    assert result.trajectory_path == trajectory
    assert result.trajectory_provenance_path == trajectory_provenance
    assert result.trajectory_authority_path is None
    assert result.trajectory_authority_ref is None


def test_acquire_captures_selection_token_usage_and_cost(monkeypatch):
    # A metered client exposes last_usage + model; the selection's token cost
    # is recorded on the result (it runs as a pre-sandbox stage).
    llm = _stub('{"selected_concepts": ["sofa2", "death"]}')
    llm._model = "deepseek-chat"
    llm.last_usage = {
        "prompt_tokens": 1000,
        "completion_tokens": 200,
        "total_tokens": 1200,
    }

    monkeypatch.setattr(
        df_mod,
        "build_available_catalog",
        lambda _d: _catalog("sofa2", "death", "age", "sex", "los_icu"),
    )
    import easyicu.research_agent.cohort.materializer as cm

    monkeypatch.setattr(
        cm,
        "materialize_to_parquet",
        lambda **kw: {"parquet": "u.parquet", "provenance": "u.json"},
    )

    res = acquire_universe_for_question(
        export_dir="/nonexistent",
        question="q",
        llm=llm,
        output_dir="/tmp/x",
        target_outcome="death",
        outcome_concepts=["death"],
    )
    assert res.selection_model == "deepseek-chat"
    assert res.selection_usage == {
        "prompt_tokens": 1000,
        "completion_tokens": 200,
        "total_tokens": 1200,
    }
    # deepseek-chat priced at (0.27, 1.10)/1M -> 0.001*0.27 + 0.0002*1.10
    assert res.selection_cost_usd is not None and res.selection_cost_usd > 0


def test_acquisition_requires_caller_owned_outcome_and_has_no_static_science_default():
    import inspect

    parameters = inspect.signature(acquire_universe_for_question).parameters
    assert parameters["target_outcome"].default is inspect.Parameter.empty
    assert parameters["outcome_concepts"].default is inspect.Parameter.empty
    assert parameters["static_concepts"].default == ()


def test_acquisition_forwards_verified_patient_grouping_to_materializer(
    monkeypatch, tmp_path
):
    mapping = tmp_path / "mapping.parquet"
    mapping.write_bytes(b"mapping")
    grouping = PatientGroupingBinding(
        mapping_path=mapping,
        mapping_sha256="a" * 64,
        mapping_stay_column="stay_id",
        mapping_patient_column="patient_key",
        authority_coordinates={"authority_ref": "owner/bridge/v1"},
    )
    monkeypatch.setattr(
        df_mod,
        "build_available_catalog",
        lambda _d: _catalog("death", "age"),
    )
    import easyicu.research_agent.cohort.materializer as cm

    captured = {}
    universe = tmp_path / "universe.parquet"
    provenance = tmp_path / "universe_provenance.json"
    universe.write_bytes(b"legacy-parquet-placeholder")
    provenance.write_text(
        json.dumps({"columns": ["patient_stay_id", "age", "death"]}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        cm,
        "materialize_to_parquet",
        lambda **kwargs: captured.update(kwargs)
        or {"parquet": universe, "provenance": provenance},
    )

    result = acquire_universe_for_question(
        export_dir=tmp_path,
        question="q",
        llm=_stub('{"selected_concepts": ["death", "age"]}'),
        output_dir=tmp_path,
        target_outcome="death",
        outcome_concepts=["death"],
        static_concepts=["age"],
        emit_trajectory=False,
        patient_grouping=grouping,
    )

    assert result.blocked is False
    assert captured["replacement_identity_path"] == mapping
    assert captured["replacement_identity_sha256"] == "a" * 64
    assert captured["output_identity_column"] == "patient_stay_id"


def test_acquisition_does_not_silently_ungroup_requested_trajectory(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        df_mod,
        "build_available_catalog",
        lambda _d: _catalog("death", "age"),
    )
    grouping = PatientGroupingBinding(
        mapping_path=tmp_path / "mapping.parquet",
        mapping_sha256="a" * 64,
        mapping_stay_column="stay_id",
        mapping_patient_column="patient_key",
    )

    with pytest.raises(MaterializedMetadataError, match="longitudinal trajectory"):
        acquire_universe_for_question(
            export_dir=tmp_path,
            question="q",
            llm=_stub('{"selected_concepts": ["death", "age"]}'),
            output_dir=tmp_path,
            target_outcome="death",
            outcome_concepts=["death"],
            static_concepts=["age"],
            emit_trajectory=True,
            patient_grouping=grouping,
        )
