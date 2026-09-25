"""Physical columns cannot overrule a canonical negative source contract."""
import json

import pytest

from easyicu.research_agent.acquisition.catalog import AvailableCatalog, CatalogConcept
from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient
from easyicu.webserver import agent_pipeline_runs as owner
from easyicu.webserver.research_pipeline_run_errors import ResearchPipelineRunError


@pytest.fixture
def source_menu(monkeypatch):
    from easyicu.research_agent.acquisition import catalog

    base = AvailableCatalog(source="canonical", concepts=[CatalogConcept("death")])
    source = AvailableCatalog(source="legacy", concepts=[
        CatalogConcept("icu_readmission", n_rows=0),
        CatalogConcept("local_measurement", description="Source-owned measurement"),
        CatalogConcept("death", description="Source-owned death definition"),
    ])
    monkeypatch.setattr(catalog, "build_database_capability_catalog", lambda _: base)
    monkeypatch.setattr(catalog, "build_available_catalog", lambda _: source)
    return source


def test_source_merge_preserves_negative_contract_and_local_metadata(source_menu):
    merged = owner._metadata_only_planning_catalog(database="miiv", export_path="/metadata")
    assert merged.ids() == ["death", "local_measurement"]
    assert merged.concepts[0].description == "Source-owned death definition"
    assert source_menu.ids() == ["icu_readmission", "local_measurement", "death"]


def test_prepared_cohort_menu_contains_only_physical_source_columns(
    source_menu, tmp_path, monkeypatch,
):
    from easyicu.research_agent.acquisition import catalog as catalog_module

    monkeypatch.setattr(
        catalog_module,
        "build_database_capability_catalog",
        lambda _: AvailableCatalog(
            source="canonical",
            concepts=[CatalogConcept("death"), CatalogConcept("crea")],
        ),
    )
    (tmp_path / "easyicu_export_manifest.json").write_text(
        json.dumps({"entry_mode": "study_local_prepared_cohort"}),
        encoding="utf-8",
    )

    catalog = owner._metadata_only_planning_catalog(
        database="miiv", export_path=tmp_path,
    )

    assert catalog.ids() == ["local_measurement", "death"]
    assert source_menu.ids() == ["icu_readmission", "local_measurement", "death"]


def test_prepared_cohort_without_verifiable_catalog_fails_closed(
    tmp_path, monkeypatch,
):
    from easyicu.research_agent.acquisition import catalog as catalog_module

    (tmp_path / "easyicu_export_manifest.json").write_text(
        json.dumps({"entry_mode": "study_local_prepared_cohort"}),
        encoding="utf-8",
    )
    def invalid_catalog(_):
        raise ValueError("invalid package")

    monkeypatch.setattr(catalog_module, "build_available_catalog", invalid_catalog)

    with pytest.raises(ResearchPipelineRunError) as caught:
        owner._metadata_only_planning_catalog(
            database="miiv", export_path=tmp_path,
        )

    assert caught.value.code == "research_pipeline_prepared_source_catalog_unavailable"


@pytest.mark.parametrize("requirement", [
    {"required_concepts": ("icu_readmission",)},
    {"target_outcome": "icu_readmission"},
    {"question": "Compare ICU readmission after treatment."},
])
def test_explicit_required_unsupported_input_blocks_before_model_or_output(
    source_menu, tmp_path, requirement,
):
    llm = ScriptedMockLLMClient([json.dumps({
        "selected_concepts": ["death"], "rationale": "An alternative outcome.",
        "inclusion_exclusion": [],
    })])
    options = {"question": "Compare hospital death.", **requirement}
    with pytest.raises(ResearchPipelineRunError) as caught:
        owner._metadata_only_planning_acquisition(
            database="miiv", export_path="/metadata", llm=llm,
            output_dir=tmp_path / "new-catalog", **options,
        )
    assert caught.value.code == "research_pipeline_required_concept_structurally_unavailable"
    assert caught.value.details["required_concepts"] == ["icu_readmission"]
    assert llm.calls == []
    assert not (tmp_path / "new-catalog").exists()


def test_optional_unsupported_column_is_not_reintroduced(source_menu, tmp_path):
    import pyarrow.parquet as pq

    llm = ScriptedMockLLMClient([json.dumps({
        "selected_concepts": ["death", "icu_readmission"],
        "rationale": "Mortality with optional covariate.", "inclusion_exclusion": [],
    })])
    result = owner._metadata_only_planning_acquisition(
        database="miiv", export_path="/metadata", question="Describe death, not readmission.",
        llm=llm, output_dir=tmp_path / "catalog",
    )
    assert not result.blocked
    assert pq.read_schema(result.universe_path).names == ["stay_id", "death"]
    assert pq.read_metadata(result.universe_path).num_rows == 0
    receipt = json.loads(result.provenance_path.read_text())
    assert receipt["unavailable_model_concepts"] == ["icu_readmission"]



def test_listed_but_unresolvable_optional_pick_is_dropped_not_fatal(tmp_path, monkeypatch):
    """A registry concept the source cannot provide leaves the menu quietly.

    The merged planning menu lists registry concepts next to the export's
    typed columns.  Once typed owners exist, an untyped registry-only entry
    can never resolve, so the model may pick something coverage will call
    missing.  Measured 2026-09-22 on the official eICU demo: one optional
    pick (``icu_unit_type``) blocked the whole run.  An optional pick is
    dropped and recorded; only an empty selection fails closed, with a code.
    """
    import pyarrow.parquet as pq

    from easyicu.research_agent.acquisition import catalog as catalog_module

    typed_death = CatalogConcept(
        "death", file_name="outcome.parquet", typed_metadata=True, column_role="event_status",
    )
    registry_only = CatalogConcept("icu_unit_type", description="ICU unit type", category="demographics")
    monkeypatch.setattr(
        catalog_module, "build_database_capability_catalog",
        lambda _: AvailableCatalog(source="canonical", concepts=[registry_only, CatalogConcept("death")]),
    )
    monkeypatch.setattr(
        catalog_module, "build_available_catalog",
        lambda _: AvailableCatalog(source="export", concepts=[typed_death]),
    )
    llm = ScriptedMockLLMClient([json.dumps({
        "selected_concepts": ["death", "icu_unit_type"],
        "rationale": "Mortality with an optional unit-type covariate.", "inclusion_exclusion": [],
    })])
    result = owner._metadata_only_planning_acquisition(
        database="eicu", export_path="/metadata", question="Describe death by unit.",
        llm=llm, output_dir=tmp_path / "catalog",
    )
    assert not result.blocked
    assert result.blocked_reason_code == ""
    assert pq.read_schema(result.universe_path).names == ["patientunitstayid", "death"]
    receipt = json.loads(result.provenance_path.read_text())
    assert receipt["unavailable_model_concepts"] == ["icu_unit_type"]

    only_unavailable = ScriptedMockLLMClient([json.dumps({
        "selected_concepts": ["icu_unit_type"],
        "rationale": "Unit type only.", "inclusion_exclusion": [],
    })])
    blocked = owner._metadata_only_planning_acquisition(
        database="eicu", export_path="/metadata", question="Describe unit type.",
        llm=only_unavailable, output_dir=tmp_path / "blocked",
    )
    assert blocked.blocked
    assert blocked.blocked_reason_code == "no_available_concepts"
    assert not (tmp_path / "blocked").exists()



def test_a_study_coordinate_the_schema_cannot_carry_stops_before_the_planner(
    tmp_path, monkeypatch
):
    """The study's exposure and outcome must be columns of the planning schema.

    A registry-only name for a concept that the export publishes under a typed
    column cannot resolve, so it left the schema without a trace. Every plan
    then failed the primary-result gate, which requires that exact column, and
    each failure cost a full Planner round. A coordinate is required: the run
    stops before any Planner call and names it.
    """
    import pyarrow.parquet as pq

    from easyicu.research_agent.acquisition import catalog as catalog_module

    typed = [
        CatalogConcept(
            "death", file_name="outcome.parquet", typed_metadata=True,
            column_role="event_status",
        ),
        CatalogConcept(
            "exposure_flag_v2", file_name="flags.parquet", typed_metadata=True,
            column_role="event_status",
        ),
    ]
    monkeypatch.setattr(
        catalog_module, "build_database_capability_catalog",
        lambda _: AvailableCatalog(
            source="canonical",
            concepts=[CatalogConcept("exposure_flag"), CatalogConcept("death")],
        ),
    )
    monkeypatch.setattr(
        catalog_module, "build_available_catalog",
        lambda _: AvailableCatalog(source="export", concepts=typed),
    )

    def run(coordinates, output):
        return owner._metadata_only_planning_acquisition(
            database="eicu", export_path="/metadata", question="Flag and death.",
            llm=ScriptedMockLLMClient([json.dumps({
                "selected_concepts": ["exposure_flag_v2", "death"],
                "rationale": "Flag and mortality.", "inclusion_exclusion": [],
            })]),
            output_dir=tmp_path / output,
            required_coordinates=coordinates,
        )

    blocked = run(("exposure_flag", "death"), "blocked")
    assert blocked.blocked
    assert blocked.blocked_reason_code == "required_concepts_unavailable"
    assert blocked.missing_concepts[0] == "exposure_flag"
    assert not (tmp_path / "blocked").exists()

    carried = run(("exposure_flag_v2", "death"), "carried")
    assert not carried.blocked
    assert {"exposure_flag_v2", "death"} <= set(
        pq.read_schema(carried.universe_path).names
    )


def test_a_cross_concept_reading_is_offered_only_when_its_inputs_are_present(
    monkeypatch,
):
    """The strict KDIGO stage is not a database capability, it is a derivation.

    No extraction produces the column, so it may only appear once the source
    physically carries every evidence receipt the reading combines.  Offering it
    otherwise would put an unmaterializable exposure on the planning menu.
    """

    from easyicu.research_agent.acquisition import catalog
    from easyicu.research_agent.contracts.host_derivations import host_derivation

    declared = host_derivation("strict_kdigo_stage")
    # Each builder returns a fresh catalog, exactly as the real owners do.
    monkeypatch.setattr(
        catalog,
        "build_database_capability_catalog",
        lambda _: AvailableCatalog(
            source="canonical", concepts=[CatalogConcept("death")]
        ),
    )

    monkeypatch.setattr(
        catalog,
        "build_available_catalog",
        lambda _: AvailableCatalog(
            source="legacy",
            concepts=[CatalogConcept(name) for name in declared.source_concepts],
        ),
    )
    offered = owner._metadata_only_planning_catalog(
        database="miiv", export_path="/metadata"
    ).ids()
    assert "aki_stage_strict" in offered
    assert "aki_ascertainment" in offered
    # Receipts travel with the cohort but are not design variables.
    assert "kidney_window_row_count" not in offered
    assert "kidney_complete_negative_observed" not in offered
    # A derived column belongs beside the concepts it reads, with its own
    # reader-facing line rather than the derivation's generic summary.
    menu = owner._metadata_only_planning_catalog(
        database="miiv", export_path="/metadata"
    )
    strict = next(c for c in menu.concepts if c.concept_id == "aki_stage_strict")
    primary = next(
        c for c in menu.concepts if c.concept_id == declared.source_concepts[0]
    )
    assert strict.category == primary.category
    assert "explicit unknown" in strict.description

    monkeypatch.setattr(
        catalog,
        "build_available_catalog",
        lambda _: AvailableCatalog(
            source="legacy",
            concepts=[
                CatalogConcept(name)
                for name in declared.source_concepts
                if name != "rrt_evidence_status"
            ],
        ),
    )
    assert "aki_stage_strict" not in owner._metadata_only_planning_catalog(
        database="miiv", export_path="/metadata"
    ).ids()


def test_a_capability_the_bound_source_lacks_is_marked_not_hidden(monkeypatch):
    """A planner-only menu says what a later extraction could produce.

    That is deliberate -- a plan is allowed to state what must still be
    extracted.  What was missing is which side of the line a concept is on: the
    model could name a concept this source does not carry and only find out
    after acquisition ran, which costs a whole planning round.  For a study
    whose source IS the final input, it can never be materialized at all.
    """

    from easyicu.research_agent.acquisition import catalog

    monkeypatch.setattr(
        catalog,
        "build_database_capability_catalog",
        lambda _: AvailableCatalog(
            source="canonical",
            # ``lact`` is extractable from the database but absent here.
            concepts=[CatalogConcept("death"), CatalogConcept("lact")],
        ),
    )
    monkeypatch.setattr(
        catalog,
        "build_available_catalog",
        lambda _: AvailableCatalog(
            source="export",
            concepts=[CatalogConcept("death"), CatalogConcept("local_measurement")],
        ),
    )

    menu = owner._metadata_only_planning_catalog(
        database="miiv", export_path="/prepared-export"
    )
    by_id = {item.concept_id: item for item in menu.concepts}

    assert sorted(by_id) == ["death", "lact", "local_measurement"]
    assert by_id["lact"].present_in_bound_source is False
    assert by_id["death"].present_in_bound_source is True
    assert by_id["local_measurement"].present_in_bound_source is True
    # The model is told, in the same place it chooses concepts.
    assert "NOT in the bound source" in menu.render_for_prompt()


def test_an_offered_cross_concept_reading_can_actually_be_selected(monkeypatch):
    """Listed on the menu must mean selectable.

    Coverage resolves only typed owners once a catalog carries any typed
    column metadata -- and every current export does.  The derived stage was
    offered untyped, so a proposed ``aki_stage_strict`` exposure was silently
    dropped from the zero-row planning schema and the landmark runtime then
    refused the launch with ``web_scientific_runtime_columns_missing``.
    """

    from easyicu.research_agent.acquisition import catalog
    from easyicu.research_agent.acquisition.catalog import assess_coverage
    from easyicu.research_agent.contracts.host_derivations import host_derivation

    declared = host_derivation("strict_kdigo_stage")
    monkeypatch.setattr(
        catalog,
        "build_database_capability_catalog",
        lambda _: AvailableCatalog(source="canonical", concepts=[CatalogConcept("death")]),
    )
    monkeypatch.setattr(
        catalog,
        "build_available_catalog",
        lambda _: AvailableCatalog(
            source="export",
            concepts=[
                CatalogConcept(name, typed_metadata=True, column_role="value")
                for name in (*declared.source_concepts, "death")
            ],
        ),
    )
    menu = owner._metadata_only_planning_catalog(database="eicu_demo", export_path="/export")

    strict = next(c for c in menu.concepts if c.concept_id == "aki_stage_strict")
    assert strict.typed_metadata is True
    assert strict.column_role == "value"
    coverage = assess_coverage(["aki_stage_strict", "aki_ascertainment", "death"], menu)
    assert coverage.missing == []
    assert set(coverage.available) == {"aki_stage_strict", "aki_ascertainment", "death"}


def test_the_planning_runner_requires_its_exposure_and_outcome_columns(
    tmp_path, monkeypatch
):
    """The runner hands the study's coordinates to the source menu as required."""
    import pandas as pd

    from easyicu.webserver import provider_adapter

    export = tmp_path / "export"
    export.mkdir()
    pd.DataFrame({"stay_id": [1], "age": [65]}).to_parquet(
        export / "demographics.parquet", index=False
    )
    (export / "_manifest.json").write_text(json.dumps({
        "database": "miiv",
        "format": "parquet",
        "concept_selection": {"mode": "explicit", "modules": {"demographics": ["age"]}},
        "feature_definitions": {"included": False},
        "files": [{
            "file": "demographics.parquet", "module": "demographics",
            "concepts": 1, "concept_ids": ["age"], "rows": 1,
        }],
    }), encoding="utf-8")
    monkeypatch.setattr(
        provider_adapter,
        "build_research_agent_provider_client",
        lambda *_args, **_kwargs: (
            ScriptedMockLLMClient([]),
            {"provider": "mock", "model": "metadata-only-test"},
        ),
    )
    captured = {}

    def capture_planning_roster(**kwargs):
        captured.update(kwargs)
        raise ResearchPipelineRunError(
            "test_planning_roster_captured", "stop after the planning roster is bound"
        )

    monkeypatch.setattr(owner, "_metadata_only_planning_acquisition", capture_planning_roster)
    runner = owner.make_research_pipeline_run_runner(
        export_path=str(export),
        study_context={
            "id": "study-planning-coordinates",
            "revision": 1,
            "question": "How common is Sepsis-3 in adult ICU stays, and is it "
            "associated with ICU mortality?",
            "data_source": {"path": str(export), "database": "miiv"},
        },
        project_root=str(tmp_path / "projects"),
        provider={"provider": "openai", "external": True},
        provider_environment={"OPENAI_API_KEY": "test-key"},
        credential_source="pi_verified",
        budget_mode="planner_canary",
    )

    class Job:
        id = "job-planning-coordinates"
        cancel_requested = False
        events: list = []

        def emit(self, event):
            self.events.append(dict(event))

    with pytest.raises(ResearchPipelineRunError) as raised:
        runner(Job())

    assert raised.value.code == "test_planning_roster_captured"
    assert tuple(captured["required_coordinates"]) == ("sep3", "death")
