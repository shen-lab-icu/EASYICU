"""A raw official-Demo study reads packages and identity through one owner.

B6, B9 and B11 (UX legs, 2026-11-19) were one class: a consumer opened the
study's bound path -- for a raw Demo study the raw folder, not an export
package -- as a package or identity source. The S1506 sweep routed the
remaining direct consumers through ``source_authority_export_path`` (identity
and package reads) or ``bound_source_planning_concepts`` (the planning menu),
and classified every other reader. A new reader of a package or identity
source must be classified here before it lands.
Synthetic official-Demo fixture; no benchmark item.
"""

import ast
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

from easyicu.research_agent.acquisition import catalog as catalog_owner
from easyicu.research_agent.intake.export_package import ExportPackageError
from easyicu.webserver import agent_runs, research_launch_scientific as launch, source_identity_authority, sources, study_contexts
from easyicu.webserver.pi_copilot import study_context_update, workflow
from easyicu.webserver.pi_copilot.demo_planning_authority import resolve_choice
from tests.support.official_demo_choice import choice as choice
from tests.support.official_demo_raw import official_raw as official_raw


@pytest.fixture
def chosen(choice, monkeypatch):
    monkeypatch.setattr(agent_runs, "list_run_history", lambda **_: {"runs": []})
    monkeypatch.setattr(sources, "_autodiscovered_paths", lambda **_: [])
    choice.service.authorize_data_source(choice.record.session_id, **choice.body)
    study = study_contexts.get_context(choice.study["id"])
    assert resolve_choice(study) is not None and study["data_source"]["path"] == str(choice.paths.raw)
    return study


def _asked(monkeypatch, name):
    asked = []

    def resolve(*, export_path, database):
        asked.append(str(export_path))
        return None

    monkeypatch.setattr(source_identity_authority, name, resolve)
    return asked


def test_the_first_plan_asks_the_official_export_whether_it_groups_stays(choice, chosen, monkeypatch):
    from easyicu.webserver import research_run_submission

    asked = _asked(monkeypatch, "resolve_study_patient_grouping")
    research_run_submission._with_repeated_stays_default(chosen, path=chosen["data_source"]["path"])
    assert asked == [str(choice.paths.export)]
    # Without a choice that resolves, the bound raw folder is asked, as before.
    asked.clear()
    unsigned = {**chosen, "demo_planning_authority_ref": ""}
    research_run_submission._with_repeated_stays_default(unsigned, path=unsigned["data_source"]["path"])
    assert asked == [str(choice.paths.raw)]


def test_the_workflow_asks_the_official_export_whether_it_groups_stays(choice, chosen, monkeypatch):
    asked = _asked(monkeypatch, "resolve_study_patient_grouping")
    workflow._source_groups_stays_at.cache_clear()
    workflow._study_source_groups_stays(chosen)
    assert asked == [str(choice.paths.export)]


def test_an_exposure_on_the_raw_folder_is_checked_on_its_planning_menu(choice, chosen):
    # The stated-variable check already planned a raw Demo on its database's
    # capability menu; the exposure check opened the raw folder as a package.
    source = chosen["data_source"]
    assert launch.checked_exposure_concept(
        "age", "age", question="Is age associated with hospital mortality?", source=source,
    ) == "age"
    assert "age" in launch.bound_source_planning_concepts(source)
    with pytest.raises(ExportPackageError):
        catalog_owner.build_available_catalog(Path(source["path"]))


def test_exact_columns_bind_against_the_official_export_or_wait_for_a_package(choice, chosen, monkeypatch):
    read = []
    real = catalog_owner.build_available_catalog

    def catalog(path):
        read.append(str(path))
        return SimpleNamespace(concepts=())

    monkeypatch.setattr(catalog_owner, "build_available_catalog", catalog)
    assert study_context_update._execution_catalog(chosen, {}, chosen["data_source"]) is not None
    assert read == [str(choice.paths.export)]
    # Unsigned, the raw folder has no package marker: the columns wait.
    monkeypatch.setattr(catalog_owner, "build_available_catalog", real)
    unsigned = {**chosen, "demo_planning_authority_ref": ""}
    assert study_context_update._execution_catalog(unsigned, {}, unsigned["data_source"]) is None

    # Any other package failure still refuses.
    def broken(path):
        raise ExportPackageError("synthetic", code="manifest_file_missing")

    monkeypatch.setattr(catalog_owner, "build_available_catalog", broken)
    with pytest.raises(ExportPackageError):
        study_context_update._execution_catalog(chosen, {}, chosen["data_source"])


@pytest.mark.parametrize(
    ("prior", "execution"),
    [
        ({}, {}),
        ({"primary_exposure": "lactate"}, {"primary_exposure": None}),
        ({"outcome": "death", "covariates": ["age"]}, {"outcome": None, "covariates": []}),
    ],
)
def test_clearing_exact_columns_reads_no_catalog(monkeypatch, prior, execution):
    # Carbon b11r1/b11r2: only an update that binds a concept reads the
    # package; clearing columns on a broken package still saves.
    from easyicu.webserver.pi_copilot import tools as tool_module
    from easyicu.webserver.pi_copilot.contracts import AuthorityBinding, PiSessionRecord, ToolExecutionContext

    current = {
        "id": "study-clear-columns", "revision": 2,
        "question": "Is lactate associated with mortality?", "active_job_id": None,
        "data_source": {"path": "/private/full-export", "database": "miiv"},
        "modules": ["demographics", "outcome"],
        # Each case clears every column it had bound; retained columns would
        # be merged and bound again, so a clear that leaves none reads nothing.
        "execution_concepts": dict(prior),
    }

    def unreadable(_path):
        raise ExportPackageError("synthetic", code="manifest_file_missing")

    writes = []
    monkeypatch.setattr(catalog_owner, "build_available_catalog", unreadable)
    monkeypatch.setattr(tool_module, "_bound_context", lambda binding: dict(current))
    monkeypatch.setattr(tool_module.study_contexts, "upsert_context",
                        lambda raw, **_kw: writes.append(dict(raw)) or {**current, **raw, "revision": 3})
    def context():
        return ToolExecutionContext(
            data_handling="zero_retention_declared",
            session=PiSessionRecord(session_id="pi-clear-columns", binding=AuthorityBinding(
                study_context_id=current["id"], study_revision=current["revision"])),
            allowed_actions={"configure"},
        )

    result = tool_module.execute_tool("easyicu_update_study_context", {"execution_concepts": execution}, context())
    assert result["code"] == "study_context_updated", result
    assert writes and "execution_concepts" in writes[0]
    # The saved mapping is cleared, not the prior binding kept.
    assert not study_context_update._execution_binds_columns(writes[0]["execution_concepts"])
    # Binding a concept on that package still refuses with the package error.
    with pytest.raises(ExportPackageError):
        tool_module.execute_tool("easyicu_update_study_context", {"execution_concepts": {"outcome": "death"}}, context())


def test_the_owner_compares_the_bound_path_as_a_path(choice, chosen, monkeypatch):
    # Carbon b11r1/b11r2: "~" or a trailing separator is the same bound folder;
    # an empty path is never the bound folder.
    raw = Path(chosen["data_source"]["path"])
    monkeypatch.setenv("HOME", str(raw.parent))
    assert launch.bound_source_authority_path(str(raw) + "/", chosen) == str(choice.paths.export)
    assert launch.bound_source_authority_path("~/" + raw.name, chosen) == str(choice.paths.export)
    assert launch.bound_source_planning_concepts({"path": "~/" + raw.name}) == launch.bound_source_planning_concepts(
        {"path": str(raw)})
    assert launch.bound_source_planning_concepts({}) == set()
    # A converted (non-Demo) folder under "~" is planned on its menu too.
    converted = raw.parent / "converted_miiv"
    converted.mkdir()
    (converted / "icustays.csv").write_text("stay_id\n")
    menu = launch.bound_source_planning_concepts({"path": str(converted), "database": "miiv"})
    assert menu and launch.bound_source_planning_concepts({"path": "~/converted_miiv", "database": "miiv"}) == menu
    dot = {**chosen, "data_source": {**chosen["data_source"], "path": "."}}
    assert launch.bound_source_authority_path("", dot) == ""
    assert launch.bound_source_authority_path("   ", dot) == "   "


def test_planning_coordinates_read_the_registered_export(choice, chosen, monkeypatch):
    # S1540 sweep: the planning coordinates' composite column and KDIGO
    # observability read the bound raw Demo folder, which is no package, so
    # both degraded silently; they read the registered export.
    from easyicu.webserver import research_launch_scientific as launch

    raw = chosen["data_source"]["path"]
    read = []
    monkeypatch.setattr(launch, "_bound_source_column", lambda path, **_k: read.append(("column", path)) or (lambda _concept: None))
    monkeypatch.setattr(launch, "_observability_preserving_exposure", lambda name, path: read.append(("kdigo", path)) or name)
    launch._metadata_only_planning_coordinates(
        question=chosen["question"], database="miiv", export_path=raw, study=chosen)
    assert read == [("kdigo", str(choice.paths.export)), ("column", str(choice.paths.export))]
    read.clear()
    launch._metadata_only_planning_coordinates(
        question=chosen["question"], database="miiv", export_path=raw, study={**chosen, "demo_planning_authority_ref": ""})
    assert read == [("kdigo", raw), ("column", raw)]


# Every call in src/easyicu/webserver that opens a path as an export package
# or asks an identity authority of it, by (file, enclosing function, reader).
# OWNER: the path comes from source_authority_export_path /
# bound_source_authority_path / bound_source_planning_concepts, or a raw folder
# is diverted first. REGISTERED: only a registry row, a validated prepared
# binding or a full_reviewed package reaches it. REFUSES: a raw folder gets a
# typed refusal, by contract. FOLDER: describes the folder itself (a source
# listing, or that the bound folder still holds tables); its modules never
# scope a run (B12). The S1506 sweep table records the evidence.
_READERS = frozenset({
    "build_available_catalog", "open_export_package", "inspect_export_layout",
    "validate_research_pipeline_source", "prepared_export_manifest_path", "index_export_package",
    "resolve_study_patient_grouping", "resolve_study_first_icu_stay",
    "resolve_study_first_icu_stay_per_hospitalization", "resolve_patient_grouping_authority",
    # Wrappers that take a path or a study view: their callers are classified too.
    "_data_foundation_profile", "_source_groups_stays_at", "bound_source_planning_concepts",
    "bound_source_authority_path",
    # B12: a folder description whose modules scope a run, and the wrapper
    # that scopes it from a study view (Carbon b12r1).
    "describe_source_tables", "_neutral_materialization_scope",
})
_CLASSIFIED = {
    ("agent_pipeline_runs.py", "_metadata_only_planning_catalog", "build_available_catalog"): (1, "OWNER"),
    ("agent_pipeline_runs.py", "make_research_pipeline_run_runner.runner", "validate_research_pipeline_source"): (1, "REGISTERED"),
    ("agent_pipeline_runs.py", "resume_research_pipeline", "validate_research_pipeline_source"): (1, "REGISTERED"),
    ("agent_review_recovery.py", "recover_pending_review", "validate_research_pipeline_source"): (1, "REGISTERED"),
    ("case_duration_materialization.py", "materialize_approved_case_duration_universe", "open_export_package"): (1, "REGISTERED"),
    ("data_package_execution_readiness.py", "_first_icu_stay_flags", "resolve_study_first_icu_stay"): (1, "REGISTERED"),
    ("data_package_execution_readiness.py", "_first_icu_stay_review", "resolve_study_first_icu_stay"): (1, "REGISTERED"),
    ("data_package_execution_readiness.py", "_hospital_first_eligibility_review", "resolve_study_first_icu_stay_per_hospitalization"): (1, "REGISTERED"),
    ("data_package_execution_readiness.py", "_hospital_first_stay_review", "resolve_study_first_icu_stay_per_hospitalization"): (1, "REGISTERED"),
    ("data_package_execution_readiness.py", "_patient_grouping_review", "resolve_study_patient_grouping"): (1, "REGISTERED"),
    ("data_package_review.py", "build_registered_data_package_review", "build_available_catalog"): (1, "REFUSES"),
    ("dataio.py", "_detect_database", "inspect_export_layout"): (1, "OWNER"),
    ("dataio.py", "scan_path", "inspect_export_layout"): (1, "OWNER"),
    ("dataio.py", "validate_research_pipeline_source", "inspect_export_layout"): (1, "REFUSES"),
    ("dataio.py", "validate_research_pipeline_source", "open_export_package"): (1, "REFUSES"),
    ("dataio.py", "validate_research_pipeline_source", "prepared_export_manifest_path"): (1, "REFUSES"),
    ("demo_source_storage.py", "egress_source_database", "inspect_export_layout"): (1, "OWNER"),
    ("manuscript_repair.py", "make_report_only_run_runner", "validate_research_pipeline_source"): (1, "REGISTERED"),
    ("pi_copilot/service.py", "PiCopilotService.apply_agent_plan_configuration", "resolve_study_first_icu_stay"): (1, "OWNER"),
    ("pi_copilot/service.py", "PiCopilotService.apply_agent_plan_configuration", "resolve_study_patient_grouping"): (1, "OWNER"),
    ("pi_copilot/study_context_update.py", "_execution_catalog", "build_available_catalog"): (1, "OWNER"),
    ("pi_copilot/tools.py", "_assess_idea_feasibility", "build_available_catalog"): (1, "REFUSES"),
    ("pi_copilot/tools.py", "_list_concept_catalog", "build_available_catalog"): (1, "REGISTERED"),
    ("pi_copilot/workflow.py", "_source_groups_stays_at", "resolve_study_patient_grouping"): (1, "OWNER"),
    ("quick_answer/preparation.py", "_verified_identity", "resolve_study_first_icu_stay"): (1, "REGISTERED"),
    ("quick_answer/preparation.py", "_verified_identity", "resolve_study_patient_grouping"): (1, "REGISTERED"),
    ("quick_answer/preparation.py", "prepare_registered_question", "validate_research_pipeline_source"): (1, "REGISTERED"),
    ("research_append_source.py", "registered_source_changed", "validate_research_pipeline_source"): (1, "REGISTERED"),
    ("research_launch_resume.py", "_development_resume_launch_scope", "validate_research_pipeline_source"): (1, "REGISTERED"),
    ("research_launch_scientific.py", "_bound_source_column", "build_available_catalog"): (1, "OWNER"),
    ("research_launch_scientific.py", "_metadata_only_planning_coordinates", "bound_source_authority_path"): (1, "OWNER"),
    ("research_launch_scientific.py", "_neutral_materialization_scope", "bound_source_authority_path"): (1, "OWNER"),
    ("research_launch_scientific.py", "_neutral_materialization_scope", "describe_source_tables"): (1, "OWNER"),
    ("research_launch_scientific.py", "_data_foundation_profile", "build_available_catalog"): (1, "OWNER"),
    ("research_launch_scientific.py", "_first_icu_stay_for_cohort", "resolve_study_first_icu_stay"): (1, "OWNER"),
    ("research_launch_scientific.py", "_first_icu_stay_per_hospitalization_for_cohort", "resolve_study_first_icu_stay_per_hospitalization"): (1, "OWNER"),
    ("research_launch_scientific.py", "bound_source_planning_concepts", "build_available_catalog"): (1, "OWNER"),
    ("research_launch_scientific.py", "verified_patient_grouping", "resolve_study_patient_grouping"): (1, "OWNER"),
    ("research_multi_extraction.py", "prepare_source", "prepared_export_manifest_path"): (1, "OWNER"),
    ("research_multi_extraction.py", "prepare_source", "validate_research_pipeline_source"): (1, "REGISTERED"),
    ("research_multi_pipeline.py", "PipelineDatabaseOwner._plan", "validate_research_pipeline_source"): (1, "REGISTERED"),
    ("research_pipeline_run_preparation.py", "_prepare_scientific_launch", "_neutral_materialization_scope"): (1, "OWNER"),
    ("research_pipeline_run_preparation.py", "_prepare_scientific_launch", "validate_research_pipeline_source"): (1, "REGISTERED"),
    ("research_plan_revision.py", "load_prepared_plan_revision", "validate_research_pipeline_source"): (1, "REGISTERED"),
    ("research_resume_package.py", "_package_offer_verdict", "validate_research_pipeline_source"): (1, "REGISTERED"),
    ("research_resume_package.py", "prepared_package_identity", "prepared_export_manifest_path"): (1, "REGISTERED"),
    ("research_run_submission.py", "_with_repeated_stays_default", "resolve_study_patient_grouping"): (1, "OWNER"),
    ("research_run_submission.py", "_with_first_stay_repair", "resolve_study_first_icu_stay_per_hospitalization"): (1, "OWNER"),
    ("research_run_submission.py", "submit_research_run", "describe_source_tables"): (1, "FOLDER"),
    ("research_run_submission.py", "submit_research_run", "prepared_export_manifest_path"): (1, "OWNER"),
    ("research_run_submission.py", "execution_input_available", "prepared_export_manifest_path"): (1, "OWNER"),
    ("research_run_submission.py", "submit_research_run", "validate_research_pipeline_source"): (2, "REGISTERED"),
    ("run_population_proof.py", "verified_first_stay_applied", "open_export_package"): (1, "REGISTERED"),
    ("sources.py", "_autodiscovered_paths.describe", "describe_source_tables"): (1, "FOLDER"),
    ("sources.py", "_source_from_path", "describe_source_tables"): (1, "FOLDER"),
    ("scientific_runtime_projection.py", "export_kdigo_strict_derivation_available", "open_export_package"): (1, "OWNER"),
    ("source_identity_authority.py", "resolve_study_first_icu_stay", "resolve_study_patient_grouping"): (1, "OWNER"),
    ("source_identity_authority.py", "resolve_study_patient_grouping", "resolve_patient_grouping_authority"): (1, "OWNER"),
    ("time_origin_source_authority.py", "attest_case_duration_for_review", "open_export_package"): (1, "REGISTERED"),
    ("time_origin_source_authority.py", "attest_registered_export_time_origin", "open_export_package"): (1, "REGISTERED"),
    ("agent_pipeline_runs.py", "make_research_pipeline_run_runner.runner", "_data_foundation_profile"): (2, "OWNER"),
    ("pi_copilot/study_context_update.py", "_execution_catalog", "bound_source_authority_path"): (1, "OWNER"),
    ("pi_copilot/workflow.py", "_study_source_groups_stays", "_source_groups_stays_at"): (1, "OWNER"),
    ("research_launch_scientific.py", "_data_foundation_profile", "bound_source_authority_path"): (1, "OWNER"),
    ("research_launch_scientific.py", "checked_exposure_concept", "bound_source_planning_concepts"): (1, "OWNER"),
    ("research_launch_scientific.py", "checked_stated_variables", "bound_source_planning_concepts"): (1, "OWNER"),
    ("research_launch_scientific.py", "target_trial_launch_inputs", "_data_foundation_profile"): (1, "REGISTERED"),
    ("research_pipeline_run_preparation.py", "_prepare_scientific_launch", "_data_foundation_profile"): (1, "REGISTERED"),
    ("research_run_submission.py", "_with_repeated_stays_default", "bound_source_authority_path"): (1, "OWNER"),
    ("research_run_submission.py", "_with_first_stay_repair", "bound_source_authority_path"): (1, "OWNER"),
}


def _reader_sites() -> Counter:
    root = Path(__file__).resolve().parents[3] / "src/easyicu/webserver"
    sites: Counter = Counter()
    for path in sorted(root.rglob("*.py")):
        relative = path.relative_to(root).as_posix()

        def walk(node, scope):
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    walk(child, [*scope, child.name])
                    continue
                if isinstance(child, ast.Call):
                    func = child.func
                    name = func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else None
                    if name in _READERS:
                        sites[(relative, ".".join(scope) or "<module>", name)] += 1
                walk(child, scope)

        walk(ast.parse(path.read_text(encoding="utf-8")), [])
    return sites


def test_every_package_or_identity_reader_is_classified():
    # Exact counts (Carbon b11r1): a second reader call inside a classified
    # function is a new site and must be classified too.
    sites = _reader_sites()
    expected = {key: count for key, (count, _cls) in _CLASSIFIED.items()}
    new = sorted((key, count) for key, count in sites.items() if expected.get(key) != count)
    assert not new, (
        "Classify these readers (route a study's bound path through "
        "source_authority_export_path / bound_source_authority_path, or record "
        f"why a raw Demo folder cannot reach them), with their call counts: {new}"
    )
    assert not sorted(expected.keys() - sites.keys()), "a classified reader is gone; drop its row"
    assert {cls for _count, cls in _CLASSIFIED.values()} <= {"OWNER", "REGISTERED", "REFUSES", "FOLDER"}


def test_a_planning_default_never_voids_the_signed_demo_choice(choice, chosen):
    # B16 (UX leg 3 zh, 2026-11-20): since S1506 the repeated-stay default reads
    # the registered export's grouping, so it applied to a raw Demo study; in a
    # staged launch its write left the study's signed choice stale and the
    # first plan was refused (research_pipeline_demo_planning_authority_stale).
    # A host default never voids the researcher's source confirmation: it is
    # left to the plan card's review-then-compile path, as a refused write is.
    from types import SimpleNamespace

    from easyicu.webserver import launch_study_updates, research_run_submission
    from easyicu.webserver.pi_copilot import plan_decisions

    design = chosen.get("analysis_design") if isinstance(chosen.get("analysis_design"), dict) else {}
    patch = {"analysis_design": {**design, **plan_decisions._PATIENT_CLUSTERED_DESIGN}}
    unsigned_id = study_contexts.upsert_context({
        "question": chosen["question"], "data_source": {"path": "/synthetic/export", "database": "miiv"},
    })["id"]
    seen = {}

    @launch_study_updates.stage_launch_updates
    def launch_with_default(request, study):
        seen[request.study_context_id] = research_run_submission._with_planning_default(
            study, patch, label="synthetic")

    launch_with_default(SimpleNamespace(study_context_id=chosen["id"]), chosen)
    assert seen[chosen["id"]] == chosen and resolve_choice(seen[chosen["id"]]) is not None
    # Without a signed choice the default is applied as before.
    unsigned = study_contexts.get_context(unsigned_id)
    launch_with_default(SimpleNamespace(study_context_id=unsigned_id), unsigned)
    assert seen[unsigned_id]["analysis_design"] == patch["analysis_design"]
    # The guard reads the patch itself: this design change would void the choice.
    assert research_run_submission._breaks_signed_demo_choice(chosen, patch) is True
