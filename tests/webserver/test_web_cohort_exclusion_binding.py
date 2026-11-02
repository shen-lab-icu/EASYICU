"""An exclusion the researcher stated has to reach the plan as an exclusion.

``CohortSpec`` has carried ``exclusion_criteria`` all along: the CLI fills it
from ``--exclusion``, the pipeline threads it through, and the outbound Planner
context publishes it as ``exclusion_contract``. The Web caller was the one
consumer that had no exclusion channel -- it folded every filter, exclusions
included, into ``inclusion_criteria``. Across 52 recorded plans not one carried
a single exclusion predicate, and a cohort-flow figure cannot draw an exclusion
stage that was never declared as one.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest

from easyicu.research_agent.research_context.outbound import (
    outbound_safe_context_payload,
)
from easyicu.webserver import agent_pipeline_runs
from easyicu.webserver.agent_pipeline_runs import (
    _exclusion_criteria,
    _inclusion_criteria,
)
from easyicu.webserver.pi_copilot import cohort_eligibility
from easyicu.webserver import primary_cohort


def _study(**cohort) -> dict:
    return {"cohort": cohort}


def test_an_exclusion_is_compiled_as_an_exclusion() -> None:
    study = _study(
        label="ICU stays",
        age_min=18,
        min_icu_los_hours=24,
        exclude_readmissions=True,
        # the diagnosis lists are only a filter once the ICD gate is on; see
        # test_a_diagnosis_list_is_declared_only_when_it_actually_filters
        icd_enabled=True,
        include_diagnoses=["condition-a"],
        exclude_diagnoses=["condition-b", "condition-c"],
    )

    inclusion = _inclusion_criteria(study)
    exclusion = _exclusion_criteria(study)

    # who enters -- by the fields the export executes, not by the label
    assert "ICU stays" not in inclusion
    assert "age range: 18 to *" in inclusion
    assert "minimum ICU length of stay: 24 hours" in inclusion
    assert "include diagnoses: condition-a" in inclusion

    # who is removed -- and none of it is still filed as inclusion
    assert "exclude diagnoses: condition-b, condition-c" in exclusion
    assert "each patient's later ICU stays across the bound source" in exclusion
    joined_inclusion = " | ".join(inclusion)
    assert "exclude" not in joined_inclusion
    assert "readmission" not in joined_inclusion


def test_a_study_with_no_filters_declares_neither_side() -> None:
    for study in ({}, _study(), _study(label=""), {"cohort": None}):
        assert _inclusion_criteria(study) == []
        assert _exclusion_criteria(study) == []


def test_a_readmission_exclusion_alone_still_produces_an_exclusion() -> None:
    """The commonest single exclusion must not need a diagnosis list."""

    study = _study(exclude_readmissions=True)
    assert _exclusion_criteria(study)
    assert _inclusion_criteria(study) == []
    # and a false flag declares nothing
    assert _exclusion_criteria(_study(exclude_readmissions=False)) == []


def test_the_pipeline_call_passes_the_exclusion_channel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Compiling exclusions and not sending them changes nothing."""

    study = _study(
        label="ICU stays",
        age_min=18,
        exclude_readmissions=True,
        icd_enabled=True,
        exclude_diagnoses=["condition-b"],
    )
    scientific = SimpleNamespace(
        study=study,
        materialization_study=study,
        patient_grouping=None,
        cohort_window=(0.0, 24.0),
        metadata_planning_coordinates={},
    )
    # Both sides, as the bound export's record allows them to be declared.
    for basis, recorded in (("export_contract", True), ("unrecorded", False)):
        monkeypatch.setattr(
            agent_pipeline_runs,
            "bound_export_selection_basis",
            lambda _study, _export_path, basis=basis: basis,
        )
        declared = agent_pipeline_runs.research_context_declarations(
            scientific, export_path=str(tmp_path / "export")
        )
        assert declared["inclusion_criteria"] == _inclusion_criteria(
            study, export_recorded=recorded
        )
        expected = ["exclude diagnoses: condition-b"] if recorded else []
        assert declared["exclusion_criteria"] == expected
        import json
        preferences = json.loads(declared["user_preferences"]["data_constraints"])
        assert preferences["cohort"]["exclude_readmissions"] is True
        assert preferences["source_selection"]["host_applied"] == []
    assert _exclusion_criteria(study, export_recorded=True)
    # the reviewer-facing cohort summary reports both sides too
    source = " ".join(
        Path("src/easyicu/webserver/agent_pipeline_runs.py").read_text(encoding="utf-8").split()
    )
    assert '"exclusion_criteria": _exclusion_criteria(study),' in source


@pytest.mark.parametrize("kind", ["patient", "hospitalization"])
@pytest.mark.parametrize("basis", ["export_contract", "unrecorded"])
def test_first_stay_contract_requires_a_matching_sealed_cohort(tmp_path, monkeypatch, kind, basis):
    """The requested rule survives, but only materialized evidence makes it applied."""
    import json
    from dataclasses import replace

    import pandas as pd
    from easyicu.research_agent.cohort.materializer import materialize_to_parquet
    from easyicu.research_agent.intake.materialized_metadata import load_verified_materialized_cohort_authority
    from easyicu.research_agent.planning.population_compile import compile_population
    from easyicu.research_agent.planning.population_spec import PopulationSpec
    from easyicu.research_agent.research_context.builder import build_research_context
    from easyicu.research_agent.research_context.export_selection import SourceIdentity, export_applied_selection
    from easyicu.webserver import raw_source_authority, source_identity_authority, sources
    from tests.support.typed_export import typed_export

    monkeypatch.setattr(source_identity_authority.state_paths, "state_root", lambda: tmp_path / "state")
    monkeypatch.setattr(raw_source_authority, "_DEFAULT_CONFIG_PATH", tmp_path / "absent.env")
    monkeypatch.setattr(source_identity_authority, "_DEFAULT_CONFIG_PATH", tmp_path / "absent-grouping.env")
    monkeypatch.setattr(sources, "_CONFIG_PATH", tmp_path / "registry.json")
    monkeypatch.setattr(sources, "path_allowed", lambda *args, **kwargs: True)
    raw = tmp_path / "raw"
    raw.mkdir()
    if kind == "patient":
        database = "miiv"
        field = "first_icu_stay"
        cohort = {"exclude_readmissions": True}
        question = "Among each patient's first ICU stays, describe age."
        pd.DataFrame({"stay_id": [1, 2], "hadm_id": [101, 102], "subject_id": [7, 7],
                      "intime": ["2026-02-01", "2026-01-01"]}).to_parquet(raw / "icustays.parquet", index=False)
        pd.DataFrame({"hadm_id": [101, 102], "dischtime": ["2026-06-01"] * 2,
                      "deathtime": [None, None], "hospital_expire_flag": [0, 0]}).to_parquet(
                          raw / "admissions.parquet", index=False)
        resolve = source_identity_authority.resolve_study_first_icu_stay
    else:
        database = "eicu"
        field = "first_icu_stay_per_hospitalization"
        cohort = {field: True}
        question = "Among first ICU stays per hospitalization, describe age."
        pd.DataFrame({"patientunitstayid": [1, 2], "patienthealthsystemstayid": [1, 1],
                      "unitvisitnumber": [2, 1], "uniquepid": ["same", "same"],
                      "unittype": ["MICU", "MICU"], "hospitaldischargeoffset": [100, 100],
                      "hospitaldischargestatus": ["Alive", "Alive"]}).to_parquet(raw / "patient.parquet", index=False)
        resolve = source_identity_authority.resolve_study_first_icu_stay_per_hospitalization
    source = typed_export(tmp_path / "export", database=database).resolve()
    manifest = source / "_manifest.json"
    manifest_data = json.loads(manifest.read_text())
    manifest_data["data_path"] = str(raw)
    manifest.write_text(json.dumps(manifest_data))
    identity = SourceIdentity(database, sources._source_id(str(source)))
    (tmp_path / "registry.json").write_text(json.dumps({"sources": [
        {"path": str(source), "id": identity.registration_id, "database": database}
    ]}))
    binding = resolve(export_path=source, database=database)
    assert binding is not None
    paths = materialize_to_parquet(tmp_path / "materialized", data_path=source, database=database,
        static_concepts=("age",), feature_concepts=("lact",), outcome_concepts=("death",),
        **binding.materializer_kwargs())
    assert pd.read_parquet(paths["parquet"])["stay_id"].tolist() == [2]
    verified = load_verified_materialized_cohort_authority(paths["parquet"])
    assert verified is not None
    acquisition = SimpleNamespace(universe_path=paths["parquet"], blocked=False,
        cohort_authority_ref=verified.reference,
        cohort_authority_path=paths["parquet"].parent / verified.reference.file)
    cohort.update(label=question, review=question)
    study = {"id": "synthetic", "question": question, "cohort": cohort}
    scientific = SimpleNamespace(study=study, materialization_study=study, patient_grouping=None,
        cohort_window=(0.0, 24.0), metadata_planning_coordinates={}, metadata_only_planning=False,
        foundation_profile={field: binding})
    monkeypatch.setattr(agent_pipeline_runs, "bound_export_selection_basis", lambda *_: basis)

    def selection():
        declared = agent_pipeline_runs.research_context_declarations(
            scientific, export_path=str(source), source_identity=identity.to_dict(), acquisition=acquisition)
        constraints = json.loads(declared["user_preferences"]["data_constraints"])
        assert constraints["cohort"]["review"] == question
        assert constraints["cohort"]["exclude_readmissions" if kind == "patient" else field] is True
        context = build_research_context(research_question=question, cohort=pd.DataFrame({"stay_id": [2], "age": [60]}),
            cohort_name="synthetic", database=database, inclusion_criteria=declared["inclusion_criteria"],
            exclusion_criteria=declared["exclusion_criteria"], user_preferences=declared["user_preferences"])
        return declared, constraints, export_applied_selection(context), context

    declared, constraints, applied, _ = selection()
    assert declared["exclusion_criteria"] == agent_pipeline_runs._host_applied_criteria(study)
    assert constraints["source_selection"]["host_applied"] == declared["exclusion_criteria"]
    assert list(applied.known_applied.exclusion) == declared["exclusion_criteria"]
    scientific.foundation_profile[field] = replace(binding, coordinate_sha256="f" * 64)
    declared, constraints, applied, context = selection()
    assert declared["exclusion_criteria"] == []
    assert constraints["source_selection"]["host_applied"] == [] and not applied.known_applied.exclusion
    spec = PopulationSpec.model_validate({"criteria": [{"id": "c1", "quote": question,
        "source": "question", "role": "include", "kind": field}]})
    (criterion,) = compile_population(spec, context).criteria
    assert criterion.disposition == "requires_extraction"
    assert criterion.reason == f"population_{field}_not_restricted"
    scientific.foundation_profile[field] = binding
    acquisition.cohort_authority_ref = None
    declared, constraints, applied, _ = selection()
    assert declared["exclusion_criteria"] == []
    assert constraints["source_selection"]["host_applied"] == [] and not applied.known_applied.exclusion
    acquisition.cohort_authority_ref = verified.reference
    scientific.metadata_only_planning = True
    declared, constraints, applied, _ = selection()
    assert declared["exclusion_criteria"] == []
    assert constraints["source_selection"]["host_applied"] == [] and not applied.known_applied.exclusion


def test_the_planner_is_shown_the_exclusion_contract() -> None:
    """A field only the host reads is a field the Planner cannot honour."""

    source = inspect.getsource(outbound_safe_context_payload)
    assert '"inclusion_contract": context.cohort.inclusion_criteria' in source
    assert '"exclusion_contract": context.cohort.exclusion_criteria' in source


def test_the_offered_eligibility_options_reach_the_exclusion_channel() -> None:
    """The two halves of this repair have to meet.

    Copilot offers first-admission options; if those never compiled into an
    exclusion, choosing one would still produce a plan with no exclusion.
    """

    by_id = {option["id"]: option for option in cohort_eligibility.ELIGIBILITY_OPTIONS}

    first_admission = _study(**by_id["adults_first_admission"]["cohort"])
    assert _exclusion_criteria(first_admission), "first-admission must exclude"
    assert _inclusion_criteria(first_admission), "and must still state its age floor"

    all_admissions = _study(**by_id["adults_all_admissions"]["cohort"])
    assert _exclusion_criteria(all_admissions) == [], "keeping repeats excludes nobody"

    declined = _study(**by_id["no_eligibility_filter"]["cohort"])
    assert _exclusion_criteria(declined) == []
    assert _inclusion_criteria(declined) == []


def test_an_idea_handoff_claims_no_eligibility_it_was_not_given() -> None:
    """Same rule on the Idea Mining path: do not assert an unchosen criterion.

    The fallback used to send "Adult ICU cohort" as a bound inclusion contract,
    so an idea that never mentioned age reached the Planner claiming an adult
    restriction the export may not apply.
    """

    from easyicu.webserver.ideas.handoff import (
        _inclusion_criteria as idea_inclusion_criteria,
    )

    assert idea_inclusion_criteria({}) == []
    assert idea_inclusion_criteria({"cohort": {}}) == []
    assert idea_inclusion_criteria({"cohort": {"default": "   "}}) == []
    assert idea_inclusion_criteria({"cohort": {"default": "adults on ventilation"}}) == [
        "adults on ventilation"
    ]
    source = Path("src/easyicu/webserver/ideas/handoff.py").read_text(encoding="utf-8")
    assert "Adult ICU cohort from the active prepared" not in source


def test_a_diagnosis_criterion_is_declared_only_when_the_owner_can_run_it() -> None:
    """Declare what the export will execute, not what the conversation typed.

    ``primary_cohort.normalize_execution_cohort`` owns what counts as an
    executable diagnosis predicate; a structured include/exclude list is one in
    its own right. The invariant here is the delegation, not any particular
    answer: a cohort the owner refuses outright declares no criterion, so an
    unexecutable filter never reaches the Planner as a real exclusion stage.
    """

    for cohort in (
        {},
        # the owner rejects these, so nothing may be declared from them
        {"preset": "icd"},  # empty_icd_filter: an ICD cohort with no tokens
        {"preset": "not-a-preset", "include_diagnoses": ["condition-a"]},
    ):
        assert _inclusion_criteria({"cohort": cohort}) == [], cohort
        assert _exclusion_criteria({"cohort": cohort}) == [], cohort

    for cohort in (
        {"include_diagnoses": ["condition-a"], "exclude_diagnoses": ["condition-b"]},
        {
            "icd_enabled": True,
            "include_diagnoses": ["condition-a"],
            "exclude_diagnoses": ["condition-b"],
        },
    ):
        assert _inclusion_criteria({"cohort": cohort}) == [
            "include diagnoses: condition-a"
        ], cohort
        assert _exclusion_criteria({"cohort": cohort}) == [
            "exclude diagnoses: condition-b"
        ], cohort

    explicitly_disabled = {
        "preset": "adult_first",
        "icd_enabled": False,
        "include_diagnoses": ["stale-condition-a"],
        "exclude_diagnoses": ["stale-condition-b"],
    }
    # The preset still executes its adult bound and first stay; only the
    # stale diagnosis lists stay undeclared.
    assert _inclusion_criteria({"cohort": explicitly_disabled}) == ["age range: 18 to *"]
    exclusion = _exclusion_criteria({"cohort": explicitly_disabled})
    assert exclusion == ["each patient's later ICU stays across the bound source"]
    assert primary_cohort.normalize_execution_cohort(explicitly_disabled)[
        "icd_include"
    ] == []

    # the dedicated ICD fields reach their own side too; before this repair the
    # web caller read only the *_diagnoses aliases and dropped these entirely
    icd = _study(preset="icd", icd_include="A41", icd_exclude="T20-T32")
    assert _inclusion_criteria(icd) == ["include diagnoses: A41"]
    assert _exclusion_criteria(icd) == ["exclude diagnoses: T20-T32"]


def test_a_removal_stated_in_prose_does_not_arrive_as_an_inclusion() -> None:
    """A removal stated in prose has its own slot, and is not an applied criterion.

    Real studies state the cohort as one free-text ``review`` blob, and a
    removal written there was delivered as an inclusion criterion. Neither
    wording is executed by the export, so neither is declared as applied.
    """

    study = _study(
        review="adult ICU stays",
        exclusion_statement="stays that ended before the landmark",
    )
    assert _inclusion_criteria(study) == []
    assert _exclusion_criteria(study) == []


def test_both_sides_stay_bounded() -> None:
    study = _study(
        icd_enabled=True,
        exclude_diagnoses=[f"condition-{index}" for index in range(40)],
        include_diagnoses=[f"other-{index}" for index in range(40)],
    )
    assert len(_inclusion_criteria(study)) <= 32
    assert len(_exclusion_criteria(study)) <= 32
    # The roster is bounded by the roster that actually executes
    # (``MAX_DIAGNOSIS_TOKENS``) rather than by a separate literal. A private
    # cap of 20 declared less than the export ran, and the manuscript's
    # inclusion criteria read this declaration.
    joined = _exclusion_criteria(study)[0]
    assert "condition-19" in joined
    assert "condition-39" in joined
    assert len(joined.split(": ", 1)[1].split(", ")) == 40


def test_an_oversized_diagnosis_roster_is_refused_not_trimmed() -> None:
    """A trimmed roster runs a different cohort than the one that was stated.

    ``I20-I52, J09-J18, E10-E50`` is three stated criteria and 84 codes. The
    cap silently kept 64, so E31-E50 never entered the executed queue and
    nothing downstream could tell that it had happened.
    """

    with pytest.raises(primary_cohort.PrimaryCohortContractError) as exc:
        primary_cohort.normalize_execution_cohort(
            {"preset": "icd", "icd_include": "I20-I52, J09-J18, E10-E50"}
        )

    assert exc.value.code == "diagnosis_filter_too_large"
    assert exc.value.detail["token_count"] == 84
    assert exc.value.detail["max_tokens"] == primary_cohort.MAX_DIAGNOSIS_TOKENS
    assert exc.value.detail["dropped_tokens"][0] == "E31"


def test_a_roster_at_the_limit_still_executes() -> None:
    """The refusal is for exceeding the bound, not for approaching it."""

    at_limit = ", ".join(
        f"A{value:02d}" for value in range(10, 10 + primary_cohort.MAX_DIAGNOSIS_TOKENS)
    )
    executed = primary_cohort.normalize_execution_cohort(
        {"preset": "icd", "icd_include": at_limit}
    )

    assert len(executed["icd_include"]) == primary_cohort.MAX_DIAGNOSIS_TOKENS
