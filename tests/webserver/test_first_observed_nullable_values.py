"""Real synthetic native materialization and raw-value reconciliation."""

from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
import json
import pandas as pd
import pytest
from tests.support.typed_export import typed_export, metadata_binding
from easyicu.concept.metadata_projection import ConceptColumnRole
from easyicu.concept.metadata_sidecar import (
    ColumnMetadataSidecar,
    write_content_addressed_sidecar,
)
from easyicu.research_agent.acquisition.first_icu_stay import (
    FirstObservedIcuCoordinate,
    FLAG_COLUMN,
    REASON_COLUMN,
)
from easyicu.research_agent.acquisition.foundation import acquire_universe_for_question
from easyicu.research_agent.intake.export_package import (
    open_export_package,
    host_deferred_export_read_scope,
)
from easyicu.webserver.first_observed_deferred_execution import (
    _verify_raw_patient_values,
)
from tests.support.published_export import record_member_digests


def test_nullable_hospital_outcome_is_preserved_and_raw_mismatch_refused(tmp_path):
    labs = pd.DataFrame(
        {
            "stay_id": [1, 1, 2, 3],
            "charttime": [1.0, 2.0, 1.0, 1.0],
            "age": [50, 50, 60, 70],
            "lact": [1.0, 2.0, 3.0, 4.0],
            "mech_vent": [False, True, False, False],
            "sex": ["Female", "Female", "Female", "Male"],
        }
    )
    outcomes = pd.DataFrame(
        {"stay_id": [1, 2, 3], "death": pd.Series([False, None, True], dtype="boolean")}
    )
    root = typed_export(
        tmp_path / "export", database="eicu", labs=labs, outcomes=outcomes
    )
    manifest_path = root / "_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    sidecar = ColumnMetadataSidecar.from_dict(
        json.loads((root / manifest["column_metadata"]["file"]).read_text())
    )
    lab = next(f for f in sidecar.files if f.module == "labs")
    lab = replace(
        lab,
        columns={
            **lab.columns,
            "sex": metadata_binding(
                "sex", "sex", ConceptColumnRole.VALUE, database="eicu"
            ),
        },
    )
    sidecar = replace(
        sidecar, files=tuple(lab if f.module == "labs" else f for f in sidecar.files)
    )
    manifest["column_metadata"] = write_content_addressed_sidecar(
        root, sidecar
    ).to_dict()
    manifest["concept_selection"]["modules"]["labs"].append("sex")
    manifest["files"][0]["concept_ids"].append("sex")
    manifest["files"][0]["concepts"] += 1
    manifest["files"][0]["column_metadata_columns"].append("sex")
    manifest_path.write_text(json.dumps(manifest))
    record_member_digests(root)  # the fixture re-publishes its rewritten members
    raw = tmp_path / "patient.parquet"
    pd.DataFrame(
        {
            "patientunitstayid": [1, 2, 3],
            "gender": ["Female", "Female", "Male"],
            "hospitaldischargestatus": ["Alive", None, "Expired"],
            "unitdischargeoffset": [2880, 3000, 4000],
        }
    ).to_parquet(raw, index=False)
    patient_sha = sha256(raw.read_bytes()).hexdigest()
    coord = tmp_path / "coordinate.parquet"
    pd.DataFrame(
        {
            "stay_id": [1, 2, 3],
            FLAG_COLUMN: [True, True, True],
            REASON_COLUMN: ["kept"] * 3,
        }
    ).to_parquet(coord, index=False)
    with open_export_package(root) as package:
        manifest_sha = package.manifest_sha256
        columns = {f.relative_path: tuple(f.columns) for f in package.files}
    coordinate = FirstObservedIcuCoordinate(
        path=coord,
        sha256=sha256(coord.read_bytes()).hexdigest(),
        archive_sha256="a" * 64,
        patient_table_sha256=patient_sha,
        export_manifest_sha256=manifest_sha,
        plan_sha256="b" * 64,
        preparation_checkpoint_sha256="c" * 64,
    )
    with host_deferred_export_read_scope(
        manifest_sha256=manifest_sha, columns_by_file=columns
    ):
        result = acquire_universe_for_question(
            export_dir=root,
            question="synthetic",
            llm=None,
            output_dir=tmp_path / "result",
            stem="cohort",
            target_outcome="death",
            primary_exposure_concept="sex",
            outcome_concepts=["death"],
            required_feature_concepts=["sex"],
            static_concepts=["sex"],
            allowed_modules=["labs", "outcomes"],
            concept_selection_authority="host_exact",
            database="eicu",
            emit_trajectory=False,
            first_observed_icu_coordinate=coordinate,
        )
    assert not result.blocked
    frame = pd.read_parquet(result.universe_path).sort_values("stay_id")
    assert frame["death"].isna().tolist() == [False, True, False]
    assert frame["death"].dropna().astype(bool).tolist() == [False, True]
    declaration = {
        "source": "registered_eicu_demo_patient_table",
        "patient_table_sha256": patient_sha,
        "columns": ["patientunitstayid", "gender", "hospitaldischargestatus"],
        "purpose": "post_approval_independent_native_export_value_comparison",
        "preserve_unknown_outcome": True,
        "post_approval_only": True,
    }
    source = SimpleNamespace(table_paths={"patient": str(raw)})
    _verify_raw_patient_values(
        cohort_path=result.universe_path,
        raw_source=source,
        coordinate=coordinate,
        declaration=declaration,
    )
    # Independently exercise the same real distribution owner: known-only denominator.
    from tests.research_agent.planning.descriptive_interval_fixtures import (
        descriptive_context,
        descriptive_plan,
    )
    from easyicu.research_agent.execution.runners.exposure_outcome_distribution_executor import (
        _distribution_rows,
    )

    spec = (
        descriptive_plan(descriptive_context(repeats=False))
        .steps[-1]
        .exposure_outcome_distribution_spec
    )
    payload = spec.model_dump(mode="python")
    payload.update(
        {
            "exposure": "sex",
            "outcome": "death",
            "exposure_levels": ["Female", "Male"],
            "outcome_levels": [0, 1],
            "risk_difference_contrast": None,
            "denominator_policy": "observed_outcome_rows",
            "missing_outcome_policy": "exclude_from_denominator",
        }
    )
    spec = type(spec).model_validate(payload)
    rows = _distribution_rows(frame, spec=spec)
    female = next(
        row
        for row in rows
        if row["row_role"] == "exposure_level" and row["exposure_level"] == "Female"
    )
    assert (
        female["n_rows"],
        female["outcome_missing_n"],
        female["outcome_denominator"],
        female["outcome_events"],
    ) == (2, 1, 1, 0)
    # Replacing the exported outcome while retaining the same manifest is now
    # refused at intake: the member no longer matches its recorded digest.
    original_outcomes = (root / "outcomes.parquet").read_bytes()
    outcomes_changed = outcomes.copy()
    outcomes_changed.loc[1, "death"] = False
    outcomes_changed.to_parquet(root / "outcomes.parquet", index=False)
    with host_deferred_export_read_scope(
        manifest_sha256=manifest_sha, columns_by_file=columns
    ):
        with pytest.raises(Exception) as replaced:
            acquire_universe_for_question(
                export_dir=root,
                question="synthetic",
                llm=None,
                output_dir=tmp_path / "mismatch",
                stem="cohort",
                target_outcome="death",
                primary_exposure_concept="sex",
                outcome_concepts=["death"],
                required_feature_concepts=["sex"],
                static_concepts=["sex"],
                allowed_modules=["labs", "outcomes"],
                concept_selection_authority="host_exact",
                database="eicu",
                emit_trajectory=False,
                first_observed_icu_coordinate=coordinate,
            )
    chain, cause = [], replaced.value
    while cause is not None:
        chain.append(f"{type(cause).__name__}: {cause}")
        cause = cause.__cause__ or cause.__context__
    assert any("member_sha256_mismatch" in item or "SHA-256" in item for item in chain), chain
    (root / "outcomes.parquet").write_bytes(original_outcomes)
    # The independent source-value check (unknown -> survivor on the raw side)
    # is exercised next against the accepted universe.
    changed = pd.read_parquet(raw)
    changed.loc[1, "hospitaldischargestatus"] = "Alive"
    changed.to_parquet(raw, index=False)
    with pytest.raises(ValueError):
        _verify_raw_patient_values(
            cohort_path=result.universe_path,
            raw_source=source,
            coordinate=coordinate,
            declaration=declaration,
        )

    # Even with a newly bound raw digest, unequal, missing and duplicate raw
    # records must fail; this checks value/identity comparison, not only hashes.
    base = pd.DataFrame(
        {
            "patientunitstayid": [1, 2, 3],
            "gender": ["Female", "Female", "Male"],
            "hospitaldischargestatus": ["Alive", None, "Expired"],
        }
    )
    for change in (
        "unknown_to_survivor",
        "sex_other",
        "sex_missing",
        "duplicate",
        "missing",
    ):
        altered = base.copy()
        if change == "unknown_to_survivor":
            altered.loc[1, "hospitaldischargestatus"] = "Alive"
        elif change == "sex_other":
            altered.loc[1, "gender"] = "Other"
        elif change == "sex_missing":
            altered.loc[1, "gender"] = None
        elif change == "duplicate":
            altered = pd.concat([altered, altered.iloc[[0]]], ignore_index=True)
        else:
            altered = altered.iloc[:-1]
        altered.to_parquet(raw, index=False)
        digest = sha256(raw.read_bytes()).hexdigest()
        new_coordinate = replace(coordinate, patient_table_sha256=digest)
        new_declaration = {**declaration, "patient_table_sha256": digest}
        with pytest.raises(ValueError):
            _verify_raw_patient_values(
                cohort_path=result.universe_path,
                raw_source=source,
                coordinate=new_coordinate,
                declaration=new_declaration,
            )


def test_raw_eicu_status_callback_preserves_long_offset_and_unknown():
    from easyicu.concept import ConceptResolver
    from easyicu.resources import load_data_sources, load_dictionary
    from easyicu.table import ICUTable

    raw = pd.DataFrame(
        {
            "patientunitstayid": [1, 2, 3],
            "unitdischargeoffset": [2880.0, 3000.0, 4000.0],
            "hospitaldischargestatus": ["Alive", None, "Expired"],
        }
    )
    source = SimpleNamespace(
        config=load_data_sources().get("eicu_demo"),
        load_table=lambda *_args, **_kwargs: ICUTable(
            raw.copy(),
            id_columns=["patientunitstayid"],
            index_column="unitdischargeoffset",
        ),
        clear_cache=lambda: None,
    )
    result = ConceptResolver(load_dictionary()).load_concepts(
        ["death"],
        source,
        merge=True,
        r_compatible=True,
        interval=pd.Timedelta(hours=1),
        align_to_admission=False,
        verbose=False,
        concept_workers=1,
    )
    actual = result.sort_values("patientunitstayid").death.reset_index(drop=True)
    pd.testing.assert_series_equal(
        actual, pd.Series([False, pd.NA, True], dtype="boolean", name="death")
    )
