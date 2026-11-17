"""Synthetic signed owners retain whole patients from export through stability."""

import hashlib
import json
import numpy as np
import pandas as pd
from easyicu.research_agent.acquisition.patient_grouping import PatientGroupingBinding
from easyicu.research_agent.cohort.materializer import materialize_to_parquet
from easyicu.research_agent.contracts.dependence import PlannedDependenceRequirement
from easyicu.research_agent.contracts.trajectory_design import (
    load_trajectory_design,
    sealed_trajectory_authority_body,
)
from easyicu.research_agent.trajectory.scientific_runtime_authority import (
    build_trajectory_scientific_runtime_authority,
)
from easyicu.research_agent.execution.runners.trajectory_scientific_representation_executor import (
    run_trajectory_scientific_representation,
)
from easyicu.research_agent.execution.runners.trajectory_scientific_candidate_executor import (
    run_trajectory_scientific_candidate_selection,
)
from easyicu.research_agent.execution.runners.trajectory_stability_executor import (
    run_trajectory_stability,
)
from tests.support.typed_export import typed_export


def test_the_signed_trajectory_suite_subsamples_patients_with_all_their_stays(tmp_path):
    p = tmp_path
    rng = np.random.default_rng(451)
    rows = [
        {
            "stay_id": i + 1,
            "charttime": t,
            "age": 60,
            "lact": (2 + 3 * (i // 80)) + rng.normal(0, 0.3),
            "mech_vent": False,
        }
        for i in range(240)
        for t in [2.0, 8.0, 14.0, 20.0]
    ]
    source = typed_export(
        p / "source",
        labs=pd.DataFrame(rows),
        outcomes=pd.DataFrame({"stay_id": range(1, 241), "death": False}),
    )
    mapping = p / "mapping.parquet"
    pd.DataFrame({"stay_id": range(1, 241), "patient": np.arange(240) // 2}).to_parquet(
        mapping, index=False
    )
    binding = PatientGroupingBinding(
        mapping_path=mapping,
        mapping_sha256=hashlib.sha256(mapping.read_bytes()).hexdigest(),
        mapping_stay_column="stay_id",
        mapping_patient_column="patient",
        authority_coordinates={},
    )
    paths = materialize_to_parquet(
        p / "prepared",
        stem="universe",
        database="miiv",
        data_path=source,
        static_concepts=("age",),
        feature_concepts=("lact",),
        outcome_concepts=("death",),
        emit_trajectory=True,
        trajectory_concepts=("lact",),
        trajectory_window=(0, 24),
        **binding.materializer_kwargs(),
    )
    design = load_trajectory_design(
        {
            "coordinate_concepts": ["lact"],
            "window_end_hours": 24,
            "grid_width_hours": 6,
            "stability_resamples": 2,
            "candidate_cluster_max": 4,
            "minimum_mean_stability": 0.0,
        }
    )
    body = sealed_trajectory_authority_body(design, protocol_content_sha256="a" * 64)
    body["dependence"] = PlannedDependenceRequirement(
        group_source="patient_stay_id",
        group_derivation="prefix_before_delimiter",
        delimiter=":s",
    ).model_dump(mode="json")
    authority = build_trajectory_scientific_runtime_authority(body)
    rep = p / "representation"
    r = run_trajectory_scientific_representation(
        authority=authority,
        runtime_projection_sha256="a" * 64,
        trajectory_path=paths["trajectory"],
        cohort_path=paths["parquet"],
        out_dir=rep,
    )

    def bindings(root, outputs, wanted=None):
        return {
            key: {
                "evidence_id": key.replace(":", "_"),
                "relative_path": str((root / name).relative_to(p)),
                "sha256": hashlib.sha256((root / name).read_bytes()).hexdigest(),
            }
            for key, name in outputs.items()
            if wanted is None or key in wanted
        }

    inputs = bindings(
        rep,
        r["output_files"],
        {
            "artifact:trajectory_representation",
            "manifest:trajectory_representation_schema",
        },
    )
    candidate = p / "candidate"
    c = run_trajectory_scientific_candidate_selection(
        authority=authority,
        runtime_projection_sha256="a" * 64,
        out_dir=candidate,
        run_dir=p,
        resolved_inputs={"inputs": inputs},
    )
    inputs.update(
        bindings(
            candidate,
            c["output_files"],
            {
                "artifact:candidate_cluster_assignments",
                "manifest:cluster_selection",
                "manifest:candidate_cluster_solution_schema",
            },
        )
    )
    stability = p / "stability"
    s = run_trajectory_stability(
        spec=authority.stability_spec,
        out_dir=stability,
        run_dir=p,
        resolved_inputs={"inputs": inputs},
        scientific_runtime_authority=authority,
        runtime_projection_sha256="a" * 64,
        include_characterization=True,
    )
    assert s["status"] != "failed_closed", s["errors"]
    spec = json.loads((stability / "cluster_stability_spec.json").read_text())
    assert spec["sampling_unit"] == "patient"
    assert spec["bound_sample_n"] == 96
    assignment = pd.read_csv(stability / "cluster_stability_assignments.csv")
    for _, sample in assignment.groupby("resample_id"):
        patients = sample.stay_id.str.split(":s").str[0].value_counts()
        assert len(patients) == 96
        assert patients.eq(2).all()
