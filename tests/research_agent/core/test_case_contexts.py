"""Case-context tests for the lactate-MAP-vasopressor demo.

These tests pin the paper's main differentiator: the agent does not see
only a CSV. It receives a formal EasyICU context contract with source
files, missingness semantics and unsafe transformations.
"""

from __future__ import annotations

import pandas as pd

from easyicu.research_agent.planning.scientific_review import requested_outcomes


def test_lactate_map_vaso_context_injects_source_and_rules(ra):
    cohort = pd.DataFrame({
        "stay_id": [1, 2, 3, 4],
        "age": [60, 70, 80, 65],
        "death": [0, 1, 0, 1],
        "los_icu": [2.0, 3.0, 4.0, 5.0],
        "los_hosp": [4.0, 5.0, 6.0, 7.0],
        "lactate_max_24h": [1.5, 3.2, None, 5.0],
        "lactate_measured_24h": [1, 1, 0, 1],
        "map_min_24h": [72, 68, 80, 60],
        "vaso_any_24h": [0, 0, 0, 1],
    })
    source_manifest = {
        "concept_sources": {
            "death": "outcome_death_los_hosp_los_icu.parquet",
            "lactate_max_24h": "blood_gas_be_cai_lact_methb_etc8.parquet",
            "map_min_24h": "vitals_dbp_hr_map_resp_etc7.parquet",
            "vaso_any_24h": "vasopressors_adh_rate_dobu60_dobu_dur_dobu_rate_etc17.parquet",
        }
    }

    ctx = ra.build_lactate_map_vaso_research_context(
        cohort=cohort,
        source_manifest=source_manifest,
        database="miiv",
    )

    lact = ctx.variable("lactate_max_24h")
    assert set(ctx.cohort.outcome_columns) == {"death", "los_icu", "los_hosp"}
    assert requested_outcomes(ctx) == ("death",)
    assert lact is not None
    assert lact.derived_from_concepts == ["lact"]
    assert lact.source_files == ["blood_gas_be_cai_lact_methb_etc8.parquet"]
    assert lact.analysis_window == "first_24h"
    assert "clinically triggered" in (lact.missingness_semantics or "")
    assert any("missing lactate with 0" in rule for rule in lact.forbidden_transformations)
    assert any("measurement frequency" in note for note in lact.cross_database_notes)

    vaso = ctx.variable("vaso_any_24h")
    assert vaso is not None
    assert vaso.role.value == "intervention"
    assert any("confounded by indication" in pitfall for pitfall in vaso.pitfalls)


def test_lactate_map_vaso_context_ablation_table_counts_added_context(ra):
    cohort = pd.DataFrame({
        "stay_id": [1, 2],
        "death": [0, 1],
        "lactate_max_24h": [1.5, 3.2],
        "lactate_measured_24h": [1, 1],
        "map_min_24h": [72, 60],
        "vaso_any_24h": [0, 1],
    })
    source_manifest = {
        "concept_sources": {
            "death": "outcome.parquet",
            "lactate_max_24h": "blood_gas.parquet",
            "map_min_24h": "vitals.parquet",
            "vaso_any_24h": "vasopressors.parquet",
        }
    }
    table = ra.build_lactate_map_vaso_context_ablation_table(
        cohort=cohort,
        source_manifest=source_manifest,
        database="miiv",
    ).set_index("context")

    assert table.loc["generic_csv_context", "variables_with_source_files"] == 0
    assert table.loc["easyicu_icu_context", "variables_with_source_files"] >= 4
    assert table.loc["generic_csv_context", "variables_with_forbidden_transformations"] == 0
    assert table.loc["easyicu_icu_context", "variables_with_forbidden_transformations"] >= 4
    assert table.loc["easyicu_icu_context", "time_windows"] > table.loc["generic_csv_context", "time_windows"]
