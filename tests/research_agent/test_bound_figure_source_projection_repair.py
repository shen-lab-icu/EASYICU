"""Regression tests for exact per-parent figure source-data projections."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from easyicu.research_agent.repair_registry import (
    RepairClass,
    automatic_repair_allowed,
    repair_metadata_for,
)
from easyicu.research_agent.repairs.source import deterministic_contract_repair

_FINDING = {
    "validator": "figure_source_data",
    "severity": "error",
    "detail": {
        "reason": "incomplete_source_lineage_coverage",
        "missing_bound_tables": ["parent_a.csv", "parent_b.csv"],
        "missing_bound_statistics": [],
    },
}


def _script(out_dir: Path) -> str:
    return f"""
from pathlib import Path
import pandas as pd

OUT_DIR = Path({str(out_dir)!r})
EXPECTED_INPUTS = ["table:parent_a", "table:parent_b"]

def make_figure_contract(**kwargs):
    return kwargs

def save_publication_figure(*, fig, out_dir, stem, contract):
    return None

def load_bound_table(input_key):
    frame = (
        pd.DataFrame({{"row_id": ["a"], "estimate": [1.25]}})
        if input_key.endswith("parent_a")
        else pd.DataFrame({{"row_id": ["b"], "count": [17]}})
    )
    source_name = "parent_a.csv" if input_key.endswith("parent_a") else "parent_b.csv"
    return (
        frame,
        {{"input_key": input_key}},
        {{"product": input_key, "relative_path": f"evidence/{{source_name}}"}},
    )

def main():
    tables = {{}}
    input_bindings = []
    input_records = {{}}
    for input_key in EXPECTED_INPUTS:
        frame, binding, record = load_bound_table(input_key)
        tables[input_key] = frame
        input_bindings.append(binding)
        input_records[input_key] = record

    mixed = pd.DataFrame({{"source_row_index": [0], "value": [1.25]}})
    source_stem = "overview_source_data.csv"
    mixed.to_csv(OUT_DIR / source_stem, index=False)

    contract = make_figure_contract(
        figure_id="figure:overview",
        source_data=source_stem,
    )
    save_publication_figure(
        fig=None,
        out_dir=OUT_DIR,
        stem="overview",
        contract=contract,
    )
    step_summary = {{
        "figure_files": ["overview.png"],
        "source_data_files": [source_stem],
    }}
    return contract, step_summary
"""


def test_bound_figure_repair_projects_each_loaded_parent_without_renaming(
    tmp_path: Path,
) -> None:
    code = _script(tmp_path)

    repair = deterministic_contract_repair(code=code, findings=[_FINDING])

    assert repair is not None
    repair_id, repaired = repair
    assert repair_id == "bound_figure_source_projection_v2"
    namespace: dict[str, object] = {}
    exec(repaired, namespace)
    contract, summary = namespace["main"]()

    declared = contract["source_data"]
    assert declared == summary["source_data_files"]
    assert len(declared) == 2
    assert "overview_source_data.csv" not in declared
    observed = [pd.read_csv(tmp_path / filename) for filename in declared]
    assert observed[0].to_dict(orient="records") == [
        {
            "source_row_index": 0,
            "source_table": "parent_a.csv",
            "row_id": "a",
            "estimate": 1.25,
        }
    ]
    assert observed[1].to_dict(orient="records") == [
        {
            "source_row_index": 0,
            "source_table": "parent_b.csv",
            "row_id": "b",
            "count": 17,
        }
    ]


def test_bound_figure_repair_requires_unambiguous_loader_and_bundle_contract(
    tmp_path: Path,
) -> None:
    code = _script(tmp_path).replace(
        "        tables[input_key] = frame\n",
        "        tables[input_key] = frame\n        shadow_tables[input_key] = frame\n",
    )

    assert deterministic_contract_repair(code=code, findings=[_FINDING]) is None
    assert (
        deterministic_contract_repair(
            code=_script(tmp_path),
            findings=[
                {
                    **_FINDING,
                    "detail": {
                        **_FINDING["detail"],
                        "missing_bound_statistics": ["statistic:primary_effect"],
                    },
                }
            ],
        )
        is None
    )


def test_direct_bound_figure_repair_projects_tables_that_also_bind_statistics(
    tmp_path: Path,
) -> None:
    code = f"""
from pathlib import Path
import pandas as pd

OUT_DIR = Path({str(tmp_path)!r})

def _export_figure_source_data(source_data):
    if isinstance(source_data, (list, tuple)) and all(
        isinstance(item, str) for item in source_data
    ):
        return list(source_data)
    exported = []
    for key, frame in source_data.items():
        name = f"{{key}}_figure_source.csv"
        frame.to_csv(OUT_DIR / name, index=False)
        exported.append(name)
    return exported

def make_figure_contract(**kwargs):
    kwargs["source_data"] = _export_figure_source_data(kwargs["source_data"])
    return kwargs

def save_publication_figure(*, fig, out_dir, stem, contract):
    return None

def main():
    loaded = {{
        "table:parent_a": (
            pd.DataFrame({{"row_id": ["a"], "primary_or": [1.25]}}),
            Path("parent_a.csv"),
        ),
        "table:parent_b": (
            pd.DataFrame({{"row_id": ["b"], "complete_case_n": [17]}}),
            Path("parent_b.csv"),
        ),
    }}
    parent_a = loaded["table:parent_a"][0].copy()
    parent_b = loaded["table:parent_b"][0].copy()
    mixed = pd.DataFrame({{"panel": ["A", "B"], "value": [1.25, 17]}})
    mixed.to_csv(OUT_DIR / "overview_source_data.csv", index=False)
    contract = make_figure_contract(
        figure_id="figure:overview",
        source_data="overview_source_data.csv",
    )
    save_publication_figure(
        fig=None,
        out_dir=OUT_DIR,
        stem="overview",
        contract=contract,
    )
    return contract
"""
    finding = {
        **_FINDING,
        "detail": {
            **_FINDING["detail"],
            "missing_bound_statistics": [
                "statistic:primary_or",
                "statistic:complete_case_n",
            ],
        },
    }

    repair = deterministic_contract_repair(code=code, findings=[finding])

    assert repair is not None
    repair_id, repaired = repair
    assert repair_id == "direct_bound_figure_source_projection_v1"
    namespace: dict[str, object] = {}
    exec(repaired, namespace)
    contract = namespace["main"]()
    assert contract["source_data"] == [
        "bound_000_parent_a_source_data.csv",
        "bound_001_parent_b_source_data.csv",
    ]
    assert pd.read_csv(tmp_path / "bound_000_parent_a_source_data.csv").to_dict(
        orient="records"
    ) == [
        {
            "row_id": "a",
            "primary_or": 1.25,
            "source_row_index": 0,
            "source_table": "parent_a.csv",
        }
    ]
    assert pd.read_csv(tmp_path / "bound_001_parent_b_source_data.csv").to_dict(
        orient="records"
    ) == [
        {
            "row_id": "b",
            "complete_case_n": 17,
            "source_row_index": 0,
            "source_table": "parent_b.csv",
        }
    ]


def test_direct_bound_figure_repair_supports_typed_loader_tuple_and_path_name(
    tmp_path: Path,
) -> None:
    code = f"""
from pathlib import Path
import pandas as pd

OUT_DIR = Path({str(tmp_path)!r})

def load_bound_table(input_key, manifest):
    if input_key == "table:absolute_risk_context":
        return (
            pd.DataFrame({{"group": ["a"], "estimate": [1.25]}}),
            {{"input_key": input_key}},
            Path("absolute_risk_context.csv"),
        )
    return (
        pd.DataFrame({{"group": ["b"], "count": [17]}}),
        {{"input_key": input_key}},
        Path("exposure_outcome_distribution.csv"),
    )

def make_figure_contract(**kwargs):
    return kwargs

def save_publication_figure(*, fig, out_dir, stem, contract):
    return None

def main():
    manifest = {{}}
    absolute_df, absolute_receipt, absolute_path = load_bound_table(
        "table:absolute_risk_context", manifest
    )
    distribution_df, distribution_receipt, distribution_path = load_bound_table(
        "table:exposure_outcome_distribution", manifest
    )
    source_path = OUT_DIR / "combined_source_data.csv"
    contract = make_figure_contract(
        figure_id="figure:overview",
        source_data=source_path.name,
    )
    save_publication_figure(
        fig=None,
        out_dir=OUT_DIR,
        stem="overview",
        contract=contract,
    )
    return contract
"""
    finding = {
        **_FINDING,
        "detail": {
            **_FINDING["detail"],
            "missing_bound_tables": [
                "absolute_risk_context.csv",
                "exposure_outcome_distribution.csv",
            ],
        },
    }

    repair = deterministic_contract_repair(code=code, findings=[finding])

    assert repair is not None
    repair_id, repaired = repair
    assert repair_id == "direct_bound_figure_source_projection_v1"
    namespace: dict[str, object] = {}
    exec(repaired, namespace)
    contract = namespace["main"]()
    assert contract["source_data"] == [
        "bound_000_absolute_risk_context_source_data.csv",
        "bound_001_exposure_outcome_distribution_source_data.csv",
    ]
    absolute = pd.read_csv(tmp_path / contract["source_data"][0])
    distribution = pd.read_csv(tmp_path / contract["source_data"][1])
    assert absolute["estimate"].tolist() == [1.25]
    assert distribution["count"].tolist() == [17]


def test_direct_bound_figure_repair_resolves_one_unkeyed_parent_from_finding(
    tmp_path: Path,
) -> None:
    code = f"""
from pathlib import Path
import pandas as pd

OUT_DIR = Path({str(tmp_path)!r})
TABLE_INPUT = "table:absolute_risk_context"

def load_bound_table(manifest, run_dir):
    return (
        pd.DataFrame({{
            "row_type": ["exposure_group"],
            "exposure_group": ["positive"],
            "prevalence_pct": [100.0],
            "prevalence_numerator": [20],
            "prevalence_denominator": [20],
        }}),
        Path("absolute_risk_context.csv"),
        {{"input_key": TABLE_INPUT}},
    )

def make_figure_contract(**kwargs):
    return kwargs

def save_publication_figure(*, fig, out_dir, stem, contract):
    return None

def main():
    manifest = {{}}
    run_dir = Path(".")
    table, table_path, binding = load_bound_table(manifest, run_dir)
    plotted = table.rename(columns={{"prevalence_pct": "value_pct"}})
    source_data_name = "prevalence_mortality_source_data.csv"
    source_path = OUT_DIR / source_data_name
    plotted.to_csv(source_path, index=False)
    contract = make_figure_contract(
        figure_id="figure:prevalence_mortality",
        source_data=source_data_name,
    )
    save_publication_figure(
        fig=None,
        out_dir=OUT_DIR,
        stem="prevalence_mortality",
        contract=contract,
    )
    return contract
"""
    finding = {
        **_FINDING,
        "detail": {
            **_FINDING["detail"],
            "missing_bound_tables": ["absolute_risk_context.csv"],
        },
    }

    repair = deterministic_contract_repair(code=code, findings=[finding])

    assert repair is not None
    repair_id, repaired = repair
    assert repair_id == "direct_bound_figure_source_projection_v1"
    namespace: dict[str, object] = {}
    exec(repaired, namespace)
    contract = namespace["main"]()
    assert contract["source_data"] == [
        "bound_000_absolute_risk_context_source_data.csv"
    ]
    projection = pd.read_csv(tmp_path / contract["source_data"][0])
    assert projection.to_dict(orient="records") == [
        {
            "source_row_index": 0,
            "source_table": "absolute_risk_context.csv",
            "row_type": "exposure_group",
            "exposure_group": "positive",
            "prevalence_pct": 100.0,
            "prevalence_numerator": 20,
            "prevalence_denominator": 20,
        }
    ]


def test_direct_bound_figure_repair_uses_frame_returned_by_typed_loader(
    tmp_path: Path,
) -> None:
    parent_path = tmp_path / "feature_quality_scaling.csv"
    pd.DataFrame({"feature": ["heart_rate"], "n_total": [10]}).to_csv(
        parent_path, index=False
    )
    code = f"""
from pathlib import Path
import pandas as pd

OUT_DIR = Path({str(tmp_path)!r})

def _load_typed_table(manifest, input_key):
    source_path = Path({str(parent_path)!r})
    loaded_df = pd.read_csv(source_path)
    return loaded_df, {{"input_key": input_key, "source_table": source_path.name}}

def make_figure_contract(**kwargs):
    return kwargs

def save_publication_figure(*, fig, out_dir, stem, contract):
    return None

def main():
    input_key = "table:feature_quality_scaling"
    df, input_receipt = _load_typed_table({{}}, input_key)
    source_name = "feature_quality_source_data.csv"
    df.assign(value=df["n_total"] * 2).to_csv(OUT_DIR / source_name, index=False)
    contract = make_figure_contract(
        figure_id="figure:feature_quality",
        source_data=source_name,
    )
    save_publication_figure(
        fig=None,
        out_dir=OUT_DIR,
        stem="feature_quality",
        contract=contract,
    )
    return contract
"""
    finding = {
        "validator": "figure_source_data",
        "severity": "error",
        "detail": {
            "best_mismatch": {
                "reason": "no_verifiable_values",
                "upstream_table": "feature_quality_scaling.csv",
            }
        },
    }

    repair = deterministic_contract_repair(code=code, findings=[finding])

    assert repair is not None
    repair_id, repaired = repair
    assert repair_id == "direct_bound_figure_source_projection_v1"
    # The loader's local ``loaded_df`` must not be used in the caller's
    # projection; the tuple-returned ``df`` is the only frame in scope there.
    assert "_easyicu_direct_bound_frame = df.copy(deep=True)" in repaired
    assert "_easyicu_direct_bound_frame = loaded_df.copy(deep=True)" not in repaired
    namespace: dict[str, object] = {}
    exec(repaired, namespace)
    contract = namespace["main"]()
    assert contract["source_data"] == [
        "bound_000_feature_quality_scaling_source_data.csv"
    ]
    projection = pd.read_csv(tmp_path / contract["source_data"][0])
    assert projection.to_dict(orient="records") == [
        {
            "source_row_index": 0,
            "source_table": "feature_quality_scaling.csv",
            "feature": "heart_rate",
            "n_total": 10,
        }
    ]


def test_direct_bound_figure_repair_resolves_single_manual_tabular_loader(
    tmp_path: Path,
) -> None:
    code = f"""
from pathlib import Path
import pandas as pd

OUT_DIR = Path({str(tmp_path)!r})

def _load_tabular(table_path):
    assert table_path.name == "absolute_risk_context.csv"
    return pd.DataFrame({{
        "row_type": ["overall", "exposure_group"],
        "exposure_group": ["overall", "positive"],
        "mortality_numerator": [30, 20],
        "mortality_denominator": [100, 40],
        "mortality_pct": [30.0, 50.0],
    }})

def make_figure_contract(**kwargs):
    return kwargs

def save_publication_figure(*, fig, out_dir, stem, contract):
    return None

def main():
    table_path = Path("absolute_risk_context.csv")
    df = _load_tabular(table_path)
    plotted = df.rename(columns={{"mortality_pct": "value_pct"}})
    source_path = OUT_DIR / "prevalence_mortality_source_data.csv"
    plotted.to_csv(source_path, index=False)
    contract = make_figure_contract(
        figure_id="figure:prevalence_mortality",
        source_data=source_path.name,
    )
    save_publication_figure(
        fig=None,
        out_dir=OUT_DIR,
        stem="prevalence_mortality",
        contract=contract,
    )
    return contract
"""
    finding = {
        **_FINDING,
        "detail": {
            **_FINDING["detail"],
            "missing_bound_tables": ["absolute_risk_context.csv"],
        },
    }

    repair = deterministic_contract_repair(code=code, findings=[finding])

    assert repair is not None
    repair_id, repaired = repair
    assert repair_id == "direct_bound_figure_source_projection_v1"
    namespace: dict[str, object] = {}
    exec(repaired, namespace)
    contract = namespace["main"]()
    assert contract["source_data"] == [
        "bound_000_absolute_risk_context_source_data.csv"
    ]
    projection = pd.read_csv(tmp_path / contract["source_data"][0])
    assert projection.to_dict(orient="records") == [
        {
            "source_row_index": 0,
            "source_table": "absolute_risk_context.csv",
            "row_type": "overall",
            "exposure_group": "overall",
            "mortality_numerator": 30,
            "mortality_denominator": 100,
            "mortality_pct": 30.0,
        },
        {
            "source_row_index": 1,
            "source_table": "absolute_risk_context.csv",
            "row_type": "exposure_group",
            "exposure_group": "positive",
            "mortality_numerator": 20,
            "mortality_denominator": 40,
            "mortality_pct": 50.0,
        },
    ]


def test_direct_bound_figure_repair_projects_manual_reader_named_as_source_table(
    tmp_path: Path,
) -> None:
    parent_path = tmp_path / "phenotype_profiles.csv"
    pd.DataFrame(
        {
            "cluster_label": [0, 1],
            "variable": ["heart_rate", "heart_rate"],
            "median": [82.0, 96.0],
            "n": [90, 10],
        }
    ).to_csv(parent_path, index=False)
    cohort_path = tmp_path / "cohort.csv"
    pd.DataFrame({"stay_id": [1, 2]}).to_csv(cohort_path, index=False)
    code = f"""
from pathlib import Path
import pandas as pd

OUT_DIR = Path({str(tmp_path)!r})

def make_figure_contract(**kwargs):
    return kwargs

def save_publication_figure(*, fig, out_dir, stem, contract):
    return None

def main():
    cohort_path = Path({str(cohort_path)!r})
    input_path = Path({str(parent_path)!r})
    cohort = pd.read_csv(cohort_path)
    profiles = pd.read_csv(input_path)
    source_table_name = input_path.name
    derived = profiles.assign(standardized_median=[-1.0, 1.0])
    profile_name = "cluster_profile_source_data.csv"
    availability_name = "cluster_availability_source_data.csv"
    derived.to_csv(OUT_DIR / profile_name, index=False)
    derived.to_csv(OUT_DIR / availability_name, index=False)
    contract = make_figure_contract(
        figure_id="figure:cluster_profile",
        source_data=[profile_name, availability_name],
    )
    save_publication_figure(
        fig=None,
        out_dir=OUT_DIR,
        stem="cluster_profile",
        contract=contract,
    )
    step_summary = {{
        "figure_files": ["cluster_profile.png"],
        "source_data": [profile_name, availability_name],
        "source_table": source_table_name,
        "cohort_rows": len(cohort),
    }}
    return contract, step_summary
"""
    finding = {
        **_FINDING,
        "detail": {
            **_FINDING["detail"],
            "missing_bound_tables": ["phenotype_profiles.csv"],
        },
    }

    repair = deterministic_contract_repair(code=code, findings=[finding])

    assert repair is not None
    repair_id, repaired = repair
    assert repair_id == "direct_bound_figure_source_projection_v1"
    namespace: dict[str, object] = {}
    exec(repaired, namespace)
    contract, summary = namespace["main"]()
    assert (
        contract["source_data"]
        == summary["source_data"]
        == ["bound_000_phenotype_profiles_source_data.csv"]
    )
    projection = pd.read_csv(tmp_path / contract["source_data"][0])
    assert projection.to_dict(orient="records") == [
        {
            "cluster_label": 0,
            "variable": "heart_rate",
            "median": 82.0,
            "n": 90,
            "source_row_index": 0,
            "source_table": "phenotype_profiles.csv",
        },
        {
            "cluster_label": 1,
            "variable": "heart_rate",
            "median": 96.0,
            "n": 10,
            "source_row_index": 1,
            "source_table": "phenotype_profiles.csv",
        },
    ]


def test_direct_bound_figure_repair_uses_unverifiable_source_parent_receipt(
    tmp_path: Path,
) -> None:
    parent_path = tmp_path / "phenotype_profiles.csv"
    pd.DataFrame(
        {
            "cluster_label": [0, 1],
            "median": [82.0, 96.0],
            "n_nonmissing": [85, 9],
            "n": [90, 10],
        }
    ).to_csv(parent_path, index=False)
    code = f"""
from pathlib import Path
import pandas as pd

OUT_DIR = Path({str(tmp_path)!r})

def make_figure_contract(**kwargs):
    return kwargs

def save_publication_figure(*, fig, out_dir, stem, contract):
    return None

def main():
    input_path = Path({str(parent_path)!r})
    if input_path.suffix == ".csv":
        profiles = pd.read_csv(input_path)
    else:
        profiles = pd.read_parquet(input_path)
    source_table_name = input_path.name
    availability = profiles.assign(
        availability_fraction=profiles["n_nonmissing"] / profiles["n"]
    )
    source_name = "cluster_availability_source_data.csv"
    availability.to_csv(OUT_DIR / source_name, index=False)
    contract = make_figure_contract(
        figure_id="figure:cluster_profile",
        source_data=[source_name],
    )
    save_publication_figure(
        fig=None,
        out_dir=OUT_DIR,
        stem="cluster_profile",
        contract=contract,
    )
    step_summary = {{
        "figure_files": ["cluster_profile.png"],
        "source_data": [source_name],
        "source_table": source_table_name,
    }}
    return contract, step_summary
"""
    finding = {
        "validator": "figure_source_data",
        "severity": "error",
        "detail": {
            "best_mismatch": {
                "reason": "no_verifiable_values",
                "upstream_table": "phenotype_profiles.csv",
                "unverified_source_value_columns": ["availability_fraction"],
            }
        },
    }

    repair = deterministic_contract_repair(code=code, findings=[finding])

    assert repair is not None
    repair_id, repaired = repair
    assert repair_id == "direct_bound_figure_source_projection_v1"
    namespace: dict[str, object] = {}
    exec(repaired, namespace)
    contract, summary = namespace["main"]()
    assert (
        contract["source_data"]
        == summary["source_data"]
        == ["bound_000_phenotype_profiles_source_data.csv"]
    )
    projection = pd.read_csv(tmp_path / contract["source_data"][0])
    assert "availability_fraction" not in projection.columns
    assert projection["median"].tolist() == [82.0, 96.0]


def test_direct_bound_figure_repair_materializes_prior_dataframe_dict_shape(
    tmp_path: Path,
) -> None:
    code = f"""
from pathlib import Path
import pandas as pd

OUT_DIR = Path({str(tmp_path)!r})

def make_figure_contract(**kwargs):
    return kwargs

def save_publication_figure(*, fig, out_dir, stem, contract):
    return None

def main():
    parent_a = pd.DataFrame({{"row_id": ["a"], "primary_or": [1.25]}})
    parent_b = pd.DataFrame({{"row_id": ["b"], "complete_case_n": [17]}})
    contract = make_figure_contract(
        figure_id="figure:overview",
        source_data={{
            "parent_a": parent_a.copy(deep=True).assign(
                source_row_index=range(len(parent_a)),
                source_table="parent_a.csv",
            ),
            "parent_b": parent_b.copy(deep=True).assign(
                source_row_index=range(len(parent_b)),
                source_table="parent_b.csv",
            ),
        }},
    )
    save_publication_figure(
        fig=None,
        out_dir=OUT_DIR,
        stem="overview",
        contract=contract,
    )
    return contract
"""
    finding = {
        "validator": "figure_source_data",
        "severity": "error",
        "detail": {"reason": "missing_source_data"},
    }

    repair = deterministic_contract_repair(
        code=code,
        findings=[finding],
        previous_repair="direct_bound_figure_source_projection_v1",
    )

    assert repair is not None
    repair_id, repaired = repair
    assert repair_id == "direct_bound_figure_source_materialization_v1"
    namespace: dict[str, object] = {}
    exec(repaired, namespace)
    contract = namespace["main"]()
    assert contract["source_data"] == [
        "bound_000_parent_a_source_data.csv",
        "bound_001_parent_b_source_data.csv",
    ]
    assert (tmp_path / contract["source_data"][0]).is_file()
    assert (tmp_path / contract["source_data"][1]).is_file()


def test_bound_figure_source_projection_is_registered_structural() -> None:
    for repair_id in (
        "bound_figure_source_projection_v2",
        "complete_bound_figure_source_bundle_v1",
        "direct_bound_figure_source_materialization_v1",
        "direct_bound_figure_source_projection_v1",
    ):
        metadata = repair_metadata_for(repair_id)
        assert metadata.classification_source == "exact"
        assert metadata.repair_class is RepairClass.STRUCTURAL
        assert metadata.introduces_numbers is False
        assert automatic_repair_allowed(metadata.repair_id)


def test_complete_bound_bundle_reuses_parquet_tables_and_statistic_receipt(
    tmp_path: Path,
) -> None:
    parent_path = tmp_path / "analysis_cohort.parquet"
    pd.DataFrame({"stay_id": [1, 2], "age": [60, 70]}).to_parquet(parent_path)
    code = f"""
import json
from pathlib import Path
import pandas as pd

OUT_DIR = Path({str(tmp_path)!r})

def resolve_bound_product(manifest, input_key):
    return Path({str(parent_path)!r}), {{}}, {{}}, ["stay_id", "age"]

def load_bound_table(path, columns, input_key):
    return pd.read_parquet(path, columns=columns)

def write_json(path, payload):
    path.write_text(json.dumps(payload))

def make_figure_contract(**kwargs):
    return kwargs

def save_publication_figure(*, fig, out_dir, stem, contract):
    return None

(
    analysis_artifact_path,
    analysis_binding,
    analysis_contract,
    analysis_columns,
) = resolve_bound_product({{}}, "artifact:analysis_cohort")
analysis_artifact = load_bound_table(
    analysis_artifact_path, analysis_columns, "artifact:analysis_cohort"
)
cluster_characteristics = pd.DataFrame({{"cluster": [0, 1], "n": [1, 1]}})
cluster_characteristics_path = OUT_DIR / "cluster_characteristics.csv"
cluster_characteristics.to_csv(cluster_characteristics_path, index=False)
phenotype_structure = pd.DataFrame(
    {{"cluster": [0, 1], "feature": ["hr", "hr"], "median": [-1.0, 1.0]}}
)
phenotype_structure_path = OUT_DIR / "phenotype_structure.csv"
phenotype_structure.to_csv(phenotype_structure_path, index=False)
selected_k = 2
cluster_count_path = OUT_DIR / "cluster_count.json"
write_json(cluster_count_path, {{"name": "cluster_count", "value": int(selected_k)}})
valid_source_path = OUT_DIR / "valid_projection_source.csv"
pd.DataFrame({{"stay_id": [1], "value": [-1.0]}}).to_csv(
    valid_source_path, index=False
)
source_data_files = [valid_source_path.name, cluster_characteristics_path.name]
contract = make_figure_contract(
    figure_id="cluster_visualization",
    source_data=source_data_files,
)
save_publication_figure(
    fig=None, out_dir=OUT_DIR, stem="cluster_visualization", contract=contract
)
step_summary = {{"figure_qa": {{"source_data_files": source_data_files}}}}
"""
    findings = [
        {
            "validator": "figure_source_data",
            "severity": "error",
            "detail": {
                "source_table": "cluster_characteristics.csv",
                "best_mismatch": {"reason": "ambiguous_join_key"},
            },
        },
        {
            "validator": "figure_source_data",
            "severity": "error",
            "detail": {
                "reason": "incomplete_source_lineage_coverage",
                "missing_bound_tables": [
                    "analysis_cohort.parquet",
                    "phenotype_structure.csv",
                ],
                "missing_bound_statistics": ["same_step:statistic:cluster_count"],
            },
        },
    ]

    repair = deterministic_contract_repair(code=code, findings=findings)

    assert repair is not None
    repair_id, repaired = repair
    assert repair_id == "complete_bound_figure_source_bundle_v1"
    namespace: dict[str, object] = {}
    exec(repaired, namespace)
    declared = namespace["contract"]["source_data"]
    assert declared == namespace["step_summary"]["figure_qa"]["source_data_files"]
    assert "valid_projection_source.csv" in declared
    assert "cluster_characteristics.csv" not in declared
    assert len(declared) == 5
    projected = [
        pd.read_csv(tmp_path / name)
        for name in declared
        if name.startswith("bound_") and not name.startswith("bound_stat_")
    ]
    assert {frame["source_table"].iloc[0] for frame in projected} == {
        "analysis_cohort.parquet",
        "cluster_characteristics.csv",
        "phenotype_structure.csv",
    }
    statistic = pd.read_csv(tmp_path / "bound_stat_000_cluster_count_source_data.csv")
    assert statistic.to_dict(orient="records") == [
        {
            "statistic": "cluster_count",
            "value": 2,
        }
    ]


def test_complete_bound_bundle_declines_unresolved_table(tmp_path: Path) -> None:
    code = """
import pandas as pd
from pathlib import Path
OUT_DIR = Path(".")
def make_figure_contract(**kwargs):
    return kwargs
def save_publication_figure(*, fig, out_dir, stem, contract):
    return None
contract = make_figure_contract(figure_id="x", source_data=["old.csv"])
save_publication_figure(fig=None, out_dir=OUT_DIR, stem="x", contract=contract)
"""
    finding = {
        "validator": "figure_source_data",
        "severity": "error",
        "detail": {
            "reason": "incomplete_source_lineage_coverage",
            "missing_bound_tables": ["unknown.parquet"],
            "missing_bound_statistics": [],
        },
    }

    assert deterministic_contract_repair(code=code, findings=[finding]) is None


def test_complete_bound_bundle_declines_upstream_statistic(tmp_path: Path) -> None:
    code = f"""
import json
from pathlib import Path
import pandas as pd
OUT_DIR = Path({str(tmp_path)!r})
def write_json(path, payload):
    path.write_text(json.dumps(payload))
def make_figure_contract(**kwargs):
    return kwargs
def save_publication_figure(*, fig, out_dir, stem, contract):
    return None
cluster_count = 2
cluster_count_path = OUT_DIR / "cluster_count.json"
write_json(cluster_count_path, {{"name": "cluster_count", "value": cluster_count}})
contract = make_figure_contract(figure_id="x", source_data=[])
save_publication_figure(fig=None, out_dir=OUT_DIR, stem="x", contract=contract)
"""
    finding = {
        "validator": "figure_source_data",
        "severity": "error",
        "detail": {
            "reason": "incomplete_source_lineage_coverage",
            "missing_bound_tables": [],
            "missing_bound_statistics": ["upstream:statistic:cluster_count"],
        },
    }

    assert deterministic_contract_repair(code=code, findings=[finding]) is None
