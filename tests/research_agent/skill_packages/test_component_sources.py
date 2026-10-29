"""Source transparency proves function identity, without running statistics."""

import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from easyicu.research_agent.method_skills import method_skill_catalog, method_skill_package
from easyicu.research_agent.skill_packages import component_sources, host_sources


REPO_ROOT = Path(__file__).resolve().parents[3]
CATALOG = method_skill_catalog(enabled=True)
IDS = [row["id"] for row in (*CATALOG["items"], *CATALOG["components"])]


@pytest.mark.parametrize("skill_id", IDS)
def test_every_package_has_consistent_structured_source_evidence(skill_id):
    package = method_skill_package(skill_id, enabled=True)
    metadata = package["implementation"]
    assert metadata["mode"] in {"fixed_host", "generated_code", "owner_projection", "unavailable"}
    assert metadata["summary_en"] and metadata["summary_zh"]
    files = {file["path"]: file for file in package["files"]}
    assert json.loads(files["references/implementation.json"]["content"]) == metadata
    if metadata["mode"] != "unavailable":
        assert metadata["entrypoints"], skill_id
    reader = host_sources._Sources(REPO_ROOT / "src/easyicu")
    for entry in metadata["entrypoints"]:
        file = files[entry["file_path"]]
        raw = (REPO_ROOT / entry["source_path"]).read_bytes()
        assert entry["source_sha256"] == file["source_sha256"] == hashlib.sha256(raw).hexdigest()
        source_excerpt = "".join(raw.decode().splitlines(keepends=True)[entry["line_start"] - 1:entry["line_end"]])
        packaged_excerpt = "".join(file["content"].splitlines(keepends=True)[entry["excerpt_line_start"] - 1:entry["excerpt_line_end"]])
        assert packaged_excerpt == source_excerpt
        node = reader.definition(host_sources.SourceDefinition(entry["module"], entry["qualname"]))
        assert entry["name"] == node.name
        assert entry["docstring"] == (ast.get_docstring(node) or "")
        assert entry["line_start"] == min([node.lineno, *(decorator.lineno for decorator in node.decorator_list)])
        assert entry["line_end"] == node.end_lineno
        # The displayed signature reconstructs precisely the source arguments,
        # defaults, annotations and return type, without evaluating them.
        signature_node = ast.parse(entry["signature"] + ":\n    pass\n").body[0]
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            assert ast.dump(signature_node.args) == ast.dump(node.args)
            assert (ast.dump(signature_node.returns) if signature_node.returns else None) == (
                ast.dump(node.returns) if node.returns else None)
        assert entry["role"] in {"entrypoint", "kernel", "helper", "gate"}
        if metadata["mode"] == "generated_code":
            assert entry["role"] in {"helper", "gate"}
    for example in metadata["examples"]:
        assert example["path"].startswith("scripts/")
        assert example["code"] == files[example["path"]]["content"]
        assert example["language"] == "python"
        ast.parse(example["code"])


def _entries(action):
    return method_skill_package(action, enabled=True)["implementation"]["entrypoints"]


@pytest.mark.parametrize("action,required,forbidden", [
    ("time_to_event.cox_hr", {"_fit_declared_cox"}, {"estimate_cif", "fit_fine_gray", "gray_test", "rmst"}),
    ("time_to_event.ph_check", {"ph_test"}, {"estimate_cif", "fit_fine_gray", "gray_test"}),
    ("time_to_event.km_logrank", {"km_estimate", "logrank_by_group"}, {"estimate_cif", "fit_fine_gray"}),
    ("time_to_event.rmst", {"rmst", "rmst_difference"}, {"ph_test", "estimate_cif"}),
    ("time_to_event.competing_risks_cif", {"estimate_cif", "fit_fine_gray", "gray_test"}, {"_fit_declared_cox", "rmst"}),
    ("association.mixed_effects", {"fit_mixed_effects"}, {"fit_gee"}),
    ("association.gee", {"fit_gee"}, {"fit_mixed_effects"}),
    ("prediction.calibration_metrics", {"_calibration_model", "evaluate_binary_predictions"}, {"continuous_nri", "idi", "net_benefit_curve"}),
    ("prediction.decision_curve", {"run_prediction_score_analysis", "net_benefit_curve"}, {"run_prediction_model", "continuous_nri"}),
    ("prediction.external_validation", {"run_external_validation_step", "run_external_validation"}, {"run_prediction_model"}),
    ("data_preparation.missing_data", {"multiple_impute", "rubin_pool"}, {"ridge_single_impute", "mice_impute"}),
    ("cross_database.meta_analysis", {"pool_database_estimates", "compute_meta_analysis"}, {"fit_estimator"}),
])
def test_components_have_exact_compute_roots_without_inheriting_other_methods(action, required, forbidden):
    names = {entry["qualname"] for entry in _entries(action)}
    assert required <= names
    assert not names & forbidden


def test_fixed_decision_curve_route_does_not_claim_to_call_the_reusable_kernel():
    entries = {entry["name"]: entry for entry in _entries("prediction.decision_curve")}
    assert entries["run_prediction_score_analysis"]["role"] == "entrypoint"
    assert entries["net_benefit_curve"]["role"] == "helper"


def test_script_renderers_are_helpers_so_computing_functions_can_be_prioritized():
    for action in ("time_to_event.rmst", "prediction.calibration_metrics", "survival-time-to-event"):
        entries = _entries(action)
        renderers = [entry for entry in entries if entry["name"].endswith(("_code", "_scaffold"))]
        assert renderers
        assert all(entry["role"] == "helper" for entry in renderers)
        assert any(entry["role"] in {"entrypoint", "kernel"} for entry in entries)


@pytest.mark.parametrize("action,name", [
    ("cohort-characterization-table-one", "table_one_spec_sha256"),
    ("descriptive.table_one", "table_one_spec_sha256"),
    ("exposure-outcome-distribution", "require_wald_group_variation"),
    ("missingness-measurement-audit", "measurement_provenance_receipt"),
])
def test_hash_and_input_checks_are_helpers_not_statistical_kernels(action, name):
    entry = next(entry for entry in _entries(action) if entry["name"] == name)
    assert entry["role"] == "helper"


def test_unit_harmonisation_projects_recorded_status_without_claiming_conversion():
    metadata = method_skill_package("data_preparation.unit_harmonisation", enabled=True)["implementation"]
    assert metadata["mode"] == "owner_projection"
    assert {entry["name"] for entry in metadata["entrypoints"]} == {"database_preparation_status"}
    assert {entry["role"] for entry in metadata["entrypoints"]} == {"helper"}
    assert "does not convert" in metadata["summary_en"]


def test_cluster_comparison_uses_its_real_runtime_not_a_neighboring_diagnostic():
    entries = _entries("phenotyping.outcome_by_cluster")
    assert {entry["name"] for entry in entries} == {"phenotype_comparison_executor_code", "run_phenotype_comparison"}
    assert all(entry["module"].endswith("phenotype_comparison_executor") for entry in entries)
    sizes = method_skill_package("phenotyping.cluster_sizes", enabled=True)["implementation"]
    assert sizes["mode"] == "fixed_host"
    assert {entry["name"] for entry in sizes["entrypoints"]} == {"cluster_sizes_executor_code", "compute_cluster_sizes", "run_cluster_sizes"}
    assert not any(entry["module"].endswith("trajectory_stability_executor") for entry in sizes["entrypoints"])


def test_actual_full_adapter_and_typed_subcontract_are_distinguished_from_legacy_labels():
    catalog = {row["id"]: row for row in CATALOG["components"]}
    assert catalog["descriptive.table_one"]["implementation"] == "llm_coded"
    assert method_skill_package("descriptive.table_one", enabled=True)["implementation"]["mode"] == "fixed_host"
    ordinal = method_skill_package("association.ordinal_trend", enabled=True)["implementation"]
    assert ordinal["mode"] == "generated_code"
    assert "subcontract" in ordinal["summary_en"]
    assert {entry["role"] for entry in ordinal["entrypoints"]} == {"helper", "gate"}


def test_no_prediction_fairness_algorithm_is_invented_from_subgroup_odds_ratios():
    metadata = method_skill_package("prediction.subgroup_fairness", enabled=True)["implementation"]
    assert metadata["mode"] == "fixed_host"
    assert {"evaluate_prediction_subgroups", "run_prediction_subgroup_step", "evaluate_binary_predictions"} <= {entry["name"] for entry in metadata["entrypoints"]}
    assert not any(entry["module"].endswith("methods.fairness") for entry in metadata["entrypoints"])


def test_balance_source_names_the_registered_action_and_real_kernel():
    metadata = method_skill_package("causal_emulation.covariate_balance", enabled=True)["implementation"]
    assert metadata["mode"] == "fixed_host"
    assert {entry["name"] for entry in metadata["entrypoints"]} == {"covariate_balance_executor_code", "compute_covariate_balance", "run_covariate_balance"}
    assert {entry["role"] for entry in metadata["entrypoints"]} == {"helper", "kernel", "entrypoint"}


def test_component_binding_drift_is_a_typed_error(monkeypatch):
    monkeypatch.setitem(component_sources._COMPONENT_FUNCTIONS, "prediction.external_validation", (
        ("methods.prediction_validation", ("renamed_external_validation",)),
    ))
    with pytest.raises(host_sources.MethodSkillSourceError, match="renamed_external_validation"):
        method_skill_package("prediction.external_validation", enabled=True)


@pytest.mark.parametrize("module", ["os", "easyicu../secrets", "easyicu./tmp", "easyicu.bad-name"])
def test_source_module_names_cannot_inject_file_paths(module):
    with pytest.raises(host_sources.MethodSkillSourceError, match="invalid source module"):
        host_sources._Sources(REPO_ROOT / "src/easyicu").read(module)


def test_source_metadata_does_not_evaluate_defaults_or_annotations(tmp_path):
    root = tmp_path / "easyicu"
    root.mkdir()
    path = root / "fixture.py"
    path.write_text('def compute(value: missing.Type = dangerous_call(), /, *, flag=True) -> absent.Result:\n    """Fixture docstring."""\n    return value\n')
    source = host_sources.SourceDefinition("easyicu.fixture", "compute")
    files = host_sources._implementation_files((source,), host_sources._Sources(root))
    metadata = host_sources.implementation_metadata(mode="fixed_host", summary_en="fixture", summary_zh="测试",
                                                   bindings={source: "kernel"}, files=files)
    entry = metadata["entrypoints"][0]
    assert "dangerous_call()" in entry["signature"]
    assert entry["docstring"] == "Fixture docstring."


def test_all_84_packages_in_fresh_process_never_import_statistical_dependencies():
    script = '''
import importlib.abc
import sys
blocked = {'statsmodels', 'lifelines', 'sklearn', 'matplotlib'}
class RejectKernels(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in blocked:
            raise AssertionError('catalogue imported a statistical dependency: ' + fullname)
sys.meta_path.insert(0, RejectKernels())
from easyicu.research_agent.method_skills import method_skill_catalog, method_skill_package
c = method_skill_catalog(enabled=True)
rows = [*c['items'], *c['components']]
assert len(rows) == 84
for row in rows:
    assert 'implementation' in method_skill_package(row['id'], enabled=True)
assert not blocked & {name.split('.')[0] for name in sys.modules}
print(len(rows))
'''
    result = subprocess.run([sys.executable, "-c", script], cwd=REPO_ROOT,
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "84"
