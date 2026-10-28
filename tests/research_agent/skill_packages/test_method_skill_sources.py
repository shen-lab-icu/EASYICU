"""Built-in packages expose the host definitions behind their results."""

import ast
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from easyicu.research_agent.method_skills import method_skill_catalog, method_skill_package
from easyicu.research_agent.skill_packages import host_sources as sources
from easyicu.research_agent.method_skills import METHOD_SKILLS
from easyicu.research_agent.planning.method_adapter_catalog import (
    DIRECT_METHOD_BINDINGS,
    HIGH_FREQUENCY_METHOD_ADAPTERS,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def mapping():
    return sources.method_skill_source_map(METHOD_SKILLS)


@pytest.mark.parametrize("skill_id", [row["id"] for row in method_skill_catalog(enabled=True)["items"]])
def test_every_builtin_has_host_implementation(skill_id):
    package = method_skill_package(skill_id, enabled=True)
    implementations = [row for row in package["files"] if row["path"].startswith("implementation/")]
    assert implementations, skill_id
    assert all(row["role"] == "host_implementation" for row in implementations)


def test_mapping_covers_exactly_the_catalog_and_resolves_every_definition(mapping):
    assert set(mapping) == {row["id"] for row in method_skill_catalog(enabled=True)["items"]}
    reader = sources._Sources(REPO_ROOT / "src/easyicu")
    for definitions in mapping.values():
        assert definitions
        assert len(definitions) == len(set(definitions))
        for definition in definitions:
            assert isinstance(reader.definition(definition), (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))


@pytest.mark.parametrize("skill_id", [spec.skill_id for spec in METHOD_SKILLS])
def test_excerpts_equal_current_source_lines_and_hashes(skill_id, mapping):
    package = method_skill_package(skill_id, enabled=True)
    seen = []
    for file in package["files"]:
        if file.get("role") != "host_implementation":
            continue
        raw = (REPO_ROOT / file["source_path"]).read_bytes()
        assert file["source_sha256"] == hashlib.sha256(raw).hexdigest()
        assert file["language"] == "python"
        assert file["size_bytes"] == len(file["content"].encode())
        assert file["sha256"] == hashlib.sha256(file["content"].encode()).hexdigest()
        module = "easyicu." + file["path"].removeprefix("implementation/").removesuffix(".py")
        chunks = []
        assert file["definitions"]
        assert file["definitions"] == sorted(file["definitions"], key=lambda row: row["line_start"])
        for definition in file["definitions"]:
            start, end = definition["line_start"], definition["line_end"]
            expected = "".join(raw.decode().splitlines(keepends=True)[start - 1:end])
            # Independently resolve the qualified name to verify the reported
            # coordinates, including the decorator lines, not just the slice.
            node = ast.parse(raw.decode())
            for part in definition["qualname"].split("."):
                node = next(child for child in node.body
                            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                            and child.name == part)
            expected_start = min([node.lineno, *(item.lineno for item in node.decorator_list)])
            assert start == expected_start
            assert end == node.end_lineno
            chunks.append(f"# {file['source_path']}:{start}-{end} sha256={file['source_sha256']}\n" + expected)
            seen.append(sources.SourceDefinition(module, definition["qualname"]))
        assert file["content"] == "\n\n".join(chunks)
        ast.parse(file["content"])
    assert tuple(seen) == mapping[skill_id]
    canonical = json.dumps([{"path": row["path"], "sha256": row["sha256"]} for row in package["files"]],
                           ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    assert package["package_sha256"] == hashlib.sha256(canonical).hexdigest()


def test_adapter_links_are_the_callables_the_executor_really_dispatches(mapping):
    from easyicu.research_agent.execution.runners.selection import STANDARD_EXECUTORS

    for spec in METHOD_SKILLS:
        for adapter in HIGH_FREQUENCY_METHOD_ADAPTERS:
            if adapter.action_id not in spec.action_ids:
                continue
            targets = [executor.render.__globals__[adapter.owner_entrypoint]
                       for executor in STANDARD_EXECUTORS.executors
                       if adapter.owner_entrypoint in executor.render.__code__.co_names]
            assert targets
            target = targets[0]
            assert all(item is target for item in targets)
            assert (target.__module__, target.__qualname__) in mapping[spec.skill_id]
            for binding in DIRECT_METHOD_BINDINGS:
                if binding.owner_module == target.__module__:
                    assert all((binding.direct_module, name) in mapping[spec.skill_id] for name in binding.direct_symbols)


def test_family_links_follow_the_runtime_registry_callables(mapping):
    from easyicu.research_agent.execution.runners.selection import STANDARD_EXECUTORS
    from easyicu.research_agent.planning.capability_registry import CAPABILITY_REGISTRY

    capabilities = {cap.capability_id: cap for cap in CAPABILITY_REGISTRY}
    for spec in METHOD_SKILLS:
        cap = capabilities.get(spec.capability_id)
        modules = list(sources._EXTRA_RUNNERS.get(spec.skill_id, ()))
        if cap is not None and cap.primary_runner_module is not None and spec.skill_id not in sources._ACTION_ONLY:
            modules.append(cap.primary_runner_module)
        for relative in modules:
            module = "easyicu.research_agent." + relative
            actual = set()
            for executor in STANDARD_EXECUTORS.executors:
                for name in executor.render.__code__.co_names:
                    target = executor.render.__globals__.get(name)
                    if callable(target) and getattr(target, "__module__", None) == module:
                        actual.add((module, target.__qualname__))
            assert actual
            assert actual <= set(mapping[spec.skill_id])


def test_workflow_sources_include_the_included_module_sources(mapping):
    for spec in METHOD_SKILLS:
        for module_id in spec.included_module_ids:
            assert set(mapping[module_id]) <= set(mapping[spec.skill_id])


def test_method_suite_runner_links_follow_the_runtime_callables(mapping):
    from easyicu.research_agent.execution.runners.selection import STANDARD_EXECUTORS
    from easyicu.research_agent.planning.analysis_method_suite import METHOD_SUITE_REGISTRY
    from easyicu.research_agent.planning.capability_registry import CAPABILITY_REGISTRY

    adapters = {row.action_id for row in HIGH_FREQUENCY_METHOD_ADAPTERS}
    methods = {f"{suite.family}.{method.key}": method for suite in METHOD_SUITE_REGISTRY for method in suite.methods}
    for spec in METHOD_SKILLS:
        for action in set(spec.action_ids) - adapters:
            runner = methods[action].runner
            for capability in CAPABILITY_REGISTRY:
                if runner is None or capability.primary_runner != runner or not capability.primary_runner_module:
                    continue
                module = "easyicu.research_agent." + capability.primary_runner_module
                targets = {(target.__module__, target.__qualname__)
                           for executor in STANDARD_EXECUTORS.executors
                           for name in executor.render.__code__.co_names
                           if callable(target := executor.render.__globals__.get(name))
                           and getattr(target, "__module__", None) == module}
                assert targets
                assert targets <= set(mapping[spec.skill_id])


def test_agent_coded_skills_disclose_the_real_host_code_origin_gate(mapping):
    for spec in METHOD_SKILLS:
        if getattr(spec, "execution_mode", "") == "agent_coded_with_host_gates":
            assert ("easyicu.research_agent.execution.step_execution", "initial_code_origin") in mapping[spec.skill_id]


def test_local_estimator_helpers_are_included(mapping):
    assert ("easyicu.research_agent.robustness.estimators", "_encode_categorical_predictors") in mapping["adjusted-association-model"]
    assert ("easyicu.research_agent.methods.rmst", "_km_integral") in mapping["restricted-mean-survival-time"]


def test_ordered_trend_discloses_the_adjusted_model_it_refits(mapping):
    from easyicu.research_agent.execution.runners import adjusted_association_executor as adjusted
    from easyicu.research_agent.execution.runners import ordered_stratified_executor as ordered

    refit = adjusted.run_adjusted_association_from_env
    assert refit.__name__ in ordered._adjusted_trend.__code__.co_names
    assert (refit.__module__, refit.__qualname__) in mapping["ordinal-dose-response"]
    fit = adjusted.fit_estimator
    assert (fit.__module__, fit.__qualname__) in mapping["ordinal-dose-response"]


def test_distribution_discloses_the_cluster_interval_it_computes(mapping):
    from easyicu.research_agent.execution.runners import exposure_outcome_distribution_executor as distribution

    for caller, target in (
        (distribution._proportion_interval, distribution.patient_weighted_hoeffding_interval),
        (distribution._risk_difference_result, distribution.require_wald_group_variation),
    ):
        assert target.__name__ in caller.__code__.co_names
        assert (target.__module__, target.__qualname__) in mapping["exposure-outcome-distribution"]


@pytest.mark.parametrize("skill_id,module,runtime", [
    ("adjusted-association-model", "adjusted_association_executor", "run_adjusted_association_from_env"),
    ("landmark-categorical-association", "landmark_categorical_association_executor", "run_landmark_categorical_primary"),
    ("landmark-spline-dose-response", "landmark_spline_executor", "run_landmark_spline_association"),
    ("time-varying-cox", "time_varying_executor", "run_time_varying_association"),
    ("survival-time-to-event", "survival_primary_executor", "run_survival_primary"),
    ("restricted-mean-survival-time", "rmst_executor", "run_rmst_contrast"),
    ("clinical-risk-prediction", "prediction_model_executor", "run_prediction_model"),
    ("cross-sectional-phenotyping", "cross_sectional_phenotyping_executor", "run_primary_phenotyping"),
    ("exposure-outcome-distribution", "exposure_outcome_distribution_executor", "run_exposure_outcome_distribution_from_env"),
    ("ordinal-dose-response", "ordered_stratified_executor", "run_ordered_stratified_from_env"),
    ("trajectory-phenotyping", "trajectory_scientific_representation_executor", "run_trajectory_scientific_representation"),
    ("trajectory-phenotyping", "trajectory_scientific_candidate_executor", "run_trajectory_scientific_candidate_selection"),
    ("trajectory-phenotyping", "trajectory_stability_executor", "run_trajectory_stability"),
    ("target-trial-emulation", "target_trial_executor", "run_target_trial_suite"),
])
def test_packages_include_the_runtime_called_by_the_generated_script(skill_id, module, runtime, mapping):
    path = REPO_ROOT / "src/easyicu/research_agent/execution/runners" / (module + ".py")
    tree = ast.parse(path.read_text())
    # These runtime names must occur inside generated script strings, rather
    # than merely naming unused definitions in the same source module.
    assert any(isinstance(node, ast.Constant) and isinstance(node.value, str) and runtime in node.value
               for node in ast.walk(tree))
    assert ("easyicu.research_agent.execution.runners." + module, runtime) in mapping[skill_id]


def test_all_packages_in_a_fresh_process_do_not_import_statistical_libraries():
    script = """
import importlib.abc
import sys
blocked = {'statsmodels', 'lifelines', 'sklearn', 'matplotlib'}
attempted = []
class RejectKernels(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in blocked:
            attempted.append(fullname)
            raise ImportError('catalogue imported a statistical kernel: ' + fullname)
sys.meta_path.insert(0, RejectKernels())
from easyicu.research_agent.method_skills import method_skill_catalog, method_skill_package
items = method_skill_catalog(enabled=True)['items']
assert len(items) == 24
for row in items:
    assert any(f.get('role') == 'host_implementation'
               for f in method_skill_package(row['id'], enabled=True)['files'])
assert not attempted, attempted
assert not (blocked & {name.split('.')[0] for name in sys.modules})
print(len(items))
"""
    result = subprocess.run([sys.executable, "-c", script], cwd=REPO_ROOT,
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "24"


@pytest.mark.parametrize("qualname", ["renamed_prediction_binary", "prediction_binary.missing_nested"])
def test_renamed_definition_fails_loudly_with_typed_error(monkeypatch, qualname):
    monkeypatch.setitem(sources._SKILL_ROOTS, "sample-size-and-power", (
        sources.SourceDefinition("easyicu.research_agent.skill_packages", qualname),
    ))
    with pytest.raises(sources.MethodSkillSourceError, match=qualname):
        method_skill_package("sample-size-and-power", enabled=True)


def test_missing_source_file_fails_with_typed_error(monkeypatch):
    monkeypatch.setitem(sources._SKILL_ROOTS, "sample-size-and-power", (
        sources.SourceDefinition("easyicu.research_agent.removed_source", "calculate"),
    ))
    with pytest.raises(sources.MethodSkillSourceError, match="unreadable source module"):
        method_skill_package("sample-size-and-power", enabled=True)


def test_missing_family_dispatch_link_fails_with_typed_error(monkeypatch):
    cap = next(cap for cap in sources.CAPABILITY_REGISTRY if cap.capability_id == "association_adjusted_v1")
    monkeypatch.setattr(sources, "CAPABILITY_REGISTRY", tuple(
        replace(row, primary_runner_module="execution.runners.removed") if row == cap else row
        for row in sources.CAPABILITY_REGISTRY
    ))
    with pytest.raises(sources.MethodSkillSourceError, match="executor render binding missing"):
        method_skill_package("adjusted-association-model", enabled=True)


@pytest.fixture
def synthetic_source(tmp_path, monkeypatch):
    root = tmp_path / "easyicu"
    selection = root / "research_agent/execution/runners/selection.py"
    selection.parent.mkdir(parents=True)
    selection.write_text("")
    module = root / "synthetic.py"
    module.write_text("# synthetic module\n\ndef decorate(obj):\n    return obj\n\n@decorate\nclass Calculator:\n    @staticmethod\n    async def compute(value):\n        return value + 1\n")
    spec = SimpleNamespace(skill_id="synthetic", capability_id=None, action_ids=(), included_module_ids=())
    monkeypatch.setitem(sources._SKILL_ROOTS, "synthetic", (sources.SourceDefinition("easyicu.synthetic", "Calculator"),))
    return root, module, spec


def test_decorated_class_and_qualified_async_method_keep_decorators(synthetic_source, monkeypatch):
    root, module, spec = synthetic_source
    files = sources.method_skill_implementation_files("synthetic", (spec,), package_root=root)
    assert files[0]["definitions"] == [
        {"qualname": "decorate", "line_start": 3, "line_end": 4},
        {"qualname": "Calculator", "line_start": 6, "line_end": 10},
    ]
    assert "@decorate\nclass Calculator" in files[0]["content"]
    monkeypatch.setitem(sources._SKILL_ROOTS, "synthetic", (sources.SourceDefinition("easyicu.synthetic", "Calculator.compute"),))
    files = sources.method_skill_implementation_files("synthetic", (spec,), package_root=root)
    assert files[0]["definitions"] == [{"qualname": "Calculator.compute", "line_start": 8, "line_end": 10}]
    assert "    @staticmethod\n    async def compute" in files[0]["content"]
    assert files[0]["source_sha256"] == hashlib.sha256(module.read_bytes()).hexdigest()


def test_source_changes_between_reads_change_both_digests(synthetic_source):
    root, module, spec = synthetic_source
    before = sources.method_skill_implementation_files("synthetic", (spec,), package_root=root)[0]
    module.write_text(module.read_text().replace("value + 1", "value + 2"))
    after = sources.method_skill_implementation_files("synthetic", (spec,), package_root=root)[0]
    assert before["source_sha256"] != after["source_sha256"]
    assert before["sha256"] != after["sha256"]
    assert "value + 2" in after["content"]
    module.write_text(module.read_text() + "\n# outside the selected definitions\n")
    outside = sources.method_skill_implementation_files("synthetic", (spec,), package_root=root)[0]
    assert after["source_sha256"] != outside["source_sha256"]
    assert after["sha256"] != outside["sha256"]


@pytest.mark.parametrize("qualname,before,after", [
    ("Calculator", "class Calculator:", "class RenamedCalculator:"),
    ("Calculator.compute", "async def compute", "async def renamed_compute"),
])
def test_renaming_an_on_disk_definition_is_not_a_silent_empty_file(synthetic_source, monkeypatch, qualname, before, after):
    root, module, spec = synthetic_source
    monkeypatch.setitem(sources._SKILL_ROOTS, "synthetic", (sources.SourceDefinition("easyicu.synthetic", qualname),))
    sources.method_skill_implementation_files("synthetic", (spec,), package_root=root)
    module.write_text(module.read_text().replace(before, after))
    with pytest.raises(sources.MethodSkillSourceError, match="easyicu.synthetic:" + qualname):
        sources.method_skill_implementation_files("synthetic", (spec,), package_root=root)


def test_package_digest_binds_current_implementation_content(monkeypatch):
    before = method_skill_package("sample-size-and-power", enabled=True)
    original = sources._Sources.read

    def changed(reader, module):
        path, raw, tree = original(reader, module)
        if module == "easyicu.research_agent.skill_packages":
            raw += b"\n# changed source snapshot\n"
        return path, raw, tree

    monkeypatch.setattr(sources._Sources, "read", changed)
    after = method_skill_package("sample-size-and-power", enabled=True)
    assert before["package_sha256"] != after["package_sha256"]
    assert before["catalog_sha256"] == after["catalog_sha256"]


@pytest.mark.parametrize("skill_id,boundary", [
    ("survival-time-to-event", "host_reference_scripts_executable"),
    ("sample-size-and-power", "host_reference_scripts_executable"),
    ("six-database-cohort-extraction", "owner_projection_no_execution"),
])
def test_transparency_preserves_execution_boundaries(skill_id, boundary):
    package = method_skill_package(skill_id, enabled=True)
    assert package["read_only"] is True
    assert package["execution_boundary"] == boundary
