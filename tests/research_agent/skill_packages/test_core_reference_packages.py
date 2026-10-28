"""Runnable reference packages stay disk-projected and ship their sources."""

from __future__ import annotations

import importlib
import ast
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from easyicu.research_agent import method_skills
from easyicu.research_agent.skill_packages.reference_io import load_prepared_cohort, plot_intervals, reference_cli


PACKAGES = {
    "cohort-characterization-table-one": "cohort_characterization_table_one",
    "adjusted-association-model": "adjusted_association_model",
    "survival-time-to-event": "survival_time_to_event",
}
PREFIX = "easyicu.research_agent.skill_packages."
ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize("skill_id,package", PACKAGES.items())
def test_registered_package_has_executable_sources(skill_id, package):
    projected = method_skills.method_skill_package(skill_id, enabled=True)
    assert method_skills.REFERENCE_SCRIPT_PACKAGES[skill_id] == package
    assert projected["execution_boundary"] == "host_reference_scripts_executable"
    files = {row["path"]: row["content"] for row in projected["files"]}
    assert {
        "scripts/spec.py", "scripts/load_cohort.py", "scripts/run_analysis.py",
        "scripts/generate_all_plots.py", "scripts/export_all.py",
        "scripts/run_all.py", "scripts/example_data.py",
        "references/reference_scripts.md", "references/methods.md",
    } <= files.keys()
    assert "dependencies" in files["references/methods.md"].lower()


def test_projecting_packages_does_not_import_estimator_or_plot_libraries():
    code = """
import sys
from easyicu.research_agent.method_skills import method_skill_catalog, method_skill_package
method_skill_catalog(enabled=True)
for skill in ('cohort-characterization-table-one', 'adjusted-association-model', 'survival-time-to-event'):
    method_skill_package(skill, enabled=True)
assert not {'statsmodels', 'lifelines', 'sklearn', 'matplotlib'} & sys.modules.keys()
"""
    completed = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=30)
    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize("package", PACKAGES.values())
def test_module_cli_runs_synthetic_example(tmp_path, package):
    output = tmp_path / package
    completed = subprocess.run(
        [sys.executable, "-m", PREFIX + package, "--example", "--out", str(output), "--quiet"],
        capture_output=True, text=True, timeout=60,
    )
    assert completed.returncode == 0, completed.stderr
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["data_provenance"] == "synthetic_example"
    assert manifest["claim_ceiling"] == "analysis_only"
    assert manifest["artifacts"]
    assert (output / "example_cohort.parquet").is_file()
    assert (output / "spec.json").is_file()
    required = {
        "cohort_characterization_table_one": {"table_one.csv"},
        "adjusted_association_model": {"adjusted_association_estimates.csv", "adjusted_association_coefficients.csv"},
        "survival_time_to_event": {"survival_effect_estimates.csv", "survival_ph_diagnostic.csv", "survival_logrank.csv"},
    }[package]
    assert required <= {row["path"] for row in manifest["artifacts"]}
    assert manifest["source_sha256"] == hashlib.sha256((output / "example_cohort.parquet").read_bytes()).hexdigest()
    assert manifest["spec_sha256"] == hashlib.sha256((output / "spec.json").read_bytes()).hexdigest()
    if package != "cohort_characterization_table_one":
        assert list(output.glob("*.svg"))
    for artifact in manifest["artifacts"]:
        assert hashlib.sha256((output / artifact["path"]).read_bytes()).hexdigest() == artifact["sha256"]
    # A second invocation consumes the persisted prepared rows and reviewed spec.
    rerun = tmp_path / "prepared"
    completed = subprocess.run(
        [sys.executable, "-m", PREFIX + package, "--cohort", str(output / "example_cohort.parquet"),
         "--spec", str(output / "spec.json"), "--out", str(rerun), "--provenance", "synthetic_rerun", "--quiet"],
        capture_output=True, text=True, timeout=60,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads((rerun / "manifest.json").read_text())["data_provenance"] == "synthetic_rerun"


@pytest.mark.parametrize("package", PACKAGES.values())
def test_spec_refuses_an_estimator_the_host_does_not_own(package):
    module = importlib.import_module(PREFIX + package + ".scripts.example_data")
    spec = module.example_spec()
    payload = spec.model_dump(mode="json")
    if package == "cohort_characterization_table_one":
        payload["table_one"]["schema_version"] = "easyicu.table_one/3"
        payload["table_one"]["p_values_required"] = False
        payload["table_one"]["p_value_adjustment"] = "not_applicable_data_derived_groups"
        for item in payload["table_one"]["variables"]:
            item["test"] = "none_descriptive_smd_only"
    elif package == "adjusted_association_model":
        payload["model"]["method_family"] = "statsmodels_glm_binomial"
    else:
        payload["primary"]["estimator"] = "weibull_aft"
    with pytest.raises(ValueError):
        type(spec).model_validate(payload)


@pytest.mark.parametrize("package", PACKAGES.values())
def test_scripts_import_only_documented_dependencies(package):
    root = ROOT / "src/easyicu/research_agent/skill_packages" / package
    dependency_line = next(line for line in (root / "references/methods.md").read_text().splitlines()
                           if line.startswith("Dependencies:"))
    declared = set(dependency_line.removeprefix("Dependencies:").strip(" .").split(", "))
    for source in [*root.rglob("*.py"), root.parent / "reference_io.py"]:
        for node in ast.walk(ast.parse(source.read_text())):
            names = [item.name for item in node.names] if isinstance(node, ast.Import) else (
                [node.module] if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module else []
            )
            for name in names:
                module = name.split(".")[0]
                assert module in sys.stdlib_module_names or module in declared, (source, module)


@pytest.mark.parametrize("problem", ["empty", "duplicate", "missing_identity", "missing_column"])
def test_prepared_loader_refuses_invalid_row_contract(problem):
    import pandas as pd

    frame = pd.DataFrame({"stay_id": ["synthetic-a", "synthetic-b"], "value": [1., 2.]})
    if problem == "empty":
        frame = frame.iloc[:0]
    elif problem == "duplicate":
        frame.loc[1, "stay_id"] = "synthetic-a"
    elif problem == "missing_identity":
        frame.loc[1, "stay_id"] = None
    else:
        frame = frame.drop(columns=["value"])
    with pytest.raises(ValueError):
        load_prepared_cohort(frame, required_columns=["value"], identity_column="stay_id")


def test_loader_does_not_select_rows_or_impute_and_does_not_invent_provenance():
    import pandas as pd

    frame = pd.DataFrame({"stay_id": ["synthetic-a", "synthetic-b"], "value": [None, 2.]})
    loaded = load_prepared_cohort(frame, required_columns=["value"], identity_column="stay_id")
    pd.testing.assert_frame_equal(loaded.frame, frame)
    assert loaded.provenance == "undeclared_by_caller"


def test_wheel_and_sdist_ship_every_reference_source_and_guide(tmp_path):
    import tarfile
    import zipfile

    pytest.importorskip("setuptools", reason="archive build needs installed setuptools; no package installs in server lane")
    pytest.importorskip("wheel", reason="archive build needs installed wheel")
    dist, build, egg = (tmp_path / name for name in ("dist", "build", "egg"))
    egg.mkdir()
    completed = subprocess.run(
        [sys.executable, "-c", "from setuptools import setup; setup()", "build", "--build-base", str(build),
         "egg_info", "--egg-base", str(egg), "sdist", "--dist-dir", str(dist),
         "bdist_wheel", "--dist-dir", str(dist), "--bdist-dir", str(tmp_path / "bdist")],
        cwd=ROOT, capture_output=True, text=True, timeout=120,
    )
    assert completed.returncode == 0, completed.stdout[-4000:] + completed.stderr[-4000:]
    with zipfile.ZipFile(next(dist.glob("*.whl"))) as wheel:
        wheel_names = set(wheel.namelist())
    with tarfile.open(next(dist.glob("*.tar.gz"))) as sdist:
        sdist_names = {name.split("/", 1)[1] for name in sdist.getnames() if "/" in name}
    root = ROOT / "src/easyicu/research_agent/skill_packages"
    files = [root / "reference_io.py"]
    for package in PACKAGES.values():
        files.extend(path for path in (root / package).rglob("*") if path.suffix in {".py", ".md"})
    for path in files:
        assert path.relative_to(ROOT / "src").as_posix() in wheel_names, path
        assert path.relative_to(ROOT).as_posix() in sdist_names, path


def test_packaging_patterns_include_guides_and_python_packages():
    import fnmatch
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib

    settings = tomllib.loads((ROOT / "pyproject.toml").read_text())["tool"]["setuptools"]
    patterns = settings["package-data"]["easyicu.research_agent.skill_packages"]
    manifest = (ROOT / "MANIFEST.in").read_text()
    assert "recursive-include src/easyicu/research_agent/skill_packages *.md" in manifest
    for package in PACKAGES.values():
        directory = ROOT / "src/easyicu/research_agent/skill_packages" / package
        assert (directory / "__init__.py").is_file()
        assert (directory / "scripts/__init__.py").is_file()
        for path in directory.rglob("*.md"):
            relative = path.relative_to(directory.parent).as_posix()
            assert any(fnmatch.fnmatch(relative, pattern) for pattern in patterns), relative


@pytest.mark.parametrize("effect_scale,estimate,low,high,axis,null", [
    ("odds_ratio", 1.5, 1.1, 2.3, "log", 1.),
    ("coefficient", 0.5, -0.1, 1.1, "linear", 0.),
])
def test_effect_plot_uses_reported_interval_and_declared_scale(
    tmp_path, monkeypatch, effect_scale, estimate, low, high, axis, null,
):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    original = plt.subplots
    figures = []

    def capture(*args, **kwargs):
        fig, ax = original(*args, **kwargs)
        figures.append((fig, ax))
        return fig, ax

    monkeypatch.setattr(plt, "subplots", capture)
    table = pd.DataFrame([{"estimate": estimate, "ci_low": low, "ci_high": high, "label": "Synthetic contrast"}])
    paths = plot_intervals(table, tmp_path, stem="effect", estimate_column="estimate",
                           label_column="label", effect_scale=effect_scale)
    fig, ax = figures[0]
    assert ax.get_xscale() == axis
    assert any(list(line.get_xdata()) == [null, null] for line in ax.lines)
    np.testing.assert_allclose(ax.collections[0].get_segments()[0][:, 0], [low, high])
    assert all(path.is_file() for path in paths)
    assert fig.number not in plt.get_fignums()


@pytest.mark.parametrize("arguments", [
    [], ["--cohort", "synthetic.parquet"],
    ["--example", "--cohort", "synthetic.parquet"],
    ["--example", "--spec", "synthetic.json"],
])
def test_cli_requires_one_complete_input_declaration(tmp_path, arguments):
    def unexpected(*args, **kwargs):
        pytest.fail("invalid CLI declaration reached analysis")

    with pytest.raises(SystemExit) as error:
        reference_cli(argv=[*arguments, "--out", str(tmp_path)], description="Synthetic CLI",
                      spec_type=None, run_all=unexpected, write_example=unexpected)
    assert error.value.code == 2
