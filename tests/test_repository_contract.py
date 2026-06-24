from __future__ import annotations

from pathlib import Path
import re
import subprocess

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.9/3.10 test runtime
    import tomli as tomllib


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_pyproject_authors_do_not_use_placeholder_contact() -> None:
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    authors = pyproject["project"]["authors"]
    assert authors, "Expected at least one project author entry."
    assert all(
        "example.com" not in author.get("email", "") for author in authors
    ), "Project metadata should not publish placeholder email addresses."


def test_pyproject_requires_modern_pyarrow_for_export_compatibility() -> None:
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    dependencies = pyproject["project"]["dependencies"]
    assert "pyarrow>=23.0.0" in dependencies


def test_pyproject_license_uses_spdx_string() -> None:
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert pyproject["project"]["license"] == "MIT"


def test_pyproject_dev_extra_includes_build_for_release_contract() -> None:
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    dev_dependencies = pyproject["project"]["optional-dependencies"]["dev"]
    assert "build>=1.2" in dev_dependencies


def test_manifest_does_not_reference_missing_optional_payloads() -> None:
    manifest = (REPO_ROOT / "MANIFEST.in").read_text(encoding="utf-8")

    assert "CHANGELOG.md" not in manifest
    assert "src/easyicu/extdata" not in manifest


def test_python39_compatible_union_annotations_use_future_import() -> None:
    files_requiring_future = [
        "src/easyicu/attach.py",
        "src/easyicu/concept.py",
        "src/easyicu/concept_callbacks.py",
        "src/easyicu/concept_parser.py",
        "src/easyicu/config.py",
        "src/easyicu/data_converter.py",
        "src/easyicu/data_utils.py",
        "src/easyicu/datasource.py",
        "src/easyicu/download.py",
        "src/easyicu/feature_compare.py",
        "src/easyicu/hosted_llm_server.py",
        "src/easyicu/import_data.py",
        "src/easyicu/logging_utils.py",
        "src/easyicu/resources.py",
        "src/easyicu/scripts/extract_features.py",
        "src/easyicu/ts_utils.py",
    ]

    missing = []
    for rel_path in files_requiring_future:
        content = (REPO_ROOT / rel_path).read_text(encoding="utf-8")
        if "from __future__ import annotations" not in content:
            missing.append(rel_path)

    assert not missing, f"Python 3.9 runtime needs deferred annotation evaluation in: {missing}"


def test_repository_includes_contribution_guide() -> None:
    assert (REPO_ROOT / "CONTRIBUTING.md").exists()


def test_repository_includes_ci_workflow() -> None:
    assert (REPO_ROOT / ".github" / "workflows" / "ci.yml").exists()


def test_ci_workflow_runs_supported_python_matrix() -> None:
    workflow = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")

    assert 'python-version: ["3.10", "3.11", "3.12"]' in workflow
    assert "python-version: ${{ matrix.python-version }}" in workflow
    assert "ruff check src tests" in workflow
    assert "pytest -q" in workflow


def test_repository_includes_citation_metadata() -> None:
    citation = REPO_ROOT / "CITATION.cff"
    assert citation.exists()

    content = citation.read_text(encoding="utf-8")
    assert 'title: "EasyICU"' in content
    assert 'type: software' in content
    assert 'repository-code: "https://github.com/shen-lab-icu/EASYICU"' in content


def test_repository_tracks_no_release_debris() -> None:
    tracked = subprocess.run(
        [
            "git",
            "-C",
            str(REPO_ROOT),
            "ls-files",
            "*.pyc",
            "**/__pycache__/*",
            "*.DS_Store",
            "__MACOSX/*",
            "*/__MACOSX/*",
            "._*",
            "*/._*",
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    assert tracked.stdout == ""


def test_readmes_link_to_citation_metadata() -> None:
    english = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    chinese = (REPO_ROOT / "README_zh.md").read_text(encoding="utf-8")

    assert "## Paper, Citation & Reproducibility" in english
    assert "[CITATION.cff](CITATION.cff)" in english

    assert "## 论文、引用与可复现" in chinese
    assert "[CITATION.cff](CITATION.cff)" in chinese


def test_readmes_include_scope_limiting_claim_language() -> None:
    chinese = (REPO_ROOT / "README_zh.md").read_text(encoding="utf-8")
    agent_readme = (REPO_ROOT / "src" / "easyicu" / "research_agent" / "README.md").read_text(
        encoding="utf-8"
    )
    agent_readme_flat = re.sub(r"\s+", " ", agent_readme)

    assert "不代表正式模型排行榜或外部 benchmark 结论" in chinese
    assert "不是全自动论文写作或自主科学发现系统" in chinese
    assert "不是临床决策支持工具" in chinese
    assert "not a formal verifier" in agent_readme_flat
    assert "not a strong security sandbox" in agent_readme_flat
    assert "controlled demonstrations" in agent_readme_flat


def test_readme_console_scripts_match_pyproject_entry_points() -> None:
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")

    scripts = pyproject["project"]["scripts"]
    for script_name, target in scripts.items():
        assert script_name in readme

        module_name, function_name = target.split(":", 1)
        module_path = REPO_ROOT / "src" / Path(*module_name.split(".")).with_suffix(".py")
        package_main_path = REPO_ROOT / "src" / Path(*module_name.split(".")) / "__main__.py"
        script_path = module_path if module_path.exists() else package_main_path
        assert script_path.exists(), f"{script_name} points to missing module {target}"
        assert f"def {function_name}(" in script_path.read_text(encoding="utf-8")


def test_readme_image_references_exist() -> None:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    image_refs = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", readme)

    assert image_refs, "README should include visual workflow references."
    local_refs = [ref for ref in image_refs if not ref.startswith(("http://", "https://"))]
    missing = [ref for ref in local_refs if not (REPO_ROOT / ref).exists()]
    assert not missing, f"README image references are missing: {missing}"


def test_research_agent_readme_example_scripts_exist() -> None:
    readme = (REPO_ROOT / "src" / "easyicu" / "research_agent" / "README.md").read_text(
        encoding="utf-8"
    )
    example_refs = sorted(set(re.findall(r"examples/[A-Za-z0-9_./-]+\.py", readme)))

    assert example_refs, "research_agent README should reference runnable examples."
    missing = [ref for ref in example_refs if not (REPO_ROOT / ref).exists()]
    assert not missing, f"research_agent README references missing examples: {missing}"


def test_repository_does_not_ignore_tests_or_contributing_guide() -> None:
    ignored = subprocess.run(
        [
            "git",
            "-C",
            str(REPO_ROOT),
            "check-ignore",
            "CONTRIBUTING.md",
            "tests/test_repository_contract.py",
        ],
        capture_output=True,
        text=True,
    )

    assert ignored.returncode == 1, ignored.stdout


def test_legacy_streamlit_package_is_decommissioned() -> None:
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    optional_dependencies = pyproject["project"]["optional-dependencies"]
    scripts = pyproject["project"]["scripts"]
    assert "webapp-legacy" not in optional_dependencies
    assert "easyicu-webapp-legacy" not in scripts
    assert not (REPO_ROOT / "src" / "easyicu" / "webapp").exists()

    tracked = subprocess.run(
        [
            "git",
            "-C",
            str(REPO_ROOT),
            "ls-files",
            "src/easyicu/webapp",
            "tests/webapp",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert tracked.stdout == ""
