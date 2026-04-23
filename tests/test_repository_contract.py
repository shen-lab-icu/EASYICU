from __future__ import annotations

from pathlib import Path
import subprocess
import tomllib


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
        "src/easyicu/webapp/__main__.py",
        "src/easyicu/webapp/app.py",
        "src/easyicu/webapp/llm_chat.py",
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


def test_repository_includes_citation_metadata() -> None:
    citation = REPO_ROOT / "CITATION.cff"
    assert citation.exists()

    content = citation.read_text(encoding="utf-8")
    assert 'title: "EasyICU"' in content
    assert 'type: software' in content
    assert 'repository-code: "https://github.com/shen-lab-icu/EASYICU"' in content


def test_readmes_link_to_citation_metadata() -> None:
    english = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    chinese = (REPO_ROOT / "README_zh.md").read_text(encoding="utf-8")

    assert "## Paper, Citation & Reproducibility" in english
    assert "[CITATION.cff](CITATION.cff)" in english

    assert "## 论文、引用与可复现" in chinese
    assert "[CITATION.cff](CITATION.cff)" in chinese


def test_repository_does_not_ignore_tests_or_contributing_guide() -> None:
    ignored = subprocess.run(
        [
            "git",
            "-C",
            str(REPO_ROOT),
            "check-ignore",
            "CONTRIBUTING.md",
            "tests/test_llm_chat.py",
            "tests/test_webapp_launch.py",
        ],
        capture_output=True,
        text=True,
    )

    assert ignored.returncode == 1, ignored.stdout


def test_webapp_buttons_do_not_use_stretch_width_keyword() -> None:
    result = subprocess.run(
        [
            "rg",
            "-n",
            "-U",
            r"st\.(button|download_button)\([\s\S]{0,200}?width\s*=\s*['\"]stretch['\"]",
            str(REPO_ROOT / "src" / "easyicu" / "webapp" / "app.py"),
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1, result.stdout
