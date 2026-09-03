from __future__ import annotations

from pathlib import Path
import re
import subprocess

import yaml

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.9/3.10 test runtime
    import tomli as tomllib


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_DIR = REPO_ROOT / ".github" / "workflows"


def _workflow_paths() -> list[Path]:
    return sorted([*WORKFLOW_DIR.glob("*.yml"), *WORKFLOW_DIR.glob("*.yaml")])


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


def test_openai_json_schema_transport_has_a_consistent_sdk_floor() -> None:
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    optional = pyproject["project"]["optional-dependencies"]

    for extra in ("webapp", "agentic"):
        assert "openai>=1.40.0" in optional[extra]

    assert '"openai>=1.40.0"' in (
        WORKFLOW_DIR / "ci.yml"
    ).read_text(encoding="utf-8")
    assert '"openai>=1.40.0"' in (
        WORKFLOW_DIR / "research_agent_ci.yml"
    ).read_text(encoding="utf-8")


def test_pyproject_license_uses_spdx_string() -> None:
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert pyproject["project"]["license"] == "MIT"


def test_pyproject_dev_extra_includes_build_for_release_contract() -> None:
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    dev_dependencies = pyproject["project"]["optional-dependencies"]["dev"]
    assert "build>=1.2" in dev_dependencies


def test_parallel_pytest_workflows_install_xdist() -> None:
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    dev_dependencies = pyproject["project"]["optional-dependencies"]["dev"]

    assert "pytest-xdist>=3.0" in dev_dependencies
    research_agent_workflow = (
        WORKFLOW_DIR / "research_agent_ci.yml"
    ).read_text(encoding="utf-8")
    assert '"pytest-xdist>=3.0"' in research_agent_workflow

    for workflow_path in (WORKFLOW_DIR / "ci.yml", WORKFLOW_DIR / "research_agent_ci.yml"):
        workflow = workflow_path.read_text(encoding="utf-8")
        assert "-n auto --dist loadfile" in workflow


def test_optional_anthropic_adapter_keeps_a_real_coverage_job() -> None:
    """An importorskip is only safe while some job installs the real SDK.

    ``anthropic`` lives in the ``webapp``/``agentic`` extras, so the minimum
    dependency stack skips the native adapter tests. Without a job that
    actually installs the extra, that skip silently removes the Claude
    Messages transport from CI entirely -- green, and never tested.
    """

    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    agentic = pyproject["project"]["optional-dependencies"]["agentic"]
    assert any(item.startswith("anthropic") for item in agentic)

    adapter_test = (
        REPO_ROOT
        / "tests"
        / "research_agent"
        / "providers"
        / "test_anthropic_messages_client.py"
    ).read_text(encoding="utf-8")
    assert 'pytest.importorskip("anthropic")' in adapter_test

    workflow = (WORKFLOW_DIR / "research_agent_ci.yml").read_text(encoding="utf-8")
    assert "anthropic-adapter:" in workflow
    assert '".[dev,agentic]"' in workflow
    # The explicit import assertion is the part that stops a missing extra
    # from degrading into a skipped file.
    assert "import anthropic; print" in workflow
    assert "tests/research_agent/providers/test_anthropic_messages_client.py" in workflow


def test_manifest_does_not_reference_missing_optional_payloads() -> None:
    manifest = (REPO_ROOT / "MANIFEST.in").read_text(encoding="utf-8")

    assert "CHANGELOG.md" not in manifest
    assert "src/easyicu/extdata" not in manifest


def test_native_webserver_static_assets_are_packaged() -> None:
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    manifest = (REPO_ROOT / "MANIFEST.in").read_text(encoding="utf-8")

    package_data = pyproject["tool"]["setuptools"]["package-data"]
    assert package_data["easyicu.webserver"] == [
        "static/index.html",
        "static/THIRD_PARTY_NOTICES.md",
        "static/css/*.css",
        "static/js/*.js",
        "static/assets/demo/*.png",
        "static/assets/demo/*.html",
        "static/assets/demo/*.pdf",
        "static/vendor/echarts/*",
        "pi_copilot/tool_catalog.json",
        "pi_copilot/node_app/package.json",
        "pi_copilot/node_app/package-lock.json",
        "pi_copilot/node_app/README.md",
        "pi_copilot/node_app/THIRD_PARTY_NOTICES.md",
        "pi_copilot/node_app/src/main.mjs",
        "pi_copilot/node_app/src/event-projection.mjs",
        "pi_copilot/node_app/src/post-tool-finalization.mjs",
        "pi_copilot/node_app/src/shell-budget.mjs",
        "pi_copilot/node_app/src/session-lifecycle.mjs",
        "pi_copilot/node_app/src/skills/web-prototype/SKILL.md",
    ]
    assert "recursive-include src/easyicu/webserver/static *.html *.css *.js" in manifest
    assert (
        "recursive-include src/easyicu/webserver/static/assets *.png *.pdf" in manifest
    )
    assert "recursive-include src/easyicu/webserver/static/vendor *" in manifest

    required_assets = [
        "src/easyicu/webserver/static/index.html",
        "src/easyicu/webserver/static/js/app.js",
        "src/easyicu/webserver/static/js/screens-viz-demo-drilldown.js",
        "src/easyicu/webserver/static/js/screens-viz-patient-features.js",
        "src/easyicu/webserver/static/css/app.css",
        "src/easyicu/webserver/static/assets/demo/e1-publication-figure.png",
        "src/easyicu/webserver/static/assets/demo/system-validation-report.html",
        "src/easyicu/webserver/static/assets/demo/system-validation-report.pdf",
        "src/easyicu/webserver/static/vendor/echarts/echarts.common.min.js",
        "src/easyicu/webserver/static/vendor/echarts/LICENSE",
        "src/easyicu/webserver/static/vendor/echarts/NOTICE",
    ]
    missing = [path for path in required_assets if not (REPO_ROOT / path).exists()]
    assert not missing, f"Native FastAPI static assets are missing from source tree: {missing}"


def test_python39_compatible_union_annotations_use_future_import() -> None:
    files_requiring_future = [
        "src/easyicu/io/attach.py",
        "src/easyicu/concept/__init__.py",
        "src/easyicu/concept/callbacks.py",
        "src/easyicu/concept/parser.py",
        "src/easyicu/config.py",
        "src/easyicu/io/data_converter.py",
        "src/easyicu/io/data_utils.py",
        "src/easyicu/datasource.py",
        "src/easyicu/io/download.py",
        "src/easyicu/feature_compare.py",
        "src/easyicu/hosted_llm_server.py",
        "src/easyicu/io/import_data.py",
        "src/easyicu/utils/logging_utils.py",
        "src/easyicu/resources.py",
        "src/easyicu/scripts/extract_features.py",
        "src/easyicu/io/ts_utils.py",
    ]

    missing = []
    for rel_path in files_requiring_future:
        content = (REPO_ROOT / rel_path).read_text(encoding="utf-8")
        if "from __future__ import annotations" not in content:
            missing.append(rel_path)

    assert not missing, f"Python 3.9 runtime needs deferred annotation evaluation in: {missing}"


def test_repository_includes_contribution_guide() -> None:
    assert (REPO_ROOT / "CONTRIBUTING.md").exists()


def test_repository_security_policy_is_private_and_data_safe() -> None:
    security_path = REPO_ROOT / "SECURITY.md"

    assert security_path.exists()
    security = security_path.read_text(encoding="utf-8")
    assert "/security/advisories/new" in security
    assert "Do not include patient-level data" in security
    assert "not a strong security sandbox" in security
    assert "Current default branch" in security


def test_pull_request_template_requires_scope_evidence_and_review() -> None:
    template_path = REPO_ROOT / ".github" / "pull_request_template.md"

    assert template_path.exists()
    template = template_path.read_text(encoding="utf-8")
    for required in (
        "Primary owner/workstream",
        "Out of scope",
        "Evidence class and claim ceiling",
        "Exact commands and results",
        "Independent domain review",
        "The PR author is not the independent reviewer",
        "docs/release_checklist.md",
    ):
        assert required in template


def test_release_checklist_preserves_formal_evidence_boundaries() -> None:
    checklist_path = REPO_ROOT / "docs" / "release_checklist.md"

    assert checklist_path.exists()
    checklist = checklist_path.read_text(encoding="utf-8")
    for required in (
        "full exact-head CI",
        "mapping_only",
        "independent clinical reviewer",
        "validated dependency snapshot",
        "vulnerability audit",
        "SBOM",
        "artifact provenance",
        "Tier 1",
        "Tier 2",
        "Tier 3",
        "release remains blocked",
    ):
        assert required in checklist


def test_repository_includes_ci_workflow() -> None:
    assert (REPO_ROOT / ".github" / "workflows" / "ci.yml").exists()


def test_ci_workflow_runs_supported_python_matrix() -> None:
    """CI must still prove every supported Python, and prove it unfiltered.

    2026-08-17: the matrix became event-conditional so a pull request gates on
    one version while ``main`` / ``workflow_dispatch`` keep the full sweep. The
    old assertion pinned the literal list ``["3.10", "3.11", "3.12"]``, so it
    failed on formatting rather than on lost coverage. This asserts the two
    properties that actually matter: every supported version is still reachable
    in the matrix, and the suite runs with the marker filter cancelled (the
    pytest.ini dev default is ``-m "not slow"``, which must never silently
    narrow a CI run).
    """
    workflow_path = REPO_ROOT / ".github" / "workflows" / "ci.yml"
    workflow = workflow_path.read_text(encoding="utf-8")
    parsed = yaml.safe_load(workflow)

    matrix_versions = parsed["jobs"]["test"]["strategy"]["matrix"]["python-version"]
    rendered = matrix_versions if isinstance(matrix_versions, str) else str(matrix_versions)
    for version in ("3.10", "3.11", "3.12"):
        assert version in rendered, f"CI no longer reaches Python {version}"

    assert "python-version: ${{ matrix.python-version }}" in workflow
    assert "ruff check src tests" in workflow

    # Every pytest invocation in CI must cancel the dev-default marker filter.
    unfiltered = [
        line.strip()
        for line in workflow.splitlines()
        if "pytest" in line
        and not line.strip().startswith("#")
        and '"pytest' not in line
        and "pip install" not in line
    ]
    assert unfiltered, "ci.yml runs no pytest at all"
    for line in unfiltered:
        assert '-m ""' in line, f"CI pytest call is marker-filtered: {line}"


def test_external_workflow_actions_are_immutable() -> None:
    violations = []
    action_ref = re.compile(r"\buses:\s*([^@\s]+)@([^\s#]+)")
    commit_sha = re.compile(r"[0-9a-f]{40}")
    image_digest = re.compile(r"sha256:[0-9a-f]{64}")

    workflow_paths = _workflow_paths()
    assert workflow_paths, "Expected at least one GitHub Actions workflow."
    for workflow_path in workflow_paths:
        for line_number, line in enumerate(
            workflow_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            match = action_ref.search(line)
            if match is None:
                continue
            action, ref = match.groups()
            if action.startswith("./"):
                continue
            if commit_sha.fullmatch(ref) or image_digest.fullmatch(ref):
                continue
            violations.append(f"{workflow_path.name}:{line_number}: {action}@{ref}")

    assert not violations, "External workflow actions must use immutable refs:\n" + "\n".join(
        violations
    )


def test_workflows_default_to_read_only_repository_access() -> None:
    violations = []
    write_permission = re.compile(r"(?m)^\s+[a-z-]+:\s*write\s*(?:#.*)?$")
    workflow_paths = _workflow_paths()
    assert workflow_paths, "Expected at least one GitHub Actions workflow."
    for workflow_path in workflow_paths:
        workflow = workflow_path.read_text(encoding="utf-8")
        top_level = workflow.split("\njobs:", maxsplit=1)[0]
        if "\npermissions:\n  contents: read\n" not in f"\n{top_level}\n":
            violations.append(workflow_path.name)
        if write_permission.search(workflow):
            violations.append(f"{workflow_path.name} declares a write permission")

    assert not violations, (
        "Workflows must declare top-level read-only repository access: "
        + ", ".join(violations)
    )


def test_dependabot_tracks_pinned_github_actions() -> None:
    dependabot = (REPO_ROOT / ".github" / "dependabot.yml").read_text(
        encoding="utf-8"
    )

    assert 'package-ecosystem: "github-actions"' in dependabot
    assert 'directory: "/"' in dependabot
    assert 'interval: "weekly"' in dependabot


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


def test_repository_does_not_ignore_contract_tests_or_governance_docs() -> None:
    ignored = subprocess.run(
        [
            "git",
            "-C",
            str(REPO_ROOT),
            "check-ignore",
            "CONTRIBUTING.md",
            "SECURITY.md",
            ".github/pull_request_template.md",
            "docs/release_checklist.md",
            "tests/governance/test_repository_contract.py",
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
