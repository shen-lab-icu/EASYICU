from __future__ import annotations

from pathlib import Path
import re
import shutil
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
    pattern = r"st\.(button|download_button)\([\s\S]{0,200}?width\s*=\s*['\"]stretch['\"]"
    app_path = REPO_ROOT / "src" / "easyicu" / "webapp" / "app.py"
    rg = shutil.which("rg")
    if rg is not None:
        result = subprocess.run(
            [
                rg,
                "-n",
                "-U",
                pattern,
                str(app_path),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 1, result.stdout
        return

    app_content = app_path.read_text(encoding="utf-8")
    assert re.search(pattern, app_content, flags=re.MULTILINE) is None


def test_webapp_concept_catalog_is_split_from_streamlit_app() -> None:
    catalog_path = REPO_ROOT / "src" / "easyicu" / "webapp" / "concept_catalog.py"
    assert catalog_path.exists(), "Large concept metadata should live outside app.py."

    app_content = (REPO_ROOT / "src" / "easyicu" / "webapp" / "app.py").read_text(encoding="utf-8")
    catalog_content = catalog_path.read_text(encoding="utf-8")

    assert "CONCEPT_DICTIONARY = {" not in app_content
    assert "CONCEPT_DESCRIPTIONS = {" not in app_content
    assert "CONCEPT_DICTIONARY = {" in catalog_content
    assert "CONCEPT_DESCRIPTIONS = {" in catalog_content


def test_webapp_global_styles_are_split_from_streamlit_app() -> None:
    styles_path = REPO_ROOT / "src" / "easyicu" / "webapp" / "styles.py"
    assert styles_path.exists(), "Large global CSS blocks should live outside app.py."

    app_content = (REPO_ROOT / "src" / "easyicu" / "webapp" / "app.py").read_text(encoding="utf-8")
    styles_content = styles_path.read_text(encoding="utf-8")

    assert "EasyICU paper-figure visual skin" not in app_content
    assert "EasyICU paper-figure visual skin" in styles_content
    assert "def render_global_styles" in styles_content


def test_webapp_i18n_texts_are_split_from_streamlit_app() -> None:
    i18n_path = REPO_ROOT / "src" / "easyicu" / "webapp" / "i18n.py"
    assert i18n_path.exists(), "Large language text dictionaries should live outside app.py."

    app_content = (REPO_ROOT / "src" / "easyicu" / "webapp" / "app.py").read_text(encoding="utf-8")
    i18n_content = i18n_path.read_text(encoding="utf-8")

    assert "TEXTS = {" not in app_content
    assert "TEXTS = {" in i18n_content
    assert "def get_text" in i18n_content
    assert "def strip_emoji" in i18n_content


def test_webapp_data_path_helpers_are_split_from_streamlit_app() -> None:
    data_paths_path = REPO_ROOT / "src" / "easyicu" / "webapp" / "data_paths.py"
    assert data_paths_path.exists(), "Real-data path and directory-browser helpers should live outside app.py."

    app_content = (REPO_ROOT / "src" / "easyicu" / "webapp" / "app.py").read_text(encoding="utf-8")
    data_paths_content = data_paths_path.read_text(encoding="utf-8")

    assert "def find_database_path" not in app_content
    assert "def _directory_input" not in app_content
    assert "def render_directory_structure_guide" not in app_content
    assert "def find_database_path" in data_paths_content
    assert "def _directory_input" in data_paths_content
    assert "def render_directory_structure_guide" in data_paths_content


def test_directory_browser_dialog_decorator_stays_with_browser_dialog() -> None:
    app_content = (REPO_ROOT / "src" / "easyicu" / "webapp" / "app.py").read_text(encoding="utf-8")
    data_paths_content = (REPO_ROOT / "src" / "easyicu" / "webapp" / "data_paths.py").read_text(
        encoding="utf-8"
    )

    decorator = '@st.dialog("Browse Server Folders / 浏览服务器目录", width="large")'
    assert decorator not in app_content
    assert f"{decorator}\ndef _render_directory_browser_dialog" in data_paths_content


def test_webapp_demo_data_helpers_are_split_from_streamlit_app() -> None:
    demo_data_path = REPO_ROOT / "src" / "easyicu" / "webapp" / "demo_data.py"
    assert demo_data_path.exists(), "Demo cohort data generation helpers should live outside app.py."

    app_content = (REPO_ROOT / "src" / "easyicu" / "webapp" / "app.py").read_text(encoding="utf-8")
    demo_data_content = demo_data_path.read_text(encoding="utf-8")

    assert "def _generate_mock_cohort_dashboard_data" not in app_content
    assert "def _generate_mock_multidb_data" not in app_content
    assert "def _build_mock_group_feature_data" not in app_content
    assert "def _generate_mock_cohort_dashboard_data" in demo_data_content
    assert "def _generate_mock_multidb_data" in demo_data_content
    assert "def _build_mock_group_feature_data" in demo_data_content


def test_webapp_home_renderers_are_split_from_streamlit_app() -> None:
    home_page_path = REPO_ROOT / "src" / "easyicu" / "webapp" / "home_page.py"
    assert home_page_path.exists(), "Home-page and data-overview renderers should live outside app.py."

    app_content = (REPO_ROOT / "src" / "easyicu" / "webapp" / "app.py").read_text(encoding="utf-8")
    home_page_content = home_page_path.read_text(encoding="utf-8")

    assert "def render_data_overview" not in app_content
    assert "def render_home_viz_mode" not in app_content
    assert "def render_data_overview" in home_page_content
    assert "def render_home_viz_mode" in home_page_content
    assert "Start Exploring" in home_page_content


def test_webapp_paper_figure_helpers_are_split_from_streamlit_app() -> None:
    paper_figures_path = REPO_ROOT / "src" / "easyicu" / "webapp" / "paper_figures.py"
    paper_figures_content = paper_figures_path.read_text(encoding="utf-8")
    app_content = (REPO_ROOT / "src" / "easyicu" / "webapp" / "app.py").read_text(encoding="utf-8")

    assert "def _render_inline_timeseries_svg" not in app_content
    assert "def _quality_snapshot_rows" not in app_content
    assert "def _render_paper_quality_panel" not in app_content
    assert "def _render_inline_timeseries_svg" in paper_figures_content
    assert "def _quality_snapshot_rows" in paper_figures_content
    assert "def _render_paper_quality_panel" in paper_figures_content


def test_webapp_cohort_filter_config_is_split_from_streamlit_app() -> None:
    cohort_config_path = REPO_ROOT / "src" / "easyicu" / "webapp" / "cohort_config.py"
    assert cohort_config_path.exists(), "Cohort filter constants should live outside app.py."

    app_content = (REPO_ROOT / "src" / "easyicu" / "webapp" / "app.py").read_text(encoding="utf-8")
    cohort_config_content = cohort_config_path.read_text(encoding="utf-8")

    assert "DISEASE_COHORT_CONFIG = {" not in app_content
    assert "SEPSIS_MODE_CONFIG = {" not in app_content
    assert "DISEASE_COHORT_CONFIG = {" in cohort_config_content
    assert "SEPSIS_MODE_CONFIG = {" in cohort_config_content
    assert "ICD_FILTER_DATABASES" in cohort_config_content


def test_webapp_cohort_filter_helpers_are_split_from_streamlit_app() -> None:
    cohort_filters_path = REPO_ROOT / "src" / "easyicu" / "webapp" / "cohort_filters.py"
    assert cohort_filters_path.exists(), "Cohort filtering helpers should live outside app.py."

    app_content = (REPO_ROOT / "src" / "easyicu" / "webapp" / "app.py").read_text(encoding="utf-8")
    cohort_filters_content = cohort_filters_path.read_text(encoding="utf-8")

    helper_defs = [
        "def _split_query_tokens",
        "def _match_ids_by_icd_tokens",
        "def _post_filter_cohort_data",
        "def _get_age_series",
        "def _get_los_hours_series",
        "def _get_sex_series",
        "def _pick_death_stay",
        "def _get_death_series",
    ]
    for helper_def in helper_defs:
        assert helper_def not in app_content
        assert helper_def in cohort_filters_content


def test_webapp_icd_preview_helpers_are_split_from_streamlit_app() -> None:
    icd_preview_path = REPO_ROOT / "src" / "easyicu" / "webapp" / "icd_preview.py"
    assert icd_preview_path.exists(), "ICD preview rendering and matching should live outside app.py."

    app_content = (REPO_ROOT / "src" / "easyicu" / "webapp" / "app.py").read_text(encoding="utf-8")
    icd_preview_content = icd_preview_path.read_text(encoding="utf-8")

    helper_defs = [
        "def _clear_icd_preview_state",
        "def _render_icd_preview_main_panel",
        "def _preview_icd_match",
    ]
    for helper_def in helper_defs:
        assert helper_def not in app_content
        assert helper_def in icd_preview_content


def test_webapp_cohort_workspace_state_helpers_are_split_from_streamlit_app() -> None:
    cohort_workspace_path = REPO_ROOT / "src" / "easyicu" / "webapp" / "cohort_workspace.py"
    assert cohort_workspace_path.exists(), "Shared cohort workspace state helpers should live outside app.py."

    app_content = (REPO_ROOT / "src" / "easyicu" / "webapp" / "app.py").read_text(encoding="utf-8")
    cohort_workspace_content = cohort_workspace_path.read_text(encoding="utf-8")

    helper_defs = [
        "def _cohort_demo_workspace_ready",
        "def _ensure_cohort_demo_workspace",
        "def _ensure_cohort_figure_demo_data",
        "def _cohort_real_workspace_ready",
        "def _cohort_real_workspace_matches_sidebar",
        "def _ensure_cohort_real_workspace",
    ]
    for helper_def in helper_defs:
        assert helper_def not in app_content
        assert helper_def in cohort_workspace_content


def test_feature_definition_panel_lives_inside_tutorial_tab() -> None:
    app_content = (REPO_ROOT / "src" / "easyicu" / "webapp" / "app.py").read_text(encoding="utf-8")

    render_home_idx = app_content.index("        render_home()")
    feature_panel_idx = app_content.index("        _render_feature_definition_panel(lang)")
    quick_viz_tab_idx = app_content.index("    with tab2:")

    assert render_home_idx < feature_panel_idx < quick_viz_tab_idx
