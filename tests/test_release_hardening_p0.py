from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

import easyicu


REPO_ROOT = Path(__file__).resolve().parents[1]


def _module_function_count(path: Path, name: str) -> int:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return sum(
        1
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name
    )


def test_public_api_reports_packaged_sources_by_default() -> None:
    sources = set(easyicu.list_available_sources())

    assert {"miiv", "eicu", "mimic", "hirid", "aumc", "sic"} <= sources


def test_get_concept_info_uses_public_dictionary_api() -> None:
    info = easyicu.get_concept_info("hr")

    assert info["name"] == "hr"
    assert info["description"] == "heart rate"
    assert info["category"] == "vitals"
    assert info["units"] == ["bpm", "/min"]
    assert info["unit"] == "bpm"
    assert {"miiv", "eicu", "mimic", "hirid", "aumc", "sic"} <= set(info["sources"])
    assert "class_name" in info
    assert "callback" in info
    assert isinstance(info["sub_concepts"], list)
    assert isinstance(info["depends_on"], list)


def test_sofa2_import_state_is_consistent() -> None:
    from easyicu import sofa2_score

    assert callable(sofa2_score)
    assert easyicu._HAS_SOFA2 is True
    assert "sofa2_score" in easyicu.__all__
    assert easyicu.sofa2_score is sofa2_score
    assert easyicu._HAS_SEPSIS_SOFA2 is True
    assert "sep3_sofa2" in easyicu.__all__
    assert "label_sep3_sofa2" in easyicu.__all__


def test_data_env_imports_with_datasource_config_alias() -> None:
    import easyicu.data_env as data_env

    assert data_env.SrcEnv.__name__ == "SrcEnv"
    assert data_env.DataEnv.__name__ == "DataEnv"


def test_known_duplicate_top_level_functions_are_collapsed() -> None:
    assert (
        _module_function_count(
            REPO_ROOT / "src" / "easyicu" / "concept_callbacks.py",
            "_callback_miiv_icu_patients_filter",
        )
        == 1
    )
    assert (
        _module_function_count(
            REPO_ROOT / "src" / "easyicu" / "webapp" / "app.py",
            "_has_any_source_recursive",
        )
        == 1
    )


def test_project_config_import_does_not_create_project_dirs(tmp_path: Path) -> None:
    project_root = tmp_path / "project-root"
    env = {
        **os.environ,
        "PYTHONPATH": str(REPO_ROOT / "src"),
        "EASYICU_PROJECT_ROOT": str(project_root),
    }

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import easyicu.project_config as c; print(c.OUTPUT_DIR); print(c.CACHE_DIR); print(c.LOGS_DIR)",
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert not (project_root / "output").exists()
    assert not (project_root / ".cache").exists()
    assert not (project_root / "logs").exists()


def test_publication_figure_path_uses_env_or_repo_relative_candidates() -> None:
    source = (REPO_ROOT / "src" / "easyicu" / "webapp" / "app.py").read_text(encoding="utf-8")

    assert "EASYICU_PUBLICATION_FIGURE_DIR" in source
    assert "/Users/haibo/Documents/GitHub" not in source
    assert "image2_generated_review" in source

