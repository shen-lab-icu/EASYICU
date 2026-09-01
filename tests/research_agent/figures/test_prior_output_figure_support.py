from pathlib import Path

from easyicu.research_agent.figures.prior_output_support import (
    figure_parent_candidate_step_dirs,
    publication_label,
    short_figure_label,
)


def test_split_figure_reads_only_its_direct_parent(tmp_path: Path) -> None:
    steps = tmp_path / "steps"
    direct = steps / "02_primary"
    unrelated = steps / "01_other"
    direct.mkdir(parents=True)
    unrelated.mkdir()

    candidates, direct_only = figure_parent_candidate_step_dirs(
        steps_dir=steps,
        current_step_id="02_primary_figure",
    )

    assert candidates == [direct]
    assert direct_only is True


def test_legacy_overview_reads_other_steps_in_stable_order(tmp_path: Path) -> None:
    steps = tmp_path / "steps"
    (steps / "03_overview").mkdir(parents=True)
    (steps / "02_model").mkdir()
    (steps / "01_cohort").mkdir()

    candidates, direct_only = figure_parent_candidate_step_dirs(
        steps_dir=steps,
        current_step_id="03_overview",
    )

    assert [path.name for path in candidates] == ["01_cohort", "02_model"]
    assert direct_only is False


def test_legacy_labels_and_truncation_remain_stable() -> None:
    assert publication_label("sepsis3") == "Sepsis-3"
    assert publication_label("sep3_sofa2_max") == (
        "Experimental SOFA-2 Sepsis-3 phenotype"
    )
    assert publication_label("sep3_sofa2_max") != "Sepsis-3"
    assert publication_label("age_per_10y") == "Age, per 10 years"
    assert publication_label("custom_first") == "Custom"
    assert short_figure_label("123456", limit=5) == "1234..."
