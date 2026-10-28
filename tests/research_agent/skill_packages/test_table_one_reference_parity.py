"""Prepared-cohort Table 1 scripts must match the governed executor cell for cell."""

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from easyicu.research_agent.execution.runners.table_one_executor import table_one_executor_code
from easyicu.research_agent.skill_packages.cohort_characterization_table_one.scripts.example_data import (
    example_cohort, example_spec,
)
from easyicu.research_agent.skill_packages.cohort_characterization_table_one.scripts.load_cohort import load_cohort
from easyicu.research_agent.skill_packages.cohort_characterization_table_one.scripts.run_analysis import run_analysis
from tests.support.reference_scripts import assert_csv_cells_equal, run_host_executor


def test_table_one_matches_host_cells_and_denominators(tmp_path, monkeypatch):
    frame, spec = example_cohort(), example_spec()
    host = run_host_executor(tmp_path, monkeypatch, frame, spec.host_step(), table_one_executor_code)
    standalone = tmp_path / "standalone"
    run_analysis(load_cohort(frame, spec), spec, standalone)
    assert_csv_cells_equal(standalone / "table_one.csv", host / "table_one.csv")
    table = pd.read_csv(standalone / "table_one.csv")
    assert table["group_missing_excluded_n"].eq(1).all()
    age = table.loc[table.variable.eq("age")]
    assert age.denominator_n.tolist() == [119, 59, 60]
    assert age.nonmissing_n.tolist() == [116, 57, 59]
    assert set(table.test_name) == {"welch_t", "mann_whitney_u", "chi_square", "fisher_exact"}


def test_other_comparison_test_fails_table_one_parity(tmp_path, monkeypatch):
    frame, spec = example_cohort(), example_spec()
    host = run_host_executor(tmp_path, monkeypatch, frame, spec.host_step(), table_one_executor_code)
    payload = spec.model_dump(mode="json")
    payload["table_one"]["variables"][0]["test"] = "mann_whitney_or_kruskal"
    changed = type(spec).model_validate(payload)
    out = tmp_path / "different_summary"
    run_analysis(load_cohort(frame, changed), changed, out)
    expected = pd.read_csv(host / "table_one.csv", dtype=str, keep_default_na=False)
    actual = pd.read_csv(out / "table_one.csv", dtype=str, keep_default_na=False)
    # Ignore the declaration digest: the numeric cells themselves must disagree.
    with pytest.raises(AssertionError):
        assert_frame_equal(actual[["p_value"]], expected[["p_value"]], check_exact=True)
