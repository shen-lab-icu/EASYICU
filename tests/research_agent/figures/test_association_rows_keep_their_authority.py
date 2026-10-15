"""Figure row identity and uncertainty must come from the same source row."""
from pathlib import Path

import pandas as pd
import pytest

from easyicu.research_agent.authority.evidence_store import EvidenceStore
from easyicu.research_agent.figures.causal import _load_effect
from easyicu.research_agent.figures.skill import _normalise_association_frame


def test_missing_interval_is_not_fabricated_from_a_point_estimate() -> None:
    frame = pd.DataFrame({"term": ["age"], "estimate": [2.1]})
    with pytest.raises(ValueError, match="confidence bounds"):
        _normalise_association_frame(frame, primary_exposure="age")


def test_grid_first_row_is_not_an_identified_primary_contrast() -> None:
    frame = pd.DataFrame({
        "exposure_value": [1.0, 2.0], "reference_exposure_value": [2.1, 2.1],
        "estimate": [0.7, 0.9], "ci_low": [0.6, 0.8], "ci_high": [0.8, 1.0],
    })
    result = _normalise_association_frame(frame, primary_exposure="lactate")
    assert not result.attrs["primary_row_identified"]
    assert result["estimate"].tolist() == [0.7, 0.9]


def test_exposure_coefficient_is_identified_even_when_not_first() -> None:
    frame = pd.DataFrame({
        "term": ["age", "lactate"], "estimate": [1.01, 1.2],
        "ci_low": [1.0, 1.1], "ci_high": [1.02, 1.3],
    })
    result = _normalise_association_frame(frame, primary_exposure="lactate")
    assert result.attrs["primary_row_identified"]
    assert result["label"].iloc[0].casefold() == "lactate"


def test_causal_point_interval_and_scale_are_read_from_one_row(tmp_path: Path) -> None:
    evidence = EvidenceStore(tmp_path)
    path = tmp_path / "causal_effect.csv"
    pd.DataFrame([
        {"estimate": 2.1, "ci_low": None, "ci_high": None, "scale": "risk_difference"},
        {"estimate": 0.5, "ci_low": 0.3, "ci_high": 0.8, "scale": "odds_ratio"},
    ]).to_csv(path, index=False)
    evidence.register_file(kind="table", source_path=path, evidence_id="causal_effect", description="Effect rows", producer="coder")
    result = _load_effect(evidence, tmp_path)
    assert result is not None
    assert result[1:] == (0.5, 0.3, 0.8, "odds_ratio")
