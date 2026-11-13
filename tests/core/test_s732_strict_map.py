"""Strict mapping is opt-in; legacy numeric/string and logical maps keep their contracts."""

import pandas as pd
import pytest

from easyicu.concept import _apply_callback
from easyicu.concept import ConceptSource


@pytest.mark.parametrize("target", ["1", "'category'"])
@pytest.mark.parametrize("var", [None, "state", "sub_var", "val_col"])
def test_strict_map_drops_unrecognized_labels(target, var):
    suffix = f", var = '{var}'" if var else ""
    source = ConceptSource(table="listitems", sub_var="state",
                           callback=f"apply_map(c(`known` = {target}){suffix}, strict = TRUE)")
    frame = pd.DataFrame({"score": ["known", "1", "unexpected", None],
                          "state": ["known", "1", "unexpected", None]})
    out = _apply_callback(frame, source, "score")
    column = "state" if var in {"state", "sub_var"} else "score"
    assert out[column].iloc[0] == (1 if target == "1" else "category")
    assert out[column].iloc[1:].isna().all()


@pytest.mark.parametrize("option", ["", ", strict = FALSE"])
@pytest.mark.parametrize("target", ["1", "'category'"])
def test_legacy_map_preserves_unmapped_values(option, target):
    source = ConceptSource(callback=f"apply_map(c(`known` = {target}){option})")
    out = _apply_callback(pd.DataFrame({"score": ["known", "2", "unexpected"]}), source, "score")
    assert out.score.tolist() == [1 if target == "1" else "category", "2", "unexpected"]


def test_legacy_logical_map_still_rejects_unknown():
    source = ConceptSource(callback="apply_map(c(`known` = TRUE), strict = FALSE)")
    out = _apply_callback(pd.DataFrame({"score": ["known", "1"]}), source, "score")
    assert out.score.iloc[0]
    assert pd.isna(out.score.iloc[1])
