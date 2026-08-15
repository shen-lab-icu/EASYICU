from __future__ import annotations

import pandas as pd
import pytest

import easyicu
from easyicu.callbacks import UnsupportedClinicalScoreError


@pytest.mark.clinical_conformance
def test_incomplete_apache_ii_public_api_fails_closed() -> None:
    values = pd.Series([75.0])

    with pytest.raises(UnsupportedClinicalScoreError) as caught:
        easyicu.apache_ii_score(values, values, values, values, values)

    assert caught.value.code == "apache_ii_not_implemented"
    assert "chronic health" in str(caught.value).lower()
