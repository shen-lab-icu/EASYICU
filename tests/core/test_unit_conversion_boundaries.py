"""Converted µmol/L values that sit on a decimal threshold decide as the threshold reads.

lead-r13 float audit group H: AUMC and HiRID chart creatinine and bilirubin
in µmol/L. 106.08 µmol/L is exactly 1.2 mg/dL, 442 is 5.0 and 34.2 is
2.0, so SOFA and KDIGO must class them on the boundary, under a float32
and a float64 load alike. ricu's rounded factors (x 0.011309, x 0.058467)
put them 1e-4 below and would not.
"""

import json
import re
from importlib import resources

import numpy as np
import pandas as pd
import pytest

from easyicu.callbacks import sofa_liver, sofa_renal
from easyicu.concept.parser import apply_binary_op, parse_binary_op
from easyicu.scores.kdigo_aki import _calc_aki_stage_creat
from easyicu.scores.sofa2 import sofa2_renal

LOADS = [np.float32, np.float64]


def conversion(concept, database):
    text = resources.files("easyicu").joinpath("data/concept-dict.json").read_text()
    callback = json.loads(text)[concept]["sources"][database][0]["callback"]
    return parse_binary_op(re.search(r"binary_op\(`.`,\s*[^)]+\)", callback).group(0))


def converted(concept, database, umol, dtype):
    symbol, value = conversion(concept, database)
    loaded = pd.Series(np.array(umol, dtype=dtype).astype(float))
    return apply_binary_op(symbol, loaded, value)


@pytest.mark.parametrize("database", ["aumc", "hirid"])
@pytest.mark.parametrize("dtype", LOADS)
def test_creatinine_on_a_threshold_is_classed_on_it(database, dtype):
    crea = converted("crea", database, [106.08, 176.8, 442.0, 441.0], dtype)
    assert sofa_renal(crea).tolist() == [1, 2, 4, 3]
    assert sofa2_renal(crea).tolist() == [0, 1, 3, 3]
    # KDIGO: 1.2 mg/dL against a 0.8 baseline is 1.5x (stage 1); 4.0 after
    # 3.7 within 48 h is >= 4.0 with a 0.3 rise (stage 3).
    stage = _calc_aki_stage_creat(
        converted("crea", database, [106.08, 353.6], dtype),
        converted("crea", database, [np.nan, 327.08], dtype),
        converted("crea", database, [70.72, np.nan], dtype),
    )
    assert stage.tolist() == [1, 3]


@pytest.mark.parametrize("database", ["aumc", "hirid"])
@pytest.mark.parametrize("dtype", LOADS)
def test_bilirubin_on_a_threshold_is_classed_on_it(database, dtype):
    bili = converted("bili", database, [20.52, 34.2, 102.6, 205.2, 34.1], dtype)
    assert sofa_liver(bili).tolist() == [1, 2, 3, 4, 1]
