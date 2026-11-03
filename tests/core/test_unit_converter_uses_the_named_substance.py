"""UnitConverter uses the substance it is given and keeps decilitre scale.

11-03 external review, reproduced through the public class: 9 mg/dL lactate
gave 0.4996 mmol/L (the glucose factor), 1 mg/dL bilirubin gave 88.4 µmol/L
(the creatinine factor), 4 mg/dL calcium gave 0.222 mmol/L; and 1 mmol/dL or
1 mEq/dL equalled 1 per litre because the aliases dropped the tenfold scale.
"""

from __future__ import annotations

import pytest

from easyicu.utils.unit_conversion import UnitConverter


@pytest.mark.parametrize("value,source,target,substance,expected", [
    (9, "mg/dL", "mmol/L", "lactate", 1.0),
    (9, "mg/dL", "mmol/L", "lact", 1.0),
    (1, "mg/dL", "µmol/L", "bilirubin", 17.1),
    (4, "mg/dL", "mmol/L", "calcium", 1.0),
    (180.16, "mg/dL", "mmol/L", "glucose", 10.0),
    (1, "mg/dL", "umol/L", "creatinine", 88.4),
    (1, "mmol/dL", "mmol/L", None, 10.0),
    (1, "mEq/dL", "mEq/L", None, 10.0),
    (12, "g/dL", "g/L", None, 120.0),
])
def test_each_conversion_uses_its_own_factor(value, source, target, substance, expected):
    assert UnitConverter.convert(value, source, target, substance=substance) == pytest.approx(expected, rel=1e-3)


@pytest.mark.parametrize("source,target", [("mg/dL", "mmol/L"), ("mg/dL", "umol/L"), ("mmol/L", "mg/dL")])
def test_mass_to_molar_without_a_substance_is_refused(source, target):
    assert not UnitConverter.can_convert(source, target)
    with pytest.raises(ValueError):
        UnitConverter.convert(1.0, source, target)


def test_a_substance_without_its_own_factor_is_refused_rather_than_guessed():
    assert not UnitConverter.can_convert("mg/dL", "mmol/L", substance="sodium")


def test_aliases_name_only_the_same_scale():
    assert UnitConverter.normalize_unit("mmol/dL") != UnitConverter.normalize_unit("mmol/L")
    assert UnitConverter.normalize_unit("mEq/dL") != UnitConverter.normalize_unit("mEq/L")
    assert UnitConverter.normalize_unit("cal") != UnitConverter.normalize_unit("kcal")
