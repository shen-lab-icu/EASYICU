from __future__ import annotations

from easyicu.webapp.concept_catalog import CONCEPT_DICTIONARY
from easyicu.webapp.mock_data import generate_mock_data


def test_mock_data_covers_current_web_concept_catalog() -> None:
    data, patient_ids = generate_mock_data(n_patients=12, hours=24)

    assert len(patient_ids) == 12
    assert set(CONCEPT_DICTIONARY) <= set(data)
