import pandas as pd
import pytest

pytest.importorskip("plotly")

from easyicu.visualization.patient import render_patient_report


def test_export_hides_identifier_and_selects_only_requested_patient(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    data = {
        "hr": pd.DataFrame(
            {"stay_id": [982763410, 982763411], "time": [1.0, 1.0], "hr": [70.0, 190.0]}
        )
    }
    figure = render_patient_report(982763410, data, output_format="figure")
    assert "982763410" not in figure.to_json()
    assert list(figure.data[0].y) == [70.0]
    path = render_patient_report(982763410, data)
    assert "982763410" not in path
    assert "982763410" not in (tmp_path / path).read_text()
    identified = render_patient_report(
        982763410, data, output_format="figure", include_identifiers=True
    )
    assert "982763410" in identified.to_json()
