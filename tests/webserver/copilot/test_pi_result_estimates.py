"""The registered primary estimate reaches the reader without being recomputed.

The adjusted-association producer writes one typed estimate table per model.
The result summary reads the primary model's contrasts from it; the research
overview leads with them.  Any doubt about which table or scale is primary
returns nothing rather than a guessed headline.
"""

import json
import shutil
import subprocess

import pytest

from tests.webserver.copilot.pi_copilot_static_fixtures import (
    _load_guided_pi_module_harness as _load_guided_pi_module_harness,
    _read,
)

ESTIMATE_HEADERS = [
    "fit_status", "estimate", "ci_low", "ci_high", "effect_scale", "exposure",
    "analysis_role", "n", "n_events", "exposure_level", "reference_level",
    "contrast", "is_primary_contrast",
]


def _estimate_row(level, estimate, low, high, primary, *, role="primary", scale="odds_ratio", status="fitted"):
    return [status, estimate, low, high, scale, "aki_stage_strict", role, "812", "88",
            level, "0", "", "True" if primary else "False"]


def run_js(script):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is not installed")
    return json.loads(subprocess.check_output([node, "--eval", script], text=True))


def summarize(tables, plan=None):
    return run_js(f"""
      global.window = {{}};
      eval({_read('js/screens-guided-pi-result-summary.js')!r});
      const result = window.EU_GUIDED_PI_RESULT_SUMMARY.summarize({json.dumps({"tables": tables})}, {json.dumps(plan or {})});
      process.stdout.write(JSON.stringify(result.estimates));
    """)


def test_the_primary_models_contrasts_lead_with_the_primary_one():
    table = {
        "name": "adjusted_association_estimates.csv",
        "headers": ESTIMATE_HEADERS,
        "rows": [
            _estimate_row("1", "1.4123", "0.8", "2.4", False),
            _estimate_row("3", "3.2051", "1.7702", "5.8031", True),
            _estimate_row("2", "2.01", "1.1", "3.6", False),
        ],
    }
    plan = {"display_labels": {'aki_stage_strict="3"': "KDIGO 3 期", 'aki_stage_strict="0"': "无 AKI"}}

    estimates = summarize([table], plan)

    assert [row["contrast"] for row in estimates] == ["3 vs 0", "1 vs 0", "2 vs 0"]
    primary = estimates[0]
    assert primary["primary"] is True
    assert primary["label"] == "KDIGO 3 期 vs 无 AKI"
    assert primary["measure"] == "OR"
    assert primary["value"] == pytest.approx(3.2051)
    assert primary["display"] == {"value": "3.21", "low": "1.77", "high": "5.80"}
    assert primary["n"] == 812


def test_sensitivity_refits_and_failed_fits_never_become_the_headline():
    primary = {
        "name": "primary.csv",
        "headers": ESTIMATE_HEADERS,
        "rows": [_estimate_row("3", "3.2", "1.7", "5.8", True),
                 _estimate_row("2", "", "", "", False, status="failed")],
    }
    sensitivity = {
        "name": "sensitivity.csv",
        "headers": ESTIMATE_HEADERS,
        "rows": [_estimate_row("3", "9.9", "1.0", "99", True, role="sensitivity")],
    }

    estimates = summarize([sensitivity, primary])

    assert [row["value"] for row in estimates] == [3.2]


@pytest.mark.parametrize(
    "tables",
    [
        # Two primary tables: which one the study reports cannot be told.
        [
            {"name": "a.csv", "headers": ESTIMATE_HEADERS, "rows": [_estimate_row("3", "3.2", "1.7", "5.8", True)]},
            {"name": "b.csv", "headers": ESTIMATE_HEADERS, "rows": [_estimate_row("3", "2.2", "1.1", "4.8", True)]},
        ],
        # An effect scale the reader does not know how to name.
        [{"name": "a.csv", "headers": ESTIMATE_HEADERS,
          "rows": [_estimate_row("3", "3.2", "1.7", "5.8", True, scale="log_odds")]}],
        # No row says it is the primary contrast.
        [{"name": "a.csv", "headers": ESTIMATE_HEADERS, "rows": [_estimate_row("3", "3.2", "1.7", "5.8", False)]}],
        # A table that only happens to have an "estimate" column.
        [{"name": "a.csv", "headers": ["estimate", "ci_low", "ci_high"], "rows": [["3.2", "1.7", "5.8"]]}],
    ],
)
def test_any_doubt_about_the_primary_estimate_returns_nothing(tables):
    assert summarize(tables) == []


def test_the_overview_leads_with_the_registered_estimate_and_readable_counts():
    table = {
        "name": "adjusted_association_estimates.csv",
        "headers": ESTIMATE_HEADERS,
        "rows": [_estimate_row("3", "3.2051", "1.7702", "5.8031", True)],
    }
    html = run_js(f"""
      global.window = {{EU_LANG:'zh'}};
      eval({_read('js/html-escape.js')!r});
      eval({_read('js/screens-guided-pi-result-summary.js')!r});
      eval({_read('js/screens-guided-pi-analysis-report.js')!r});
      const payload = {{
        manuscript_provenance: {{claims: [
          {{source_field: 'n_total', display_value: '48969'}},
          {{source_field: 'overall_outcome.risk_pct', display_value: '13.8271%'}},
        ]}},
        result_tables: {json.dumps({"tables": [table]})},
      }};
      process.stdout.write(JSON.stringify(window.EU_GUIDED_PI_ANALYSIS_REPORT.render(payload)));
    """)

    assert "主要估计：" in html
    assert "OR 3.21（95% CI 1.77–5.80）" in html
    assert "已登记的主要估计为 OR 3.21（95% CI 1.77–5.80）" in html
    assert "当前没有可直接汇总的分组表" not in html
    assert "48,969" in html and "13.83%" in html
