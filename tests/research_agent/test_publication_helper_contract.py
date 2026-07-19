from __future__ import annotations

from pathlib import Path

import pytest

from easyicu.research_agent.gates.preflight import audit_mechanical_code_contracts
from easyicu.research_agent.repairs.reasons import repair_reason_for_finding


def _step(ra):
    return ra.AnalysisStep(
        step_id="figure_step",
        intent="Render a publication figure from declared source products.",
        inputs=["table:declared_source"],
        expected_outputs=["publication_figure"],
        method="publication_figure",
    )


def _findings(script: str, ra):
    return audit_mechanical_code_contracts(script, _step(ra))


def test_preflight_emits_typed_host_helper_signature_reason(ra) -> None:
    script = """
import inspect
from easyicu.research_agent.publication_figures import save_publication_figure
signature = inspect.signature(save_publication_figure)
"""

    host_findings = [
        finding
        for finding in _findings(script, ra)
        if (finding.detail or {}).get("reason")
        == "host_helper_runtime_introspection"
    ]

    assert len(host_findings) == 1
    assert host_findings[0].severity == "error"
    assert repair_reason_for_finding(host_findings[0]).value == (
        "INVALID_HELPER_SIGNATURE"
    )


@pytest.mark.parametrize(
    "script",
    [
        """
import inspect as ins
from easyicu.research_agent.publication_figures import save_publication_figure
signature = ins.signature(save_publication_figure)
""",
        """
from inspect import signature as inspect_signature
from easyicu.research_agent.publication_figures import save_publication_figure as save_fig
signature = inspect_signature(save_fig)
""",
        """
import easyicu.research_agent.publication_figures as publication_helpers
signature = publication_helpers.save_publication_figure.__signature__
""",
        """
from easyicu.research_agent import publication_figures as publication_helpers
signature = publication_helpers.save_publication_figure.__signature__
""",
    ],
)
def test_preflight_blocks_host_helper_introspection_aliases(script: str, ra) -> None:
    findings = _findings(script, ra)
    reasons = {(finding.detail or {}).get("reason") for finding in findings}

    assert "host_helper_runtime_introspection" in reasons
    assert all(
        finding.severity == "error"
        for finding in findings
        if (finding.detail or {}).get("reason")
        == "host_helper_runtime_introspection"
    )


def test_custom_local_names_do_not_claim_host_authority(ra) -> None:
    custom = """
def run(inspect, save_publication_figure):
    return inspect.signature(save_publication_figure)
"""

    assert not [
        finding
        for finding in _findings(custom, ra)
        if (finding.detail or {}).get("reason")
        == "host_helper_runtime_introspection"
    ]


def test_stable_direct_helper_api_writes_declared_bundle(tmp_path) -> None:
    plt = pytest.importorskip("matplotlib.pyplot")
    from easyicu.research_agent.publication_figures import (
        make_figure_contract,
        save_publication_figure,
    )

    fig, ax = plt.subplots(figsize=(2.5, 1.8))
    ax.plot([0, 1], [0, 1])
    contract = make_figure_contract(
        figure_id="stable_host_api",
        core_claim="The declared line is rendered from the supplied figure.",
        panels=[
            {
                "panel_id": "a",
                "title": "Line",
                "role": "overview",
                "claim": "The declared line is present.",
            }
        ],
        export_formats=["png", "svg"],
    )
    paths = save_publication_figure(
        fig=fig,
        out_dir=tmp_path,
        stem="stable_host_api",
        contract=contract,
    )
    plt.close(fig)

    assert {"png", "svg", "contract"} <= set(paths)


def test_coder_contract_requires_direct_host_api() -> None:
    prompt = (
        Path(__file__).parents[2]
        / "src/easyicu/research_agent/providers/prompts/v1/coder.txt"
    ).read_text(encoding="utf-8")

    assert "save_publication_figure(fig=fig, out_dir=out_dir, stem=stem," in prompt
    assert "`inspect.signature`" in prompt
    assert "EasyICU owns helper-version" in prompt
