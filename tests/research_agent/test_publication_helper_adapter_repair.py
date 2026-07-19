from __future__ import annotations

import ast
from pathlib import Path

import pytest

from easyicu.research_agent.gates.preflight import audit_mechanical_code_contracts
from easyicu.research_agent.repairs.source import deterministic_concept_audit_repair
from easyicu.research_agent.legacy_code_migrations import (
    migrate_legacy_publication_helper_adapter_v1,
)
from easyicu.research_agent.repairs.reasons import repair_reason_for_finding
from easyicu.research_agent.repair_registry import RepairClass, repair_metadata_for

_LEGACY_SCRIPT = """
import inspect
import pandas as pd
from easyicu.research_agent.publication_figures import save_publication_figure

def strict_numeric(series):
    original = series.copy()
    coerced = pd.to_numeric(series, errors="coerce")
    newly_invalid = int((original.notna() & coerced.isna()).sum())
    if newly_invalid > 0:
        raise ValueError("numeric coercion invalidated observed values")
    return coerced

def export_bundle(fig, contract, out_dir, stem):
    # Use the EasyICU publication helper when its installed signature is known.
    signature = inspect.signature(save_publication_figure)
    kwargs = {}
    positional = []
    for name, parameter in signature.parameters.items():
        if name in {"fig", "figure"}:
            kwargs[name] = fig
        elif name in {"contract", "figure_contract"}:
            kwargs[name] = contract
        elif name in {"out_dir", "output_dir", "directory"}:
            kwargs[name] = out_dir
        elif name in {"stem", "figure_stem", "filename_stem"}:
            kwargs[name] = stem
        elif parameter.default is inspect.Parameter.empty and name not in kwargs:
            if not positional:
                positional.append(fig)
            elif len(positional) == 1:
                positional.append(contract)
            elif len(positional) == 2:
                positional.append(out_dir)
            elif len(positional) == 3:
                positional.append(stem)

    try:
        save_publication_figure(*positional, **kwargs)
    except TypeError:
        # The explicit exports below remain the same source figure and ensure
        # the requested files exist even if helper signatures differ by version.
        pass
    fig.savefig(out_dir / f"{stem}.png")
    return "manual-export-preserved"
"""


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


def _repair(script: str, ra):
    findings = _findings(script, ra)
    return deterministic_concept_audit_repair(
        script,
        [finding.message for finding in findings],
        repair_reasons=[repair_reason_for_finding(finding) for finding in findings],
        repair_findings=findings,
    )


def test_preflight_emits_typed_host_helper_signature_reason(ra) -> None:
    host_findings = [
        finding
        for finding in _findings(_LEGACY_SCRIPT, ra)
        if (finding.detail or {}).get("reason") == "host_helper_runtime_introspection"
    ]

    assert len(host_findings) == 1
    assert repair_reason_for_finding(host_findings[0]).value == (
        "INVALID_HELPER_SIGNATURE"
    )


def test_exact_legacy_adapter_migrates_without_changing_manual_export(ra) -> None:
    repaired, names = _repair(_LEGACY_SCRIPT, ra)

    assert names == ["legacy_publication_helper_adapter_v1"]
    assert "import inspect" not in repaired
    assert "inspect.signature" not in repaired
    assert "manual-export-preserved" in repaired
    assert "fig.savefig" in repaired
    assert not {
        (finding.detail or {}).get("reason") for finding in _findings(repaired, ra)
    } & {"host_helper_runtime_introspection", "lossy_numeric_coercion"}

    calls = [
        node
        for node in ast.walk(ast.parse(repaired))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "save_publication_figure"
    ]
    assert len(calls) == 1
    assert calls[0].args == []
    assert {keyword.arg: keyword.value.id for keyword in calls[0].keywords} == {
        "fig": "fig",
        "out_dir": "out_dir",
        "stem": "stem",
        "contract": "contract",
    }


def test_legacy_migration_is_idempotent_and_not_prose_routed(ra) -> None:
    repaired, _ = _repair(_LEGACY_SCRIPT, ra)
    again, names = _repair(repaired, ra)
    prose_only, prose_names = deterministic_concept_audit_repair(
        _LEGACY_SCRIPT,
        ["host_helper_runtime_introspection"],
    )

    assert again == repaired
    assert names == []
    assert prose_only == _LEGACY_SCRIPT
    assert prose_names == []


def test_unrelated_helper_finding_cannot_rewrite_inert_legacy_text(ra) -> None:
    legacy_block = _LEGACY_SCRIPT.split(
        "def export_bundle(fig, contract, out_dir, stem):\n", 1
    )[1].split('    fig.savefig(out_dir / f"{stem}.png")', 1)[0]
    script = f'''\
import inspect
from easyicu.research_agent.publication_figures import save_publication_figure

LEGACY_TEXT = """{legacy_block}"""

def local_helper(required):
    return required

local_helper()
'''
    findings = _findings(script, ra)
    repaired, names = deterministic_concept_audit_repair(
        script,
        [finding.message for finding in findings],
        repair_reasons=[repair_reason_for_finding(finding) for finding in findings],
        repair_findings=findings,
    )

    assert {(finding.detail or {}).get("reason") for finding in findings} == {
        "invalid_local_helper_call"
    }
    assert migrate_legacy_publication_helper_adapter_v1(script) == script
    assert repaired == script
    assert names == []


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
import inspect
import easyicu.research_agent.publication_figures as publication_helpers
signature = inspect.signature(publication_helpers.save_publication_figure)
""",
        """
from easyicu.research_agent import publication_figures as publication_helpers
signature = publication_helpers.save_publication_figure.__signature__
""",
    ],
)
def test_preflight_blocks_straightforward_introspection_aliases(
    script: str, ra
) -> None:
    reasons = {
        (finding.detail or {}).get("reason") for finding in _findings(script, ra)
    }

    assert "host_helper_runtime_introspection" in reasons
    repaired, names = _repair(script, ra)
    assert repaired == script
    assert names == []


@pytest.mark.parametrize(
    "old,new",
    [
        ("import inspect", "import inspect; SENTINEL = 41"),
        ("import inspect", "import inspect as reflection"),
        ('{"fig", "figure"}', '{"fig", "figure", "canvas"}'),
        ("except TypeError:", "except Exception:"),
        (
            '    fig.savefig(out_dir / f"{stem}.png")',
            '    globals()\n    fig.savefig(out_dir / f"{stem}.png")',
        ),
        (
            '    fig.savefig(out_dir / f"{stem}.png")',
            "    save_publication_figure = fig.savefig\n"
            '    fig.savefig(out_dir / f"{stem}.png")',
        ),
    ],
)
def test_legacy_migration_rejects_any_nonexact_or_ambiguous_shape(
    ra, old: str, new: str
) -> None:
    variant = _LEGACY_SCRIPT.replace(old, new, 1)
    repaired, names = _repair(variant, ra)

    assert repaired == variant
    assert names == []


def test_legacy_migration_handles_import_after_function(ra) -> None:
    variant = _LEGACY_SCRIPT.replace("import inspect\n", "") + "\nimport inspect\n"

    repaired, names = _repair(variant, ra)

    assert names == ["legacy_publication_helper_adapter_v1"]
    assert "import inspect" not in repaired
    ast.parse(repaired)


def test_custom_local_names_do_not_claim_host_authority(ra) -> None:
    custom = """
def run(inspect, save_publication_figure):
    return inspect.signature(save_publication_figure)
"""

    assert not [
        finding
        for finding in _findings(custom, ra)
        if (finding.detail or {}).get("reason") == "host_helper_runtime_introspection"
    ]


def test_migration_is_registered_as_case_neutral_syntax() -> None:
    metadata = repair_metadata_for("legacy_publication_helper_adapter_v1")

    assert metadata.repair_class is RepairClass.SYNTACTIC
    assert metadata.introduces_numbers is False
    assert metadata.requires_disclosure is False


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
