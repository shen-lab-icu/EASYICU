"""AST structural contract for the execute-phase GateEvaluator seam.

Batch 1c follow-up (Codex-ordered): converge the brittle
``inspect.getsource(...).index("literal")`` ordering/purity anchors — the ones
that broke three or four times as gate implementations moved this session — into
ONE AST-based contract suite. Moving a gate implementation should now break at
most this file's semantic assertions, not a scatter of fragile string offsets.

It locks three GateEvaluator invariants:
  1. the gate pipeline runs in a fixed ORDER inside run_execute_phase;
  2. the gate components reference NONE of a fixed deny-list of orchestration
     primitives (a boundary SENTINEL — not a complete "no side effects / pure
     component" proof; it catches the specific control-flow/authority leaks we
     care about);
  3. the orchestrator KEEPS the authority + control-flow primitives (they did not
     leak into the components).

The reusable AST helpers (``gate_call_order`` / ``component_identifiers`` /
``has_continue``) are imported by the visual-gate governance tests so their
ordering checks are AST-based too, not string-index based.
"""

from __future__ import annotations

import ast
import inspect
import textwrap

from easyicu.research_agent.execution import phase as pe

# --- reusable AST helpers (shared with test_visual_repair_governance) ---


def _module(func) -> ast.Module:
    return ast.parse(textwrap.dedent(inspect.getsource(func)))


def _call_name(node: ast.AST):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def gate_call_order(func, names):
    """First source line at which each name in ``names`` is CALLED inside ``func``.

    ``ast.walk`` yields nodes without a source-order guarantee, so we keep the
    minimum lineno per name — the first occurrence, which is what an ordering
    contract cares about. Returns ``{name: lineno}`` (missing names absent).
    """
    wanted = set(names)
    first: dict[str, int] = {}
    for node in ast.walk(_module(func)):
        if isinstance(node, ast.Call):
            name = _call_name(node.func)
            if name in wanted:
                first[name] = min(first.get(name, node.lineno), node.lineno)
    return first


def component_identifiers(func) -> set[str]:
    """Every Name.id / Attribute.attr referenced in ``func`` (docstrings excluded
    — they are ast.Constant, not identifiers)."""
    ids: set[str] = set()
    for node in ast.walk(_module(func)):
        if isinstance(node, ast.Name):
            ids.add(node.id)
        elif isinstance(node, ast.Attribute):
            ids.add(node.attr)
    return ids


def has_continue(func) -> bool:
    return any(isinstance(n, ast.Continue) for n in ast.walk(_module(func)))


# --- the GateEvaluator seam under contract ---

GATE_COMPONENTS = (
    "collect_visual_gate_result",
    "decide_visual_repair",
    "_step_deterministic_contract_findings",
    "_post_canonicalization_figure_findings",
)

# A fixed deny-list of orchestration primitives a gate component must never
# reference. This is a boundary SENTINEL over known names — NOT a proof of
# purity or of "no side effects"; it catches the specific control-flow/authority
# leaks that would mean a component started driving the run instead of reporting.
FORBIDDEN_ORCHESTRATION = {
    "_consume_llm_repair_budget",
    "_repair_with_capsule",
    "shared_lock",
    "_append_terminal_step_record",
    "_flush_partial_manifest",
    "emit_progress",
    "_clear_output_dir",
    "_deterministic_fallback_code",
    "step_record",
}


def test_gate_pipeline_runs_in_canonical_order():
    order = gate_call_order(
        pe.run_execute_phase,
        {
            "collect_visual_gate_result",
            "_step_deterministic_contract_findings",
            "_install_figure_contract_source_data_canonicalization",
            "_post_canonicalization_figure_findings",
        },
    )
    assert set(order) == {
        "collect_visual_gate_result",
        "_step_deterministic_contract_findings",
        "_install_figure_contract_source_data_canonicalization",
        "_post_canonicalization_figure_findings",
    }, f"a gate stage is missing from run_execute_phase: {sorted(order)}"
    # Visual QA -> shared deterministic contract gate -> figure-contract
    # canonicalization repair -> post-canonicalization figure audits.
    assert (
        order["collect_visual_gate_result"]
        < order["_step_deterministic_contract_findings"]
        < order["_install_figure_contract_source_data_canonicalization"]
        < order["_post_canonicalization_figure_findings"]
    )


def test_gate_components_reference_no_orchestration_primitives():
    # Boundary sentinel, NOT a full side-effect/purity proof: each gate component
    # must reference none of the fixed deny-list of orchestration primitives and
    # must not drive loop control flow.
    for name in GATE_COMPONENTS:
        func = getattr(pe, name)
        leaked = component_identifiers(func) & FORBIDDEN_ORCHESTRATION
        assert not leaked, f"{name} leaked orchestration symbols {sorted(leaked)}"
        assert not has_continue(func), f"{name} must not drive loop control flow"


def test_orchestrator_retains_authority_and_control_flow():
    # The primitives the components must NOT own DO live in the orchestrator
    # (run_execute_phase source includes the nested _execute_one_step). This is
    # the other half of the boundary: the gate reports, the host decides.
    orchestrator_ids = component_identifiers(pe.run_execute_phase)
    for symbol in (
        "_consume_llm_repair_budget",
        "_repair_with_capsule",
        "shared_lock",
        "emit_progress",
    ):
        assert symbol in orchestrator_ids, f"orchestrator lost {symbol}"
