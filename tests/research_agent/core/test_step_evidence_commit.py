"""Behavior + structural-contract tests for StepEvidenceCommit (batch 1c).

StepEvidenceCommit encapsulates the validated-step commit boundary that used to
be inline in ``_execute_one_step``: numeric/derived claims and result aliases
become current as ONE durable generation inside the store's existing
``success_publication_transaction``, or roll back together.

It is NOT a second evidence authority: it opens the store's existing
transaction, delegates alias promotion to EvidenceRegistrar, does no
validation/gate, scans no ``current``, and owns no control flow. The
``status == "ok"`` guard, the ``step_record`` mutation, and the failure finding
stay in the caller.

Written before the implementation (Codex discipline: tests first). The
structural-contract test is AST-based on purpose — it is the seed of the
convergence away from brittle ``inspect.getsource(...).index("literal")``
anchors that break every time a function moves.
"""

from __future__ import annotations

import ast
import inspect
from contextlib import contextmanager

import pytest


class _FakeStore:
    """Minimal EvidenceStore surface with an observable transaction."""

    def __init__(self, *, promote_result="PROMO", records=(), aliases=None):
        self.calls: list[str] = []
        self._records = list(records)
        self._aliases = dict(aliases or {})
        self._promote_result = promote_result
        self.publish_args: list[dict] = []

    # --- promotion-store surface ---
    def records(self):
        return list(self._records)

    def aliases(self):
        return dict(self._aliases)

    def publish_step_success_aliases(
        self, bindings, *, step_id, suppressed_basename_evidence_ids=()
    ):
        self.calls.append("publish")
        self.publish_args.append(
            {
                "bindings": dict(bindings),
                "step_id": step_id,
                "suppressed": list(suppressed_basename_evidence_ids),
            }
        )
        # Echo an empty published map — StepEvidenceCommit does not interpret it.
        return {}

    @contextmanager
    def success_publication_transaction(self):
        self.calls.append("txn_enter")
        try:
            yield
        except BaseException:
            self.calls.append("txn_rollback")
            raise
        else:
            self.calls.append("txn_commit")


def _commit(store):
    from easyicu.research_agent.authority.registration import StepEvidenceCommit

    return StepEvidenceCommit(store)


# =====================================================================
# Behavior
# =====================================================================


def test_numeric_and_promote_run_inside_one_transaction_in_order():
    store = _FakeStore()
    thunk_calls = []

    def register_numeric_claims():
        store.calls.append("numeric")
        thunk_calls.append(1)

    _commit(store).commit_validated_step(
        step_id="01_summary",
        pending_aliases={},
        allowed_evidence_ids=[],
        register_numeric_claims=register_numeric_claims,
    )

    # Numeric registration and alias promotion both happen strictly between the
    # transaction enter and its commit — the single-generation guarantee.
    assert store.calls == ["txn_enter", "numeric", "publish", "txn_commit"]
    assert thunk_calls == [1]


def test_commit_returns_promotion_result():
    store = _FakeStore()
    result = _commit(store).commit_validated_step(
        step_id="01_summary",
        pending_aliases={},
        allowed_evidence_ids=[],
        register_numeric_claims=lambda: None,
    )
    from easyicu.research_agent.authority.registration import EvidencePromotionResult

    assert isinstance(result, EvidencePromotionResult)


def test_numeric_thunk_failure_rolls_back_and_skips_promote():
    store = _FakeStore()

    def boom():
        store.calls.append("numeric")
        raise ValueError("numeric registration exploded")

    with pytest.raises(ValueError, match="numeric registration exploded"):
        _commit(store).commit_validated_step(
            step_id="01_summary",
            pending_aliases={},
            allowed_evidence_ids=[],
            register_numeric_claims=boom,
        )
    # Promote never runs, and the transaction exits via the rollback path so the
    # store restores its entry state — numeric claims cannot go current alone.
    assert store.calls == ["txn_enter", "numeric", "txn_rollback"]
    assert "publish" not in store.calls


def test_promote_failure_propagates_through_transaction_rollback():
    store = _FakeStore()

    def exploding_publish(*_args, **_kwargs):
        store.calls.append("publish")
        raise OSError("alias store unavailable")

    store.publish_step_success_aliases = exploding_publish

    with pytest.raises(OSError, match="alias store unavailable"):
        _commit(store).commit_validated_step(
            step_id="01_summary",
            pending_aliases={"ev_1": ["01_summary_table"]},
            allowed_evidence_ids=["ev_1"],
            register_numeric_claims=lambda: None,
        )
    assert store.calls == ["txn_enter", "publish", "txn_rollback"]


def test_pending_alias_outside_attempt_is_rejected_before_any_commit():
    # The attempt-bound guard lives in EvidenceRegistrar; lock that the commit
    # boundary surfaces it (a pending alias for evidence not in this attempt is a
    # ValueError, raised before promotion mutates anything).
    store = _FakeStore()
    with pytest.raises(ValueError, match="outside the current attempt"):
        _commit(store).commit_validated_step(
            step_id="01_summary",
            pending_aliases={"stranger_ev": ["x"]},
            allowed_evidence_ids=["ev_1"],
            register_numeric_claims=lambda: None,
        )
    assert "publish" not in store.calls


# =====================================================================
# Structural contract (AST — the convergence away from brittle anchors)
# =====================================================================


def _commit_method_ast():
    import textwrap

    from easyicu.research_agent.authority import registration as evidence_registration

    src = inspect.getsource(
        evidence_registration.StepEvidenceCommit.commit_validated_step
    )
    return ast.parse(textwrap.dedent(src))


def test_numeric_and_promote_are_lexically_inside_the_transaction_with_block():
    """AST contract: both the numeric thunk call and promote_validated_step are
    in the body of the ``with success_publication_transaction()`` block."""
    tree = _commit_method_ast()

    with_nodes = [n for n in ast.walk(tree) if isinstance(n, ast.With)]
    txn_withs = [
        w
        for w in with_nodes
        if any(
            isinstance(item.context_expr, ast.Call)
            and _attr_name(item.context_expr.func) == "success_publication_transaction"
            for item in w.items
        )
    ]
    assert len(txn_withs) == 1, "exactly one success_publication_transaction block"
    body_src = "\n".join(ast.dump(n) for n in ast.walk(txn_withs[0]))
    assert "promote_validated_step" in body_src
    # the injected numeric thunk is invoked inside the block
    called = {
        _attr_name(n.func) for n in ast.walk(txn_withs[0]) if isinstance(n, ast.Call)
    }
    assert "promote_validated_step" in called
    assert "register_numeric_claims" in called


def test_commit_boundary_is_not_a_second_authority_or_a_gate():
    """AST contract: the commit boundary neither validates/gates nor touches
    orchestration control-flow state.

    Checks *identifiers* (ast.Name / ast.Attribute), not raw source text, so the
    docstring may still explain the boundary without tripping the guard — this is
    the AST-based convergence away from brittle string-index anchors.
    """
    import textwrap

    from easyicu.research_agent.authority import registration as evidence_registration

    src = textwrap.dedent(inspect.getsource(evidence_registration.StepEvidenceCommit))
    tree = ast.parse(src)
    identifiers = {
        node.attr if isinstance(node, ast.Attribute) else node.id
        for node in ast.walk(tree)
        if isinstance(node, (ast.Name, ast.Attribute))
    }
    forbidden = {
        "step_record",
        "_evaluate_final_deterministic_gates",
        "CriticAgent",
        "register_file",
        "load_current_evidence_snapshot",
    }
    leaked = identifiers & forbidden
    assert not leaked, f"commit boundary must not reference {sorted(leaked)}"


def _attr_name(node) -> str:
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Name):
        return node.id
    return ""
