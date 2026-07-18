"""Old-path re-export compatibility lock (Codex-ordered, bundle2).

The contract / figure-contract subsystem was extracted from ``pipeline_execute``
into ``contract_gate`` (read-only findings gates) and
``figure_contract_preparation`` (writes-files shaping/canonicalization). The
promise of every such extraction is old-path back-compat: everything the sub
module publishes in ``__all__`` must remain importable from ``pipeline_execute``
AS THE SAME OBJECT (identity), so existing ``from ...pipeline_execute import X``
call sites and monkeypatch targets keep resolving.

This is the "compat table identity test": it walks each sub-module's ``__all__``
and asserts ``pipeline_execute`` re-exports every name with identical identity.
It fails closed if a future symbol is added to a sub-module's ``__all__`` but not
forwarded (the exact gap Codex caught: 3 helpers + 3 constants dropped their old
path after the figure-shaping move).
"""

from __future__ import annotations

import pytest

from easyicu.research_agent import (
    contract_gate,
    figure_contract_preparation,
    pipeline_execute,
    publication_figure_execution,
)

_COMPAT_MODULES = (
    contract_gate,
    figure_contract_preparation,
    publication_figure_execution,
)


def _compat_table():
    for module in _COMPAT_MODULES:
        for name in module.__all__:
            yield module, name


@pytest.mark.parametrize(
    "module,name",
    [(m, n) for m, n in _compat_table()],
    ids=[f"{m.__name__.rsplit('.', 1)[-1]}.{n}" for m, n in _compat_table()],
)
def test_pipeline_execute_reexports_every_all_symbol_with_identity(module, name):
    assert hasattr(pipeline_execute, name), (
        f"{name} is in {module.__name__}.__all__ but is NOT re-exported by "
        f"pipeline_execute — old import path from ...pipeline_execute import "
        f"{name} would break."
    )
    assert getattr(pipeline_execute, name) is getattr(module, name), (
        f"pipeline_execute.{name} is not the same object as {module.__name__}."
        f"{name} — the re-export must preserve identity so monkeypatch and "
        f"isinstance/identity checks keep working."
    )


def test_compat_table_is_nonempty_and_covers_both_modules():
    # Guards against the test silently passing if __all__ ever disappears.
    covered = {m.__name__.rsplit(".", 1)[-1] for m, _ in _compat_table()}
    assert covered == {
        "contract_gate",
        "figure_contract_preparation",
        "publication_figure_execution",
    }
    assert len(contract_gate.__all__) >= 14
    assert len(figure_contract_preparation.__all__) == 8
    assert len(publication_figure_execution.__all__) == 6
