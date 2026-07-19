"""Execute-phase collaborator identity lock (Codex-ordered, bundle2).

The contract / figure-contract subsystem was extracted from the execute phase
into ``contract_gate`` (read-only findings gates) and
``figure_contract_preparation`` (writes-files shaping/canonicalization).
Everything the submodule publishes in ``__all__`` must be the same object used
by the canonical execute-phase consumer so monkeypatch targets cannot split.

This is the collaborator identity test: it walks each sub-module's ``__all__``
and asserts the execution phase uses every name with identical identity.
It fails closed if a future symbol is added to a sub-module's ``__all__`` but not
forwarded (the exact gap Codex caught: 3 helpers + 3 constants dropped their old
path after the figure-shaping move).
"""

from __future__ import annotations

import pytest

from easyicu.research_agent.execution import phase as execution_phase
from easyicu.research_agent.execution import (
    figure_preparation as figure_contract_preparation,
)
from easyicu.research_agent.execution import (
    publication_figure as publication_figure_execution,
)
from easyicu.research_agent.gates import contract as contract_gate

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
def test_execution_phase_uses_every_collaborator_with_identity(module, name):
    assert hasattr(execution_phase, name), (
        f"{name} is in {module.__name__}.__all__ but is NOT re-exported by "
        f"the canonical execution phase."
    )
    assert getattr(execution_phase, name) is getattr(module, name), (
        f"execution_phase.{name} is not the same object as {module.__name__}."
        f"{name} — the re-export must preserve identity so monkeypatch and "
        f"isinstance/identity checks keep working."
    )


def test_compat_table_is_nonempty_and_covers_both_modules():
    # Guards against the test silently passing if __all__ ever disappears.
    covered = {m.__name__.rsplit(".", 1)[-1] for m, _ in _compat_table()}
    assert covered == {
        "contract",
        "figure_preparation",
        "publication_figure",
    }
    assert len(contract_gate.__all__) >= 14
    assert len(figure_contract_preparation.__all__) == 8
    assert len(publication_figure_execution.__all__) == 6
