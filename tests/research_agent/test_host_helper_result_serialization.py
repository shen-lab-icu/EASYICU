"""``json.dump`` cannot serialize a host helper's typed result.

fresh17, step ``05_missingness_event_timing_audit_figure``. The figure rendered
completely -- PDF, PNG, SVG, TIFF and source data all written -- and the step
still failed, because the metadata sidecar was zero bytes::

    TypeError: Object of type FigureContract is not JSON serializable

``json.dump`` had opened the file, then raised while serializing.

Two figure steps in that same run prove the outcome is a coin flip rather than
a capability difference: both call the host's ``make_figure_contract`` and both
write the result with ``json.dump`` plus a hand-written ``default=`` hook.
``08``'s hook happened to reach ``to_dict`` and passed; ``05``'s handled
numpy/pandas only and failed.

There is a worse sibling the same check catches. A hook ending in
``return str(value)`` does not raise -- it writes a *valid* JSON file whose
whole content is the object's repr as a string, so the contract reader gets a
string where it expects an object and nothing looks wrong.
"""

from __future__ import annotations

import ast
import json

from easyicu.research_agent.figures.publication import make_figure_contract
from easyicu.research_agent.gates.host_helper_serialization import (
    host_helper_result_serialization_findings,
)

_IMPORT = (
    "import json\n"
    "from easyicu.research_agent.figures.publication import make_figure_contract\n"
)


def _findings(body: str):
    source = _IMPORT + body
    return host_helper_result_serialization_findings(
        ast.parse(source), script_text=source
    )


def test_the_live_failure_is_reported() -> None:
    """No hook at all: exactly the TypeError that emptied the sidecar."""

    findings = _findings(
        "contract = make_figure_contract(figure_id='f')\n"
        "with open('c.json', 'w') as handle:\n"
        "    json.dump(contract, handle)\n"
    )

    assert len(findings) == 1
    assert findings[0].detail["binding"] == "contract"
    assert findings[0].detail["helper"] == "make_figure_contract"
    assert findings[0].detail["returns"] == "FigureContract"


def test_a_hook_that_cannot_reach_the_model_is_reported() -> None:
    """The shape ``05`` actually had: numpy/pandas only."""

    findings = _findings(
        "def json_default(value):\n"
        "    if isinstance(value, complex):\n"
        "        return str(value)\n"
        "    raise TypeError(value)\n"
        "contract = make_figure_contract(figure_id='f')\n"
        "json.dump(contract, open('c.json', 'w'), default=json_default)\n"
    )

    assert len(findings) == 1


def test_a_hook_that_reaches_the_model_is_left_alone() -> None:
    """The shape ``08`` had, which really works.

    Reporting it would block correct code and spend a repair on it -- the cost
    this gate exists to avoid, not to cause.
    """

    findings = _findings(
        "def json_default(value):\n"
        "    if hasattr(value, 'to_dict'):\n"
        "        return value.to_dict()\n"
        "    raise TypeError(value)\n"
        "contract = make_figure_contract(figure_id='f')\n"
        "json.dump(contract, open('c.json', 'w'), default=json_default)\n"
    )

    assert findings == []


def test_the_sanctioned_accessor_is_not_reported() -> None:
    for body in (
        "contract = make_figure_contract(figure_id='f')\n"
        "open('c.json', 'w').write(contract.to_json())\n",
        "contract = make_figure_contract(figure_id='f')\n"
        "json.dump(contract.to_dict(), open('c.json', 'w'))\n",
    ):
        assert _findings(body) == []


def test_an_unrelated_dump_is_not_reported() -> None:
    findings = _findings(
        "contract = make_figure_contract(figure_id='f')\n"
        "summary = {'status': 'ok'}\n"
        "json.dump(summary, open('s.json', 'w'))\n"
    )

    assert findings == []


def test_a_rebound_name_is_not_reported() -> None:
    """Once reassigned from a non-call expression its type is unknown."""

    findings = _findings(
        "contract = make_figure_contract(figure_id='f')\n"
        "contract = {'figure_id': 'f'}\n"
        "json.dump(contract, open('c.json', 'w'))\n"
    )

    assert findings == []


def test_the_check_reads_the_declared_return_type_not_a_helper_name() -> None:
    """A host helper that does not return a model must not be reported.

    The gate resolves the annotation from the host, so a new helper returning a
    typed model is covered the day it is added and nothing is re-listed here.
    """

    source = (
        "import json\n"
        "from easyicu.research_agent.figures.publication import figure_contract_path\n"
        "value = figure_contract_path('x')\n"
        "json.dump(value, open('v.json', 'w'))\n"
    )
    try:
        tree = ast.parse(source)
    except SyntaxError:  # pragma: no cover
        raise
    # Whatever that helper returns, it is not a pydantic model, so nothing is
    # reported; the test still passes if the symbol does not exist at all.
    assert host_helper_result_serialization_findings(tree, script_text=source) == []


# ---------------------------------------------------------------------------
# The runtime facts the check is asserting about.
# ---------------------------------------------------------------------------


def _contract():
    return make_figure_contract(
        figure_id="figure:t",
        core_claim="c" * 12,
        panels=[{"panel_id": "A", "role": "r", "claim": "x" * 12}],
        source_data=["a.csv"],
    )


def test_json_dump_really_does_raise_on_this_type() -> None:
    """If this ever stops raising, the gate is guarding a dead failure mode."""

    import pytest

    with pytest.raises(TypeError):
        json.dumps(_contract())


def test_a_str_fallback_silently_writes_a_string_not_a_contract() -> None:
    """The quieter failure: valid JSON, wrong content, nothing looks wrong."""

    written = json.dumps(_contract(), default=str)

    assert isinstance(json.loads(written), str)


def test_the_sanctioned_accessor_really_produces_the_object() -> None:
    restored = json.loads(_contract().to_json())

    assert isinstance(restored, dict)
    assert restored["figure_id"] == "figure:t"
