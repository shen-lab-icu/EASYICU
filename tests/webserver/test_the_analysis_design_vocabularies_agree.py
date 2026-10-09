"""The three lists of what an analysis design may state agree.

The StudyContext owner, the dependence owner that binds plans in the
execution kernel, and the Copilot tool schema each list the analysis units,
variance estimators and cluster units a study may state.  Until one owner
holds them, a value one list has and another lacks fails here, whichever
list it is.
"""

from __future__ import annotations

import re
import typing
from pathlib import Path

import pytest

from easyicu.research_agent.planning import dependence_authority
from easyicu.webserver import study_contexts

_SCHEMA = (
    Path(study_contexts.__file__).parent / "pi_copilot" / "node_app" / "src" / "main.mjs"
)
_STUDY_CONTEXT = {
    "analysis_unit": study_contexts._ANALYSIS_UNITS,
    "variance_estimator": study_contexts._VARIANCE_ESTIMATORS,
    "cluster_unit": study_contexts._CLUSTER_UNITS,
}


def _schema_values(field: str) -> frozenset[str]:
    source = _SCHEMA.read_text(encoding="utf-8")
    design = source.split("const analysisDesign = Type.Object({", 1)[1]
    design = design.split("}, { additionalProperties: false });", 1)[0]
    union = re.search(
        rf"\n    {field}: (?:Type\.Optional\()?Type\.Union\(\[(.*?)\]\)", design, re.S
    )
    assert union is not None, field
    return frozenset(re.findall(r'Type\.Literal\("([a-z_]+)"\)', union.group(1)))


def _literal_values(annotation: object) -> frozenset[str]:
    if typing.get_origin(annotation) is typing.Literal:
        return frozenset(typing.get_args(annotation))
    return frozenset(
        value for arg in typing.get_args(annotation) for value in _literal_values(arg)
    )


def _kernel_values(field: str) -> frozenset[str]:
    model = dependence_authority._AnalysisDesign
    return _literal_values(model.model_fields[field].annotation)


@pytest.mark.parametrize("field", sorted(_STUDY_CONTEXT))
def test_each_owner_lists_the_same_values(field: str) -> None:
    owner = frozenset(_STUDY_CONTEXT[field])

    assert owner
    assert _kernel_values(field) == owner
    assert _schema_values(field) == owner
