"""Completeness checks for current-case runtime plan compilation."""

from types import SimpleNamespace
from typing import get_args

import pytest

from easyicu.research_agent.authority.current_case_scientific_runtime import (
    CurrentCaseScientificRuntimeAuthority,
)
from easyicu.research_agent.orchestration import scientific_runtime
from easyicu.research_agent.schema import AnalysisPlan


def _authority_classes() -> tuple[type, ...]:
    union = get_args(CurrentCaseScientificRuntimeAuthority)[0]
    return get_args(union)


def _authority_kind(authority_class: type) -> str:
    annotation = authority_class.model_fields["authority_kind"].annotation
    return get_args(annotation)[0]


def test_every_current_case_authority_has_a_plan_compiler() -> None:
    authority_kinds = {_authority_kind(cls) for cls in _authority_classes()}

    assert set(scientific_runtime._CURRENT_CASE_PLAN_COMPILERS) == authority_kinds


def test_every_development_projection_has_a_registered_compiler() -> None:
    projection_kinds = {
        _authority_kind(cls)
        for cls in _authority_classes()
        if hasattr(cls, "development_execution_only_plan")
    }

    assert set(
        scientific_runtime._CURRENT_CASE_DEVELOPMENT_PLAN_COMPILERS
    ) == projection_kinds


def test_unregistered_current_case_authority_fails_closed() -> None:
    unknown = SimpleNamespace(authority_kind="new_unregistered_authority")
    plan = AnalysisPlan(research_question="Registry coverage check", steps=[])

    with pytest.raises(TypeError, match="has no plan compiler"):
        scientific_runtime._compile_current_case_plan(unknown, plan)
