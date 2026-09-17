"""Copilot workflow fixture sink (E-P2-12).

New convention: shared StudyContext fixtures live here as pytest fixtures.
Importing them via ``from test_* import`` is banned for new code (see
``tests/governance/test_test_organization.py::test_no_cross_test_module_imports``);
the legacy ``research_workflow_fixtures`` module stays as a grandfathered
implementation detail and is re-exported below so existing importers keep
working without a 50-file refactor.
"""

from __future__ import annotations

import pytest

from tests.webserver.copilot import research_workflow_fixtures as _legacy


@pytest.fixture
def workflow_complete_study() -> dict:
    """A bounded, confirmed StudyContext dict for workflow contract tests."""

    return _legacy.complete_study()


@pytest.fixture
def workflow_confirmed_cohort_decision():
    """The ``confirmed_cohort_decision`` helper as a fixture factory."""

    return _legacy.confirmed_cohort_decision


__all__ = ["workflow_complete_study", "workflow_confirmed_cohort_decision"]
