from __future__ import annotations

from pathlib import Path
import re

import pytest

from easyicu.research_agent.figures.publication import make_figure_contract
from easyicu.research_agent.reporting.article_display_policy import (
    ARTICLE_DISPLAY_FAILED_CLOSED_RESULT_FORBIDDEN,
    ARTICLE_DISPLAY_ROLE_UNSUPPORTED,
    ArticleDisplayPolicyError,
    ArticleDisplayPolicyRequest,
    decide_article_display,
)


@pytest.mark.parametrize(
    ("variable_label", "role", "expected_placement", "expected_purpose"),
    [
        ("flux_7q", "primary_estimand", "main", "scientific_result"),
        ("ion_zeta", "baseline_context", "main", "context"),
        ("split_sigma", "validation_design", "main", "context"),
        ("marker_kappa", "robustness", "main", "scientific_result"),
        ("signal_omega", "data_quality", "supplementary", "audit"),
    ],
)
def test_unseen_labels_cannot_change_typed_display_policy(
    variable_label: str,
    role: str,
    expected_placement: str,
    expected_purpose: str,
) -> None:
    decision = decide_article_display(ArticleDisplayPolicyRequest(article_role=role))

    assert variable_label not in decision.model_dump_json()
    assert decision.placement == expected_placement
    assert decision.display_purpose == expected_purpose


def test_measurement_is_main_only_when_it_is_the_typed_research_focus() -> None:
    decision = decide_article_display(
        ArticleDisplayPolicyRequest(
            article_role="data_quality",
            analysis_type="data_quality_audit",
            central_to_question=True,
        )
    )

    assert decision.placement == "main"
    assert decision.display_purpose == "scientific_result"
    assert decision.reason_code == "MEASUREMENT_PROCESS_IS_RESEARCH_RESULT"


def test_failed_closed_result_is_rejected_but_terminal_diagnostic_is_main() -> None:
    with pytest.raises(ArticleDisplayPolicyError) as error:
        decide_article_display(
            ArticleDisplayPolicyRequest(
                article_role="primary_estimand",
                scientific_status="failed_closed",
            )
        )
    assert error.value.reason_code == ARTICLE_DISPLAY_FAILED_CLOSED_RESULT_FORBIDDEN

    diagnostic = decide_article_display(
        ArticleDisplayPolicyRequest(
            article_role="diagnostics",
            scientific_status="failed_closed",
            terminal_diagnostic=True,
            central_to_question=True,
        )
    )
    assert diagnostic.placement == "main"
    assert diagnostic.display_purpose == "diagnostic"


def test_unknown_role_fails_at_the_policy_owner() -> None:
    with pytest.raises(ArticleDisplayPolicyError) as error:
        decide_article_display(
            ArticleDisplayPolicyRequest(article_role="custom_answer_panel")
        )
    assert error.value.reason_code == ARTICLE_DISPLAY_ROLE_UNSUPPORTED


def test_figure_contract_receives_policy_receipt_without_case_routing() -> None:
    contract = make_figure_contract(
        figure_id="figure:unseen_task_delta",
        core_claim="A typed result and its uncertainty are displayed.",
        panels=[
            {
                "panel_id": "a",
                "title": "Registered contrast",
                "role": "primary_estimand",
                "article_role": "primary_estimand",
                "chart_type": "forest",
                "claim": "The registered contrast is shown.",
                "evidence_ids": ["evidence_random_42"],
            }
        ],
        source_data=["random_source.csv"],
    )

    metadata = contract.panels[0].metadata
    assert metadata["placement"] == "main"
    assert metadata["display_purpose"] == "scientific_result"
    assert metadata["display_policy_reason_code"] == "SCIENTIFIC_RESULT_DISPLAY"


def test_policy_source_contains_no_development_case_or_variable_names() -> None:
    source = (
        (
            Path(__file__).parents[3]
            / "src/easyicu/research_agent/reporting/article_display_policy.py"
        )
        .read_text(encoding="utf-8")
        .casefold()
    )

    forbidden = (
        "e1_",
        "e2_",
        "h2_",
        "h3_",
        "sepsis",
        "lactate",
        "bilirubin",
        "kdigo",
        "vasopressor",
    )
    assert not [token for token in forbidden if token in source]


def test_all_statically_declared_runtime_roles_are_owned_by_the_policy() -> None:
    source_root = Path(__file__).parents[3] / "src/easyicu/research_agent"
    roles: set[str] = set()
    patterns = (
        re.compile(r'article_role\s*=\s*["\']([a-z][a-z0-9_]*)["\']'),
        re.compile(r'["\']article_role["\']\s*:\s*["\']([a-z][a-z0-9_]*)["\']'),
    )
    for path in source_root.rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        for pattern in patterns:
            roles.update(pattern.findall(source))

    for role in sorted(roles):
        decide_article_display(ArticleDisplayPolicyRequest(article_role=role))
