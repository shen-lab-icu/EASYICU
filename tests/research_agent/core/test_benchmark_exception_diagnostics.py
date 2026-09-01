"""Benchmark item failures retain safe structured-retry evidence."""

from __future__ import annotations


def test_benchmark_projects_attempts_without_response_or_error_text():
    import tools.run_research_agent_bench as benchmark
    from easyicu.research_agent.providers.structured_retry import (
        StructuredAttempt,
        StructuredResponseFailure,
    )

    failure = StructuredResponseFailure(
        [
            StructuredAttempt(
                attempt=0,
                raw_head="patient payload sk-secret",
                raw_chars=123,
                error_class="ValidationError",
                error_message="secret parser detail",
                finish_reason="stop",
                usage_summary={
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                    "provider_private": "must disappear",
                },
                transport_attempts=1,
                validation_stage="schema_validation",
                validation_issues=[
                    {
                        "location": ["steps", 0, "method"],
                        "issue_type": "literal_error",
                    },
                ],
                violation_sha256="a" * 64,
            )
        ],
        role="planner",
    )

    projected = benchmark._safe_benchmark_structured_attempts(failure)

    assert projected == [
        {
            "attempt": 1,
            "raw_chars": 123,
            "error_class": "validation",
            "finish_reason": "stop",
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
            "transport_attempts": 1,
            "validation_stage": "schema_validation",
            "validation_issues": [
                {
                    "location": ["steps", 0, "method"],
                    "issue_type": "literal_error",
                }
            ],
            "violation_sha256": "a" * 64,
        }
    ]
    assert "secret" not in str(projected)
