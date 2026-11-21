"""A review displays only the exact typed population bound by its authority."""

from types import SimpleNamespace

import pytest

from easyicu.webserver.pi_copilot.population_spec_review import reviewed_population_spec
from tests.support.bound_population_review import bound_population_review


@pytest.fixture
def review(tmp_path):
    return bound_population_review(tmp_path)


def test_review_displays_exact_compiled_spec_from_the_bound_source(review):
    root, pending = review
    result = reviewed_population_spec(run_dir=root, pending=pending)
    assert result is not None
    row = result["criteria"][0]
    assert row["criterion"]["quote"] == "first creatinine up to 4 mg/dL in the first 24 h"
    assert row["criterion"]["unit"] == "mg/dL"
    assert row["criterion"]["window"] == {"start_hours": 0.0, "end_hours": 24.0}
    assert row["criterion"]["missing_policy"] == "exclude"
    assert row["disposition"] == "applied_by_plan"


@pytest.mark.parametrize("filename", ["progressive_plan_skeleton.json", "research_context.json",
                                      "progressive_planning_authority.json", "progressive_plan_compile_receipt.json"])
def test_review_refuses_modified_or_missing_bound_artifacts(review, filename):
    root, pending = review
    path = root / filename
    path.write_bytes(path.read_bytes() + b" ")
    assert reviewed_population_spec(run_dir=root, pending=pending) is None


def test_review_refuses_another_plan_and_missing_authority(review):
    root, pending = review
    raw = pending.requests[0].payload["plan_review_authority"]
    altered = dict(raw["plan_payload"]); altered["research_question"] = "A different question"
    raw["plan_payload"] = altered
    assert reviewed_population_spec(run_dir=root, pending=pending) is None
    assert reviewed_population_spec(run_dir=root, pending=SimpleNamespace(requests=[])) is None


def test_multiple_requests_can_include_an_unrelated_request_without_plan_authority(review):
    root, pending = review
    pending.requests.append(SimpleNamespace(payload={"kind": "another_review"}))
    assert reviewed_population_spec(run_dir=root, pending=pending) is not None


def test_multiple_different_review_authorities_fail_closed(review):
    root, pending = review
    raw = pending.requests[0].payload["plan_review_authority"]
    different = dict(raw)
    different["evidence_sha256"] = dict(raw["evidence_sha256"], other="f" * 64)
    pending.requests.append(SimpleNamespace(payload={"plan_review_authority": different}))
    assert reviewed_population_spec(run_dir=root, pending=pending) is None
