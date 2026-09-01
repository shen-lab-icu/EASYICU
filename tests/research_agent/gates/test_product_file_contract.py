from __future__ import annotations

from easyicu.research_agent.contracts.product_files import (
    descriptor_path_is_compatible,
    file_kinds,
)


def test_physical_suffixes_project_to_closed_product_kinds() -> None:
    assert file_kinds("table.csv") == {"table", "artifact", "dataset", "test"}
    assert file_kinds("figure.svg") == {"figure"}
    assert file_kinds("model.joblib") == {"model", "artifact"}
    assert file_kinds("unknown.bin") == set()


def test_semantic_roles_require_compatible_physical_payloads() -> None:
    assert descriptor_path_is_compatible(kind="audit", path="audit.csv")
    assert descriptor_path_is_compatible(kind="audit", path="audit.json")
    assert descriptor_path_is_compatible(kind="report", path="report.md")
    assert not descriptor_path_is_compatible(kind="report", path="report.csv")
    assert not descriptor_path_is_compatible(kind="audit", path="plot.png")


def test_suffix_inference_never_creates_semantic_roles() -> None:
    assert "audit" not in file_kinds("audit.csv")
    assert "report" not in file_kinds("report.md")
