from pathlib import Path

from tools.audit_repository_hygiene import audit_repository


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_repository_layout_has_one_owner_per_content_class() -> None:
    assert audit_repository(REPO_ROOT) == ()


def test_hygiene_audit_rejects_competing_and_generated_tracked_roots(
    tmp_path: Path,
) -> None:
    for relative in (
        "benchmarks/README.md",
        "benchmarks/cases/.keep",
        "benchmarks/catalogs/.keep",
        "benchmarks/idea_mining/.keep",
        "docs/qa/.keep",
        "docs/repository_layout.md",
        "docs/research_agent_capability_inventory.md",
        ".codegraph/.gitignore",
        "tools/arch_baselines/research_agent_top_level_ownership.json",
    ):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = (
            '{"modules": {}}'
            if path.name == "research_agent_top_level_ownership.json"
            else "*\n!.gitignore\n"
        )
        path.write_text(payload, encoding="utf-8")

    (tmp_path / "benchmark").mkdir()
    findings = audit_repository(
        tmp_path,
        tracked_files=(Path("dist/easyicu.whl"), Path("src/pkg/__pycache__/x.pyc")),
    )

    assert "competing top-level benchmark/ owner exists; use benchmarks/" in findings
    assert "generated or retired root is tracked: dist/easyicu.whl" in findings
    assert "cache payload is tracked: src/pkg/__pycache__/x.pyc" in findings
