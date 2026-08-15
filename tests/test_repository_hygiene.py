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
        "docs/research_agent_duplication_audit.md",
        ".codegraph/.gitignore",
        "tools/arch_baselines/research_agent_top_level_ownership.json",
        "tools/arch_baselines/research_agent_duplicate_helpers.json",
    ):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = (
            '{"modules": {}}'
            if path.name == "research_agent_top_level_ownership.json"
            else '{"helpers": {"_sha256_file": {}, "_finite": {}}}'
            if path.name == "research_agent_duplicate_helpers.json"
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


def test_hygiene_audit_rejects_new_duplicate_helpers(tmp_path: Path) -> None:
    baseline = tmp_path / "tools/arch_baselines/research_agent_duplicate_helpers.json"
    baseline.parent.mkdir(parents=True)
    baseline.write_text(
        '{"helpers": {"_sha256_file": {}, "_finite": {}}}',
        encoding="utf-8",
    )
    helper = tmp_path / "src/easyicu/research_agent/new_helper.py"
    helper.parent.mkdir(parents=True)
    helper.write_text(
        "def _sha256_file(path):\n    return path.read_bytes()\n",
        encoding="utf-8",
    )

    findings = audit_repository(tmp_path, tracked_files=())

    assert (
        "new local duplicate helper definition: "
        "_sha256_file x1 in new_helper.py"
    ) in findings


def test_hygiene_audit_scans_methods_and_nested_helpers(tmp_path: Path) -> None:
    baseline = tmp_path / "tools/arch_baselines/research_agent_duplicate_helpers.json"
    baseline.parent.mkdir(parents=True)
    baseline.write_text('{"helpers": {"_sha256_file": {}}}', encoding="utf-8")
    helper = tmp_path / "src/easyicu/research_agent/nested_helper.py"
    helper.parent.mkdir(parents=True)
    helper.write_text(
        "class Owner:\n"
        "    def _sha256_file(self, path):\n"
        "        return path.read_bytes()\n"
        "def outer():\n"
        "    def _sha256_file(path):\n"
        "        return path.read_bytes()\n"
        "    return _sha256_file\n",
        encoding="utf-8",
    )

    findings = audit_repository(tmp_path, tracked_files=())

    assert (
        "new local duplicate helper definition: "
        "_sha256_file x2 in nested_helper.py"
    ) in findings
