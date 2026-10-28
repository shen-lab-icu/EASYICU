"""Uploaded packages remain frozen instruction data, including their references."""

from io import BytesIO
import stat
import zipfile

import pytest

from easyicu.extensions import ExtensionRegistry, ExtensionRegistryError


SKILL = "---\nname: synthetic-helper\ndescription: Explain synthetic examples.\n---\nRead references/guide.md.\n"


def archive(files, *, symlink=False):
    out = BytesIO()
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as package:
        for name, content in files.items():
            info = zipfile.ZipInfo(name)
            if symlink:
                info.create_system = 3
                info.external_attr = (stat.S_IFLNK | 0o777) << 16
            package.writestr(info, content)
    return out.getvalue()


def test_package_references_reach_frozen_attachment_and_writing_consumers(tmp_path):
    from easyicu.webserver.pi_copilot.contracts import PiSessionRecord
    from easyicu.webserver.pi_copilot.skill_attachments import load_attached_users, validate_attachments

    registry = ExtensionRegistry(tmp_path / "extensions")
    result = registry.install_skill_package(
        archive_bytes=archive({"bundle/SKILL.md": SKILL, "bundle/references/guide.md": "Use the blue example."}),
        stages=["conversation", "writing"],
    )
    frozen = registry.snapshot()
    refs = validate_attachments([{"source": "user", "id": result["name"], "digest": result["digest"]}],
                                activation=frozen, registry=registry)
    loaded = load_attached_users(PiSessionRecord(session_id="synthetic", extension_activation=frozen), refs, registry)
    assert "Use the blue example." in loaded[0]["instructions"]
    assert "Use the blue example." in registry.pipeline_activation(frozen)["writing_advisory"]
    changed = registry.install_skill_package(files={"SKILL.md": SKILL, "references/guide.md": "Use the green example."})
    assert changed["digest"] != result["digest"]
    assert "blue example" in registry.load_skill(name=result["name"], digest=result["digest"])["instructions"]
    assert "green example" in registry.current_skill(result["name"])["instructions"]


def test_folder_import_edit_export_delete_uses_the_same_registry(tmp_path):
    registry = ExtensionRegistry(tmp_path / "extensions")
    registry.install_skill_package(files={"folder/SKILL.md": SKILL, "folder/references/guide.md": "Explain clearly."})
    current = registry.current_skill("synthetic-helper")
    assert current["reference_files"] == {"references/guide.md": "Explain clearly."}
    registry.install_skill(SKILL.replace("Read references", "First read references"), reference_files=current["reference_files"])
    exported = registry.export_skill_package("synthetic-helper")
    with zipfile.ZipFile(BytesIO(exported)) as package:
        assert package.namelist() == ["SKILL.md", "references/guide.md"]
        assert b"First read" in package.read("SKILL.md")
        assert package.read("references/guide.md") == b"Explain clearly."
    registry.remove(kind="skill", name="synthetic-helper")
    assert registry.list_public()["skills"] == []
    with pytest.raises(ExtensionRegistryError):
        registry.export_skill_package("synthetic-helper")


@pytest.mark.parametrize("path", ["../escape.md", "/escape.md", "x/../../escape.md", "x\\escape.md", "C:/escape.md", "references/./escape.md"])
def test_import_rejects_traversal_even_in_ignored_payloads(tmp_path, path):
    registry = ExtensionRegistry(tmp_path / "extensions")
    with pytest.raises(ExtensionRegistryError, match="path"):
        registry.install_skill_package(archive_bytes=archive({"SKILL.md": SKILL, path: "unsafe"}))
    assert not registry.state_path.exists()


@pytest.mark.parametrize("files", [
    {"references/a.md": "No main document."},
    {"SKILL.md": "No frontmatter."},
    {"SKILL.md": SKILL, "references/a.md": "x" * 48_001},
    {"SKILL.md": SKILL, **{f"references/{i}.md": "x" for i in range(65)}},
    {"SKILL.md": SKILL, "references/a.md": b"\xff"},
])
def test_import_rejects_missing_invalid_oversize_and_non_text_packages(tmp_path, files):
    with pytest.raises(ExtensionRegistryError):
        ExtensionRegistry(tmp_path / "extensions").install_skill_package(archive_bytes=archive(files), enabled=False)


def test_import_rejects_archive_limit_duplicates_and_symlinks(tmp_path):
    registry = ExtensionRegistry(tmp_path / "extensions")
    for content in [b"x" * 256_001, b"not a zip", archive({"SKILL.md": SKILL}, symlink=True), archive({"SKILL.mdXextra": SKILL}).replace(b"SKILL.mdX", b"SKILL.md\x00")]:
        with pytest.raises(ExtensionRegistryError):
            registry.install_skill_package(archive_bytes=content)
    out = BytesIO()
    with pytest.warns(UserWarning), zipfile.ZipFile(out, "w") as package:
        package.writestr("SKILL.md", SKILL)
        package.writestr("SKILL.md", SKILL)
    with pytest.raises(ExtensionRegistryError):
        registry.install_skill_package(archive_bytes=out.getvalue())

    # A valid ZIP can exceed the compressed-upload limit through metadata alone.
    out = BytesIO()
    with zipfile.ZipFile(out, "w") as package:
        for name in ["SKILL.md", *(f"references/{i}.md" for i in range(4))]:
            info = zipfile.ZipInfo(name)
            info.extra = b"\xfe\xca" + (59_996).to_bytes(2, "little") + b"x" * 59_996
            package.writestr(info, SKILL if name == "SKILL.md" else "Synthetic reference.")
    with pytest.raises(ExtensionRegistryError):
        registry.install_skill_package(archive_bytes=out.getvalue())


def test_executable_payload_is_ignored_and_never_stored_or_exported(tmp_path):
    sentinel = tmp_path / "executed"
    registry = ExtensionRegistry(tmp_path / "extensions")
    result = registry.install_skill_package(files={
        "SKILL.md": SKILL, "references/guide.md": "Use synthetic examples.",
        "scripts/run.py": f"open({str(sentinel)!r}, 'w').write('bad')",
        "setup.sh": "exit 1", "references/payload.py": "raise RuntimeError('bad')",
    })
    assert result["ignored_files"] == ["references/payload.py", "scripts/run.py", "setup.sh"]
    assert not sentinel.exists()
    with zipfile.ZipFile(BytesIO(registry.export_skill_package(result["name"]))) as package:
        assert package.namelist() == ["SKILL.md", "references/guide.md"]
    assert "RuntimeError" not in registry.current_skill(result["name"])["instructions"]


def test_package_import_revision_and_schema_cannot_grant_host_authority(tmp_path):
    registry = ExtensionRegistry(tmp_path / "extensions")
    old = registry.snapshot().activation_sha256
    registry.install_skill(SKILL)
    with pytest.raises(ExtensionRegistryError) as stale:
        registry.install_skill_package(files={"SKILL.md": SKILL}, expected_activation_sha256=old)
    assert stale.value.code == "extension_revision_mismatch"
    with pytest.raises(ExtensionRegistryError):
        registry.install_skill_package(files={"SKILL.md": SKILL.replace("description:", "execute: true\ndescription:")})


def test_active_budget_counts_instruction_bytes_instead_of_json_escaping(tmp_path):
    registry = ExtensionRegistry(tmp_path / "extensions")
    skill = registry.install_skill_package(files={"SKILL.md": SKILL, "references/a.txt": '"' * 30_000})
    assert skill["size_bytes"] < 31_000
    assert registry.snapshot().skills[0].name == "synthetic-helper"
