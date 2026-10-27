"""Synthetic installers bind both acceptance and draft assets to one build."""

import hashlib
import json

import pytest

from tests.support.desktop_bundle import load_script

INSTALLERS = {
    "windows": ("cargo/release/bundle/nsis/EasyICU 临床 setup.exe", "x86_64-pc-windows-msvc", ".exe"),
    "macos": ("cargo/release/bundle/dmg/EasyICU 临床_aarch64.dmg", "aarch64-apple-darwin", ".dmg"),
}


@pytest.fixture(params=sorted(INSTALLERS))
def artifact(request, tmp_path):
    platform = request.param
    relative, target, _ = INSTALLERS[platform]
    root = tmp_path / f"download-{platform}"
    installer = root / relative
    installer.parent.mkdir(parents=True)
    installer.write_bytes(f"synthetic {platform} installer".encode() + b"\x00\xff")
    receipt = root / "build/build-inputs.json"
    receipt.parent.mkdir()
    receipt.write_text(json.dumps({
        "source_commit": "a" * 40, "source_dirty": False, "target": target,
        "installer": {"name": installer.name, "sha256": hashlib.sha256(installer.read_bytes()).hexdigest()},
    }))
    return platform, root, installer, receipt


def test_verified_artifact_is_the_exact_installer_and_checksum(artifact, tmp_path, capsys):
    module = load_script("verify_desktop_artifact")
    platform, root, installer, _ = artifact
    output = tmp_path / "release"
    assert module.main([platform, str(root), "--source-commit", "a" * 40, "--release-dir", str(output)]) == 0
    published = output / module.release_name(installer.name)
    assert capsys.readouterr().out.strip() == str(published.resolve())
    assert published.read_bytes() == installer.read_bytes()
    digest = hashlib.sha256(installer.read_bytes()).hexdigest()
    assert (output / "SHA256SUMS.txt").read_bytes() == f"{digest}  {published.name}\n".encode()
    assert sorted(p.name for p in output.iterdir()) == sorted([published.name, "SHA256SUMS.txt"])


def test_one_release_lists_every_platform_in_one_checksum_file(tmp_path, capsys):
    module = load_script("verify_desktop_artifact")
    output = tmp_path / "release"
    lines = []
    for platform, (relative, target, _) in INSTALLERS.items():
        root = tmp_path / platform
        installer = root / relative
        installer.parent.mkdir(parents=True)
        installer.write_bytes(platform.encode())
        digest = hashlib.sha256(platform.encode()).hexdigest()
        (root / "build-inputs.json").write_text(json.dumps({
            "source_commit": "a" * 40, "source_dirty": False, "target": target,
            "installer": {"name": installer.name, "sha256": digest},
        }))
        for _ in range(2):  # A rerun replaces its own line instead of adding one.
            assert module.main([platform, str(root), "--source-commit", "a" * 40, "--release-dir", str(output)]) == 0
        lines.append(f"{digest}  {module.release_name(installer.name)}\n")
    capsys.readouterr()
    assert (output / "SHA256SUMS.txt").read_text(encoding="utf-8") == "".join(
        sorted(lines, key=lambda line: line.split("  ", 1)[1]))


@pytest.mark.parametrize(("built", "uploaded"), [
    # v0.1.0: GitHub stored the space as "." and SHA256SUMS named the space.
    ("EasyICU Desktop_0.1.0_x64-setup.exe", "EasyICU.Desktop_0.1.0_x64-setup.exe"),
    ("EasyICU_0.1.0_aarch64.dmg", "EasyICU_0.1.0_aarch64.dmg"),
    ("EasyICU 临床 setup.exe", "EasyICU.setup.exe"),
])
def test_the_release_copy_and_its_checksum_use_the_name_github_keeps(built, uploaded):
    module = load_script("verify_desktop_artifact")
    assert module.release_name(built) == uploaded
    assert module.release_name(uploaded) == uploaded


def test_acceptance_verifies_without_copying_or_creating_release_files(artifact, capsys):
    module = load_script("verify_desktop_artifact")
    platform, root, installer, _ = artifact
    before = sorted(root.rglob("*"))
    assert module.main([platform, str(root), "--source-commit", "a" * 40]) == 0
    assert capsys.readouterr().out.strip() == str(installer.resolve())
    assert sorted(root.rglob("*")) == before


def test_a_platform_never_accepts_the_other_platforms_build(artifact, tmp_path):
    module = load_script("verify_desktop_artifact")
    platform, root, _, _ = artifact
    other = "macos" if platform == "windows" else "windows"
    with pytest.raises(ValueError, match=f"{other}_artifact_ambiguous_or_missing"):
        module.main([other, str(root), "--source-commit", "a" * 40, "--release-dir", str(tmp_path / "release")])
    assert not (tmp_path / "release").exists()


@pytest.mark.parametrize("change", [
    "commit", "dirty", "target", "name", "digest", "tamper", "missing_installer", "extra_installer",
    "missing_receipt", "extra_receipt",
])
def test_unrelated_or_corrupt_builds_cannot_be_accepted_or_released(artifact, tmp_path, change):
    module = load_script("verify_desktop_artifact")
    platform, root, installer, receipt_path = artifact
    receipt = json.loads(receipt_path.read_text())
    if change in {"commit", "dirty", "target"}:
        key, value = {"commit": ("source_commit", "b" * 40), "dirty": ("source_dirty", True),
                      "target": ("target", "another-platform")}[change]
        receipt[key] = value
    elif change in {"name", "digest"}:
        receipt["installer"]["name" if change == "name" else "sha256"] = "wrong"
    elif change == "tamper":
        installer.write_bytes(b"replaced installer")
    elif change == "missing_installer":
        installer.unlink()
    elif change == "extra_installer":
        (root / f"extra{INSTALLERS[platform][2]}").write_bytes(b"second installer")
    receipt_path.write_text(json.dumps(receipt))
    if change == "missing_receipt":
        receipt_path.unlink()
    elif change == "extra_receipt":
        (root / "build-inputs.json").write_text(json.dumps(receipt))
    output = tmp_path / "release"
    with pytest.raises(ValueError, match=f"{platform}_artifact_"):
        module.main([platform, str(root), "--source-commit", "a" * 40, "--release-dir", str(output)])
    assert not output.exists()
