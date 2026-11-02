"""Root-staged Node CLI launchers survive bundler link materialization."""
from __future__ import annotations

import errno
import hashlib
import os
from pathlib import Path
import shutil
import subprocess

import pytest

from tests.support.desktop_bundle import load_script


def make_symlink(link, target, *, directory=False):
    try:
        link.symlink_to(target, target_is_directory=directory)
    except OSError as exc:
        if exc.errno in {errno.EACCES, errno.EPERM, errno.ENOSYS, errno.ENOTSUP} or getattr(exc, "winerror", None) in {50, 1314}:
            pytest.skip("Symlink creation is unavailable on this host")
        raise


def make_stage(tmp_path, package_name):
    root = tmp_path / "runtime with space"
    bins = root / "node_modules/.bin"
    bins.mkdir(parents=True)
    (root / "package.json").write_text('{"type":"module"}', encoding="utf-8")
    target = root / "node_modules" / package_name / "bin.mjs"
    target.parent.mkdir()
    target.write_text(
        '#!/usr/bin/env node\n'
        'process.stdin.setEncoding("utf8"); let body="";\n'
        'process.stdin.on("data", value => body += value);\n'
        'process.stdin.on("end", () => { console.log(JSON.stringify([process.argv.slice(2), body, import.meta.url.endsWith("bin.mjs")])); '
        'if (process.argv.includes("bad")) { console.error("explicit failure"); process.exitCode=7; } });\n',
        encoding="utf-8",
    )
    target.chmod(0o755)
    make_symlink(bins / "sample", os.path.relpath(target, bins))
    return root, bins, target


@pytest.fixture
def staged(tmp_path):
    return make_stage(tmp_path, "package 临床 data")


def run_cli(node, entry, args, stdin, cwd):
    return subprocess.run([node, str(entry), *args], input=stdin, text=True, capture_output=True, cwd=cwd, timeout=10)


def fixture_builder(monkeypatch, root, target):
    common = load_script("build_common")
    monkeypatch.setattr(common, "_AUDITED_NODE_BIN_LINKS", {
        "node_modules/.bin/sample": (target.relative_to(root).as_posix(), hashlib.sha256(target.read_bytes()).hexdigest()),
    })
    return common


@pytest.mark.parametrize("args,stdin", [(["a b", "#?", "中文"], "payload\n"), (["bad"], "")])
def test_launcher_matches_original_after_relocation(staged, tmp_path, monkeypatch, args, stdin):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is required")
    root, bins, target = staged
    before = run_cli(node, bins / "sample", args, stdin, root)
    result = fixture_builder(monkeypatch, root, target).normalize_node_bin_links(root)
    assert result == {"converted": 1}
    assert not (bins / "sample").is_symlink()
    assert (bins / "sample").stat().st_mode & 0o777 == target.stat().st_mode & 0o777
    after = run_cli(node, bins / "sample", args, stdin, root)
    assert (after.returncode, after.stdout, after.stderr) == (before.returncode, before.stdout, before.stderr)
    moved = tmp_path / "relocated 临床 path"
    shutil.move(root, moved)
    relocated = run_cli(node, moved / "node_modules/.bin/sample", args, stdin, moved)
    assert (relocated.returncode, relocated.stdout, relocated.stderr) == (before.returncode, before.stdout, before.stderr)


@pytest.mark.parametrize("kind", ["outside", "missing", "directory", "non_node", "indirect_link"])
def test_invalid_targets_are_rejected_before_any_mutation(staged, tmp_path, monkeypatch, kind):
    root, bins, valid_target = staged
    bad = bins / "zbad"  # Validate the earlier valid entry before rejecting this one.
    if kind == "outside":
        target = tmp_path / "outside"
        target.write_text("#!/usr/bin/env node\n")
    elif kind == "missing":
        target = root / "node_modules/absent"
    elif kind == "directory":
        target = root / "node_modules/dir"
        target.mkdir()
    elif kind == "non_node":
        target = root / "node_modules/shell"
        target.write_text("#!/bin/sh\n")
    else:
        target = root / "node_modules/indirect"
        make_symlink(target, root / "node_modules/package 临床 data/bin.mjs")
    make_symlink(bad, os.path.relpath(target, bins), directory=kind == "directory")
    before = os.readlink(bins / "sample")
    with pytest.raises((OSError, RuntimeError)):
        fixture_builder(monkeypatch, root, valid_target).normalize_node_bin_links(root)
    assert (bins / "sample").is_symlink()
    assert os.readlink(bins / "sample") == before
    assert bad.is_symlink()


def test_ordinary_windows_shims_are_unchanged(tmp_path):
    root = tmp_path / "runtime"
    bins = root / "node_modules/.bin"
    bins.mkdir(parents=True)
    files = {"tool.cmd": b"@echo off\r\n", "tool.ps1": b"Write-Output 'hello'\n", "tool": b"#!/bin/sh\n"}
    for name, content in files.items():
        (bins / name).write_bytes(content)
    modes = {name: (bins / name).stat().st_mode for name in files}
    assert load_script("build_common").normalize_node_bin_links(root) == {"converted": 0}
    assert {name: (bins / name).read_bytes() for name in files} == files
    assert {name: (bins / name).stat().st_mode for name in files} == modes


@pytest.mark.skipif(os.name == "nt", reason="Windows filenames cannot contain #? together with ?")
def test_url_reserved_characters_in_posix_target_are_encoded(tmp_path, monkeypatch):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is required")
    root, bins, target = make_stage(tmp_path, "package #?% 临床")
    before = run_cli(node, bins / "sample", ["reserved"], "payload\n", root)
    fixture_builder(monkeypatch, root, target).normalize_node_bin_links(root)
    launcher = (bins / "sample").read_text()
    assert "%23%3F%25" in launcher
    after = run_cli(node, bins / "sample", ["reserved"], "payload\n", root)
    assert (after.returncode, after.stdout, after.stderr) == (before.returncode, before.stdout, before.stderr)


@pytest.mark.parametrize("change", ["unknown_link", "modified_target"])
def test_only_exact_audited_link_and_target_are_converted(staged, monkeypatch, change):
    root, bins, target = staged
    common = fixture_builder(monkeypatch, root, target)
    if change == "unknown_link":
        (bins / "sample").rename(bins / "another")
        link = bins / "another"
    else:
        target.write_text(target.read_text() + "// changed\n")
        link = bins / "sample"
    with pytest.raises(RuntimeError, match="not been audited"):
        common.normalize_node_bin_links(root)
    assert link.is_symlink()


def test_seal_remains_valid_after_copy_and_detects_launcher_tamper(staged, tmp_path, monkeypatch):
    from easyicu.webserver.pi_copilot.install import _packaged_file_inventory

    root, bins, target = staged
    common = fixture_builder(monkeypatch, root, target)
    common.normalize_node_bin_links(root)
    common.seal_packaged_pi(root)
    copy = tmp_path / "copied"
    shutil.copytree(root, copy, symlinks=False)
    assert _packaged_file_inventory(root) == _packaged_file_inventory(copy)
    (copy / "node_modules/.bin/sample").write_text("// tampered\n")
    assert _packaged_file_inventory(root) != _packaged_file_inventory(copy)


@pytest.mark.parametrize("helper", ["draft-repair.mjs", "attached-skills.mjs"])
def test_existing_staged_import_check_rejects_broken_required_helper(tmp_path, helper):
    from tests.support.desktop_bundle import node_app

    node = shutil.which("node")
    if not node:
        pytest.skip("Node is required")
    root = node_app(tmp_path / "synthetic-stage")
    (root / "src" / helper).write_text("throw new Error('synthetic required helper failure');\n", encoding="utf-8")
    common = load_script("build_common")

    def run(command, **kwargs):
        subprocess.run(command, **kwargs, check=True, text=True, timeout=15)

    with pytest.raises(common.StagedPiImportError, match=helper):
        common.check_staged_pi(root, Path(node), run=run)
