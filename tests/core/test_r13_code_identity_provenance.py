"""Native-export code identity: a checkout, a build stamp, or a typed unknown.

lead-r13 provenance finding: bench runs imported EasyICU from a tree copied out
of Git, and every native manifest recorded ``easyicu_git_commit = None`` with
nothing saying why. These tests pin the three identities the owner
(``easyicu.code_identity``) may report and how the manifest records them.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from easyicu import code_identity
from easyicu.api import extraction

REPO = Path(__file__).resolve().parents[2]


def _git(*arguments: str, cwd: Path = REPO) -> str:
    return subprocess.run(
        ["git", *arguments], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()


def _is_checkout() -> bool:
    try:
        return Path(_git("rev-parse", "--show-toplevel")).resolve() == REPO.resolve()
    except (OSError, subprocess.SubprocessError):
        return False


needs_checkout = pytest.mark.skipif(
    shutil.which("git") is None or not _is_checkout(),
    reason="needs this repository as a Git checkout",
)


def _copied_package(root: Path) -> Path:
    """The bench layout: ``src/easyicu`` copied out of Git without ``.git``."""

    root.mkdir(parents=True)
    archive = subprocess.run(
        ["git", "archive", "HEAD", "src/easyicu"],
        cwd=REPO, check=True, capture_output=True,
    ).stdout
    subprocess.run(["tar", "-x", "-C", str(root)], input=archive, check=True)
    return root / "src" / "easyicu"


def _provenance_of(package: Path) -> dict:
    home = package.parents[2] / "home"
    home.mkdir(exist_ok=True)
    code = (
        "import json; from easyicu.api import extraction; "
        "print(json.dumps(extraction._native_export_runtime_provenance()))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        cwd=package.parents[1],
        env={
            "PYTHONPATH": str(package.parent),
            "PATH": "/usr/bin:/bin",
            "HOME": str(home),
            "EASYICU_HOME": str(home / ".easyicu"),
        },
        check=True, capture_output=True, text=True, timeout=300,
    ).stdout
    provenance = json.loads(out.strip().splitlines()[-1])
    assert Path(provenance["easyicu_import_path"]).resolve() == package.resolve()
    return provenance


@needs_checkout
def test_a_checkout_run_records_the_commit_of_the_code_it_ran():
    provenance = extraction._native_export_runtime_provenance()
    commit = provenance["easyicu_git_commit"]
    assert isinstance(commit, str) and len(commit) == 40
    assert commit == _git("rev-parse", "HEAD")
    assert provenance["easyicu_code_identity_source"] == code_identity.SOURCE_GIT_CHECKOUT
    assert isinstance(provenance["easyicu_git_dirty"], bool)
    assert "easyicu_code_identity_reason" not in provenance


@needs_checkout
def test_a_tree_copied_out_of_git_says_unknown_with_a_reason(tmp_path):
    provenance = _provenance_of(_copied_package(tmp_path / "copy"))

    assert provenance["easyicu_code_identity_source"] == code_identity.SOURCE_UNKNOWN
    assert provenance["easyicu_code_identity_reason"] == code_identity.REASON_UNAVAILABLE
    assert provenance["easyicu_git_commit"] is None


@needs_checkout
def test_a_stamped_install_records_the_stamped_commit_and_version(tmp_path):
    package = _copied_package(tmp_path / "wheel")
    stamp = code_identity.build_stamp_payload(REPO, version="9.9.9")
    assert stamp is not None
    (package / code_identity.BUILD_IDENTITY_FILE).write_text(json.dumps(stamp))

    provenance = _provenance_of(package)

    assert provenance["easyicu_code_identity_source"] == code_identity.SOURCE_BUILD_STAMP
    assert provenance["easyicu_git_commit"] == _git("rev-parse", "HEAD")
    assert provenance["easyicu_git_dirty"] is stamp["dirty"]


def test_a_malformed_stamp_is_a_typed_unknown(tmp_path):
    package = tmp_path / "site" / "lib" / "easyicu"
    package.mkdir(parents=True)
    (package / code_identity.BUILD_IDENTITY_FILE).write_text(
        json.dumps({"schema": code_identity.BUILD_IDENTITY_SCHEMA, "commit": "abc"})
    )

    identity = code_identity.resolve_code_identity(package)

    assert identity["source"] == code_identity.SOURCE_UNKNOWN
    assert identity["reason_code"] == code_identity.REASON_STAMP_INVALID
    assert identity["commit"] is None


@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
def test_a_package_inside_another_repository_never_borrows_its_commit(tmp_path):
    outer = tmp_path / "outer"
    outer.mkdir()
    _git("init", "-q", cwd=outer)
    (outer / "README").write_text("outer\n")
    _git("add", "README", cwd=outer)
    _git(
        "-c", "user.name=t", "-c", "user.email=t@example.invalid",
        "commit", "-q", "-m", "outer", cwd=outer,
    )
    # A virtual environment (or a copied tree) under an unrelated checkout.
    package = outer / ".venv" / "lib" / "site-packages" / "easyicu"
    package.mkdir(parents=True)

    identity = code_identity.resolve_code_identity(package)

    assert identity["source"] == code_identity.SOURCE_UNKNOWN
    assert identity["commit"] is None
