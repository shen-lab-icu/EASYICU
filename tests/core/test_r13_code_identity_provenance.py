"""Native-export code identity: a checkout, a build stamp, or a typed unknown.

lead-r13 provenance finding: bench runs imported EasyICU from a tree copied out
of Git, and every native manifest recorded ``easyicu_git_commit = None`` with
nothing saying why. These tests pin the three identities the owner
(``easyicu.code_identity``) may report and how the manifest records them.
"""

from __future__ import annotations

import json
import os
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

    # The tracked files as they are in this tree (so uncommitted edits are
    # what the copy runs), without Git metadata.
    tracked = _git("ls-files", "-z", "src/easyicu").split("\0")
    for relative in filter(None, tracked):
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / relative, target)
    return root / "src" / "easyicu"


def _provenance_of(package: Path, *, extra_path: Path | None = None) -> dict:
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
            "PYTHONPATH": os.pathsep.join(
                ([str(extra_path)] if extra_path else []) + [str(package.parent)]
            ),
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


# delta7p D7P-OUTER-HEAD: git_checkout only for <root>/src/easyicu where <root>
# is the Git top level, names the EasyICU project, and tracks the package.

needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="needs git")


def _repo(root: Path, *, name: str | None, tracked: tuple[str, ...], untracked=()) -> Path:
    root.mkdir(parents=True)
    _git("init", "-q", cwd=root)
    (root / "README").write_text("r\n")
    if name is not None:
        (root / "pyproject.toml").write_text(f'[project]\nname = "{name}"\nversion = "0"\n')
    for relative in (*tracked, *untracked):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# x\n")
    _git("add", "README", *(["pyproject.toml"] if name is not None else []), *tracked, cwd=root)
    _git(
        "-c", "user.name=t", "-c", "user.email=t@example.invalid",
        "commit", "-q", "-m", "r", cwd=root,
    )
    return root


_PACKAGE_FILES = ("__init__.py", "code_identity.py")


def _assert_not_borrowed(package: Path) -> None:
    identity = code_identity.resolve_code_identity(package)
    assert identity["source"] == code_identity.SOURCE_UNKNOWN
    assert identity["reason_code"] == code_identity.REASON_UNAVAILABLE
    assert identity["commit"] is None


@needs_git
def test_an_easyicu_checkout_with_the_package_tracked_is_a_git_checkout(tmp_path):
    root = _repo(
        tmp_path / "checkout", name="easyicu",
        tracked=tuple(f"src/easyicu/{f}" for f in _PACKAGE_FILES),
    )

    identity = code_identity.resolve_code_identity(root / "src" / "easyicu")

    assert identity["source"] == code_identity.SOURCE_GIT_CHECKOUT
    assert identity["commit"] == _git("rev-parse", "HEAD", cwd=root)
    assert identity["dirty"] is False


@needs_git
def test_a_vendored_install_inside_another_repository_is_not_a_checkout(tmp_path):
    outer = _repo(
        tmp_path / "outer", name="other-project",
        tracked=tuple(f"vendor/easyicu/{f}" for f in _PACKAGE_FILES),
    )
    _assert_not_borrowed(outer / "vendor" / "easyicu")


@needs_git
def test_a_package_under_src_of_a_different_project_is_not_a_checkout(tmp_path):
    outer = _repo(
        tmp_path / "outer", name="other-project",
        tracked=tuple(f"src/easyicu/{f}" for f in _PACKAGE_FILES),
    )
    _assert_not_borrowed(outer / "src" / "easyicu")


@needs_git
def test_a_look_alike_tree_whose_package_git_does_not_track_is_not_a_checkout(tmp_path):
    outer = _repo(
        tmp_path / "lookalike", name="easyicu", tracked=(),
        untracked=tuple(f"src/easyicu/{f}" for f in _PACKAGE_FILES),
    )
    _assert_not_borrowed(outer / "src" / "easyicu")


@needs_git
def test_a_symlinked_package_is_judged_where_it_really_lives(tmp_path):
    outer = _repo(
        tmp_path / "outer", name="other-project",
        tracked=tuple(f"vendor/easyicu/{f}" for f in _PACKAGE_FILES),
    )
    link = tmp_path / "shim" / "src" / "easyicu"
    link.parent.mkdir(parents=True)
    link.symlink_to(outer / "vendor" / "easyicu", target_is_directory=True)

    _assert_not_borrowed(link)


def test_a_stamp_without_every_field_is_a_typed_unknown(tmp_path):
    package = tmp_path / "site" / "lib" / "easyicu"
    package.mkdir(parents=True)
    (package / code_identity.BUILD_IDENTITY_FILE).write_text(json.dumps({
        "schema": code_identity.BUILD_IDENTITY_SCHEMA,
        "version": "0.1.0", "commit": "a" * 40, "dirty": False,
    }))

    identity = code_identity.resolve_code_identity(package)

    assert identity["source"] == code_identity.SOURCE_UNKNOWN
    assert identity["reason_code"] == code_identity.REASON_STAMP_INVALID



# Supervisor ruling (B): easyicu_version is the version of the code that ran.

def _committed_checkout(root: Path, *, version: str) -> Path:
    package = _copied_package(root)
    (root / "pyproject.toml").write_text(
        f'[project]\nname = "easyicu"\nversion = "{version}"\n'
    )
    _git("init", "-q", cwd=root)
    _git("add", "-A", cwd=root)
    _git(
        "-c", "user.name=t", "-c", "user.email=t@example.invalid",
        "commit", "-q", "-m", "checkout", cwd=root,
    )
    return package


@needs_checkout
def test_the_manifest_version_is_the_checkouts_not_the_installed_dists(tmp_path):
    package = _committed_checkout(tmp_path / "checkout", version="7.7.7")
    # A different easyicu distribution installed ahead of everything else.
    dist = tmp_path / "site" / "easyicu-9.9.9.dist-info"
    dist.mkdir(parents=True)
    (dist / "METADATA").write_text("Metadata-Version: 2.1\nName: easyicu\nVersion: 9.9.9\n")

    provenance = _provenance_of(package, extra_path=dist.parent)

    assert provenance["easyicu_code_identity_source"] == code_identity.SOURCE_GIT_CHECKOUT
    assert provenance["easyicu_version"] == "7.7.7"
    assert provenance["easyicu_installed_dist_version"] == "9.9.9"


@needs_checkout
def test_a_stamped_install_records_the_stamps_version(tmp_path):
    package = _copied_package(tmp_path / "wheel")
    stamp = code_identity.build_stamp_payload(REPO, version="5.5.5")
    (package / code_identity.BUILD_IDENTITY_FILE).write_text(json.dumps(stamp))

    assert _provenance_of(package)["easyicu_version"] == "5.5.5"


@needs_checkout
def test_an_unknown_identity_records_no_version(tmp_path):
    provenance = _provenance_of(_copied_package(tmp_path / "copy"))

    assert provenance["easyicu_version"] is None
    assert provenance["easyicu_code_identity_reason"] == code_identity.REASON_UNAVAILABLE


@needs_git
def test_a_pyproject_whose_project_is_not_a_table_is_a_typed_unknown(tmp_path):
    # delta9p NB9P-PROJECT-TYPE: valid TOML, but [project] is not a table.
    root = _repo(
        tmp_path / "checkout", name=None,
        tracked=tuple(f"src/easyicu/{f}" for f in _PACKAGE_FILES),
    )
    (root / "pyproject.toml").write_text('project = "easyicu"\n')
    _git("add", "pyproject.toml", cwd=root)
    _git(
        "-c", "user.name=t", "-c", "user.email=t@example.invalid",
        "commit", "-q", "-m", "p", cwd=root,
    )

    identity = code_identity.resolve_code_identity(root / "src" / "easyicu")

    assert identity["source"] == code_identity.SOURCE_UNKNOWN
    assert identity["reason_code"] == code_identity.REASON_UNAVAILABLE
