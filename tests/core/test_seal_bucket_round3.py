"""Seal-tip bucket tool, review round 3 (delta29t): one regression per B finding.

Each test provokes the counterexample the review demonstrated against a2aea5db4 and asserts the typed
refusal (or the repaired behaviour). Synthetic installs only: no clinical data, no real builder.
"""

from __future__ import annotations

import ast
import builtins
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from easyicu import storage_layout as sl
from tests.support.seal_buckets import make_bucket, make_writable
from tools.seal_buckets import bucket_compare as bc
from tools.seal_buckets import footer_scan
from tools.seal_buckets import seal_rebuild as sr

TOOLS = Path(sr.__file__).parent
ORDINAL = sl.ORDINAL


def _tool(name: str, *args: str):
    return subprocess.run(
        [sys.executable, "-I", str(TOOLS / name), *args], capture_output=True, text=True
    )


# ---- D29-EMPTY-STRUCT: the comparator enforces the resolver's full contract, checked for parity


def _part(path: Path, key_type=pa.int64(), variable=False) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = {"k": pa.array(range(10), key_type), "v": pa.array([1.5] * 10, pa.float64())}
    if variable:
        cols["s"] = pa.array(range(10), pa.int64())
    cols[ORDINAL] = pa.array(range(10), pa.int64())
    pq.write_table(pa.table(cols), path, use_dictionary=False)
    return {"row_count": 10, "size": path.stat().st_size}


def _install(base: Path, variable=False) -> Path:
    """A valid install whose layout has EMPTY structure: an empty bucket directory (hash) or an empty
    partition and an empty bucket inside a populated partition (variable)."""
    root = base / "t_bucket"
    source = base / "t.parquet"
    base.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({"k": [1]}), source)
    inventory = []
    if variable:
        rel = "var_id=0/bucket_id=0/part.parquet"
        inventory.append({"relative_path": rel, **_part(root / rel, variable=True)})
        (root / "var_id=0" / "bucket_id=1").mkdir(parents=True)
        (root / "var_id=1" / "bucket_id=0").mkdir(parents=True)
    else:
        rel = "bucket_id=0/part.parquet"
        inventory.append({"relative_path": rel, **_part(root / rel)})
        (root / "bucket_id=1").mkdir(parents=True)
    st = source.stat()
    manifest = {
        "schema": sl.SCHEMA,
        "scheme": sl.VARIABLE_SCHEME if variable else "hash",
        "key_column": "k",
        "key_type": "BIGINT",
        "modulus": 2,
        "row_group_target": 131072,
        "builder_version": sl.BUILDER_VERSION,
        "sort_keys": ["k", ORDINAL],
        "source_order": ORDINAL,
        "inventory": inventory,
        "derived_from": {
            "table": "t",
            "files": [
                {
                    "path": "t.parquet",
                    "sha256": sr.sha256_file(source),
                    "size": st.st_size,
                    "mtime_ns": st.st_mtime_ns,
                }
            ],
        },
    }
    if variable:
        manifest.update(
            stay_column="s",
            stay_key_type="BIGINT",
            stay_breakpoints=[5],
            secondary_keys=[],
            partitions=[
                {"id": 0, "value": 1, "modulus": 2, "secondary": {}},
                {"id": 1, "value": 2, "modulus": 1, "secondary": {}},
            ],
        )
    else:
        manifest["hash"] = {"name": "duckdb.hash", "duckdb_version": duckdb.__version__}
    (root / "_BUCKET_MANIFEST.json").write_text(json.dumps(manifest))
    (root / "_BUCKET_BUILD_RECEIPT.json").write_text(json.dumps({"rows": 10}))
    (root / "_COMPLETE").write_text("complete\n")
    return root


def _edit_manifest(root: Path, change) -> None:
    m = json.loads((root / "_BUCKET_MANIFEST.json").read_text())
    change(m)
    (root / "_BUCKET_MANIFEST.json").write_text(json.dumps(m))


def _wrong_schema(root: Path) -> None:
    _part(root / "bucket_id=0" / "part.parquet", key_type=pa.int32())
    size = (root / "bucket_id=0" / "part.parquet").stat().st_size
    _edit_manifest(root, lambda m: m["inventory"][0].update(size=size))


def _non_canonical(root: Path) -> None:
    (root / "bucket_id=0").rename(root / "bucket_id=00")
    _edit_manifest(
        root,
        lambda m: m["inventory"][0].update(relative_path="bucket_id=00/part.parquet"),
    )


MUTATIONS = {
    "hash_good": (False, None),
    "hash_missing_empty_bucket": (False, lambda r: (r / "bucket_id=1").rmdir()),
    "hash_extra_bucket_dir": (False, lambda r: (r / "bucket_id=2").mkdir()),
    "hash_extra_file": (
        False,
        lambda r: shutil.copy(
            r / "bucket_id=0" / "part.parquet", r / "bucket_id=1" / "x.parquet"
        ),
    ),
    "hash_wrong_key_schema": (False, _wrong_schema),
    "hash_non_canonical_path": (False, _non_canonical),
    "hash_wrong_storage_schema": (
        False,
        lambda r: _edit_manifest(
            r, lambda m: m.update(schema="easyicu.bucket_storage/0")
        ),
    ),
    "variable_good": (True, None),
    "variable_missing_empty_partition": (True, lambda r: shutil.rmtree(r / "var_id=1")),
    "variable_missing_empty_bucket": (
        True,
        lambda r: (r / "var_id=0" / "bucket_id=1").rmdir(),
    ),
    "variable_extra_partition": (
        True,
        lambda r: (r / "var_id=2" / "bucket_id=0").mkdir(parents=True),
    ),
    "variable_partition_id_out_of_order": (
        True,
        lambda r: _edit_manifest(r, lambda m: m["partitions"][1].update(id=5)),
    ),
    "manifest_missing": (False, lambda r: (r / "_BUCKET_MANIFEST.json").unlink()),
    "complete_marker_missing": (False, lambda r: (r / "_COMPLETE").unlink()),
    "hash_other_duckdb": (
        False,
        lambda r: _edit_manifest(r, lambda m: m["hash"].update(duckdb_version="0.0.0")),
    ),
    "derived_from_malformed": (
        False,
        lambda r: _edit_manifest(
            r, lambda m: m["derived_from"]["files"][0].update(sha256="x")
        ),
    ),
    "inventory_duplicate_without_size": (  # NB-PARITY-ORDER: a combined defect, same reason
        False,
        lambda r: _edit_manifest(
            r,
            lambda m: m["inventory"].append(
                {"relative_path": m["inventory"][0]["relative_path"], "row_count": 10}
            ),
        ),
    ),
    "variable_inventory_duplicate_without_size": (
        True,
        lambda r: _edit_manifest(
            r,
            lambda m: m["inventory"].append(
                {"relative_path": m["inventory"][0]["relative_path"], "row_count": 10}
            ),
        ),
    ),
    "derived_from_source_missing": (False, lambda r: (r.parent / "t.parquet").unlink()),
    "derived_from_source_changed": (
        False,
        lambda r: (r.parent / "t.parquet").write_bytes(b"changed source, other size"),
    ),
}


def _decision(fn, root):
    try:
        fn(root)
    except (sl.StorageIntegrityError, bc.StorageIntegrityError) as exc:
        return exc.reason
    return "ok"


def test_the_comparator_and_the_resolver_decide_every_install_alike(tmp_path):
    decisions = {}
    for name, (variable, mutate) in MUTATIONS.items():
        root = _install(tmp_path / name, variable)
        if mutate:
            mutate(root)
        resolver = _decision(sl.read_bucket_contract, root)
        comparator = _decision(bc.validate_contract, root)
        assert comparator == resolver, name
        decisions[name] = resolver
    assert decisions["hash_good"] == decisions["variable_good"] == "ok"
    assert decisions["hash_missing_empty_bucket"] == "bucket_modulus_mismatch"
    assert decisions["variable_missing_empty_partition"] == "bucket_partition_missing"
    assert decisions["variable_missing_empty_bucket"] == "bucket_modulus_mismatch"
    assert all(v != "ok" for k, v in decisions.items() if not k.endswith("_good"))


# Reached only with allow_legacy=True (the receipted AUMC itemid cache); a seal never accepts a legacy
# layout, and BC (like SL with allow_legacy=False) reports one as bucket_manifest_invalid.
LEGACY_ONLY = {"bucket_legacy_unverifiable": {"_legacy_manifest"}}


class UnsupportedReasonForm(AssertionError):
    """A raise or constructor form the collector cannot read a literal reason code from."""


def _resolver_reason_codes(
    source: str | None = None,
    entries=("read_bucket_contract", "_read_variable_contract"),
) -> dict[str, set[str]]:
    """Every reason code read_bucket_contract and _read_variable_contract can raise, derived from the
    storage_layout source: string first arguments of StorageIntegrityError-family constructors in those
    readers and every module function they (transitively) call. Code -> raising functions.

    Any form it cannot read a literal code from FAILS explicitly (never skipped): a family constructor
    whose code is not a string literal (a variable, an f-string, none), a family constructor reached as
    an attribute (module.Error(...)), an exception constructor imported from another module, or a raise
    of anything but a call (``raise err``); a bare re-raise is fine."""
    tree = ast.parse(source if source is not None else Path(sl.__file__).read_text())
    functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    errors = {"StorageIntegrityError"}
    classes = [n for n in tree.body if isinstance(n, ast.ClassDef)]
    while True:
        more = {
            c.name
            for c in classes
            if any(isinstance(b, ast.Name) and b.id in errors for b in c.bases)
        } - errors
        if not more:
            break
        errors |= more
    builtin_errors = {
        n
        for n in dir(builtins)
        if isinstance(getattr(builtins, n), type)
        and issubclass(getattr(builtins, n), BaseException)
    }
    codes: dict[str, set[str]] = {}
    todo, seen = list(entries), set()
    while todo:
        name = todo.pop()
        if name in seen:
            continue
        seen.add(name)
        for node in ast.walk(functions[name]):
            if (
                isinstance(node, ast.Raise)
                and node.exc is not None
                and not isinstance(node.exc, ast.Call)
            ):
                raise UnsupportedReasonForm(f"{name}: raise of a non-call")
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if isinstance(func, ast.Attribute) and (
                func.attr in errors or func.attr.endswith("Error")
            ):
                raise UnsupportedReasonForm(
                    f"{name}: attribute constructor {func.attr}"
                )
            if not isinstance(func, ast.Name):
                continue
            if func.id in functions:
                todo.append(func.id)
            if func.id in errors:
                first = node.args[0] if node.args else None
                if not (
                    isinstance(first, ast.Constant) and isinstance(first.value, str)
                ):
                    raise UnsupportedReasonForm(
                        f"{name}: non-literal code for {func.id}"
                    )
                codes.setdefault(first.value, set()).add(name)
            elif func.id.endswith("Error") and func.id not in builtin_errors:
                raise UnsupportedReasonForm(f"{name}: foreign constructor {func.id}")
    return codes


_FAMILY = (
    "class StorageIntegrityError(Exception):\n    pass\n"
    "class BucketLayoutError(StorageIntegrityError):\n    pass\n"
)


@pytest.mark.parametrize(
    "body",
    [
        "    code = 'bucket_x'\n    raise BucketLayoutError(code, root)\n",
        "    raise BucketLayoutError(f'bucket_{root}', root)\n",
        "    raise BucketLayoutError()\n",
        "    raise errors.BucketLayoutError('bucket_x', root)\n",
        "    raise OtherLayoutError('bucket_x', root)\n",
        "    err = BucketLayoutError('bucket_x', root)\n    raise err\n",
    ],
    ids=[
        "variable",
        "f-string",
        "no-code",
        "attribute",
        "cross-module",
        "raise-variable",
    ],
)
def test_the_reason_code_collector_fails_on_unsupported_forms(body):
    source = (
        "from elsewhere import OtherLayoutError, errors\n"
        + _FAMILY
        + "def _read_variable_contract(root):\n    raise BucketLayoutError('bucket_ok', root)\n"
        + "def read_bucket_contract(root):\n"
        + body
    )
    with pytest.raises(UnsupportedReasonForm):
        _resolver_reason_codes(source)


def test_the_reason_code_collector_reads_supported_forms():
    source = _FAMILY + (
        "def helper(root):\n    raise BucketLayoutError('bucket_helper', root) from None\n"
        "def _read_variable_contract(root):\n    raise ValueError('internal')\n"
        "def read_bucket_contract(root):\n"
        "    try:\n        helper(root)\n    except BucketLayoutError:\n        raise\n"
        "    raise BucketLayoutError('bucket_top', root)\n"
    )
    assert _resolver_reason_codes(source) == {
        "bucket_helper": {"helper"},
        "bucket_top": {"read_bucket_contract"},
    }


def test_every_resolver_reason_code_is_exercised_with_parity(tmp_path):
    codes = _resolver_reason_codes()
    assert len(codes) >= 10  # the walk found the readers' codes
    for code, raisers in LEGACY_ONLY.items():
        assert codes.get(code) == raisers, code
    covered = set()
    for name, (variable, mutate) in MUTATIONS.items():
        root = _install(tmp_path / name, variable)
        if mutate:
            mutate(root)
        resolver = _decision(sl.read_bucket_contract, root)
        assert _decision(bc.validate_contract, root) == resolver, name
        covered.add(resolver)
    uncovered = sorted(set(codes) - set(LEGACY_ONLY) - covered)
    assert not uncovered, f"resolver reason codes without a parity case: {uncovered}"


@pytest.mark.parametrize("variable", [False, True])
def test_twins_both_missing_their_empty_structure_are_invalid_not_identical(
    tmp_path, variable
):
    a, b = _install(tmp_path / "a", variable), _install(tmp_path / "b", variable)
    for root in (a, b):
        if variable:
            shutil.rmtree(root / "var_id=1")
        else:
            (root / "bucket_id=1").rmdir()
    result = _tool(
        "bucket_compare.py", str(a), str(b), "--out", str(tmp_path / "c.json")
    )
    out = json.loads((tmp_path / "c.json").read_text())
    assert result.returncode == 1 and out["verdict"] == "INVALID"
    assert out["integrity"]["reason"] == (
        "bucket_partition_missing" if variable else "bucket_modulus_mismatch"
    )


# ---- D29-PUBLISH: the final name appears read-only, verified, and unpublished until the commit


def _child(tmp_path: Path, body: str) -> subprocess.CompletedProcess:
    script = tmp_path / "child.py"
    script.write_text(
        "import importlib.util, json, os, sys\nfrom pathlib import Path\n"
        f"spec = importlib.util.spec_from_file_location('sr', {str(TOOLS / 'seal_rebuild.py')!r})\n"
        "sr = importlib.util.module_from_spec(spec); spec.loader.exec_module(sr)\n"
        + body
    )
    return subprocess.run(
        [sys.executable, "-B", str(script)], capture_output=True, text=True
    )


def test_a_crash_right_after_the_rename_leaves_a_read_only_root_refused_as_pending(
    tmp_path,
):
    from tests.support.seal_buckets import make_mirror

    mirror, sha = make_mirror(tmp_path)
    staging, target = tmp_path / ".root.seal_staging", tmp_path / "root"
    bucket = make_bucket(staging)
    sr.write_json(staging / sr.CONVERSION_MANIFEST, {})
    inner = {
        "verdict": "PASS",
        "seal_sha": sha,
        "tool": sr.sealed_tool_fingerprint(str(mirror), sha),
        "validator": sr.load_validator_from_git(str(mirror), sha).fingerprint,
        "conversion_manifest_sha256": sr.sha256_file(staging / sr.CONVERSION_MANIFEST),
        "links": ["t"],
        "link_set": sr.link_set(
            staging,
            {"t_bucket", sr.CONVERSION_MANIFEST, sr.IN_ROOT_RECEIPT},
            identities=True,
        ),
        "build": "build1",
        "db": "sic",
        "tables": {
            "t": {
                "manifest_sha256": sr.sha256_file(bucket / "_BUCKET_MANIFEST.json"),
                "rows": 40,
                "parts": 2,
            }
        },
        "bucket_set": sr.bucket_set(staging, ["t"]),
    }
    body = (
        "real = sr.rename_noreplace\n"
        "def crash(a, b):\n    real(a, b)\n    os._exit(79)\n"
        "sr.rename_noreplace = crash\n"
        f"sr.install(Path({str(staging)!r}), Path({str(target)!r}), {inner!r})\n"
    )
    assert _child(tmp_path, body).returncode == 79
    assert (
        target.is_dir() and sr.writable_entries(target) == []
    )  # never visible writable
    assert sr.pending_marker(target).is_file()
    # even a forged external PASS receipt does not publish it
    external = tmp_path / "receipt.json"
    external.write_text(
        json.dumps(
            dict(
                inner,
                install={
                    "read_only_verified": True,
                    "in_root_receipt_sha256": sr.sha256_file(
                        target / sr.IN_ROOT_RECEIPT
                    ),
                },
            )
        )
    )
    with pytest.raises(sr.SealError, match="root_pending"):
        sr.accept_root(target, external, sha, "build1", "sic", git_dir=str(mirror))
    sr.commit_install(target, sr.load_validator_from_git(str(mirror), sha))
    assert (
        sr.accept_root(target, external, sha, "build1", "sic", git_dir=str(mirror))[
            "verdict"
        ]
        == "PASS"
    )
    make_writable(target)


def test_a_crash_before_the_rename_never_shows_the_final_name(tmp_path):
    staging, target = tmp_path / ".root.seal_staging", tmp_path / "root"
    make_bucket(staging)
    body = (
        "def crash(*a, **k):\n    os._exit(79)\n"
        "sr.rename_noreplace = crash\n"
        f"sr.install(Path({str(staging)!r}), Path({str(target)!r}), {{'verdict': 'PASS'}})\n"
    )
    assert _child(tmp_path, body).returncode == 79
    assert not target.exists() and sr.writable_entries(staging) == []
    make_writable(staging)


# ---- D29-NOREPLACE: an atomic no-replace rename; no primitive, no publish; locks in b6_root


def test_the_rename_never_replaces_an_empty_directory_that_appeared(
    tmp_path, monkeypatch
):
    staging, target = tmp_path / ".root.seal_staging", tmp_path / "root"
    make_bucket(staging)
    real = sr.verify_installed

    def race(root, sha, **kw):
        real(root, sha, **kw)
        if root == staging:
            target.mkdir()  # appears between the checks and the rename

    monkeypatch.setattr(sr, "verify_installed", race)
    with pytest.raises(sr.SealError, match="target_exists"):
        sr.install(staging, target, {"verdict": "PASS"})
    assert target.is_dir() and os.listdir(target) == []
    assert (staging / sr.IN_ROOT_RECEIPT).is_file()
    make_writable(staging)


def test_without_a_no_replace_primitive_the_publish_is_refused(tmp_path, monkeypatch):
    staging, target = tmp_path / ".root.seal_staging", tmp_path / "root"
    make_bucket(staging)
    monkeypatch.setattr(sr, "_noreplace_primitive", lambda: None)
    with pytest.raises(sr.SealError, match="rename_noreplace_unavailable"):
        sr.install(staging, target, {"verdict": "PASS"})
    assert not target.exists() and staging.is_dir()
    make_writable(staging)


def test_the_run_lock_holds_in_the_target_space_across_work_roots(tmp_path):
    b6 = tmp_path / "b6"

    def seal(work):
        return SimpleNamespace(
            cfg={"work_root": str(tmp_path / work)},
            b6=b6,
            s9="f" * 9,
            mode="build1",
            dry_run=False,
            dbs=["synthetic"],
            target=lambda db: b6 / "target",
            farm_target=lambda db: None,
        )

    first = sr.acquire_run_lock(seal("work_a"))
    try:
        with pytest.raises(sr.SealError, match="seal_run_locked"):
            sr.acquire_run_lock(seal("work_b"))
    finally:
        first.close()
    sr.acquire_run_lock(seal("work_b")).close()


# ---- D29-CONVERSION-RACE: the bytes that are verified are the bytes that are copied


def test_a_manifest_update_after_the_hash_is_never_copied(tmp_path, monkeypatch):
    old, new = tmp_path / "old", tmp_path / "new"
    old.mkdir()
    derived = tmp_path / "derived.json"
    derived.write_text("{}")
    manifest = old / sr.CONVERSION_MANIFEST
    planned = {
        "data_path": str(old),
        "derived_from": {"path": str(derived), "sha256": sr.sha256_file(derived)},
        "tables": [{"name": "t", "rows": 20}],
    }
    manifest.write_text(json.dumps(planned))
    pin = sr.manifest_source(manifest)
    changed = dict(planned, tables=[{"name": "t", "rows": 40}])
    fired = []

    def after_first_read(fn):
        def wrapped(path, *a, **k):
            result = fn(path, *a, **k)
            if Path(path) == manifest and not fired:
                fired.append(True)
                manifest.write_text(
                    json.dumps(changed)
                )  # lands right after the first read
            return result

        return wrapped

    monkeypatch.setattr(sr, "sha256_file", after_first_read(sr.sha256_file))
    monkeypatch.setattr(Path, "read_bytes", after_first_read(Path.read_bytes))
    monkeypatch.setattr(Path, "read_text", after_first_read(Path.read_text))
    dest = tmp_path / "dest.json"
    try:
        sr.write_conversion_manifest(manifest, dest, old, new, pin)
    except sr.SealError as exc:
        assert exc.reason == "conversion_manifest_source_changed"
        return
    assert fired
    assert json.loads(dest.read_text()) == dict(planned, data_path=str(new))


# ---- D29-FOOTER-ALIAS: an alias never starves the formal bucket path of its footers


def test_an_alias_of_a_bucket_keeps_the_bucket_footers(tmp_path):
    bucket = make_bucket(tmp_path)
    os.symlink(bucket, tmp_path / "aaa_alias")  # sorts (and is walked) before t_bucket
    os.symlink(tmp_path, bucket / "bucket_id=0" / "back")  # a loop still ends
    result = footer_scan.scan(tmp_path)
    assert result["ok"], result["failures"]
    assert result["buckets"]["t_bucket"]["rows_footer"] == 40


def test_bucket_metadata_outside_the_allowed_roots_is_not_read(tmp_path):
    root = tmp_path / "root"
    bucket = make_bucket(root)
    outside = tmp_path / "outside_manifest.json"
    (bucket / "_BUCKET_MANIFEST.json").rename(outside)
    os.symlink(outside, bucket / "_BUCKET_MANIFEST.json")
    result = footer_scan.scan(root)
    assert not result["ok"]
    assert {
        "path": "t_bucket/_BUCKET_MANIFEST.json",
        "reason": "path_escapes_allowed_roots",
    } in result["failures"]


# ---- D29-BOOT-HOOK: no start-up hook runs; every easyicu/duckdb/pyarrow module's origin is asserted


def _clone(tmp_path: Path, builder_text: str) -> tuple[Path, str, Path]:
    clone = tmp_path / "clone"
    pkg = clone / "src" / "easyicu"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "storage_layout.py").write_text("FLAG = 'clone'\n")
    builder = clone / "scripts" / "builder.py"
    builder.parent.mkdir()
    builder.write_text(builder_text)
    env = {
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@t",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@t",
        "PATH": os.environ["PATH"],
    }
    for args in (["init", "-q"], ["add", "-A"], ["commit", "-q", "-m", "c"]):
        subprocess.run(
            ["git", *args], cwd=clone, check=True, env=env, capture_output=True
        )
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=clone, capture_output=True, text=True
    ).stdout.strip()
    return clone, sha, builder


def _boot(tmp_path, clone, sha, builder, site, cwd=None):
    origin = tmp_path / "origin.json"
    env = {
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "PYTHONPATH": str(clone / "src"),
        "PYTHONDONTWRITEBYTECODE": "1",
        "SEAL_CLONE": str(clone),
        "SEAL_SHA": sha,
        "SEAL_ORIGIN_OUT": str(origin),
        "SEAL_SITE_PACKAGES": str(site),
    }
    # the tool's own launch line (a2aea5db4 had none: `python -c BOOT <builder>`)
    launch = getattr(
        sr, "builder_argv", lambda py, b: [py, "-c", sr.BUILDER_BOOT, str(b)]
    )
    done = subprocess.run(
        launch(sys.executable, builder),
        cwd=cwd or tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )
    return done, json.loads(origin.read_text())


def test_site_hooks_cannot_preload_a_foreign_easyicu_submodule(tmp_path):
    foreign = tmp_path / "foreign_storage_layout.py"
    foreign.write_text("FLAG = 'foreign'\n")
    preload = (
        "import importlib.util, sys; "
        f"s = importlib.util.spec_from_file_location('easyicu.storage_layout', {str(foreign)!r}); "
        "m = importlib.util.module_from_spec(s); s.loader.exec_module(m); "
        "sys.modules['easyicu.storage_layout'] = m"
    )
    clone, sha, builder = _clone(
        tmp_path,
        "from pathlib import Path\nfrom easyicu.storage_layout import FLAG\n"
        f"Path({str(tmp_path / 'result')!r}).write_text(FLAG)\n",
    )
    site = (
        tmp_path / "site"
    )  # stands for the interpreter's site-packages, carrying both hooks
    site.mkdir()
    (site / "hook.pth").write_text(preload + "\n")
    (site / "sitecustomize.py").write_text(preload.replace("; ", "\n") + "\n")
    (clone / "src" / "sitecustomize.py").write_text(preload.replace("; ", "\n") + "\n")
    done, origin = _boot(tmp_path, clone, sha, builder, site)
    assert done.returncode == 0, done.stderr
    assert (tmp_path / "result").read_text() == "clone"
    assert sr.origin_ok(origin, clone, sha)
    assert origin["modules"]["easyicu.storage_layout"] == os.path.realpath(
        clone / "src" / "easyicu" / "storage_layout.py"
    )


def test_a_foreign_easyicu_or_pyarrow_module_loaded_any_other_way_fails_the_build(
    tmp_path,
):
    foreign = tmp_path / "foreign_storage_layout.py"
    foreign.write_text("FLAG = 'foreign'\n")
    clone, sha, builder = _clone(
        tmp_path,
        "import importlib.util, sys\n"
        f"s = importlib.util.spec_from_file_location('easyicu.storage_layout', {str(foreign)!r})\n"
        "m = importlib.util.module_from_spec(s); s.loader.exec_module(m)\n"
        "sys.modules['easyicu.storage_layout'] = m\n"
        "import pyarrow\n",  # resolves to the shadow pyarrow.py beside the builder
    )
    (builder.parent / "pyarrow.py").write_text("")
    site = Path(pa.__file__).parent.parent  # this interpreter's site-packages
    done, origin = _boot(tmp_path, clone, sha, builder, site)
    assert done.returncode == sr.ORIGIN_EXIT
    assert "builder_import_origin_mismatch" in done.stderr
    assert origin["mismatched"] == ["easyicu.storage_layout", "pyarrow"]
    assert not sr.origin_ok(origin, clone, sha)


# ---- D34-BOOT-ISOLATION (delta34s): neither the current directory nor PYTHONPATH reaches the boot


def test_boot_cwd_stdlib_shadow(tmp_path):
    """Reviewer probe boot_cwd_stdlib_shadow: a data-root entry json.py (a link the containment rules
    allow) in the builder's cwd must never run before the path is fixed."""
    mark = tmp_path / "shadow_executed"
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    poison = allowed / "json.py"
    poison.write_text(
        f"open({str(mark)!r}, 'w').write('executed')\n"
        "import os, sys, importlib\n"
        "here = os.path.realpath(os.getcwd())\n"
        "sys.path[:] = [x for x in sys.path if os.path.realpath(x or os.getcwd()) != here]\n"
        "sys.modules.pop('json', None)\n"
        "globals().update(importlib.import_module('json').__dict__)\n"
    )
    stage = tmp_path / "stage"
    stage.mkdir()
    sr.materialize(
        [("link", "json.py", str(poison))], stage / "data", allowed, stage / "final"
    )
    sr.check_links(stage / "data", [allowed, stage])
    clone, sha, builder = _clone(tmp_path, "import json\n")
    done, origin = _boot(
        tmp_path,
        clone,
        sha,
        builder,
        Path(pa.__file__).parent.parent,
        cwd=stage / "data",
    )
    assert done.returncode == 0, done.stderr
    assert not mark.exists()
    assert sr.origin_ok(origin, clone, sha)


def test_site_packages_environment_shadow(tmp_path, monkeypatch):
    """Reviewer probe site_packages_environment_shadow: the non-venv site-packages lookup inherits
    nothing; a foreign sysconfig on PYTHONPATH cannot pick the site-packages directory."""
    poison, fake = tmp_path / "poison", tmp_path / "fake_site"
    poison.mkdir()
    fake.mkdir()
    (poison / "sysconfig.py").write_text(
        "def get_paths():\n    return {'purelib': " + repr(str(fake)) + "}\n"
    )
    monkeypatch.setenv("PYTHONPATH", str(poison))
    base_python = Path(sys.base_prefix) / "bin" / "python3"
    assert not (
        base_python.parent.parent / "pyvenv.cfg"
    ).exists()  # the non-venv branch
    selected = sr.site_packages(str(base_python))
    assert selected != str(fake)
    assert Path(selected).is_dir()
