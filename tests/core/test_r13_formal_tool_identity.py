"""Formal tool execution identity: copy-then-exec, the boot, exit status (delta32p ruling).

The wrapper (tools/formal/bench7v3f.sh) stages every tool of the sealed
manifest (tools/formal/FORMAL_TOOLS.json) by content into the run's private
``.formal_tools`` and executes only those copies; every Python step runs
under the staged boot (tools/formal/formal_boot.py, ``python -I -S -P``).
The receipt (tools/formal_run_receipt.py) verifies the staged set against
the seal and each step's import-origin record. Nothing here reads processes,
links or timestamps (the fd-255 ancestry, link following and ctime/START
heuristics were removed: B32-FD255, B32-DRIVER-LINK, B32-SHADOW, B32-START).
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import sysconfig
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
TOOL = REPO / "tools" / "formal_run_receipt.py"
FORMAL = REPO / "tools" / "formal"
WRAPPER = FORMAL / "bench7v3f.sh"
BOOT = FORMAL / "formal_boot.py"
SPEC = importlib.util.spec_from_file_location("formal_run_receipt_tool_identity", TOOL)
assert SPEC and SPEC.loader
receipt = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = receipt
SPEC.loader.exec_module(receipt)

INSTALLED = {  # sha256 of the installed server copies these were committed from
    "bench7v3e.sh": "3222d90bfc20a913bda77b48e862b29f559237142c236a69b7d5c5900760b9dd",
    "bench_local.py": "523190f23d9703e3d7377604f3389f9e3e1a1adf71c1934668c78891486ff6a9",
    "inputs_bench_local.py": "f06eca62a1fbac7695c3abc4f4f51dfc27bda27eee4348148480cf0c1503e1ca",
    "memgate.py": "5a86ade410ec74babb077a447996df2113fa73132b467193cb13d3d212c8f23e",
}
TOOL_SET = {"bench7v3f.sh", "formal_boot.py", "bench_local.py", "inputs_bench_local.py",
            "memgate.py", "formal_run_receipt.py", "formal_declared_absent.json"}


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(cwd), *args], check=True, capture_output=True,
                          text=True).stdout.strip()


def _manifest() -> dict:
    return json.loads((FORMAL / "FORMAL_TOOLS.json").read_text())


@contextlib.contextmanager
def _writable(folder: Path):
    os.chmod(folder, 0o700)
    try:
        yield
    finally:
        os.chmod(folder, 0o500)


# The committed tool set.

@pytest.mark.parametrize("name", sorted(INSTALLED))
def test_the_committed_copies_are_byte_identical_to_the_installed_ones(name):
    assert _sha((FORMAL / name).read_bytes()) == INSTALLED[name]


def test_the_sealed_manifest_is_the_formal_tool_set_at_its_current_bytes():
    manifest = _manifest()

    assert manifest["schema"] == "easyicu.formal_tools/1"
    assert {t["name"] for t in manifest["tools"]} == TOOL_SET  # memgate.py included
    stale = {t["name"]: _sha((REPO / t["path"]).read_bytes()) for t in manifest["tools"]
             if _sha((REPO / t["path"]).read_bytes()) != t["sha256"]}
    assert not stale, f"update tools/formal/FORMAL_TOOLS.json: {stale}"
    assert all(Path(t["path"]).name == t["name"] for t in manifest["tools"])


# A seal tree and a run staged from it by the wrapper's own stage_tools.

def _seal_files(**override: bytes) -> dict:
    files = {t["path"]: (REPO / t["path"]).read_bytes() for t in _manifest()["tools"]}
    files["src/easyicu/__init__.py"] = b""
    for name, data in override.items():
        (key,) = [k for k in files if Path(k).name == name]
        files[key] = data
    manifest = _manifest()
    for tool in manifest["tools"]:
        tool["sha256"] = _sha(files[tool["path"]])
    files["tools/formal/FORMAL_TOOLS.json"] = json.dumps(manifest).encode()
    return files


def _seal_repo(tmp_path: Path, files: dict, name: str = "seal") -> tuple[Path, str]:
    repo = tmp_path / name
    for relative, data in files.items():
        (repo / relative).parent.mkdir(parents=True, exist_ok=True)
        (repo / relative).write_bytes(data)
        if relative.endswith(".sh"):
            os.chmod(repo / relative, 0o700)
    _git(tmp_path, "init", "-q", str(repo))
    _git(repo, "add", "-A")
    _git(repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "seal")
    return repo, _git(repo, "rev-parse", "HEAD")


def _bash(script: str, *args: str, env: dict | None = None, cwd: Path | None = None):
    return subprocess.run(["bash", "-c", script, "_", *args], capture_output=True, text=True,
                          env=env, cwd=cwd)


def _stage(run: Path, repo: Path, sha: str) -> subprocess.CompletedProcess:
    """The wrapper's own stage_tools (the script sourced: its functions only)."""
    run.mkdir(exist_ok=True)
    return _bash('source "$1"; R=$2; FT=$2/.formal_tools; K=$3; SHA=$4; PY=$5; stage_tools',
                 str(WRAPPER), str(run), str(repo / ".git"), sha, sys.executable)


@pytest.fixture
def staged(tmp_path, monkeypatch):
    repo, sha = _seal_repo(tmp_path, _seal_files())
    run = tmp_path / "run"
    done = _stage(run, repo, sha)
    assert done.returncode == 0, done.stderr
    folder = run / ".formal_tools"
    # The receipt under test runs as the staged copy, as in a formal run.
    monkeypatch.setattr(receipt, "__file__", str(folder / "formal_run_receipt.py"))
    monkeypatch.setattr(receipt, "APPROVED_DECLARED_ABSENT_FILE",
                        folder / "formal_declared_absent.json")
    yield types.SimpleNamespace(repo=repo, sha=sha, run=run, folder=folder)
    os.chmod(folder, 0o700)


def _check(st) -> list:
    return receipt.formal_tool_scripts(st.sha, {"import_path": str(st.repo / "src" / "easyicu")},
                                       st.run, receipt.approved_declared_absent()[1])


def _refused(st, code: str, reason: str | None = None):
    with pytest.raises(receipt.ReceiptError) as error:
        _check(st)
    assert error.value.code == code, error.value
    if reason is not None:
        assert error.value.details["reason"] == reason
    return error.value


def test_staging_copies_by_content_read_only_and_records_the_shas(staged):
    folder = staged.folder

    assert {p.name for p in folder.iterdir()} == TOOL_SET
    assert not os.stat(folder).st_mode & 0o222  # the dir itself: nothing renamed in
    for path in folder.iterdir():
        assert path.is_file() and not path.is_symlink() and not os.stat(path).st_mode & 0o222
    staged_shas = dict(line.split("  ")[::-1]
                       for line in (staged.run / "FORMAL_TOOLS.staged").read_text().splitlines())
    assert staged_shas == {t["name"]: _sha((REPO / t["path"]).read_bytes())
                           for t in _manifest()["tools"]}


def test_a_staged_run_passes_however_fast_it_was_installed(staged):
    # B32-START: staged and checked within the same second; nothing compares times.
    records = _check(staged)

    assert {r["name"] for r in records} == TOOL_SET
    assert all(r["path"] == str(staged.folder / r["name"]) for r in records)
    assert {r["name"]: r["sha256"] for r in records}["memgate.py"] == INSTALLED["memgate.py"]


def test_the_identity_reads_no_process_tree(staged, monkeypatch):
    # B32-FD255: a shell holding the wrapper open proves nothing; only the staged
    # copies count, and a receipt not running as its staged copy is refused.
    source = TOOL.read_text()
    assert "/proc" not in source and "executing_wrapper" not in source
    assert "st_ctime" not in source and "started_ns" not in source
    monkeypatch.setattr(receipt, "__file__", str(TOOL))

    error = _refused(staged, "formal_tool_script_mismatch", "not_staged")
    assert error.details["role"] == "formal_run_receipt.py"


def test_a_staged_driver_swapped_for_a_link_is_refused(staged, tmp_path):
    # B32-DRIVER-LINK: a link to the very same bytes is still not the staged copy.
    elsewhere = tmp_path / "bench_local.py"
    elsewhere.write_bytes((staged.folder / "bench_local.py").read_bytes())
    with _writable(staged.folder):
        (staged.folder / "bench_local.py").unlink()
        os.symlink(elsewhere, staged.folder / "bench_local.py")

    error = _refused(staged, "formal_tool_script_mismatch", "not_a_staged_copy")
    assert error.details["role"] == "bench_local.py"


def test_a_hard_linked_staged_copy_is_refused(staged, tmp_path):
    os.link(staged.folder / "bench_local.py", tmp_path / "alias.py")

    _refused(staged, "formal_tool_script_mismatch", "not_a_staged_copy")


def test_a_linked_tools_dir_is_refused(staged, tmp_path):
    real_dir = tmp_path / "elsewhere"
    os.chmod(staged.folder, 0o700)  # renaming a directory needs write on it (macOS)
    os.rename(staged.folder, real_dir)
    os.chmod(real_dir, 0o500)
    os.symlink(real_dir, staged.folder)
    try:
        _refused(staged, "formal_tools_dir_invalid")
    finally:
        os.unlink(staged.folder)
        os.chmod(real_dir, 0o700)
        os.rename(real_dir, staged.folder)


def test_a_writable_tools_dir_is_refused(staged):
    os.chmod(staged.folder, 0o700)
    try:
        _refused(staged, "formal_tools_dir_invalid")
    finally:
        os.chmod(staged.folder, 0o500)


@pytest.mark.parametrize("extra", ["bench_local.pyc", "json.pyc", "__pycache__", "json.py"])
def test_anything_beside_the_staged_tools_is_refused(staged, extra):
    # B32-SHADOW: a sourceless .pyc, a cache dir or a shadow module next to the copies.
    with _writable(staged.folder):
        if extra == "__pycache__":
            (staged.folder / extra).mkdir()
        else:
            (staged.folder / extra).write_bytes(b"")

    error = _refused(staged, "formal_tools_dir_unexpected")
    assert error.details["unexpected"] == [extra]


@pytest.mark.parametrize("name", ["bench_local.py", "memgate.py", "formal_boot.py"])
def test_a_staged_tool_removed_before_the_receipt_is_refused(staged, name):
    # B32-SHADOW (removed .py): the copies must still all be there at the end.
    with _writable(staged.folder):
        (staged.folder / name).unlink()

    error = _refused(staged, "formal_tools_dir_unexpected")
    assert error.details["missing"] == [name]


def test_a_staged_copy_changed_after_staging_is_refused(staged):
    copy = staged.folder / "bench_local.py"
    data = copy.read_bytes()
    with _writable(staged.folder):
        copy.unlink()
        copy.write_bytes(data.replace(b"20261104", b"20261105", 1))
        os.chmod(copy, 0o444)

    error = _refused(staged, "formal_tool_script_mismatch", "changed_since_staged")
    assert error.details["staged_sha256"] == error.details["seal_sha256"] != error.details["sha256"]


def test_a_staged_record_that_is_not_the_seal_is_refused(staged):
    record = staged.run / "FORMAL_TOOLS.staged"
    os.chmod(record, 0o644)
    lines = record.read_text().splitlines()
    lines = [("0" * 64 + "  bench_local.py") if line.endswith("  bench_local.py") else line
             for line in lines]
    record.write_text("\n".join(lines) + "\n")

    _refused(staged, "formal_tool_script_mismatch", "staged_not_seal")


def test_a_sealed_manifest_that_is_not_its_tree_is_refused(tmp_path, monkeypatch):
    files = _seal_files()
    manifest = json.loads(files["tools/formal/FORMAL_TOOLS.json"])
    manifest["tools"][0]["sha256"] = "0" * 64
    files["tools/formal/FORMAL_TOOLS.json"] = json.dumps(manifest).encode()
    repo, sha = _seal_repo(tmp_path, files)

    with pytest.raises(receipt.ReceiptError) as error:
        receipt.sealed_tool_manifest(repo, sha)
    assert error.value.code == "formal_tool_script_mismatch"
    assert error.value.details["reason"] == "manifest_not_seal"


def test_staging_refuses_a_copy_that_differs_from_the_manifest(tmp_path):
    files = _seal_files()
    manifest = json.loads(files["tools/formal/FORMAL_TOOLS.json"])
    manifest["tools"][2]["sha256"] = "0" * 64
    files["tools/formal/FORMAL_TOOLS.json"] = json.dumps(manifest).encode()
    repo, sha = _seal_repo(tmp_path, files)

    done = _stage(tmp_path / "run", repo, sha)
    assert done.returncode != 0 and "formal_tool_copy_mismatch" in done.stderr


def test_staging_refuses_an_existing_tools_dir(tmp_path):
    repo, sha = _seal_repo(tmp_path, _seal_files())
    (tmp_path / "run" / ".formal_tools").mkdir(parents=True)

    assert _stage(tmp_path / "run", repo, sha).returncode != 0


# The wrapper's stage 1 under every launch mode execs the same staged copy.

_STUB_STAGE2 = b"""#!/bin/bash
D=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
echo "${BENCH7_STAGE:-} $D/$(basename "${BASH_SOURCE[0]}")" > "$D/../stage2.txt"
"""


@pytest.fixture
def bench_host(tmp_path):
    """A bench root, a mirror holding a seal whose wrapper copy is a recording stub."""
    repo, sha = _seal_repo(tmp_path, _seal_files(**{"bench7v3f.sh": _STUB_STAGE2}))
    mirror = tmp_path / "mirror.git"
    subprocess.run(["git", "clone", "-q", "--bare", str(repo), str(mirror)], check=True)
    _git(mirror, "config", "uploadpack.allowAnySHA1InWant", "true")
    root = tmp_path / "bench"
    root.mkdir()
    shims = tmp_path / "shims"
    shims.mkdir()
    if shutil.which("flock") is None:  # util-linux on the server; a no-op lock here
        (shims / "flock").write_text("#!/bin/sh\nexit 0\n")
        os.chmod(shims / "flock", 0o700)
    env = {"PATH": f"{shims}:{os.environ['PATH']}", "HOME": str(tmp_path),
           "BENCH7_ROOT": str(root), "BENCH7_MIRROR": str(mirror), "BENCH7_PY": sys.executable,
           "GIT_CONFIG_NOSYSTEM": "1"}
    return types.SimpleNamespace(sha=sha, root=root, env=env, tmp=tmp_path)


_LAUNCH = {
    "exec": lambda w, a: ["bash", "-c", 'exec bash "$0" "$@"', w, *a],
    "sh_c": lambda w, a: ["sh", "-c", 'bash "$0" "$@"', w, *a],
    "nohup": lambda w, a: ["nohup", "bash", w, *a],
    "setsid": lambda w, a: ["setsid", "bash", w, *a],
}


@pytest.mark.parametrize("mode", sorted(_LAUNCH))
def test_every_launch_mode_execs_the_same_staged_copy(bench_host, mode):
    if mode == "setsid" and shutil.which("setsid") is None:
        pytest.skip("setsid (util-linux) is not installed here")
    tag = f"t_{mode}"
    argv = _LAUNCH[mode](str(WRAPPER), [tag, bench_host.sha, "sample", "sic:/nowhere:188-189"])

    done = subprocess.run(argv, capture_output=True, text=True, env=bench_host.env,
                          cwd=bench_host.tmp)

    run = bench_host.root / f"run_{tag}"
    assert done.returncode == 0, done.stderr
    staged_copy = os.path.realpath(run / ".formal_tools" / "bench7v3f.sh")
    assert (run / "stage2.txt").read_text().split() == ["2", staged_copy]
    assert _sha(Path(staged_copy).read_bytes()) == _sha(_STUB_STAGE2)


def test_stage_one_refuses_an_existing_run_root(bench_host):
    (bench_host.root / "run_t_again").mkdir()

    done = subprocess.run(["bash", str(WRAPPER), "t_again", bench_host.sha, "sample",
                           "sic:/nowhere:188-189"], capture_output=True, text=True,
                          env=bench_host.env, cwd=bench_host.tmp)

    assert done.returncode == 2 and "created fresh" in done.stderr
    assert not (bench_host.root / "run_t_again" / "stage2.txt").exists()


def test_stage_two_runs_only_as_the_staged_copy(bench_host):
    argv = ["bash", str(WRAPPER), "t_two", bench_host.sha, "sample", "sic:/nowhere:188-189"]
    assert subprocess.run(argv, env=bench_host.env, cwd=bench_host.tmp).returncode == 0

    # The installed wrapper claiming to be stage 2 is refused.
    done = subprocess.run(argv, capture_output=True, text=True, cwd=bench_host.tmp,
                          env={**bench_host.env, "BENCH7_STAGE": "2"})
    assert done.returncode == 2 and "stage 2 runs only as" in done.stderr


def test_a_stage_one_copy_mismatch_execs_nothing(tmp_path):
    files = _seal_files(**{"bench7v3f.sh": _STUB_STAGE2})
    manifest = json.loads(files["tools/formal/FORMAL_TOOLS.json"])
    manifest["tools"][0]["sha256"] = "0" * 64  # the wrapper's own entry
    files["tools/formal/FORMAL_TOOLS.json"] = json.dumps(manifest).encode()
    repo, sha = _seal_repo(tmp_path, files)
    mirror = tmp_path / "mirror.git"
    subprocess.run(["git", "clone", "-q", "--bare", str(repo), str(mirror)], check=True)
    _git(mirror, "config", "uploadpack.allowAnySHA1InWant", "true")
    root = tmp_path / "bench"
    root.mkdir()
    shims = tmp_path / "shims"
    shims.mkdir()
    (shims / "flock").write_text("#!/bin/sh\nexit 0\n")
    os.chmod(shims / "flock", 0o700)
    env = {"PATH": f"{shims}:{os.environ['PATH']}", "HOME": str(tmp_path),
           "BENCH7_ROOT": str(root), "BENCH7_MIRROR": str(mirror), "BENCH7_PY": sys.executable}

    done = subprocess.run(["bash", str(WRAPPER), "t_bad", sha, "sample", "sic:/x:188-189"],
                          capture_output=True, text=True, env=env, cwd=tmp_path)

    assert done.returncode == 2
    assert (root / "run_t_bad" / "RUN_INVALID").read_text().strip() == "formal_tools_not_staged"
    assert not (root / "run_t_bad" / "stage2.txt").exists()


# Exit status: DONE never masks a failure.

def _finish(tmp_path: Path, status: str, call: str) -> subprocess.CompletedProcess:
    run = tmp_path / "run"
    run.mkdir(exist_ok=True)
    (run / "status").write_text(status)
    return _bash(f'source "$1"; R=$2; {call}; echo continued', str(WRAPPER), str(run))


def test_a_refused_receipt_invalidates_the_run_and_exits_non_zero(tmp_path):
    done = _finish(tmp_path, "sic exit=0\nreceipt exit=2\n", "finish_receipt 2")

    assert done.returncode == 3 and "continued" not in done.stdout
    assert (tmp_path / "run" / "RUN_INVALID").read_text().strip() == "receipt_refused exit=2"
    assert (tmp_path / "run" / "status").read_text().splitlines()[-1] == "DONE"


def test_an_accepted_receipt_continues(tmp_path):
    done = _finish(tmp_path, "sic exit=0\nreceipt exit=0\n", "finish_receipt 0")

    assert done.returncode == 0 and "continued" in done.stdout
    assert not (tmp_path / "run" / "RUN_INVALID").exists()


@pytest.mark.parametrize("status,code", [
    ("inputs_pre exit=0\nsic exit=0\ninputs_post exit=0\n", 0),
    ("inputs_pre exit=0\nsic exit=1\ninputs_post exit=0\n", 1),
    ("sic exit=refused_window_closed\n", 1),
    ("sic exit=formal_tool_changed\n", 1),
    ("sic exit=0\nreceipt exit=97\n", 1),
])
def test_done_never_masks_a_failed_step(tmp_path, status, code):
    done = _finish(tmp_path, status, "finish_run")

    assert done.returncode == code
    assert (tmp_path / "run" / "status").read_text().splitlines()[-1] == "DONE"


# The boot.

def _site() -> str:
    return os.path.realpath(sysconfig.get_paths()["purelib"])


@pytest.fixture
def boot_env(tmp_path):
    clone, sha = _seal_repo(tmp_path, {"src/easyicu/__init__.py": b"VALUE = 1\n"}, "clone")
    tools = tmp_path / "tools"
    tools.mkdir()
    shutil.copy(BOOT, tools / "formal_boot.py")
    tmpdir = tmp_path / "private_tmp"
    tmpdir.mkdir()
    return types.SimpleNamespace(clone=clone, sha=sha, tools=tools, tmp=tmp_path,
                                 out=tmp_path / "origin.json", tmpdir=tmpdir)


def _boot(b, script: str, body: str, *, sha: str | None = None, site: str | None = None,
          env: dict | None = None, cwd: Path | None = None, place: Path | None = None):
    target = (place or b.tools) / script
    target.write_text(body)
    argv = [sys.executable, "-I", "-S", "-P", str(b.tools / "formal_boot.py"), str(b.tools),
            str(b.clone), sha or b.sha, site or _site(), str(b.out), str(target), "a1"]
    base = {"PATH": os.environ["PATH"], "TMPDIR": str(b.tmpdir)}
    return subprocess.run(argv, capture_output=True, text=True, cwd=cwd or b.tmp,
                          env={**base, **env} if env is not None else base)


def test_the_boot_runs_a_staged_script_with_clean_origins(boot_env):
    done = _boot(boot_env, "bench_local.py",
                 "import sys, json, easyicu, pyarrow\nassert sys.argv[1:] == ['a1']\nsys.exit(5)\n")

    record = json.loads(boot_env.out.read_text())
    assert done.returncode == 5, done.stderr
    assert record["complete"] is True and record["mismatched"] == []
    assert record["head"] == boot_env.sha
    assert record["path"] == os.path.realpath(boot_env.clone / "src" / "easyicu" / "__init__.py")
    assert record["script"] == str(boot_env.tools / "bench_local.py")
    assert all(f.startswith(_site() + os.sep) for n, f in record["modules"].items()
               if n.split(".")[0] == "pyarrow" and f)


def test_a_module_from_outside_the_four_roots_is_exit_97(boot_env):
    evil = boot_env.tmp / "evil"
    evil.mkdir()
    (evil / "evil_mod.py").write_text("X = 1\n")

    done = _boot(boot_env, "bench_local.py",
                 f"import sys\nsys.path.append({str(evil)!r})\nimport evil_mod\n")

    assert done.returncode == 97
    assert "evil_mod" in json.loads(boot_env.out.read_text())["mismatched"]


def test_a_sourceless_pyc_in_the_tools_dir_is_exit_97(boot_env):
    stub = boot_env.tmp / "shadowmod.py"
    stub.write_text("X = 1\n")
    import py_compile

    py_compile.compile(str(stub), cfile=str(boot_env.tools / "shadowmod.pyc"))

    done = _boot(boot_env, "bench_local.py", "import shadowmod\n")

    assert done.returncode == 97
    assert "shadowmod" in json.loads(boot_env.out.read_text())["mismatched"]


def test_a_stdlib_shadow_in_the_cwd_never_runs(boot_env):
    # delta34s probe: json.py in the cwd (or staging) must not be imported.
    cwd = boot_env.tmp / "cwd"
    cwd.mkdir()
    (cwd / "json.py").write_text("raise SystemExit(42)\n")
    (cwd / "decimal.py").write_text("raise SystemExit(43)\n")

    done = _boot(boot_env, "bench_local.py", "import json, decimal\n", cwd=cwd)

    assert done.returncode == 0, done.stderr
    record = json.loads(boot_env.out.read_text())
    assert record["mismatched"] == [] and str(cwd) not in json.dumps(record["sys_path"])


def test_a_pythonpath_site_shadow_never_runs(boot_env):
    # delta34s probe: PYTHONPATH/PYTHONHOME/PYTHONSTARTUP are ignored (-I).
    shadow = boot_env.tmp / "pp"
    shadow.mkdir()
    (shadow / "json.py").write_text("raise SystemExit(42)\n")
    (shadow / "pyarrow.py").write_text("raise SystemExit(44)\n")
    (shadow / "startup.py").write_text("raise SystemExit(45)\n")
    env = {"PATH": os.environ["PATH"], "PYTHONPATH": str(shadow),
           "PYTHONSTARTUP": str(shadow / "startup.py")}

    done = _boot(boot_env, "bench_local.py", "import json, pyarrow\n", env=env)

    assert done.returncode == 0, done.stderr
    assert str(shadow) not in json.dumps(json.loads(boot_env.out.read_text())["sys_path"])


def test_no_pth_line_of_the_named_site_packages_runs(boot_env):
    site = boot_env.tmp / "site"
    site.mkdir()
    (site / "zz_hook.pth").write_text("import os; os.environ['PTH_RAN'] = '1'\n")

    done = _boot(boot_env, "bench_local.py",
                 "import os, sys\nsys.exit(1 if os.environ.get('PTH_RAN') else 0)\n",
                 site=str(site))

    assert done.returncode == 0, done.stderr


def test_a_script_outside_the_tools_dir_is_exit_97_before_it_runs(boot_env):
    marker = boot_env.tmp / "ran"
    done = _boot(boot_env, "bench_local.py", f"open({str(marker)!r}, 'w').close()\n",
                 place=boot_env.tmp)

    assert done.returncode == 97 and not marker.exists()


def test_a_linked_script_in_the_tools_dir_is_exit_97(boot_env):
    real = boot_env.tmp / "real.py"
    real.write_text("pass\n")
    os.symlink(real, boot_env.tools / "bench_local.py")
    argv = [sys.executable, "-I", "-S", "-P", str(boot_env.tools / "formal_boot.py"),
            str(boot_env.tools), str(boot_env.clone), boot_env.sha, _site(), str(boot_env.out),
            str(boot_env.tools / "bench_local.py")]

    env = {"PATH": os.environ["PATH"], "TMPDIR": str(boot_env.tmpdir)}
    assert subprocess.run(argv, capture_output=True, env=env).returncode == 97


def test_a_clone_not_at_the_seal_is_exit_97(boot_env):
    done = _boot(boot_env, "bench_local.py", "pass\n", sha="0" * 40)

    assert done.returncode == 97


# The receipt's import-origin check.

def _origin(run: Path, tools: Path, src: Path, sha: str, script: str, **change) -> dict:
    return {"complete": True, "mismatched": [], "head": sha, "src": os.path.realpath(src),
            "tools": os.path.realpath(tools), "path": os.path.realpath(src / "easyicu" / "x.py"),
            "script": os.path.join(os.path.realpath(tools), script), "modules": {},
            "tmpdir": os.path.join(os.path.realpath(run), ".tmp"), **change}


@pytest.fixture
def origins(tmp_path, monkeypatch):
    run, clone = tmp_path / "run", tmp_path / "clone"
    tools, src = run / ".formal_tools", clone / "src"
    for d in (tools, src / "easyicu", run / "sic"):
        d.mkdir(parents=True)
    sha = "c" * 40

    def write(step_path: Path, script: str, **change):
        step_path.write_text(json.dumps(_origin(run, tools, src, sha, script, **change)))

    write(run / "sic" / "IMPORT_ORIGIN.json", "bench_local.py")
    for step in ("snapshot", "verify"):
        write(run / f"IMPORT_ORIGIN.inputs_{step}.json", "inputs_bench_local.py")
    live = types.SimpleNamespace(check=lambda: [],
                                 record=_origin(run, tools, src, sha, "formal_run_receipt.py"))
    monkeypatch.setitem(sys.modules, "formal_boot_live", live)
    code = {"import_path": str(src / "easyicu")}
    return types.SimpleNamespace(run=run, sha=sha, code=code, write=write, live=live, tools=tools,
                                 src=src)


def test_clean_step_records_and_a_clean_live_receipt_pass(origins):
    out = receipt.formal_import_origins(origins.run, origins.sha, origins.code, ["sic"])

    assert set(out) == {"extract:sic", "inputs:snapshot", "inputs:verify", "receipt"}


@pytest.mark.parametrize("change,code", [
    ("missing", "formal_import_origin_missing"),
    ("mismatched", "formal_import_origin_mismatch"),
    ("incomplete", "formal_import_origin_mismatch"),
    ("other_script", "formal_import_origin_mismatch"),
    ("other_head", "formal_import_origin_mismatch"),
    ("tmp_outside", "formal_import_origin_mismatch"),
])
def test_a_step_without_clean_origins_refuses_the_receipt(origins, change, code):
    path = origins.run / "sic" / "IMPORT_ORIGIN.json"
    if change == "missing":
        path.unlink()
    else:
        origins.write(path, "memgate.py" if change == "other_script" else "bench_local.py",
                      **{"mismatched": {"mismatched": ["evil_mod"]},
                         "incomplete": {"complete": False},
                         "other_head": {"head": "d" * 40},
                         "tmp_outside": {"tmpdir": "/tmp"}}.get(change, {}))

    with pytest.raises(receipt.ReceiptError) as error:
        receipt.formal_import_origins(origins.run, origins.sha, origins.code, ["sic"])
    assert error.value.code == code


def test_a_receipt_outside_the_boot_or_with_unclean_origins_is_refused(origins, monkeypatch):
    origins.live.check = lambda: ["json"]
    with pytest.raises(receipt.ReceiptError) as error:
        receipt.formal_import_origins(origins.run, origins.sha, origins.code, ["sic"])
    assert error.value.code == "formal_import_origin_mismatch"

    monkeypatch.delitem(sys.modules, "formal_boot_live")
    with pytest.raises(receipt.ReceiptError) as error:
        receipt.formal_import_origins(origins.run, origins.sha, origins.code, ["sic"])
    assert error.value.code == "formal_boot_missing"


# No formal step writes to /tmp (supervisor rule for the smoke run).

def test_the_boot_refuses_to_run_without_a_private_tmpdir(boot_env):
    marker = boot_env.tmp / "ran"
    target = boot_env.tools / "bench_local.py"
    target.write_text(f"open({str(marker)!r}, 'w').close()\n")
    argv = [sys.executable, "-I", "-S", "-P", str(boot_env.tools / "formal_boot.py"),
            str(boot_env.tools), str(boot_env.clone), boot_env.sha, _site(), str(boot_env.out),
            str(target)]

    done = subprocess.run(argv, capture_output=True, text=True, env={"PATH": os.environ["PATH"]})
    assert done.returncode == 97 and "formal_tmpdir_unset" in done.stderr and not marker.exists()


def test_the_boot_records_the_temp_dir_in_use(boot_env):
    done = _boot(boot_env, "bench_local.py", "import tempfile\ntempfile.mkstemp()\n")

    assert done.returncode == 0, done.stderr
    assert json.loads(boot_env.out.read_text())["tmpdir"] == os.path.realpath(boot_env.tmpdir)
    assert list(boot_env.tmpdir.iterdir())  # the temp file went to the private dir


def test_every_isolated_step_of_the_wrapper_sets_a_private_tmpdir():
    lines = WRAPPER.read_text().splitlines()
    launches = [i for i, line in enumerate(lines)
                if "env -i" in line and not line.lstrip().startswith("#")]
    assert len(launches) == 5
    for i in launches:
        block = " ".join(lines[i:i + 3])
        assert "TMPDIR=" in block, lines[i]
    body = WRAPPER.read_text()
    for step in ("TMPDIR=$W/tmp JOBLIB_TEMP_FOLDER=$W/tmp", "TMPDIR=$IT JOBLIB_TEMP_FOLDER=$IT",
                 "TMPDIR=$RS/tmp JOBLIB_TEMP_FOLDER=$RS/tmp",
                 'export TMPDIR=$R/.tmp JOBLIB_TEMP_FOLDER=$R/.tmp'):
        assert step in body


def test_a_script_that_imports_multiprocessing_keeps_clean_origins(boot_env):
    # Smoke run A (32b034cc5): importing multiprocessing aliases the running
    # __main__ (the staged script) as __mp_main__; that alias is not a mismatch.
    done = _boot(boot_env, "inputs_bench_local.py",
                 "import multiprocessing, sys\nassert '__mp_main__' in sys.modules\n")

    assert done.returncode == 0, done.stderr
    assert json.loads(boot_env.out.read_text())["mismatched"] == []


def test_a_sourceless_pyc_imported_as_mp_main_is_still_exit_97(boot_env):
    import py_compile

    stub = boot_env.tmp / "stub.py"
    stub.write_text("X = 1\n")
    py_compile.compile(str(stub), cfile=str(boot_env.tools / "stub.pyc"))
    done = _boot(boot_env, "bench_local.py",
                 "import sys, importlib\nsys.modules['__mp_main__'] = importlib.import_module('stub')\n")

    assert done.returncode == 97
