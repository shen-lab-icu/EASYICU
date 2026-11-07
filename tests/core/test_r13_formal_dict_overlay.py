"""Receipt-bound formal dictionary input, using only synthetic sources and real boots.

Like the existing formal identity/retry tests, source the real wrapper's functions
and stage its sealed tools; no systemd scope or installed copy is changed.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path

import pandas as pd
import pytest

from tests.core.test_r13_formal_isolated_retry_origin import HOOK as RETRY_HOOK
from tests.core.test_r13_formal_tool_identity import _git, _stage, receipt
from tests.core.test_r13_formal_input_snapshot import _manifest as input_manifest

REPO = Path(__file__).resolve().parents[2]
BASE = "162d847d8c73f08c51cf303222e2b060172be110"

# Inject only a deterministic file accident at the existing Process seam.
# For retry, the existing synthetic OOM seam runs first; damage occurs on its successor.
HOOK = '''
def install_mutation(api):
    original = api._get_extraction_mp_context
    def context(mp):
        ctx = original(mp)
        class Context:
            @staticmethod
            def Process(*, target, args, daemon):
                mode = os.environ.get("FORMAL_FIXTURE_MUTATION", "")
                marker = Path(os.environ["RETRY_SENTINEL"])
                if mode and mode != "retry_pass" and (mode != "retry" or marker.exists()):
                    overlay = Path(os.environ["FORMAL_FIXTURE_OVERLAY"])
                    if mode == "removed":
                        overlay.unlink(missing_ok=True)
                    elif mode == "replaced":
                        overlay.unlink(missing_ok=True)
                        overlay.mkdir(exist_ok=True)
                    else:
                        overlay.write_text("{}")
                return ctx.Process(target=target, args=args, daemon=daemon)
        return Context()
    api._get_extraction_mp_context = context
'''


def _blob(path):
    return subprocess.check_output(["git", "show", f"{BASE}:{path}"], cwd=REPO)


def _clone(tmp, *, baseline=False):
    clone = tmp / ("base_clone" if baseline else "clone")
    for relative in _git(REPO, "ls-files", "src", "pyproject.toml").splitlines():
        dest = clone / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / relative, dest)
    (clone / ".gitignore").write_text("__pycache__/\n*.pyc\n")
    (clone / "src/easyicu/_formal_overlay_fixture.py").write_text(RETRY_HOOK + HOOK)
    with (clone / "src/easyicu/api/extraction.py").open("a") as fh:
        fh.write("\nimport os as _fixture_os\nimport sys as _fixture_sys\n"
                 "from easyicu._formal_overlay_fixture import install, install_mutation\n"
                 "if _fixture_os.environ.get('FORMAL_FIXTURE_ISOLATED'):\n"
                 "    _ISOLATED_STREAM_BATCH_TARGETS = frozenset({('hirid', 'outcome')})\n"
                 "if _fixture_os.environ.get('FORMAL_FIXTURE_MUTATION') in ('retry', 'retry_pass'):\n"
                 "    install(_fixture_sys.modules[__name__])\n"
                 "install_mutation(_fixture_sys.modules[__name__])\n")
    manifest = json.loads((REPO / "tools/formal/FORMAL_TOOLS.json").read_text())
    for entry in manifest["tools"]:
        dest = clone / entry["path"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(_blob(entry["path"]) if baseline else (REPO / entry["path"]).read_bytes())
        entry["sha256"] = hashlib.sha256(dest.read_bytes()).hexdigest()
    (clone / "tools/formal/FORMAL_TOOLS.json").write_text(json.dumps(manifest))
    _git(tmp, "init", "-q", str(clone))
    _git(clone, "add", ".")
    _git(clone, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "synthetic seal")
    return clone, _git(clone, "rev-parse", "HEAD")


@pytest.fixture(scope="module")
def sources(tmp_path_factory):
    import easyicu
    from easyicu.synthetic_sources import generate_source

    origin = Path(easyicu.__file__).resolve()
    head = _git(REPO, "rev-parse", "HEAD")
    print(f"easyicu={origin}\nHEAD={head}", flush=True)
    assert origin.is_relative_to(REPO / "src")
    assert head == os.environ["CLAIMED_SHA"]
    tmp = tmp_path_factory.mktemp("dict_overlay")
    source = tmp / "source"
    generate_source(source, "hirid", stays=12)
    input_manifest(source)
    yield {"tmp": tmp, "source": source, "fixed": _clone(tmp),
           "base": _clone(tmp, baseline=True)}
    # Staged tools are deliberately read-only, including when a baseline assertion fails.
    for path in tmp.rglob(".formal_tools"):
        path.chmod(0o700)


def _overlay(tmp):
    path = tmp / "overlay.json"
    dictionary = json.loads((REPO / "src/easyicu/data/concept-dict.json").read_text())
    source = dictionary["hr"]["sources"]["hirid"][0]
    path.write_text(json.dumps({"hr": {"sources": {"hirid": [{**source, "ids": [999999]}]}}}))
    return path


def _job(sources, name, *, overlay=None, caller=False, mutation="", isolated=False,
         modules="outcome", baseline=False):
    clone, sha = sources["base" if baseline else "fixed"]
    run = sources["tmp"] / name
    staged = _stage(run, clone, sha)
    assert staged.returncode == 0, staged.stderr
    tools = run / ".formal_tools"
    (run / "hirid").mkdir()
    private = run / "tmp"
    private.mkdir()
    cpu = str(min(os.sched_getaffinity(0)))
    spec = f"hirid:{sources['source']}:{cpu}-{cpu}"
    env = {"PATH": os.environ["PATH"], "TMPDIR": str(private),
           "JOBLIB_TEMP_FOLDER": str(private), "EASYICU_HOME": str(run / "home"),
           "EASYICU_DUCKDB_TEMP_DIR": str(private), "BENCH_GATED": "1",
           "BENCH_BATCH": "4", "BENCH_MODULES": modules, "BENCH_NC": "1",
           "BENCH_SEG": f"{cpu}-{cpu}", "EASYICU_PARALLEL_MAX_WORKERS": "1",
           "RETRY_SENTINEL": str(run / "synthetic_oom"),
           "FORMAL_FIXTURE_MUTATION": mutation,
           "FORMAL_FIXTURE_ISOLATED": "1" if isolated else "",
           "FORMAL_FIXTURE_OVERLAY": str(overlay or ""),
           **dict.fromkeys(("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                            "POLARS_MAX_THREADS", "RAYON_NUM_THREADS", "EASYICU_DUCKDB_THREADS",
                            "EASYICU_ARROW_THREADS", "NUMEXPR_MAX_THREADS"), "1")}
    if caller:
        # Values must not be forwarded or appear in a receipt.
        env.update(EASYICU_DICT_PATH=str(overlay), EASYICU_DICT_DIR="unrelated-private-value")
    elif overlay is not None:
        env["BENCH_DICT_PATH"] = str(overlay)

    def boot(script, out, *args):
        # Real wrapper configuration, env -i boundary, staged boot and staged driver.
        # The fallback is precisely BASE's empty dictionary configuration.
        shell = '''source "$1"
shift
BOOT=("$1" -I -S -P "$2" "$3" "$4" "$5" "$6")
shift 6
DENV=(); DPARAM=()
if declare -F configure_dictionary >/dev/null; then configure_dictionary; fi
if [ "$1" = v2_bench_local.py ]; then
  FORWARD=("${DENV[@]}")
else
  FORWARD=()
fi
SCRIPT=$1; OUT=$2; shift 2
if [ "$SCRIPT" = formal_run_receipt.py ]; then
  set -- "${DPARAM[@]}" "$@"
fi
PASS=()
for name in PATH TMPDIR JOBLIB_TEMP_FOLDER EASYICU_HOME EASYICU_DUCKDB_TEMP_DIR BENCH_GATED BENCH_BATCH BENCH_MODULES BENCH_NC BENCH_SEG EASYICU_PARALLEL_MAX_WORKERS RETRY_SENTINEL FORMAL_FIXTURE_MUTATION FORMAL_FIXTURE_ISOLATED FORMAL_FIXTURE_OVERLAY OMP_NUM_THREADS OPENBLAS_NUM_THREADS MKL_NUM_THREADS POLARS_MAX_THREADS RAYON_NUM_THREADS EASYICU_DUCKDB_THREADS EASYICU_ARROW_THREADS NUMEXPR_MAX_THREADS; do
  PASS+=("$name=${!name}")
done
exec env -i "${PASS[@]}" "${FORWARD[@]}" taskset -c "''' + cpu + '''" "${BOOT[@]}" "$OUT" "''' + str(tools) + '''/$SCRIPT" "$@"
'''
        done = subprocess.run(["bash", "-c", shell, "_", str(tools / "bench7v3f.sh"),
            sys.executable, str(tools / "formal_boot.py"), str(tools), str(clone), sha,
            os.path.realpath(sysconfig.get_paths()["purelib"]), script, str(out),
            *map(str, args)], cwd=run, env=env, capture_output=True, text=True, timeout=150)
        (run / f"{out.name}.log").write_text(done.stdout + done.stderr)
        return done

    done = boot("inputs_bench_local.py", run / "IMPORT_ORIGIN.inputs_snapshot.json",
                "snapshot", "--out", run / "INPUT_SNAPSHOT.json", spec)
    if done.returncode:
        return {"run": run, "done": done, "boot": boot}
    done = boot("v2_bench_local.py", run / "hirid/IMPORT_ORIGIN.json", "hirid",
                sources["source"], run / "hirid", "full")
    summary_path = run / "hirid/summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    if done.returncode or summary.get("exception") or summary.get("errors") or summary.get("worker_failures"):
        return {"run": run, "done": done, "summary": summary, "boot": boot}
    done = boot("inputs_bench_local.py", run / "IMPORT_ORIGIN.inputs_verify.json",
                "verify", "--pre", run / "INPUT_SNAPSHOT.json", "--out",
                run / "INPUT_SNAPSHOT_POST.json")
    assert done.returncode == 0, done.stderr
    (run / "status").write_text("inputs_pre exit=0\nhirid exit=0\ninputs_post exit=0\n")
    native = json.loads((run / "hirid/export/_manifest.json").read_text())
    (run / "REQUIRED_LAYOUT.json").write_text(json.dumps({
        "schema": "easyicu.required_layout/2", "complete": True,
        "databases": {"hirid": {r["table"]: {"layout": "flat"} for r in native["storage_reads"]}},
    }))
    done = boot("formal_run_receipt.py", run / "IMPORT_ORIGIN.receipt.json",
                "--run-root", run, "--sealed-sha", sha, "--scratch", run / "scratch", spec)
    return {"run": run, "done": done, "summary": summary, "boot": boot}


def _published(arm):
    assert arm["done"].returncode == 0, arm["done"].stderr
    path = arm["run"] / "RUN_RECEIPT.json"
    assert path.exists(), arm.get("summary")
    return json.loads(path.read_text())


def _parquets(arm):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (arm["run"] / "hirid/export").glob("*.parquet")}


@pytest.fixture(scope="module")
def main_runs(sources):
    return (_job(sources, "no_overlay", modules="outcome,vitals"),
            _job(sources, "base_no_overlay", modules="outcome,vitals", baseline=True))


def test_no_overlay_is_null_and_exports_match_base(main_runs):
    fixed, base = main_runs
    assert _published(fixed)["dict_overlay"] is None
    assert _published(base).get("dict_overlay") is None
    assert _parquets(fixed) and _parquets(fixed) == _parquets(base)
    assert not (fixed["run"] / "DICT_OVERLAY.json").exists()


def test_no_overlay_rest_of_receipt_is_byte_identical_to_base(tmp_path, monkeypatch):
    # Same fixed synthetic input/output records, same code/tool identity and origins.
    # Comparing full real runs would also compare intentionally different seals and timings.
    from tests.core.test_r13_formal_run_receipt import CODE, _prepared, _run_root

    original = tmp_path / "base_receipt.py"
    original.write_bytes(_blob("tools/formal_run_receipt.py"))
    shutil.copyfile(REPO / "tools/formal_declared_absent.json", tmp_path / "formal_declared_absent.json")
    spec = importlib.util.spec_from_file_location("base_overlay_receipt", original)
    base = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = base
    spec.loader.exec_module(base)
    prepared = _prepared(tmp_path / "prepared")
    run = _run_root(tmp_path, prepared, commit=CODE["commit"])
    monkeypatch.setattr(receipt.time, "time", lambda: 123.0)
    for module in (receipt, base):
        monkeypatch.setattr(module, "sealed_code_identity", lambda sha: dict(CODE))
        monkeypatch.setattr(module, "formal_tool_scripts", lambda *args: [])
        monkeypatch.setattr(module, "formal_import_origins", lambda *args: {})
        monkeypatch.setattr(module, "_seal_published", lambda *args: [])
        monkeypatch.setattr(module, "_admit_consumed", lambda a, b, p: {
            "target_realpath": os.path.realpath(p), "admitted_by": "data_root",
            "size": None, "sha256": None})
    kwargs = {"scratch": tmp_path / "scratch", "started_at": 0, "parameters": {"modules": "outcome"}}
    fixed = receipt.build_run_receipt(run, CODE["commit"], [f"sic:{prepared}:144-145"], **kwargs)
    old = base.build_run_receipt(run, CODE["commit"], [f"sic:{prepared}:144-145"], **kwargs)
    assert fixed.pop("dict_overlay") is None
    assert json.dumps(fixed, sort_keys=True).encode() == json.dumps(old, sort_keys=True).encode()


def test_configured_overlay_is_bound_and_changes_worker_selector(sources, main_runs):
    overlay = _overlay(sources["tmp"])
    # A symlink to a regular file is accepted, and the receipt pins its realpath.
    link = sources["tmp"] / "overlay-link.json"
    link.symlink_to(overlay)
    arm = _job(sources, "overlay", overlay=link, modules="outcome,vitals")
    saved = _published(arm)
    expected = {"path": str(overlay.resolve()), "sha256": hashlib.sha256(overlay.read_bytes()).hexdigest(),
                "size": overlay.stat().st_size}
    assert saved["dict_overlay"] == expected
    assert saved["parameters"]["dict_overlay_path"] == str(link)
    assert receipt.validate_dict_overlay(saved)["file_status"] == "verified"
    workers = list((arm["run"] / "hirid/worker_origin").glob("*.json"))
    assert workers and all(json.loads(p.read_text())["dict_overlay"] == expected for p in workers)
    assert all(json.loads(p.read_text())["ok"] for p in workers)
    normal = pd.read_parquet(main_runs[0]["run"] / "hirid/export/vitals.parquet")
    changed = pd.read_parquet(arm["run"] / "hirid/export/vitals.parquet")
    assert normal["hr"].notna().any() and changed["hr"].isna().all()
    assert _parquets(arm)["vitals.parquet"] != _parquets(main_runs[0])["vitals.parquet"]


@pytest.mark.parametrize("mode,isolated", [("changed", False), ("removed", False),
                                         ("replaced", False), ("changed", True), ("retry", True)])
def test_changed_overlay_is_refused_before_worker_load(sources, tmp_path, mode, isolated):
    overlay = _overlay(tmp_path)
    arm = _job(sources, f"accident_{mode}_{isolated}", overlay=overlay,
               mutation=mode, isolated=isolated)
    invalid = arm["run"] / "RUN_INVALID"
    assert invalid.exists(), arm.get("summary")
    assert invalid.read_text().strip() == "dict_overlay_changed"
    assert not (arm["run"] / "RUN_RECEIPT.json").exists()
    assert not list((arm["run"] / "hirid/export").glob("*.parquet"))
    records = [json.loads(p.read_text()) for p in (arm["run"] / "hirid/worker_origin").glob("*.json")]
    assert any(r.get("reason") == "dict_overlay_changed" and not r["ok"] for r in records)
    assert any("exit code 97" in str(v) or "exitcode=97" in str(v)
               or "97" in str(v) for v in arm.get("summary", {}).values())


def test_caller_dictionary_environment_is_ignored_and_names_are_receipted(sources, main_runs):
    overlay = _overlay(sources["tmp"])
    arm = _job(sources, "caller_env", overlay=overlay, caller=True, modules="outcome,vitals")
    saved = _published(arm)
    assert saved["dict_overlay"] is None
    assert saved["dict_overlay_caller_env_ignored"] is True
    assert saved["dict_overlay_caller_env_ignored_names"] == ["EASYICU_DICT_PATH", "EASYICU_DICT_DIR"]
    assert str(overlay) not in json.dumps(saved)
    assert "unrelated-private-value" not in json.dumps(saved)
    assert _parquets(arm) == _parquets(main_runs[0])


def test_configured_overlay_survives_certified_isolated_retry(sources, tmp_path):
    overlay = _overlay(tmp_path)
    arm = _job(sources, "overlay_retry", overlay=overlay, mutation="retry_pass", isolated=True)
    saved = _published(arm)
    expected = saved["dict_overlay"]
    assert expected["sha256"] == hashlib.sha256(overlay.read_bytes()).hexdigest()
    workers = saved["import_origin"]["workers:hirid"]
    assert workers["aborted_retried"] == 1
    assert all(json.loads(Path(r["path"]).read_text())["dict_overlay"] == expected
               for r in workers["records"])


@pytest.mark.parametrize("kind,reason", [("directory", "dict_overlay_not_regular"),
                                        ("fifo_link", "dict_overlay_not_regular"),
                                        ("missing", "dict_overlay_unavailable")])
def test_non_file_overlay_is_refused_at_boot(sources, tmp_path, kind, reason):
    path = tmp_path / "dictionary"
    if kind == "directory":
        path.mkdir()
    elif kind == "fifo_link":
        fifo = tmp_path / "fifo"
        os.mkfifo(fifo)
        path.symlink_to(fifo)
    arm = _job(sources, f"bad_path_{kind}", overlay=path)
    assert arm["done"].returncode == 97, arm["done"].stderr
    assert (arm["run"] / "RUN_INVALID").read_text().strip() == reason
    assert not (arm["run"] / "INPUT_SNAPSHOT.json").exists()
    assert not (arm["run"] / "hirid/summary.json").exists()


def test_receipt_validator_requires_key_and_checks_existing_or_missing_file(tmp_path, capsys):
    overlay = _overlay(tmp_path)
    saved = {"dict_overlay": {"path": str(overlay), "sha256": hashlib.sha256(overlay.read_bytes()).hexdigest(),
                              "size": overlay.stat().st_size}}
    target = tmp_path / "RUN_RECEIPT.json"
    target.write_text(json.dumps(saved))
    assert hasattr(receipt, "validate_dict_overlay")
    with pytest.raises(receipt.ReceiptError) as caught:
        receipt.validate_dict_overlay({})
    assert caught.value.code == "dict_overlay_missing"
    assert receipt.main(["--validate-dict-overlay", str(target)]) == 0
    assert json.loads(capsys.readouterr().out)["file_status"] == "verified"
    overlay.write_text("{}")
    with pytest.raises(receipt.ReceiptError) as caught:
        receipt.validate_dict_overlay(saved)
    assert caught.value.code == "dict_overlay_changed"
    overlay.unlink()
    assert receipt.main(["--validate-dict-overlay", str(target)]) == 0
    checked = json.loads(capsys.readouterr().out)
    assert checked["file_status"] == "missing" and "not rechecked" in checked["message"]
    with pytest.raises(receipt.ReceiptError) as caught:
        receipt.write_run_receipt(tmp_path, saved)
    assert caught.value.code == "dict_overlay_changed"


def test_wrapper_preserves_dictionary_refusal_reason(tmp_path):
    invalid = tmp_path / "RUN_INVALID"
    invalid.write_text("dict_overlay_changed\n")
    done = subprocess.run(["bash", "-c", 'source "$1"; R=$2; finish_receipt 97', "_",
                           str(REPO / "tools/formal/bench7v3f.sh"), str(tmp_path)],
                          capture_output=True, text=True)
    assert done.returncode == 3
    assert invalid.read_text() == "dict_overlay_changed\n"
