"""B51: a certified isolated retry reaches a PASS through the staged formal tools.

The only injected behavior is the existing OOM-test context/counter seam:
one real spawned child exits abruptly, leaving an uncommitted part, and
the synthetic cgroup counter rises. All extraction, publication, boot,
snapshot and receipt code runs normally. No real OOM is induced.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path

import pytest

from tests.core.test_r13_formal_tool_identity import _git, _stage, receipt
from tests.core.test_r13_formal_input_snapshot import _manifest as input_manifest

REPO = Path(__file__).resolve().parents[2]

HOOK = '''"""Synthetic OOM evidence seam, sealed only in this test clone."""
import os
from pathlib import Path


def abort_batch(*args):
    part = Path(args[-1])
    part.write_bytes(b"uncommitted synthetic part")
    Path(os.environ["RETRY_SENTINEL"]).touch()
    os._exit(137)


def install(api):
    original = api._get_extraction_mp_context

    def context(mp):
        ctx = original(mp)

        class Context:
            @staticmethod
            def Process(*, target, args, daemon):
                if (target is api._write_isolated_stream_batch
                        and not Path(os.environ["RETRY_SENTINEL"]).exists()):
                    target = abort_batch
                return ctx.Process(target=target, args=args, daemon=daemon)

        return Context()

    api._get_extraction_mp_context = context
    api._ISOLATED_STREAM_BATCH_TARGETS = frozenset({("hirid", "outcome")})
    api._cgroup_oom_kills = lambda: int(Path(os.environ["RETRY_SENTINEL"]).exists())
'''


@pytest.fixture(scope="module")
def recovered_run(tmp_path_factory):
    import easyicu
    from easyicu.synthetic_sources import generate_source

    origin = Path(easyicu.__file__).resolve()
    head = _git(REPO, "rev-parse", "HEAD")
    print(f"easyicu={origin}\nHEAD={head}", flush=True)
    assert origin.is_relative_to(REPO / "src")
    assert head == os.environ.get("CLAIMED_SHA", head)

    tmp = tmp_path_factory.mktemp("formal_retry")
    clone = tmp / "clone"
    # Copy tracked source at the working bytes, including the proposed fix.
    for relative in _git(REPO, "ls-files", "src", "pyproject.toml").splitlines():
        dest = clone / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / relative, dest)
    (clone / ".gitignore").write_text("__pycache__/\n*.pyc\n")
    hook = clone / "src/easyicu/_formal_retry_fixture.py"
    hook.write_text(HOOK)
    extraction = clone / "src/easyicu/api/extraction.py"
    with extraction.open("a") as fh:
        fh.write("\nfrom easyicu._formal_retry_fixture import install\n"
                 "import sys as _fixture_sys\ninstall(_fixture_sys.modules[__name__])\n")
    manifest = json.loads((REPO / "tools/formal/FORMAL_TOOLS.json").read_text())
    for entry in manifest["tools"]:
        dest = clone / entry["path"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / entry["path"], dest)
        entry["sha256"] = hashlib.sha256(dest.read_bytes()).hexdigest()
    (clone / "tools/formal/FORMAL_TOOLS.json").write_text(json.dumps(manifest))
    _git(tmp, "init", "-q", str(clone))
    _git(clone, "add", ".")
    _git(clone, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "synthetic seal")
    sha = _git(clone, "rev-parse", "HEAD")
    source = tmp / "source"
    generate_source(source, "hirid", stays=6)
    input_manifest(source)
    run = tmp / "run"
    staged = _stage(run, clone, sha)
    assert staged.returncode == 0, staged.stderr
    tools = run / ".formal_tools"
    (run / "hirid").mkdir()
    private = run / "tmp"
    private.mkdir()
    cpu = str(min(os.sched_getaffinity(0)))
    spec = f"hirid:{source}:{cpu}-{cpu}"
    env = {**os.environ, "TMPDIR": str(private), "JOBLIB_TEMP_FOLDER": str(private),
           "EASYICU_HOME": str(run / "home"), "EASYICU_DUCKDB_TEMP_DIR": str(private),
           "RETRY_SENTINEL": str(run / "synthetic_oom"), "BENCH_GATED": "1",
           "BENCH_BATCH": "4", "BENCH_MODULES": "outcome", "BENCH_NC": "1",
           "BENCH_SEG": f"{cpu}-{cpu}", "EASYICU_PARALLEL_MAX_WORKERS": "1",
           **dict.fromkeys(("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                            "POLARS_MAX_THREADS", "RAYON_NUM_THREADS", "EASYICU_DUCKDB_THREADS",
                            "EASYICU_ARROW_THREADS", "NUMEXPR_MAX_THREADS"), "1")}
    env.pop("PYTHONPATH", None)

    def boot(script, out, *args):
        done = subprocess.run(["taskset", "-c", cpu, sys.executable, "-I", "-S", "-P",
            str(tools / "formal_boot.py"), str(tools), str(clone), sha,
            os.path.realpath(sysconfig.get_paths()["purelib"]), str(out),
            str(tools / script), *map(str, args)], cwd=run, env=env,
            capture_output=True, text=True, timeout=120)
        (run / f"{out.name}.log").write_text(done.stdout + done.stderr)
        return done

    done = boot("inputs_bench_local.py", run / "IMPORT_ORIGIN.inputs_snapshot.json",
                "snapshot", "--out", run / "INPUT_SNAPSHOT.json", spec)
    assert done.returncode == 0, done.stderr
    done = boot("v2_bench_local.py", run / "hirid/IMPORT_ORIGIN.json", "hirid", source,
                run / "hirid", "full")
    assert done.returncode == 0, done.stderr
    summary = json.loads((run / "hirid/summary.json").read_text())
    assert not summary.get("exception"), summary
    assert not summary["errors"] and not summary["worker_failures"]
    done = boot("inputs_bench_local.py", run / "IMPORT_ORIGIN.inputs_verify.json",
                "verify", "--pre", run / "INPUT_SNAPSHOT.json", "--out",
                run / "INPUT_SNAPSHOT_POST.json")
    assert done.returncode == 0, done.stderr
    (run / "status").write_text("inputs_pre exit=0\nhirid exit=0\ninputs_post exit=0\n")
    native_path = run / "hirid/export/_manifest.json"
    native_bytes = native_path.read_bytes()
    native = json.loads(native_bytes)
    (run / "REQUIRED_LAYOUT.json").write_text(json.dumps({
        "schema": "easyicu.required_layout/2", "complete": True,
        "databases": {"hirid": {r["table"]: {"layout": "flat"}
                                for r in native["storage_reads"]}},
    }))
    done = boot("formal_run_receipt.py", run / "IMPORT_ORIGIN.receipt.json",
                "--run-root", run, "--sealed-sha", sha, "--scratch", run / "scratch", spec)
    # Keep the baseline refusal available to each regression assertion.
    yield {"run": run, "native": native, "native_path": native_path,
           "native_bytes": native_bytes, "done": done,
           "src": str(clone / "src"), "sha": sha, "tools": str(tools)}
    os.chmod(tools, 0o700)


def _pass(arm):
    assert arm["done"].returncode == 0, arm["done"].stderr
    published = json.loads((arm["run"] / "RUN_RECEIPT.json").read_text())
    assert published["databases"]["hirid"]["native_manifest"]["sha256"] == (
        hashlib.sha256(arm["native_path"].read_bytes()).hexdigest())
    checked = published["import_origin"]["workers:hirid"]
    assert checked["aborted_retried"] == 1
    aborted = [r for r in checked["records"] if r["status"] == "aborted_retried"]
    assert len(aborted) == 1
    return checked, aborted[0]


def test_certified_isolated_retry_passes_the_real_formal_receipt(recovered_run):
    arm = recovered_run
    checked, aborted = _pass(arm)
    event = arm["native"]["stream_resume_events"]["outcome"][0]
    assert event["evidence"] == "oom_kill_event"
    assert aborted["stream_attempt"] == event["stream_attempt"]
    assert aborted["pid"] == event["worker_pid"]
    assert event["part_discarded"] is True
    assert all(json.loads(Path(r["path"]).read_text())["complete"]
               for r in checked["records"] if r["status"] == "completed")
    assert not list((arm["run"] / "hirid/export").glob(".*.batch-*"))


@pytest.mark.parametrize("damage", [
    "no_ledger_entry", "uncertified", "no_successor", "kept_part", "wrong_attempt",
    "wrong_batch", "wrong_pid", "successor_incomplete", "bad_startup_origin",
    "unverified_retry", "leftover_part", "leftover_partial",
])
def test_incomplete_origin_requires_exact_certified_discarded_recovery(recovered_run, damage):
    arm = recovered_run
    checked, aborted = _pass(arm)  # Also proves these tests fail on the reviewed base.
    native = copy.deepcopy(arm["native"])
    event = native["stream_resume_events"]["outcome"][0]
    origin_path = None
    old_origin = None
    leftover = None
    if damage == "no_ledger_entry":
        native["stream_resume_events"] = {}
    elif damage == "uncertified":
        event["termination_evidence"]["oom_kill_delta"] = 0  # Bare 137 is not evidence.
    elif damage == "no_successor":
        native["stream_batches"] = {}
    elif damage == "kept_part":
        event["part_discarded"] = False
    elif damage == "wrong_attempt":
        event["stream_attempt"]["attempt_id"] = "unmatched"
    elif damage == "wrong_batch":
        event["stream_attempt"]["batch_index"] += 1
    elif damage == "wrong_pid":
        event["worker_pid"] += 1
    elif damage == "unverified_retry":
        event["kind"] = "unverified_kill_retry"
    elif damage in ("leftover_part", "leftover_partial"):
        name = event["discarded_part"]
        leftover = arm["native_path"].parent / (
            name if damage == "leftover_part" else f".{name}.partial")
        leftover.write_bytes(b"kept synthetic part")
    else:
        origin_path = Path(aborted["path"])
        if damage == "successor_incomplete":
            successor = native["stream_batches"]["outcome"][0]["stream_attempt"]
            origin_path = Path(next(r["path"] for r in checked["records"]
                                    if r["stream_attempt"] == successor))
        old_origin = origin_path.read_text()
        record = json.loads(old_origin)
        record.update({"complete": False} if damage == "successor_incomplete"
                      else {"mismatched": ["outside_module"]})
        origin_path.write_text(json.dumps(record))
    arm["native_path"].write_text(json.dumps(native))
    try:
        with pytest.raises(receipt.ReceiptError) as caught:
            receipt._worker_origins("hirid", arm["run"] / "hirid/worker_origin",
                arm["src"], arm["sha"], arm["tools"], str(arm["run"]))
        assert caught.value.code == "formal_import_origin_mismatch"
    finally:
        arm["native_path"].write_bytes(arm["native_bytes"])
        if origin_path is not None:
            origin_path.write_text(old_origin)
        if leftover is not None:
            leftover.unlink()
