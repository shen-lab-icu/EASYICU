#!/usr/bin/env python3
"""Run unchanged legacy Figure 2 contracts at their exact historical source.

This is an engineering replay only: no Provider, research execution, signing
or paper authority is granted. The current-source suite runs independently.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from importlib.metadata import version
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile

REPO_ROOT = Path(__file__).resolve().parents[1]
# dbbdae6a5489 (tree 8d751c8ae88f) before the 2026-11-12 history rewrite.
HISTORY_COMMIT = "cde6fe4e820564c99275f82d1519cdef7eed6bff"
HISTORY_TREE = "b02d4b6a5766e3cd99d282a13dc6edf6e030242f"
HISTORY_SCORER = "8d146e10dfa02f43af1540dd4a018df5e6b6bf05838e1f55db30806884593911"
HISTORY_TESTS = "tests/benchmarks/figure2_canonical9/evaluator"


def _run(command: list[str], *, cwd: Path, env: dict[str, str]) -> str:
    return subprocess.check_output(command, cwd=cwd, env=env, text=True).strip()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--git-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--receipt-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    git_root = args.git_root.resolve()
    output = args.receipt_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="figure2-history-", dir=output)).resolve()
    source = work / "source"
    source.mkdir()
    env = {"PATH": os.environ.get("PATH", os.defpath), "LANG": os.environ.get("LANG", "C.UTF-8"), "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    private_home = work / "home"
    private_tmp = work / "tmp"
    private_home.mkdir()
    private_tmp.mkdir()
    env.update(HOME=str(private_home), EASYICU_HOME=str(private_home / "easyicu"), TMPDIR=str(private_tmp), PYTHONDONTWRITEBYTECODE="1", EASYICU_DISABLE_PROVIDER_FILE_READ="1")
    receipt = {"schema_version": "easyicu.figure2_history_ci/1", "status": "running", "authority_status": "engineering_replay_no_formal_authority", "current_checkout_commit": _run(["git", "rev-parse", "HEAD"], cwd=git_root, env=env), "history_commit": HISTORY_COMMIT, "history_tree": HISTORY_TREE, "history_scorer_sha256": HISTORY_SCORER, "python": platform.python_version(), "source_root": str(source), "test_scope": HISTORY_TESTS}
    target = output / "historical-v3-receipt.json"
    target.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    try:
        if _run(["git", "rev-parse", HISTORY_COMMIT + "^{tree}"], cwd=git_root, env=env) != HISTORY_TREE:
            raise RuntimeError("Historical Figure2 tree coordinate changed")
        archive = work / "history.tar"
        with archive.open("wb") as handle:
            subprocess.run(["git", "archive", HISTORY_COMMIT], cwd=git_root, env=env, stdout=handle, check=True)
        receipt["archive_sha256"] = hashlib.sha256(archive.read_bytes()).hexdigest()
        with tarfile.open(archive) as stream:
            for member in stream.getmembers():
                path = Path(member.name)
                if path.is_absolute() or ".." in path.parts or not (member.isdir() or member.isfile()):
                    raise RuntimeError("Unexpected non-regular or escaped file in pinned historical archive")
            for member in stream.getmembers():
                destination = source / member.name
                if member.isdir():
                    destination.mkdir(parents=True, exist_ok=True)
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                payload = stream.extractfile(member)
                if payload is None:
                    raise RuntimeError("Pinned archive file has no payload")
                with payload, destination.open("xb") as handle:
                    shutil.copyfileobj(payload, handle)
        policy = json.loads((REPO_ROOT / "benchmarks/figure2_canonical9/figure2_versioned_ci_v4.json").read_text(encoding="utf-8"))
        for relative, entry in policy["test_sources"].items():
            if hashlib.sha256((source / relative).read_bytes()).hexdigest() != entry["sha256"]:
                raise RuntimeError("Historical test bytes differ from the declared unchanged source")
        env["PYTHONPATH"] = os.pathsep.join([str(source / "src"), str(source)])
        # An OS-neutral test-only boundary: no INET connection may reach a
        # real evaluator, including a local model endpoint. Unix IPC remains
        # available for the normal subprocess/test machinery.
        bootstrap = work / "offline_guard"
        bootstrap.mkdir()
        (bootstrap / "sitecustomize.py").write_text("import socket\n_old_connect=socket.socket.connect\n_old_connect_ex=socket.socket.connect_ex\ndef _guard(self,address):\n    if self.family in (socket.AF_INET,socket.AF_INET6): raise RuntimeError('historical_ci_network_forbidden')\n    return _old_connect(self,address)\ndef _guard_ex(self,address):\n    if self.family in (socket.AF_INET,socket.AF_INET6): raise RuntimeError('historical_ci_network_forbidden')\n    return _old_connect_ex(self,address)\nsocket.socket.connect=_guard\nsocket.socket.connect_ex=_guard_ex\n", encoding="utf-8")
        env["PYTHONPATH"] = os.pathsep.join([str(bootstrap), env["PYTHONPATH"]])
        identity_code = "import json,easyicu; from pathlib import Path; from benchmarks.figure2_canonical9.evaluator.paper_rubric_v3 import load_figure2_paper_rubric,scorer_tree_sha256; m=load_figure2_paper_rubric(); print(json.dumps({'easyicu_origin':str(Path(easyicu.__file__).resolve()),'scorer_sha256':scorer_tree_sha256(),'rubric_ref':m.rubric_ref}))"
        identity = json.loads(_run([sys.executable, "-c", identity_code], cwd=source, env=env))
        if not Path(identity["easyicu_origin"]).is_relative_to(source) or identity["scorer_sha256"] != HISTORY_SCORER:
            raise RuntimeError("Historical run is importing the wrong source")
        receipt["loaded_identity"] = identity
        receipt["runtime_versions"] = {name: version(name) for name in ("pytest", "pydantic", "numpy", "pandas", "pyarrow")}
        original_source = {path.relative_to(source).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in source.rglob("*.py")}
        for path in sorted(source.rglob("*"), reverse=True):
            path.chmod(0o555 if path.is_dir() else 0o444)
        source.chmod(0o555)
        for name, extra in [("collection", ["--collect-only"]), ("execution", ["--junitxml=" + str(output / "historical-v3-results.xml")])]:
            command = [sys.executable, "-m", "pytest", "-q", "-m", "", "-p", "no:cacheprovider", HISTORY_TESTS, *extra]
            with (output / f"historical-v3-{name}.log").open("w", encoding="utf-8") as handle:
                code = subprocess.call(command, cwd=source, env=env, stdout=handle, stderr=subprocess.STDOUT)
            receipt[name + "_exit"] = code
            if code:
                raise RuntimeError("Historical Figure2 " + name + " failed")
        after = {path.relative_to(source).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in source.rglob("*.py")}
        if original_source != after:
            raise RuntimeError("Historical replay changed frozen source bytes")
        receipt.update(status="passed", source_unchanged=True, python_inet_connects="forbidden by test-only sitecustomize")
    except Exception as error:
        receipt.update(status="failed", error=f"{type(error).__name__}: {error}")
    finally:
        target.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))
    return 0 if receipt["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
