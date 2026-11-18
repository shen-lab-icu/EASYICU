"""Fixture support for synthetic runs through the sealed formal tools."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

FORMAL = Path(__file__).resolve().parents[2] / "tools" / "formal"
WRAPPER = FORMAL / "bench7v3f.sh"


def load_receipt_module(name):
    spec = importlib.util.spec_from_file_location(name, FORMAL.parent / "formal_run_receipt.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


tool_identity_receipt = load_receipt_module("formal_run_receipt_tool_identity")


def git(cwd: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(cwd), *args], check=True, capture_output=True,
                          text=True).stdout.strip()


def bash(script: str, *args: str, env: dict | None = None, cwd: Path | None = None):
    return subprocess.run(["bash", "-c", script, "_", *args], capture_output=True, text=True,
                          env=env, cwd=cwd)


def stage_formal_tools(run: Path, repo: Path, sha: str) -> subprocess.CompletedProcess:
    """The wrapper's own stage_tools (the script sourced: its functions only)."""
    run.mkdir(exist_ok=True)
    return bash('source "$1"; R=$2; FT=$2/.formal_tools; K=$3; SHA=$4; PY=$5; stage_tools',
                 str(WRAPPER), str(run), str(repo / ".git"), sha, sys.executable)


def input_manifest(root: Path) -> None:
    """A conversion manifest listing every flat parquet output under ``root``."""
    tables: dict = {}
    for path in sorted(root.rglob("*.parquet")):
        relative = path.relative_to(root)
        if relative.parts[0].endswith("_bucket"):
            continue
        table = relative.parts[0] if len(relative.parts) > 1 else path.stem
        tables.setdefault(table, []).append(
            {"relative_path": relative.as_posix(), "size_bytes": path.stat().st_size}
        )
    (root / "conversion_manifest.json").write_text(json.dumps({
        "schema_version": 1,
        "tables": [{"table": t, "status": "completed", "outputs": o} for t, o in tables.items()],
    }))


RETRY_HOOK = '''"""Synthetic OOM evidence seam, sealed only in this test clone."""
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
