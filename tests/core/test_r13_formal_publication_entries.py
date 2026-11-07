"""Formal entry gates refuse synthetic inputs before content reads or listings."""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from easyicu import formal_inputs as inputs
from easyicu import storage_seal_publication as publication
from tests.core.test_r13_formal_run_receipt import receipt

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def identity():
    import easyicu

    origin = Path(easyicu.__file__).resolve()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    print(f"easyicu={origin}\nHEAD={head}", flush=True)
    assert origin.is_relative_to(REPO / "src")
    assert head == os.environ["CLAIMED_SHA"]


def _root(path):
    path.mkdir()
    (path / "sample.parquet").write_bytes(b"synthetic sentinel; never opened on refusal")
    (path / "conversion_manifest.json").write_text(json.dumps({"tables": []}))
    return path


def _closure(tmp, shape):
    target = _root(tmp / "target")
    if shape == "direct":
        return target, target
    root = _root(tmp / "farm")
    if shape == "linked":
        (root / "unused").symlink_to(target, target_is_directory=True)
    else:
        (root / "conversion_manifest.json").write_text(json.dumps({
            "tables": [], "derived_from": {
                "path": str(target / "conversion_manifest.json"), "sha256": "0" * 64,
            },
        }))
    return root, target


def _audit(roots):
    events, enabled = [], [True]

    def hook(event, args):
        if not enabled[0] or event not in {"open", "os.listdir", "os.scandir"}:
            return
        if not args or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.path.realpath(os.fsdecode(args[0])))
        if not any(path == root or root in path.parents for root in roots):
            return
        frame = sys._getframe(1)
        gate = False
        while frame:
            if frame.f_globals.get("__name__") == publication.__name__:
                gate = True
                break
            frame = frame.f_back
        events.append((event, path, gate))

    sys.addaudithook(hook)
    return events, enabled


@pytest.mark.parametrize("entry", ["snapshot", "plan", "verify", "receipt"])
@pytest.mark.parametrize("shape", ["direct", "linked", "provenance"])
@pytest.mark.parametrize("suffix,reason", [
    (publication.PENDING_SUFFIX, publication.UNPUBLISHED),
    (publication.REFUSAL_SUFFIX, publication.REFUSED),
])
def test_marked_closure_is_refused_before_content_or_listing(
    tmp_path, monkeypatch, entry, shape, suffix, reason,
):
    root, marked = _closure(tmp_path, shape)
    (marked.parent / f".{marked.name}{suffix}").write_text("{}")
    spec = f"hirid:{root}:144-145"
    pre = {"schema": inputs.SNAPSHOT_SCHEMA, "databases": {
        "hirid": {"data_path": str(root), "files": [], "digest": "unused"},
    }}
    # Let the independent receipt reach its real database reader on the old tree.
    # No publication or input reader is replaced.
    monkeypatch.setattr(receipt, "sealed_code_identity", lambda sha: {"version": "test"})
    monkeypatch.setattr(receipt, "approved_declared_absent", lambda: ({}, "unused"))
    monkeypatch.setattr(receipt, "formal_tool_scripts", lambda *args: [])
    monkeypatch.setattr(receipt, "_snapshots", lambda *args: ({}, {}, {}))
    monkeypatch.setattr(receipt, "load_required_layout", lambda *args: (
        {"databases": {"hirid": {"sample": {"layout": "flat"}}}}, "unused"))
    monkeypatch.setattr(receipt, "_status_exits", lambda *args: {})
    calls = {
        "snapshot": lambda: inputs.take_snapshot([spec]),
        "plan": lambda: inputs.plan_input_files("hirid", root),
        "verify": lambda: inputs.verify_snapshot(pre),
        # A healthy flat first spec ensures the receipt gates every spec at entry.
        "receipt": lambda: receipt.build_run_receipt(
            tmp_path / "run", "0" * 40,
            [f"miiv:{tmp_path / 'healthy'}:144-145", spec],
            scratch=tmp_path / "scratch"),
    }
    _root(tmp_path / "healthy")
    events, enabled = _audit([root, marked, tmp_path / "healthy"])
    error = None
    try:
        calls[entry]()
    except Exception as exc:
        error = exc
    finally:
        enabled[0] = False
    assert not [e for e in events if not e[2]], "content/listing before the publication gate"
    assert not [e for e in events if e[1] == marked or marked in e[1].parents]
    assert isinstance(error, publication.SealPublicationError)
    assert error.reason == reason


def _boot():
    spec = importlib.util.spec_from_file_location("entry_test_boot", REPO / "tools/formal/formal_boot.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("shape", ["direct", "linked", "provenance"])
@pytest.mark.parametrize("existing", [False, True])
def test_dictionary_binding_refuses_closure_file_but_accepts_work_root(
    tmp_path, shape, existing,
):
    root, target = _closure(tmp_path, shape)
    config = target / "dictionary.json"
    config.write_text("{}")
    tools = tmp_path / "run" / ".formal_tools"
    tools.mkdir(parents=True)
    boot = _boot()
    if existing:
        (tools.parent / "DICT_OVERLAY.json").write_text(json.dumps({
            "path": str(config), "size": config.stat().st_size,
            "sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
        }))
    # Compatibility lets the regression exercise the old, actually-reading helper.
    kwargs = {"input_roots": [root]} if "input_roots" in inspect.signature(
        boot.bind_dictionary).parameters else {}
    events, enabled = _audit([target])
    error = None
    try:
        boot.bind_dictionary(str(tools), str(config), **kwargs)
    except Exception as exc:
        error = exc
    finally:
        enabled[0] = False
    assert not [e for e in events if e[0] == "open" and e[1] == config]
    assert isinstance(error, boot.DictOverlayError)
    assert error.reason == "formal_configuration_inside_input"
    (tools.parent / "DICT_OVERLAY.json").unlink(missing_ok=True)
    outside = tmp_path / "dictionary.json"
    outside.write_text("{}")
    assert boot.bind_dictionary(str(tools), str(outside), **kwargs)["path"] == str(outside)
    assert boot.bind_dictionary(str(tools), str(outside), **kwargs)["path"] == str(outside)


@pytest.mark.parametrize("shape", ["direct", "linked", "provenance"])
def test_required_layout_wrapper_refuses_before_cmp_json_or_copy_and_allows_work_root(
    tmp_path, shape,
):
    root, target = _closure(tmp_path, shape)
    layout = {"schema": "easyicu.required_layout/2", "complete": True,
              "databases": {"hirid": {"sample": {"layout": "flat"}}}}
    inside = target / "layout.json"
    inside.write_text(json.dumps(layout))
    outside = tmp_path / "layout.json"
    outside.write_text(json.dumps(layout))
    # Execute the actual wrapper block. Replace only the staged boot launch seam;
    # the helper runs the real formal_inputs CLI with an audit hook armed.
    wrapper = (REPO / "tools/formal/bench7v3f.sh").read_text()
    start = wrapper.index('if [ "${BENCH_FORMAL_RECEIPT:-0}" = 1 ]; then')
    end = wrapper.index('IFS=: read -r _db _dir IC', start)
    shim = tmp_path / "audit_boot.py"
    shim.write_text('''import json, os, pathlib, subprocess, sys
import easyicu
origin = pathlib.Path(easyicu.__file__).resolve()
head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
print('easyicu:', origin, 'HEAD:', head, flush=True)
assert origin.is_relative_to(pathlib.Path(REPO) / 'src')
assert head == CLAIM
def audit(event, args):
    if event == 'open' and args and isinstance(args[0], str) and os.path.realpath(args[0]) == CONFIG:
        with open(LOG, 'a') as out:
            out.write('configuration content open\\n')
sys.addaudithook(audit)
from easyicu.formal_inputs import main
sys.exit(main(sys.argv[3:]))
'''.replace("REPO", repr(str(REPO))).replace("CLAIM", repr(os.environ["CLAIMED_SHA"]))
        .replace("CONFIG", repr(str(inside))).replace("LOG", repr(str(tmp_path / "opens"))))
    env = dict(os.environ, PYTHONPATH=f"{REPO / 'src'}:{REPO}:{os.environ['RUN_WW3_GUARD']}")
    for candidate, allowed in ((inside, False), (outside, True)):
        run = tmp_path / ("outside_run" if allowed else "inside_run")
        run.mkdir()
        (run / ".tmp").mkdir()
        # Ensure the cmp branch is exercised if the pre-read check is absent.
        (run / "REQUIRED_LAYOUT.json").write_text(json.dumps(layout))
        shell = '''set -eu
R=$1; RL=$2; PY=$3; SHIM=$4; FT=$R/.formal_tools
BENCH_REQUIRED_LAYOUT=$RL; BENCH_FORMAL_RECEIPT=1
BOOT=(env "PYTHONPATH=$PYTHONPATH" "RUN_WW3_GUARD=$RUN_WW3_GUARD" "$PY" "$SHIM")
cmp() { echo cmp >> "$R/reads"; command cmp "$@"; }
cp() { echo cp >> "$R/reads"; command cp "$@"; }
shift 4
''' + wrapper[start:end]
        result = subprocess.run(["bash", "-c", shell, "test", str(run), str(candidate),
                                 sys.executable, str(shim), f"hirid:{root}:144-145"],
                                env=env, capture_output=True, text=True)
        if allowed:
            assert result.returncode == 0, result.stderr
            assert (run / "reads").read_text().strip() == "cmp"
        else:
            assert result.returncode == 2
            assert "formal_configuration_inside_input" in result.stderr
            assert not (run / "reads").exists()
            assert not (tmp_path / "opens").exists()
