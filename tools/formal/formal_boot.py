"""Boot for every Python step of a formal run (delta32p ruling; the BUILDER_BOOT pattern as
corrected by delta34s).

usage: python -I -S -P formal_boot.py TOOLS CLONE SHA SITE_PACKAGES ORIGIN_OUT SCRIPT [ARGS...]

The wrapper runs each formal Python tool (the extraction and snapshot
drivers, the receipt) only as a staged copy in TOOLS (the run's
``.formal_tools``), and only through this boot, itself a staged copy:

* ``-I``: no PYTHON* variable (PYTHONPATH, PYTHONHOME, PYTHONSTARTUP) and
  no user site; ``-S``: no site hook runs (no sitecustomize/usercustomize, no
  ``.pth`` line, so no editable install or coverage hook redirects an
  import); ``-P``: neither the script's directory nor the cwd is prepended;
  the wrapper also runs every step under ``env -i``;
* the FIRST statement (after ``import sys`` and the frozen ``os``) cuts
  sys.path to the interpreter's standard library under ``sys.base_prefix``,
  before any other import; then the import path is set explicitly: TOOLS, the clone's ``src``,
  that standard library and the named venv SITE_PACKAGES;
* before the script runs, ``easyicu`` must import from CLONE/src and CLONE's
  HEAD must be SHA, and SCRIPT must be a regular file directly in TOOLS;
* the staged-tool allowlist and digests come from SHA's sealed
  ``FORMAL_TOOLS.json``; extra files in TOOLS grant no import permission;
* after it ends (normally or by ``sys.exit``), every loaded ``easyicu.*``
  module must come from CLONE/src, every ``duckdb.*``/``pyarrow.*`` module
  from SITE_PACKAGES, every module named like a staged tool from that tool's
  own source in TOOLS (nothing else from TOOLS: no sourceless ``.pyc``), and
  no module file from anywhere outside those four roots;
* TMPDIR must name a writable directory (the wrapper's private one under
  the run root); tempfile.tempdir is pinned to its realpath before a write
  probe, with no fallback. Spawned extraction workers repeat this binding
  and check their own modules before and after their target.

Any mismatch is exit 97 (``import_origin_mismatch``). The record (paths,
head, modules, mismatched, complete) is written to ORIGIN_OUT for the
receipt. A step can check itself while running through the
``formal_boot_live`` module this boot registers (the receipt does, before it
writes RUN_RECEIPT.json).
"""

import sys
import os  # frozen in the interpreter (never looked up on sys.path); needed for realpath

_BASE = os.path.realpath(sys.base_prefix) + os.sep
if __name__ == "__main__":
    sys.path[:] = [p for p in sys.path if p and os.path.realpath(p).startswith(_BASE)]  # stdlib only

import hashlib  # noqa: E402
import json  # noqa: E402  (every other import after the cut)
import runpy  # noqa: E402
import stat  # noqa: E402
import subprocess  # noqa: E402
import tempfile  # noqa: E402
import types  # noqa: E402
import uuid  # noqa: E402

ORIGIN_EXIT = 97


def bind_private_temp():
    """Bind before the first tempfile call; an unusable private directory is fatal."""
    value = os.environ.get("TMPDIR", "")
    if not os.path.isabs(value) or not os.path.isdir(value):
        raise OSError("formal_tmpdir_unset")
    private = os.path.realpath(value)
    tempfile.tempdir = private  # Never let tempfile search its fallback candidates.
    fd, path = tempfile.mkstemp(prefix=".formal_write_probe_", dir=private)
    os.close(fd)
    os.unlink(path)
    return private


def sealed_tool_names(tools, clone, sha):
    """The allowlist is the verified commit's manifest, never a directory listing."""
    raw = subprocess.check_output(["git", "cat-file", "blob",
        sha + ":tools/formal/FORMAL_TOOLS.json"], cwd=clone)
    manifest = json.loads(raw)
    if manifest.get("schema") != "easyicu.formal_tools/1":
        raise ValueError("formal_tools_manifest_invalid")
    names = set()
    for entry in manifest["tools"]:
        name = entry["name"]
        if not isinstance(name, str) or name in names or name in ("", ".", "..") or (
                os.path.basename(name) != name):
            raise ValueError("formal_tools_manifest_invalid")
        path = os.path.join(tools, name)
        if not stat.S_ISREG(os.lstat(path).st_mode) or os.path.islink(path):
            raise ValueError("formal_tool_not_staged")
        with open(path, "rb") as fh:
            digest = hashlib.sha256(fh.read()).hexdigest()
        if digest != entry["sha256"]:
            raise ValueError("formal_tool_sha_mismatch")
        names.add(name)
    return {n[:-3] for n in names if n.endswith(".py")}


def check_modules(record, tool_names, allowed):
    tools, src, site = record["tools"], record["src"], record["site_packages"]
    roots = {"easyicu": [src], "duckdb": [site], "pyarrow": [site]}
    real = os.path.realpath

    def under(path, dirs):
        return any(os.path.commonpath((path, d)) == d and path != d for d in dirs)

    mods, bad = {}, []
    for name, mod in list(sys.modules.items()):
        raw_file = getattr(mod, "__file__", None)
        f = real(raw_file) if isinstance(raw_file, str) else None
        top = name.split(".")[0]
        if f is not None:
            mods[name] = f
        # This exception is bound to the verified script, before generic root rules.
        if top == "__mp_main__" and (name != "__mp_main__" or f is None or (
                f != real(record["script"]))):
            bad.append(name)
        elif top in roots:
            if (f is None and top == "easyicu") or (f is not None and not under(f, roots[top])):
                bad.append(name)
        elif f is not None and under(f, [tools]):
            stem = os.path.basename(f)[:-3] if f.endswith(".py") else None
            if stem not in tool_names or f != os.path.join(tools, stem + ".py") or (
                    top not in (stem, "__main__", "__mp_main__")):
                bad.append(name)
        elif top in tool_names or (f is not None and not under(f, allowed)):
            bad.append(name)
    record["modules"], record["mismatched"] = mods, sorted(set(bad))
    return record["mismatched"]


def worker_origin(initial_ok):
    """Run the boot's full check in this worker, at startup and after its target."""
    context = json.loads(os.environ["FORMAL_BOOT_CONTEXT"])
    record = dict(context["record"], complete=False, modules={}, mismatched=[],
                  pid=os.getpid(), isolated=sys.flags.isolated, no_site=sys.flags.no_site,
                  safe_path=sys.flags.safe_path, sys_path=list(sys.path),
                  worker_id=uuid.uuid4().hex,
                  stream_attempt=json.loads(os.environ.get("FORMAL_STREAM_ATTEMPT", "null")))
    out = os.path.join(os.environ["FORMAL_WORKER_ORIGIN_DIR"],
                       f"{os.getpid()}-{record['worker_id']}.json")

    def check(complete=False):
        try:
            record["tmpdir"] = bind_private_temp()
            record["head"] = subprocess.check_output(["git", "rev-parse", "HEAD"],
                cwd=context["clone"], text=True).strip()
            names = sealed_tool_names(record["tools"], context["clone"], record["sha"])
            import easyicu
            record["easyicu"] = record["path"] = os.path.realpath(easyicu.__file__)
            check_modules(record, names, context["allowed"])
            record["ok"] = bool(initial_ok and record["head"] == record["sha"] and (
                record["isolated"] and record["no_site"] and record["safe_path"]) and (
                not record["mismatched"]))
        except Exception as exc:
            record["ok"] = False
            record["error"] = type(exc).__name__
        record["complete"] = complete
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(record, fh, sort_keys=True)
        if not record["ok"]:
            os._exit(ORIGIN_EXIT)

    check()
    # multiprocessing uses os._exit, so atexit handlers do not run in workers.
    from multiprocessing.util import Finalize
    Finalize(None, check, args=(True,), exitpriority=-100)


def main() -> None:
    if len(sys.argv) < 7:
        print("usage: formal_boot.py TOOLS CLONE SHA SITE_PACKAGES ORIGIN_OUT SCRIPT [ARGS...]",
              file=sys.stderr)
        sys.exit(2)
    tools_arg, clone, sha, site_arg, out, script = sys.argv[1:7]
    real = os.path.realpath
    tools, src, site = real(tools_arg), real(os.path.join(clone, "src")), real(site_arg)
    base = real(sys.base_prefix) + os.sep
    others = tuple(r + os.sep for r in (tools, src, site))
    stdlib = [p for p in sys.path if p and real(p).startswith(base)
              and not real(p).startswith(others)]
    sys.path[:] = [tools, src, *stdlib, site]
    sys.dont_write_bytecode = True
    allowed = [tools, src, *(real(p) for p in stdlib), site]
    try:
        private = bind_private_temp()
        tool_names = sealed_tool_names(tools, clone, sha)
    except (OSError, ValueError, TypeError, KeyError, subprocess.SubprocessError) as exc:
        print(f"formal_tmpdir_unset or formal_tools_invalid: {exc}", file=sys.stderr)
        sys.exit(ORIGIN_EXIT)
    record = {"tools": tools, "src": src, "site_packages": site, "sha": sha,
              "tmpdir": private,
              "sys_path": list(sys.path), "script": None, "path": None, "head": None,
              "modules": {}, "mismatched": [], "complete": False}

    def check():
        return check_modules(record, tool_names, allowed)

    def dump():
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(record, fh, sort_keys=True)

    path = os.path.abspath(script)
    try:
        # A regular file (never a link) directly in TOOLS.
        staged = stat.S_ISREG(os.lstat(path).st_mode) and real(os.path.dirname(path)) == tools \
            and path.endswith(".py") and os.path.basename(path)[:-3] in tool_names
    except OSError:
        staged = False
    record["script"] = path if staged else None
    try:
        import easyicu
        record["path"] = real(easyicu.__file__)
        record["head"] = subprocess.run(["git", "rev-parse", "HEAD"], cwd=clone,
                                        capture_output=True, text=True).stdout.strip()
        check()
    finally:
        dump()
    if not staged or not record["path"].startswith(src + os.sep) or record["head"] != sha \
            or record["mismatched"]:
        print("import_origin_mismatch (before the script)", file=sys.stderr)
        sys.exit(ORIGIN_EXIT)
    # A spawned worker (a fresh interpreter) checks its own origin against this.
    os.environ["FORMAL_BOOT_SRC"] = src
    os.environ["FORMAL_BOOT_CONTEXT"] = json.dumps({"record": record, "clone": real(clone),
                                                  "allowed": allowed})
    live = types.ModuleType("formal_boot_live")
    live.check, live.record = check, record
    sys.modules["formal_boot_live"] = live
    sys.argv = [path, *sys.argv[7:]]
    code = 0
    try:
        runpy.run_path(path, run_name="__main__")
    except SystemExit as exc:
        code = exc.code
    finally:
        check()
        record["complete"] = True
        dump()
    if record["mismatched"]:
        print("import_origin_mismatch " + " ".join(record["mismatched"][:5]), file=sys.stderr)
        sys.exit(ORIGIN_EXIT)
    sys.exit(code)


if __name__ == "__main__":
    main()
