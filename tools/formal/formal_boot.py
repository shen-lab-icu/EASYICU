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
* after it ends (normally or by ``sys.exit``), every loaded ``easyicu.*``
  module must come from CLONE/src, every ``duckdb.*``/``pyarrow.*`` module
  from SITE_PACKAGES, every module named like a staged tool from that tool's
  own source in TOOLS (nothing else from TOOLS: no sourceless ``.pyc``), and
  no module file from anywhere outside those four roots.

Any mismatch is exit 97 (``import_origin_mismatch``). The record (paths,
head, modules, mismatched, complete) is written to ORIGIN_OUT for the
receipt. A step can check itself while running through the
``formal_boot_live`` module this boot registers (the receipt does, before it
writes RUN_RECEIPT.json).
"""

import sys
import os  # frozen in the interpreter (never looked up on sys.path); needed for realpath

_BASE = os.path.realpath(sys.base_prefix) + os.sep
sys.path[:] = [p for p in sys.path if p and os.path.realpath(p).startswith(_BASE)]  # stdlib only

import json  # noqa: E402  (every other import after the cut)
import runpy  # noqa: E402
import stat  # noqa: E402
import subprocess  # noqa: E402
import types  # noqa: E402

ORIGIN_EXIT = 97


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
    tool_names = {n[:-3] for n in os.listdir(tools) if n.endswith(".py")}
    roots = {"easyicu": [src], "duckdb": [site], "pyarrow": [site]}
    allowed = [tools, src, *(real(p) for p in stdlib), site]
    record = {"tools": tools, "src": src, "site_packages": site, "sha": sha,
              "sys_path": list(sys.path), "script": None, "path": None, "head": None,
              "modules": {}, "mismatched": [], "complete": False}

    def under(path, dirs):
        return any(path.startswith(d + os.sep) for d in dirs)

    def check():
        mods, bad = {}, []
        for name, mod in list(sys.modules.items()):
            f = getattr(mod, "__file__", None)
            f = real(f) if isinstance(f, str) else None
            top = name.split(".")[0]
            if top in roots:
                mods[name] = f
                if (f is None and top == "easyicu") or (f is not None and not under(f, roots[top])):
                    bad.append(name)
            elif f is not None and under(f, [tools]):
                # From TOOLS only a staged tool's own source (never a .pyc or another file),
                # under its own name or as __main__ (this boot, the script).
                stem = os.path.basename(f)[:-3] if f.endswith(".py") else None
                if stem not in tool_names or f != os.path.join(tools, stem + ".py") or (
                        top not in (stem, "__main__")):
                    bad.append(name)
            elif top in tool_names or (f is not None and not under(f, allowed)):
                bad.append(name)
        record["modules"] = mods
        record["mismatched"] = sorted(set(bad))
        return record["mismatched"]

    def dump():
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(record, fh, sort_keys=True)

    path = os.path.abspath(script)
    try:
        # A regular file (never a link) directly in TOOLS.
        staged = stat.S_ISREG(os.lstat(path).st_mode) and real(os.path.dirname(path)) == tools
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
