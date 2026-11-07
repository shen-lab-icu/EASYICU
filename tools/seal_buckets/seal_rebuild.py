#!/usr/bin/env python3
"""Seal-tip rebuild of the formal bucket roots (lead-r13 seal plan, supervisor approval 2026-11-06).

  (A) build1  rebuild every formal bucket root ONCE at the seal SHA with the float-fidelity builder:
              strict QC per table, footer scan per root; any failure stops the seal. CareVue builds a
              bucket root and then assembles the formal link farm mimic_cv_dir_r13 around it.
  (B) build2  a second, fully independent build of every root from the same SHA (twin roots);
      compare v3 compare (bucket_compare.py) build1 vs build2: every part must be IDENTICAL.
  (C) the per-DB 20/20 formal-start gate is not run by this tool.

Usage (entry point: seal_rebuild.sh; never launched without coordinator-assigned cores):

  seal_rebuild.sh SEAL_SHA build1|build2|compare --cores sic=a-b --cores eicu=a-b ... [--only sic,eicu]
                  [--dry-run [--probe]] [--config CONFIG.json]

  --dry-run  checks the SHA, the builder, cores, every current root/bucket/parameter, the link
             containment of everything that would be mirrored, and that no target exists. Builds nothing.
  --probe    (with --dry-run) clones the SHA and runs its builder on tiny synthetic tables in the work
             directory, applying the same strict QC (float fidelity, builder_code == SHA, clean).

Exit codes: 0 every selected DB PASS (or DRY_RUN_OK / IDENTICAL), 1 any typed failure, 2 usage.
Every failure is a typed reason in the status file and on stdout; nothing is swallowed. The exit code is
decided from structured verdicts only (never from status text); any untyped exception is a failure.
One run per seal SHA and mode: exclusive fcntl locks ($WORK_ROOT/.seal_<sha9>_<mode>.lock and, unless
--dry-run, the same name in b6_root, the real target space) are taken before planning; a second entry is
refused (seal_run_locked).

Review round 2 (delta26t) contracts:
  * checkouts and scratch are created exclusively (an existing path or a link below the tool's roots is
    refused); scratch carries this run's owner token and only owned scratch is removed;
  * the builder runs as `python -I -S -c BUILDER_BOOT <builder>` (builder_argv), with cwd outside the
    staging root: no PYTHONPATH, cwd or site hook adds code; the boot first resets the path to the
    standard library, then adds the clone's src and the interpreter's site-packages (SEAL_SITE_PACKAGES,
    named explicitly; the lookup itself runs -I -S with a scrubbed env); easyicu must import from that clone at the seal SHA, and after the builder ends every
    easyicu.* module must come from the clone and every duckdb.*/pyarrow.* module from that
    site-packages (builder_import_origin_mismatch otherwise); the receipt records import_origin {path,
    head, modules, mismatched}; footer_scan/bucket_compare/negzero_table never import easyicu;
  * conversion_manifest.json is always an owned copy of the bytes verified against the plan-time sha256
    (read once; a link included), and every link --
    and everything reachable through a directory link -- must stay inside the allowed roots;
  * a failure after the rename leaves .<target>.seal_REFUSED and a FAILED receipt; compare (and any
    acceptance) goes through accept_root: external PASS receipt, bound in-root receipt, no refusal,
    read-only; CareVue also needs its farm's own in-root receipt.

Per DB and build (order SIC -> eICU -> CareVue -> MIIV -> AUMC; each starts once its cores are free and
the memgate admits it, then DBs run in parallel):
  1. one fresh clone of SEAL_SHA per build (build1 and build2 never share a checkout); HEAD must equal the
     SHA and `git status --porcelain --untracked-files=all --ignored` must be empty, before AND after;
  2. a hidden staging root  b6_root/.<target>.seal_staging  (refused if it or the target exists):
     every entry of the current formal root except the rebuilt buckets, as symlinks (per-file links inside
     real dirs; every link target must resolve inside the allowed roots); conversion_manifest.json is
     copied with data_path re-pointed to the target (its derived_from sha256 is re-verified first);
  3. the builder with exactly the parameters of the current bucket (asserted against its manifest and
     receipt), in a minimal environment built from scratch: its own HOME / EASYICU_HOME / XDG_CACHE_HOME /
     TMPDIR / DuckDB spill under b6_root/_scratch/seal_<sha9>/<build>/<db>/, PYTHONPATH = that build's
     checkout; no value may name the other build's checkout, scratch or targets (build_env_crosses_builds);
     confined by systemd-run --scope (MemoryMax, MemorySwapMax=0) + taskset + a timeout that stops the scope;
  4. strict QC per bucket: builder exit 0 (it fails typed on bucket_float_fidelity_mismatch), _COMPLETE,
     float_fidelity recorded (dictionary encoding disabled whenever there are float columns, columns =
     float columns, receipt == manifest), builder_code present with commit == SEAL SHA and dirty == false
     (builder_code_missing / builder_code_not_seal / builder_code_dirty), inventory rows == receipt rows,
     layout parameters equal to the current bucket's;
  5. footer_scan.py over the whole staging root; its per-bucket footer row totals must equal the QC rows;
  6. ORDER OF THE LAST STEPS (no partially installed root is ever visible under its final name):
       a. after the builds, everything under the staging root is made read-only; each bucket manifest
          must still be the one QC recorded, then the footer scan runs on the frozen tree, and each new
          bucket must satisfy the full storage contract (bucket_compare.validate_contract; otherwise
          bucket_contract_invalid <table>: <contract reason>);
       b. the in-root receipt <root>/_SEAL_BUILD_RECEIPT.json is written as the LAST content step;
       c. the staging root itself is made read-only and the whole tree is verified (read-only, in-root
          receipt sha) -- the final name never exists writable or unverified;
       d. a durable pending marker b6_root/.<target>.seal_PENDING is written (O_EXCL, fsync'd);
       e. one atomic NO-REPLACE rename staging -> target (renameat2 RENAME_NOREPLACE on Linux, renamex_np
          RENAME_EXCL on macOS; without one the publish is refused, rename_noreplace_unavailable); an
          existing target, even an empty directory that just appeared, is target_exists;
       f. the external receipt $WORK/<build>/receipt_<db>.json is written; then the commit removes the
          pending marker. accept_root refuses a root with a pending (root_pending) or refusal marker.
     A failure before (e) leaves only the hidden staging root; a crash between (e) and (f) leaves a
     read-only, verified root that stays refused (pending) until an operator decides.
  CareVue (build1 only, after its bucket root PASSES): the farm b6_root/mimic_cv_dir_r13 is assembled the
  same way (staging, ro, rename): the current farm's flat links and per-file dirs replicated exactly,
  chartevents_bucket -> the new build1 bucket, conversion_manifest.json with data_path re-pointed; refused
  if mimic_cv_dir_r13 already exists; every link target must resolve inside the allowed roots.
Writes: only new roots/staging under b6_root, b6_root/_scratch/seal_<sha9>/, and the work directory.
Never databases/, the CareVue view, or any current root. Receipts hold aggregates and paths only.

Operator runbook (there is no automatic resume; a refused root stays refused until an operator acts):
  1. Find the marker beside the target in b6_root: .<target>.seal_PENDING (renamed, never committed:
     a crash between the rename and the commit) or .<target>.seal_REFUSED (verification failed after
     the rename). First make sure no run of that SHA and mode is alive: no seal_rebuild process, no
     seal-<sha9>-<mode>-* scope, and the .seal_<sha9>_<mode>.lock files in work_root and b6_root free.
  2. Check both receipts: the in-root <target>/_SEAL_BUILD_RECEIPT.json (verdict PASS, seal_sha, build,
     db) and the external $WORK_ROOT/seal_<sha9>/<build>/receipt_<db>.json (verdict PASS, no failure,
     install.read_only_verified, install.in_root_receipt_sha256 == sha256 of the in-root receipt).
  3. COMMIT only when the target carries a pending marker, no refusal marker, and both receipts pass
     every acceptance check: operator_commit() re-runs accept_root (pending aside), removes the pending
     marker and accepts the root again (one line; arguments: tool, target, external receipt, seal sha,
     build, db):
python3 -I -c 'import runpy,sys;from pathlib import Path;m=runpy.run_path(sys.argv[1]);print(m["operator_commit"](Path(sys.argv[2]),Path(sys.argv[3]),*sys.argv[4:])["verdict"])' tools/seal_buckets/seal_rebuild.py TARGET RECEIPT SHA BUILD DB
  4. REMOVE in every other case (a refusal marker, a FAILED, missing or mismatched receipt): the target
     and its markers are deleted by the operator (make the tree writable first), then the run is
     repeated. The tool never commits a refused root and never removes anything on its own.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import fcntl
import glob
import hashlib
import importlib.util
import json
import os
import re
import shlex
import shutil
import stat
import subprocess
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

EXIT_OK, EXIT_FAIL, EXIT_USAGE = 0, 1, 2
MODES = ("build1", "build2", "compare")
BUILDER_REL = "scripts/build_itemid_bucket_cache.py"
BUILDER_MARK = "bucket_float_fidelity_mismatch"
FLOAT_TYPES = frozenset({"FLOAT", "DOUBLE", "REAL", "FLOAT4", "FLOAT8"})
SKIP_RE = re.compile(
    r"^(_LEAD_R13_BUILD_RECEIPT\.json|_SEAL_BUILD_RECEIPT\.json|_lead_r13_build_.*\.log|_unused_.*|\..*)$"
)
CONVERSION_MANIFEST = "conversion_manifest.json"
IN_ROOT_RECEIPT = "_SEAL_BUILD_RECEIPT.json"
REFUSAL_SUFFIX = ".seal_REFUSED"
PENDING_SUFFIX = ".seal_PENDING"
OWNER_TOKEN = ".seal_owner"
OK_VERDICTS = frozenset({"PASS", "IDENTICAL", "DRY_RUN_OK"})
ADOPTION_RECEIPT = "_BUCKET_ADOPTION_RECEIPT.json"
SOURCE_ORDINAL = "_easyicu_source_ordinal"
FORBIDDEN_CORES = ((5, 105), (146, 179), (224, 1 << 20))
ORIGIN_EXIT = 97
# Launches the seal clone's builder like `python <builder>` (sys.path[0] = scripts/), under `python -I -S`:
# no PYTHONPATH, no current directory and no site hook (sitecustomize, usercustomize, .pth) can add code.
# The boot's first statements (only the builtin sys module is touched) reset sys.path to the standard
# library under sys.base_prefix; then the clone's src and the interpreter's own site-packages (named by
# the seal as SEAL_SITE_PACKAGES; its .pth lines are not executed) are added, and only then is anything
# imported. Before the builder runs, `easyicu` must import from the clone's src at the seal SHA; after it
# ends (normally or by sys.exit), every loaded easyicu.* module must come from the clone's src and every
# duckdb.*/pyarrow.* module from that site-packages. Any mismatch is exit 97
# (builder_import_origin_mismatch); the record (paths, head, modules) goes to SEAL_ORIGIN_OUT.
BUILDER_BOOT = r"""
import sys
_lib = sys.base_prefix + "/" + sys.platlibdir + "/python%d.%d" % sys.version_info[:2]
sys.path[:] = [sys.base_prefix + "/" + sys.platlibdir + "/python%d%d.zip" % sys.version_info[:2],
               _lib, _lib + "/lib-dynload"]
import json, os, runpy, subprocess
clone, sha, out = os.environ["SEAL_CLONE"], os.environ["SEAL_SHA"], os.environ["SEAL_ORIGIN_OUT"]
real = os.path.realpath
src = real(os.path.join(clone, "src"))
site = [real(os.environ["SEAL_SITE_PACKAGES"])]
sys.path[:] = [src, *sys.path, *site]
roots = {"easyicu": [src], "duckdb": site, "pyarrow": site}
record = {"path": None, "head": None, "sys_path": list(sys.path), "site_packages": site,
          "modules": {}, "mismatched": [], "complete": False}


def check():
    mods = {}
    for name, mod in list(sys.modules.items()):
        top = name.split(".")[0]
        if top in roots:
            f = getattr(mod, "__file__", None)
            mods[name] = real(f) if f else None
    record["modules"] = mods
    record["mismatched"] = sorted(
        n for n, f in mods.items()
        if (f is None and n.startswith("easyicu"))
        or (f is not None and not any(f.startswith(r + os.sep) for r in roots[n.split(".")[0]]))
    )


def dump():
    with open(out, "w") as fh:
        json.dump(record, fh, sort_keys=True)


try:
    import easyicu
    record["path"] = real(easyicu.__file__)
    record["head"] = subprocess.run(["git", "rev-parse", "HEAD"], cwd=clone, capture_output=True,
                                    text=True).stdout.strip()
    check()
finally:
    dump()
if not (record["path"] or "").startswith(src + os.sep) or record["head"] != sha or record["mismatched"]:
    print("builder_import_origin_mismatch", file=sys.stderr)
    sys.exit(97)
builder = sys.argv[1]
sys.argv = sys.argv[1:]
sys.path.insert(0, os.path.dirname(os.path.abspath(builder)))
code = 0
try:
    runpy.run_path(builder, run_name="__main__")
except SystemExit as exc:
    code = exc.code
finally:
    check()
    record["complete"] = True
    dump()
if record["mismatched"]:
    print("builder_import_origin_mismatch " + " ".join(record["mismatched"][:5]), file=sys.stderr)
    sys.exit(97)
sys.exit(code)
"""

_B6 = "/home/zhuhb/workspace/claude-ws/_data/b6_root"
_BL = "/home/zhuhb/easyicu_dev9/demo_sources/bench_local"
_CV_VIEW = "/home/zhuhb/workspace/claude-ws/_data/v6_97f508c6/refresh578/mimicview/raw"
_HASH64 = "--key itemid --scheme hash --modulus 64 --row-group-target 131072 --threads 2 --memory-limit 2GB"
PRODUCTION = {
    "b6_root": _B6,
    "work_root": f"{_BL}/diag_build",
    "mirror": "/home/zhuhb/easyicu_dev9/codex_ws/_mirror.git",
    "python": "/home/zhuhb/easyicu_dev9/venv311/bin/python",
    "allowed_roots": [_B6, "/home/zhuhb/workspace/databases", _CV_VIEW],
    "confine": "systemd",
    "memgate": {"lock": f"{_BL}/memgate4.lock", "module_dir": _BL, "budget_gib": 260},
    "order": ["sic", "eicu", "cv", "miiv", "aumc"],
    "dbs": {
        "sic": {
            "current_root": f"{_B6}/sic_108",
            "database": "sic",
            "mem_gib": 48,
            "threads": 4,
            "timeout_s": 21600,
            "tables": [
                [
                    "medication",
                    "--key DrugID --stay-key CaseID --scheme variable_stay_range --threads 4 --memory-limit 32GB",
                ],
                [
                    "data_float_h",
                    "--key DataID --scheme hash --sort-keys DataID,CaseID,Offset --threads 4 --memory-limit 32GB",
                ],
                [
                    "laboratory",
                    "--key LaboratoryID --scheme hash --sort-keys LaboratoryID,CaseID,Offset --threads 4 --memory-limit 32GB",
                ],
            ],
        },
        "eicu": {
            "current_root": f"{_B6}/eicu_var",
            "database": "eicu",
            "mem_gib": 48,
            "threads": 4,
            "timeout_s": 21600,
            "tables": [
                [
                    "nursecharting",
                    "--scheme variable_stay_range --threads 4 --memory-limit 32GB",
                ],
                ["lab", "--scheme variable_stay_range --threads 4 --memory-limit 32GB"],
                [
                    "infusiondrug",
                    "--key drugname --scheme hash --sort-keys drugname,patientunitstayid,infusionoffset --threads 4 --memory-limit 32GB",
                ],
            ],
        },
        "cv": {
            "current_root": f"{_B6}/mimic_cv_dir",
            "database": "mimic",
            "mem_gib": 12,
            "threads": 2,
            "timeout_s": 21600,
            "tables": [["chartevents", _HASH64]],
            "farm": {
                "name": "mimic_cv_dir",
                "bucket_root": "mimic_cv_bucket",
                "reference": f"{_B6}/mimic_cv_dir_r13",
            },
        },
        "miiv": {
            "current_root": f"{_B6}/miiv_r13",
            "database": "miiv",
            "mem_gib": 12,
            "threads": 2,
            "timeout_s": 21600,
            "tables": [["chartevents", _HASH64]],
        },
        "aumc": {
            "current_root": f"{_B6}/aumc_r13",
            "database": "aumc",
            "mem_gib": 48,
            "threads": 4,
            "timeout_s": 21600,
            "tables": [
                ["numericitems", "--scheme hash --threads 4 --memory-limit 32GB"]
            ],
        },
    },
}


class SealError(Exception):
    """A typed failure: ``reason`` is the stable code written to the status file."""

    def __init__(self, reason: str, detail: str = ""):
        super().__init__(f"{reason}{': ' + detail if detail else ''}")
        self.reason, self.detail = reason, detail


# ----------------------------------------------------------------------------- small helpers
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, data) -> str:
    text = json.dumps(data, indent=1, sort_keys=True) + "\n"
    path.write_text(text)
    return hashlib.sha256(text.encode()).hexdigest()


def git(args: list[str], cwd: Path | None = None, git_dir: str | None = None) -> str:
    cmd = ["git"] + (["--git-dir", git_dir] if git_dir else []) + args
    out = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if out.returncode != 0:
        raise SealError(
            "git_failed", f"{' '.join(args[:2])}: {out.stderr.strip()[:200]}"
        )
    return out.stdout


def git_dir_of(mirror: str) -> str:
    return str(Path(mirror) / ".git") if (Path(mirror) / ".git").is_dir() else mirror


def inside(path: str | Path, roots) -> bool:
    real = os.path.realpath(path)
    for root in roots:
        r = os.path.realpath(root)
        if real == r or real.startswith(r.rstrip("/") + "/"):
            return True
    return False


def parse_cores(text: str) -> tuple[int, int]:
    m = re.fullmatch(r"(\d+)-(\d+)", text or "")
    if not m or int(m.group(1)) > int(m.group(2)):
        raise SealError("cores_invalid", text or "missing")
    lo, hi = int(m.group(1)), int(m.group(2))
    for a, b in FORBIDDEN_CORES:
        if lo <= b and hi >= a:
            raise SealError("forbidden_cores", text)
    return lo, hi


def cores_busy(lo: int, hi: int) -> bool:
    """True when a pinned process of this user (affinity narrower than 100 cores) overlaps [lo, hi]."""
    for status in glob.glob("/proc/[0-9]*/status"):
        try:
            text = Path(status).read_text()
        except OSError:
            continue
        if f"\nUid:\t{os.getuid()}\t" not in text:
            continue
        m = re.search(r"Cpus_allowed_list:\s*(\S+)", text)
        if not m or "," in m.group(1):
            continue
        a, _, b = m.group(1).partition("-")
        al, ah = int(a), int(b or a)
        if ah - al < 100 and lo <= ah and hi >= al:
            return True
    return False


# ----------------------------------------------------------------------------- configuration
def load_config(path: Path | None) -> dict:
    cfg = json.loads(path.read_text()) if path else json.loads(json.dumps(PRODUCTION))
    if cfg.get("confine") not in ("systemd", "none"):
        raise SealError("config_confine_invalid", str(cfg.get("confine")))
    if cfg["confine"] == "none" and os.path.realpath(
        cfg["b6_root"]
    ) == os.path.realpath(PRODUCTION["b6_root"]):
        raise SealError("unconfined_on_production_root")
    if cfg["confine"] == "systemd" and not sys.platform.startswith("linux"):
        raise SealError("systemd_confinement_needs_linux")
    return cfg


class Seal:
    def __init__(
        self,
        cfg: dict,
        sha: str,
        mode: str,
        dry_run: bool,
        only: list[str],
        cores: dict,
    ):
        self.cfg, self.mode, self.dry_run = cfg, mode, dry_run
        self.b6 = Path(cfg["b6_root"])
        self.gitd = git_dir_of(cfg["mirror"])
        try:
            self.full = git(
                ["rev-parse", "--verify", "--quiet", f"{sha}^{{commit}}"],
                git_dir=self.gitd,
            ).strip()
        except SealError as exc:
            raise SealError("seal_sha_unknown", sha) from exc
        if not re.fullmatch(r"[0-9a-f]{40}", self.full):
            raise SealError("seal_sha_unknown", sha)
        self.s9 = self.full[:9]
        self.work = Path(cfg["work_root"]) / (
            f"seal_dryrun_{self.s9}" if dry_run else f"seal_{self.s9}"
        )
        self.scratch = self.b6 / "_scratch" / f"seal_{self.s9}"
        unknown = [d for d in only if d not in cfg["dbs"]]
        if unknown:
            raise SealError("unknown_db", ",".join(unknown))
        self.dbs = [d for d in cfg["order"] if not only or d in only]
        self.cores = cores
        self.lock = threading.Lock()
        # db -> (verdict, line): the exit code is decided from the verdict, never from line text
        self.status: dict[str, tuple[str, str]] = {}
        self.token = uuid.uuid4().hex

    # ---- naming
    def target(self, db: str, mode: str | None = None) -> Path:
        mode = mode or self.mode
        spec = self.cfg["dbs"][db]
        base = (
            spec["farm"]["bucket_root"]
            if "farm" in spec
            else Path(spec["current_root"]).name
        )
        return self.b6 / f"{base}_s{self.s9}{'_build2' if mode == 'build2' else ''}"

    def farm_target(self, db: str) -> Path | None:
        farm = self.cfg["dbs"][db].get("farm")
        return self.b6 / f"{farm['name']}_s{self.s9}" if farm else None

    def staging(self, target: Path) -> Path:
        return target.parent / f".{target.name}.seal_staging"

    def code_dir(self, mode: str) -> Path:
        return self.work / mode / "code"

    def scratch_dir(self, mode: str, db: str) -> Path:
        return self.scratch / mode / db

    def log(self, msg: str) -> None:
        line = f"{time.strftime('%H:%M:%SZ', time.gmtime())} [{self.mode}] {msg}"
        with self.lock:
            print(line, flush=True)
            if (self.work / self.mode).is_dir():
                with open(self.work / self.mode / "run.log", "a") as fh:
                    fh.write(line + "\n")

    def set_status(self, db: str, verdict: str, detail: str = "") -> None:
        if verdict not in OK_VERDICTS and verdict != "FAILED":
            raise ValueError(f"unknown verdict {verdict!r}")
        with self.lock:
            self.status[db] = (verdict, f"{db} {verdict} {detail}".rstrip())

    def verdict(self, db: str) -> str:
        return self.status.get(db, ("FAILED", ""))[0]

    # ---- builder checks
    def check_builder_mark(self) -> None:
        source = git(["show", f"{self.full}:{BUILDER_REL}"], git_dir=self.gitd)
        if BUILDER_MARK not in source:
            raise SealError("builder_lacks_float_fidelity", self.s9)

    def prepare_checkout(self, code: Path) -> Path:
        """A fresh checkout, created exclusively by this run: an existing path (a former checkout, a link
        to another build's) is refused, never reused (D26-ISOLATION)."""
        work_root = Path(self.cfg["work_root"])
        code.parent.mkdir(parents=True, exist_ok=True)
        assert_no_links_below(code.parent, work_root)
        if code.exists() or code.is_symlink():
            raise SealError("checkout_exists", str(code))
        try:
            code.mkdir()
        except FileExistsError as exc:
            raise SealError("checkout_exists", str(code)) from exc
        git(["clone", "-q", "--no-checkout", self.cfg["mirror"], str(code)])
        git(["-c", "advice.detachedHead=false", "checkout", "-q", self.full], cwd=code)
        verify_checkout(code, self.full)
        return code


def verify_checkout(code: Path, full: str) -> None:
    if not (code / ".git").exists():
        raise SealError("checkout_missing", str(code))
    head = git(["rev-parse", "HEAD"], cwd=code).strip()
    if head != full:
        raise SealError("checkout_not_at_seal", head[:9])
    # --ignored: an ignored execution artifact (bytecode, build output) is not clean either
    if git(
        ["status", "--porcelain", "--untracked-files=all", "--ignored"], cwd=code
    ).strip():
        raise SealError("checkout_not_clean", str(code))


def builder_argv(python: str, builder: Path) -> list[str]:
    """How every builder is launched: -I -S (no PYTHONPATH, no cwd, no site hooks) and the boot, never
    `python <builder>`; callers also run it with a scrubbed env and a cwd outside the staging root.
    -I ignores every PYTHON* variable, so -B is given explicitly: no bytecode is written into the
    checkout (it must stay clean, ignored files included)."""
    return [python, "-I", "-S", "-B", "-c", BUILDER_BOOT, str(builder)]


# The site-packages query runs isolated like the builder: -I -S, a scrubbed env, cwd "/", and the path
# reset to the standard library before sysconfig is imported.
_SITE_QUERY = (
    "import sys\n"
    "_lib = sys.base_prefix + '/' + sys.platlibdir + '/python%d.%d' % sys.version_info[:2]\n"
    "sys.path[:] = [_lib, _lib + '/lib-dynload']\n"
    "import sysconfig\n"
    "print(sysconfig.get_paths()['purelib'])\n"
)


def site_packages(python: str) -> str:
    """The configured interpreter's own site-packages, named explicitly for the -S builder boot: a
    venv's lib/python*/site-packages (from its pyvenv.cfg layout, nothing executed), else the
    interpreter's purelib asked under -I -S with a scrubbed env (nothing inherited)."""
    venv = Path(python).parent.parent
    if (venv / "pyvenv.cfg").is_file():
        found = sorted(venv.glob("lib/python3*/site-packages"))
        if len(found) != 1:
            raise SealError("site_packages_unresolved", str(venv))
        return str(found[0])
    done = subprocess.run(
        [python, "-I", "-S", "-c", _SITE_QUERY],
        capture_output=True,
        text=True,
        cwd="/",
        env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"},
    )
    path = done.stdout.strip()
    if done.returncode != 0 or not Path(path).is_dir():
        raise SealError("site_packages_unresolved", python)
    return path


def read_origin(path: Path) -> dict:
    try:
        rec = json.loads(path.read_text())
    except (OSError, ValueError):
        return {"path": None, "head": None}
    return rec if isinstance(rec, dict) else {"path": None, "head": None}


def origin_ok(origin: dict, code: Path, full: str) -> bool:
    """The boot's record: easyicu from the clone's src at the SHA, the builder ran to completion under
    the boot, and no easyicu.*/duckdb.*/pyarrow.* module came from anywhere else (D29-BOOT-HOOK)."""
    src = os.path.realpath(code / "src") + os.sep
    return (
        bool(origin.get("path"))
        and str(origin["path"]).startswith(src)
        and origin.get("head") == full
        and origin.get("complete") is True
        and origin.get("mismatched") == []
    )


def assert_no_links_below(path: Path, base: Path) -> None:
    """``path`` and each ancestor below ``base`` must be a real directory or file, never a symlink."""
    path, base = Path(os.path.abspath(path)), Path(os.path.abspath(base))
    try:
        rel = path.relative_to(base)
    except ValueError as exc:
        raise SealError("path_outside_tool_root", str(path)) from exc
    cur = base
    for part in rel.parts:
        cur = cur / part
        if cur.is_symlink():
            raise SealError("path_is_link", str(cur))


class RunLocks:
    def __init__(self) -> None:
        self.handles: list = []

    def close(self) -> None:
        for fh in self.handles:
            fh.close()
        self.handles = []


def acquire_run_lock(seal: "Seal") -> RunLocks:
    """Exclusive cross-process locks per seal SHA and mode, taken before planning (D26-CONCURRENCY):
    one in the work root and one in the real target space, b6_root (D29-NOREPLACE), so a run from
    another work root or config that targets the same b6 names is refused too.

    The second entry is refused instead of racing (it never waits, and never touches the first run's
    directories). Publishing itself never relies on the locks: the rename is RENAME_NOREPLACE."""
    locks = RunLocks()
    roots = [Path(seal.cfg["work_root"])]
    if not seal.dry_run:  # a dry run publishes nothing and writes nothing under b6_root
        roots.append(seal.b6)
    for root in roots:
        root.mkdir(parents=True, exist_ok=True)
        fh = open(root / f".seal_{seal.s9}_{seal.mode}.lock", "a")
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            fh.close()
            locks.close()
            raise SealError("seal_run_locked", f"{seal.s9} {seal.mode} {root}") from exc
        locks.handles.append(fh)
    return locks


def claim_scratch(scr: Path, base: Path, token: str) -> None:
    """Create a scratch directory exclusively and mark it with this run's ownership token."""
    scr.parent.mkdir(parents=True, exist_ok=True)
    assert_no_links_below(scr.parent, base)
    try:
        scr.mkdir()
    except FileExistsError as exc:
        raise SealError("scratch_exists", str(scr)) from exc
    (scr / OWNER_TOKEN).write_text(token)


def release_scratch(scr: Path, base: Path, token: str) -> None:
    """Remove a scratch directory only when it carries this run's token (never another run's)."""
    try:
        owned = (
            not scr.is_symlink()
            and inside(scr, [base])
            and (scr / OWNER_TOKEN).read_text() == token
        )
    except OSError:
        owned = False
    if owned:
        shutil.rmtree(scr)


# ----------------------------------------------------------------------------- parameters and QC
def _opts(args: str) -> dict:
    a = shlex.split(args)
    return {
        a[i]: a[i + 1]
        for i in range(len(a) - 1)
        if a[i].startswith("--") and not a[i + 1].startswith("--")
    }


def assert_params(current_bucket: Path, args: str) -> None:
    try:
        m = json.loads((current_bucket / "_BUCKET_MANIFEST.json").read_text())
        r = json.loads((current_bucket / "_BUCKET_BUILD_RECEIPT.json").read_text())
    except (OSError, ValueError) as exc:
        raise SealError(
            "current_bucket_unreadable", f"{current_bucket.name}: {type(exc).__name__}"
        ) from exc
    opt, bad = _opts(args), []

    def chk(name, want, got):
        if want is not None and str(want).lower() != str(got).lower():
            bad.append(name)

    chk("scheme", opt.get("--scheme"), m.get("scheme"))
    chk("threads", opt.get("--threads"), r.get("threads"))
    chk("memory_limit", opt.get("--memory-limit"), r.get("memory_limit"))
    chk("key", opt.get("--key"), m.get("key_column"))
    chk("stay_key", opt.get("--stay-key"), m.get("stay_column"))
    chk("modulus", opt.get("--modulus"), m.get("modulus"))
    chk("row_group_target", opt.get("--row-group-target"), m.get("row_group_target"))
    if "--sort-keys" in opt:
        chk("sort_keys", opt["--sort-keys"], ",".join(m.get("sort_keys", [])[:-1]))
    if bad:
        raise SealError("params_mismatch", f"{current_bucket.name}:{','.join(bad)}")


LAYOUT_KEYS = (
    "scheme",
    "key_column",
    "stay_column",
    "modulus",
    "row_group_target",
    "sort_keys",
)


def qc_bucket(bucket: Path, full: str, current_bucket: Path | None = None) -> dict:
    """Strict QC of one freshly built bucket; returns the aggregate record, raises on the first finding set."""
    findings: list[str] = []
    try:
        m = json.loads((bucket / "_BUCKET_MANIFEST.json").read_text())
        r = json.loads((bucket / "_BUCKET_BUILD_RECEIPT.json").read_text())
    except (OSError, ValueError) as exc:
        raise SealError(
            "bucket_manifest_missing", f"{bucket.name}: {type(exc).__name__}"
        ) from exc
    if not (bucket / "_COMPLETE").is_file():
        findings.append("bucket_complete_marker_missing")
    schema = r.get("source_schema")
    if not schema:
        findings.append("source_schema_missing")
    floats = sorted(c for c, t in (schema or []) if str(t).upper() in FLOAT_TYPES)
    ff = m.get("float_fidelity")
    if ff is None:
        findings.append("float_fidelity_missing")
    else:
        if floats and ff.get("dictionary_encoding") != "disabled":
            findings.append("float_dictionary_encoding_not_disabled")
        if sorted(ff.get("columns", {})) != floats:
            findings.append("float_fidelity_columns_mismatch")
        if r.get("float_fidelity") != ff:
            findings.append("float_fidelity_receipt_vs_manifest")
    bc = m.get("builder_code")
    if not isinstance(bc, dict) or not bc.get("commit"):
        findings.append("builder_code_missing")
    elif bc["commit"] != full:
        findings.append("builder_code_not_seal")
    elif bc.get("dirty") is not False:
        findings.append("builder_code_dirty")
    if "builder_code" in r and r["builder_code"] != bc:
        findings.append("builder_code_receipt_vs_manifest")
    # Formal roots are built, never adopted: an adopted legacy cache has no source ordinal.
    if (bucket / ADOPTION_RECEIPT).exists() or m.get("source_order") != SOURCE_ORDINAL:
        findings.append("adopted_bucket_not_formal")
    if ff is not None and ff.get("rows") != r.get("rows"):
        findings.append("float_fidelity_rows_vs_receipt")
    inventory = m.get("inventory") or []
    rows = r.get("rows")
    if rows is None or sum(int(e.get("row_count", 0)) for e in inventory) != int(rows):
        findings.append("bucket_rows_inventory_vs_receipt")
    if current_bucket is not None:
        cm = json.loads((current_bucket / "_BUCKET_MANIFEST.json").read_text())
        differ = [
            k for k in LAYOUT_KEYS if str(cm.get(k)).lower() != str(m.get(k)).lower()
        ]
        if differ:
            findings.append("params_differ_from_current:" + ",".join(differ))
    record = {
        "rows": rows,
        "parts": len(inventory),
        "elapsed_seconds": r.get("elapsed_seconds"),
        "builder_code": bc,
        "float_columns": floats,
        "float_fidelity": None
        if ff is None
        else {
            c: {
                k: v.get(k)
                for k in ("values", "nulls", "neg_zero", "pos_zero", "nan", "sha256")
            }
            for c, v in ff.get("columns", {}).items()
        },
        "manifest_sha256": sha256_file(bucket / "_BUCKET_MANIFEST.json"),
        "qc_findings": findings,
    }
    if findings:
        err = SealError(
            findings[0].split(":")[0], f"{bucket.name}: {','.join(findings)}"
        )
        err.record = record
        raise err
    return record


# ----------------------------------------------------------------------------- roots: mirror, ro, install
def manifest_source(src: Path) -> dict:
    """Pin the conversion manifest's source at plan time: the file it resolves to and its sha256."""
    real = Path(os.path.realpath(src))
    if not real.is_file():
        raise SealError("conversion_manifest_unreadable", src.parent.name)
    return {
        "source": str(real),
        "sha256": sha256_file(real),
        "linked": src.is_symlink(),
    }


def _pinned_bytes(src: Path, pinned: dict) -> bytes:
    """Read the pinned source ONCE and verify the plan-time sha256 on exactly these bytes."""
    real = Path(pinned["source"])
    if Path(os.path.realpath(src)) != real:
        raise SealError("conversion_manifest_source_changed", src.parent.name)
    try:
        data = real.read_bytes()
    except OSError as exc:
        raise SealError("conversion_manifest_unreadable", src.parent.name) from exc
    if hashlib.sha256(data).hexdigest() != pinned["sha256"]:
        raise SealError("conversion_manifest_source_changed", src.parent.name)
    return data


def _conversion_decision(
    src: Path, old_root: Path, new_root: Path, pinned: dict
) -> tuple[bytes, dict | None]:
    data = _pinned_bytes(src, pinned)
    real = Path(pinned["source"])
    try:
        c = json.loads(data)
    except ValueError as exc:
        raise SealError("conversion_manifest_unreadable", src.parent.name) from exc
    if not isinstance(c, dict):
        raise SealError("conversion_manifest_unreadable", src.parent.name)
    d = c.get("derived_from")
    if d is None and not pinned["linked"]:
        raise SealError("conversion_manifest_unbound", src.parent.name)
    if d is not None:
        if not isinstance(d, dict) or not d.get("path") or not d.get("sha256"):
            raise SealError("conversion_manifest_unbound", src.parent.name)
        if not Path(d["path"]).is_file() or sha256_file(Path(d["path"])) != d["sha256"]:
            raise SealError("conversion_manifest_derived_from_changed", src.parent.name)
    data_path = c.get("data_path")
    if data_path == str(old_root):
        c["data_path"] = str(new_root)
        return data, c
    if data_path is None or (pinned["linked"] and data_path == str(real.parent)):
        return data, None
    raise SealError("conversion_manifest_data_path_unexpected", src.parent.name)


def rewritten_conversion_manifest(
    src: Path, old_root: Path, new_root: Path, pinned: dict | None = None
) -> dict | None:
    """The mirrored conversion manifest is always an owned file (a link is read, never re-linked).

    Its source is read once and verified on those bytes (D29-CONVERSION-RACE): the file it resolves
    to must have the sha256 pinned at plan time, and a ``derived_from`` binding (always required of a
    root's own, non-linked manifest) must still match its source. Only ``data_path`` may change: the
    current root's path is re-pointed to the new root; a linked source manifest that names its own
    directory keeps it. Returns the new content, or None when the content is unchanged."""
    return _conversion_decision(
        src, old_root, new_root, pinned or manifest_source(src)
    )[1]


def write_conversion_manifest(
    src: Path, dest: Path, old_root: Path, new_root: Path, pinned: dict
) -> None:
    """Write exactly the verified bytes (or their parsed rewrite): never a second read of the source."""
    data, content = _conversion_decision(src, old_root, new_root, pinned)
    if content is None:
        with open(dest, "xb") as fh:
            fh.write(data)
    else:
        write_json(dest, content)


def plan_entries(
    src_root: Path, skip_names: set[str], allowed
) -> list[tuple[str, str, object]]:
    """What a mirror of ``src_root`` will contain: ('link', name, target) | ('dir', name, [(file, target)])
    | ('manifest', name, None). Every link target must resolve inside the allowed roots."""
    plan = []
    for entry in sorted(os.scandir(src_root), key=lambda e: e.name):
        name = entry.name
        if name in skip_names or SKIP_RE.match(name):
            continue
        path = Path(entry.path)
        if name == CONVERSION_MANIFEST:
            # the reserved name is dispatched first, link or not: it is always copied and re-verified
            plan.append(("manifest", name, manifest_source(path)))
            continue
        if entry.is_symlink():
            if not path.exists():
                raise SealError("mirror_link_dangling", name)
            try:
                check_closure(path, allowed, "link_outside_allowed_roots")
            except SealError as exc:
                raise SealError("link_outside_allowed_roots", name) from exc
            plan.append(("link", name, os.readlink(path)))
        elif entry.is_dir():
            files = []
            for f in sorted(os.scandir(path), key=lambda e: e.name):
                fp = Path(f.path)
                if os.path.isdir(fp):
                    raise SealError("nested_dir_unsupported", f"{name}/{f.name}")
                try:
                    check_closure(fp, allowed, "link_outside_allowed_roots")
                except SealError as exc:
                    raise SealError(
                        "link_outside_allowed_roots", f"{name}/{f.name}"
                    ) from exc
                files.append(
                    (
                        f.name,
                        os.readlink(fp) if os.path.islink(fp) else os.path.realpath(fp),
                    )
                )
            plan.append(("dir", name, files))
        elif entry.is_file():
            if not inside(path, allowed):
                raise SealError("link_outside_allowed_roots", name)
            plan.append(("link", name, os.path.realpath(path)))
        else:
            raise SealError("mirror_entry_unsupported", name)
    return plan


def materialize(
    plan,
    staging: Path,
    src_root: Path,
    final: Path,
    extra_links: dict[str, str] | None = None,
) -> None:
    staging.mkdir()
    for kind, name, payload in plan:
        if kind == "link":
            os.symlink(payload, staging / name)
        elif kind == "dir":
            (staging / name).mkdir()
            for fname, target in payload:
                os.symlink(target, staging / name / fname)
        else:
            write_conversion_manifest(
                src_root / name, staging / name, src_root, final, payload
            )
    for name, target in (extra_links or {}).items():
        os.symlink(target, staging / name)


def check_closure(path: str | Path, allowed, reason: str) -> None:
    """``path`` and, when it resolves to a directory, every effective descendant (following links,
    each real directory visited once) must exist and resolve inside the allowed roots."""
    seen: set[str] = set()
    stack = [str(path)]
    while stack:
        p = stack.pop()
        if not os.path.exists(p) or not inside(p, allowed):
            raise SealError(reason, str(p))
        if os.path.isdir(p):
            real = os.path.realpath(p)
            if real in seen:
                continue
            seen.add(real)
            stack.extend(e.path for e in os.scandir(p))


def check_links(
    root: Path, allowed, reason: str = "link_outside_allowed_roots"
) -> None:
    """Every symlink under ``root`` (files and directories) must resolve inside the allowed roots,
    and so must everything reachable through a directory link."""
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        for name in [*dirnames, *filenames]:
            p = os.path.join(dirpath, name)
            if os.path.islink(p):
                try:
                    check_closure(p, allowed, reason)
                except SealError as exc:
                    raise SealError(reason, os.path.relpath(p, root)) from exc


def make_read_only(root: Path, include_root: bool) -> None:
    for dirpath, dirnames, filenames in os.walk(root, topdown=False, followlinks=False):
        for name in filenames:
            p = os.path.join(dirpath, name)
            if not os.path.islink(p):
                os.chmod(p, stat.S_IMODE(os.lstat(p).st_mode) & ~0o222)
        for name in dirnames:
            p = os.path.join(dirpath, name)
            if not os.path.islink(p):
                os.chmod(p, stat.S_IMODE(os.lstat(p).st_mode) & ~0o222)
    if include_root:
        os.chmod(root, stat.S_IMODE(os.lstat(root).st_mode) & ~0o222)


def writable_entries(root: Path) -> list[str]:
    out = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        for name in [*dirnames, *filenames, *([] if dirpath != str(root) else ["."])]:
            p = os.path.join(dirpath, name) if name != "." else dirpath
            if not os.path.islink(p) and stat.S_IMODE(os.lstat(p).st_mode) & 0o222:
                out.append(os.path.relpath(p, root))
    return out


_CONTRACT = None


def check_bucket_contract(bucket: Path, table: str) -> None:
    """bucket_compare.validate_contract (the resolver's contract, ported; easyicu-free) on one new
    bucket: a violation is bucket_contract_invalid with the contract's own reason."""
    global _CONTRACT
    if _CONTRACT is None:
        spec = importlib.util.spec_from_file_location(
            "seal_bucket_contract", Path(__file__).with_name("bucket_compare.py")
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _CONTRACT = module
    try:
        _CONTRACT.validate_contract(bucket)
    except _CONTRACT.StorageIntegrityError as exc:
        raise SealError(
            "bucket_contract_invalid", f"{table}: {exc.reason} {exc.detail}".rstrip()
        ) from exc


def refusal_marker(target: Path) -> Path:
    """A persistent refusal beside a final name whose post-rename verification failed."""
    return target.parent / f".{target.name}{REFUSAL_SUFFIX}"


def pending_marker(target: Path) -> Path:
    """Written durably BEFORE the rename and removed only after the external PASS receipt: a final name
    with this marker beside it is unpublished (a crash between the two) and every consumer refuses it."""
    return target.parent / f".{target.name}{PENDING_SUFFIX}"


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _noreplace_primitive():
    """(name, call) for an atomic no-replace rename on this platform, or None."""
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        if sys.platform.startswith("linux"):
            fn = libc.renameat2  # glibc >= 2.28
            fn.argtypes = [
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_uint,
            ]
            return "renameat2", lambda a, b: fn(
                -100, a, -100, b, 1
            )  # AT_FDCWD, RENAME_NOREPLACE
        if sys.platform == "darwin":
            fn = libc.renamex_np
            fn.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
            return "renamex_np", lambda a, b: fn(a, b, 0x4)  # RENAME_EXCL
    except AttributeError:
        return None
    return None


def rename_noreplace(src: Path, dst: Path) -> str:
    """Atomically rename ``src`` to ``dst`` only if ``dst`` does not exist (D29-NOREPLACE): an existing
    target, even an empty directory created a moment earlier, is target_exists; without the primitive
    the publish is refused (rename_noreplace_unavailable), never done with a replacing rename."""
    prim = _noreplace_primitive()
    if prim is None:
        raise SealError("rename_noreplace_unavailable", sys.platform)
    name, call = prim
    if call(os.fsencode(src), os.fsencode(dst)) != 0:
        err = ctypes.get_errno()
        if err == errno.EEXIST or (err == errno.ENOTEMPTY and dst.exists()):
            raise SealError("target_exists", str(dst))
        if err in (errno.ENOSYS, errno.EINVAL, errno.ENOTSUP, errno.EOPNOTSUPP):
            raise SealError("rename_noreplace_unavailable", f"{name} errno={err}")
        raise OSError(err, os.strerror(err), str(src))
    return name


def verify_installed(root: Path, in_root_sha: str) -> None:
    left = writable_entries(root)
    if left:
        raise SealError("root_not_read_only", f"{root.name}: {len(left)} writable")
    if sha256_file(root / IN_ROOT_RECEIPT) != in_root_sha:
        raise SealError("in_root_receipt_changed", root.name)


def install(staging: Path, target: Path, in_root_receipt: dict) -> dict:
    """Publish a staged root (D29-PUBLISH): the in-root receipt is the last content step; the WHOLE
    tree, root included, is made read-only and verified BEFORE the final name exists; a durable pending
    marker is written; then one atomic no-replace rename. The final name is never visible writable or
    unverified. The caller writes the external receipt and then commit_install() removes the pending
    marker; until then (a crash in between included) every consumer refuses the root. A failure after
    the rename leaves a persistent refusal marker as well."""
    if (
        not isinstance(in_root_receipt, dict)
        or in_root_receipt.get("verdict") != "PASS"
    ):
        raise SealError("in_root_receipt_missing", target.name)
    for p in (target, refusal_marker(target), pending_marker(target)):
        if p.exists() or p.is_symlink():
            raise SealError("target_exists", str(p))
    in_root_sha = write_json(staging / IN_ROOT_RECEIPT, in_root_receipt)
    make_read_only(staging, include_root=True)
    verify_installed(staging, in_root_sha)
    try:
        fd = os.open(
            pending_marker(target), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444
        )
    except FileExistsError as exc:
        raise SealError("target_exists", str(pending_marker(target))) from exc
    with os.fdopen(fd, "w") as fh:
        json.dump({"root": str(target), "verdict": "PENDING"}, fh)
        fh.flush()
        os.fsync(fh.fileno())
    _fsync_dir(target.parent)
    primitive = rename_noreplace(staging, target)
    _fsync_dir(target.parent)
    try:
        verify_installed(target, in_root_sha)
    except (SealError, OSError) as exc:
        reason = exc.reason if isinstance(exc, SealError) else type(exc).__name__
        write_json(
            refusal_marker(target),
            {"root": str(target), "verdict": "REFUSED", "reason": reason},
        )
        raise SealError("install_verify_failed", f"{target.name}: {reason}") from exc
    return {
        "root": str(target),
        "read_only_verified": True,
        "in_root_receipt_sha256": in_root_sha,
        "rename": primitive,
    }


def operator_commit(
    target: Path, external: Path, seal_sha: str, build: str, db: str
) -> dict:
    """Runbook step 3: commit a pending root only when every acceptance check passes (pending aside)."""
    if not pending_marker(target).is_file():
        raise SealError("root_not_pending", target.name)
    accept_root(target, external, seal_sha, build, db, allow_pending=True)
    commit_install(target)
    return accept_root(target, external, seal_sha, build, db)


def commit_install(target: Path) -> None:
    """The commit: remove the pending marker once the external PASS receipt is durable."""
    pending_marker(target).unlink()
    _fsync_dir(target.parent)


def accept_root(
    root: Path,
    external: Path,
    seal_sha: str,
    build: str,
    db: str,
    *,
    allow_pending: bool = False,
) -> dict:
    """The acceptance gate for an installed root: a verified external PASS receipt, no refusal marker,
    no pending marker (only operator_commit passes allow_pending), the in-root receipt bound by sha256
    and itself PASS for this SHA, build and DB, and a read-only tree. Returns the in-root receipt."""
    try:
        ext = json.loads(external.read_text())
    except (OSError, ValueError) as exc:
        raise SealError("receipt_missing", str(external.name)) from exc
    if (
        not isinstance(ext, dict)
        or ext.get("verdict") != "PASS"
        or "failure" in ext
        or ext.get("seal_sha") != seal_sha
        or ext.get("build") != build
        or ext.get("db") != db
    ):
        raise SealError("receipt_not_pass", external.name)
    bound = (ext.get("install") or {}).get("in_root_receipt_sha256")
    if not bound or ext["install"].get("read_only_verified") is not True:
        raise SealError("receipt_unbound", external.name)
    if refusal_marker(root).exists() or refusal_marker(root).is_symlink():
        raise SealError("root_refused", root.name)
    if not allow_pending and (
        pending_marker(root).exists() or pending_marker(root).is_symlink()
    ):
        raise SealError("root_pending", root.name)
    if (
        not (root / IN_ROOT_RECEIPT).is_file()
        or sha256_file(root / IN_ROOT_RECEIPT) != bound
    ):
        raise SealError("in_root_receipt_changed", root.name)
    inner = json.loads((root / IN_ROOT_RECEIPT).read_text())
    if (
        inner.get("verdict") != "PASS"
        or "failure" in inner
        or inner.get("seal_sha") != seal_sha
        or inner.get("build") != build
        or inner.get("db") != db
    ):
        raise SealError("in_root_receipt_not_pass", root.name)
    if writable_entries(root):
        raise SealError("root_not_read_only", root.name)
    return inner


# ----------------------------------------------------------------------------- isolation and confinement
def build_env(seal: Seal, db: str, code: Path) -> dict:
    scr = seal.scratch_dir(seal.mode, db)
    t = str(seal.cfg["dbs"][db]["threads"])
    env = {
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "HOME": str(scr / "home"),
        "EASYICU_HOME": str(scr / "home" / ".easyicu"),
        "XDG_CACHE_HOME": str(scr / "home" / ".cache"),
        "TMPDIR": str(scr / "tmp"),
        "TMP": str(scr / "tmp"),
        "TEMP": str(scr / "tmp"),
        "EASYICU_DUCKDB_TEMP_DIR": str(scr / "spill"),
        "PYTHONPATH": str(code / "src"),
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "SEAL_CLONE": str(code),
        "SEAL_SHA": seal.full,
        "SEAL_SITE_PACKAGES": site_packages(seal.cfg["python"]),
        # a private bytecode cache: no build reads another build's (or the checkout's) cached bytecode
        "PYTHONPYCACHEPREFIX": str(scr / "pycache"),
        "OMP_NUM_THREADS": t,
        "POLARS_MAX_THREADS": t,
        "RAYON_NUM_THREADS": t,
        "OPENBLAS_NUM_THREADS": t,
    }
    if seal.cfg["confine"] == "systemd":
        for key in ("XDG_RUNTIME_DIR", "DBUS_SESSION_BUS_ADDRESS"):
            if os.environ.get(key):
                env[key] = os.environ[key]
    check_isolation(env, foreign_paths(seal))
    return env


def foreign_paths(seal: Seal) -> list[str]:
    other = "build1" if seal.mode == "build2" else "build2"
    paths = [seal.work / other, seal.scratch / other]
    paths += [seal.target(d, other) for d in seal.cfg["dbs"]]
    return [str(p) for p in paths]


def check_isolation(env: dict, foreign: list[str]) -> None:
    """No env value may name -- as text or after resolving links -- the other build's paths."""
    real_foreign = [os.path.realpath(f) for f in foreign if f]
    for key, value in env.items():
        for f in foreign:
            if f and f in value:
                raise SealError("build_env_crosses_builds", key)
        for item in value.split(os.pathsep):
            if not item.startswith("/"):
                continue
            real = os.path.realpath(item)
            for rf in real_foreign:
                if real == rf or real.startswith(rf.rstrip("/") + "/"):
                    raise SealError("build_env_crosses_builds", key)


def confined_run(
    seal: Seal,
    db: str,
    unit: str,
    argv: list[str],
    env: dict,
    log: Path,
    cwd: Path,
    mem_gib: int,
    timeout_s: int,
) -> tuple[int, int | None]:
    """Run argv confined; returns (rc, peak_rss_kb). Timeout stops the whole scope."""
    cores = seal.cores[db]
    if seal.cfg["confine"] == "systemd":
        argv = [
            "systemd-run",
            "--user",
            "--scope",
            "-q",
            "--unit",
            unit,
            "-p",
            f"MemoryMax={mem_gib}G",
            "-p",
            "MemorySwapMax=0",
            "taskset",
            "-c",
            cores,
            "/usr/bin/time",
            "-v",
            *argv,
        ]
    with open(log, "w") as fh:
        proc = subprocess.Popen(
            argv,
            cwd=cwd,
            env=env,
            stdout=fh,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            rc = proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            if seal.cfg["confine"] == "systemd":
                subprocess.run(
                    [
                        "systemctl",
                        "--user",
                        "kill",
                        "--signal=SIGKILL",
                        f"{unit}.scope",
                    ],
                    capture_output=True,
                )
            os.killpg(proc.pid, 9)
            proc.wait()
            raise SealError("timeout", unit) from None
    peak = None
    m = re.search(
        r"Maximum resident set size \(kbytes\): (\d+)", log.read_text(errors="replace")
    )
    if m:
        peak = int(m.group(1))
    return rc, peak


def admit(seal: Seal, need_gib: int) -> None:
    gate = seal.cfg.get("memgate")
    if seal.cfg["confine"] != "systemd" or not gate:
        return
    sys.path.insert(0, gate["module_dir"])
    import memgate  # noqa: PLC0415 - server-side module, real runs only

    budget = int(gate["budget_gib"]) << 30
    uid = os.getuid()
    pattern = f"/sys/fs/cgroup/user.slice/user-{uid}.slice/user@{uid}.service/app.slice/seal-{seal.s9}-*.scope"
    while True:
        with open(gate["lock"], "a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            used = memgate.committed()
            for scope in glob.glob(pattern):
                try:
                    mx = Path(scope, "memory.max").read_text().strip()
                except OSError:
                    continue
                used += budget if mx == "max" else int(mx)
            ok = used + (need_gib << 30) <= budget
            fcntl.flock(lock, fcntl.LOCK_UN)
        seal.log(
            f"memgate seal: need={need_gib}G committed={used >> 30}G budget={budget >> 30}G ok={ok}"
        )
        if ok:
            return
        time.sleep(60)


# ----------------------------------------------------------------------------- per-DB plan and build
def farm_entries(src_root: Path, skip_names: set[str], allowed):
    """plan_entries for a link farm: a link leading outside the allowed roots is a farm refusal."""
    try:
        return plan_entries(src_root, skip_names, allowed)
    except SealError as exc:
        if exc.reason == "link_outside_allowed_roots":
            raise SealError("farm_link_outside_allowed_roots", exc.detail) from exc
        raise


def plan_db(seal: Seal, db: str) -> dict:
    spec = seal.cfg["dbs"][db]
    cur = Path(spec["current_root"])
    allowed = seal.cfg["allowed_roots"]
    if not cur.is_dir():
        raise SealError("current_root_missing", str(cur))
    tables = [t for t, _ in spec["tables"]]
    for t, args in spec["tables"]:
        if not (cur / f"{t}_bucket" / "_COMPLETE").is_file():
            raise SealError("current_bucket_missing", t)
        assert_params(cur / f"{t}_bucket", args)
    target = seal.target(db)
    for p in (target, seal.staging(target)):
        if p.exists() or p.is_symlink():
            raise SealError("target_exists", str(p))
    if not inside(target.parent, [seal.b6]) or target == cur:
        raise SealError("target_outside_b6_root", str(target))
    plan: dict = {
        "db": db,
        "current_root": str(cur),
        "target": str(target),
        "tables": tables,
    }
    if "farm" in spec:
        # the bucket root holds only the source entries of the rebuilt tables, same targets as the farm
        keep = {n for t in tables for n in (f"{t}.parquet", t)}
        entries = [e for e in farm_entries(cur, set(), allowed) if e[1] in keep]
        if not entries:
            raise SealError("farm_source_missing", ",".join(tables))
        plan["entries"] = entries
        if seal.mode == "build1":
            farm = seal.farm_target(db)
            for p in (farm, seal.staging(farm)):
                if p.exists() or p.is_symlink():
                    raise SealError("farm_target_exists", str(p))
            plan["farm_entries"] = farm_entries(
                cur, {f"{t}_bucket" for t in tables}, allowed
            )
            for kind, _, payload in plan["farm_entries"]:
                if kind == "manifest":
                    rewritten_conversion_manifest(
                        cur / CONVERSION_MANIFEST, cur, farm, payload
                    )
            reference = spec["farm"].get("reference")
            if reference:
                plan["farm_reference"] = check_farm_reference(
                    plan["farm_entries"], Path(reference), tables
                )
            plan["farm"] = str(farm)
    else:
        plan["entries"] = plan_entries(cur, {f"{t}_bucket" for t in tables}, allowed)
        for kind, _, payload in plan["entries"]:
            if kind == "manifest":
                rewritten_conversion_manifest(
                    cur / CONVERSION_MANIFEST, cur, target, payload
                )
    plan["links"] = sorted(n for _, n, _ in plan["entries"])
    return plan


def link_set(root: Path, skip: set[str]) -> dict[str, object]:
    """Entry names and link targets of a farm: {name: target | {file: target} | 'file'}."""
    out: dict[str, object] = {}
    for entry in sorted(os.scandir(root), key=lambda e: e.name):
        if entry.name in skip:
            continue
        if entry.is_symlink():
            out[entry.name] = os.readlink(entry.path)
        elif entry.is_dir():
            out[entry.name] = {
                f.name: os.readlink(f.path) if f.is_symlink() else "file"
                for f in os.scandir(entry.path)
            }
        else:
            out[entry.name] = "file"
    return out


def check_farm_reference(plan_entries, reference: Path, tables) -> dict:
    """The planned farm must have the reference farm's entry names and link targets (bucket links aside)."""
    if not reference.is_dir():
        raise SealError("farm_reference_missing", str(reference))
    planned: dict[str, object] = {}
    for kind, name, payload in plan_entries:
        planned[name] = (
            payload if kind == "link" else dict(payload) if kind == "dir" else "file"
        )
    skip = {f"{t}_bucket" for t in tables} | {
        n for n in os.listdir(reference) if SKIP_RE.match(n)
    }
    ref = link_set(reference, skip)
    differ = sorted(n for n in set(planned) | set(ref) if planned.get(n) != ref.get(n))
    if differ:
        raise SealError("farm_reference_mismatch", ",".join(differ[:10]))
    return {"reference": str(reference), "entries_compared": len(ref), "differences": 0}


def run_db(seal: Seal, db: str, plan: dict, code: Path) -> None:
    spec = seal.cfg["dbs"][db]
    cur, target = Path(spec["current_root"]), Path(plan["target"])
    staging = seal.staging(target)
    receipt = {
        "schema": "lead_r13.seal_build_receipt/2",
        "seal_sha": seal.full,
        "build": seal.mode,
        "db": db,
        "current_formal_root": str(cur),
        "staging": str(staging),
        "new_root": str(target),
        "cores": seal.cores[db],
        "verdict": "FAILED",
        "tables": {},
    }
    out = seal.work / seal.mode
    scr = seal.scratch_dir(seal.mode, db)
    try:
        claim_scratch(scr, seal.b6, seal.token)
        env = build_env(seal, db, code)
        receipt["build_env_keys"] = sorted(env)
        for key in (
            "HOME",
            "TMPDIR",
            "EASYICU_DUCKDB_TEMP_DIR",
            "XDG_CACHE_HOME",
            "PYTHONPYCACHEPREFIX",
        ):
            Path(env[key]).mkdir(parents=True, exist_ok=True)
        (scr / "cwd").mkdir()
        materialize(plan["entries"], staging, cur, target)
        check_links(staging, seal.cfg["allowed_roots"])
        for table, args in spec["tables"]:
            unit = f"seal-{seal.s9}-{seal.mode}-{db}-{table}".replace("_", "-")
            origin_out = out / f"origin_{db}_{table}.json"
            argv = [
                *builder_argv(seal.cfg["python"], code / BUILDER_REL),
                "--data-path",
                str(staging),
                "--database",
                spec["database"],
                "--table",
                table,
                *shlex.split(args),
            ]
            seal.log(
                f"{db} {table} build start cores={seal.cores[db]} mem={spec['mem_gib']}G"
            )
            t0 = time.time()
            rc, peak = confined_run(
                seal,
                db,
                unit,
                argv,
                {**env, "SEAL_ORIGIN_OUT": str(origin_out)},
                out / f"build_{db}_{table}.log",
                scr / "cwd",  # never the staging root (D34-BOOT-ISOLATION)
                spec["mem_gib"],
                spec["timeout_s"],
            )
            wall = round(time.time() - t0, 1)
            seal.log(f"{db} {table} build rc={rc} wall={wall}s peak_kb={peak}")
            origin = read_origin(origin_out)
            receipt.setdefault("import_origin", {})[table] = origin
            if rc == ORIGIN_EXIT or not origin_ok(origin, code, seal.full):
                raise SealError(
                    "builder_import_origin_mismatch", f"{table} {origin.get('path')}"
                )
            if rc != 0:
                text = (out / f"build_{db}_{table}.log").read_text(errors="replace")
                reason = BUILDER_MARK if BUILDER_MARK in text else "builder_error"
                raise SealError("build_failed", f"{table} rc={rc} {reason}")
            try:
                rec = qc_bucket(
                    staging / f"{table}_bucket", seal.full, cur / f"{table}_bucket"
                )
            except SealError as exc:
                receipt["tables"][table] = getattr(exc, "record", None)
                raise
            rec.update(wall_seconds=wall, peak_rss_kb=peak)
            receipt["tables"][table] = rec
        verify_checkout(code, seal.full)
        # read-only first, then every remaining check runs on the frozen tree (D29-PUBLISH)
        make_read_only(staging, include_root=False)
        for table, rec in receipt["tables"].items():
            manifest = staging / f"{table}_bucket" / "_BUCKET_MANIFEST.json"
            if sha256_file(manifest) != rec.get("manifest_sha256"):
                raise SealError("bucket_manifest_changed_after_qc", table)
        scan_json = out / f"{db}_footer_scan.json"
        rc, _ = confined_run(
            seal,
            db,
            f"seal-{seal.s9}-{seal.mode}-{db}-footer",
            [
                seal.cfg["python"],
                "-I",
                str(Path(__file__).with_name("footer_scan.py")),
                str(staging),
                "--out",
                str(scan_json),
                *[
                    arg
                    for root in seal.cfg["allowed_roots"]
                    for arg in ("--allowed-root", str(root))
                ],
            ],
            env,
            out / f"{db}_footer_scan.txt",
            out,
            4,
            3600,
        )
        if rc != 0:
            raise SealError("footer_scan_failed", db)
        scan = json.loads(scan_json.read_text())
        receipt["footer_scan"] = {
            k: scan[k]
            for k in (
                "ok",
                "parquet_files",
                "parquet_rows_total",
                "buckets",
                "tool_sha256",
            )
        }
        for table, rec in receipt["tables"].items():
            b = scan["buckets"].get(f"{table}_bucket") or {}
            if (
                b.get("rows_footer") != rec["rows"]
                or b.get("parts_on_disk_readable") != rec["parts"]
            ):
                raise SealError("footer_totals_vs_receipt", table)
        # the full bucket storage contract, as every reader and the comparator enforce it, on the
        # frozen tree before anything is published (delta29t (b))
        for table in receipt["tables"]:
            check_bucket_contract(staging / f"{table}_bucket", table)
        receipt["links"] = plan["links"]
        in_root = dict(receipt, verdict="PASS")
        installed = install(staging, target, in_root)
        receipt["install"] = installed
        receipt["verdict"] = "PASS"
        write_json(out / f"receipt_{db}.json", receipt)
        commit_install(target)
        seal.set_status(db, "PASS", str(target))
        seal.log(f"{db} PASS -> {target}")
    except SealError as exc:
        receipt["verdict"] = "FAILED"
        receipt["failure"] = {"reason": exc.reason, "detail": exc.detail}
        write_json(out / f"receipt_{db}.json", receipt)
        seal.set_status(db, "FAILED", f"{exc.reason} {exc.detail}")
        seal.log(f"{db} FAILED {exc}")
    except Exception as exc:  # noqa: BLE001 - any untyped error is a failure, recorded with its type
        receipt["verdict"] = "FAILED"
        receipt["failure"] = {
            "reason": "unexpected_error",
            "detail": type(exc).__name__,
        }
        write_json(out / f"receipt_{db}.json", receipt)
        seal.set_status(db, "FAILED", f"unexpected_error {type(exc).__name__}")
        seal.log(f"{db} FAILED unexpected_error {exc!r}")
    finally:
        release_scratch(scr, seal.scratch, seal.token)
    if seal.mode == "build1" and plan.get("farm") and seal.verdict(db) == "PASS":
        try:
            assemble_farm(seal, db, plan, target)
        except SealError as exc:
            write_json(
                out / f"receipt_{db}_farm.json",
                {
                    "farm": plan["farm"],
                    "verdict": "FAILED",
                    "failure": {"reason": exc.reason, "detail": exc.detail},
                },
            )
            seal.set_status(db, "FAILED", f"{exc.reason} {exc.detail}")
            seal.log(f"{db} farm FAILED {exc}")


def assemble_farm(seal: Seal, db: str, plan: dict, bucket_root: Path) -> None:
    spec = seal.cfg["dbs"][db]
    cur, farm = Path(spec["current_root"]), Path(plan["farm"])
    staging = seal.staging(farm)
    links = {f"{t}_bucket": str(bucket_root / f"{t}_bucket") for t in plan["tables"]}
    materialize(plan["farm_entries"], staging, cur, farm, links)
    check_links(
        staging, seal.cfg["allowed_roots"], reason="farm_link_outside_allowed_roots"
    )
    for name in links:
        if not inside(staging / name, [seal.b6]):
            raise SealError("farm_link_outside_allowed_roots", name)
    names = sorted(os.listdir(staging))
    manifest = staging / CONVERSION_MANIFEST
    record = {
        "schema": "lead_r13.seal_farm_receipt/2",
        "seal_sha": seal.full,
        "build": seal.mode,
        "db": db,
        "farm": str(farm),
        "replicates": str(cur),
        "entries": names,
        "bucket_links": links,
        "bucket_root": str(bucket_root),
        "reference_check": plan.get("farm_reference"),
        "conversion_manifest_sha256": sha256_file(manifest)
        if manifest.is_file()
        else None,
        "verdict": "PASS",
    }
    # the farm carries its own in-root receipt; a bucket root's _COMPLETE never stands for the farm
    installed = install(staging, farm, record)
    record["install"] = installed
    write_json(seal.work / seal.mode / f"receipt_{db}_farm.json", record)
    commit_install(farm)
    seal.set_status(db, "PASS", f"{bucket_root} farm={farm}")
    seal.log(f"{db} farm PASS -> {farm} ({len(names)} entries)")


# ----------------------------------------------------------------------------- compare
def compare_db(seal: Seal, db: str) -> None:
    spec = seal.cfg["dbs"][db]
    a, b = seal.target(db, "build1"), seal.target(db, "build2")
    out = seal.work / "compare"
    try:
        receipts = []
        work = Path(seal.cfg["work_root"]) / f"seal_{seal.s9}"
        for mode, root in (("build1", a), ("build2", b)):
            p = work / mode / f"receipt_{db}.json"
            if not p.is_file():
                raise SealError("compare_missing_build", f"{mode}:{db}")
            # the root is accepted only through its bound in-root receipt (D26-INSTALL, D26-COMPARE-SET)
            inner = accept_root(root, p, seal.full, mode, db)
            for t, _ in spec["tables"]:
                rec = (inner.get("tables") or {}).get(t) or {}
                manifest = root / f"{t}_bucket" / "_BUCKET_MANIFEST.json"
                if not (root / f"{t}_bucket" / "_COMPLETE").is_file():
                    raise SealError("compare_missing_build", f"{root.name}/{t}_bucket")
                if (
                    not rec.get("manifest_sha256")
                    or sha256_file(manifest) != rec["manifest_sha256"]
                ):
                    raise SealError(
                        "compare_manifest_unbound", f"{root.name}/{t}_bucket"
                    )
            receipts.append(inner)
        if "farm" in spec:
            # formal acceptance of CareVue needs the farm receipt as well as the bucket receipts
            farm = seal.farm_target(db)
            accept_root(
                farm,
                work / "build1" / f"receipt_{db}_farm.json",
                seal.full,
                "build1",
                db,
            )
        if seal.dry_run:
            seal.set_status(db, "DRY_RUN_OK")
            return
        verdicts = {}
        for t, _ in spec["tables"]:
            fa = (receipts[0]["tables"].get(t) or {}).get("float_fidelity")
            fb = (receipts[1]["tables"].get(t) or {}).get("float_fidelity")
            if fa != fb:
                raise SealError("compare_float_profile_differs", t)
            env = {
                "PATH": "/usr/local/bin:/usr/bin:/bin",
                "LANG": "C.UTF-8",
                "PYTHONDONTWRITEBYTECODE": "1",
            }
            rc, _ = confined_run(
                seal,
                db,
                f"seal-{seal.s9}-compare-{db}-{t}".replace("_", "-"),
                [
                    seal.cfg["python"],
                    "-I",
                    str(Path(__file__).with_name("bucket_compare.py")),
                    str(a / f"{t}_bucket"),
                    str(b / f"{t}_bucket"),
                    "--out",
                    str(out / f"compare_{db}_{t}.json"),
                    "--seal-sha",
                    seal.full,
                ],
                env,
                out / f"compare_{db}_{t}.txt",
                out,
                16,
                7200,
            )
            verdicts[t] = (
                (out / f"compare_{db}_{t}.txt").read_text().strip().splitlines()[-1:]
            )
            seal.log(f"{db} {t} build1-vs-build2 rc={rc} {verdicts[t]}")
            if rc != 0:
                raise SealError("build1_vs_build2_differs", t)
        seal.set_status(db, "IDENTICAL")
    except SealError as exc:
        seal.set_status(db, "FAILED", f"{exc.reason} {exc.detail}")
        seal.log(f"{db} FAILED {exc}")


# ----------------------------------------------------------------------------- probe (dry run)
PROBE_DATA = r"""
import sys, numpy as np, pyarrow as pa, pyarrow.parquet as pq
from pathlib import Path
root = Path(sys.argv[1]); n = 4000; rng = np.random.default_rng(3)
v = rng.choice([0.0, 1.5, 2.5], n); v[rng.random(n) < 0.1] = -0.0; v[0] = -0.0; v[rng.random(n) < 0.02] = np.nan
t = pa.table({"stay_id": pa.array(rng.integers(1, 30, n), pa.int64()), "itemid": pa.array(rng.integers(1, 5, n), pa.int64()),
              "charttime": pa.array(np.arange(n), pa.int64()), "valuenum": pa.array(v, pa.float64())})
for name in ("probe_hash", "probe_var"):
    (root / name).mkdir(parents=True); pq.write_table(t, root / name / "1.parquet", use_dictionary=False)
"""


def probe(seal: Seal) -> list[tuple[str, str]]:
    base = seal.work / "probe"
    if base.exists():
        raise SealError("probe_dir_exists", str(base))
    code = seal.prepare_checkout(base / "code")
    data = base / "data"
    data.mkdir(parents=True)
    subprocess.run([seal.cfg["python"], "-c", PROBE_DATA, str(data)], check=True)
    schemes = {
        _opts(a).get("--scheme", "hash")
        for d in seal.dbs
        for _, a in seal.cfg["dbs"][d]["tables"]
    }
    results = []
    env = {
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "LANG": "C.UTF-8",
        "HOME": str(base / "home"),
        "EASYICU_HOME": str(base / "home/.easyicu"),
        "TMPDIR": str(base),
        "PYTHONPATH": str(code / "src"),
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "EASYICU_DUCKDB_TEMP_DIR": str(base / "spill"),
        "SEAL_CLONE": str(code),
        "SEAL_SHA": seal.full,
        "SEAL_SITE_PACKAGES": site_packages(seal.cfg["python"]),
    }
    for scheme, table, extra in (
        ("hash", "probe_hash", ["--key", "itemid", "--modulus", "4"]),
        (
            "variable_stay_range",
            "probe_var",
            ["--key", "itemid", "--stay-key", "stay_id"],
        ),
    ):
        if scheme not in schemes:
            continue
        origin_out = base / f"origin_{table}.json"
        cmd = [
            *builder_argv(seal.cfg["python"], code / BUILDER_REL),
            "--data-path",
            str(data),
            "--table",
            table,
            "--scheme",
            scheme,
            *extra,
            "--threads",
            "1",
            "--memory-limit",
            "1GB",
        ]
        (base / "cwd").mkdir(exist_ok=True)
        done = subprocess.run(
            cmd,
            cwd=base / "cwd",
            env={**env, "SEAL_ORIGIN_OUT": str(origin_out)},
            capture_output=True,
            text=True,
            timeout=900,
        )
        if done.returncode == ORIGIN_EXIT or not origin_ok(
            read_origin(origin_out), code, seal.full
        ):
            results.append(
                ("FAILED", f"probe {scheme}: FAILED builder_import_origin_mismatch")
            )
            continue
        if done.returncode != 0:
            results.append(
                ("FAILED", f"probe {scheme}: FAILED build_failed rc={done.returncode}")
            )
            continue
        try:
            qc_bucket(data / f"{table}_bucket", seal.full)
            results.append(("PASS", f"probe {scheme}: PASS"))
        except SealError as exc:
            results.append(
                ("FAILED", f"probe {scheme}: FAILED {exc.reason} {exc.detail}")
            )
    shutil.rmtree(base)
    return results


# ----------------------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        epilog=__doc__[__doc__.index("Operator runbook") :],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("sha")
    ap.add_argument("mode", choices=MODES)
    ap.add_argument("--config", type=Path)
    ap.add_argument("--only", default="")
    ap.add_argument(
        "--cores",
        action="append",
        default=[],
        help="db=lo-hi; required for every selected db",
    )
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--probe", action="store_true")
    args = ap.parse_args(argv)
    if args.probe and not args.dry_run:
        ap.error("--probe needs --dry-run")
    # (verdict, line) pairs; success is decided from verdicts only (D26-EXIT)
    status_lines: list[tuple[str, str]] = []
    seal: Seal | None = None
    run_lock = None
    try:
        cfg = load_config(args.config)
        cores = dict(c.split("=", 1) for c in args.cores if "=" in c)
        seal = Seal(
            cfg,
            args.sha,
            args.mode,
            args.dry_run,
            [d for d in args.only.split(",") if d],
            cores,
        )
        run_lock = acquire_run_lock(seal)
        (seal.work / seal.mode).mkdir(parents=True, exist_ok=True)
        seal.log(
            f"seal {seal.full} mode={seal.mode}{' dry-run' if seal.dry_run else ''} "
            f"dbs={','.join(seal.dbs)} confine={cfg['confine']}"
        )
        for db in seal.dbs:
            if db not in cores:
                raise SealError("no_cores", db)
            parse_cores(cores[db])
        if args.mode == "compare":
            for db in seal.dbs:
                compare_db(seal, db)
        else:
            seal.check_builder_mark()
            run_builds(seal, cfg, cores, args.probe, status_lines)
    except SealError as exc:
        status_lines.append(
            ("FAILED", f"all FAILED {exc.reason} {exc.detail}".rstrip())
        )
    except Exception as exc:  # noqa: BLE001 - any untyped error fails the run, with its type
        status_lines.append(
            ("FAILED", f"all FAILED unexpected_error {type(exc).__name__}")
        )
    finally:
        if run_lock is not None:
            run_lock.close()
    entries: list[tuple[str, str]] = []
    if seal:
        entries = [
            seal.status.get(d, ("FAILED", f"{d} FAILED not_run")) for d in seal.dbs
        ]
    entries += status_lines
    lines = [line for _, line in entries]
    if seal and (seal.work / seal.mode).is_dir():
        (seal.work / seal.mode / "status").write_text("\n".join(lines) + "\n")
    for line in lines:
        print(line, flush=True)
    ok = (
        seal is not None
        and bool(seal.dbs)
        and bool(entries)
        and all(verdict in OK_VERDICTS for verdict, _ in entries)
    )
    return EXIT_OK if ok else EXIT_FAIL


def run_builds(
    seal: Seal,
    cfg: dict,
    cores: dict,
    with_probe: bool,
    status_lines: list[tuple[str, str]],
) -> None:
    plans = {}
    for db in seal.dbs:
        try:
            plans[db] = plan_db(seal, db)
        except SealError as exc:
            seal.set_status(db, "FAILED", f"{exc.reason} {exc.detail}")
            continue
        if seal.dry_run:
            p = plans[db]
            seal.log(
                f"{db} plan ok: {p['current_root']} -> {p['target']}"
                f"{' farm=' + p['farm'] if p.get('farm') else ''}; tables={','.join(p['tables'])}; "
                f"mirrored entries={len(p['entries'])}"
                + (
                    f"; farm entries={len(p['farm_entries'])}"
                    if p.get("farm_entries")
                    else ""
                )
                + (
                    f"; farm reference {p['farm_reference']}"
                    if p.get("farm_reference")
                    else ""
                )
            )
            seal.set_status(db, "DRY_RUN_OK", p["target"])
    if seal.dry_run:
        if with_probe:
            for verdict, line in probe(seal):
                seal.log(line)
                status_lines.append((verdict, line))
        return
    if len(plans) != len(seal.dbs):
        return  # a planning failure stops the seal before anything is built
    code = seal.prepare_checkout(seal.code_dir(seal.mode))
    with ThreadPoolExecutor(max_workers=len(seal.dbs)) as pool:
        futures = []
        for db in seal.dbs:
            lo, hi = parse_cores(cores[db])
            while cfg["confine"] == "systemd" and cores_busy(lo, hi):
                seal.log(f"{db} waiting: cores {cores[db]} busy")
                time.sleep(60)
            admit(seal, cfg["dbs"][db]["mem_gib"])
            futures.append(pool.submit(run_db, seal, db, plans[db], code))
            if cfg["confine"] == "systemd":
                time.sleep(30)
        for future in futures:
            future.result()


if __name__ == "__main__":
    sys.exit(main())
