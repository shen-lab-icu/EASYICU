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
Every failure is a typed reason in the status file and on stdout; nothing is swallowed.

Per DB and build (order SIC -> eICU -> CareVue -> MIIV -> AUMC; each starts once its cores are free and
the memgate admits it, then DBs run in parallel):
  1. one clean clone of SEAL_SHA per build (build1 and build2 never share a checkout); HEAD must equal the
     SHA and `git status --porcelain --untracked-files=all` must be empty, before AND after the builds;
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
       a. the in-root receipt <root>/_SEAL_BUILD_RECEIPT.json is written as the LAST content step;
       b. read-only is applied to every directory and regular file under the staging root (symlinks and
          their targets untouched), except the staging directory itself;
       c. atomic rename staging -> target (same parent directory);
       d. the target directory itself is made read-only; the whole tree is verified read-only;
       e. the external receipt $WORK/<build>/receipt_<db>.json is written last; it records the verified
          read-only state and the sha256 of the in-root receipt.
     A failure before (c) leaves only the hidden staging root (never renamed); a rerun refuses until an
     operator removes it.
  CareVue (build1 only, after its bucket root PASSES): the farm b6_root/mimic_cv_dir_r13 is assembled the
  same way (staging, ro, rename): the current farm's flat links and per-file dirs replicated exactly,
  chartevents_bucket -> the new build1 bucket, conversion_manifest.json with data_path re-pointed; refused
  if mimic_cv_dir_r13 already exists; every link target must resolve inside the allowed roots.
Writes: only new roots/staging under b6_root, b6_root/_scratch/seal_<sha9>/, and the work directory.
Never databases/, the CareVue view, or any current root. Receipts hold aggregates and paths only.
"""

from __future__ import annotations

import argparse
import fcntl
import glob
import hashlib
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
ADOPTION_RECEIPT = "_BUCKET_ADOPTION_RECEIPT.json"
SOURCE_ORDINAL = "_easyicu_source_ordinal"
FORBIDDEN_CORES = ((5, 105), (146, 179), (224, 1 << 20))

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
        self.status: dict[str, str] = {}

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

    def set_status(self, db: str, line: str) -> None:
        with self.lock:
            self.status[db] = line

    # ---- builder checks
    def check_builder_mark(self) -> None:
        source = git(["show", f"{self.full}:{BUILDER_REL}"], git_dir=self.gitd)
        if BUILDER_MARK not in source:
            raise SealError("builder_lacks_float_fidelity", self.s9)

    def prepare_checkout(self, code: Path) -> Path:
        if not code.exists():
            code.parent.mkdir(parents=True, exist_ok=True)
            git(["clone", "-q", "--no-checkout", self.cfg["mirror"], str(code)])
            git(
                ["-c", "advice.detachedHead=false", "checkout", "-q", self.full],
                cwd=code,
            )
        verify_checkout(code, self.full)
        return code


def verify_checkout(code: Path, full: str) -> None:
    if not (code / ".git").exists():
        raise SealError("checkout_missing", str(code))
    head = git(["rev-parse", "HEAD"], cwd=code).strip()
    if head != full:
        raise SealError("checkout_not_at_seal", head[:9])
    if git(["status", "--porcelain", "--untracked-files=all"], cwd=code).strip():
        raise SealError("checkout_not_clean", str(code))


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
def rewritten_conversion_manifest(src: Path, old_root: Path, new_root: Path) -> dict:
    c = json.loads(src.read_text())
    d = c.get("derived_from") or {}
    if d.get("path"):
        if not Path(d["path"]).is_file() or sha256_file(Path(d["path"])) != d.get(
            "sha256"
        ):
            raise SealError("conversion_manifest_derived_from_changed", src.parent.name)
    if c.get("data_path") not in (None, str(old_root)):
        raise SealError("conversion_manifest_data_path_unexpected", src.parent.name)
    if c.get("data_path") is not None:
        c["data_path"] = str(new_root)
    return c


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
        if entry.is_symlink():
            if not path.exists():
                raise SealError("mirror_link_dangling", name)
            if not inside(path, allowed):
                raise SealError("link_outside_allowed_roots", name)
            plan.append(("link", name, os.readlink(path)))
        elif entry.is_dir():
            files = []
            for f in sorted(os.scandir(path), key=lambda e: e.name):
                fp = Path(f.path)
                if os.path.isdir(fp):
                    raise SealError("nested_dir_unsupported", f"{name}/{f.name}")
                if not inside(fp, allowed):
                    raise SealError("link_outside_allowed_roots", f"{name}/{f.name}")
                files.append(
                    (
                        f.name,
                        os.readlink(fp) if os.path.islink(fp) else os.path.realpath(fp),
                    )
                )
            plan.append(("dir", name, files))
        elif name == CONVERSION_MANIFEST:
            plan.append(("manifest", name, None))
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
            write_json(
                staging / name,
                rewritten_conversion_manifest(src_root / name, src_root, final),
            )
    for name, target in (extra_links or {}).items():
        os.symlink(target, staging / name)


def check_links(
    root: Path, allowed, reason: str = "link_outside_allowed_roots"
) -> None:
    """Every symlink under ``root`` (files and directories) must resolve, inside the allowed roots."""
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        for name in [*dirnames, *filenames]:
            p = os.path.join(dirpath, name)
            if os.path.islink(p) and (not os.path.exists(p) or not inside(p, allowed)):
                raise SealError(reason, os.path.relpath(p, root))


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


def install(staging: Path, target: Path, in_root_receipt: dict | None) -> dict:
    """Steps 6a-6d: in-root receipt last, ro, atomic rename, ro root, verify. Returns the install record."""
    if target.exists() or target.is_symlink():
        raise SealError("target_exists", str(target))
    in_root_sha = None
    if in_root_receipt is not None:
        in_root_sha = write_json(staging / IN_ROOT_RECEIPT, in_root_receipt)
    make_read_only(staging, include_root=False)
    os.rename(staging, target)
    make_read_only(target, include_root=True)
    left = writable_entries(target)
    if left:
        raise SealError("root_not_read_only", f"{target.name}: {len(left)} writable")
    if in_root_sha is not None and sha256_file(target / IN_ROOT_RECEIPT) != in_root_sha:
        raise SealError("in_root_receipt_changed", target.name)
    return {
        "root": str(target),
        "read_only_verified": True,
        "in_root_receipt_sha256": in_root_sha,
    }


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
    for key, value in env.items():
        for f in foreign:
            if f and f in value:
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
            if any(kind == "manifest" for kind, _, _ in plan["farm_entries"]):
                rewritten_conversion_manifest(cur / CONVERSION_MANIFEST, cur, farm)
            reference = spec["farm"].get("reference")
            if reference:
                plan["farm_reference"] = check_farm_reference(
                    plan["farm_entries"], Path(reference), tables
                )
            plan["farm"] = str(farm)
    else:
        plan["entries"] = plan_entries(cur, {f"{t}_bucket" for t in tables}, allowed)
        if any(kind == "manifest" for kind, _, _ in plan["entries"]):
            rewritten_conversion_manifest(cur / CONVERSION_MANIFEST, cur, target)
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
    try:
        env = build_env(seal, db, code)
        receipt["build_env_keys"] = sorted(env)
        for key in ("HOME", "TMPDIR", "EASYICU_DUCKDB_TEMP_DIR", "XDG_CACHE_HOME"):
            Path(env[key]).mkdir(parents=True, exist_ok=True)
        materialize(plan["entries"], staging, cur, target)
        check_links(staging, seal.cfg["allowed_roots"])
        for table, args in spec["tables"]:
            unit = f"seal-{seal.s9}-{seal.mode}-{db}-{table}".replace("_", "-")
            argv = [
                seal.cfg["python"],
                str(code / BUILDER_REL),
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
                env,
                out / f"build_{db}_{table}.log",
                staging,
                spec["mem_gib"],
                spec["timeout_s"],
            )
            wall = round(time.time() - t0, 1)
            seal.log(f"{db} {table} build rc={rc} wall={wall}s peak_kb={peak}")
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
        receipt["links"] = plan["links"]
        receipt["verdict"] = "PASS"
        in_root = dict(receipt)
        installed = install(staging, target, in_root)
        receipt["install"] = installed
        write_json(out / f"receipt_{db}.json", receipt)
        seal.set_status(db, f"{db} PASS {target}")
        seal.log(f"{db} PASS -> {target}")
    except SealError as exc:
        receipt["failure"] = {"reason": exc.reason, "detail": exc.detail}
        write_json(out / f"receipt_{db}.json", receipt)
        seal.set_status(db, f"{db} FAILED {exc.reason} {exc.detail}".rstrip())
        seal.log(f"{db} FAILED {exc}")
    finally:
        spill = seal.scratch_dir(seal.mode, db)
        if spill.is_dir() and inside(spill, [seal.scratch]):
            shutil.rmtree(spill)
    if (
        seal.mode == "build1"
        and plan.get("farm")
        and seal.status.get(db, "").startswith(f"{db} PASS")
    ):
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
            seal.set_status(db, f"{db} FAILED {exc.reason} {exc.detail}".rstrip())
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
    installed = install(staging, farm, None)
    manifest = farm / CONVERSION_MANIFEST
    record = {
        "schema": "lead_r13.seal_farm_receipt/1",
        "seal_sha": seal.full,
        "farm": str(farm),
        "replicates": str(cur),
        "entries": names,
        "bucket_links": links,
        "reference_check": plan.get("farm_reference"),
        "conversion_manifest_sha256": sha256_file(manifest)
        if manifest.is_file()
        else None,
        "install": installed,
        "verdict": "PASS",
    }
    write_json(seal.work / seal.mode / f"receipt_{db}_farm.json", record)
    seal.set_status(db, f"{db} PASS {bucket_root} farm={farm}")
    seal.log(f"{db} farm PASS -> {farm} ({len(names)} entries)")


# ----------------------------------------------------------------------------- compare
def compare_db(seal: Seal, db: str) -> None:
    spec = seal.cfg["dbs"][db]
    a, b = seal.target(db, "build1"), seal.target(db, "build2")
    out = seal.work / "compare"
    try:
        receipts = []
        for mode in ("build1", "build2"):
            p = (
                Path(seal.cfg["work_root"])
                / f"seal_{seal.s9}"
                / mode
                / f"receipt_{db}.json"
            )
            r = json.loads(p.read_text()) if p.is_file() else {}
            if r.get("verdict") != "PASS" or r.get("seal_sha") != seal.full:
                raise SealError("compare_missing_build", f"{mode}:{db}")
            receipts.append(r)
        for t, _ in spec["tables"]:
            for root in (a, b):
                if not (root / f"{t}_bucket" / "_COMPLETE").is_file():
                    raise SealError("compare_missing_build", f"{root.name}/{t}_bucket")
        if seal.dry_run:
            seal.set_status(db, f"{db} DRY_RUN_OK")
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
        seal.set_status(db, f"{db} IDENTICAL")
    except SealError as exc:
        seal.set_status(db, f"{db} FAILED {exc.reason} {exc.detail}".rstrip())
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


def probe(seal: Seal) -> list[str]:
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
        cmd = [
            seal.cfg["python"],
            str(code / BUILDER_REL),
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
        done = subprocess.run(
            cmd, cwd=data, env=env, capture_output=True, text=True, timeout=900
        )
        if done.returncode != 0:
            results.append(f"probe {scheme}: FAILED build_failed rc={done.returncode}")
            continue
        try:
            qc_bucket(data / f"{table}_bucket", seal.full)
            results.append(f"probe {scheme}: PASS")
        except SealError as exc:
            results.append(f"probe {scheme}: FAILED {exc.reason} {exc.detail}")
    shutil.rmtree(base)
    return results


# ----------------------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
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
    status_lines: list[str] = []
    seal: Seal | None = None
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
        status_lines.append(f"all FAILED {exc.reason} {exc.detail}".rstrip())
    lines = [seal.status[d] for d in seal.dbs if d in seal.status] if seal else []
    if seal and not status_lines:
        lines += [f"{d} FAILED not_run" for d in seal.dbs if d not in seal.status]
    lines += status_lines
    if seal:
        (seal.work / seal.mode / "status").write_text("\n".join(lines) + "\n")
    for line in lines:
        print(line, flush=True)
    ok = bool(lines) and all(
        re.search(r" (PASS|IDENTICAL|DRY_RUN_OK)( |$)|: PASS$", ln) for ln in lines
    )
    return EXIT_OK if ok else EXIT_FAIL


def run_builds(
    seal: Seal, cfg: dict, cores: dict, with_probe: bool, status_lines: list[str]
) -> None:
    plans = {}
    for db in seal.dbs:
        try:
            plans[db] = plan_db(seal, db)
        except SealError as exc:
            seal.set_status(db, f"{db} FAILED {exc.reason} {exc.detail}".rstrip())
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
            seal.set_status(db, f"{db} DRY_RUN_OK {p['target']}")
    if seal.dry_run:
        if with_probe:
            for line in probe(seal):
                seal.log(line)
                status_lines.append(line)
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
