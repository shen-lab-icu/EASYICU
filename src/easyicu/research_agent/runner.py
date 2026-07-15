"""Run agent-generated Python code in a constrained subprocess.

The runner is the line between LLM output and real numbers. It must
be deterministic, isolated, and fully captured.

What it gives you:

* a fresh working directory per step (``STEP_OUT_DIR``);
* the cohort path injected as ``COHORT_PARQUET``;
* stdout, stderr and exit-code captured into a run log;
* a hard wall-clock timeout (default 5 min) so a runaway script
  never blocks the whole pipeline;
* a curated PYTHONPATH so the script picks up the EasyICU
  installation but no surprise plugins.

Two runner backends ship in this module:

* :class:`CodeRunner` — macOS ``sandbox-exec`` confinement when available;
  otherwise fail-closed unless development explicitly opts into unsafe host
  execution.
* :class:`DockerRunner` — host-side ``docker run`` with
  ``--network none`` and read-only cohort mount, so an LLM that emits
  ``rm -rf /`` or attempts to call ``urllib.request`` cannot escape
  or exfiltrate anything. Same ``run(step_id=, code=)`` contract
  as :class:`CodeRunner`, so the pipeline can swap them in via
  ``runner_kind="docker"``.

Hashes live in :mod:`evidence`, not here — these classes only produce
the artefacts.
"""

from __future__ import annotations

import ast
import glob
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import textwrap
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .code_hygiene import reorder_forward_references
from .method_capabilities import (
    BASELINE_PACKAGES,
    CURATED_METHOD_PACKAGES,
    OPTIONAL_BASELINE_PACKAGES,
    set_runtime_capability_snapshot_provider,
)

_SAFE_INHERITED_ENV_KEYS = (
    "PATH",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "TZ",
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
)

_RUN_ARTIFACT_AUTHORITY_SNAPSHOT_SCHEMA = "easyicu.run_artifact_authority_snapshot/1"
_RUN_ARTIFACT_AUTHORITY_SNAPSHOT_ENV = "EASYICU_RUN_ARTIFACT_AUTHORITY_SNAPSHOT"
_RUN_ARTIFACT_AUTHORITY_SNAPSHOT_SHA_ENV = (
    "EASYICU_RUN_ARTIFACT_AUTHORITY_SNAPSHOT_SHA256"
)
_RUN_ARTIFACT_AUTHORITY_ERROR_ENV = "EASYICU_RUN_ARTIFACT_AUTHORITY_ERROR"
_ROBUSTNESS_AUTHORITY_ENTRYPOINT = "_run_robustness_preflight_from_env"


def _canonical_json_bytes(payload: object) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        + "\n"
    ).encode("utf-8")


def _replace_regular_file_atomically(destination: Path, payload: bytes) -> None:
    """Replace one host-owned control file without following an old link."""

    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _remove_authority_snapshot(path: Path) -> None:
    """Remove a stale runner-control snapshot without traversing a symlink."""

    try:
        metadata = os.lstat(path)
    except FileNotFoundError:
        return
    if stat.S_ISDIR(metadata.st_mode) and not stat.S_ISLNK(metadata.st_mode):
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)


def _capture_run_artifact_authority_snapshot(
    *, workdir: Path, step_dir: Path
) -> tuple[Optional[Path], Optional[str], Optional[str]]:
    """Freeze current host-selected artifact authority for one subprocess.

    Generated and deterministic code must not choose between live/final
    manifests itself.  The trusted host resolves the newest checkpoint once,
    serializes that exact authority into a receipt, and binds the receipt bytes
    to an environment-supplied SHA-256 digest.  Missing or corrupt current
    authority deliberately yields no snapshot, so consumers can fail closed
    without replaying an older checkpoint.
    """

    from .runtime_artifacts import (
        RunArtifactAuthorityError,
        current_evidence_records,
        current_step_records,
        load_run_artifact_authority,
    )

    destination = step_dir / ".run_artifact_authority_snapshot.json"
    try:
        authority = load_run_artifact_authority(workdir)
    except RunArtifactAuthorityError as exc:
        _remove_authority_snapshot(destination)
        return None, None, str(exc)
    if authority is None:
        _remove_authority_snapshot(destination)
        return None, None, "No current per-step checkpoint authority is available."

    raw_step_records = authority.get("per_step_records")
    if not isinstance(raw_step_records, list):
        _remove_authority_snapshot(destination)
        return None, None, "Current checkpoint has no valid per-step ledger."
    active_step_records = [
        dict(record) for record in current_step_records(raw_step_records)
    ]
    raw_evidence = authority.get("evidence")
    evidence_records = raw_evidence if isinstance(raw_evidence, list) else []
    active_evidence = current_evidence_records(
        evidence_records,
        active_step_records,
    )
    # The subprocess receives only the current scientific authority closure,
    # never the complete manifest (findings, prompts, repairs, notes, etc.).
    authority = {
        "run_id": authority.get("run_id"),
        "checkpoint_sequence": authority.get("checkpoint_sequence"),
        "per_step_records": active_step_records,
        "evidence": [
            dict(record) if isinstance(record, dict) else record
            for record in active_evidence
        ],
    }
    authority_bytes = _canonical_json_bytes(authority)
    authority_sha256 = hashlib.sha256(authority_bytes).hexdigest()
    snapshot = {
        "schema_version": _RUN_ARTIFACT_AUTHORITY_SNAPSHOT_SCHEMA,
        "checkpoint_sequence": authority.get("checkpoint_sequence"),
        "authority_sha256": authority_sha256,
        "authority": authority,
    }
    snapshot_bytes = _canonical_json_bytes(snapshot)
    snapshot_sha256 = hashlib.sha256(snapshot_bytes).hexdigest()
    _replace_regular_file_atomically(destination, snapshot_bytes)
    return destination.resolve(), snapshot_sha256, None


def _code_requests_robustness_authority_snapshot(code: str) -> bool:
    """Recognise the exact host-owned robustness entrypoint import."""

    try:
        tree = ast.parse(code)
    except (SyntaxError, TypeError, ValueError):
        return False
    return any(
        isinstance(node, ast.ImportFrom)
        and node.module == "easyicu.research_agent.deterministic_robustness"
        and any(
            alias.name == _ROBUSTNESS_AUTHORITY_ENTRYPOINT for alias in node.names
        )
        for node in ast.walk(tree)
    )


def _authority_snapshot_for_code(
    *, code: str, workdir: Path, step_dir: Path
) -> tuple[Optional[Path], Optional[str], Optional[str]]:
    if _code_requests_robustness_authority_snapshot(code):
        return _capture_run_artifact_authority_snapshot(
            workdir=workdir,
            step_dir=step_dir,
        )
    _remove_authority_snapshot(
        step_dir / ".run_artifact_authority_snapshot.json"
    )
    return None, None, None


def _safe_path_component(value: str, *, label: str) -> str:
    text = str(value or "")
    if (
        not text
        or text in {".", ".."}
        or "\x00" in text
        or "/" in text
        or "\\" in text
        or Path(text).is_absolute()
        or Path(text).name != text
    ):
        raise ValueError(f"{label} must be a single safe path component")
    return text


def _validated_resolved_inputs_path(
    value: Optional[Path],
    *,
    workdir: Path,
) -> Optional[Path]:
    """Accept only a regular, non-symlink manifest inside the run root."""

    if value is None:
        return None
    candidate = Path(value).expanduser()
    if candidate.is_symlink():
        raise ValueError("resolved_inputs_path must not be a symlink")
    try:
        resolved = candidate.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ValueError("resolved_inputs_path must exist") from exc
    if not resolved.is_file():
        raise ValueError("resolved_inputs_path must be a regular file")
    try:
        resolved.relative_to(Path(workdir).resolve())
    except ValueError as exc:
        raise ValueError("resolved_inputs_path must be inside the run workdir") from exc
    return resolved


def _sandbox_quote(path: Path | str) -> str:
    return str(path).replace("\\", "\\\\").replace('"', '\\"')


def _env_flag(name: str, *, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _normalise_distribution_name(value: str) -> str:
    return value.strip().lower().replace("_", "-").replace(".", "-")


def _distributions_from_freeze(requirements: str) -> frozenset[str]:
    names: set[str] = set()
    for raw_line in requirements.splitlines():
        line = raw_line.strip()
        if not line or line.startswith(("#", "-")):
            continue
        if " @ " in line:
            name = line.split(" @ ", 1)[0]
        elif "==" in line:
            name = line.split("==", 1)[0]
        else:
            continue
        if name:
            names.add(_normalise_distribution_name(name))
    return frozenset(names)


def _as_text(stream: object) -> str:
    """Normalise a subprocess stdout/stderr capture to ``str``.

    On ``subprocess.TimeoutExpired`` the partial output can come back as
    ``bytes`` even when the process was launched in text mode (a CPython
    wrinkle), so concatenating it with the ``str`` timeout message raises
    ``TypeError: can't concat str to bytes``. Decode defensively; ``None`` -> "".
    """
    if stream is None:
        return ""
    if isinstance(stream, bytes):
        return stream.decode("utf-8", errors="replace")
    return str(stream)


@dataclass
class RunResult:
    """Everything captured from one code execution."""

    step_id: str
    script_path: Path
    cwd: Path
    out_dir: Path
    stdout: str
    stderr: str
    returncode: int
    duration_seconds: float
    artefacts: List[Path] = field(default_factory=list)
    timed_out: bool = False
    requested_network_policy: str = "none"
    effective_isolation: str = "unknown"
    isolation_degraded: bool = False
    isolation_degradation_reason: Optional[str] = None
    runtime_provenance: Dict[str, object] = field(default_factory=dict)
    # False means callers must not scan or hash anything under ``out_dir``.
    outputs_safe_to_collect: bool = True
    runner_log_path: Optional[Path] = None

    @property
    def succeeded(self) -> bool:
        return (
            self.returncode == 0
            and not self.timed_out
            and self.outputs_safe_to_collect
        )


class CodeRunner:
    """Run agent-generated Python in a fresh per-step directory."""

    def __init__(
        self,
        *,
        workdir: Path,
        cohort_parquet: Path,
        timeout_seconds: float = 300.0,
        python_executable: Optional[str] = None,
        extra_env: Optional[Dict[str, str]] = None,
        network_policy: str = "none",
        allow_unsafe_host_fallback: Optional[bool] = None,
    ) -> None:
        self.workdir = Path(workdir).expanduser().resolve()
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.cohort_parquet = Path(cohort_parquet).resolve()
        if not self.cohort_parquet.exists():
            raise FileNotFoundError(
                f"Cohort parquet does not exist: {self.cohort_parquet}"
            )
        self.timeout_seconds = timeout_seconds
        selected_python = str(python_executable or sys.executable)
        python_path = Path(selected_python).expanduser()
        if python_path.is_absolute():
            try:
                # Resolve symlinked *directory* ancestors (for example a
                # worktree-local ``.venv`` link) so sandbox-exec sees the same
                # runtime path that its profile allows.  Preserve the final
                # ``bin/python`` entry point: resolving that symlink all the
                # way to a base interpreter would discard virtualenv prefix
                # discovery and could silently run with the wrong packages.
                resolved_parent = python_path.parent.resolve(strict=True)
                normalized_python = resolved_parent / python_path.name
                if normalized_python.exists():
                    selected_python = str(normalized_python)
            except (OSError, RuntimeError):
                pass
        self.python_executable = selected_python
        self.extra_env = dict(extra_env or {})
        self.network_policy = (network_policy or "none").lower()
        self.allow_unsafe_host_fallback = (
            _env_flag("EASYICU_ALLOW_UNSAFE_HOST_FALLBACK")
            if allow_unsafe_host_fallback is None
            else bool(allow_unsafe_host_fallback)
        )
        # A host runner must never inherit a Docker capability snapshot left in
        # the same context by an earlier run.
        set_runtime_capability_snapshot_provider(None)

    def _isolation_backend_for_cmd(self, cmd: Sequence[str]) -> str:
        if self.network_policy not in {"none", "disabled"}:
            return "network_allowed"
        if cmd and Path(cmd[0]).name == "sandbox-exec":
            return "macos_sandbox_exec"
        if cmd and Path(cmd[0]).name == "unshare":
            return "linux_unshare_network_namespace"
        return "host_subprocess"

    def _macos_sandbox_profile(self, *, script_path: Path) -> str:
        """Return a network-denied, workdir-confined macOS profile.

        Python and its native wheels need read access to their installation
        prefixes and system libraries.  Generated code receives read access to
        those runtime locations, the cohort, the run directory, and explicitly
        supplied path-valued inputs only.  Writes stay under the run directory.
        """

        read_dirs = {
            Path("/System/Library"),
            Path("/usr/lib"),
            Path("/usr/share/locale"),
            Path("/Library/Frameworks"),
            Path("/Library/Fonts"),
            Path("/private/var/db/timezone"),
            Path(sys.prefix).resolve(),
            Path(sys.base_prefix).resolve(),
            Path(__file__).resolve().parents[2],
            Path(self.workdir).resolve(),
            Path(self.python_executable).resolve().parent,
        }
        read_files = {
            # CPython/conda enumerate the filesystem root while resolving the
            # executable prefix. Granting data access to this directory entry
            # (not a subpath) lets Python, pandas and EasyICU initialise while
            # preserving denial of files outside the explicit allow-list.
            Path("/"),
            self.cohort_parquet,
            script_path.resolve(),
            Path("/dev/null"),
            Path("/dev/urandom"),
            Path("/dev/random"),
        }
        for value in self.extra_env.values():
            candidate = Path(str(value)).expanduser()
            if not candidate.is_absolute() or not candidate.exists():
                continue
            resolved = candidate.resolve()
            if resolved.is_dir():
                read_dirs.add(resolved)
            else:
                read_files.add(resolved)

        rules = [
            "(version 1)",
            "(deny default)",
            "(allow process*)",
            "(allow sysctl-read)",
            "(allow mach-lookup)",
            "(allow ipc-posix-shm)",
        ]
        rules.extend(
            f'(allow file-read* (subpath "{_sandbox_quote(path)}"))'
            for path in sorted(read_dirs, key=str)
            if path.exists()
        )
        rules.extend(
            f'(allow file-read* (literal "{_sandbox_quote(path)}"))'
            for path in sorted(read_files, key=str)
            if path.exists()
        )
        rules.append(
            "(allow file-write* "
            f'(subpath "{_sandbox_quote(script_path.parent.resolve())}"))'
        )
        rules.append("(deny network*)")
        return "\n".join(rules) + "\n"

    def build_command(self, *, script_path: Path) -> List[str]:
        base = [self.python_executable, str(script_path)]
        if self.network_policy not in {"none", "disabled"}:
            return base
        sandbox_exec = shutil.which("sandbox-exec")
        if sandbox_exec and sys.platform == "darwin":
            profile = self._macos_sandbox_profile(script_path=script_path)
            return [sandbox_exec, "-p", profile, *base]
        unshare = shutil.which("unshare")
        if unshare and sys.platform.startswith("linux"):
            return [unshare, "-n", "--", *base]
        return base

    def run(
        self,
        *,
        step_id: str,
        code: str,
        resolved_inputs_path: Optional[Path] = None,
    ) -> RunResult:
        step_id = _safe_path_component(step_id, label="step_id")
        resolved_inputs_path = _validated_resolved_inputs_path(
            resolved_inputs_path,
            workdir=self.workdir,
        )
        step_dir = self.workdir / "steps" / step_id
        step_dir.mkdir(parents=True, exist_ok=True)
        script_path = step_dir / "analysis.py"
        out_dir = step_dir / "outputs"
        out_dir.mkdir(parents=True, exist_ok=True)
        log_path = step_dir / "run.log"

        # Hoist any top-level `def` / `class` that is referenced before it is
        # defined. Small coder models (notably qwen3-coder-30b) occasionally
        # emit helpers at the bottom of the file and reference them from the
        # top, which would otherwise fail with NameError and stall the
        # self-repair loop. No-op when the agent's code is already well-
        # ordered, so well-behaved models pay zero cost.
        code = reorder_forward_references(code)

        # Persist the script BEFORE running so it is hashable as evidence
        # even if execution crashes.
        script_path.write_text(code, encoding="utf-8")
        (
            authority_snapshot_path,
            authority_snapshot_sha256,
            authority_snapshot_error,
        ) = _authority_snapshot_for_code(
            code=code,
            workdir=self.workdir,
            step_dir=step_dir,
        )

        # Generated code gets only a small non-secret ambient environment.
        # API keys, cloud credentials, SSH agent sockets and unrelated project
        # variables are excluded unless a caller deliberately supplies them in
        # ``extra_env``.
        env = {
            key: os.environ[key]
            for key in _SAFE_INHERITED_ENV_KEYS
            if os.environ.get(key)
        }
        private_home = step_dir / ".home"
        private_tmp = step_dir / ".tmp"
        private_home.mkdir(parents=True, exist_ok=True)
        private_tmp.mkdir(parents=True, exist_ok=True)
        env["HOME"] = str(private_home)
        env["TMPDIR"] = str(private_tmp)
        env["TMP"] = str(private_tmp)
        env["TEMP"] = str(private_tmp)
        env["PYTHONNOUSERSITE"] = "1"
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])
        env["COHORT_PARQUET"] = str(self.cohort_parquet)
        env["STEP_OUT_DIR"] = str(out_dir)
        env["EASYICU_RUN_DIR"] = str(self.workdir.resolve())
        env["EASYICU_EVIDENCE_DIR"] = str((self.workdir / "evidence").resolve())
        env["EASYICU_MANIFEST_PARTIAL"] = str(
            (self.workdir / "manifest_partial.json").resolve()
        )
        if authority_snapshot_path is not None and authority_snapshot_sha256:
            env[_RUN_ARTIFACT_AUTHORITY_SNAPSHOT_ENV] = str(authority_snapshot_path)
            env[_RUN_ARTIFACT_AUTHORITY_SNAPSHOT_SHA_ENV] = authority_snapshot_sha256
        elif authority_snapshot_error:
            env[_RUN_ARTIFACT_AUTHORITY_ERROR_ENV] = authority_snapshot_error
        # Defensive aliases: agent-emitted scripts frequently invent
        # alternative env-var names for the canonical inputs (e.g.
        # ``STEP_OUTPUT_DIR``, ``EASYICU_OUTPUT_DIR``, ``OUT_DIR``,
        # ``COHORT_PATH``, ``EASYICU_COHORT_PATH``). Exposing every
        # observed variant under the same value prevents a hallucinated
        # name from silently writing artefacts to the wrong directory
        # (and later being missed by the evidence registrar) or aborting
        # with ``KeyError``/``FileNotFoundError``.
        for cohort_alias in (
            "COHORT_PATH",
            "EASYICU_COHORT_PATH",
            "EASYICU_COHORT_PARQUET",
        ):
            env[cohort_alias] = str(self.cohort_parquet)
        for out_alias in (
            "STEP_OUTPUT_DIR",
            "STEP_OUTPUT",
            "OUT_DIR",
            "OUTPUT_DIR",
            "EASYICU_OUTPUT_DIR",
            "EASYICU_STEP_OUT_DIR",
        ):
            env[out_alias] = str(out_dir)
        env["MPLBACKEND"] = "Agg"
        mpl_config_dir = step_dir / ".matplotlib"
        mpl_config_dir.mkdir(parents=True, exist_ok=True)
        env["MPLCONFIGDIR"] = str(mpl_config_dir)
        env["PYTHONIOENCODING"] = "utf-8"
        # macOS sandbox-exec can block shared-memory paths used by threaded
        # BLAS/OpenMP runtimes. Single-thread defaults keep generated ICU
        # analyses deterministic and avoid OMP SHM crashes inside the sandbox.
        # Force, rather than default, because users' shells often export
        # multi-threaded BLAS/OpenMP settings. Under macOS sandbox-exec those
        # inherited settings can abort before Python reaches user code
        # (for example: ``OMP: Error #179: Can't open SHM2``).
        env["OMP_NUM_THREADS"] = "1"
        env["MKL_NUM_THREADS"] = "1"
        env["OPENBLAS_NUM_THREADS"] = "1"
        env["VECLIB_MAXIMUM_THREADS"] = "1"
        env["NUMEXPR_NUM_THREADS"] = "1"
        env["JOBLIB_MULTIPROCESSING"] = "0"
        env["KMP_INIT_AT_FORK"] = "FALSE"
        env.update(self.extra_env)
        if resolved_inputs_path is not None:
            env["EASYICU_RESOLVED_INPUTS_JSON"] = str(resolved_inputs_path)

        timed_out = False
        started = time.monotonic()
        cmd = self.build_command(script_path=script_path)
        original_cmd = list(cmd)
        requested_isolation = self._isolation_backend_for_cmd(original_cmd)
        isolation_degraded = False
        isolation_degradation_reason: Optional[str] = None
        unsafe_direct_backends = {
            "host_subprocess",
            "linux_unshare_network_namespace",
            "network_allowed",
        }
        if (
            requested_isolation in unsafe_direct_backends
            and not self.allow_unsafe_host_fallback
        ):
            duration = time.monotonic() - started
            stderr = (
                "[CodeRunner] execution blocked by fail-closed policy: "
                f"{requested_isolation} does not isolate generated code from "
                "the host filesystem. Use DockerRunner, macOS sandbox-exec, or "
                "set allow_unsafe_host_fallback=True for explicit development-only "
                "degraded execution."
            )
            log_path.write_text(
                textwrap.dedent(f"""
                    === step {step_id} ===
                    cmd: {' '.join(cmd)}
                    cwd: {step_dir}
                    cohort: {self.cohort_parquet}
                    network_policy: {self.network_policy}
                    requested_isolation: {requested_isolation}
                    effective_isolation: blocked_fail_closed
                    allow_unsafe_host_fallback: {self.allow_unsafe_host_fallback}
                    isolation_degraded: False
                    returncode: 126
                    timed_out: False
                    duration_seconds: {duration:.3f}
                    ---- stdout ----

                    ---- stderr ----
                    {stderr}
                    """).strip(),
                encoding="utf-8",
            )
            return RunResult(
                step_id=step_id,
                script_path=script_path,
                cwd=step_dir,
                out_dir=out_dir,
                stdout="",
                stderr=stderr,
                returncode=126,
                duration_seconds=duration,
                artefacts=[],
                timed_out=False,
                requested_network_policy=self.network_policy,
                effective_isolation="blocked_fail_closed",
                isolation_degraded=False,
                isolation_degradation_reason=None,
                runner_log_path=log_path,
            )
        if requested_isolation == "host_subprocess":
            isolation_degraded = True
            isolation_degradation_reason = (
                "No filesystem-isolating backend was available; generated code is "
                "running as a host subprocess with a scrubbed environment."
            )
        elif requested_isolation == "linux_unshare_network_namespace":
            isolation_degraded = True
            isolation_degradation_reason = (
                "Linux unshare isolates the network namespace but not the host "
                "filesystem; use DockerRunner for paper-facing execution."
            )
        elif requested_isolation == "network_allowed":
            isolation_degraded = True
            isolation_degradation_reason = (
                "Network access was explicitly enabled and the host filesystem "
                "is not isolated; use DockerRunner for untrusted generated code."
            )
        try:
            proc = subprocess.run(  # noqa: S603 - intentional, generated script
                cmd,
                cwd=str(step_dir),
                env=env,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                encoding="utf-8",
                errors="replace",
            )
            stdout, stderr, returncode = proc.stdout, proc.stderr, proc.returncode
            if (
                returncode != 0
                and self.allow_unsafe_host_fallback
                and original_cmd
                and Path(original_cmd[0]).name == "unshare"
                and sys.platform.startswith("linux")
                and "unshare failed" in stderr.lower()
            ):
                retry_cmd = [self.python_executable, str(script_path)]
                retry_timeout = max(
                    self.timeout_seconds - (time.monotonic() - started), 1.0
                )
                retry_proc = (
                    subprocess.run(  # noqa: S603 - intentional, generated script
                        retry_cmd,
                        cwd=str(step_dir),
                        env=env,
                        capture_output=True,
                        text=True,
                        timeout=retry_timeout,
                        encoding="utf-8",
                        errors="replace",
                    )
                )
                stdout = retry_proc.stdout
                stderr = (
                    "[CodeRunner] unshare network isolation unavailable; "
                    "retrying without Linux network namespace isolation.\n"
                    f"[CodeRunner] original stderr:\n{stderr}\n"
                    f"[CodeRunner] fallback stderr:\n{retry_proc.stderr}"
                )
                returncode = retry_proc.returncode
                cmd = retry_cmd
                isolation_degraded = True
                isolation_degradation_reason = "unshare network namespace isolation failed; retried as a host subprocess."
            if (
                returncode != 0
                and self.allow_unsafe_host_fallback
                and original_cmd
                and Path(original_cmd[0]).name == "sandbox-exec"
                and sys.platform == "darwin"
                and "sandbox_apply" in stderr.lower()
                and "operation not permitted" in stderr.lower()
            ):
                retry_cmd = [self.python_executable, str(script_path)]
                retry_timeout = max(
                    self.timeout_seconds - (time.monotonic() - started), 1.0
                )
                retry_proc = subprocess.run(  # noqa: S603 - argv list, no shell
                    retry_cmd,
                    cwd=str(step_dir),
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=retry_timeout,
                    encoding="utf-8",
                    errors="replace",
                )
                stdout = retry_proc.stdout
                stderr = (
                    "[CodeRunner] macOS sandbox-exec could not apply its profile "
                    "inside the current outer sandbox; retrying without sandbox-exec "
                    "while keeping generated code under captured provenance.\n"
                    f"[CodeRunner] original stderr:\n{stderr}\n"
                    f"[CodeRunner] fallback stderr:\n{retry_proc.stderr}"
                )
                returncode = retry_proc.returncode
                cmd = retry_cmd
                isolation_degraded = True
                isolation_degradation_reason = (
                    "macOS sandbox-exec profile application was denied by the "
                    "outer sandbox; retried as a host subprocess."
                )
            if (
                returncode != 0
                and self.allow_unsafe_host_fallback
                and original_cmd
                and Path(original_cmd[0]).name == "sandbox-exec"
                and sys.platform == "darwin"
                and (
                    "init_sys_streams" in stderr.lower()
                    or "can't initialize sys standard streams" in stderr.lower()
                    or "bad file descriptor" in stderr.lower()
                )
            ):
                retry_cmd = [self.python_executable, str(script_path)]
                retry_timeout = max(
                    self.timeout_seconds - (time.monotonic() - started), 1.0
                )
                retry_proc = subprocess.run(  # noqa: S603 - argv list, no shell
                    retry_cmd,
                    cwd=str(step_dir),
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=retry_timeout,
                    encoding="utf-8",
                    errors="replace",
                )
                stdout = retry_proc.stdout
                stderr = (
                    "[CodeRunner] macOS sandbox-exec prevented Python stdio initialisation; "
                    "retrying without sandbox-exec while keeping generated code under "
                    "captured provenance.\n"
                    f"[CodeRunner] original stderr:\n{stderr}\n"
                    f"[CodeRunner] fallback stderr:\n{retry_proc.stderr}"
                )
                returncode = retry_proc.returncode
                cmd = retry_cmd
                isolation_degraded = True
                isolation_degradation_reason = (
                    "macOS sandbox-exec blocked Python stdio initialisation; "
                    "retried as a host subprocess."
                )
            if (
                returncode != 0
                and self.allow_unsafe_host_fallback
                and original_cmd
                and Path(original_cmd[0]).name == "sandbox-exec"
                and sys.platform == "darwin"
                and "omp: error #179" in stderr.lower()
                and "shm" in stderr.lower()
            ):
                retry_cmd = [self.python_executable, str(script_path)]
                retry_timeout = max(
                    self.timeout_seconds - (time.monotonic() - started), 1.0
                )
                retry_proc = subprocess.run(  # noqa: S603 - argv list, no shell
                    retry_cmd,
                    cwd=str(step_dir),
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=retry_timeout,
                    encoding="utf-8",
                    errors="replace",
                )
                stdout = retry_proc.stdout
                stderr = (
                    "[CodeRunner] macOS sandbox-exec blocked numeric runtime shared memory; "
                    "retrying without sandbox-exec while keeping network-free generated code "
                    "under captured provenance.\n"
                    f"[CodeRunner] original stderr:\n{stderr}\n"
                    f"[CodeRunner] fallback stderr:\n{retry_proc.stderr}"
                )
                returncode = retry_proc.returncode
                cmd = retry_cmd
                isolation_degraded = True
                isolation_degradation_reason = (
                    "macOS sandbox-exec blocked numeric runtime shared memory; "
                    "retried as a host subprocess."
                )
            if (
                returncode < 0
                and self.allow_unsafe_host_fallback
                and original_cmd
                and Path(original_cmd[0]).name == "sandbox-exec"
                and sys.platform == "darwin"
                and not stderr.strip()
            ):
                retry_cmd = [self.python_executable, str(script_path)]
                retry_timeout = max(
                    self.timeout_seconds - (time.monotonic() - started), 1.0
                )
                retry_proc = subprocess.run(  # noqa: S603 - argv list, no shell
                    retry_cmd,
                    cwd=str(step_dir),
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=retry_timeout,
                    encoding="utf-8",
                    errors="replace",
                )
                stdout = retry_proc.stdout
                stderr = (
                    "[CodeRunner] macOS sandbox-exec terminated the Python "
                    "runtime without diagnostics; retrying with the scrubbed "
                    "environment but without filesystem isolation.\n"
                    f"[CodeRunner] sandbox returncode: {returncode}\n"
                    f"[CodeRunner] fallback stderr:\n{retry_proc.stderr}"
                )
                returncode = retry_proc.returncode
                cmd = retry_cmd
                isolation_degraded = True
                isolation_degradation_reason = (
                    "macOS sandbox-exec terminated the Python runtime; retried "
                    "as a host subprocess with a scrubbed environment."
                )
            if (
                returncode != 0
                and not self.allow_unsafe_host_fallback
                and original_cmd
                and Path(original_cmd[0]).name in {"sandbox-exec", "unshare"}
            ):
                stderr = (
                    "[CodeRunner] isolation backend failed; fail-closed policy "
                    "forbids retrying generated code as a host subprocess. "
                    "Set allow_unsafe_host_fallback=True (development only) to "
                    "opt in to degraded execution.\n"
                    f"{stderr}"
                )
            duration = time.monotonic() - started
        except subprocess.TimeoutExpired as exc:
            # exc.stdout/stderr may be bytes even under text mode — decode before
            # concatenating the str status line (else TypeError: can't concat
            # str to bytes; observed once on a slow clustering step).
            stdout = _as_text(exc.stdout)
            stderr = _as_text(exc.stderr) + (
                f"\n[CodeRunner] timed out after {self.timeout_seconds}s\n"
            )
            returncode = -1
            duration = time.monotonic() - started
            timed_out = True

        log_path.write_text(
            textwrap.dedent(f"""
                === step {step_id} ===
                cmd: {' '.join(cmd)}
                original_cmd: {' '.join(original_cmd)}
                cwd: {step_dir}
                cohort: {self.cohort_parquet}
                network_policy: {self.network_policy}
                allow_unsafe_host_fallback: {self.allow_unsafe_host_fallback}
                requested_isolation: {requested_isolation}
                effective_isolation: {self._isolation_backend_for_cmd(cmd)}
                isolation_degraded: {isolation_degraded}
                isolation_degradation_reason: {isolation_degradation_reason or ""}
                returncode: {returncode}
                timed_out: {timed_out}
                duration_seconds: {duration:.3f}
                ---- stdout ----
                {stdout}
                ---- stderr ----
                {stderr}
                """).strip(),
            encoding="utf-8",
        )

        artefacts = sorted(p for p in out_dir.iterdir() if p.is_file())
        return RunResult(
            step_id=step_id,
            script_path=script_path,
            cwd=step_dir,
            out_dir=out_dir,
            stdout=stdout,
            stderr=stderr,
            returncode=returncode,
            duration_seconds=duration,
            artefacts=artefacts,
            timed_out=timed_out,
            requested_network_policy=self.network_policy,
            effective_isolation=self._isolation_backend_for_cmd(cmd),
            isolation_degraded=isolation_degraded,
            isolation_degradation_reason=isolation_degradation_reason,
            runner_log_path=log_path,
        )


class DockerRunner:
    """Run agent-generated Python inside a ``docker run`` container.

    T3.1 — opt-in sandbox. Same contract as :class:`CodeRunner`
    (``run(step_id=..., code=...) -> RunResult``) so the pipeline
    can swap backends without any other code changes.

    Defaults are chosen to be safe-by-default:

    * ``--network none`` — the agent-emitted script cannot make a
      network call. If you need PubMed lookups *inside* a step,
      pass ``network="bridge"`` explicitly.
    * cohort parquet is mounted **read-only** at a fixed container
      path (``/cohort.parquet``) and ``COHORT_PARQUET`` points at it.
    * the run root, including the script and step directory, is mounted
      read-only at ``/easyicu-run``; only the current step's ``outputs``
      directory is overlaid read-write.
    * ``--rm`` so containers don't pile up; ``--init`` so signal
      handling is sane.
    * the host's ``docker`` binary must be on PATH and the image
      must already be present (``docker pull`` is opt-in via
      ``pull_image=True``).

    The image is expected to provide Python plus the agent script's
    runtime deps advertised by :mod:`method_capabilities`. A reference
    Dockerfile ships at
    ``src/easyicu/research_agent/runner_image/Dockerfile``; build
    with::

        docker build -t easyicu-research-agent:latest \\
            -f src/easyicu/research_agent/runner_image/Dockerfile .

    Subclassing for OpenHands or any other sandbox is intentionally
    cheap: override :meth:`build_command` (which returns the argv
    list passed to ``subprocess.run``) and :meth:`prepare_step_dir`
    if your sandbox needs a different mount strategy.
    """

    DEFAULT_IMAGE = "easyicu-research-agent:latest"
    manages_output_cleanup = True
    CONTAINER_RUN_ROOT = "/easyicu-run"
    CONTAINER_COHORT_PATH = "/cohort.parquet"
    CONTAINER_INPUT_ROOT = "/easyicu-inputs"

    def __init__(
        self,
        *,
        workdir: Path,
        cohort_parquet: Path,
        timeout_seconds: float = 300.0,
        image: Optional[str] = None,
        docker_executable: Optional[str] = None,
        network: str = "none",
        extra_mounts: Optional[Sequence[Tuple[str, str, str]]] = None,
        extra_env: Optional[Dict[str, str]] = None,
        pull_image: bool = False,
        cpu_limit: Optional[str] = None,
        memory_limit: Optional[str] = None,
        user: Optional[str] = None,
        platform: Optional[str] = None,
    ) -> None:
        set_runtime_capability_snapshot_provider(None)
        self.workdir = Path(workdir).expanduser().resolve()
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.cohort_parquet = Path(cohort_parquet).resolve()
        if not self.cohort_parquet.exists():
            raise FileNotFoundError(
                f"Cohort parquet does not exist: {self.cohort_parquet}"
            )
        self.timeout_seconds = timeout_seconds
        self.image = image or os.environ.get("EASYICU_RUNNER_IMAGE", self.DEFAULT_IMAGE)
        self.docker_executable = (
            docker_executable or os.environ.get("EASYICU_DOCKER_EXECUTABLE") or "docker"
        )
        self.network = network
        self.extra_mounts = list(extra_mounts or [])
        self.extra_env = dict(extra_env or {})
        self.pull_image = bool(pull_image)
        self.cpu_limit = cpu_limit
        self.memory_limit = memory_limit
        if user is not None:
            self.user = user
        elif os.name == "posix" and hasattr(os, "getuid") and hasattr(os, "getgid"):
            self.user = f"{os.getuid()}:{os.getgid()}"
        else:
            self.user = None
        self.platform = platform
        self._provenance_lock = threading.Lock()
        self._cached_runtime_provenance: Optional[Dict[str, object]] = None
        self._cached_runtime_requirements: Optional[str] = None
        # Resolve the docker binary up front so we can produce a
        # readable error before the pipeline gets too far.
        resolved = shutil.which(self.docker_executable)
        if resolved is None:
            raise FileNotFoundError(
                f"Docker executable {self.docker_executable!r} not found on PATH. "
                "Either install Docker, set EASYICU_DOCKER_EXECUTABLE to the binary, "
                "or fall back to the subprocess CodeRunner "
                "(``runner_kind='subprocess'`` in ResearchAgentPipeline)."
            )
        self.docker_executable = resolved
        # The coder prompt is rendered after the runner is constructed. Its
        # allow-list must come from this image snapshot, never from host packages.
        set_runtime_capability_snapshot_provider(self._method_capability_snapshot)

    def _container_step_dir(self, step_id: str) -> str:
        safe_step_id = _safe_path_component(step_id, label="step_id")
        return f"{self.CONTAINER_RUN_ROOT}/steps/{safe_step_id}"

    def _containerise_extra_env(
        self,
    ) -> Tuple[Dict[str, str], List[Tuple[str, str, str]]]:
        """Rewrite explicit host paths to mounted read-only container paths."""

        rewritten: Dict[str, str] = {}
        mounts: List[Tuple[str, str, str]] = []
        run_root = self.workdir.resolve()
        for index, (key, raw_value) in enumerate(sorted(self.extra_env.items())):
            value = str(raw_value)
            candidate = Path(value).expanduser()
            if not candidate.is_absolute() or not candidate.exists():
                rewritten[key] = value
                continue
            resolved = candidate.resolve()
            if resolved == self.cohort_parquet:
                rewritten[key] = self.CONTAINER_COHORT_PATH
                continue
            try:
                relative = resolved.relative_to(run_root)
            except ValueError:
                target = (
                    f"{self.CONTAINER_INPUT_ROOT}/{index:03d}_"
                    f"{resolved.name or 'input'}"
                )
                mounts.append((str(resolved), target, "ro"))
                rewritten[key] = target
            else:
                rewritten[key] = f"{self.CONTAINER_RUN_ROOT}/{relative.as_posix()}"
        return rewritten, mounts

    # ------------------------------------------------------------------
    # Hooks subclasses may override
    # ------------------------------------------------------------------

    def prepare_step_dir(self, step_id: str) -> Tuple[Path, Path, Path]:
        """Lay out the per-step directory and return the key paths."""
        step_id = _safe_path_component(step_id, label="step_id")
        steps_dir = self.workdir / "steps"
        self._ensure_real_directory(steps_dir, replace_unsafe=False)
        step_dir = steps_dir / step_id
        if step_dir.parent != steps_dir:
            raise ValueError("step directory must remain under the run steps root")
        self._ensure_real_directory(step_dir, replace_unsafe=False)
        script_path = step_dir / "analysis.py"
        out_dir = step_dir / "outputs"
        self._ensure_real_directory(out_dir, replace_unsafe=True)
        return step_dir, script_path, out_dir

    @staticmethod
    def _remove_lexical_path(path: Path) -> None:
        """Remove exactly ``path`` without following a final symlink."""

        try:
            mode = os.lstat(path).st_mode
        except FileNotFoundError:
            return
        if stat.S_ISDIR(mode) and not stat.S_ISLNK(mode):
            shutil.rmtree(path)
        else:
            path.unlink()

    @classmethod
    def _ensure_real_directory(
        cls,
        path: Path,
        *,
        replace_unsafe: bool,
    ) -> None:
        """Ensure ``path`` itself is a directory, never a symlink target."""

        try:
            mode = os.lstat(path).st_mode
        except FileNotFoundError:
            path.mkdir(parents=False)
            return
        if stat.S_ISDIR(mode) and not stat.S_ISLNK(mode):
            return
        if not replace_unsafe:
            raise RuntimeError(f"DockerRunner requires a real directory: {path}")
        cls._remove_lexical_path(path)
        path.mkdir(parents=False)

    @classmethod
    def _write_regular_file(cls, path: Path, content: str) -> None:
        """Replace a possibly hostile path with one single-link regular file."""

        cls._remove_lexical_path(path)
        path.write_text(content, encoding="utf-8")
        metadata = os.lstat(path)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            cls._remove_lexical_path(path)
            raise RuntimeError(f"DockerRunner could not secure host file: {path}")

    @staticmethod
    def _clear_step_outputs(out_dir: Path) -> None:
        """Clear a quiescent step output directory without following symlinks."""

        DockerRunner._ensure_real_directory(out_dir, replace_unsafe=True)
        for child in out_dir.iterdir():
            DockerRunner._remove_lexical_path(child)

    def build_command(
        self,
        *,
        step_id: str,
        script_path: Path,
        out_dir: Path,
        runtime_image: Optional[str] = None,
        resolved_inputs_path: Optional[Path] = None,
        authority_snapshot_path: Optional[Path] = None,
        authority_snapshot_sha256: Optional[str] = None,
        authority_snapshot_error: Optional[str] = None,
    ) -> List[str]:
        """Compose the ``docker run`` argv for a single step."""
        step_id = _safe_path_component(step_id, label="step_id")
        resolved_inputs_path = _validated_resolved_inputs_path(
            resolved_inputs_path,
            workdir=self.workdir,
        )
        authority_snapshot_path = _validated_resolved_inputs_path(
            authority_snapshot_path,
            workdir=self.workdir,
        )
        if authority_snapshot_path is not None:
            digest = str(authority_snapshot_sha256 or "").strip().lower()
            if len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest
            ):
                raise ValueError(
                    "authority_snapshot_sha256 must be a SHA-256 hex digest"
                )
            authority_snapshot_sha256 = digest
        elif authority_snapshot_sha256:
            raise ValueError(
                "authority_snapshot_path is required when its digest is supplied"
            )
        container_step_dir = self._container_step_dir(step_id)
        cmd: List[str] = [
            self.docker_executable,
            "run",
            "--rm",
            "--init",
            "--read-only",
            "--cap-drop=ALL",
            "--security-opt=no-new-privileges",
            "--tmpfs=/tmp:rw,noexec,nosuid,size=64m",
            f"--network={self.network}",
            f"--workdir={container_step_dir}",
        ]
        if self.platform:
            cmd.append(f"--platform={self.platform}")
        if self.user:
            cmd.append(f"--user={self.user}")
        if self.cpu_limit:
            cmd.append(f"--cpus={self.cpu_limit}")
        if self.memory_limit:
            cmd.append(f"--memory={self.memory_limit}")

        # Mount the complete run tree read-only so deterministic runners can
        # resolve ``STEP_OUT_DIR.parents[2]`` to the run root. Overlay only the
        # current outputs directory read-write; the script, cwd, all other run
        # artefacts, and the cohort remain immutable to generated code.
        cmd.extend(
            [
                "--mount",
                (
                    f"type=bind,source={str(self.workdir.resolve())},"
                    f"target={self.CONTAINER_RUN_ROOT},readonly"
                ),
                "--mount",
                (
                    f"type=bind,source={str(out_dir.resolve())},"
                    f"target={container_step_dir}/outputs"
                ),
                "--mount",
                (
                    f"type=bind,source={str(self.cohort_parquet)},"
                    f"target={self.CONTAINER_COHORT_PATH},readonly"
                ),
            ]
        )
        rewritten_extra_env, path_mounts = self._containerise_extra_env()
        for source, target, mode in [*self.extra_mounts, *path_mounts]:
            entry = f"type=bind,source={source},target={target}"
            if mode and "ro" in mode.lower():
                entry += ",readonly"
            cmd.extend(["--mount", entry])

        # Env. The container sees absolute container paths; the host
        # path is irrelevant inside.
        env = {
            "COHORT_PARQUET": self.CONTAINER_COHORT_PATH,
            "COHORT_PATH": self.CONTAINER_COHORT_PATH,
            "EASYICU_COHORT_PATH": self.CONTAINER_COHORT_PATH,
            "EASYICU_COHORT_PARQUET": self.CONTAINER_COHORT_PATH,
            "STEP_OUT_DIR": f"{container_step_dir}/outputs",
            "STEP_OUTPUT_DIR": f"{container_step_dir}/outputs",
            "STEP_OUTPUT": f"{container_step_dir}/outputs",
            "OUT_DIR": f"{container_step_dir}/outputs",
            "OUTPUT_DIR": f"{container_step_dir}/outputs",
            "EASYICU_OUTPUT_DIR": f"{container_step_dir}/outputs",
            "EASYICU_STEP_OUT_DIR": f"{container_step_dir}/outputs",
            "EASYICU_RUN_DIR": self.CONTAINER_RUN_ROOT,
            "EASYICU_EVIDENCE_DIR": f"{self.CONTAINER_RUN_ROOT}/evidence",
            "EASYICU_MANIFEST_PARTIAL": (
                f"{self.CONTAINER_RUN_ROOT}/manifest_partial.json"
            ),
            "MPLBACKEND": "Agg",
            "PYTHONIOENCODING": "utf-8",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
            "HOME": "/tmp",
            "MPLCONFIGDIR": "/tmp/matplotlib",
            "XDG_CACHE_HOME": "/tmp/.cache",
        }
        env.update(rewritten_extra_env)
        if resolved_inputs_path is not None:
            relative_manifest = resolved_inputs_path.relative_to(
                self.workdir.resolve()
            )
            env["EASYICU_RESOLVED_INPUTS_JSON"] = (
                f"{self.CONTAINER_RUN_ROOT}/{relative_manifest.as_posix()}"
            )
        if authority_snapshot_path is not None and authority_snapshot_sha256:
            relative_snapshot = authority_snapshot_path.relative_to(
                self.workdir.resolve()
            )
            env[_RUN_ARTIFACT_AUTHORITY_SNAPSHOT_ENV] = (
                f"{self.CONTAINER_RUN_ROOT}/{relative_snapshot.as_posix()}"
            )
            env[_RUN_ARTIFACT_AUTHORITY_SNAPSHOT_SHA_ENV] = authority_snapshot_sha256
        elif authority_snapshot_error:
            env[_RUN_ARTIFACT_AUTHORITY_ERROR_ENV] = " ".join(
                str(authority_snapshot_error).split()
            )[:1000]
        for key, value in env.items():
            cmd.extend(["-e", f"{key}={value}"])

        # Production passes the immutable sha256 id captured before execution;
        # the mutable tag remains metadata only.
        cmd.append(runtime_image or self.image)
        # Use python -u for
        # unbuffered stdout so streaming logs don't surprise people.
        cmd.extend(
            [
                "python",
                "-u",
                f"{container_step_dir}/{script_path.name}",
            ]
        )
        return cmd

    def _capture_runtime_provenance(self) -> Tuple[Dict[str, object], str]:
        """Inspect the exact image and capture its installed Python packages.

        The result is cached per runner so concurrent/repeated steps share one
        immutable environment snapshot.  Failure is fatal: a Docker run without
        an image identity and execution-runtime lockfile is not submission-grade.
        """

        with self._provenance_lock:
            if (
                self._cached_runtime_provenance is not None
                and self._cached_runtime_requirements is not None
            ):
                return (
                    dict(self._cached_runtime_provenance),
                    self._cached_runtime_requirements,
                )
            inspect_proc = subprocess.run(  # noqa: S603 - argv list, no shell
                [
                    self.docker_executable,
                    "image",
                    "inspect",
                    "--format={{json .}}",
                    self.image,
                ],
                capture_output=True,
                text=True,
                timeout=max(30.0, min(self.timeout_seconds, 120.0)),
                encoding="utf-8",
                errors="replace",
            )
            if inspect_proc.returncode != 0:
                raise RuntimeError(
                    "Docker image provenance inspection failed: "
                    f"{inspect_proc.stderr.strip() or self.image}"
                )
            try:
                inspected = json.loads(inspect_proc.stdout)
                image_id = str(inspected["Id"])
                repo_digests = [str(x) for x in inspected.get("RepoDigests") or []]
            except Exception as exc:
                raise RuntimeError(
                    "Docker image provenance inspection returned invalid JSON"
                ) from exc
            if not image_id.startswith("sha256:"):
                raise RuntimeError(
                    "Docker image provenance is missing a sha256 image id"
                )

            freeze_proc = subprocess.run(  # noqa: S603 - argv list, no shell
                [
                    self.docker_executable,
                    "run",
                    "--rm",
                    "--network=none",
                    "--read-only",
                    "--cap-drop=ALL",
                    "--security-opt=no-new-privileges",
                    "--tmpfs=/tmp:rw,noexec,nosuid,size=32m",
                    *([f"--user={self.user}"] if self.user else []),
                    "-e",
                    "HOME=/tmp",
                    image_id,
                    "python",
                    "-m",
                    "pip",
                    "freeze",
                    "--disable-pip-version-check",
                ],
                capture_output=True,
                text=True,
                timeout=max(60.0, min(self.timeout_seconds, 180.0)),
                encoding="utf-8",
                errors="replace",
            )
            requirements = freeze_proc.stdout.strip()
            if freeze_proc.returncode != 0 or not requirements:
                raise RuntimeError(
                    "Docker execution-runtime dependency capture failed: "
                    f"{freeze_proc.stderr.strip() or 'empty pip freeze output'}"
                )
            installed_distributions = _distributions_from_freeze(requirements)
            import_to_distribution = {
                **{name: name for name in BASELINE_PACKAGES},
                "sklearn": "scikit-learn",
                **{name: name for name in OPTIONAL_BASELINE_PACKAGES},
                **{
                    package.import_name: package.pip_name
                    for package in CURATED_METHOD_PACKAGES
                },
            }
            method_capabilities = sorted(
                import_name
                for import_name, distribution_name in import_to_distribution.items()
                if _normalise_distribution_name(distribution_name)
                in installed_distributions
            )
            missing_baseline = sorted(
                set(BASELINE_PACKAGES).difference(method_capabilities)
            )
            if missing_baseline:
                raise RuntimeError(
                    "Docker execution runtime is missing required baseline "
                    f"packages: {', '.join(missing_baseline)}"
                )
            requirements_text = (
                "# easyicu.research_agent — execution requirements.lock\n"
                "# runtime=docker\n"
                f"# docker_image_reference={self.image}\n"
                f"# docker_image_id={image_id}\n"
                f"# docker_repo_digests={','.join(repo_digests)}\n"
                "# generated_by=easyicu.research_agent.runner.DockerRunner\n"
                f"{requirements}\n"
            )
            provenance: Dict[str, object] = {
                "runtime": "docker",
                "image_reference": self.image,
                "image_id": image_id,
                "repo_digests": repo_digests,
                "network": self.network,
                "requirements_sha256": hashlib.sha256(
                    requirements_text.encode("utf-8")
                ).hexdigest(),
                "method_capabilities": method_capabilities,
            }
            self._cached_runtime_provenance = dict(provenance)
            self._cached_runtime_requirements = requirements_text
            return provenance, requirements_text

    def _method_capability_snapshot(self) -> Sequence[str]:
        provenance, _requirements = self._capture_runtime_provenance()
        snapshot = provenance.get("method_capabilities")
        if not isinstance(snapshot, list) or not all(
            isinstance(name, str) for name in snapshot
        ):
            raise RuntimeError(
                "Docker runtime provenance lacks a method capability snapshot"
            )
        return tuple(snapshot)

    @staticmethod
    def _container_reference(
        cidfile: Path,
        *,
        fallback_name: Optional[str] = None,
    ) -> Optional[str]:
        """Return a validated container id/name without trusting mounted code."""

        try:
            value = cidfile.read_text(encoding="utf-8").strip()
        except OSError:
            value = ""
        if value.startswith("name:"):
            value = value.removeprefix("name:")
        try:
            if len(value) == 64:
                int(value, 16)
                return value
        except ValueError:
            pass
        if value.startswith("easyicu-ra-") and all(
            char.isalnum() or char in "-_" for char in value
        ):
            return value
        return fallback_name

    def _teardown_container(self, container_ref: str) -> Tuple[bool, str]:
        """Stop, force-remove if needed, and wait for one container."""

        def _control(
            args: Sequence[str], *, timeout: float
        ) -> Optional[subprocess.CompletedProcess[str]]:
            try:
                return subprocess.run(  # noqa: S603 - argv list, no shell
                    [self.docker_executable, *args],
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    encoding="utf-8",
                    errors="replace",
                )
            except (OSError, subprocess.TimeoutExpired):
                return None

        teardown_confirmed = False
        cleanup_notes: List[str] = []
        teardown_commands = (
            ("stop", ("stop", "--timeout=5", container_ref), 15.0),
            ("kill", ("kill", container_ref), 10.0),
            ("rm", ("rm", "--force", container_ref), 10.0),
        )
        for label, args, timeout in teardown_commands:
            proc = _control(args, timeout=timeout)
            teardown_confirmed = proc is not None and proc.returncode == 0
            if teardown_confirmed:
                break
            cleanup_notes.append(f"{label} failed")

        wait_proc = _control(("wait", container_ref), timeout=10.0)
        if not teardown_confirmed and (
            wait_proc is None or wait_proc.returncode != 0
        ):
            cleanup_notes.append("wait failed")

        if not teardown_confirmed:
            inspect_proc = _control(
                ("container", "inspect", container_ref),
                timeout=10.0,
            )
            if inspect_proc is not None:
                inspect_error = inspect_proc.stderr.lower()
                teardown_confirmed = inspect_proc.returncode != 0 and (
                    "no such object" in inspect_error
                    or "no such container" in inspect_error
                )
            if not teardown_confirmed:
                cleanup_notes.append("container presence could not be excluded")

        if not teardown_confirmed:
            return False, (
                "[DockerRunner] timed-out container cleanup: "
                f"{', '.join(cleanup_notes)}\n"
            )
        return True, (
            "[DockerRunner] container teardown confirmed before output collection\n"
        )

    def _retry_stale_container_cleanup(self, step_id: str) -> None:
        """Resolve prior unconfirmed teardown before reusing a step directory."""

        pattern = f".docker-{glob.escape(step_id)}-*.sentinel"
        for sentinel in self.workdir.glob(pattern):
            container_ref = self._container_reference(sentinel)
            if container_ref is None:
                raise RuntimeError(
                    "DockerRunner found an invalid stale container sentinel; "
                    "refusing to reuse the step output directory"
                )
            teardown_confirmed, _note = self._teardown_container(container_ref)
            if not teardown_confirmed:
                raise RuntimeError(
                    "DockerRunner could not confirm stale container teardown; "
                    "refusing to reuse the step output directory"
                )
            sentinel.unlink(missing_ok=True)
            for suffix in (".cid", ".analysis.py", ".run.log"):
                sentinel.with_suffix(suffix).unlink(missing_ok=True)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(
        self,
        *,
        step_id: str,
        code: str,
        resolved_inputs_path: Optional[Path] = None,
    ) -> RunResult:
        step_id = _safe_path_component(step_id, label="step_id")
        self._retry_stale_container_cleanup(step_id)
        resolved_inputs_path = _validated_resolved_inputs_path(
            resolved_inputs_path,
            workdir=self.workdir,
        )
        # Same forward-reference hoisting as CodeRunner; see code_hygiene
        # docstring for the qwen3-coder-30b regression that motivates it.
        code = reorder_forward_references(code)
        step_dir, script_path, out_dir = self.prepare_step_dir(step_id)
        self._clear_step_outputs(out_dir)
        log_path = step_dir / "run.log"
        self._write_regular_file(script_path, code)
        self._remove_lexical_path(log_path)
        (
            authority_snapshot_path,
            authority_snapshot_sha256,
            authority_snapshot_error,
        ) = _authority_snapshot_for_code(
            code=code,
            workdir=self.workdir,
            step_dir=step_dir,
        )

        if self.pull_image:
            try:
                subprocess.run(  # noqa: S603 - argv list, no shell
                    [self.docker_executable, "pull", self.image],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=max(60.0, self.timeout_seconds),
                )
            except Exception:
                # pull failures are non-fatal; the run will surface
                # any "image not found" error in stderr below.
                pass

        runtime_provenance, runtime_requirements = self._capture_runtime_provenance()

        cmd = self.build_command(
            step_id=step_id,
            script_path=script_path,
            out_dir=out_dir,
            runtime_image=str(runtime_provenance["image_id"]),
            resolved_inputs_path=resolved_inputs_path,
            authority_snapshot_path=authority_snapshot_path,
            authority_snapshot_sha256=authority_snapshot_sha256,
            authority_snapshot_error=authority_snapshot_error,
        )
        # Keep the host-written cidfile outside the step's read-write mount so
        # generated code cannot replace the container id used for teardown.
        attempt_id = uuid.uuid4().hex
        cidfile = self.workdir / f".docker-{step_id}-{attempt_id}.cid"
        sentinel = self.workdir / f".docker-{step_id}-{attempt_id}.sentinel"
        control_script_path = sentinel.with_suffix(".analysis.py")
        control_log_path = sentinel.with_suffix(".run.log")
        container_name = f"easyicu-ra-{attempt_id}"
        self._write_regular_file(sentinel, f"name:{container_name}\n")
        self._write_regular_file(control_script_path, code)
        cmd.insert(2, f"--cidfile={cidfile}")
        cmd.insert(3, f"--name={container_name}")

        timed_out = False
        teardown_confirmed = False
        started = time.monotonic()
        try:
            proc = subprocess.run(  # noqa: S603 - argv list, no shell
                cmd,
                cwd=str(step_dir),
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                encoding="utf-8",
                errors="replace",
            )
            stdout, stderr, returncode = (
                proc.stdout,
                proc.stderr,
                proc.returncode,
            )
            if returncode == 0:
                teardown_confirmed = True
            else:
                container_ref = self._container_reference(
                    cidfile,
                    fallback_name=container_name,
                )
                assert container_ref is not None
                teardown_confirmed, cleanup_note = self._teardown_container(
                    container_ref
                )
                stderr = _as_text(stderr) + "\n" + cleanup_note
        except subprocess.TimeoutExpired as exc:
            stdout = _as_text(exc.stdout)
            stderr = _as_text(exc.stderr) + (
                f"\n[DockerRunner] timed out after {self.timeout_seconds}s\n"
            )
            container_ref = self._container_reference(
                cidfile,
                fallback_name=container_name,
            )
            assert container_ref is not None
            teardown_confirmed, cleanup_note = self._teardown_container(container_ref)
            stderr += cleanup_note
            returncode = -1
            timed_out = True
        duration = time.monotonic() - started

        log_content = textwrap.dedent(f"""
                === step {step_id} (DockerRunner) ===
                image: {self.image}
                image_id: {runtime_provenance.get("image_id")}
                repo_digests: {runtime_provenance.get("repo_digests")}
                network: {self.network}
                cohort: {self.cohort_parquet}
                cmd: {' '.join(cmd)}
                returncode: {returncode}
                timed_out: {timed_out}
                duration_seconds: {duration:.3f}
                ---- stdout ----
                {stdout}
                ---- stderr ----
                {stderr}
                """).strip()

        if teardown_confirmed:
            self._ensure_real_directory(step_dir, replace_unsafe=False)
            self._ensure_real_directory(out_dir, replace_unsafe=True)
            for output_path in list(out_dir.iterdir()):
                metadata = os.lstat(output_path)
                if stat.S_ISLNK(metadata.st_mode) or (
                    stat.S_ISREG(metadata.st_mode) and metadata.st_nlink != 1
                ):
                    self._remove_lexical_path(output_path)
            self._write_regular_file(script_path, code)
            safe_script_path = script_path
            safe_log_path = log_path
            self._write_regular_file(safe_log_path, log_content)
            requirements_path = out_dir / "runner_requirements.lock.txt"
            self._write_regular_file(requirements_path, runtime_requirements)
            provenance_path = out_dir / "runner_provenance.json"
            self._write_regular_file(
                provenance_path,
                json.dumps(runtime_provenance, indent=2, ensure_ascii=False) + "\n",
            )
            artefacts = []
            for output_path in out_dir.iterdir():
                metadata = os.lstat(output_path)
                if stat.S_ISREG(metadata.st_mode) and metadata.st_nlink == 1:
                    artefacts.append(output_path)
            artefacts.sort()
            for control_path in (cidfile, sentinel, control_script_path):
                control_path.unlink(missing_ok=True)
        else:
            safe_script_path = control_script_path
            safe_log_path = control_log_path
            self._write_regular_file(safe_log_path, log_content)
            artefacts = []
        return RunResult(
            step_id=step_id,
            script_path=safe_script_path,
            cwd=step_dir,
            out_dir=out_dir,
            stdout=stdout,
            stderr=stderr,
            returncode=returncode,
            duration_seconds=duration,
            artefacts=artefacts,
            timed_out=timed_out,
            requested_network_policy=f"docker:{self.network}",
            effective_isolation=f"docker_network_{self.network}",
            isolation_degraded=False,
            isolation_degradation_reason=None,
            runtime_provenance=runtime_provenance,
            outputs_safe_to_collect=teardown_confirmed,
            runner_log_path=safe_log_path,
        )


class SafeRunnerUnavailableError(RuntimeError):
    """No filesystem-isolating execution backend is ready for use."""


def select_safe_runner_kind(
    *,
    image: Optional[str] = None,
    docker_executable: Optional[str] = None,
    probe_timeout_seconds: float = 5.0,
) -> str:
    """Select a usable safe backend without silently falling back to host.

    Docker is preferred on every platform, but only after the configured image
    can be inspected through a live daemon. macOS may fall back to
    ``sandbox-exec``. Other hosts fail before generated code is launched and
    must configure Docker (or explicitly opt into the unsafe development-only
    host runner).
    """

    runtime_image = image or os.environ.get(
        "EASYICU_RUNNER_IMAGE", DockerRunner.DEFAULT_IMAGE
    )
    requested_executable = (
        docker_executable
        or os.environ.get("EASYICU_DOCKER_EXECUTABLE")
        or "docker"
    )
    resolved_docker = shutil.which(requested_executable)
    docker_detail = f"Docker executable {requested_executable!r} was not found"
    if resolved_docker is not None:
        try:
            probe = subprocess.run(  # noqa: S603 - fixed Docker argv, no shell
                [
                    resolved_docker,
                    "image",
                    "inspect",
                    runtime_image,
                    "--format={{.Id}}",
                ],
                capture_output=True,
                text=True,
                timeout=max(0.1, float(probe_timeout_seconds)),
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            docker_detail = f"Docker probe failed: {exc}"
        else:
            image_id = str(probe.stdout or "").strip()
            if probe.returncode == 0 and image_id.startswith("sha256:"):
                return "docker"
            detail = str(probe.stderr or probe.stdout or "").strip()
            docker_detail = (
                f"Docker image {runtime_image!r} is not ready"
                + (f": {detail[:240]}" if detail else "")
            )

    if sys.platform == "darwin" and shutil.which("sandbox-exec"):
        return "subprocess"

    raise SafeRunnerUnavailableError(
        "No safe generated-code runner is available. "
        f"{docker_detail}. Build or pull {runtime_image!r} with a live Docker "
        "daemon. For explicit development-only host execution, set "
        "runner_kind='subprocess' and "
        "runner_kwargs={'allow_unsafe_host_fallback': True}."
    )


__all__ = [
    "RunResult",
    "CodeRunner",
    "DockerRunner",
    "SafeRunnerUnavailableError",
    "select_safe_runner_kind",
]
