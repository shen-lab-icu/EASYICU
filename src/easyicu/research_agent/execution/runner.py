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

Hashes live in :mod:`easyicu.research_agent.authority.evidence_store`, not here
— these classes only produce the artefacts.
"""

from __future__ import annotations

import ast
import glob
import hashlib
import json
import os
import re
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import uuid
from pathlib import Path, PurePosixPath
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from .code_hygiene import reorder_forward_references
from ..contracts.method_packages import (
    BASELINE_PACKAGES,
    CURATED_METHOD_PACKAGES,
    FINGERPRINT_ONLY_DISTRIBUTIONS,
    OPTIONAL_BASELINE_PACKAGES,
)
from ..contracts.runtime import RunResult
from ..orchestration.profiles import is_paper_facing_profile
from .method_capabilities import set_runtime_capability_snapshot_provider

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


def _python_source_tree_sha256(root: Path) -> str:
    """Digest Python sources by relative path and bytes."""

    digest = hashlib.sha256()
    for path in sorted(Path(root).rglob("*.py")):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        payload = path.read_bytes()
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


# These coordinates are owned by the host runtime.  ``extra_env`` remains a
# supported extension surface for credentials and auxiliary read-only inputs,
# but it must never redirect the cohort, outputs, evidence, or replay receipts
# selected by the pipeline.
HOST_OWNED_RUNNER_ENV_KEYS = frozenset(
    {
        "COHORT_PARQUET",
        "COHORT_PATH",
        "EASYICU_COHORT_PATH",
        "EASYICU_COHORT_PARQUET",
        "EASYICU_COHORT_ROWS",
        "STEP_OUT_DIR",
        "STEP_OUTPUT_DIR",
        "STEP_OUTPUT",
        "OUT_DIR",
        "OUTPUT_DIR",
        "EASYICU_OUTPUT_DIR",
        "EASYICU_STEP_OUT_DIR",
        "EASYICU_RUN_DIR",
        "EASYICU_EVIDENCE_DIR",
        "EASYICU_MANIFEST_PARTIAL",
        _RUN_ARTIFACT_AUTHORITY_SNAPSHOT_ENV,
        _RUN_ARTIFACT_AUTHORITY_SNAPSHOT_SHA_ENV,
        _RUN_ARTIFACT_AUTHORITY_ERROR_ENV,
        "EASYICU_RESOLVED_INPUTS_JSON",
        "HOME",
        "TMPDIR",
        "TMP",
        "TEMP",
        "PYTHONPATH",
        "PYTHONNOUSERSITE",
        "PYTHONDONTWRITEBYTECODE",
        "MPLBACKEND",
        "MPLCONFIGDIR",
        "XDG_CACHE_HOME",
        "PYTHONIOENCODING",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "JOBLIB_MULTIPROCESSING",
        "KMP_INIT_AT_FORK",
    }
)
_RUNNER_ENV_KEY_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")


def _parquet_row_count(path: Path) -> Optional[int]:
    """Read the Parquet footer cardinality without scanning cohort columns."""

    try:
        import pyarrow.parquet as pq

        count = int(pq.read_metadata(Path(path)).num_rows)
    except (ImportError, OSError, TypeError, ValueError):
        return None
    return count if count >= 0 else None


def _reject_docker_mount_field(value: str, *, label: str) -> None:
    if any(character in value for character in ",=") or any(
        ord(character) < 32 or ord(character) == 127 for character in value
    ):
        raise ValueError(f"DockerRunner {label} contains unsafe mount syntax")


def _validated_real_mount_source(raw_source: str, *, label: str) -> Path:
    """Return one canonical regular-file/real-directory auxiliary input."""

    source = Path(raw_source).expanduser()
    try:
        resolved = source.resolve(strict=True)
        metadata = os.lstat(source)
    except OSError as exc:
        raise ValueError(f"DockerRunner {label} must be real input") from exc
    if not source.is_absolute() or source != resolved or stat.S_ISLNK(metadata.st_mode):
        raise ValueError(f"DockerRunner {label} must be real input")
    if stat.S_ISREG(metadata.st_mode):
        if metadata.st_nlink != 1:
            raise ValueError(f"DockerRunner {label} must be singly linked")
    elif stat.S_ISDIR(metadata.st_mode):
        if resolved == Path(resolved.anchor):
            raise ValueError(f"DockerRunner {label} cannot expose a filesystem root")
    else:
        raise ValueError(
            f"DockerRunner {label} must be a regular file or real directory"
        )
    _reject_docker_mount_field(str(resolved), label=label)
    return resolved


def _docker_mount_entry(source: str, target: str, *, readonly: bool) -> str:
    _reject_docker_mount_field(source, label="mount source")
    _reject_docker_mount_field(target, label="mount target")
    entry = f"type=bind,source={source},target={target}"
    return f"{entry},readonly" if readonly else entry


#: Depth limit for the generated-output sweep. Deep enough for the layouts
#: generated code actually uses (``figures/``, ``tables/``, ``models/``), shallow
#: enough that a runaway mkdir loop cannot stall evidence collection.
MAX_OUTPUT_ARTIFACT_DEPTH = 8

#: Ceilings on the generated-output tree. An evidence-bound run must not be
#: able to produce output the sweep silently skips, nor exhaust inodes or disk
#: while the sweep tries to enumerate it.
MAX_OUTPUT_ARTIFACT_FILES = 5_000
MAX_OUTPUT_ARTIFACT_DIRECTORIES = 1_000
MAX_OUTPUT_ARTIFACT_TOTAL_BYTES = 2 * 1024**3
MAX_OUTPUT_ARTIFACT_FILE_BYTES = 512 * 1024**2


class OutputArtifactPolicyError(RuntimeError):
    """Raised when generated output breaks the evidence-collection contract.

    Fail closed rather than skip: in an evidence-bound design an artefact the
    sweep cannot register must not be able to coexist with a successful run.
    """


def _collect_safe_output_artifacts(out_dir: Path) -> List[Path]:
    """Collect lexical single-link regular files from generated output.

    Recurses into subdirectories: generated code routinely writes
    ``outputs/figures/fig1.png`` and ``outputs/tables/table1.csv``, and a
    top-level-only sweep dropped those from ``RunResult.artefacts`` — meaning
    they never reached the SHA-256 evidence store even though the manuscript
    could cite them.

    Symlinks, hardlinked files and special files are still rejected (and
    removed) rather than collected: they can point outside the sandbox.
    """

    artefacts: List[Path] = []
    pending: List[tuple[Path, int]] = [(out_dir, 0)]
    directories = 0
    total_bytes = 0
    while pending:
        current, depth = pending.pop()
        try:
            entries = sorted(current.iterdir())
        except OSError as exc:
            # Skipping here would let an unreadable directory — or a transient
            # I/O error — drop its files from the evidence list while the run
            # still reported success. Under an evidence-bound contract an
            # unenumerable output directory is a failure, not a gap.
            raise OutputArtifactPolicyError(
                "cannot enumerate generated output directory " f"{current}: {exc}"
            ) from exc
        for output_path in entries:
            metadata = os.lstat(output_path)
            if stat.S_ISREG(metadata.st_mode) and metadata.st_nlink == 1:
                if metadata.st_size > MAX_OUTPUT_ARTIFACT_FILE_BYTES:
                    raise OutputArtifactPolicyError(
                        f"generated output {output_path.relative_to(out_dir)} is "
                        f"{metadata.st_size} bytes, over the "
                        f"{MAX_OUTPUT_ARTIFACT_FILE_BYTES}-byte per-file limit"
                    )
                total_bytes += metadata.st_size
                if total_bytes > MAX_OUTPUT_ARTIFACT_TOTAL_BYTES:
                    raise OutputArtifactPolicyError(
                        "generated output exceeds the "
                        f"{MAX_OUTPUT_ARTIFACT_TOTAL_BYTES}-byte total limit"
                    )
                artefacts.append(output_path)
                if len(artefacts) > MAX_OUTPUT_ARTIFACT_FILES:
                    raise OutputArtifactPolicyError(
                        f"generated output has more than {MAX_OUTPUT_ARTIFACT_FILES} "
                        "files; evidence collection refuses to enumerate it"
                    )
                continue
            if stat.S_ISDIR(metadata.st_mode):
                if depth >= MAX_OUTPUT_ARTIFACT_DEPTH:
                    raise OutputArtifactPolicyError(
                        f"generated output directory "
                        f"{output_path.relative_to(out_dir)} is nested deeper than "
                        f"{MAX_OUTPUT_ARTIFACT_DEPTH} levels; anything below it "
                        "would be omitted from the evidence artefact list"
                    )
                directories += 1
                if directories > MAX_OUTPUT_ARTIFACT_DIRECTORIES:
                    raise OutputArtifactPolicyError(
                        f"generated output has more than "
                        f"{MAX_OUTPUT_ARTIFACT_DIRECTORIES} directories"
                    )
                pending.append((output_path, depth + 1))
                continue
            # Symlink, hardlink, fifo, socket, device — never an artefact.
            output_path.unlink(missing_ok=True)
    artefacts.sort()
    return artefacts


def reject_reserved_runner_env(
    extra_env: Dict[str, str],
    *,
    reserved: Sequence[str] = tuple(HOST_OWNED_RUNNER_ENV_KEYS),
    owner: str = "runner",
) -> None:
    """Reject caller overrides of host-owned execution coordinates."""

    for key, value in extra_env.items():
        if not isinstance(key, str) or _RUNNER_ENV_KEY_RE.fullmatch(key) is None:
            raise ValueError(f"{owner} extra_env contains an invalid environment key")
        if not isinstance(value, str) or "\x00" in value:
            raise ValueError(f"{owner} extra_env contains an invalid environment value")
    conflicts = sorted(set(extra_env).intersection(reserved))
    if conflicts:
        raise ValueError(
            f"{owner} extra_env cannot override host-owned key(s): "
            + ", ".join(conflicts)
        )


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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_directory_tree(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(path.rglob("*"), key=lambda value: value.as_posix()):
        relative = item.relative_to(path).as_posix().encode("utf-8")
        mode = os.lstat(item).st_mode
        if stat.S_ISLNK(mode):
            raise RuntimeError(
                "CodeRunner authority cannot bind a symlinked extra_env tree"
            )
        digest.update(relative)
        digest.update(b"\0")
        if stat.S_ISREG(mode):
            digest.update(b"file\0")
            digest.update(_sha256_file(item).encode("ascii"))
        elif stat.S_ISDIR(mode):
            digest.update(b"dir\0")
        else:
            raise RuntimeError(
                "CodeRunner authority cannot bind a special extra_env path"
            )
        digest.update(b"\0")
    return digest.hexdigest()


def _path_bound_authority_value(value: object) -> object:
    candidate = Path(str(value)).expanduser()
    if candidate.is_absolute() and candidate.is_file():
        return {
            "path": str(candidate.resolve()),
            "sha256": _sha256_file(candidate),
        }
    if candidate.is_absolute() and candidate.is_dir():
        return {
            "path": str(candidate.resolve()),
            "tree_sha256": _sha256_directory_tree(candidate),
        }
    return str(value)


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


def _terminate_process_group(proc: "subprocess.Popen") -> None:
    """Best-effort SIGKILL of the child's whole process group (POSIX only).

    ``start_new_session=True`` makes the child a session/group leader, so its
    pgid equals its pid and signalling the group also reaps double-forked
    descendants that stayed in it. Descendants that call ``setsid`` themselves
    escape -- an inherent limit of group signalling, not specific to this code.
    """

    try:
        pgid = os.getpgid(proc.pid)
    except (ProcessLookupError, OSError):
        pgid = None
    if pgid is not None:
        try:
            os.killpg(pgid, signal.SIGKILL)
            return
        except (ProcessLookupError, OSError):
            pass
    try:
        proc.kill()
    except (ProcessLookupError, OSError):
        pass


def _run_capturing_with_descendant_reaping(
    cmd: Sequence[str],
    *,
    cwd: str,
    env: Mapping[str, str],
    timeout: float,
) -> subprocess.CompletedProcess:
    """Run ``cmd`` capturing text output; on timeout kill the whole group.

    ``subprocess.run(timeout=...)`` sends the timeout kill only to the direct
    child, so a background process spawned by generated code survives and can
    keep mutating step outputs after evidence has been collected. On POSIX the
    child is launched in a new session and, on timeout, the whole process group
    is signalled before ``TimeoutExpired`` is re-raised with the captured
    partial output (so the caller's bytes-safe timeout handler is unchanged).
    Non-POSIX platforms keep the plain ``subprocess.run`` behaviour.
    """

    capture = dict(
        cwd=cwd,
        env=dict(env),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if os.name != "posix":
        return subprocess.run(  # noqa: S603 - configured argv, no shell
            cmd, timeout=timeout, **capture
        )

    with subprocess.Popen(  # noqa: S603 - configured argv, no shell
        cmd, start_new_session=True, **capture
    ) as proc:
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            _terminate_process_group(proc)
            stdout, stderr = proc.communicate()
            raise subprocess.TimeoutExpired(cmd, timeout, output=stdout, stderr=stderr)
        return subprocess.CompletedProcess(cmd, proc.returncode, stdout, stderr)


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

    from ..authority.runtime_artifacts import (
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
        and node.module
        == "easyicu.research_agent.execution.runners.deterministic_robustness"
        and any(alias.name == _ROBUSTNESS_AUTHORITY_ENTRYPOINT for alias in node.names)
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
    _remove_authority_snapshot(step_dir / ".run_artifact_authority_snapshot.json")
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


def macos_sandbox_permission_denied(stderr: object) -> bool:
    """Return whether macOS refused the sandbox profile or its target exec.

    A nested application sandbox can reject either ``sandbox_apply`` itself or
    the subsequent ``execvp`` of a project virtualenv interpreter.  Require the
    exact sandbox marker together with ``Operation not permitted`` so ordinary
    generated-code failures never activate the development-only host fallback.
    """

    detail = _as_text(stderr).lower()
    return "operation not permitted" in detail and (
        "sandbox_apply" in detail or "sandbox-exec: execvp()" in detail
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
        reject_reserved_runner_env(self.extra_env, owner="CodeRunner")
        self.network_policy = (network_policy or "none").lower()
        # Strict: this flag disables host-isolation, so a non-bool must never be
        # silently coerced. ``bool("false")`` is ``True`` -- a quoted YAML/TOML/
        # env value or JSON string would otherwise enable unsafe host execution.
        if allow_unsafe_host_fallback is None:
            self.allow_unsafe_host_fallback = _env_flag(
                "EASYICU_ALLOW_UNSAFE_HOST_FALLBACK"
            )
        elif isinstance(allow_unsafe_host_fallback, bool):
            self.allow_unsafe_host_fallback = allow_unsafe_host_fallback
        else:
            raise TypeError(
                "allow_unsafe_host_fallback must be True, False, or None, not "
                f"{type(allow_unsafe_host_fallback).__name__}"
            )
        self._authority_identity_lock = threading.Lock()
        self._cached_authority_identity_sha256: Optional[str] = None
        # A host runner must never inherit a Docker capability snapshot left in
        # the same context by an earlier run.
        set_runtime_capability_snapshot_provider(None)

    def validate_runtime_capabilities(self) -> Tuple[str, ...]:
        """Verify the selected interpreter before planning executable work.

        Probe the exact configured interpreter, not the host process, and
        publish the verified allow-list used by the Coder prompt. Package
        installation is never attempted here.
        """

        import_names = tuple(
            dict.fromkeys(
                (
                    *BASELINE_PACKAGES,
                    *OPTIONAL_BASELINE_PACKAGES,
                    *(package.import_name for package in CURATED_METHOD_PACKAGES),
                )
            )
        )
        probe = (
            "import importlib.util, json\n"
            f"names = {list(import_names)!r}\n"
            "print(json.dumps([name for name in names "
            "if importlib.util.find_spec(name) is not None]))\n"
        )
        try:
            proc = subprocess.run(  # noqa: S603 - configured interpreter, no shell
                [self.python_executable, "-c", probe],
                capture_output=True,
                text=True,
                timeout=max(15.0, min(self.timeout_seconds, 60.0)),
                check=False,
                encoding="utf-8",
                errors="replace",
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise RuntimeError(
                "Python execution-runtime capability probe failed before planning"
            ) from exc
        if proc.returncode != 0:
            raise RuntimeError(
                "Python execution-runtime capability probe failed before planning: "
                + (proc.stderr.strip() or f"returncode={proc.returncode}")
            )
        try:
            snapshot = tuple(str(name) for name in json.loads(proc.stdout))
        except Exception as exc:
            raise RuntimeError(
                "Python execution-runtime capability probe returned invalid JSON"
            ) from exc
        missing_baseline = sorted(set(BASELINE_PACKAGES).difference(snapshot))
        if missing_baseline:
            raise RuntimeError(
                "Python execution runtime is missing required baseline packages: "
                + ", ".join(missing_baseline)
            )
        frozen_snapshot = tuple(snapshot)
        set_runtime_capability_snapshot_provider(lambda: frozen_snapshot)
        return frozen_snapshot

    @property
    def authority_identity_sha256(self) -> str:
        """Bind replay to this interpreter, packages, inputs, and isolation."""

        with self._authority_identity_lock:
            if self._cached_authority_identity_sha256 is not None:
                return self._cached_authority_identity_sha256
            distributions = {
                "scikit-learn" if package == "sklearn" else package
                for package in (*BASELINE_PACKAGES, *OPTIONAL_BASELINE_PACKAGES)
            }
            distributions.update(
                package.pip_name for package in CURATED_METHOD_PACKAGES
            )
            distributions.update(FINGERPRINT_ONLY_DISTRIBUTIONS)
            probe = (
                "import json, platform, sys\n"
                "from importlib import metadata\n"
                f"names = {sorted(distributions)!r}\n"
                "versions = {}\n"
                "for name in names:\n"
                "    try:\n"
                "        versions[name] = metadata.version(name)\n"
                "    except metadata.PackageNotFoundError:\n"
                "        versions[name] = 'unavailable'\n"
                "print(json.dumps({\n"
                "    'executable': sys.executable,\n"
                "    'implementation': platform.python_implementation(),\n"
                "    'python_version': platform.python_version(),\n"
                "    'platform_system': platform.system(),\n"
                "    'platform_machine': platform.machine(),\n"
                "    'packages': versions,\n"
                "}, sort_keys=True))\n"
            )
            probe_env = {
                key: os.environ[key]
                for key in _SAFE_INHERITED_ENV_KEYS
                if os.environ.get(key)
            }
            probe_env["PYTHONNOUSERSITE"] = "1"
            result = subprocess.run(  # noqa: S603 - configured argv, no shell
                [self.python_executable, "-c", probe],
                capture_output=True,
                text=True,
                timeout=30.0,
                env=probe_env,
                encoding="utf-8",
                errors="replace",
            )
            if result.returncode != 0:
                raise RuntimeError(
                    "CodeRunner interpreter authority probe failed: "
                    f"{result.stderr.strip() or self.python_executable}"
                )
            try:
                interpreter = json.loads(result.stdout)
                if not isinstance(interpreter, dict):
                    raise TypeError("interpreter authority is not an object")
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    "CodeRunner interpreter authority probe returned invalid JSON"
                ) from exc
            extra_env_identity: dict[str, object] = {}
            for key, value in sorted(self.extra_env.items()):
                extra_env_identity[key] = _path_bound_authority_value(value)
            python_binary = Path(self.python_executable).resolve(strict=True)
            payload = {
                "schema": "easyicu.code_runner_authority/1",
                "interpreter": interpreter,
                "python_entrypoint": {
                    "configured": self.python_executable,
                    "resolved": str(python_binary),
                    "sha256": _sha256_file(python_binary),
                },
                "extra_env": extra_env_identity,
                "network_policy": self.network_policy,
                "allow_unsafe_host_fallback": self.allow_unsafe_host_fallback,
            }
            self._cached_authority_identity_sha256 = hashlib.sha256(
                _canonical_json_bytes(payload)
            ).hexdigest()
            return self._cached_authority_identity_sha256

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
            Path(__file__).resolve().parents[3],
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
        # even if execution crashes. Route through the atomic, symlink-safe
        # writer: a previous attempt's sandboxed code can leave ``analysis.py``
        # as a symlink inside the reused step dir, and a raw ``write_text`` would
        # follow it and clobber a host file outside the sandbox.
        _replace_regular_file_atomically(script_path, code.encode("utf-8"))
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
        env["PYTHONPATH"] = str(Path(__file__).resolve().parents[3])
        env["COHORT_PARQUET"] = str(self.cohort_parquet)
        cohort_rows = _parquet_row_count(self.cohort_parquet)
        if cohort_rows is not None:
            env["EASYICU_COHORT_ROWS"] = str(cohort_rows)
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
            _replace_regular_file_atomically(
                log_path,
                textwrap.dedent(
                    f"""
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
                    """
                )
                .strip()
                .encode("utf-8"),
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
            proc = _run_capturing_with_descendant_reaping(
                cmd,
                cwd=str(step_dir),
                env=env,
                timeout=self.timeout_seconds,
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
                retry_proc = _run_capturing_with_descendant_reaping(
                    retry_cmd,
                    cwd=str(step_dir),
                    env=env,
                    timeout=retry_timeout,
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
                and macos_sandbox_permission_denied(stderr)
            ):
                retry_cmd = [self.python_executable, str(script_path)]
                retry_timeout = max(
                    self.timeout_seconds - (time.monotonic() - started), 1.0
                )
                retry_proc = _run_capturing_with_descendant_reaping(
                    retry_cmd,
                    cwd=str(step_dir),
                    env=env,
                    timeout=retry_timeout,
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
                retry_proc = _run_capturing_with_descendant_reaping(
                    retry_cmd,
                    cwd=str(step_dir),
                    env=env,
                    timeout=retry_timeout,
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
                retry_proc = _run_capturing_with_descendant_reaping(
                    retry_cmd,
                    cwd=str(step_dir),
                    env=env,
                    timeout=retry_timeout,
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
                retry_proc = _run_capturing_with_descendant_reaping(
                    retry_cmd,
                    cwd=str(step_dir),
                    env=env,
                    timeout=retry_timeout,
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

        # Symlink-safe: sandboxed code may have replaced ``run.log`` with a
        # symlink during execution; the atomic writer overwrites the link with a
        # fresh regular file rather than writing through it to a host victim.
        _replace_regular_file_atomically(
            log_path,
            textwrap.dedent(
                f"""
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
                """
            )
            .strip()
            .encode("utf-8"),
        )

        artefacts = _collect_safe_output_artifacts(out_dir)
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
      read-only at ``/easyicu-run``; the current attempt's output directory is
      mounted separately at ``/easyicu-step-output``.  Keeping the writable
      bind outside the read-only bind avoids Docker Desktop teardown stalls
      caused by nested bind mounts.
    * analysis containers are explicitly removed by the host after ``wait``;
      avoiding ``docker run --rm`` prevents Docker Desktop from blocking the
      client while it releases bind mounts. ``--init`` keeps signal handling
      sane.
    * the host's ``docker`` binary must be on PATH and the image
      must already be present (``docker pull`` is opt-in via
      ``pull_image=True``).

    The image is expected to provide Python plus the agent script's
    runtime deps declared by
    :mod:`easyicu.research_agent.contracts.method_packages`. A reference
    Dockerfile ships at
    ``src/easyicu/research_agent/runner_image/Dockerfile``; build
    with::

        docker build -t easyicu-research-agent:1.0.0 \\
            -f src/easyicu/research_agent/runner_image/Dockerfile .

    Subclassing for OpenHands or any other sandbox is intentionally
    cheap: override :meth:`build_command` (which returns the argv
    list passed to ``subprocess.run``) and :meth:`prepare_step_dir`
    if your sandbox needs a different mount strategy.
    """

    DEFAULT_IMAGE = "easyicu-research-agent:1.0.0"
    manages_output_cleanup = True
    CONTAINER_RUN_ROOT = "/easyicu-run"
    CONTAINER_OUTPUT_ROOT = "/easyicu-step-output"
    CONTAINER_SCRIPT_PATH = "/easyicu-analysis.py"
    CONTAINER_COHORT_PATH = "/cohort.parquet"
    CONTAINER_INPUT_ROOT = "/easyicu-inputs"
    CONTAINER_EXTRA_ROOT = "/easyicu-extra"
    GHOST_MONITOR_GRACE_SECONDS = 2.0
    GHOST_MONITOR_INTERVAL_SECONDS = 0.25
    RUNTIME_PROVENANCE_MAX_ATTEMPTS = 2

    #: Default container resource caps. Chosen to fit an ordinary analysis
    #: step (a Cox fit, a bootstrap, a figure render) with headroom, while
    #: keeping one runaway step from taking the host — or a sibling step —
    #: down with it.
    DEFAULT_CPU_LIMIT = "4"
    DEFAULT_MEMORY_LIMIT = "8g"
    DEFAULT_PIDS_LIMIT = 256
    DEFAULT_OPEN_FILES_LIMIT = 4096

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
        pids_limit: Optional[int] = None,
        open_files_limit: Optional[int] = None,
        submission_profile_name: Optional[str] = None,
        user: Optional[str] = None,
        platform: Optional[str] = None,
    ) -> None:
        set_runtime_capability_snapshot_provider(None)
        self.workdir = Path(workdir).expanduser().resolve()
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.cohort_parquet = Path(cohort_parquet).resolve()
        _reject_docker_mount_field(str(self.workdir), label="workdir")
        _reject_docker_mount_field(str(self.cohort_parquet), label="cohort path")
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
        self.extra_mounts = self._validated_extra_mounts(extra_mounts or ())
        self.extra_env = dict(extra_env or {})
        reject_reserved_runner_env(self.extra_env, owner="DockerRunner")
        self.pull_image = bool(pull_image)
        # Resource caps are ON by default. The timeout alone does not bound
        # damage: generated code can exhaust host memory, fork until the host
        # runs out of PIDs, or open enough files to starve sibling steps well
        # inside a five-minute budget. Pass an explicit value to widen or
        # narrow a limit; pass "" / 0 to opt out of one.
        self.cpu_limit = self.DEFAULT_CPU_LIMIT if cpu_limit is None else cpu_limit
        self.memory_limit = (
            self.DEFAULT_MEMORY_LIMIT if memory_limit is None else memory_limit
        )
        self.pids_limit = (
            self.DEFAULT_PIDS_LIMIT if pids_limit is None else int(pids_limit or 0)
        )
        self.open_files_limit = (
            self.DEFAULT_OPEN_FILES_LIMIT
            if open_files_limit is None
            else int(open_files_limit or 0)
        )
        # A paper-facing profile pins the execution environment into the run's
        # authority identity. "Docker was used" is not that environment if the
        # caller could also pass ``memory_limit=""``: two runs claiming the
        # same profile would then have run under different, unrecorded
        # ceilings. Development profiles keep the opt-out.
        if is_paper_facing_profile(submission_profile_name):
            disabled = [
                name
                for name, value in (
                    ("cpu_limit", self.cpu_limit),
                    ("memory_limit", self.memory_limit),
                    ("pids_limit", self.pids_limit),
                    ("open_files_limit", self.open_files_limit),
                )
                if not value
            ]
            if disabled:
                raise ValueError(
                    "submission profile "
                    f"{submission_profile_name!r} requires every container "
                    "resource ceiling to be set; disabled: "
                    + ", ".join(sorted(disabled))
                )
        if user is not None:
            self.user = user
        elif os.name == "posix" and hasattr(os, "getuid") and hasattr(os, "getgid"):
            self.user = f"{os.getuid()}:{os.getgid()}"
        else:
            self.user = None
        self.platform = platform
        self._provenance_lock = threading.Lock()
        self._pull_lock = threading.Lock()
        self._pull_attempted = False
        self._image_identity_lock = threading.Lock()
        self._cached_image_identity: Optional[Tuple[str, Tuple[str, ...]]] = None
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

    @classmethod
    def _validated_extra_mounts(
        cls,
        mounts: Sequence[Tuple[str, str, str]],
    ) -> List[Tuple[str, str, str]]:
        """Confine caller mounts to one read-only auxiliary namespace."""

        validated: List[Tuple[str, str, str]] = []
        targets: List[PurePosixPath] = []
        extra_root = PurePosixPath(cls.CONTAINER_EXTRA_ROOT)
        for raw_source, raw_target, raw_mode in mounts:
            resolved_source = _validated_real_mount_source(
                raw_source,
                label="extra mount source",
            )
            target = PurePosixPath(str(raw_target))
            mode = str(raw_mode).strip().lower()
            _reject_docker_mount_field(str(target), label="mount target")
            if (
                not target.is_absolute()
                or target == extra_root
                or extra_root not in target.parents
                or ".." in target.parts
            ):
                raise ValueError(
                    "DockerRunner extra mount target must be below /easyicu-extra"
                )
            if mode not in {"ro", "readonly"}:
                raise ValueError("DockerRunner extra mounts must be read-only")
            if any(
                target == existing
                or target in existing.parents
                or existing in target.parents
                for existing in targets
            ):
                raise ValueError("DockerRunner extra mount targets must not overlap")
            targets.append(target)
            validated.append((str(resolved_source), str(target), "ro"))
        return validated

    @staticmethod
    def _docker_cidfile_path(attempt_id: str) -> Path:
        """Keep Docker control paths short and outside mounted run trees."""

        return Path(tempfile.gettempdir()) / f"easyicu-docker-{attempt_id}.cid"

    def _container_step_dir(self, step_id: str) -> str:
        safe_step_id = _safe_path_component(step_id, label="step_id")
        _reject_docker_mount_field(safe_step_id, label="step_id")
        return f"{self.CONTAINER_RUN_ROOT}/steps/{safe_step_id}"

    def _container_cohort_path(self) -> str:
        """Return the canonical container name for the selected cohort bytes."""

        try:
            relative = self.cohort_parquet.relative_to(self.workdir.resolve())
        except ValueError:
            return self.CONTAINER_COHORT_PATH
        return f"{self.CONTAINER_RUN_ROOT}/{relative.as_posix()}"

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
            resolved = _validated_real_mount_source(
                value,
                label="extra_env path source",
            )
            if resolved == self.cohort_parquet:
                rewritten[key] = self._container_cohort_path()
                continue
            try:
                relative = resolved.relative_to(run_root)
            except ValueError:
                target = (
                    f"{self.CONTAINER_INPUT_ROOT}/{index:03d}_"
                    f"{resolved.name or 'input'}"
                )
                _reject_docker_mount_field(target, label="extra_env path target")
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
        _reject_docker_mount_field(step_id, label="step_id")
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
        """Recreate a quiescent output mount instead of reusing its inode."""

        DockerRunner._remove_lexical_path(out_dir)
        out_dir.mkdir(parents=False)

    @staticmethod
    def _publish_step_outputs(staged_out_dir: Path, out_dir: Path) -> None:
        """Publish one quiescent attempt directory at the canonical path."""

        DockerRunner._remove_lexical_path(out_dir)
        os.replace(staged_out_dir, out_dir)
        DockerRunner._ensure_real_directory(out_dir, replace_unsafe=False)

    def build_command(
        self,
        *,
        step_id: str,
        script_path: Path,
        out_dir: Path,
        immutable_script_path: Optional[Path] = None,
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
        _safe_path_component(out_dir.name, label="output directory name")
        container_output_dir = self.CONTAINER_OUTPUT_ROOT
        cmd: List[str] = [
            self.docker_executable,
            "run",
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
            # Without an equal swap cap the memory limit is advisory: the
            # container simply swaps, and the host thrashes instead of the
            # step failing fast.
            cmd.append(f"--memory-swap={self.memory_limit}")
        if self.pids_limit:
            cmd.append(f"--pids-limit={self.pids_limit}")
        if self.open_files_limit:
            cmd.append(
                f"--ulimit=nofile={self.open_files_limit}:{self.open_files_limit}"
            )

        # Mount the complete run tree read-only.  Mount both the writable
        # attempt output and immutable script at independent container paths.
        # Docker Desktop can leave ``docker run --rm`` blocked after PID 1 has
        # exited when *any* child bind target is nested below the read-only run
        # root, even when that child is also read-only.  Generated code must use
        # EASYICU_RUN_DIR for run-root reads rather than deriving paths from
        # ``__file__``.
        cmd.extend(
            [
                "--mount",
                _docker_mount_entry(
                    str(self.workdir.resolve()),
                    self.CONTAINER_RUN_ROOT,
                    readonly=True,
                ),
                "--mount",
                _docker_mount_entry(
                    str(out_dir.resolve()),
                    container_output_dir,
                    readonly=False,
                ),
                "--mount",
                _docker_mount_entry(
                    str(self.cohort_parquet),
                    self.CONTAINER_COHORT_PATH,
                    readonly=True,
                ),
            ]
        )
        if immutable_script_path is not None:
            immutable_script_path = _validated_real_mount_source(
                immutable_script_path,
                label="immutable attempt script",
            )
            cmd.extend(
                [
                    "--mount",
                    _docker_mount_entry(
                        str(immutable_script_path),
                        self.CONTAINER_SCRIPT_PATH,
                        readonly=True,
                    ),
                ]
            )
        rewritten_extra_env, path_mounts = self._containerise_extra_env()
        for source, target, mode in [*self.extra_mounts, *path_mounts]:
            entry = _docker_mount_entry(
                source,
                target,
                readonly=bool(mode and "ro" in mode.lower()),
            )
            cmd.extend(["--mount", entry])

        # Env. The container sees absolute container paths; the host
        # path is irrelevant inside.
        container_cohort_path = self._container_cohort_path()
        env = {
            "COHORT_PARQUET": container_cohort_path,
            "COHORT_PATH": container_cohort_path,
            "EASYICU_COHORT_PATH": container_cohort_path,
            "EASYICU_COHORT_PARQUET": container_cohort_path,
            "STEP_OUT_DIR": container_output_dir,
            "STEP_OUTPUT_DIR": container_output_dir,
            "STEP_OUTPUT": container_output_dir,
            "OUT_DIR": container_output_dir,
            "OUTPUT_DIR": container_output_dir,
            "EASYICU_OUTPUT_DIR": container_output_dir,
            "EASYICU_STEP_OUT_DIR": container_output_dir,
            "EASYICU_STEP_ID": step_id,
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
        cohort_rows = _parquet_row_count(self.cohort_parquet)
        if cohort_rows is not None:
            env["EASYICU_COHORT_ROWS"] = str(cohort_rows)
        env.update(rewritten_extra_env)
        if resolved_inputs_path is not None:
            relative_manifest = resolved_inputs_path.relative_to(self.workdir.resolve())
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
                self.CONTAINER_SCRIPT_PATH,
            ]
        )
        return cmd

    def _ensure_image_ready_for_authority(self) -> None:
        """Perform the optional mutable-tag pull once, before any identity seal."""

        with self._pull_lock:
            if self._pull_attempted:
                return
            self._pull_attempted = True
            if not self.pull_image:
                return
            try:
                subprocess.run(  # noqa: S603 - argv list, no shell
                    [self.docker_executable, "pull", self.image],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=max(60.0, self.timeout_seconds),
                )
            except Exception:
                # Inspection below remains the fail-closed source of truth.
                pass

    def _inspect_image_identity(self) -> Tuple[str, Tuple[str, ...]]:
        """Resolve a mutable tag to one immutable image id without running it."""

        self._ensure_image_ready_for_authority()
        with self._image_identity_lock:
            if self._cached_image_identity is not None:
                return self._cached_image_identity
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
                repo_digests = tuple(str(x) for x in inspected.get("RepoDigests") or [])
            except Exception as exc:
                raise RuntimeError(
                    "Docker image provenance inspection returned invalid JSON"
                ) from exc
            if not image_id.startswith("sha256:"):
                raise RuntimeError(
                    "Docker image provenance is missing a sha256 image id"
                )
            self._cached_image_identity = (image_id, repo_digests)
            return self._cached_image_identity

    def _capture_runtime_provenance(self) -> Tuple[Dict[str, object], str]:
        """Inspect the exact image and capture its installed Python packages.

        The result is cached per runner so concurrent/repeated steps share one
        immutable environment snapshot.  Failure is fatal: a Docker run without
        an image identity and execution-runtime lockfile is not submission-grade.
        The short-lived probe is named and tracked just like an analysis
        container so a host-side timeout cannot strand an anonymous metadata
        probe and stall later resumes.
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
            self._retry_stale_container_cleanup("runtime-provenance")
            image_id, repo_digests = self._inspect_image_identity()
            host_source_sha256 = _python_source_tree_sha256(
                Path(__file__).resolve().parents[1]
            )

            distribution_script = (
                "import hashlib\n"
                "import os\n"
                "import sys\n"
                "from pathlib import Path\n"
                "import easyicu.research_agent as research_agent\n"
                "from importlib.metadata import distributions\n"
                "root = Path(research_agent.__file__).resolve().parent\n"
                "digest = hashlib.sha256()\n"
                "for path in sorted(root.rglob('*.py')):\n"
                "    relative = path.relative_to(root).as_posix().encode('utf-8')\n"
                "    digest.update(len(relative).to_bytes(8, 'big'))\n"
                "    digest.update(relative)\n"
                "    payload = path.read_bytes()\n"
                "    digest.update(len(payload).to_bytes(8, 'big'))\n"
                "    digest.update(payload)\n"
                f"expected = {host_source_sha256!r}\n"
                "if digest.hexdigest() != expected:\n"
                "    raise RuntimeError(\n"
                "        'EasyICU research-agent source mismatch: ' \n"
                "        f'expected {expected}, observed {digest.hexdigest()}'\n"
                "    )\n"
                "rows = {}\n"
                "for dist in distributions():\n"
                "    name = str(dist.metadata.get('Name') or '').strip()\n"
                "    version = str(dist.version or '').strip()\n"
                "    if name and version:\n"
                "        rows[name.casefold()] = f'{name}=={version}'\n"
                "sys.stdout.write('\\n'.join(rows[key] for key in sorted(rows)) + '\\n')\n"
                "sys.stdout.flush()\n"
                "# This read-only metadata probe has no cleanup contract inside the\n"
                "# container.  Exit directly after flushing so third-party atexit\n"
                "# handlers cannot strand an otherwise completed probe.\n"
                "os._exit(0)\n"
            )
            capture_proc: Optional[subprocess.CompletedProcess[str]] = None
            for attempt_index in range(self.RUNTIME_PROVENANCE_MAX_ATTEMPTS):
                attempt_id = uuid.uuid4().hex
                cidfile = self._docker_cidfile_path(attempt_id)
                sentinel = self.workdir / (
                    f".docker-runtime-provenance-{attempt_id}.sentinel"
                )
                container_name = f"easyicu-ra-{attempt_id}"
                self._write_regular_file(sentinel, f"name:{container_name}\n")
                capture_cmd = [
                    self.docker_executable,
                    "run",
                    f"--cidfile={cidfile}",
                    f"--name={container_name}",
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
                    "-c",
                    distribution_script,
                ]
                ghost_monitor = self._start_ghost_container_monitor(
                    cidfile=cidfile,
                    fallback_name=container_name,
                )
                try:
                    capture_proc = subprocess.run(  # noqa: S603 - argv list, no shell
                        capture_cmd,
                        capture_output=True,
                        text=True,
                        timeout=max(15.0, min(self.timeout_seconds, 60.0)),
                        encoding="utf-8",
                        errors="replace",
                    )
                except subprocess.TimeoutExpired as exc:
                    container_ref = self._required_container_reference(
                        cidfile,
                        fallback_name=container_name,
                    )
                    teardown_confirmed, cleanup_note = self._teardown_container(
                        container_ref
                    )
                    if teardown_confirmed:
                        sentinel.unlink(missing_ok=True)
                        cidfile.unlink(missing_ok=True)
                    if (
                        teardown_confirmed
                        and attempt_index + 1 < self.RUNTIME_PROVENANCE_MAX_ATTEMPTS
                    ):
                        continue
                    raise RuntimeError(
                        "Docker execution-runtime dependency capture timed out "
                        f"after {attempt_index + 1} attempt(s). " + cleanup_note.strip()
                    ) from exc
                finally:
                    if ghost_monitor is not None:
                        monitor_stop, monitor_thread = ghost_monitor
                        monitor_stop.set()
                        monitor_thread.join(timeout=1.0)
                container_ref = self._required_container_reference(
                    cidfile,
                    fallback_name=container_name,
                )
                teardown_confirmed, cleanup_note = self._teardown_container(
                    container_ref
                )
                if not teardown_confirmed:
                    raise RuntimeError(
                        "Docker execution-runtime dependency capture completed, but "
                        "container teardown could not be confirmed. "
                        + cleanup_note.strip()
                    )
                sentinel.unlink(missing_ok=True)
                cidfile.unlink(missing_ok=True)
                break
            if capture_proc is None:  # pragma: no cover - loop contract guard
                raise RuntimeError(
                    "Docker execution-runtime dependency capture produced no result"
                )
            requirements = capture_proc.stdout.strip()
            if capture_proc.returncode != 0 or not requirements:
                raise RuntimeError(
                    "Docker execution-runtime dependency capture failed: "
                    f"{capture_proc.stderr.strip() or 'empty metadata output'}"
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
                f"# research_agent_source_sha256={host_source_sha256}\n"
                "# capture_method=importlib.metadata.distributions\n"
                "# generated_by=easyicu.research_agent.execution.runner.DockerRunner\n"
                f"{requirements}\n"
            )
            provenance: Dict[str, object] = {
                "runtime": "docker",
                "image_reference": self.image,
                "image_id": image_id,
                "repo_digests": list(repo_digests),
                "network": self.network,
                "dependency_capture_method": "importlib.metadata.distributions",
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

    def validate_runtime_capabilities(self) -> Tuple[str, ...]:
        """Validate the immutable image and publish its package allow-list."""

        snapshot = tuple(self._method_capability_snapshot())
        frozen_snapshot = tuple(snapshot)
        set_runtime_capability_snapshot_provider(lambda: frozen_snapshot)
        return frozen_snapshot

    def export_validated_runtime_bundle(self) -> Dict[str, object]:
        """Export the preflighted immutable-image receipt for runner rebuilds.

        A pipeline may need to rebuild its runner after the Agent materializes
        a locked analysis cohort.  Re-running the distribution metadata probe
        at that point is redundant.  The bundle is safe to reuse only for the
        exact immutable image id and runner network policy;
        :meth:`adopt_validated_runtime_bundle` rechecks those bindings.
        """

        provenance, requirements = self._capture_runtime_provenance()
        return {
            "schema": "easyicu.docker_runtime_preflight/2",
            "provenance": dict(provenance),
            "requirements": requirements,
        }

    def adopt_validated_runtime_bundle(self, bundle: Mapping[str, object]) -> None:
        """Reuse one digest-bound preflight receipt without another probe."""

        if (
            set(bundle) != {"schema", "provenance", "requirements"}
            or bundle.get("schema") != "easyicu.docker_runtime_preflight/2"
        ):
            raise RuntimeError("Docker runtime preflight bundle schema mismatch")
        raw_provenance = bundle.get("provenance")
        requirements = bundle.get("requirements")
        if not isinstance(raw_provenance, Mapping) or not isinstance(requirements, str):
            raise RuntimeError("Docker runtime preflight bundle is incomplete")
        provenance = dict(raw_provenance)
        if set(provenance) != {
            "runtime",
            "image_reference",
            "image_id",
            "repo_digests",
            "network",
            "dependency_capture_method",
            "requirements_sha256",
            "method_capabilities",
        }:
            raise RuntimeError("Docker runtime preflight provenance schema mismatch")
        image_id, repo_digests = self._inspect_image_identity()
        expected_requirements_sha = hashlib.sha256(
            requirements.encode("utf-8")
        ).hexdigest()
        if (
            provenance.get("runtime") != "docker"
            or provenance.get("image_reference") != self.image
            or provenance.get("image_id") != image_id
            or provenance.get("repo_digests") != list(repo_digests)
            or provenance.get("network") != self.network
            or provenance.get("dependency_capture_method")
            != "importlib.metadata.distributions"
            or provenance.get("requirements_sha256") != expected_requirements_sha
        ):
            raise RuntimeError(
                "Docker runtime changed after preflight; refusing cached receipt"
            )
        capabilities = provenance.get("method_capabilities")
        if (
            not isinstance(capabilities, list)
            or not capabilities
            or not all(isinstance(name, str) and name for name in capabilities)
        ):
            raise RuntimeError(
                "Docker runtime preflight lacks verified method capabilities"
            )
        with self._provenance_lock:
            self._cached_runtime_provenance = dict(provenance)
            self._cached_runtime_requirements = requirements
        frozen_snapshot = tuple(capabilities)
        set_runtime_capability_snapshot_provider(lambda: frozen_snapshot)

    def runtime_capability_report(self) -> Dict[str, object]:
        """Return the verified, digest-bound image capability report."""

        provenance, _requirements = self._capture_runtime_provenance()
        return dict(provenance)

    @property
    def authority_identity_sha256(self) -> str:
        """Bind replay cheaply to the immutable image and runner policy."""

        image_id, _repo_digests = self._inspect_image_identity()
        extra_mount_identity = [
            {
                "source": _path_bound_authority_value(source),
                "destination": destination,
                "mode": mode,
            }
            for source, destination, mode in self.extra_mounts
        ]
        payload = {
            "schema": "easyicu.docker_runner_authority/1",
            "image_id": image_id,
            "network": self.network,
            "extra_mounts": extra_mount_identity,
            "extra_env": {
                key: _path_bound_authority_value(value)
                for key, value in sorted(self.extra_env.items())
            },
            "cpu_limit": self.cpu_limit,
            "memory_limit": self.memory_limit,
            "pids_limit": self.pids_limit,
            "open_files_limit": self.open_files_limit,
            "user": self.user,
            "platform": self.platform,
        }
        return hashlib.sha256(
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()

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

    @staticmethod
    def _required_container_reference(
        cidfile: Path,
        *,
        fallback_name: str,
    ) -> str:
        """Resolve the container reference that teardown is about to act on.

        With a non-empty ``fallback_name`` this cannot legitimately be None;
        raise instead of asserting so the invariant survives ``python -O`` and
        produces an actionable error if the naming contract ever changes.
        """

        container_ref = DockerRunner._container_reference(
            cidfile,
            fallback_name=fallback_name,
        )
        if container_ref is None:
            raise RuntimeError(
                "Docker container reference could not be resolved for teardown "
                f"(cidfile={cidfile.name!r}, fallback_name={fallback_name!r})."
            )
        return container_ref

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

        cleanup_notes: List[str] = []
        stop_proc = _control(("stop", "--timeout=5", container_ref), timeout=15.0)
        if stop_proc is None or stop_proc.returncode != 0:
            cleanup_notes.append("stop failed")
            kill_proc = _control(("kill", container_ref), timeout=10.0)
            if kill_proc is None or kill_proc.returncode != 0:
                cleanup_notes.append("kill failed")
        wait_proc = _control(("wait", container_ref), timeout=10.0)
        if wait_proc is None or wait_proc.returncode != 0:
            cleanup_notes.append("wait failed")

        # Analysis containers deliberately omit ``--rm`` so the synchronous
        # docker client returns as soon as PID 1 exits instead of waiting for
        # Docker Desktop bind-mount teardown.  Removal is therefore always an
        # explicit host-owned phase after ``wait``.
        rm_proc = _control(("rm", "--force", container_ref), timeout=10.0)
        teardown_confirmed = rm_proc is not None and rm_proc.returncode == 0
        if not teardown_confirmed:
            cleanup_notes.append("rm failed")

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

    def _confirm_successful_container_teardown(
        self,
        container_ref: str,
    ) -> Tuple[bool, str]:
        """Confirm that the completed analysis container has been removed.

        A zero process return code proves that the generated program exited.
        It does not prove that its bind mounts are quiescent, so an analysis
        container that remains visible is explicitly waited and removed before
        the host reuses or scans the output directory.
        """

        try:
            inspect_proc = subprocess.run(  # noqa: S603 - argv list, no shell
                [
                    self.docker_executable,
                    "container",
                    "inspect",
                    container_ref,
                ],
                capture_output=True,
                text=True,
                timeout=10.0,
                encoding="utf-8",
                errors="replace",
            )
        except (OSError, subprocess.TimeoutExpired):
            inspect_proc = None
        if inspect_proc is not None:
            inspect_error = inspect_proc.stderr.lower()
            if inspect_proc.returncode != 0 and (
                "no such object" in inspect_error
                or "no such container" in inspect_error
            ):
                return True, (
                    "[DockerRunner] successful container removal confirmed "
                    "before output collection\n"
                )
        return self._teardown_container(container_ref)

    def _start_ghost_container_monitor(
        self,
        *,
        cidfile: Path,
        fallback_name: str,
    ) -> Optional[Tuple[threading.Event, threading.Thread]]:
        """Release a Docker Desktop container whose PID 1 already vanished.

        On macOS Docker Desktop can keep the attached ``docker run`` client
        blocked while ``container inspect`` still says running, even though
        ``docker top`` reports no process.  Two empty process snapshots after a
        startup grace period prove there is no generated process left to kill.
        Force-removing that ghost releases the attached client, which still
        returns the generated process's original exit status. This monitor is
        deliberately limited to macOS and requires two processless snapshots;
        ordinary running containers and non-Docker-Desktop hosts are untouched.
        """

        if sys.platform != "darwin":
            return None

        stop = threading.Event()

        def monitor() -> None:
            if stop.wait(self.GHOST_MONITOR_GRACE_SECONDS):
                return
            empty_snapshots = 0
            while not stop.is_set():
                container_ref = self._container_reference(
                    cidfile,
                    fallback_name=fallback_name,
                )
                if container_ref is None:
                    if stop.wait(self.GHOST_MONITOR_INTERVAL_SECONDS):
                        return
                    continue
                try:
                    top_proc = subprocess.run(  # noqa: S603 - fixed Docker argv
                        [self.docker_executable, "top", container_ref, "-eo", "pid"],
                        capture_output=True,
                        text=True,
                        timeout=5.0,
                        encoding="utf-8",
                        errors="replace",
                    )
                except (OSError, subprocess.TimeoutExpired):
                    top_proc = None
                if top_proc is not None and top_proc.returncode == 0:
                    process_rows = [
                        line
                        for line in top_proc.stdout.splitlines()[1:]
                        if line.strip()
                    ]
                    empty_snapshots = 0 if process_rows else empty_snapshots + 1
                    if empty_snapshots >= 2:
                        try:
                            subprocess.run(  # noqa: S603 - fixed Docker argv
                                [
                                    self.docker_executable,
                                    "rm",
                                    "--force",
                                    container_ref,
                                ],
                                capture_output=True,
                                text=True,
                                timeout=10.0,
                                encoding="utf-8",
                                errors="replace",
                            )
                        except (OSError, subprocess.TimeoutExpired):
                            pass
                        return
                else:
                    empty_snapshots = 0
                if stop.wait(self.GHOST_MONITOR_INTERVAL_SECONDS):
                    return

        thread = threading.Thread(
            target=monitor,
            name=f"easyicu-docker-ghost-{fallback_name}",
            daemon=True,
        )
        thread.start()
        return stop, thread

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
            prefix = f".docker-{step_id}-"
            attempt_id = sentinel.name[len(prefix) : -len(".sentinel")]
            self._remove_lexical_path(
                self.workdir / "steps" / step_id / f".outputs-{attempt_id}"
            )
            sentinel.unlink(missing_ok=True)
            for suffix in (".cid", ".analysis.py", ".run.log"):
                sentinel.with_suffix(suffix).unlink(missing_ok=True)
            self._docker_cidfile_path(attempt_id).unlink(missing_ok=True)

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

        runtime_provenance, runtime_requirements = self._capture_runtime_provenance()

        attempt_id = uuid.uuid4().hex
        staged_out_dir = step_dir / f".outputs-{attempt_id}"
        self._clear_step_outputs(staged_out_dir)
        # Docker Desktop can retain a stale view of a repeatedly replaced file
        # inside the long-lived run-root bind mount.  Execute an immutable,
        # attempt-owned copy overlaid at the canonical container path; publish
        # the same bytes back to ``steps/<id>/analysis.py`` only after teardown.
        cidfile = self._docker_cidfile_path(attempt_id)
        sentinel = self.workdir / f".docker-{step_id}-{attempt_id}.sentinel"
        control_script_path = sentinel.with_suffix(".analysis.py")
        control_log_path = sentinel.with_suffix(".run.log")
        container_name = f"easyicu-ra-{attempt_id}"
        self._write_regular_file(sentinel, f"name:{container_name}\n")
        self._write_regular_file(control_script_path, code)
        cmd = self.build_command(
            step_id=step_id,
            script_path=script_path,
            out_dir=staged_out_dir,
            immutable_script_path=control_script_path,
            runtime_image=str(runtime_provenance["image_id"]),
            resolved_inputs_path=resolved_inputs_path,
            authority_snapshot_path=authority_snapshot_path,
            authority_snapshot_sha256=authority_snapshot_sha256,
            authority_snapshot_error=authority_snapshot_error,
        )
        # Keep the host-written cidfile outside the step's read-write mount so
        # generated code cannot replace the container id used for teardown.
        cmd.insert(2, f"--cidfile={cidfile}")
        cmd.insert(3, f"--name={container_name}")

        timed_out = False
        teardown_confirmed = False
        started = time.monotonic()
        ghost_monitor = self._start_ghost_container_monitor(
            cidfile=cidfile,
            fallback_name=container_name,
        )
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
            container_ref = self._required_container_reference(
                cidfile,
                fallback_name=container_name,
            )
            if returncode == 0:
                teardown_confirmed, cleanup_note = (
                    self._confirm_successful_container_teardown(container_ref)
                )
                stderr = _as_text(stderr) + "\n" + cleanup_note
            else:
                teardown_confirmed, cleanup_note = self._teardown_container(
                    container_ref
                )
                stderr = _as_text(stderr) + "\n" + cleanup_note
        except subprocess.TimeoutExpired as exc:
            stdout = _as_text(exc.stdout)
            stderr = _as_text(exc.stderr) + (
                f"\n[DockerRunner] timed out after {self.timeout_seconds}s\n"
            )
            container_ref = self._required_container_reference(
                cidfile,
                fallback_name=container_name,
            )
            teardown_confirmed, cleanup_note = self._teardown_container(container_ref)
            stderr += cleanup_note
            returncode = -1
            timed_out = True
        finally:
            if ghost_monitor is not None:
                monitor_stop, monitor_thread = ghost_monitor
                monitor_stop.set()
                monitor_thread.join(timeout=1.0)
        duration = time.monotonic() - started

        log_content = textwrap.dedent(
            f"""
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
                """
        ).strip()

        if teardown_confirmed:
            self._ensure_real_directory(step_dir, replace_unsafe=False)
            self._ensure_real_directory(staged_out_dir, replace_unsafe=True)
            for output_path in list(staged_out_dir.iterdir()):
                metadata = os.lstat(output_path)
                if stat.S_ISLNK(metadata.st_mode) or (
                    stat.S_ISREG(metadata.st_mode) and metadata.st_nlink != 1
                ):
                    self._remove_lexical_path(output_path)
            self._write_regular_file(script_path, code)
            safe_script_path = script_path
            safe_log_path = log_path
            self._write_regular_file(safe_log_path, log_content)
            requirements_path = staged_out_dir / "runner_requirements.lock.txt"
            self._write_regular_file(requirements_path, runtime_requirements)
            provenance_path = staged_out_dir / "runner_provenance.json"
            self._write_regular_file(
                provenance_path,
                json.dumps(runtime_provenance, indent=2, ensure_ascii=False) + "\n",
            )
            self._publish_step_outputs(staged_out_dir, out_dir)
            artefacts = _collect_safe_output_artifacts(out_dir)
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
        docker_executable or os.environ.get("EASYICU_DOCKER_EXECUTABLE") or "docker"
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
            docker_detail = f"Docker image {runtime_image!r} is not ready" + (
                f": {detail[:240]}" if detail else ""
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
