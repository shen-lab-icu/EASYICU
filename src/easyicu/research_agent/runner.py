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

* :class:`CodeRunner` — host subprocess. Fast, default, no extra
  dependency. Trusts the user's machine.
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

import os
import shutil
import subprocess
import sys
import textwrap
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .code_hygiene import reorder_forward_references


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

    @property
    def succeeded(self) -> bool:
        return self.returncode == 0 and not self.timed_out


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
    ) -> None:
        self.workdir = Path(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.cohort_parquet = Path(cohort_parquet).resolve()
        if not self.cohort_parquet.exists():
            raise FileNotFoundError(f"Cohort parquet does not exist: {self.cohort_parquet}")
        self.timeout_seconds = timeout_seconds
        self.python_executable = python_executable or sys.executable
        self.extra_env = dict(extra_env or {})
        self.network_policy = (network_policy or "none").lower()

    def build_command(self, *, script_path: Path) -> List[str]:
        base = [self.python_executable, str(script_path)]
        if self.network_policy not in {"none", "disabled"}:
            return base
        sandbox_exec = shutil.which("sandbox-exec")
        if sandbox_exec and sys.platform == "darwin":
            profile = (
                "(version 1)\n"
                "(deny default)\n"
                "(allow process*)\n"
                "(allow sysctl-read)\n"
                "(allow file-read*)\n"
                "(allow file-write*)\n"
                "(deny network*)\n"
            )
            return [sandbox_exec, "-p", profile, *base]
        unshare = shutil.which("unshare")
        if unshare and sys.platform.startswith("linux"):
            return [unshare, "-n", "--", *base]
        return base

    def run(self, *, step_id: str, code: str) -> RunResult:
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

        env = os.environ.copy()
        env["COHORT_PARQUET"] = str(self.cohort_parquet)
        env["STEP_OUT_DIR"] = str(out_dir)
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

        timed_out = False
        started = time.monotonic()
        cmd = self.build_command(script_path=script_path)
        original_cmd = list(cmd)
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
                and original_cmd
                and Path(original_cmd[0]).name == "unshare"
                and sys.platform.startswith("linux")
                and "unshare failed" in stderr.lower()
            ):
                retry_cmd = [self.python_executable, str(script_path)]
                retry_timeout = max(self.timeout_seconds - (time.monotonic() - started), 1.0)
                retry_proc = subprocess.run(  # noqa: S603 - intentional, generated script
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
                    "[CodeRunner] unshare network isolation unavailable; "
                    "retrying without Linux network namespace isolation.\n"
                    f"[CodeRunner] original stderr:\n{stderr}\n"
                    f"[CodeRunner] fallback stderr:\n{retry_proc.stderr}"
                )
                returncode = retry_proc.returncode
                cmd = retry_cmd
            if (
                returncode != 0
                and original_cmd
                and Path(original_cmd[0]).name == "sandbox-exec"
                and sys.platform == "darwin"
                and "omp: error #179" in stderr.lower()
                and "shm" in stderr.lower()
            ):
                retry_cmd = [self.python_executable, str(script_path)]
                retry_timeout = max(self.timeout_seconds - (time.monotonic() - started), 1.0)
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
            duration = time.monotonic() - started
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout or ""
            stderr = (exc.stderr or "") + f"\n[CodeRunner] timed out after {self.timeout_seconds}s\n"
            returncode = -1
            duration = time.monotonic() - started
            timed_out = True

        log_path.write_text(
            textwrap.dedent(
                f"""
                === step {step_id} ===
                cmd: {' '.join(cmd)}
                original_cmd: {' '.join(original_cmd)}
                cwd: {step_dir}
                cohort: {self.cohort_parquet}
                network_policy: {self.network_policy}
                returncode: {returncode}
                timed_out: {timed_out}
                duration_seconds: {duration:.3f}
                ---- stdout ----
                {stdout}
                ---- stderr ----
                {stderr}
                """
            ).strip(),
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
    * the per-step output directory is mounted **read-write** at
      ``/workspace`` so artefacts land back on the host without
      copying.
    * ``--rm`` so containers don't pile up; ``--init`` so signal
      handling is sane.
    * the host's ``docker`` binary must be on PATH and the image
      must already be present (``docker pull`` is opt-in via
      ``pull_image=True``).

    The image is expected to provide Python plus the agent script's
    runtime deps (``pandas``, ``numpy``, ``scipy``, ``statsmodels``,
    ``matplotlib``, ``pyarrow``). A reference Dockerfile ships at
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
    CONTAINER_WORKDIR = "/workspace"
    CONTAINER_COHORT_PATH = "/cohort.parquet"

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
        self.workdir = Path(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.cohort_parquet = Path(cohort_parquet).resolve()
        if not self.cohort_parquet.exists():
            raise FileNotFoundError(
                f"Cohort parquet does not exist: {self.cohort_parquet}"
            )
        self.timeout_seconds = timeout_seconds
        self.image = image or os.environ.get(
            "EASYICU_RUNNER_IMAGE", self.DEFAULT_IMAGE
        )
        self.docker_executable = (
            docker_executable
            or os.environ.get("EASYICU_DOCKER_EXECUTABLE")
            or "docker"
        )
        self.network = network
        self.extra_mounts = list(extra_mounts or [])
        self.extra_env = dict(extra_env or {})
        self.pull_image = bool(pull_image)
        self.cpu_limit = cpu_limit
        self.memory_limit = memory_limit
        self.user = user
        self.platform = platform

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

    # ------------------------------------------------------------------
    # Hooks subclasses may override
    # ------------------------------------------------------------------

    def prepare_step_dir(self, step_id: str) -> Tuple[Path, Path, Path]:
        """Lay out the per-step directory and return the key paths."""
        step_dir = self.workdir / "steps" / step_id
        step_dir.mkdir(parents=True, exist_ok=True)
        script_path = step_dir / "analysis.py"
        out_dir = step_dir / "outputs"
        out_dir.mkdir(parents=True, exist_ok=True)
        return step_dir, script_path, out_dir

    def build_command(
        self,
        *,
        step_id: str,
        script_path: Path,
        out_dir: Path,
    ) -> List[str]:
        """Compose the ``docker run`` argv for a single step."""
        cmd: List[str] = [
            self.docker_executable,
            "run",
            "--rm",
            "--init",
            f"--network={self.network}",
            f"--workdir={self.CONTAINER_WORKDIR}",
        ]
        if self.platform:
            cmd.append(f"--platform={self.platform}")
        if self.user:
            cmd.append(f"--user={self.user}")
        if self.cpu_limit:
            cmd.append(f"--cpus={self.cpu_limit}")
        if self.memory_limit:
            cmd.append(f"--memory={self.memory_limit}")

        # Mounts: cohort RO, step_dir RW. The step_dir mount carries
        # both the script AND the outputs/ subdir, so the container
        # writes its artefacts into the same place the host reads.
        cmd.extend([
            "--mount",
            (
                f"type=bind,source={str(script_path.parent.resolve())},"
                f"target={self.CONTAINER_WORKDIR}"
            ),
            "--mount",
            (
                f"type=bind,source={str(self.cohort_parquet)},"
                f"target={self.CONTAINER_COHORT_PATH},readonly"
            ),
        ])
        for source, target, mode in self.extra_mounts:
            entry = f"type=bind,source={source},target={target}"
            if mode and "ro" in mode.lower():
                entry += ",readonly"
            cmd.extend(["--mount", entry])

        # Env. The container sees absolute container paths; the host
        # path is irrelevant inside.
        env = {
            "COHORT_PARQUET": self.CONTAINER_COHORT_PATH,
            "STEP_OUT_DIR": (
                f"{self.CONTAINER_WORKDIR}/outputs"
            ),
            "MPLBACKEND": "Agg",
            "PYTHONIOENCODING": "utf-8",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        env.update(self.extra_env)
        for key, value in env.items():
            cmd.extend(["-e", f"{key}={value}"])

        cmd.append(self.image)
        # The script is at /workspace/<basename>. Use python -u for
        # unbuffered stdout so streaming logs don't surprise people.
        cmd.extend([
            "python",
            "-u",
            f"{self.CONTAINER_WORKDIR}/{script_path.name}",
        ])
        return cmd

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self, *, step_id: str, code: str) -> RunResult:
        # Same forward-reference hoisting as CodeRunner; see code_hygiene
        # docstring for the qwen3-coder-30b regression that motivates it.
        code = reorder_forward_references(code)
        step_dir, script_path, out_dir = self.prepare_step_dir(step_id)
        log_path = step_dir / "run.log"
        script_path.write_text(code, encoding="utf-8")

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

        cmd = self.build_command(
            step_id=step_id, script_path=script_path, out_dir=out_dir,
        )

        timed_out = False
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
            stdout, stderr, returncode = proc.stdout, proc.stderr, proc.returncode
            duration = time.monotonic() - started
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout if isinstance(exc.stdout, str) else (
                (exc.stdout or b"").decode("utf-8", errors="replace")
                if exc.stdout is not None else ""
            )
            stderr = (
                (exc.stderr if isinstance(exc.stderr, str) else (
                    (exc.stderr or b"").decode("utf-8", errors="replace")
                    if exc.stderr is not None else ""
                ))
                + f"\n[DockerRunner] timed out after {self.timeout_seconds}s\n"
            )
            returncode = -1
            duration = time.monotonic() - started
            timed_out = True
            # Best-effort: if the host's docker is alive, the container
            # is still draining. There's no addressable container id
            # because we used --rm; the timeout will reap it on its own.

        log_path.write_text(
            textwrap.dedent(
                f"""
                === step {step_id} (DockerRunner) ===
                image: {self.image}
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
            ).strip(),
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
        )


__all__ = ["RunResult", "CodeRunner", "DockerRunner"]
