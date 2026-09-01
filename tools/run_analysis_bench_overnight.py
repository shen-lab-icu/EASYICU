"""Sequential overnight runner for the 10-task Analysis Bench.

This wrapper is designed for free-tier or rate-limited LLM providers.
Instead of asking one long benchmark invocation to survive the entire
night, it runs each analysis item sequentially, retries failed items,
and then performs a final reuse-only aggregation pass.

Typical usage::

    python tools/run_analysis_bench_overnight.py
    python tools/run_analysis_bench_overnight.py --models openai/gpt-oss-120b:free
    python tools/run_analysis_bench_overnight.py --provider mock

The wrapper stores:

* ``overnight_plan.json`` — run configuration
* ``overnight_progress.json`` — per-item attempt history
* ``overnight_runner.log`` — append-only text log
* per-model ``bench_results.json`` / ``bench_results.md``
* optional multi-model ``bench_model_matrix.json`` / ``bench_model_matrix.md``
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence


def _bootstrap_imports() -> Path:
    here = Path(__file__).resolve().parent
    repo_root = here.parent
    src_path = repo_root / "src"
    if src_path.exists() and str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    return repo_root


def _slugify_model(model: str) -> str:
    safe = []
    for ch in model.strip():
        safe.append(ch if ch.isalnum() or ch in "._-" else "_")
    slug = "".join(safe).strip("._-")
    return slug or "model"


@dataclass
class AttemptRecord:
    item: str
    model: str
    attempt: int
    started_at: str
    completed_at: str | None = None
    return_code: int | None = None
    status: str = "running"
    elapsed_seconds: float | None = None


@dataclass
class ProgressState:
    generated_at: str
    out_root: str
    provider: str
    models: List[str]
    items: List[str]
    request_timeout: float
    max_retries: int
    sleep_seconds: int
    aggregate_after_each_model: bool
    attempts: List[AttemptRecord] = field(default_factory=list)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _append_log(log_path: Path, message: str) -> None:
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(message.rstrip() + "\n")


def _progress_payload(state: ProgressState) -> Dict[str, Any]:
    return {
        "generated_at": state.generated_at,
        "out_root": state.out_root,
        "provider": state.provider,
        "models": state.models,
        "items": state.items,
        "request_timeout": state.request_timeout,
        "max_retries": state.max_retries,
        "sleep_seconds": state.sleep_seconds,
        "aggregate_after_each_model": state.aggregate_after_each_model,
        "attempts": [asdict(a) for a in state.attempts],
    }


def _run_command(
    *, cmd: Sequence[str], cwd: Path, env: Dict[str, str], log_path: Path
) -> int:
    _append_log(log_path, "")
    _append_log(log_path, f"$ {' '.join(cmd)}")
    started = time.monotonic()
    with log_path.open("a", encoding="utf-8") as fh:
        proc = subprocess.Popen(
            list(cmd),
            cwd=str(cwd),
            env=env,
            stdout=fh,
            stderr=subprocess.STDOUT,
            text=True,
        )
        rc = proc.wait()
    elapsed = round(time.monotonic() - started, 2)
    _append_log(log_path, f"[exit={rc} elapsed_seconds={elapsed}]")
    return rc


def _item_command(
    *,
    python_bin: str,
    runner_path: Path,
    provider: str,
    model: str,
    item: str,
    request_timeout: float,
    out_root: Path,
    seed: int,
) -> List[str]:
    cmd = [
        python_bin,
        "-u",
        str(runner_path),
        "--bench-kind",
        "analysis",
        "--provider",
        provider,
        "--items",
        item,
        "--request-timeout",
        str(request_timeout),
        "--reuse-existing",
        "--out-root",
        str(out_root),
        "--seed",
        str(seed),
    ]
    if provider != "mock":
        cmd.extend(["--model", model])
    return cmd


def _aggregate_command(
    *,
    python_bin: str,
    runner_path: Path,
    provider: str,
    model: str,
    items: Sequence[str],
    request_timeout: float,
    out_root: Path,
    seed: int,
) -> List[str]:
    cmd = [
        python_bin,
        "-u",
        str(runner_path),
        "--bench-kind",
        "analysis",
        "--provider",
        provider,
        "--items",
        *items,
        "--request-timeout",
        str(request_timeout),
        "--reuse-existing",
        "--out-root",
        str(out_root),
        "--seed",
        str(seed),
    ]
    if provider != "mock":
        cmd.extend(["--model", model])
    return cmd


def _matrix_command(
    *,
    python_bin: str,
    runner_path: Path,
    provider: str,
    models: Sequence[str],
    items: Sequence[str],
    request_timeout: float,
    out_root: Path,
    seed: int,
) -> List[str]:
    cmd = [
        python_bin,
        "-u",
        str(runner_path),
        "--bench-kind",
        "analysis",
        "--provider",
        provider,
        "--items",
        *items,
        "--request-timeout",
        str(request_timeout),
        "--reuse-existing",
        "--out-root",
        str(out_root),
        "--seed",
        str(seed),
    ]
    if provider != "mock":
        cmd.extend(["--models", *models])
    return cmd


def main() -> int:
    repo_root = _bootstrap_imports()
    from easyicu.research_agent.providers import (
        SUPPORTED_CLI_ACCOUNT_NAMES,
        SUPPORTED_PROVIDER_NAMES,
        cli_account_profile,
        provider_profile,
    )
    from tests.support.benchmark_cases import ANALYSIS_BENCH_ITEMS  # type: ignore

    all_items = [it.key for it in ANALYSIS_BENCH_ITEMS]

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--items",
        nargs="+",
        default=None,
        help="Subset of analysis bench items (default: all 10).",
    )
    parser.add_argument(
        "--provider",
        choices=["mock", *SUPPORTED_CLI_ACCOUNT_NAMES, *SUPPORTED_PROVIDER_NAMES],
        default="openrouter",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Single model to use when --models is not set. Account CLIs use "
            "their logged-in default when omitted."
        ),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Optional multiple models to run sequentially.",
    )
    parser.add_argument("--request-timeout", type=float, default=300.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--sleep-seconds", type=int, default=45)
    parser.add_argument(
        "--aggregate-after-each-model",
        action="store_true",
        help="Write per-model bench_results after each model finishes.",
    )
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument(
        "--out-root",
        default=None,
        help="Default: research_output/bench/analysis_overnight_<UTC timestamp>",
    )
    args = parser.parse_args()

    if not args.model and not args.models:
        if args.provider == "mock":
            args.model = "mock"
        elif args.provider in SUPPORTED_CLI_ACCOUNT_NAMES:
            account = cli_account_profile(args.provider)
            _source, configured_model = (
                account.model(os.environ) if account is not None else (None, "")
            )
            args.model = configured_model or "cli-default"
        else:
            profile = provider_profile(args.provider)
            _source, configured_model = (
                profile.model(os.environ) if profile is not None else (None, "")
            )
            args.model = configured_model or os.environ.get(
                "EASYICU_HOSTED_DEFAULT_MODEL",
                "",
            )
            if not args.model and args.provider == "openrouter":
                args.model = "openai/gpt-oss-120b:free"
            if not args.model:
                parser.error(f"--model is required for --provider {args.provider}")

    items = list(args.items or all_items)
    unknown = sorted(set(items) - set(all_items))
    if unknown:
        raise SystemExit(f"Unknown analysis bench items: {unknown}")

    if args.provider == "mock":
        models = ["mock"]
    else:
        models = list(args.models or [args.model])

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_root = Path(
        args.out_root
        or (repo_root / "research_output" / "bench" / f"analysis_overnight_{timestamp}")
    ).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    log_path = out_root / "overnight_runner.log"
    plan_path = out_root / "overnight_plan.json"
    progress_path = out_root / "overnight_progress.json"
    runner_path = (repo_root / "tools" / "run_research_agent_bench.py").resolve()

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    state = ProgressState(
        generated_at=_utc_now(),
        out_root=str(out_root),
        provider=args.provider,
        models=models,
        items=items,
        request_timeout=float(args.request_timeout),
        max_retries=int(args.max_retries),
        sleep_seconds=int(args.sleep_seconds),
        aggregate_after_each_model=bool(args.aggregate_after_each_model),
    )
    _write_json(plan_path, _progress_payload(state))
    _write_json(progress_path, _progress_payload(state))

    _append_log(log_path, f"Overnight analysis bench started at {_utc_now()}")
    _append_log(log_path, f"Provider={args.provider}")
    _append_log(log_path, f"Models={models}")
    _append_log(log_path, f"Items={items}")
    _append_log(log_path, f"Out root={out_root}")

    for model_idx, model in enumerate(models, start=1):
        model_root = (
            out_root if len(models) == 1 else (out_root / _slugify_model(model))
        )
        model_root.mkdir(parents=True, exist_ok=True)
        _append_log(log_path, "")
        _append_log(log_path, f"=== Model {model_idx}/{len(models)}: {model} ===")
        for item_idx, item in enumerate(items, start=1):
            success = False
            for attempt in range(1, args.max_retries + 1):
                record = AttemptRecord(
                    item=item,
                    model=model,
                    attempt=attempt,
                    started_at=_utc_now(),
                )
                state.attempts.append(record)
                _write_json(progress_path, _progress_payload(state))
                _append_log(
                    log_path,
                    f"--- Item {item_idx}/{len(items)}: {item} | attempt {attempt}/{args.max_retries} ---",
                )
                started = time.monotonic()
                rc = _run_command(
                    cmd=_item_command(
                        python_bin=args.python_bin,
                        runner_path=runner_path,
                        provider=args.provider,
                        model=model,
                        item=item,
                        request_timeout=float(args.request_timeout),
                        out_root=model_root,
                        seed=int(args.seed),
                    ),
                    cwd=repo_root,
                    env=env,
                    log_path=log_path,
                )
                record.completed_at = _utc_now()
                record.return_code = rc
                record.elapsed_seconds = round(time.monotonic() - started, 2)
                record.status = "ok" if rc == 0 else "failed"
                _write_json(progress_path, _progress_payload(state))
                if rc == 0:
                    success = True
                    break
                if attempt < args.max_retries:
                    _append_log(
                        log_path, f"Retrying {item} after {args.sleep_seconds}s"
                    )
                    time.sleep(args.sleep_seconds)
            if not success:
                _append_log(
                    log_path, f"Item failed after {args.max_retries} attempts: {item}"
                )

        if args.aggregate_after_each_model:
            _append_log(log_path, f"Aggregating completed items for model: {model}")
            _run_command(
                cmd=_aggregate_command(
                    python_bin=args.python_bin,
                    runner_path=runner_path,
                    provider=args.provider,
                    model=model,
                    items=items,
                    request_timeout=float(args.request_timeout),
                    out_root=model_root,
                    seed=int(args.seed),
                ),
                cwd=repo_root,
                env=env,
                log_path=log_path,
            )

    if len(models) == 1:
        _append_log(log_path, "Running final single-model aggregation pass")
        _run_command(
            cmd=_aggregate_command(
                python_bin=args.python_bin,
                runner_path=runner_path,
                provider=args.provider,
                model=models[0],
                items=items,
                request_timeout=float(args.request_timeout),
                out_root=out_root,
                seed=int(args.seed),
            ),
            cwd=repo_root,
            env=env,
            log_path=log_path,
        )
    else:
        _append_log(log_path, "Running final multi-model matrix aggregation pass")
        _run_command(
            cmd=_matrix_command(
                python_bin=args.python_bin,
                runner_path=runner_path,
                provider=args.provider,
                models=models,
                items=items,
                request_timeout=float(args.request_timeout),
                out_root=out_root,
                seed=int(args.seed),
            ),
            cwd=repo_root,
            env=env,
            log_path=log_path,
        )

    _append_log(log_path, f"Overnight analysis bench finished at {_utc_now()}")
    _write_json(progress_path, _progress_payload(state))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
