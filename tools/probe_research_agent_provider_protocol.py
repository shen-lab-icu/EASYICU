#!/usr/bin/env python3
"""Run six bounded non-clinical calls against the local E1 Provider path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Mapping


def _bootstrap() -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    for path in (repo_root, repo_root / "src"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    return repo_root


def _parse_env_file(path: Path) -> dict[str, str]:
    candidate = path.expanduser()
    if candidate.is_symlink() or not candidate.is_file():
        raise SystemExit("env file must be a regular non-symlink file")
    values: dict[str, str] = {}
    for line_number, raw in enumerate(
        candidate.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise SystemExit(f"invalid env assignment at line {line_number}")
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or not key.replace("_", "").isalnum() or not key[0].isalpha():
            raise SystemExit(f"invalid env key at line {line_number}")
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        values[key] = value
    return values


def _build_clients(
    *,
    environment: Mapping[str, str],
    model: str,
) -> dict[str, object]:
    from easyicu.research_agent import OpenAIClient
    from easyicu.research_agent.providers.factory import build_provider_client

    return {
        effort: build_provider_client(
            provider="openai",
            model=model,
            request_timeout=180.0,
            title="EasyICU E1 provider protocol probe",
            client_cls=OpenAIClient,
            environment=environment,
            extra_body={"reasoning": {"effort": effort}},
            max_retries=1,
            stream_enabled=False,
            allow_environment_overrides=False,
        )
        for effort in ("low", "medium")
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, default=Path(".env.local"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model", default="gpt-5.6-luna")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    repo_root = _bootstrap()
    environment = _parse_env_file(args.env_file)
    base_url = environment.get("OPENAI_BASE_URL", "")
    clients = _build_clients(environment=environment, model=args.model)

    from benchmarks.figure2_canonical9.provider_protocol_probe import (
        PROVIDER_PROTOCOL_REPORT_FILENAME,
        run_provider_protocol_probe,
    )

    report = run_provider_protocol_probe(
        output_dir=args.out_dir,
        model=args.model,
        base_url=base_url,
        client_for_effort=lambda effort: clients[effort],
        repo_root=repo_root,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "model": report["model"],
                "call_count": report["call_count"],
                "transport_attempts": report["transport_attempts"],
                "transport_retries": report["transport_retries"],
                "truncation_finish_reason": report["truncation_finish_reason"],
                "report": str(
                    args.out_dir.expanduser().resolve()
                    / PROVIDER_PROTOCOL_REPORT_FILENAME
                ),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
