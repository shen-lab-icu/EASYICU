#!/usr/bin/env python3
"""Repair a sealed run's manuscript without invoking scientific execution."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any


_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_ROOT = _REPO_ROOT / "src"
for _path in (_SRC_ROOT, _REPO_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from easyicu.research_agent.agents.reporting import WriterAgent  # noqa: E402
from easyicu.research_agent.authority.provider_hard_stop import (  # noqa: E402
    ProviderHardStopLedger,
    ProviderHardStopLimits,
    validate_provider_transport_reservation_capacity,
)
from easyicu.research_agent.providers.cost import CostMeter, MeteredClient  # noqa: E402
from easyicu.research_agent.providers.hard_stop import HardStopClient  # noqa: E402
from easyicu.research_agent.providers.llm import build_llm_client  # noqa: E402
from easyicu.research_agent.reporting.writer_only_migration import (  # noqa: E402
    prepare_writer_only_migration,
    publish_writer_only_failure,
    publish_writer_only_result,
    repair_writer_only,
    writer_only_preflight_payload,
)
from easyicu.research_agent.reporting.manuscript_quality import (  # noqa: E402
    audit_manuscript_quality,
)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


class _RecordingWriterAgent(WriterAgent):
    """Persist aggregate-only candidate sections for failed-call diagnosis."""

    def __init__(self, *args: Any, runtime_dir: Path, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._runtime_dir = runtime_dir
        self._attempt = 0

    def _call_section(self, **kwargs: Any) -> str:
        section = super()._call_section(**kwargs)
        self._attempt += 1
        stem = f"writer_candidate_{self._attempt:02d}_{str(kwargs['section_name']).lower()}"
        path = self._runtime_dir / "writer_candidates" / f"{stem}.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(section, encoding="utf-8")
        _write_json(
            path.with_suffix(".quality.json"),
            audit_manuscript_quality(
                section,
                require_administrative_sections=False,
            ).to_dict(),
        )
        return section


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--provider", default="codex")
    parser.add_argument("--model", default="gpt-5.6-luna")
    parser.add_argument("--request-timeout", type=float, default=300.0)
    parser.add_argument("--max-calls", type=int, default=6)
    parser.add_argument("--max-total-tokens", type=int, default=1_500_000)
    parser.add_argument("--max-cost-usd", type=float, default=100.0)
    parser.add_argument("--max-seconds", type=float, default=900.0)
    parser.add_argument("--input-price-upper", type=float, default=10.0)
    parser.add_argument("--output-price-upper", type=float, default=50.0)
    return parser


def main() -> int:
    args = _parser().parse_args()
    prepared = prepare_writer_only_migration(args.source_run)
    output = args.output_dir.expanduser().resolve()
    if args.dry_run:
        payload = writer_only_preflight_payload(prepared)
        _write_json(output / "writer_only_preflight.json", payload)
        print(json.dumps(payload, ensure_ascii=False))
        return 0

    if str(os.environ.get("EASYICU_ALLOW_EXTERNAL_LLM") or "").lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }:
        raise SystemExit(
            "Real Writer transport requires EASYICU_ALLOW_EXTERNAL_LLM=1."
        )

    limits = ProviderHardStopLimits(
        max_provider_attempts_per_run=args.max_calls,
        max_provider_attempts_per_batch=args.max_calls,
        max_total_tokens_per_run=args.max_total_tokens,
        max_total_tokens_per_batch=args.max_total_tokens,
        max_estimated_cost_usd_per_batch=args.max_cost_usd,
        max_wall_clock_seconds_per_task=args.max_seconds,
        input_cost_usd_per_million_tokens=args.input_price_upper,
        output_cost_usd_per_million_tokens=args.output_price_upper,
    )
    validate_provider_transport_reservation_capacity(limits)
    runtime = output / "runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    ledger_path = runtime / "provider_hard_stop.json"
    task_id = prepared.source_run_dir.parent.parent.name
    ledger = ProviderHardStopLedger(
        path=ledger_path,
        task_ids=[task_id],
        limits=limits,
        batch_id=f"writer-only:{task_id}",
    )
    task = ledger.start_task(task_id)
    meter = CostMeter(runtime_dir=runtime)
    selection = build_llm_client(
        prefer=args.provider,
        model=args.model,
        allow_mock=False,
        ladder=[args.provider],
        request_timeout=task.cap_timeout(args.request_timeout),
        environment=os.environ,
    )
    hard_stopped = HardStopClient(selection.client, role="writer", task=task)
    metered = MeteredClient(
        hard_stopped,
        role="writer",
        meter=meter,
        model_override=args.model,
    )
    writer = _RecordingWriterAgent(
        metered,
        language="en",
        nature_writing_enabled=True,
        runtime_dir=runtime,
    )
    try:
        result = repair_writer_only(prepared, writer=writer)
    except BaseException as exc:
        task.finish(error=f"{type(exc).__name__}: {exc}")
        summary = meter.summary(
            hard_stop_accounting=task.accounting_summary()
        )
        publish_writer_only_failure(
            prepared,
            output_dir=output,
            error=exc,
            provider=args.provider,
            model=args.model,
            provider_summary=summary,
            provider_ledger=str(ledger_path),
        )
        raise
    task.finish(
        score={
            "quality_status": result.quality_audit.status,
            "literature_status": result.literature_audit.status,
            "publication_authorized": False,
        }
    )
    summary = meter.summary(hard_stop_accounting=task.accounting_summary())
    receipt = publish_writer_only_result(
        prepared,
        result,
        output_dir=output,
        provider=args.provider,
        model=args.model,
        provider_summary=summary,
        provider_ledger=str(ledger_path),
    )
    print(json.dumps(receipt, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
