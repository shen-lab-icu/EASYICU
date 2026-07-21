#!/usr/bin/env python3
"""Run a bounded, blinded Planner-only know-how A/B evaluation.

This tool deliberately stops before Coder, sandbox execution, evidence
registration, and manuscript generation.  It exists to answer one narrow
question: does reviewed research know-how improve the plan for the same
immutable research context without increasing retries or prompt cost
unacceptably?

Online use fails closed when a selected card is not clinically *and*
methodologically reviewed.  ``--allow-curated-development-card`` is an
explicit development-only escape hatch; its use is recorded in the result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


def _bootstrap_imports() -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    src = repo_root / "src"
    for value in (src, repo_root):
        if str(value) not in sys.path:
            sys.path.insert(0, str(value))
    return repo_root


REPO_ROOT = _bootstrap_imports()

from easyicu.research_agent.agents.core import PlannerAgent  # noqa: E402
from easyicu.research_agent.authority.evidence_store import EvidenceStore  # noqa: E402
from easyicu.research_agent.planning.preplan_know_how import (  # noqa: E402
    PlannerKnowHowBinding,
    prepare_preplan_know_how,
)
from easyicu.research_agent.providers import (  # noqa: E402
    ProviderConfigurationError,
    build_provider_client,
)
from easyicu.research_agent.schema import ResearchContext  # noqa: E402


SCHEMA_VERSION = "easyicu.research_know_how_planner_ab/1"
BLIND_RUBRIC = {
    "schema_version": "easyicu.research_know_how_blind_rubric/1",
    "instructions": (
        "Score plans without opening operator/arm_key.json. Use 0=absent/unsafe, "
        "1=partial, 2=complete and appropriate for the declared question/data."
    ),
    "dimensions": [
        "question_and_estimand_alignment",
        "eligibility_and_time_zero",
        "exposure_outcome_and_window_definition",
        "missingness_and_source_status",
        "data_answerability_and_stop_conditions",
        "method_and_uncertainty",
        "robustness_and_diagnostics",
        "claim_level_evidence_discipline",
    ],
    "critical_errors": [
        "unsupported_disease_specific_exclusion",
        "causal_claim_for_descriptive_question",
        "unmeasured_or_source_missing_treated_as_zero",
        "irrelevant_protocol_card_adopted",
        "required_but_unavailable_data_silently_assumed",
    ],
}


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _git_head() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _load_env_file(path: Path | None) -> None:
    """Load a simple KEY=VALUE file without overriding the operator's env."""
    if path is None:
        return
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            raise SystemExit(f"invalid env line {path}:{line_number}")
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"'")
        if not key:
            raise SystemExit(f"empty env key at {path}:{line_number}")
        os.environ.setdefault(key, value)


def build_schedule(repeats: int, seed: int) -> list[str]:
    if repeats < 2:
        raise ValueError("repeats must be at least 2 per arm")
    rng = random.Random(seed)
    arms: list[str] = []
    for _ in range(repeats):
        first = rng.choice(("off", "on"))
        arms.extend((first, "on" if first == "off" else "off"))
    return arms


def blinded_label(*, seed: int, trial_index: int) -> str:
    digest = hashlib.sha256(f"{seed}:{trial_index}".encode("ascii")).hexdigest()[:12]
    return f"plan_{digest}"


def require_reviewed_cards(
    binding: PlannerKnowHowBinding,
    prepared: Any,
    *,
    allow_curated_development_card: bool,
) -> None:
    if not binding.selected_ids or allow_curated_development_card:
        return
    unreviewed = sorted(
        card_id
        for card_id in binding.selected_ids
        if prepared.registry.get(card_id).review_status != "clinical_reviewed"
    )
    if unreviewed:
        raise RuntimeError(
            "online Planner A/B refused unreviewed know-how cards: "
            f"{unreviewed!r}. Complete clinical+methods attestation, or use "
            "--allow-curated-development-card for a clearly labelled, "
            "non-submission development comparison."
        )


@dataclass
class CountingClient:
    """Transparent LLM wrapper recording every structured-retry call."""

    inner: Any
    max_calls: int = 2
    raw_output_dir: Path | None = None
    calls: list[dict[str, Any]] = field(default_factory=list)

    @property
    def name(self) -> str:
        return str(getattr(self.inner, "name", self.inner.__class__.__name__))

    @property
    def last_usage(self) -> Any:
        return getattr(self.inner, "last_usage", None)

    def complete(
        self,
        messages: Sequence[Any],
        *,
        max_tokens: int = 2048,
        temperature: float = 0.2,
    ) -> str:
        if len(self.calls) >= self.max_calls:
            raise RuntimeError(
                "Planner A/B per-trial provider-call budget exhausted: "
                f"{self.max_calls}"
            )
        started = time.monotonic()
        response = self.inner.complete(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        usage = getattr(self.inner, "last_usage", None)
        call_number = len(self.calls) + 1
        if self.raw_output_dir is not None:
            self.raw_output_dir.mkdir(parents=True, exist_ok=True)
            (self.raw_output_dir / f"raw_response_{call_number:02d}.txt").write_text(
                response or "",
                encoding="utf-8",
            )
        self.calls.append(
            {
                "active_wall_seconds": round(time.monotonic() - started, 6),
                "raw_chars": len(response or ""),
                "raw_sha256": _sha256_bytes((response or "").encode("utf-8")),
                "raw_head": (response or "").strip().replace("\n", " ")[:240],
                "usage": dict(usage) if isinstance(usage, Mapping) else None,
            }
        )
        return response


def _build_client(provider: str, model: str, request_timeout: float) -> Any:
    try:
        return build_provider_client(
            provider=provider,
            model=model,
            request_timeout=request_timeout,
            title="EasyICU research know-how Planner A/B",
        )
    except ProviderConfigurationError as exc:
        raise SystemExit(str(exc)) from exc


def _prompt_payload(
    context: ResearchContext,
    binding: PlannerKnowHowBinding,
) -> dict[str, Any]:
    messages = PlannerAgent.request_messages(context, know_how_context=binding.prompt)
    metrics = PlannerAgent.request_metrics(context, know_how_context=binding.prompt)
    return {
        "metrics": metrics,
        "messages": [
            {"role": message.role, "content": message.content} for message in messages
        ],
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context-json", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--database", default="miiv")
    parser.add_argument("--provider", default="openai")
    parser.add_argument("--model", default="gpt-5.6-luna")
    parser.add_argument("--request-timeout", type=float, default=300.0)
    parser.add_argument("--max-provider-calls-per-trial", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--min-score", type=float, default=0.15)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--allow-curated-development-card", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.max_provider_calls_per_trial < 1:
        raise SystemExit("--max-provider-calls-per-trial must be at least 1")
    _load_env_file(args.env_file)
    raw_context = args.context_json.read_bytes()
    context = ResearchContext.model_validate_json(raw_context)
    out_dir = args.out_dir.resolve()
    if out_dir.exists() and any(out_dir.iterdir()):
        raise SystemExit(f"refusing non-empty output directory: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    schedule = build_schedule(args.repeats, args.seed)
    run_manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_head": _git_head(),
        "context_path": str(args.context_json.resolve()),
        "context_sha256": _sha256_bytes(raw_context),
        "provider": args.provider,
        "model": args.model,
        "database": args.database,
        "repeats_per_arm": args.repeats,
        "randomization_seed": args.seed,
        "max_provider_calls_per_trial": args.max_provider_calls_per_trial,
        "prepare_only": bool(args.prepare_only),
        "allow_curated_development_card": bool(
            args.allow_curated_development_card
        ),
        "trials": [],
    }
    private_key: dict[str, str] = {}
    _write_json(out_dir / "blinded" / "rubric.json", BLIND_RUBRIC)

    for trial_index, arm in enumerate(schedule, 1):
        label = blinded_label(seed=args.seed, trial_index=trial_index)
        trial_dir = out_dir / "operator" / f"trial_{trial_index:02d}_{arm}"
        trial_dir.mkdir(parents=True, exist_ok=False)
        binding = PlannerKnowHowBinding()
        prepared = None
        if arm == "on":
            prepared = prepare_preplan_know_how(
                context=context,
                run_dir=trial_dir,
                evidence=EvidenceStore(trial_dir),
                database=args.database,
                top_k=args.top_k,
                min_score=args.min_score,
            )
            binding = PlannerKnowHowBinding.from_prepared(prepared)
            if not args.prepare_only:
                require_reviewed_cards(
                    binding,
                    prepared,
                    allow_curated_development_card=(
                        args.allow_curated_development_card
                    ),
                )

        prompt_payload = _prompt_payload(context, binding)
        _write_json(trial_dir / "planner_request.json", prompt_payload)
        private_key[label] = arm
        trial_record: dict[str, Any] = {
            "trial_index": trial_index,
            "blinded_label": label,
            "arm": arm,
            "selected_card_ids": list(binding.selected_ids),
            "prompt_metrics": prompt_payload["metrics"],
            "provider_calls": 0,
        }
        if not args.prepare_only:
            client = CountingClient(
                _build_client(args.provider, args.model, args.request_timeout),
                max_calls=args.max_provider_calls_per_trial,
                raw_output_dir=trial_dir,
            )
            planner = PlannerAgent(client)
            started = time.monotonic()
            try:
                plan = planner.run(context, **binding.planner_kwargs)
            except Exception as exc:  # noqa: BLE001 - failure is an A/B outcome
                trial_record.update(
                    {
                        "status": "failed",
                        "active_wall_seconds": round(time.monotonic() - started, 6),
                        "provider_calls": len(client.calls),
                        "calls": client.calls,
                        "error_class": exc.__class__.__name__,
                        "error_message": str(exc)[:1_000],
                    }
                )
                blinded_payload = {
                    "schema_version": "easyicu.blinded_planner_plan/1",
                    "label": label,
                    "status": "failed",
                    "error_class": exc.__class__.__name__,
                }
            else:
                trial_record.update(
                    {
                        "status": "ok",
                        "active_wall_seconds": round(time.monotonic() - started, 6),
                        "provider_calls": len(client.calls),
                        "calls": client.calls,
                    }
                )
                blinded_payload = {
                    "schema_version": "easyicu.blinded_planner_plan/1",
                    "label": label,
                    "status": "ok",
                    "plan": plan.model_dump(mode="json"),
                }
            _write_json(out_dir / "blinded" / f"{label}.json", blinded_payload)
        run_manifest["trials"].append(trial_record)
        _write_json(out_dir / "operator" / "arm_key.json", private_key)
        _write_json(out_dir / "manifest.json", run_manifest)

    _write_json(out_dir / "operator" / "arm_key.json", private_key)
    _write_json(out_dir / "manifest.json", run_manifest)
    print(json.dumps(run_manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
