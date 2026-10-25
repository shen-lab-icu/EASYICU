#!/usr/bin/env python3
"""Record an explicitly authorized live public-demo acceptance, with human review.

Use an existing registered public-demo export. This tool never downloads,
converts, selects credentials or grants external model access. The ordinary
provider factory must already authorize the configured live provider.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from easyicu.research_agent.facade import record_review, run_public_demo
from easyicu.research_agent.orchestration.profiles import PUBLIC_DEMO_ACCEPTANCE
from easyicu.research_agent.providers.factory import build_provider_client
from easyicu.research_agent.providers.clients import (
    RecordingLLMClient, recording_model_identity,
)

from easyicu.webserver.demo_source_storage import public_demo_replay_scope


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-path", required=True, type=Path)
    parser.add_argument("--database", required=True)
    parser.add_argument("--provider", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--strict-json-schema", action="store_true")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--question", required=True)
    parser.add_argument("--primary-exposure", required=True)
    parser.add_argument("--target-outcome", required=True)
    parser.add_argument("--covariate", action="append", default=[])
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("output already exists; choose a new cassette path")
    with public_demo_replay_scope(export_path=args.export_path, database=args.database,
                                  profile_ref=PUBLIC_DEMO_ACCEPTANCE.ref):
        inner = build_provider_client(provider=args.provider, model=args.model,
            request_timeout=120, title="Public demo acceptance recording", max_retries=0,
            supports_strict_json_schema=args.strict_json_schema)
        recorder = RecordingLLMClient(inner, model=recording_model_identity(inner))
        result = run_public_demo(export_path=args.export_path, database=args.database, work_dir=args.work_dir,
            client=recorder, question=args.question, primary_exposure=args.primary_exposure,
            target_outcome=args.target_outcome, covariates=args.covariate,
            review=lambda pending: record_review(pending, recorder))
        cassette = recorder.cassette()
        cassette.write(args.output)
        print(json.dumps({**result, "cassette_sha256": cassette.cassette_sha256,
                          "recorded_requests": len(cassette.entries)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
