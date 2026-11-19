"""Build the offline desktop M1 cassette from reviewed, invented responses.

Run as a module from the repository with external sockets disabled. Use a
fresh work directory and approve the generated plan at the acceptance prompt.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from easyicu.research_agent.canonical_json import canonical_json
from easyicu.research_agent.facade import record_review, run_public_demo
from easyicu.research_agent.orchestration.profiles import PUBLIC_DEMO_ACCEPTANCE
from easyicu.research_agent.providers.clients import RecordingLLMClient, ReplayLLMClient
from easyicu.research_agent.providers.protocol import (
    SCRIPTED_PROVIDER, ModelIdentity, RecordedResponse, ReplayCassette,
)
from easyicu.synthetic_desktop_source import ANALYSIS_INPUTS, source_spec
from tools.windows_laptop_acceptance import laptop_environment, prepare_synthetic_dataset, validate_work_dir


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic-source", action="store_true", required=True)
    parser.add_argument("--responses", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if output.exists():
        parser.error("output already exists; choose a new cassette path")

    table = json.loads(args.responses.read_text(encoding="utf-8"))
    model = ModelIdentity.model_validate(table["model"])
    if model.provider != SCRIPTED_PROVIDER:
        parser.error("response table must have a scripted model identity")
    entries = tuple(
        RecordedResponse.model_validate({**entry, "response": (
            entry["response"] if isinstance(entry["response"], str) else canonical_json(entry["response"])
        )})
        for entry in table["entries"]
    )
    work = args.work_dir.resolve()
    validate_work_dir(work)
    work.mkdir(parents=True, exist_ok=True)
    with laptop_environment(work, work / "cache", desktop_first_result=True):
        from easyicu.webserver import demo_source_storage as storage

        source = source_spec()
        export = prepare_synthetic_dataset(source, storage.source_paths(source), {})
        with storage.public_demo_replay_scope(export_path=export, database=source.database,
                                              profile_ref=PUBLIC_DEMO_ACCEPTANCE.ref, synthetic_source=True) as scope:
            script = ReplayCassette.create(source=scope.source, model=model, entries=entries)
            recorder = RecordingLLMClient(ReplayLLMClient(script), model=model)
            result = run_public_demo(export_path=export, database=source.database, work_dir=work / "recording",
                                     client=recorder, review=lambda pending: record_review(pending, recorder, scripted=True),
                                     **ANALYSIS_INPUTS)
            cassette = recorder.cassette()
            if cassette.entries != script.entries:
                parser.error("reviewed response table differs from the responses consumed by this run")
            cassette.write(output)
            print(json.dumps({**result, "cassette_sha256": cassette.cassette_sha256,
                              "recorded_requests": len(cassette.entries)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
