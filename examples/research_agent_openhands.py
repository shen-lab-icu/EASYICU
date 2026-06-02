"""Run the research agent with an OpenHands-style runtime image.

This is the concrete demo for the ``runner_factory`` / sandbox
story. It uses the existing DockerRunner backend, but points it at an
OpenHands-compatible runtime image supplied by the user:

    export OPENHANDS_RUNTIME_IMAGE=ghcr.io/all-hands-ai/runtime:latest
    python examples/research_agent_openhands.py

If the image is not present locally, pass ``--pull``. The cohort and
script are mounted exactly like DockerRunner does: cohort read-only,
step workspace read-write, network disabled by default.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _bootstrap():
    here = Path(__file__).resolve().parent
    repo_root = here.parent
    src_path = repo_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))
    from research_agent_mortality_sofa import build_synthetic_cohort  # type: ignore
    from easyicu.research_agent import MockLLMClient, ResearchAgentPipeline
    return repo_root, build_synthetic_cohort, MockLLMClient, ResearchAgentPipeline


def main() -> int:
    repo_root, build_synthetic_cohort, MockLLMClient, ResearchAgentPipeline = _bootstrap()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image",
        default=os.environ.get(
            "OPENHANDS_RUNTIME_IMAGE",
            "ghcr.io/all-hands-ai/runtime:latest",
        ),
        help="OpenHands-compatible runtime image.",
    )
    parser.add_argument("--pull", action="store_true", help="docker pull image before running.")
    parser.add_argument("--network", default="none", help="Docker network mode (default: none).")
    parser.add_argument(
        "--workdir",
        default=str(repo_root / "research_output" / "openhands_demo"),
    )
    args = parser.parse_args()

    pipeline = ResearchAgentPipeline(
        workdir=args.workdir,
        llm=MockLLMClient(),
        runner_kind="docker",
        runner_image=args.image,
        runner_network=args.network,
        runner_kwargs={"pull_image": args.pull},
    )
    result = pipeline.run(
        question="Is admission SOFA-2 score associated with ICU mortality?",
        cohort=build_synthetic_cohort(),
        cohort_name="openhands_runtime_demo",
        database="synthetic",
        target_outcome="death",
        inclusion_criteria=["Synthetic first ICU admissions"],
    )
    print("OpenHands-style runtime demo complete")
    print(f"manifest:   {result.manifest_path}")
    print(f"report:     {result.report_path}")
    print(f"manuscript: {result.manuscript_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
