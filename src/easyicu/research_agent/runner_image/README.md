# `easyicu-research-agent` runner image

A reference Docker image for the T3.1 sandbox runner
(:class:`easyicu.research_agent.DockerRunner`).

## Why a separate image?

The code runner mounts the cohort parquet read-only and runs each
agent-generated step inside a container with `--network none` by
default. Because the network is off, every Python dependency the
script might import has to already be in the image — installing
packages at run time would defeat the isolation guarantee.

## Build

From the repository root:

```bash
docker build \
    -t easyicu-research-agent:latest \
    -f src/easyicu/research_agent/runner_image/Dockerfile \
    .
```

If you tag it differently, point the pipeline at your tag:

```python
from easyicu.research_agent import ResearchAgentPipeline

pipe = ResearchAgentPipeline(
    workdir="./research_output",
    runner_kind="docker",
    runner_image="my-org/easyicu-research-agent:2026-05-04",
)
```

You can also set the `EASYICU_RUNNER_IMAGE` environment variable to
override the default tag without touching code.

## What's installed

The image pins:

* Python 3.11-slim
* numpy, pandas, pyarrow
* scipy, statsmodels (for the T1.6 logistic-regression step)
* matplotlib (set to the `Agg` backend, no display required)

If your pipeline emits scripts that need anything else, fork the
Dockerfile and rebuild.

## Determinism switches

The image bakes in a few environment variables that protect the
hash-stability claim:

* `PYTHONHASHSEED=0` — set/dict ordering is deterministic across runs.
* `PYTHONDONTWRITEBYTECODE=1` — no `__pycache__` litter in the mounted
  step directory.
* `MPLBACKEND=Agg` — figures render without a display server.

The runner host also injects:

* `COHORT_PARQUET=/cohort.parquet` — read-only mount of the materialised
  cohort.
* `STEP_OUT_DIR=/workspace/outputs` — read-write mount that captures
  every artefact the script writes.

## Non-root user

The image creates a `runner` user (uid 1000) so `--user=1000:1000`
works out of the box. This means the host-mounted step directory
ends up owned by the calling user instead of root.

## OpenHands?

The pipeline interface is small enough that OpenHands or any other
sandbox can be plugged in without subclassing — pass a
`runner_factory` callable to :class:`ResearchAgentPipeline` that
returns any object with a `run(step_id=, code=) -> RunResult`
method.
