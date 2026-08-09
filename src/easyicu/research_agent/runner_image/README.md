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
    -t easyicu-research-agent:1.0.0 \
    -f src/easyicu/research_agent/runner_image/Dockerfile \
    .
```

The default base image is digest-pinned in both `Dockerfile` and
`base-image.lock`. Updating it requires reviewing the new upstream manifest,
running the image smoke job, and regenerating the uploaded CycloneDX SBOM.

The public default downloads from official PyPI. On a slow route, select a
trusted mirror at build time without changing the locked package set:

```bash
docker build \
    --build-arg PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
    --build-arg DEBIAN_MIRROR=https://mirrors.tuna.tsinghua.edu.cn/debian \
    --build-arg DEBIAN_SECURITY_MIRROR=https://mirrors.tuna.tsinghua.edu.cn/debian-security \
    -t easyicu-research-agent:1.0.0 \
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

Validate the exact image before starting a run:

```bash
python tools/check_agent_runtime.py \
    --image easyicu-research-agent:1.0.0
```

You can also set the `EASYICU_RUNNER_IMAGE` environment variable to
override the default tag without touching code.

Before a Planner call, EasyICU verifies the selected image's immutable id and
baseline package set. A stale or incomplete image therefore fails without
spending an LLM call.

## What's installed

The image pins:

* Python 3.11-slim, pinned by multi-platform OCI digest
The direct scientific stack is pinned in `requirements.lock`. It includes the
baseline numpy/pandas/pyarrow/scipy/statsmodels/scikit-learn/matplotlib stack and
the curated optional method packages declared in
`contracts/method_packages.py`. DockerRunner records the fully resolved
transitive `pip freeze` and immutable image id for every run.

If a new method needs another package, add it to the host-owned curated package
registry and the lock file, build a new versioned image tag, run the capability
check, and select that image with `EASYICU_RUNNER_IMAGE`. Generated code cannot
install packages at analysis time; an unavailable package must use its declared
fallback or fail closed.

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
