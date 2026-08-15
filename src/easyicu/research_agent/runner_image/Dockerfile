# Reference image for `DockerRunner` (T3.1 — research-agent sandbox).
#
# Build from the repository root:
#
#     docker build -t easyicu-research-agent:1.0.0 \
#         -f src/easyicu/research_agent/runner_image/Dockerfile .
#
# Then point the pipeline at it:
#
#     pipe = ResearchAgentPipeline(
#         workdir="./research_output",
#         runner_kind="docker",
#         runner_image="easyicu-research-agent:1.0.0",
#     )
#
# The container is invoked with `--network none` by default, so the
# script *cannot* call `pip install` at run time. Every dep the agent
# might import has to be present in this image.
#
# Determinism notes:
# * `MPLBACKEND=Agg`              — no DISPLAY required for plots.
# * `PYTHONDONTWRITEBYTECODE=1`   — no `__pycache__` litter inside the
#                                    mounted step dir.
# * `PIP_NO_CACHE_DIR=1`          — image stays small.
# * `PYTHONHASHSEED=0`            — best-effort reproducibility for
#                                    set/dict ordering in artefacts.

# Release builds default to the multi-platform digest recorded in
# base-image.lock. Updating the base is an explicit reviewed change; callers may
# still override BASE_IMAGE for a local experiment, but that changes execution
# identity and is recorded by DockerRunner.
ARG BASE_IMAGE=python:3.11-slim@sha256:90744cff8f32887f075c47d747a173ff333e9e98801667af93c357fa9f5e28ff
FROM ${BASE_IMAGE}

# Official PyPI remains the reproducible public default. Builders on a slow
# route may override it without changing the image contract, for example:
#   --build-arg PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
ARG PIP_INDEX_URL=https://pypi.org/simple
ARG DEBIAN_MIRROR=http://deb.debian.org/debian
ARG DEBIAN_SECURITY_MIRROR=http://deb.debian.org/debian-security

LABEL org.opencontainers.image.title="easyicu-research-agent runner"
LABEL org.opencontainers.image.description="Sandbox image for the EasyICU research-agent code runner (T3.1)."
LABEL org.opencontainers.image.source="https://github.com/shen-lab-icu/EASYICU"
LABEL org.opencontainers.image.base.digest="sha256:90744cff8f32887f075c47d747a173ff333e9e98801667af93c357fa9f5e28ff"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONHASHSEED=0 \
    MPLBACKEND=Agg

# Minimal native deps for pyarrow / scipy wheels.
RUN sed -i \
        -e "s|http://deb.debian.org/debian-security|${DEBIAN_SECURITY_MIRROR}|g" \
        -e "s|http://deb.debian.org/debian|${DEBIAN_MIRROR}|g" \
        /etc/apt/sources.list.d/debian.sources \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        libstdc++6 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install the complete, version-locked package set declared by
# contracts/method_packages.py. The image is network-disabled at execution
# time, so a package advertised to the coder must already be importable here.
COPY src/easyicu/research_agent/runner_image/requirements.lock /opt/easyicu-runner/
RUN pip install --no-cache-dir --index-url "${PIP_INDEX_URL}" \
        -r /opt/easyicu-runner/requirements.lock

# Generated analysis imports EasyICU's publication-figure and audit helpers.
# Install the repository package itself without resolving dependencies again;
# the complete runtime set is pinned above and is captured by DockerRunner.
COPY pyproject.toml README.md /opt/easyicu/
COPY src/easyicu /opt/easyicu/src/easyicu
RUN pip install --no-cache-dir --no-deps /opt/easyicu

# Non-root user so `--user=$(id -u):$(id -g)` works out of the box and
# the script cannot accidentally `chown` host-mounted files to root.
RUN useradd --create-home --shell /usr/sbin/nologin --uid 1000 runner
USER runner

WORKDIR /workspace

# `runner.py` invokes `python -u /workspace/analysis.py`, so no
# ENTRYPOINT / CMD is needed — the host explicitly composes the argv.
