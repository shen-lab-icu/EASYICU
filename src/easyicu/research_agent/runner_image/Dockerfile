# Reference image for `DockerRunner` (T3.1 — research-agent sandbox).
#
# Build from the repository root:
#
#     docker build -t easyicu-research-agent:latest \
#         -f src/easyicu/research_agent/runner_image/Dockerfile .
#
# Then point the pipeline at it:
#
#     pipe = ResearchAgentPipeline(
#         workdir="./research_output",
#         runner_kind="docker",
#         runner_image="easyicu-research-agent:latest",
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

FROM python:3.11-slim

LABEL org.opencontainers.image.title="easyicu-research-agent runner"
LABEL org.opencontainers.image.description="Sandbox image for the EasyICU research-agent code runner (T3.1)."
LABEL org.opencontainers.image.source="https://github.com/your-org/EASYICU"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONHASHSEED=0 \
    MPLBACKEND=Agg

# Minimal native deps for pyarrow / scipy wheels.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libstdc++6 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install the complete capability set advertised by execution/method_capabilities.py.
# The image is network-disabled at execution time, so a package advertised to
# the coder must already be importable here. Keep this list aligned with the
# project core dependencies plus the ``methods`` extra.
RUN pip install --no-cache-dir \
        "numpy>=1.26,<3" \
        "pandas>=2.0,<3" \
        "pyarrow>=23,<24" \
        "scipy>=1.11,<2" \
        "statsmodels>=0.14,<1" \
        "matplotlib>=3.7,<4" \
        "seaborn>=0.12,<1" \
        "scikit-learn>=1.3,<2" \
        "lifelines>=0.27,<1" \
        "shap>=0.44,<1" \
        "xgboost>=2.0,<4" \
        "pydantic>=2,<3" \
        "pyyaml>=6,<7" \
        "requests>=2.28,<3" \
        "tqdm>=4.65,<5" \
        "duckdb>=0.9,<2"

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
