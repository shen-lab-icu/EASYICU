"""Zero-Provider Docker resource acceptance for frozen Canonical9 inputs.

This is a development-only admission check.  It exercises the exact
``DockerRunner`` isolation path and complete materialized inputs, but never
constructs a Provider client and never produces a scientific estimate for
publication.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import subprocess
import tempfile
from typing import Any, Callable, Mapping, Sequence

import pyarrow.parquet as pq

from easyicu.research_agent.authority.evidence_store import (
    EvidenceStore,
    sha256_of_file,
)
from easyicu.research_agent.execution.runner import DockerRunner
from easyicu.research_agent.intake.materialized_metadata import (
    MaterializedCohortAuthorityRef,
)
from easyicu.research_agent.intake.materialized_trajectory import (
    StagedTrajectoryBinding,
)

from benchmarks.figure2_canonical9.evaluator.rubric_v1 import FIGURE2_TASK_IDS
from benchmarks.figure2_canonical9.prompt_preflight import (
    _authority_ref,
    _strict_rows,
)

RESOURCE_PREFLIGHT_SCHEMA_VERSION = "easyicu.canonical9_resource_preflight/1"
RESOURCE_PROBE_SCHEMA_VERSION = "easyicu.canonical9_resource_probe/1"
RESOURCE_REPORT_FILENAME = "canonical9_resource_preflight.json"

_CONTAINER_SOURCE_ROOT = "/easyicu-extra/canonical-source"
_REQUIRED_IMPORTS = (
    "duckdb",
    "lifelines",
    "matplotlib",
    "numpy",
    "pandas",
    "patsy",
    "pyarrow",
    "scipy",
    "seaborn",
    "shap",
    "sklearn",
    "statsmodels",
    "xgboost",
)
_FAMILY_BY_TASK = {
    "e1_sepsis3_prevalence_mortality": "prevalence",
    "e2_lactate_mortality": "association",
    "e3_kdigo_gradient": "ordinal_gradient",
    "m1_hepatobiliary_missingness": "missingness",
    "m2_mortality_prediction": "prediction",
    "m3_sepsis_subphenotype": "clustering",
    "h1_ventilation_survival": "survival",
    "h2_vasopressor_causal": "causal",
    "h3_trajectory_clustering": "trajectory_clustering",
}


class ResourcePreflightError(RuntimeError):
    """The final Docker/input resource envelope is not admissible."""


@dataclass(frozen=True)
class ResourceCase:
    """One verified materialized input selected by the frozen JSONL."""

    task_id: str
    case_dir: Path
    cohort_path: Path
    cohort_authority_ref: MaterializedCohortAuthorityRef
    cohort_rows: int
    cohort_columns: tuple[str, ...]
    trajectory_binding: StagedTrajectoryBinding | None
    trajectory_rows: int | None
    trajectory_columns: tuple[str, ...]

    @property
    def trajectory_path(self) -> Path | None:
        return (
            self.trajectory_binding.path
            if self.trajectory_binding is not None
            else None
        )


def _strict_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ResourcePreflightError(f"invalid strict JSON artifact {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ResourcePreflightError(f"JSON artifact is not an object: {path}")
    return payload


def _reject_duplicate_pairs(
    pairs: Sequence[tuple[str, object]],
) -> dict[str, object]:
    payload: dict[str, object] = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError(f"duplicate JSON key {key!r}")
        payload[key] = value
    return payload


def _reject_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON constant {value!r}")


def _load_cases(jsonl_path: Path) -> list[ResourceCase]:
    rows = _strict_rows(jsonl_path)
    cases: list[ResourceCase] = []
    for row in rows:
        task_id = str(row["key"])
        cohort_path, cohort_ref, trajectory_binding = _authority_ref(row)
        # ``_authority_ref`` has already performed the complete typed authority
        # verification. Project dimensions from Parquet metadata without
        # repeating the expensive full semantic scan (especially H3).
        cohort_metadata = pq.ParquetFile(cohort_path)
        trajectory_rows: int | None = None
        trajectory_columns: tuple[str, ...] = ()
        if trajectory_binding is not None:
            trajectory_metadata = pq.ParquetFile(trajectory_binding.path)
            trajectory_rows = int(trajectory_metadata.metadata.num_rows)
            trajectory_columns = tuple(trajectory_metadata.schema_arrow.names)
        cases.append(
            ResourceCase(
                task_id=task_id,
                case_dir=cohort_path.parent,
                cohort_path=cohort_path,
                cohort_authority_ref=cohort_ref,
                cohort_rows=int(cohort_metadata.metadata.num_rows),
                cohort_columns=tuple(cohort_metadata.schema_arrow.names),
                trajectory_binding=trajectory_binding,
                trajectory_rows=trajectory_rows,
                trajectory_columns=trajectory_columns,
            )
        )
    observed = tuple(case.task_id for case in cases)
    if observed != tuple(FIGURE2_TASK_IDS):
        raise ResourcePreflightError(
            f"Canonical9 resource order mismatch: {observed!r}"
        )
    return cases


def _case_source_paths(case: ResourceCase) -> tuple[Path, ...]:
    paths: list[Path] = []
    if case.case_dir.is_symlink() or not case.case_dir.is_dir():
        raise ResourcePreflightError(
            f"case source must be a real directory: {case.case_dir}"
        )
    for path in sorted(case.case_dir.iterdir()):
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise ResourcePreflightError(f"case source contains a symlink: {path}")
        if stat.S_ISREG(metadata.st_mode):
            if metadata.st_nlink != 1:
                raise ResourcePreflightError(
                    f"case source must be singly linked: {path}"
                )
            paths.append(path)
        elif not stat.S_ISDIR(metadata.st_mode):
            raise ResourcePreflightError(
                f"case source contains a special filesystem entry: {path}"
            )
    if case.cohort_path not in paths:
        raise ResourcePreflightError(
            f"cohort is not a direct regular child of its case directory: {case.task_id}"
        )
    if case.trajectory_path is not None and case.trajectory_path not in paths:
        raise ResourcePreflightError(
            f"trajectory is not a direct regular child of its case directory: {case.task_id}"
        )
    return tuple(paths)


def _fingerprint(paths: Sequence[Path]) -> dict[str, tuple[int, int, str]]:
    result: dict[str, tuple[int, int, str]] = {}
    for path in paths:
        metadata = path.lstat()
        if (
            not stat.S_ISREG(metadata.st_mode)
            or stat.S_ISLNK(metadata.st_mode)
            or metadata.st_nlink != 1
        ):
            raise ResourcePreflightError(
                f"resource source changed type during admission: {path}"
            )
        result[str(path)] = (
            int(metadata.st_size),
            int(metadata.st_mtime_ns),
            sha256_of_file(path),
        )
    return result


def _source_inventory(
    case: ResourceCase,
    fingerprint: Mapping[str, tuple[int, int, str]],
) -> list[dict[str, object]]:
    return [
        {
            "file": path.name,
            "size_bytes": fingerprint[str(path)][0],
            "sha256": fingerprint[str(path)][2],
        }
        for path in _case_source_paths(case)
    ]


def _probe_code(case: ResourceCase) -> str:
    task_id = json.dumps(case.task_id)
    family = json.dumps(_FAMILY_BY_TASK[case.task_id])
    has_trajectory = repr(case.trajectory_binding is not None)
    required_imports = json.dumps(list(_REQUIRED_IMPORTS))
    return f'''
import gc
import hashlib
import importlib
from importlib import metadata as importlib_metadata
import json
import math
import os
from pathlib import Path
import resource
import time

import numpy as np
import pandas as pd

from easyicu.research_agent.authority.evidence_store import EvidenceStore
from easyicu.research_agent.intake.materialized_metadata import (
    load_verified_materialized_cohort_authority,
)
from easyicu.research_agent.intake.materialized_trajectory import (
    load_verified_materialized_trajectory_authority,
)
from easyicu.research_agent.methods.table_one import build_grouped_table_one

SCHEMA_VERSION = {RESOURCE_PROBE_SCHEMA_VERSION!r}
TASK_ID = {task_id}
FAMILY = {family}
HAS_TRAJECTORY = {has_trajectory}
REQUIRED_IMPORTS = {required_imports}
SOURCE_ROOT = Path({_CONTAINER_SOURCE_ROOT!r})
COHORT_PATH = SOURCE_ROOT / "cohort.parquet"
TRAJECTORY_PATH = SOURCE_ROOT / "cohort_trajectory.parquet"
OUT_DIR = Path(os.environ["STEP_OUT_DIR"])
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _mount_options(target):
    wanted = str(Path(target))
    matches = []
    for raw in Path("/proc/self/mountinfo").read_text(encoding="utf-8").splitlines():
        fields = raw.split()
        if len(fields) > 6 and fields[4] == wanted:
            matches.append(set(fields[5].split(",")))
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one direct mount for {{wanted}}")
    return matches[0]


def _python_scalar(value):
    return value.item() if isinstance(value, np.generic) else value


def _numeric_matrix(frame, *, rows=4096, excluded=()):
    excluded_set = {{str(value) for value in excluded}}
    columns = [
        str(column)
        for column in frame.select_dtypes(include=["number", "bool"]).columns
        if str(column) not in excluded_set
        and not str(column).lower().endswith(("_id", "_time"))
    ][:8]
    if not columns:
        raise RuntimeError("family executor found no numeric feature columns")
    matrix = frame.loc[:, columns].head(rows).apply(pd.to_numeric, errors="coerce")
    matrix = matrix.replace([np.inf, -np.inf], np.nan)
    matrix = matrix.fillna(matrix.median(numeric_only=True)).fillna(0.0)
    return matrix


def _binary_target(frame, requested="death", *, rows=4096):
    candidates = [requested, "death", "mort_28d_max", "adm"]
    for column in candidates:
        if column not in frame.columns:
            continue
        target = pd.to_numeric(frame[column], errors="coerce").head(rows)
        observed = sorted(target.dropna().unique().tolist())
        if len(observed) == 2:
            return column, target
    raise RuntimeError("family executor found no usable binary target")


def _run_table_one(frame):
    if "sex" not in frame.columns or "age" not in frame.columns:
        raise RuntimeError("Table 1 resource probe requires sex and age")
    table_frame = frame.loc[:, ["sex", "age"]].dropna(subset=["sex"]).copy()
    levels = sorted(
        (_python_scalar(value) for value in table_frame["sex"].unique().tolist()),
        key=lambda value: (type(value).__name__, str(value)),
    )
    if len(levels) < 2:
        raise RuntimeError("Table 1 resource probe requires two observed sex levels")
    table = build_grouped_table_one(
        table_frame,
        {{
            "group_by": "sex",
            "group_levels": levels,
            "variables": [
                {{
                    "name": "age",
                    "variable_kind": "continuous",
                    "summary": "median_iqr",
                    "test": "mann_whitney_or_kruskal",
                }}
            ],
        }},
    )
    if table.empty:
        raise RuntimeError("deterministic Table 1 returned no rows")
    return int(len(table))


def _run_family(frame):
    if FAMILY == "prevalence":
        _, target = _binary_target(frame)
        target.value_counts(dropna=False)
        return "pandas_prevalence", int(target.notna().sum())
    if FAMILY == "missingness":
        frame.isna().sum()
        return "pandas_missingness", int(len(frame))
    if FAMILY == "ordinal_gradient":
        from scipy import stats

        exposure = next(
            (name for name in ("aki_stage_max", "sofa2_max") if name in frame.columns),
            None,
        )
        if exposure is None:
            raise RuntimeError("ordinal resource probe found no declared score column")
        _, target = _binary_target(frame)
        pair = pd.concat(
            [pd.to_numeric(frame[exposure], errors="coerce").head(4096), target],
            axis=1,
        ).dropna()
        if len(pair) < 10:
            raise RuntimeError("ordinal resource probe has too few complete rows")
        stats.spearmanr(pair.iloc[:, 0], pair.iloc[:, 1])
        return "scipy_ordinal_gradient", int(len(pair))
    if FAMILY == "association":
        import statsmodels.api as sm

        _, target = _binary_target(frame)
        exposure = next(
            (name for name in ("lact_max", "lact_first", "age") if name in frame.columns),
            None,
        )
        if exposure is None:
            raise RuntimeError("association resource probe found no numeric exposure")
        design = pd.concat(
            [pd.to_numeric(frame[exposure], errors="coerce").head(4096), target],
            axis=1,
        ).dropna()
        if len(design) < 20:
            raise RuntimeError("association resource probe has too few complete rows")
        sm.GLM(
            design.iloc[:, 1],
            sm.add_constant(design.iloc[:, [0]], has_constant="add"),
            family=sm.families.Binomial(),
        ).fit(maxiter=25, disp=0)
        return "statsmodels_binomial", int(len(design))
    if FAMILY in {{"prediction", "causal"}}:
        from sklearn.linear_model import LogisticRegression

        target_name, target = _binary_target(frame)
        exposure_name = next(
            (
                name
                for name in ("vaso_ind_max", "vaso_ind_first")
                if name in frame.columns
            ),
            None,
        )
        excluded = [target_name]
        if FAMILY == "causal" and exposure_name is not None:
            causal_target = pd.to_numeric(
                frame[exposure_name], errors="coerce"
            ).head(4096)
            if causal_target.dropna().nunique() == 2:
                target = causal_target
                excluded.append(exposure_name)
        matrix = _numeric_matrix(frame, excluded=excluded)
        paired = matrix.copy()
        paired["__target"] = target.reindex(paired.index)
        paired = paired.dropna()
        if paired["__target"].nunique() != 2:
            raise RuntimeError("logistic resource probe target is not binary")
        LogisticRegression(max_iter=100, random_state=0).fit(
            paired.drop(columns="__target"),
            paired["__target"],
        )
        return (
            "sklearn_propensity" if FAMILY == "causal" else "sklearn_prediction",
            int(len(paired)),
        )
    if FAMILY == "clustering":
        from sklearn.cluster import KMeans

        matrix = _numeric_matrix(frame)
        if len(matrix) < 20:
            raise RuntimeError("clustering resource probe has too few rows")
        KMeans(n_clusters=2, n_init=1, random_state=0).fit(matrix)
        return "sklearn_clustering", int(len(matrix))
    if FAMILY == "survival":
        from lifelines import KaplanMeierFitter

        duration_name = next(
            (name for name in ("los_hosp", "los_icu") if name in frame.columns),
            None,
        )
        if duration_name is None:
            raise RuntimeError("survival resource probe found no duration")
        _, event = _binary_target(frame)
        survival = pd.concat(
            [pd.to_numeric(frame[duration_name], errors="coerce").head(4096), event],
            axis=1,
        ).dropna()
        survival = survival[survival.iloc[:, 0] > 0]
        if len(survival) < 20:
            raise RuntimeError("survival resource probe has too few rows")
        KaplanMeierFitter().fit(survival.iloc[:, 0], survival.iloc[:, 1])
        return "lifelines_survival", int(len(survival))
    if FAMILY == "trajectory_clustering":
        # The actual trajectory executor runs after the full trajectory load.
        return "trajectory_pending", 0
    raise RuntimeError(f"unknown resource family {{FAMILY}}")


def _load_parquet(path):
    started = time.monotonic()
    frame = pd.read_parquet(path)
    return frame, time.monotonic() - started


source_options = _mount_options(SOURCE_ROOT)
cohort_options = _mount_options(Path(os.environ["COHORT_PARQUET"]))
output_options = _mount_options(OUT_DIR)
if "ro" not in source_options or "ro" not in cohort_options:
    raise RuntimeError("Canonical source mounts are not read-only")
if "ro" in output_options:
    raise RuntimeError("step output mount is unexpectedly read-only")

package_versions = {{}}
for module_name in REQUIRED_IMPORTS:
    module = importlib.import_module(module_name)
    distribution_name = "scikit-learn" if module_name == "sklearn" else module_name
    try:
        package_versions[module_name] = importlib_metadata.version(distribution_name)
    except importlib_metadata.PackageNotFoundError:
        package_versions[module_name] = str(getattr(module, "__version__", "unknown"))

tmp_probe = Path("/tmp/easyicu-resource-probe.tmp")
tmp_probe.write_bytes(b"easyicu")
tmp_probe.unlink()
shm_probe = Path("/dev/shm/easyicu-resource-probe.tmp")
shm_probe.write_bytes(b"easyicu")
shm_probe.unlink()

verified_cohort = load_verified_materialized_cohort_authority(COHORT_PATH)
if verified_cohort is None:
    raise RuntimeError("cohort authority verification returned None")
cohort_frame, cohort_load_seconds = _load_parquet(COHORT_PATH)
if len(cohort_frame) != verified_cohort.authority.cohort_rows:
    raise RuntimeError("loaded cohort row count disagrees with authority")
if tuple(str(column) for column in cohort_frame.columns) != tuple(
    verified_cohort.authority.cohort_columns
):
    raise RuntimeError("loaded cohort columns disagree with authority")

table_one_rows = _run_table_one(cohort_frame)
family_executor, family_rows = _run_family(cohort_frame)
cohort_metrics = {{
    "compressed_size_bytes": int(COHORT_PATH.stat().st_size),
    "rows": int(len(cohort_frame)),
    "columns": int(len(cohort_frame.columns)),
    "load_seconds": float(cohort_load_seconds),
    "authority_sha256": verified_cohort.reference.sha256,
}}
del cohort_frame
gc.collect()

trajectory_metrics = None
if HAS_TRAJECTORY:
    verified_trajectory = load_verified_materialized_trajectory_authority(
        TRAJECTORY_PATH,
        expected_universe_authority=verified_cohort.reference,
    )
    if verified_trajectory is None:
        raise RuntimeError("trajectory authority verification returned None")
    trajectory_frame, trajectory_load_seconds = _load_parquet(TRAJECTORY_PATH)
    if len(trajectory_frame) != verified_trajectory.authority.trajectory_rows:
        raise RuntimeError("loaded trajectory row count disagrees with authority")
    if tuple(str(column) for column in trajectory_frame.columns) != tuple(
        verified_trajectory.authority.trajectory_columns
    ):
        raise RuntimeError("loaded trajectory columns disagree with authority")
    aggregated = (
        trajectory_frame.groupby(
            ["stay_id", "concept"],
            observed=True,
            sort=False,
        )["value_num"]
        .mean()
        .reset_index()
    )
    if aggregated.empty:
        raise RuntimeError("trajectory aggregation returned no rows")
    if FAMILY == "trajectory_clustering":
        from sklearn.cluster import KMeans

        selected_ids = aggregated["stay_id"].drop_duplicates().head(2048)
        matrix = (
            aggregated[aggregated["stay_id"].isin(selected_ids)]
            .pivot(index="stay_id", columns="concept", values="value_num")
            .replace([np.inf, -np.inf], np.nan)
        )
        matrix = matrix.fillna(matrix.median(numeric_only=True)).dropna(axis=1)
        matrix = matrix.fillna(0.0)
        if len(matrix) < 20 or matrix.shape[1] < 1:
            raise RuntimeError("trajectory clustering probe has insufficient matrix")
        KMeans(n_clusters=2, n_init=1, random_state=0).fit(matrix)
        family_executor = "sklearn_trajectory_clustering"
        family_rows = int(len(matrix))
    trajectory_metrics = {{
        "compressed_size_bytes": int(TRAJECTORY_PATH.stat().st_size),
        "rows": int(len(trajectory_frame)),
        "columns": int(len(trajectory_frame.columns)),
        "load_seconds": float(trajectory_load_seconds),
        "aggregated_rows": int(len(aggregated)),
        "authority_sha256": verified_trajectory.reference.sha256,
    }}
    del aggregated
    del trajectory_frame
    gc.collect()

usage = resource.getrusage(resource.RUSAGE_SELF)
peak_rss_bytes = int(usage.ru_maxrss) * 1024
tmp_stats = os.statvfs("/tmp")
shm_stats = os.statvfs("/dev/shm")
payload = {{
    "schema_version": SCHEMA_VERSION,
    "status": "passed",
    "development_only": True,
    "paper_authorized": False,
    "provider_calls": 0,
    "task_id": TASK_ID,
    "family": FAMILY,
    "family_executor": {{
        "name": family_executor,
        "status": "passed",
        "input_rows": int(family_rows),
    }},
    "table_one": {{
        "status": "passed",
        "result_rows": table_one_rows,
    }},
    "cohort": cohort_metrics,
    "trajectory": trajectory_metrics,
    "packages": package_versions,
    "mounts": {{
        "source_read_only": True,
        "cohort_read_only": True,
        "output_writable": True,
    }},
    "scratch": {{
        "tmp_write": "passed",
        "tmp_capacity_bytes": int(tmp_stats.f_blocks * tmp_stats.f_frsize),
        "shm_write": "passed",
        "shm_capacity_bytes": int(shm_stats.f_blocks * shm_stats.f_frsize),
    }},
    "peak_rss_bytes": peak_rss_bytes,
}}
artifact_path = OUT_DIR / "resource_probe.json"
artifact_path.write_text(
    json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\\n",
    encoding="utf-8",
)
store_root = OUT_DIR / "temporary_evidence_store"
record = EvidenceStore(store_root).register_file(
    kind="log",
    description="Development-only zero-Provider Docker resource probe.",
    source_path=artifact_path,
    produced_by_step=f"resource_{{TASK_ID}}",
    evidence_id=f"resource_{{TASK_ID}}",
    producer="canonical9_resource_preflight",
    generation_mode="deterministic",
)
reopened = EvidenceStore(store_root).get(record.evidence_id)
if reopened is None or reopened.sha256 != record.sha256:
    raise RuntimeError("temporary EvidenceStore registration did not reopen")
(OUT_DIR / "resource_probe_registration.json").write_text(
    json.dumps(
        {{
            "evidence_id": record.evidence_id,
            "sha256": record.sha256,
            "reopened": True,
        }},
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
    )
    + "\\n",
    encoding="utf-8",
)
print(
    json.dumps(
        {{
            "status": "passed",
            "task_id": TASK_ID,
            "peak_rss_bytes": peak_rss_bytes,
        }},
        ensure_ascii=False,
        allow_nan=False,
    )
)
'''.strip()


def _validate_probe(case: ResourceCase, payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != RESOURCE_PROBE_SCHEMA_VERSION:
        raise ResourcePreflightError(
            f"task {case.task_id} resource probe schema mismatch"
        )
    fixed = {
        "status": "passed",
        "development_only": True,
        "paper_authorized": False,
        "provider_calls": 0,
        "task_id": case.task_id,
    }
    for key, expected in fixed.items():
        if payload.get(key) != expected:
            raise ResourcePreflightError(
                f"task {case.task_id} resource probe changed {key!r}"
            )
    cohort = payload.get("cohort")
    if not isinstance(cohort, Mapping):
        raise ResourcePreflightError(f"task {case.task_id} has no cohort metrics")
    if (
        cohort.get("rows") != case.cohort_rows
        or cohort.get("columns") != len(case.cohort_columns)
        or cohort.get("authority_sha256") != case.cohort_authority_ref.sha256
    ):
        raise ResourcePreflightError(
            f"task {case.task_id} cohort metrics disagree with authority"
        )
    trajectory = payload.get("trajectory")
    if case.trajectory_binding is None:
        if trajectory is not None:
            raise ResourcePreflightError(
                f"task {case.task_id} unexpectedly reported a trajectory"
            )
    else:
        if not isinstance(trajectory, Mapping):
            raise ResourcePreflightError(
                f"task {case.task_id} has no trajectory metrics"
            )
        if (
            trajectory.get("rows") != case.trajectory_rows
            or trajectory.get("columns") != len(case.trajectory_columns)
            or trajectory.get("authority_sha256")
            != case.trajectory_binding.authority_ref.sha256
        ):
            raise ResourcePreflightError(
                f"task {case.task_id} trajectory metrics disagree with authority"
            )
    packages = payload.get("packages")
    if not isinstance(packages, Mapping) or set(packages) != set(_REQUIRED_IMPORTS):
        raise ResourcePreflightError(
            f"task {case.task_id} did not import the complete package envelope"
        )
    mounts = payload.get("mounts")
    if not isinstance(mounts, Mapping) or mounts != {
        "source_read_only": True,
        "cohort_read_only": True,
        "output_writable": True,
    }:
        raise ResourcePreflightError(
            f"task {case.task_id} did not prove its mount permissions"
        )
    for section in ("table_one", "family_executor"):
        value = payload.get(section)
        if not isinstance(value, Mapping) or value.get("status") != "passed":
            raise ResourcePreflightError(
                f"task {case.task_id} {section} did not pass"
            )
    for label, value in (
        ("peak_rss_bytes", payload.get("peak_rss_bytes")),
        ("cohort load_seconds", cohort.get("load_seconds")),
        ("cohort compressed_size_bytes", cohort.get("compressed_size_bytes")),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0
        ):
            raise ResourcePreflightError(
                f"task {case.task_id} has invalid {label}"
            )
    scratch = payload.get("scratch")
    if (
        not isinstance(scratch, Mapping)
        or scratch.get("tmp_write") != "passed"
        or scratch.get("shm_write") != "passed"
        or int(scratch.get("tmp_capacity_bytes") or 0) <= 0
        or int(scratch.get("shm_capacity_bytes") or 0) <= 0
    ):
        raise ResourcePreflightError(
            f"task {case.task_id} scratch-space probe did not pass"
        )


def _container_ids(docker_executable: str) -> frozenset[str]:
    process = subprocess.run(
        [
            docker_executable,
            "ps",
            "-aq",
            "--filter",
            "name=easyicu-ra-",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=30.0,
        encoding="utf-8",
        errors="replace",
    )
    if process.returncode != 0:
        raise ResourcePreflightError(
            "could not inspect DockerRunner container cleanup state"
        )
    return frozenset(line.strip() for line in process.stdout.splitlines() if line.strip())


def _git_identity(repo_root: Path) -> tuple[str, str]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
        timeout=30.0,
        encoding="utf-8",
        errors="replace",
    )
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
        timeout=30.0,
        encoding="utf-8",
        errors="replace",
    )
    if commit.returncode != 0 or status.returncode != 0:
        raise ResourcePreflightError("could not bind resource preflight to Git")
    return commit.stdout.strip(), status.stdout


def _atomic_report(path: Path, report: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise ResourcePreflightError(f"unsafe resource report destination: {path}")
    raw = (
        json.dumps(
            report,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=2,
        ).encode("utf-8")
        + b"\n"
    )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
        os.chmod(path, 0o600)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def run_canonical9_resource_preflight(
    *,
    jsonl_path: Path,
    output_dir: Path,
    image: str,
    timeout_seconds: float = 900.0,
    docker_executable: str = "docker",
    runner_factory: Callable[..., Any] = DockerRunner,
    require_clean_git: bool = True,
) -> dict[str, Any]:
    """Run the complete-input, zero-Provider acceptance sequentially."""

    source_candidate = Path(jsonl_path).expanduser()
    if source_candidate.is_symlink():
        raise ResourcePreflightError(
            "Canonical9 source JSONL must not be a symlink"
        )
    source_jsonl = source_candidate.resolve(strict=True)
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    repo_root = Path(__file__).resolve().parents[2]
    commit, dirty = _git_identity(repo_root)
    if require_clean_git and dirty:
        raise ResourcePreflightError(
            "resource preflight requires a clean Git tree so image/source identity "
            "cannot drift during admission"
        )
    if not str(image or "").strip():
        raise ResourcePreflightError("an explicit final Docker image is required")
    if not math.isfinite(float(timeout_seconds)) or float(timeout_seconds) <= 0:
        raise ResourcePreflightError("timeout_seconds must be finite and positive")

    cases = _load_cases(source_jsonl)
    source_paths = [source_jsonl]
    case_paths: dict[str, tuple[Path, ...]] = {}
    for case in cases:
        case_paths[case.task_id] = _case_source_paths(case)
        source_paths.extend(case_paths[case.task_id])
    before = _fingerprint(tuple(dict.fromkeys(source_paths)))
    containers_before = _container_ids(docker_executable)

    task_reports: list[dict[str, Any]] = []
    image_ids: set[str] = set()
    for case in cases:
        with tempfile.TemporaryDirectory(
            prefix=f".canonical9-resource-{case.task_id}-",
            dir=destination,
        ) as temporary:
            workdir = Path(temporary) / "run"
            runner = runner_factory(
                workdir=workdir,
                cohort_parquet=case.cohort_path,
                image=image,
                docker_executable=docker_executable,
                network="none",
                timeout_seconds=float(timeout_seconds),
                extra_mounts=(
                    (
                        str(case.case_dir),
                        _CONTAINER_SOURCE_ROOT,
                        "ro",
                    ),
                ),
            )
            result = runner.run(
                step_id=f"resource_{case.task_id}",
                code=_probe_code(case),
            )
            if not result.succeeded:
                raise ResourcePreflightError(
                    f"task {case.task_id} Docker probe failed: "
                    f"returncode={result.returncode}, timed_out={result.timed_out}, "
                    f"stderr_sha256={hashlib.sha256(result.stderr.encode()).hexdigest()}"
                )
            if result.requested_network_policy != "docker:none":
                raise ResourcePreflightError(
                    f"task {case.task_id} did not run with Docker network none"
                )
            probe_path = result.out_dir / "resource_probe.json"
            registration_path = result.out_dir / "resource_probe_registration.json"
            probe = _strict_json(probe_path)
            registration = _strict_json(registration_path)
            _validate_probe(case, probe)
            evidence_id = str(registration.get("evidence_id") or "")
            reopened = EvidenceStore(
                result.out_dir / "temporary_evidence_store"
            ).get(evidence_id)
            if (
                not evidence_id
                or reopened is None
                or reopened.sha256 != registration.get("sha256")
                or reopened.sha256 != sha256_of_file(probe_path)
            ):
                raise ResourcePreflightError(
                    f"task {case.task_id} EvidenceStore registration is invalid"
                )
            provenance = dict(result.runtime_provenance or {})
            image_id = str(provenance.get("image_id") or "")
            if not image_id.startswith("sha256:"):
                raise ResourcePreflightError(
                    f"task {case.task_id} has no immutable Docker image id"
                )
            image_ids.add(image_id)
            task_reports.append(
                {
                    **probe,
                    "docker_duration_seconds": float(result.duration_seconds),
                    "docker_image_id": image_id,
                    "docker_repo_digests": list(
                        provenance.get("repo_digests") or ()
                    ),
                    "runner_requirements_sha256": provenance.get(
                        "requirements_sha256"
                    ),
                    "probe_artifact_sha256": reopened.sha256,
                    "evidence_registration": "passed",
                    "source_files": _source_inventory(case, before),
                }
            )

    after = _fingerprint(tuple(dict.fromkeys(source_paths)))
    if before != after:
        changed = sorted(set(before) | set(after))
        changed = [path for path in changed if before.get(path) != after.get(path)]
        raise ResourcePreflightError(
            "resource preflight changed source files: "
            + ", ".join(Path(path).name for path in changed)
        )
    containers_after = _container_ids(docker_executable)
    if containers_after != containers_before:
        raise ResourcePreflightError(
            "DockerRunner left a new resource-preflight container behind"
        )
    if len(image_ids) != 1:
        raise ResourcePreflightError(
            "Canonical9 tasks did not execute from one immutable Docker image"
        )

    peak = max(int(task["peak_rss_bytes"]) for task in task_reports)
    report: dict[str, Any] = {
        "schema_version": RESOURCE_PREFLIGHT_SCHEMA_VERSION,
        "status": "passed",
        "development_only": True,
        "paper_authorized": False,
        "provider_calls": 0,
        "git_commit": commit,
        "git_tree_clean": not bool(dirty),
        "requested_image": image,
        "docker_image_id": next(iter(image_ids)),
        "source_jsonl_sha256": before[str(source_jsonl)][2],
        "source_zero_write_verified": True,
        "container_cleanup_verified": True,
        "sequential_execution": True,
        "task_order": [case.task_id for case in cases],
        "task_count": len(task_reports),
        "peak_rss_bytes": peak,
        "tasks": task_reports,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_report(destination / RESOURCE_REPORT_FILENAME, report)
    return report


__all__ = [
    "RESOURCE_PREFLIGHT_SCHEMA_VERSION",
    "RESOURCE_PROBE_SCHEMA_VERSION",
    "RESOURCE_REPORT_FILENAME",
    "ResourceCase",
    "ResourcePreflightError",
    "run_canonical9_resource_preflight",
]
