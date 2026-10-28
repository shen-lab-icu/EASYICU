"""File and CLI support for offline reference scripts; no scientific defaults."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Sequence

import pandas as pd


@dataclass(frozen=True)
class LoadedCohort:
    frame: pd.DataFrame
    source_path: Path | None
    source_sha256: str | None
    provenance: str


def load_prepared_cohort(
    source: str | Path | pd.DataFrame,
    *,
    required_columns: Sequence[str],
    identity_column: str,
    provenance: str | None = None,
) -> LoadedCohort:
    """Read prepared rows without imputing, recoding, or selecting a population."""
    path = None if isinstance(source, pd.DataFrame) else Path(source).resolve()
    if path is None:
        frame = source.copy()
    elif path.suffix == ".parquet":
        frame = pd.read_parquet(path)
    elif path.suffix in {".csv", ".tsv"}:
        frame = pd.read_csv(path, sep="\t" if path.suffix == ".tsv" else ",")
    else:
        raise ValueError("Use a prepared .parquet, .csv or .tsv cohort")
    missing = set([identity_column, *required_columns]) - set(frame.columns)
    if missing:
        raise ValueError(f"Prepared cohort lacks columns: {sorted(missing)}")
    if frame.empty or frame[identity_column].isna().any() or frame[identity_column].duplicated().any():
        raise ValueError("Prepared cohort needs nonempty rows with unique, nonmissing row identities")
    declared = provenance or frame.attrs.get("provenance") or "undeclared_by_caller"
    digest = hashlib.sha256(path.read_bytes()).hexdigest() if path else None
    return LoadedCohort(frame, path, digest, str(declared))


def export_manifest(
    *,
    out_dir: str | Path,
    skill_id: str,
    loaded: LoadedCohort,
    spec: Any,
    artifacts: Sequence[Path],
) -> dict[str, Any]:
    """Bind exported results and the exact specification, without granting host authority."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    spec_path = out / "spec.json"
    spec_path.write_text(spec.model_dump_json(indent=2) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": "easyicu.reference_scripts/1",
        "skill_id": skill_id,
        "claim_ceiling": "analysis_only",
        "data_provenance": loaded.provenance,
        "n_source_rows": len(loaded.frame),
        "source_path": str(loaded.source_path) if loaded.source_path else None,
        "source_sha256": loaded.source_sha256,
        "spec_sha256": hashlib.sha256(spec_path.read_bytes()).hexdigest(),
        "artifacts": [
            {"path": path.relative_to(out).as_posix(),
             "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for path in sorted(set([spec_path, *artifacts]))
        ],
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def reference_cli(
    *,
    argv: Sequence[str] | None,
    description: str,
    spec_type: Any,
    run_all: Callable[..., Any],
    write_example: Callable[..., Any],
) -> int:
    parser = argparse.ArgumentParser(description=description)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--example", action="store_true", help="Write and analyze synthetic example rows")
    inputs.add_argument("--cohort", type=Path, help="Prepared cohort; requires --spec")
    parser.add_argument("--spec", type=Path, help="Exact reviewed specification in JSON")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--provenance", help="Caller-declared source provenance")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    if args.example and args.spec:
        parser.error("--example uses its own synthetic specification")
    if not args.example and not args.spec:
        parser.error("--cohort requires --spec")
    args.out.mkdir(parents=True, exist_ok=True)
    if args.example:
        cohort, spec = write_example(args.out)
        provenance = "synthetic_example"
    else:
        cohort, spec = args.cohort, spec_type.model_validate_json(args.spec.read_text())
        provenance = args.provenance
    manifest = run_all(cohort, spec, args.out, provenance=provenance)
    if not args.quiet:
        print(json.dumps(manifest, ensure_ascii=False))
    return 0


def plot_intervals(
    table: pd.DataFrame, out_dir: str | Path, *, stem: str,
    estimate_column: str, label_column: str, effect_scale: str,
) -> list[Path]:
    """Draw existing estimates; never refit a model or compute new intervals."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from ..execution.runners.effect_scale import describe_effect_scale
    from ..figures.publication import apply_publication_style

    palette = apply_publication_style(font_size=8)
    scale = describe_effect_scale(effect_scale)
    fig, ax = plt.subplots(figsize=(6, 1.5 + 0.35 * len(table)), layout="constrained")
    try:
        for index, row in enumerate(table.to_dict("records")):
            estimate, low, high = (float(row[key]) for key in (estimate_column, "ci_low", "ci_high"))
            ax.errorbar(estimate, index, xerr=[[estimate - low], [high - estimate]],
                        fmt="o", color=palette["blue"], capsize=3)
        ax.set_yticks(range(len(table)), table[label_column].astype(str))
        ax.invert_yaxis()
        ax.set_xlabel(f"{scale.name} (95% CI)")
        if scale.null_value is not None:
            ax.axvline(scale.null_value, color=palette["neutral"], linestyle="--")
        if scale.multiplicative:
            ax.set_xscale("log")
        paths = [Path(out_dir) / f"{stem}.{suffix}" for suffix in ("svg", "png")]
        for path in paths:
            fig.savefig(path, dpi=180)
        return paths
    finally:
        plt.close(fig)
