"""Figure-contract PREPARATION — shaping / canonicalization / install helpers.

Split out of ``contract_gate`` (Codex-ordered) to keep a clean responsibility
boundary: ``contract_gate`` holds only READ-ONLY findings gates, while this module
holds the figure-contract preparation helpers that WRITE the figure contract file
and RETURN a canonicalization code candidate:

* ``_ensure_step_figure_contract`` — materialises the per-step figure contract file.
* ``_figure_contract_source_data_canonicalization_candidate`` /
  ``_install_figure_contract_source_data_canonicalization`` — build / install the
  source-data-schema canonicalization repair candidate.
* ``_step_has_figure_only_output_contract`` / ``_infer_step_figure_panel_role`` /
  ``_reader_label_from_stem`` / ``_step_summary_paths`` — figure-shape predicates
  and label/path helpers they rely on.
* ``_family_has_deterministic_figure_renderer`` — renderer-family predicate (its
  consuming demotion stays in the execution layer).

The DECISION to install a canonicalization and the demotion that consumes
``_family_has_deterministic_figure_renderer`` stay in ``pipeline_execute``; only
the preparation helpers live here. Imports only leaf modules (schema /
publication_figures / stdlib) so there is no import cycle with ``pipeline_execute``
or ``contract_gate``. ``pipeline_execute`` re-exports every name here for
back-compat.
"""

from __future__ import annotations

import json
import os
import stat
import tempfile
from pathlib import Path
from typing import Any, List, Mapping, Optional, Sequence, Tuple

from ..publication_figures import make_figure_contract
from ..schema import AnalysisStep


def _step_has_figure_only_output_contract(step: AnalysisStep) -> bool:
    """Whether replacing ``outputs/`` can only replace presentation artifacts.

    Deterministic renderers install a complete staged bundle.  They are safe as
    a preflight or whole-directory repair only for an explicitly figure-only
    step; a mixed table/model + figure contract must stay with the coder so a
    renderer cannot erase or silently stand in for scientific products.
    """

    outputs = [
        str(output or "").strip()
        for output in (step.expected_outputs or [])
        if str(output or "").strip()
    ]

    def _is_typed_figure_product(output: str) -> bool:
        token = str(output or "").strip().lower()
        kind, separator, _product = token.partition(":")
        if separator:
            # The artifact kind is authoritative. A scientific table/model
            # whose product name happens to contain ``figure`` or ``plot`` is
            # still a mixed contract and must remain coder-owned.
            return kind.strip() in {"figure", "plot", "chart", "fig", "heatmap"}
        # Legacy bare declarations are figure-only only when they name an
        # actual image/vector export, never from a keyword in the stem.
        return token.endswith((".png", ".svg", ".pdf", ".tif", ".tiff"))

    return bool(outputs) and all(_is_typed_figure_product(output) for output in outputs)


def _reader_label_from_stem(stem: str) -> str:
    words = [
        token for token in stem.replace("-", "_").replace(".", "_").split("_") if token
    ]
    if not words:
        return "Manuscript figure"
    return " ".join(
        word.capitalize() if len(word) > 3 else word.upper() for word in words
    )


def _infer_step_figure_panel_role(step: AnalysisStep, stem: str) -> str:
    text = " ".join(
        [
            step.step_id,
            step.intent or "",
            step.method or "",
            stem,
            " ".join(step.expected_outputs or []),
        ]
    ).lower()
    if any(token in text for token in ("robustness", "sensitivity", "specification")):
        return "robustness"
    if any(
        token in text
        for token in (
            "missingness",
            "measurement",
            "quality",
            "baseline",
            "table one",
            "attrition",
            "cohort",
            "audit",
        )
    ):
        return "audit"
    if any(
        token in text
        for token in ("association", "effect", "forest", "estimate", "outcome")
    ):
        return "relationship"
    return "overview"


def _step_summary_paths(
    value: Any,
    *,
    out_dir: Path,
    allowed_suffixes: Optional[set[str]] = None,
) -> List[Path]:
    raw_values: List[Any] = []
    if isinstance(value, (str, Path)):
        raw_values = [value]
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        raw_values = list(value)
    paths: List[Path] = []
    for raw in raw_values:
        path = Path(str(raw))
        if not path.is_absolute():
            path = out_dir / path
        if not path.exists() or not path.is_file():
            continue
        if allowed_suffixes is not None and path.suffix.lower() not in allowed_suffixes:
            continue
        paths.append(path)
    return sorted(dict.fromkeys(paths))


def _ensure_step_figure_contract(
    *,
    step: AnalysisStep,
    out_dir: Path,
    step_summary: Mapping[str, Any],
    evidence_ids: Sequence[str],
) -> Optional[Path]:
    """Create a minimal manuscript-facing contract for valid figure exports.

    Coder prompts already ask for ``*.figure_contract.json``. This runner-level
    fallback covers the common successful-plot / missing-boilerplate case without
    weakening result-bearing figure gates: association and robustness figures
    still keep their result-like roles, so the contract validator can require
    multi-panel evidence when appropriate.
    """

    if sorted(out_dir.glob("*.figure_contract.json")):
        return None
    figure_suffixes = {".svg", ".pdf", ".png", ".tiff", ".tif", ".pptx"}
    figure_paths = _step_summary_paths(
        step_summary.get("figure_files") or step_summary.get("figure_path"),
        out_dir=out_dir,
        allowed_suffixes=figure_suffixes,
    )
    if not figure_paths:
        figure_paths = sorted(
            path
            for path in out_dir.iterdir()
            if path.is_file() and path.suffix.lower() in figure_suffixes
        )
    if not figure_paths:
        return None
    source_paths = _step_summary_paths(
        step_summary.get("source_data_files")
        or step_summary.get("source_data")
        or step_summary.get("source_table"),
        out_dir=out_dir,
    )
    primary_stem = figure_paths[0].stem
    label = _reader_label_from_stem(primary_stem)
    role = _infer_step_figure_panel_role(step, primary_stem)
    contract = make_figure_contract(
        figure_id=primary_stem,
        core_claim=(
            f"{label} summarizes the planned manuscript figure from registered "
            "source data."
        ),
        panels=[
            {
                "panel_id": "A",
                "title": label,
                "role": role,
                "claim": (
                    "This panel displays the step result using registered "
                    "source data and preserved code provenance."
                ),
                "evidence_ids": list(evidence_ids),
                "review_risk": (
                    "Review the source data and upstream step contract before "
                    "using the panel in manuscript text."
                ),
            }
        ],
        export_formats=[
            suffix.lstrip(".")
            for suffix in (".svg", ".pdf", ".png", ".tiff")
            if any(path.suffix.lower() == suffix for path in figure_paths)
        ]
        or ["svg", "png"],
        source_data=[path.name for path in source_paths],
        statistics_note="Auto-generated by the runner from step summary metadata.",
        image_integrity_note="No values were invented or visually altered by this contract synthesis.",
    )
    contract_path = out_dir / f"{primary_stem}.figure_contract.json"
    contract_path.write_text(
        json.dumps(contract.model_dump(mode="json"), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return contract_path


def _figure_contract_source_data_canonicalization_candidate(
    *,
    contract_path: Path,
    out_dir: Path,
) -> Optional[Tuple[str, str, List[str]]]:
    """Return an exact legacy-descriptor -> flat-basename JSON rewrite.

    ``make_figure_contract`` accepts small path mappings as an in-memory input
    compatibility layer but persists canonical ``List[str]`` source data.
    Some legacy agent scripts wrote those mappings directly to JSON.  This
    representation-only migration is deliberately strict: every populated path
    alias must agree, every source must be an existing ordinary local CSV in
    the exact step output directory, and non-empty evidence references are not
    discarded.  Anything else is left untouched for the validator to block.
    """

    output_root = Path(out_dir).resolve()
    candidate_path = Path(contract_path)
    try:
        if (
            candidate_path.parent.resolve() != output_root
            or candidate_path.resolve(strict=True).parent != output_root
            or not candidate_path.is_file()
            or candidate_path.is_symlink()
            or candidate_path.stat().st_nlink != 1
        ):
            return None
        before = candidate_path.read_text(encoding="utf-8")
        payload = json.loads(before)
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None

    raw_sources = payload.get("source_data")
    if isinstance(raw_sources, Mapping):
        source_items: List[Any] = [raw_sources]
    elif isinstance(raw_sources, list):
        source_items = list(raw_sources)
    else:
        return None
    if not source_items or not any(isinstance(item, Mapping) for item in source_items):
        return None

    path_keys = ("file", "filename", "path", "relative_path")
    canonical_names: List[str] = []
    for item in source_items:
        if isinstance(item, str):
            source_name = item.strip()
        elif isinstance(item, Mapping):
            if item.get("evidence_id") not in (None, "") or item.get(
                "evidence_ids"
            ) not in (None, "", []):
                return None
            populated: List[str] = []
            for key in path_keys:
                value = item.get(key)
                if value in (None, ""):
                    continue
                if not isinstance(value, str) or not value.strip():
                    return None
                populated.append(value.strip())
            if len(set(populated)) != 1:
                return None
            source_name = populated[0]
        else:
            return None
        if (
            not source_name
            or Path(source_name).name != source_name
            or "/" in source_name
            or "\\" in source_name
            or Path(source_name).suffix.lower() != ".csv"
        ):
            return None
        source_path = output_root / source_name
        try:
            if (
                source_path.resolve(strict=True).parent != output_root
                or not source_path.is_file()
                or source_path.is_symlink()
                or source_path.stat().st_nlink != 1
            ):
                return None
        except OSError:
            return None
        canonical_names.append(source_name)

    canonical_payload = dict(payload)
    canonical_payload["source_data"] = canonical_names
    after = json.dumps(canonical_payload, indent=2, ensure_ascii=False) + "\n"
    if before == after:
        return None
    return before, after, canonical_names


def _install_figure_contract_source_data_canonicalization(
    *,
    contract_path: Path,
    expected_before: str,
    canonical_text: str,
) -> None:
    """Atomically install one pre-authorized contract-schema rewrite.

    The generated step controls its output directory, so a predictable temp
    path is unsafe: it could be pre-created as a symlink before the host writes.
    ``mkstemp`` gives us an exclusive random regular file.  The destination is
    also reopened without following symlinks and must still match the exact
    content reviewed by the authorization boundary.
    """

    contract_path = Path(contract_path)
    parent = contract_path.parent
    read_flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    contract_fd = os.open(contract_path, read_flags)
    try:
        opened_stat = os.fstat(contract_fd)
        if not stat.S_ISREG(opened_stat.st_mode) or opened_stat.st_nlink != 1:
            raise ValueError("figure contract must remain one ordinary file")
        with os.fdopen(contract_fd, "r", encoding="utf-8") as handle:
            contract_fd = -1
            observed_before = handle.read()
        if observed_before != expected_before:
            raise ValueError("figure contract changed after canonicalization review")

        temporary_fd, temporary_name = tempfile.mkstemp(
            prefix=f".{contract_path.name}.",
            suffix=".schema.tmp",
            dir=parent,
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(temporary_fd, "w", encoding="utf-8") as handle:
                handle.write(canonical_text)
                handle.flush()
                os.fsync(handle.fileno())
            current_stat = os.stat(contract_path, follow_symlinks=False)
            if (
                not stat.S_ISREG(current_stat.st_mode)
                or current_stat.st_nlink != 1
                or current_stat.st_dev != opened_stat.st_dev
                or current_stat.st_ino != opened_stat.st_ino
            ):
                raise ValueError("figure contract identity changed before replace")
            os.replace(temporary_path, contract_path)
            try:
                directory_fd = os.open(
                    parent,
                    os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
                )
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
            except OSError:
                pass
        finally:
            temporary_path.unlink(missing_ok=True)
    finally:
        if contract_fd >= 0:
            os.close(contract_fd)


def _family_has_deterministic_figure_renderer(context: Any) -> bool:
    """True when this study-design family builds its PRIMARY publication figure
    deterministically in the write phase (``render_family_figure``).

    Lazy import keeps ``pipeline_execute`` free of a ``figures`` /
    ``study_design`` import-order dependency and fail-safes to False (strict) if
    the family cannot be inferred.
    """
    try:
        from ..figures import FAMILY_RENDERERS
        from ..study_design import infer_study_design_family

        return str(infer_study_design_family(context)) in FAMILY_RENDERERS
    except Exception:
        return False


__all__ = [
    "_step_has_figure_only_output_contract",
    "_reader_label_from_stem",
    "_infer_step_figure_panel_role",
    "_step_summary_paths",
    "_ensure_step_figure_contract",
    "_figure_contract_source_data_canonicalization_candidate",
    "_install_figure_contract_source_data_canonicalization",
    "_family_has_deterministic_figure_renderer",
]
