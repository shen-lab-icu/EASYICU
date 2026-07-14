"""Execution-time boundary between planned and produced step products.

The planner owns scientific scope through typed ``kind:product`` entries in
``AnalysisStep.expected_outputs``.  A successful script must realise those
products in its machine-readable summary, and it may not silently widen a
non-figure/non-effect step into a publication figure or effect analysis.

This module only validates declarations and registrations.  It never chooses
an exposure, outcome, cohort, estimator, or analysis method.
"""

from __future__ import annotations

import json
import hashlib
import math
import os
import re
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .repair_registry import is_sealed_renderer_repair, repair_metadata_for
from .schema import AnalysisStep, ValidationFinding
from .trajectory_plan_contract import (
    trajectory_role_result_findings,
    trajectory_role_scope_summary_findings,
)

_FIGURE_KINDS = frozenset({"figure", "plot", "chart", "fig", "heatmap"})
_FAILED_STATUSES = frozenset(
    {
        "blocked",
        "error",
        "failed",
        "execution_failed",
        "contract_failed",
        "fail_closed",
        "failed_closed",
        "repair_failed",
        "skipped",
        "skipped_dependency_failed",
    }
)
_OUTPUT_CONTAINER_KEYS = frozenset(
    {"output_files", "output_artifacts", "outputs", "figure_files"}
)
_DIRECT_FIGURE_KEYS = frozenset({"figure_file", "figure_path"})
_FIGURE_SUFFIXES = frozenset({".png", ".svg", ".pdf", ".tif", ".tiff"})
_KNOWN_FILE_SUFFIXES = frozenset(
    {
        *_FIGURE_SUFFIXES,
        ".csv",
        ".tsv",
        ".parquet",
        ".feather",
        ".json",
        ".jsonl",
        ".md",
        ".txt",
        ".log",
        ".pkl",
        ".pickle",
        ".joblib",
        ".npy",
        ".npz",
    }
)
_EFFECT_PRODUCT_BASES = frozenset(
    {
        "adjusted_effect",
        "adjusted_effect_estimate",
        "adjusted_effect_estimates",
        "adjusted_association",
        "adjusted_associations",
        "adjusted_association_estimate",
        "adjusted_association_estimates",
        "adjusted_hr",
        "adjusted_odds_ratio",
        "adjusted_odds_ratios",
        "adjusted_or",
        "adjusted_rd",
        "adjusted_rr",
        "association_estimate",
        "association_estimates",
        "causal_effect",
        "coefficient",
        "coefficients",
        "effect_estimate",
        "effect_estimates",
        "effect_forest",
        "hazard_ratio",
        "interaction_pvalue",
        "odds_ratio",
        "or_estimate",
        "or_estimates",
        "or_forest",
        "overall_effect",
        "primary_association",
        "primary_association_estimate",
        "primary_effect",
        "primary_estimate",
        "primary_hr",
        "primary_or",
        "primary_rd",
        "primary_rr",
        "relative_risk",
        "rd_estimate",
        "rd_estimates",
        "rd_forest",
        "rr_estimate",
        "rr_estimates",
        "rr_forest",
        "risk_difference",
        "risk_ratio",
        "hr_estimate",
        "hr_estimates",
        "hr_forest",
        "subgroup_effect",
        "subgroup_effects",
        "treatment_effect",
        "adjusted_logistic_regression_primary",
    }
)

_PRODUCT_SLOT_SUFFIXES: Mapping[str, tuple[tuple[str, ...], ...]] = {
    "absolute_risk": (("absolute", "risk"),),
    "distribution": (("distribution",),),
    "availability": (
        ("availability",),
        ("availability", "panel"),
        ("measurement", "coverage"),
        ("source", "coverage"),
    ),
    "cohort_flow": (
        ("cohort", "flow"),
        ("eligibility", "flow"),
    ),
    "attrition_audit": (
        ("attrition",),
        ("attrition", "audit"),
    ),
    "primary_estimand": (
        ("adjusted", "effect"),
        ("adjusted", "association"),
        ("primary", "estimand"),
    ),
    "precision_audit": (
        ("precision", "audit"),
        ("interval", "width", "audit"),
    ),
    "robustness_plot": (
        ("robustness", "plot"),
        ("sensitivity", "plot"),
        ("robustness", "forest"),
        ("sensitivity", "forest"),
    ),
    "robustness_denominator_audit": (
        ("robustness", "denominator", "audit"),
        ("sensitivity", "denominator", "audit"),
        ("model", "denominator", "audit"),
    ),
}

# A planner product's subject must name the scientific entity being displayed,
# not another display archetype.  Without this structural guard a nested role
# such as ``kaplan_meier_curve_distribution`` can be stripped to the subject
# ``kaplan_meier_curve`` and then laundered into an ordinary distribution slot.
# These are generic display-role terminals, not benchmark or clinical tokens.
_NESTED_DISPLAY_ROLE_SUFFIXES: tuple[tuple[str, ...], ...] = (
    ("curve",),
    ("diagram",),
    ("forest",),
    ("heatmap",),
    ("plot",),
)

_OPERATIONAL_SUBJECT_SUFFIXES = frozenset(
    {
        "any",
        "count",
        "ever",
        "first",
        "flag",
        "indicator",
        "last",
        "max",
        "mean",
        "measured",
        "median",
        "min",
        "n",
        "observed",
        "raw",
        "sum",
        "value",
    }
)

# This subtree is a host-verified receipt, not a registry of scientific
# outputs.  The execution layer separately rejects these markers when they
# were not installed by an authorized sealed renderer, so excluding the digest
# map here cannot let generated code claim renderer authority.
_HOST_RECEIPT_SUBTREES = frozenset({"sealed_renderer_parent_digests"})


def _normalise(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower()).strip("_")


def _canonical_kind(value: object) -> str:
    kind = _normalise(value)
    if kind in _FIGURE_KINDS:
        return "figure"
    if kind in {"metric", "statistics"}:
        return "statistic"
    return kind


def typed_product(value: object) -> tuple[str, str] | None:
    """Return the shared canonical identity for a ``kind:product`` token."""

    kind, separator, product = str(value or "").strip().partition(":")
    if not separator:
        return None
    canonical_kind = _canonical_kind(kind)
    product_name = _normalise(Path(product).name)
    for suffix in sorted(_KNOWN_FILE_SUFFIXES, key=len, reverse=True):
        normalised_suffix = _normalise(suffix)
        if product_name.endswith(f"_{normalised_suffix}"):
            product_name = product_name[: -(len(normalised_suffix) + 1)]
            break
    if not canonical_kind or not product_name:
        return None
    return canonical_kind, product_name


def _contained_regular_output_file(out_dir: Path, value: object) -> Path:
    """Return one link-free regular file lexically contained in ``out_dir``.

    This helper runs inside the macOS execution sandbox.  Calling
    :meth:`Path.resolve` there asks the kernel to inspect ancestors such as
    ``/private`` that are intentionally outside the sandbox allowlist.  The
    host already supplies the absolute output root, so containment is enforced
    without ancestor traversal: require a relative path, reject parent
    components, reject symlinks at every output-local component, and reject
    hard-linked files.
    """

    root = Path(out_dir)
    raw = str(value or "").strip()
    relative = Path(raw)
    if not raw or relative.is_absolute() or ".." in relative.parts:
        raise ValueError("output path must be relative and contained")
    try:
        if root.is_symlink() or not root.is_dir():
            raise ValueError("output root is not a regular directory")
        candidate = root
        for part in relative.parts:
            candidate = candidate / part
            if candidate.is_symlink():
                raise ValueError("output path traverses a symlink")
        stat_result = candidate.stat()
    except OSError as exc:
        raise ValueError("output file is unavailable") from exc
    if not candidate.is_file() or stat_result.st_nlink != 1:
        raise ValueError("output path is not a single-link regular file")
    return candidate


def read_digest_bound_artifact_snapshot(
    *, parent_out: Path, artifact_digests: Mapping[str, str]
) -> dict[str, bytes]:
    """Read one link-free parent snapshot and verify the exact bytes once.

    Renderers must parse the returned bytes (for example through
    :class:`io.BytesIO`) rather than reopening the source paths.  This makes a
    file replacement after verification irrelevant to the authorized render.
    """

    root = Path(parent_out)
    if root.is_symlink() or not root.is_dir() or not artifact_digests:
        raise ValueError("parent artifact root is unavailable")
    snapshot: dict[str, bytes] = {}
    for raw_name, raw_digest in artifact_digests.items():
        name = str(raw_name or "").strip()
        digest = str(raw_digest or "").strip().lower()
        path = Path(name)
        candidate = root / name
        if (
            not name
            or path.name != name
            or path.is_absolute()
            or not re.fullmatch(r"[0-9a-f]{64}", digest)
            or candidate.is_symlink()
        ):
            raise ValueError("parent artifact seal contains an unsafe entry")
        try:
            stat_result = candidate.stat()
            payload = candidate.read_bytes()
        except OSError as exc:
            raise ValueError("parent artifact is unavailable") from exc
        if (
            not candidate.is_file()
            or stat_result.st_nlink != 1
            or hashlib.sha256(payload).hexdigest() != digest
        ):
            raise ValueError("parent artifact does not match its authorized digest")
        snapshot[name] = payload
    return snapshot


def _planner_product_slot_and_subject(
    product_name: str, allowed_slots: set[str]
) -> tuple[str, tuple[str, ...], str]:
    tokens = tuple(part for part in _normalise(product_name).split("_") if part)
    matches: set[tuple[str, tuple[str, ...], tuple[str, ...], str]] = set()
    for slot in allowed_slots:
        for role_tokens in _PRODUCT_SLOT_SUFFIXES.get(slot, ()):
            if (
                len(tokens) >= len(role_tokens)
                and tokens[-len(role_tokens) :] == role_tokens
            ):
                matches.add(
                    (slot, role_tokens, tokens[: -len(role_tokens)], "subject_role")
                )
            if (
                len(tokens) > len(role_tokens) + 1
                and tokens[: len(role_tokens)] == role_tokens
                and tokens[len(role_tokens)] == "by"
            ):
                matches.add(
                    (
                        slot,
                        role_tokens,
                        tokens[len(role_tokens) + 1 :],
                        "role_by_subject",
                    )
                )
    if len(matches) != 1:
        raise ValueError(
            "Planner figure role does not map uniquely to an authorized product slot"
        )
    slot, _role_tokens, subject, syntax = matches.pop()
    if not subject and syntax == "role_by_subject":
        raise ValueError("Planner figure role has an empty scientific subject")
    if any(
        tuple(subject[index : index + len(role_suffix)]) == role_suffix
        for role_suffix in _NESTED_DISPLAY_ROLE_SUFFIXES
        for index in range(len(subject) - len(role_suffix) + 1)
    ):
        raise ValueError(
            "Planner figure role nests an incompatible display archetype in "
            "its scientific subject"
        )
    return slot, subject, syntax


def _subject_tokens_compatible(
    declared_subject: tuple[str, ...], authoritative_subject: tuple[str, ...]
) -> bool:
    """Match a display label to one host-owned operational exposure.

    Operational columns may add a standard aggregation suffix (for example
    ``marker_max``). Human-readable aliases must be supplied explicitly by the
    host context; lexical prefix guesses are intentionally forbidden. This
    comparison is only one half of authorization: the exact declared subject
    must also occur in a typed direct-parent anchor.
    """

    reduced = list(authoritative_subject)
    while len(reduced) > 1 and reduced[-1] in _OPERATIONAL_SUBJECT_SUFFIXES:
        reduced.pop()
    return bool(declared_subject) and declared_subject in {
        authoritative_subject,
        tuple(reduced),
    }


def _contains_token_sequence(
    container: tuple[str, ...], candidate: tuple[str, ...]
) -> bool:
    return bool(candidate) and any(
        container[index : index + len(candidate)] == candidate
        for index in range(len(container) - len(candidate) + 1)
    )


def authorize_declared_figure_product_slots(
    *,
    declared_products: Sequence[str],
    renderer_repair_id: str,
    planner_parent_anchors: Sequence[str],
    authoritative_display_subjects: Sequence[str] = (),
) -> dict[str, str]:
    """Build the host-owned exact product-to-slot authorization.

    Product suffixes describe a display role, but a suffix alone is not
    authority: ``kaplan_meier_curve_distribution`` must not become an ordinary
    descriptive distribution merely because its last token matches.  A
    non-empty subject prefix is therefore accepted only when it is anchored to
    the beginning of a typed product or structured input from the host-recorded
    parent planning request.  Physical filenames and coder-written summaries
    are never semantic authority.  The exact Planner method/family selects the
    sealed renderer separately.
    """

    if not is_sealed_renderer_repair(renderer_repair_id):
        raise ValueError("renderer repair id is not an exact sealed renderer")
    allowed_slots = set(repair_metadata_for(renderer_repair_id).figure_product_slots)
    parent_token_sequences: set[tuple[str, ...]] = set()
    for value in planner_parent_anchors:
        parsed_parent = typed_product(value)
        parent_name = (
            parsed_parent[1]
            if parsed_parent is not None
            else (
                _file_stem(value)
                if Path(str(value or "")).name != "step_summary.json"
                else ""
            )
        )
        tokens = tuple(part for part in parent_name.split("_") if part)
        if tokens:
            parent_token_sequences.add(tokens)
    canonical: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    slot_bindings: dict[str, str] = {}
    claimed_slots: set[str] = set()
    for raw in declared_products:
        parsed = typed_product(raw)
        if parsed is None or parsed[0] != "figure" or parsed in seen:
            raise ValueError("Planner products are not unique typed figure roles")
        seen.add(parsed)
        canonical.append(parsed)
        slot, subject, syntax = _planner_product_slot_and_subject(
            parsed[1], allowed_slots
        )
        parent_subject_anchored = any(
            (
                (
                    len(parent_tokens) >= len(subject)
                    and parent_tokens[: len(subject)] == subject
                )
                if syntax == "subject_role"
                else _contains_token_sequence(parent_tokens, subject)
            )
            for parent_tokens in parent_token_sequences
        )
        if subject and not parent_subject_anchored:
            raise ValueError(
                "Planner figure role subject is not anchored to a verified "
                "Planner direct-parent product"
            )
        if syntax == "role_by_subject":
            authoritative_tokens = {
                tuple(part for part in _normalise(value).split("_") if part)
                for value in authoritative_display_subjects
                if _normalise(value)
            }
            if not authoritative_tokens or not any(
                _subject_tokens_compatible(subject, expected)
                for expected in authoritative_tokens
            ):
                raise ValueError(
                    "Planner role-by-subject figure does not match the host-owned "
                    "display subject"
                )
        if slot in claimed_slots:
            raise ValueError(
                "Planner figure roles do not map to distinct contract product slots"
            )
        slot_bindings[f"{parsed[0]}:{parsed[1]}"] = slot
        claimed_slots.add(slot)
    if not canonical:
        raise ValueError("Planner declared no typed figure products")
    return slot_bindings


def bind_declared_figure_products(
    *,
    out_dir: Path,
    declared_products: Sequence[str],
    authorized_product_slots: Mapping[str, str],
    renderer_repair_id: str,
    renderer_implementation_sha256: str,
    renderer_parent_digests: Mapping[str, str],
) -> bool:
    """Bind Planner-owned figure roles to one verified multi-panel bundle.

    Closed renderers own only rendering.  The Planner still owns the logical
    products, so the host embeds its exact ``expected_outputs`` in the trusted
    adapter and this helper registers those roles without inventing a new
    figure, statistic, cohort, or method.  Multiple planned roles may point to
    one canonical bundle only when its contract contains at least one panel per
    role; otherwise the binding fails closed.
    """

    if not is_sealed_renderer_repair(renderer_repair_id):
        raise ValueError("renderer repair id is not an exact sealed renderer")
    metadata = repair_metadata_for(renderer_repair_id)
    allowed_slots = set(metadata.figure_product_slots)
    if not allowed_slots:
        raise ValueError("sealed renderer declares no authorized product slots")
    if not re.fullmatch(r"[0-9a-f]{64}", renderer_implementation_sha256):
        raise ValueError("renderer implementation digest is invalid")
    canonical_parent_digests = {
        str(name): str(digest).strip().lower()
        for name, digest in renderer_parent_digests.items()
    }
    if "step_summary.json" not in canonical_parent_digests or any(
        Path(name).name != name or not re.fullmatch(r"[0-9a-f]{64}", digest)
        for name, digest in canonical_parent_digests.items()
    ):
        raise ValueError("renderer parent digest seal is invalid")

    out_dir = Path(out_dir)
    try:
        summary_path = _contained_regular_output_file(out_dir, "step_summary.json")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("rendered step_summary.json is unreadable") from exc
    if not isinstance(summary, dict) or summary.get("rendering_only") is not True:
        raise ValueError("rendered summary does not declare rendering_only=true")

    canonical: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str]] = set()
    for raw in declared_products:
        parsed = typed_product(raw)
        if parsed is None or parsed[0] != "figure" or parsed in seen:
            raise ValueError("Planner products are not unique typed figure roles")
        seen.add(parsed)
        canonical.append((parsed[0], parsed[1], str(raw).strip()))
    if not canonical:
        raise ValueError("Planner declared no typed figure products")
    canonical_names = {f"{kind}:{name}" for kind, name, _raw in canonical}
    supplied_slots = {
        str(product): _normalise(slot)
        for product, slot in authorized_product_slots.items()
    }
    if set(supplied_slots) != canonical_names:
        raise ValueError(
            "host product-slot authorization does not match Planner products"
        )
    if any(slot not in allowed_slots for slot in supplied_slots.values()) or len(
        supplied_slots.values()
    ) != len(set(supplied_slots.values())):
        raise ValueError("host product-slot authorization is invalid")

    figure_path_value = summary.get("figure_path")
    if not isinstance(figure_path_value, str) or not figure_path_value.strip():
        raise ValueError("rendered summary has no canonical figure_path")
    try:
        figure_path = _contained_regular_output_file(out_dir, figure_path_value)
    except ValueError as exc:
        raise ValueError("canonical figure is missing or outside STEP_OUT_DIR") from exc
    if figure_path.suffix.lower() not in _FIGURE_SUFFIXES:
        raise ValueError("canonical figure_path is not a supported figure file")

    contract_value = summary.get("figure_contract")
    if not isinstance(contract_value, str) or not contract_value.strip():
        raise ValueError("rendered summary has no figure_contract")
    try:
        contract_path = _contained_regular_output_file(out_dir, contract_value)
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(
            "figure contract is unreadable or outside STEP_OUT_DIR"
        ) from exc
    panels = contract.get("panels") if isinstance(contract, Mapping) else None
    panel_ids: set[str] = set()
    slot_to_panels: dict[str, set[str]] = defaultdict(set)
    for panel in panels or []:
        if not isinstance(panel, Mapping):
            continue
        panel_id = str(panel.get("panel_id") or "").strip()
        if not panel_id or panel_id in panel_ids:
            raise ValueError("figure contract panel ids are missing or duplicated")
        panel_ids.add(panel_id)
        panel_metadata = panel.get("metadata")
        raw_slots = (
            panel_metadata.get("planner_product_slots")
            if isinstance(panel_metadata, Mapping)
            else None
        )
        if raw_slots is None:
            continue
        if not isinstance(raw_slots, list) or not raw_slots:
            raise ValueError("panel planner_product_slots must be a non-empty list")
        normalized_slots = [_normalise(slot) for slot in raw_slots]
        if any(
            not slot or slot not in allowed_slots for slot in normalized_slots
        ) or len(normalized_slots) != len(set(normalized_slots)):
            raise ValueError("panel declares an unauthorized or duplicate product slot")
        for slot in normalized_slots:
            slot_to_panels[slot].add(panel_id)

    slot_bindings: dict[str, dict[str, object]] = {}
    for kind, name, _raw in canonical:
        canonical_name = f"{kind}:{name}"
        slot = supplied_slots[canonical_name]
        candidate_panels = slot_to_panels.get(slot, set())
        if not candidate_panels:
            raise ValueError(
                "authorized product slot is not anchored to a contract panel"
            )
        slot_bindings[canonical_name] = {
            "slot": slot,
            "panel_ids": sorted(candidate_panels),
        }

    output_files = summary.get("output_files")
    if not isinstance(output_files, dict):
        output_files = {}
    else:
        output_files = dict(output_files)
    for kind, name, _raw in canonical:
        output_files[f"{kind}:{name}"] = figure_path_value
    summary["output_files"] = output_files
    summary["planner_bound_figure_products"] = [
        f"{kind}:{name}" for kind, name, _raw in canonical
    ]
    summary["planner_product_slot_bindings"] = slot_bindings
    summary["planner_product_binding"] = "shared_multi_panel_bundle"
    summary["sealed_renderer_repair"] = renderer_repair_id
    summary["sealed_renderer_implementation_sha256"] = renderer_implementation_sha256
    summary["sealed_renderer_parent_digests"] = dict(
        sorted(canonical_parent_digests.items())
    )

    payload = json.dumps(summary, indent=2, ensure_ascii=False, default=str)
    temporary_fd, temporary_name = tempfile.mkstemp(
        dir=out_dir,
        prefix=".step_summary.planner_binding.",
        suffix=".tmp",
        text=True,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(temporary_fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, summary_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return True


# Internal compatibility alias; lineage and product-scope validation must share
# the public parser above rather than growing independent token grammars.
_typed_product = typed_product


def _file_stem(value: object) -> str:
    name = Path(str(value or "").strip()).name
    suffix = Path(name).suffix.lower()
    if suffix in _KNOWN_FILE_SUFFIXES:
        name = name[: -len(suffix)]
    return _normalise(name)


def _file_kinds(value: object) -> frozenset[str]:
    suffix = Path(str(value or "").strip()).suffix.lower()
    if suffix in _FIGURE_SUFFIXES:
        return frozenset({"figure"})
    if suffix in {".csv", ".tsv"}:
        return frozenset({"table", "artifact", "dataset", "test"})
    if suffix in {".parquet", ".feather"}:
        return frozenset({"artifact", "dataset", "table"})
    if suffix in {".pkl", ".pickle", ".joblib"}:
        return frozenset({"model", "artifact"})
    if suffix in {".npy", ".npz"}:
        return frozenset({"artifact", "dataset", "model"})
    if suffix in {".md", ".txt", ".log", ".jsonl"}:
        return frozenset({"log", "artifact"})
    if suffix == ".json":
        return frozenset({"artifact", "manifest", "log", "model", "test"})
    return frozenset()


def _is_file_path(value: object) -> bool:
    return isinstance(value, str) and Path(value.strip()).suffix.lower() in (
        _KNOWN_FILE_SUFFIXES
    )


def _iter_paths(value: Any) -> Iterable[str]:
    if _is_file_path(value):
        yield str(value).strip()
    elif isinstance(value, Mapping):
        for child in value.values():
            yield from _iter_paths(child)
    elif isinstance(value, (list, tuple, set)):
        for child in value:
            yield from _iter_paths(child)


def _summary_scalar_products(value: Any) -> set[tuple[str, str]]:
    """Return exact statistic/log keys backed by non-null scalar values."""

    products: set[tuple[str, str]] = set()

    def visit(node: Any) -> None:
        if isinstance(node, Mapping):
            for raw_key, child in node.items():
                key = _normalise(raw_key)
                if key in _HOST_RECEIPT_SUBTREES:
                    continue
                if isinstance(child, Mapping) or isinstance(child, (list, tuple)):
                    visit(child)
                    continue
                valid = child is not None and child != "" and not _is_file_path(child)
                if isinstance(child, float):
                    valid = math.isfinite(child)
                if valid and key:
                    products.add(("statistic", key))
                    products.add(("log", key))
        elif isinstance(node, (list, tuple)):
            for child in node:
                visit(child)

    visit(value)
    return products


def _registered_products(
    summary: Mapping[str, Any],
    *,
    out_dir: Path | None = None,
) -> tuple[set[tuple[str, str]], list[tuple[str, bool]]]:
    """Collect typed/file products and figure paths from output containers."""

    products: set[tuple[str, str]] = set()
    figure_paths: list[tuple[str, bool]] = []

    def is_actual_output(path: str) -> bool:
        if out_dir is None:
            return True
        root = out_dir.resolve()
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = root / candidate
        try:
            resolved = candidate.resolve(strict=True)
            resolved.relative_to(root)
        except (FileNotFoundError, OSError, ValueError):
            return False
        return resolved.is_file()

    def add_path(path: str, *, explicit_figure_list: bool = False) -> None:
        kinds = _file_kinds(path)
        stem = _file_stem(path)
        products.update((kind, stem) for kind in kinds if stem)
        if "figure" in kinds:
            figure_paths.append((path, explicit_figure_list))

    def add_container(value: Any, *, explicit_figure_list: bool = False) -> None:
        if isinstance(value, Mapping):
            for raw_role, child in value.items():
                role = _typed_product(raw_role)
                paths = [path for path in _iter_paths(child) if is_actual_output(path)]
                if role is not None:
                    role_kind, role_name = role
                    compatible_path = any(
                        role_kind in _file_kinds(path) for path in paths
                    )
                    scalar_registration = (
                        role_kind in {"statistic", "log"}
                        and not isinstance(child, (Mapping, list, tuple, set))
                        and child is not None
                        and child != ""
                    )
                    if compatible_path or scalar_registration:
                        products.add(role)
                elif paths:
                    role_name = _normalise(raw_role)
                    for path in paths:
                        products.update(
                            (kind, role_name) for kind in _file_kinds(path) if role_name
                        )
                for path in paths:
                    # A JSON file registered under an exact matching typed log
                    # role is an auxiliary sidecar, not four extra scientific
                    # products merely because JSON is a multi-purpose format.
                    # Keep suffix inference for every scientific role and for
                    # mismatched log filenames so an innocuous key cannot
                    # launder an effect-bearing output path.
                    if (
                        role is not None
                        and role[0] == "log"
                        and role[0] in _file_kinds(path)
                        and _file_stem(path) == role[1]
                    ):
                        continue
                    add_path(path, explicit_figure_list=explicit_figure_list)
            return
        for path in _iter_paths(value):
            add_path(path, explicit_figure_list=explicit_figure_list)

    def visit(node: Any) -> None:
        if isinstance(node, Mapping):
            for raw_key, child in node.items():
                key = _normalise(raw_key)
                if key in _OUTPUT_CONTAINER_KEYS:
                    add_container(child, explicit_figure_list=key == "figure_files")
                elif key in _DIRECT_FIGURE_KEYS:
                    add_container(child, explicit_figure_list=True)
                if isinstance(child, (Mapping, list, tuple)):
                    visit(child)
        elif isinstance(node, (list, tuple)):
            for child in node:
                visit(child)

    visit(summary)
    products.update(_summary_scalar_products(summary))
    return products, figure_paths


def _has_product_registry(value: Any) -> bool:
    """Whether a summary opted into the machine-readable output registry."""

    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            if _normalise(raw_key) in {
                "output_files",
                "output_artifacts",
                "outputs",
            }:
                return True
            if isinstance(child, (Mapping, list, tuple)) and _has_product_registry(
                child
            ):
                return True
    elif isinstance(value, (list, tuple)):
        return any(_has_product_registry(child) for child in value)
    return False


def _effect_bearing_name(name: str) -> bool:
    normalised = _normalise(name)
    # Input/design audits and provenance sidecars may name the effect they
    # inspect, but they are not authority to estimate or register that effect.
    # Any numeric effect nested inside such a summary is still caught by the
    # scalar-path walk below; this exemption applies only to the typed product
    # role itself.
    if any(
        normalised.endswith(f"_{suffix}")
        for suffix in (
            "audit",
            "contract",
            "definition",
            "diagnostic",
            "diagnostics",
            "input_audit",
            "lineage",
            "provenance",
            "trace",
        )
    ):
        return False
    return any(
        _contains_product_role(normalised, base) for base in _EFFECT_PRODUCT_BASES
    )


def _contains_product_role(name: str, role: str) -> bool:
    """Match a normalized multi-token role at any underscore boundary."""

    return name == role or f"_{role}_" in f"_{name}_"


def effect_bearing_product(value: object) -> bool:
    """Return whether a typed product name denotes a scientific effect."""

    parsed = typed_product(value)
    return parsed is not None and _effect_bearing_name(parsed[1])


def effect_bearing_name(value: object) -> bool:
    """Return whether an untyped/canonical product name denotes an effect."""

    return _effect_bearing_name(str(value or ""))


_EFFECT_MEASURE_PREFIXES: Mapping[str, tuple[str, ...]] = {
    "odds_ratio": (
        "odds_ratio",
        "adjusted_odds_ratio",
        "adjusted_odds_ratios",
        "primary_or",
        "adjusted_or",
        "or_estimate",
        "or_estimates",
        "or_forest",
    ),
    "risk_ratio": (
        "risk_ratio",
        "relative_risk",
        "primary_rr",
        "adjusted_rr",
        "rr_estimate",
        "rr_estimates",
        "rr_forest",
    ),
    "hazard_ratio": (
        "hazard_ratio",
        "primary_hr",
        "adjusted_hr",
        "hr_estimate",
        "hr_estimates",
        "hr_forest",
    ),
    "risk_difference": (
        "risk_difference",
        "primary_rd",
        "adjusted_rd",
        "rd_estimate",
        "rd_estimates",
        "rd_forest",
    ),
    "coefficient": ("coefficient", "coefficients"),
}

# These abbreviations are meaningful only when they are the complete typed
# product/column name.  Boundary matching would misclassify ordinary ICU
# products such as ``hr_trajectory`` (heart rate), ``rr_distribution``
# (respiratory rate), or prose-like ``included_or_excluded_counts``.
_EXACT_EFFECT_MEASURE_NAMES: Mapping[str, str] = {
    "or": "odds_ratio",
    "rr": "risk_ratio",
    "hr": "hazard_ratio",
    "rd": "risk_difference",
}


def effect_measure_family(value: object) -> str | None:
    """Return an explicit effect scale encoded in a typed product name."""

    parsed = typed_product(value)
    if parsed is None:
        return None
    name = parsed[1]
    exact_family = _EXACT_EFFECT_MEASURE_NAMES.get(name)
    if exact_family is not None:
        return exact_family
    for family, prefixes in _EFFECT_MEASURE_PREFIXES.items():
        if any(_contains_product_role(name, prefix) for prefix in prefixes):
            return family
    return None


_EFFECT_ROLE_PREFIXES: Mapping[str, tuple[str, ...]] = {
    "interaction": ("interaction", "interaction_pvalue"),
    "subgroup": ("subgroup", "subgroup_effect", "subgroup_effects"),
    "treatment": ("treatment", "treatment_effect"),
    "causal": ("causal", "causal_effect"),
}


def effect_role_family(value: object) -> str | None:
    """Return a non-interchangeable scientific effect role, if explicit."""

    parsed = typed_product(value)
    if parsed is None:
        return None
    name = parsed[1]
    for family, prefixes in _EFFECT_ROLE_PREFIXES.items():
        if any(_contains_product_role(name, prefix) for prefix in prefixes):
            return family
    return None


_EFFECT_ESTIMAND_TIER_PREFIXES: Mapping[str, tuple[str, ...]] = {
    "primary": ("primary",),
    "secondary": ("secondary",),
    "sensitivity": ("sensitivity", "robust", "robustness"),
    "corroborative": ("corroborative",),
}


def effect_estimand_tier(value: object) -> str | None:
    """Return a non-interchangeable primary/supporting estimand tier."""

    parsed = typed_product(value)
    if parsed is None:
        return None
    name = parsed[1]
    for tier, prefixes in _EFFECT_ESTIMAND_TIER_PREFIXES.items():
        if any(_contains_product_role(name, prefix) for prefix in prefixes):
            return tier
    return None


def effect_adjustment_family(value: object) -> str | None:
    """Return an explicit adjusted versus crude/unadjusted qualifier."""

    parsed = typed_product(value)
    if parsed is None:
        return None
    name = parsed[1]
    if _contains_product_role(name, "adjusted"):
        return "adjusted"
    if any(
        _contains_product_role(name, qualifier) for qualifier in ("unadjusted", "crude")
    ):
        return "unadjusted"
    return None


_AUXILIARY_LOG_SUFFIXES = (
    "_audit",
    "_contract",
    "_diagnostic",
    "_diagnostics",
    "_lineage",
    "_process",
    "_provenance",
    "_render_trace",
    "_rendering_trace",
    "_trace",
)


def _auxiliary_log_name(name: str) -> bool:
    normalised = _normalise(name)
    return any(normalised.endswith(suffix) for suffix in _AUXILIARY_LOG_SUFFIXES)


def _effect_summary_paths(summary: Mapping[str, Any]) -> list[str]:
    paths: list[str] = []

    def visit(node: Any, prefix: str = "") -> None:
        if isinstance(node, Mapping):
            for raw_key, child in node.items():
                key = _normalise(raw_key)
                path = f"{prefix}.{key}" if prefix else key
                if key in _HOST_RECEIPT_SUBTREES:
                    continue
                if isinstance(child, Mapping) or isinstance(child, (list, tuple)):
                    visit(child, path)
                elif (
                    key
                    and _effect_bearing_name(key)
                    and child is not None
                    and child != ""
                    and not _is_file_path(child)
                ):
                    paths.append(path)
        elif isinstance(node, (list, tuple)):
            for index, child in enumerate(node):
                visit(child, f"{prefix}[{index}]")

    visit(summary)
    return sorted(set(paths))


def _undeclared_figure_bundle(
    figure_paths: Sequence[tuple[str, bool]],
) -> dict[str, list[str]]:
    by_stem: dict[str, set[str]] = defaultdict(set)
    explicit_stems: set[str] = set()
    for path, explicit in figure_paths:
        stem = _file_stem(path)
        if not stem:
            continue
        by_stem[stem].add(Path(path).suffix.lower())
        if explicit:
            explicit_stems.add(stem)
    return {
        stem: sorted(suffixes)
        for stem, suffixes in by_stem.items()
        if len(suffixes) >= 2 or stem in explicit_stems
    }


def _assignment_model_completion_findings(
    *,
    step: AnalysisStep,
    step_summary: Mapping[str, Any],
    declared: set[tuple[str, str]],
) -> list[ValidationFinding]:
    """Require a declared assignment-model artifact to contain a fitted model.

    This validates realization of the Planner-owned method; it does not select
    the exposure, covariates, model family, cohort, or estimand.
    """

    if ("artifact", "assignment_model") not in declared:
        return []

    raw_models = step_summary.get("assignment_models")
    models = raw_models if isinstance(raw_models, list) else []
    fitted_models = [
        model
        for model in models
        if isinstance(model, Mapping)
        and _normalise(model.get("fit_status") or model.get("status")) == "fitted"
    ]
    if fitted_models:
        return []

    exposure = step_summary.get("exposure")
    exposure_resolution = (
        exposure.get("resolution")
        if isinstance(exposure, Mapping)
        and isinstance(exposure.get("resolution"), Mapping)
        else {}
    )
    model_diagnostics = [
        {
            "model_id": model.get("model_id"),
            "fit_status": _normalise(model.get("fit_status") or model.get("status")),
            "n": model.get("n"),
            "exposure_event_n": model.get("exposure_event_n"),
            "exposure_non_event_n": model.get("exposure_non_event_n"),
            "error": model.get("error"),
        }
        for model in models
        if isinstance(model, Mapping)
    ]
    overall_event_n = exposure.get("event_n") if isinstance(exposure, Mapping) else None
    overall_non_event_n = (
        exposure.get("non_event_n") if isinstance(exposure, Mapping) else None
    )
    exposure_class_collapse = bool(
        isinstance(overall_event_n, int)
        and isinstance(overall_non_event_n, int)
        and overall_event_n > 0
        and overall_non_event_n > 0
        and any(
            (
                diagnostic.get("exposure_event_n") == 0
                or diagnostic.get("exposure_non_event_n") == 0
            )
            for diagnostic in model_diagnostics
        )
    )
    return [
        ValidationFinding(
            validator="declared_product_contract",
            severity="error",
            message=(
                f"Step {step.step_id} declared an assignment-model artifact but "
                "registered no assignment model whose summary roster has "
                "`fit_status` exactly `fitted`. Empty, placeholder, or "
                "noncanonical model rosters are not completed products."
            ),
            detail={
                "kind": "assignment_model_unfitted",
                "step_id": step.step_id,
                "model_statuses": [
                    _normalise(model.get("fit_status") or model.get("status"))
                    for model in models
                    if isinstance(model, Mapping)
                ],
                "exposure_resolution_status": _normalise(
                    exposure_resolution.get("status")
                ),
                "exposure_resolution_reason": exposure_resolution.get("reason"),
                "overall_exposure_event_n": overall_event_n,
                "overall_exposure_non_event_n": overall_non_event_n,
                "model_diagnostics": model_diagnostics,
                "exposure_class_collapse_after_eligibility": exposure_class_collapse,
                "repair_constraint": (
                    "The resolved exposure has both classes before model-set "
                    "eligibility but at least one model set has lost a class. "
                    "Audit analysis-set and timing eligibility symmetrically "
                    "for exposed and unexposed rows; do not redefine the "
                    "Planner-owned exposure, cohort, model, or estimand."
                    if exposure_class_collapse
                    else None
                ),
            },
        )
    ]


def declared_product_contract_findings(
    *,
    step: AnalysisStep,
    step_summary: Mapping[str, Any],
    effect_method_authorized: bool,
    effect_figure_source_authorized: bool = False,
    out_dir: Path | None = None,
) -> list[ValidationFinding]:
    """Validate declared-product realization and scientific output scope."""

    reported_status = _normalise(step_summary.get("status"))
    if reported_status in _FAILED_STATUSES:
        return []

    declared = {
        product
        for raw in (step.expected_outputs or [])
        if (product := _typed_product(raw)) is not None
    }
    registered, figure_paths = _registered_products(step_summary, out_dir=out_dir)
    findings: list[ValidationFinding] = []
    findings.extend(
        trajectory_role_scope_summary_findings(
            step=step,
            step_summary=step_summary,
        )
    )
    findings.extend(
        trajectory_role_result_findings(
            step=step,
            step_summary=step_summary,
        )
    )
    findings.extend(
        _assignment_model_completion_findings(
            step=step,
            step_summary=step_summary,
            declared=declared,
        )
    )

    # Older direct unit fixtures predate ``output_files`` and validate only
    # their own numeric payload.  A real execution supplies ``out_dir`` and is
    # always held to the product boundary, even if its script tries to evade the
    # gate by omitting the modern registry entirely.
    missing = (
        sorted(declared - registered)
        if out_dir is not None or _has_product_registry(step_summary)
        else []
    )
    if missing:
        findings.append(
            ValidationFinding(
                validator="declared_product_contract",
                severity="error",
                message=(
                    f"Step {step.step_id} did not realise every typed product "
                    "declared by the plan in step_summary output registrations."
                ),
                detail={
                    "kind": "declared_product_missing",
                    "step_id": step.step_id,
                    "missing_products": [f"{kind}:{name}" for kind, name in missing],
                    "declared_products": [
                        f"{kind}:{name}" for kind, name in sorted(declared)
                    ],
                    "registered_products": [
                        f"{kind}:{name}" for kind, name in sorted(registered)
                    ],
                },
            )
        )

    declares_figure = any(kind == "figure" for kind, _name in declared)
    figure_bundle = _undeclared_figure_bundle(figure_paths)
    if figure_bundle and not declares_figure:
        findings.append(
            ValidationFinding(
                validator="declared_product_contract",
                severity="error",
                message=(
                    f"Step {step.step_id} produced a figure bundle without a "
                    "typed figure product in expected_outputs. Figure rendering "
                    "must remain in its declared figure owner step."
                ),
                detail={
                    "kind": "undeclared_figure_bundle",
                    "step_id": step.step_id,
                    "figure_bundle": figure_bundle,
                },
            )
        )

    if not effect_method_authorized:
        declared_effects = sorted(
            f"{kind}:{name}"
            for kind, name in declared
            if _effect_bearing_name(name)
            and not (kind == "log" and _auxiliary_log_name(name))
            and not (effect_figure_source_authorized and kind == "figure")
        )
        registered_effects = sorted(
            f"{kind}:{name}"
            for kind, name in registered
            if _effect_bearing_name(name)
            and not (
                kind == "log" and (kind, name) in declared and _auxiliary_log_name(name)
            )
            and not (
                effect_figure_source_authorized
                and kind == "figure"
                and (kind, name) in declared
            )
        )
        summary_effects = _effect_summary_paths(step_summary)
        if declared_effects or registered_effects or summary_effects:
            findings.append(
                ValidationFinding(
                    validator="declared_product_contract",
                    severity="error",
                    message=(
                        f"Step {step.step_id} lacks a closed effect-output contract "
                        "but declared or registered effect-bearing scientific output. "
                        "Use an agent-planned effect-method owner with a typed, "
                        "machine-readable effect result."
                    ),
                    detail={
                        "kind": "unauthorized_effect_product",
                        "step_id": step.step_id,
                        "planned_method": step.method,
                        "declared_effect_products": declared_effects,
                        "registered_effect_products": registered_effects,
                        "summary_effect_paths": summary_effects,
                    },
                )
            )

    return findings


__all__ = [
    "bind_declared_figure_products",
    "declared_product_contract_findings",
    "effect_adjustment_family",
    "effect_bearing_name",
    "effect_bearing_product",
    "effect_estimand_tier",
    "effect_measure_family",
    "effect_role_family",
    "typed_product",
]
