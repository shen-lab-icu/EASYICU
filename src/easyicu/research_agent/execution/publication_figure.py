"""Publication-figure deterministic execution — the sealed-renderer figure repair.

Extracted from ``pipeline_execute`` (Codex-ordered, Bundle 3) as a real
cross-file core-execution boundary. Holds:

* ``SealedRendererState`` — the mutable value object for the sealed-renderer
  figure-repair state (created per step, passed to the generator; concurrent
  steps never share one).
* ``_deterministic_publication_figure_code`` — the deterministic
  publication-figure generator: it produces the rendering-only adapter code for a
  sealed / distribution-availability figure repair, records the repair, and writes
  the sealed-renderer provenance into ``step_record`` + ``SealedRendererState``.
  It is a rendering-only adapter over already-authorized typed parent outputs — it
  selects NO exposure, outcome, cohort, model, or estimand.
* the four sealed-renderer digest/product helpers it uses exclusively
  (``_sealed_renderer_source_digests`` / ``_sealed_renderer_implementation_digest``
  / ``_sealed_parent_planner_anchors`` / ``_sealed_typed_figure_products``).

This module imports neither ``pipeline_execute`` nor ``pipeline``.  Host-owned
authority functions arrive through an immutable service object, while the
generator takes all former ``_execute_one_step`` closure reads as explicit
keyword-only parameters. ``pipeline_execute`` re-exports every public name here
for back-compat.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from ..contracts.declared_product import (
    authorize_declared_figure_product_slots,
    typed_product as _canonical_typed_product,
)
from ..authority.parent_artifact import _resolve_upstream_manifest_step

from ..authority.evidence_store import sha256_of_bytes, sha256_of_file
from .figure_preparation import _step_has_figure_only_output_contract
from .host_services import ExecutePhaseHost, PublicationFigureAuthorityServices
from ..repair_registry import is_sealed_renderer_repair, repair_metadata_for
from ..schema import AnalysisStep, ResearchContext
from .step_worker_state import StepWorkerProgress


class SealedRendererState:
    """Mutable value object for the sealed-renderer figure-repair state.

    The deterministic publication-figure generator produces this state (repair id,
    implementation digest, parent digests, authorized product slots) and the
    downstream visual-revalidation gate consumes it. It replaces four ``nonlocal``
    closure variables in ``_execute_one_step`` so the generator can be lifted out
    of that god function: the generator mutates the object's attributes in place
    (identical semantics to the old nonlocal writes) instead of rebinding closure
    names. Defaults match the old inline initialisers exactly (None / None / {} / {}).
    """

    __slots__ = (
        "repair_id",
        "implementation_sha256",
        "parent_digests",
        "authorized_product_slots",
    )

    def __init__(self) -> None:
        self.repair_id: Optional[str] = None
        self.implementation_sha256: Optional[str] = None
        self.parent_digests: Dict[str, str] = {}
        self.authorized_product_slots: Dict[str, str] = {}


def _sealed_renderer_source_digests(repair_id: str) -> Dict[str, str]:
    """Hash every repository module loaded by an exact sealed renderer."""

    if not is_sealed_renderer_repair(repair_id):
        raise ValueError(f"{repair_id!r} is not an exact sealed renderer")
    metadata = repair_metadata_for(repair_id)
    if not metadata.implementation_modules:
        raise ValueError(f"{repair_id!r} declares no implementation modules")
    digests: Dict[str, str] = {}
    for module_name in metadata.implementation_modules:
        module = importlib.import_module(module_name)
        module_file = getattr(module, "__file__", None)
        if not module_file:
            raise ValueError(f"Cannot locate implementation module {module_name!r}")
        digests[module_name] = sha256_of_file(Path(module_file))
    return dict(sorted(digests.items()))


def _sealed_renderer_implementation_digest(source_digests: Mapping[str, str]) -> str:
    """Return one stable authority digest for a renderer's source modules."""

    payload = json.dumps(
        dict(sorted(source_digests.items())),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256_of_bytes(payload)


def _sealed_parent_planner_anchors(
    *,
    run_dir: Path,
    figure_step_id: str,
) -> tuple[str, ...]:
    """Return only products and inputs from the host-recorded parent request.

    A physical filename or coder summary field can prove bytes and schema, but
    it cannot define the scientific subject that a Planner figure role claims.
    """

    request_step = _resolve_upstream_manifest_step(run_dir, figure_step_id)
    if not isinstance(request_step, Mapping):
        return ()
    anchors: list[str] = []
    for raw in request_step.get("expected_outputs") or []:
        parsed = _canonical_typed_product(raw)
        if parsed is not None:
            anchors.append(f"{parsed[0]}:{parsed[1]}")
    anchors.extend(
        str(raw).strip()
        for raw in (request_step.get("inputs") or [])
        if str(raw).strip()
    )
    return tuple(dict.fromkeys(anchors))


def _sealed_typed_figure_products(
    expected_outputs: Sequence[str],
) -> Optional[List[str]]:
    """Return unique typed figure roles, never legacy bare export filenames."""

    products = [
        str(product).strip() for product in expected_outputs if str(product).strip()
    ]
    typed_roles = [_canonical_typed_product(product) for product in products]
    if (
        not typed_roles
        or any(role is None or role[0] != "figure" for role in typed_roles)
        or len(typed_roles) != len(set(typed_roles))
    ):
        return None
    return products


def _deterministic_publication_figure_code(
    reason: str,
    *,
    run_dir: Path,
    step: AnalysisStep,
    worker_progress: StepWorkerProgress,
    pipeline: ExecutePhaseHost,
    authority_services: PublicationFigureAuthorityServices,
    agent_context: ResearchContext,
    step_record: Dict[str, Any],
    sealed_renderer_state: SealedRendererState,
    _authorize_automatic_repair,
    _record_repair,
) -> Optional[str]:
    exact_repair_id = authority_services.deterministic_repair_id_for_upstream(
        run_dir, step.step_id
    )
    if (
        worker_progress.deterministic_fallback_used
        or not pipeline._enable_deterministic_runner_repair
        or not _step_has_figure_only_output_contract(step)
        or exact_repair_id is None
    ):
        return None
    sealed_renderer = is_sealed_renderer_repair(exact_repair_id)
    declared_figure_products = list(step.expected_outputs or [])
    sealed_source_digests: Dict[str, str] = {}
    sealed_implementation_digest = ""
    sealed_product_slots: Dict[str, str] = {}
    if sealed_renderer:
        typed_products = _sealed_typed_figure_products(declared_figure_products)
        if typed_products is None:
            # Legacy bare exports are file requirements, not logical
            # Planner product roles.  They retain the ordinary coder
            # path rather than entering a sealed binder that cannot
            # prove their semantics.
            return None
        declared_figure_products = typed_products
        try:
            sealed_source_digests = _sealed_renderer_source_digests(exact_repair_id)
            sealed_implementation_digest = _sealed_renderer_implementation_digest(
                sealed_source_digests
            )
        except (ImportError, OSError, ValueError):
            return None
        if not authority_services.sealed_renderer_step_matches_parent(
            run_dir,
            step,
            exact_repair_id,
        ):
            return None
    sealed_parent_digests: Optional[Dict[str, str]] = None
    if sealed_renderer:
        sealed_parent_digests = authority_services.sealed_renderer_parent_digest_seal(
            run_dir,
            step.step_id,
            exact_repair_id,
        )
        if not sealed_parent_digests:
            return None
        try:
            primary_descriptor = (
                agent_context.variable(agent_context.primary_exposure)
                if agent_context.primary_exposure
                else None
            )
            sealed_product_slots = authorize_declared_figure_product_slots(
                declared_products=declared_figure_products,
                renderer_repair_id=exact_repair_id,
                planner_parent_anchors=_sealed_parent_planner_anchors(
                    run_dir=run_dir,
                    figure_step_id=step.step_id,
                ),
                authoritative_display_subjects=(
                    [
                        value
                        for value in (
                            agent_context.primary_exposure,
                            (
                                primary_descriptor.description
                                if primary_descriptor is not None
                                else None
                            ),
                        )
                        if value
                    ]
                ),
            )
        except ValueError:
            return None
    if exact_repair_id == (
        "distribution_availability_publication_bundle_from_parent_outputs_v1"
    ):
        if not authority_services.distribution_availability_step_matches_parent(
            run_dir, step
        ):
            return None
    candidate_code = """
import hashlib
import importlib
import json
import os
from pathlib import Path

out_dir = Path(os.environ["STEP_OUT_DIR"])
run_dir = out_dir.parents[2]
current_step_id = out_dir.parent.name

expected_source_digests = __EXPECTED_SOURCE_DIGESTS__
loaded_modules = {}
actual_source_digests = {}
for module_name, expected_digest in expected_source_digests.items():
    module = importlib.import_module(module_name)
    module_path = Path(module.__file__)
    actual_digest = hashlib.sha256(module_path.read_bytes()).hexdigest()
    loaded_modules[module_name] = module
    actual_source_digests[module_name] = actual_digest
if actual_source_digests != expected_source_digests:
    raise RuntimeError(
        "A sealed renderer implementation module changed after authorization."
    )

pipeline_module = loaded_modules.get("easyicu.research_agent.pipeline")
if pipeline_module is None:
    pipeline_module = importlib.import_module("easyicu.research_agent.pipeline")
expected_repair_id = __EXPECTED_REPAIR_ID__
if __IS_SEALED_RENDERER__:
    render_publication_bundle = getattr(
        pipeline_module,
        "_render_authorized_sealed_publication_bundle",
    )
    repair_id = render_publication_bundle(
        repair_id=expected_repair_id,
        run_dir=run_dir,
        current_step_id=current_step_id,
        out_dir=out_dir,
        parent_artifact_digests=__PREVERIFIED_PARENT_DIGESTS__,
    )
else:
    render_publication_bundle = getattr(
        pipeline_module,
        "_render_publication_bundle_from_prior_outputs_for_step",
    )
    repair_id = render_publication_bundle(
        run_dir=run_dir,
        current_step_id=current_step_id,
        out_dir=out_dir,
        preverified_parent_digests=__PREVERIFIED_PARENT_DIGESTS__,
    )

if repair_id != expected_repair_id:
    summary = {
        "rendering_only": True,
        "deterministic_publication_figure_rescue": "typed_renderer_mismatch",
        "expected_repair_id": expected_repair_id,
        "observed_repair_id": repair_id,
        "figure_files": [],
        "warning": "The evidence-bound renderer did not return its authorized repair id.",
    }
    with open(out_dir / "step_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
else:
    if __IS_SEALED_RENDERER__:
        contract_module = loaded_modules[
            "easyicu.research_agent.contracts.declared_product"
        ]
        bind_declared_figure_products = getattr(
            contract_module,
            "bind_declared_figure_products",
        )
        bind_declared_figure_products(
            out_dir=out_dir,
            declared_products=__DECLARED_FIGURE_PRODUCTS__,
            authorized_product_slots=__AUTHORIZED_PRODUCT_SLOTS__,
            renderer_repair_id=expected_repair_id,
            renderer_implementation_sha256=__IMPLEMENTATION_DIGEST__,
            renderer_parent_digests=__PREVERIFIED_PARENT_DIGESTS__,
        )
    print(json.dumps({"deterministic_publication_figure_rescue": repair_id}))
"""
    candidate_code = candidate_code.replace(
        "__EXPECTED_REPAIR_ID__", repr(exact_repair_id)
    )
    candidate_code = candidate_code.replace(
        "__PREVERIFIED_PARENT_DIGESTS__",
        repr(
            dict(sorted(sealed_parent_digests.items()))
            if sealed_parent_digests is not None
            else None
        ),
    )
    candidate_code = candidate_code.replace(
        "__DECLARED_FIGURE_PRODUCTS__",
        repr(declared_figure_products),
    )
    candidate_code = candidate_code.replace(
        "__AUTHORIZED_PRODUCT_SLOTS__",
        repr(dict(sorted(sealed_product_slots.items()))),
    )
    candidate_code = candidate_code.replace(
        "__EXPECTED_SOURCE_DIGESTS__",
        repr(sealed_source_digests),
    )
    candidate_code = candidate_code.replace(
        "__IS_SEALED_RENDERER__",
        repr(sealed_renderer),
    )
    candidate_code = candidate_code.replace(
        "__IMPLEMENTATION_DIGEST__",
        repr(sealed_implementation_digest),
    )
    repair_id = exact_repair_id
    authorized = _authorize_automatic_repair(
        (repair_id, candidate_code),
        step=step,
        source=reason,
        before_code="",
        sealed_renderer_wrapper=sealed_renderer,
    )
    if authorized is None:
        return None
    worker_progress.deterministic_fallback_used = True
    worker_progress.preexecution_runner_repair_name = repair_id
    if sealed_renderer:
        sealed_renderer_state.repair_id = repair_id
        sealed_renderer_state.implementation_sha256 = sealed_implementation_digest
        sealed_renderer_state.parent_digests = dict(
            sorted((sealed_parent_digests or {}).items())
        )
        sealed_renderer_state.authorized_product_slots = dict(
            sorted(sealed_product_slots.items())
        )
        step_record["sealed_renderer_repair"] = repair_id
        step_record["post_execution_mutation_policy"] = "audit_only"
        step_record["sealed_renderer_source_digests"] = dict(sealed_source_digests)
        step_record["sealed_renderer_implementation_sha256"] = (
            sealed_implementation_digest
        )
        step_record["sealed_renderer_parent_digests"] = dict(
            sealed_renderer_state.parent_digests
        )
        step_record["sealed_renderer_authorized_product_slots"] = dict(
            sealed_renderer_state.authorized_product_slots
        )
        step_record["planner_product_slot_binding_source"] = (
            "planner_parent_typed_product_prefix_v2"
        )
    step_record["deterministic_code_fallback"] = reason
    step_record["runner_repair"] = repair_id
    _record_repair(
        repair_id=repair_id,
        step_id=step.step_id,
        trigger={"source": reason},
        transformation=(
            "Executed a rendering-only adapter over the typed direct "
            "parent outputs; no estimand, cohort, or method was selected."
        ),
        before_code="",
        after_code=candidate_code,
    )
    return authorized[1]


__all__ = [
    "SealedRendererState",
    "_deterministic_publication_figure_code",
    "_sealed_renderer_source_digests",
    "_sealed_renderer_implementation_digest",
    "_sealed_parent_planner_anchors",
    "_sealed_typed_figure_products",
]
