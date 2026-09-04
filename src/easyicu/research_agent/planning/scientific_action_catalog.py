"""Planner-facing scientific actions compiled from the existing ICU registries.

The analysis-type catalog tells the Planner *which family* a question belongs
to.  The method-suite registry tells reviewers *which methods* normally belong
to that family.  Coder resources separately declare reviewed kernels and
packages.  Until this module those three truthful inventories never met before
the Plan was written, so a Planner could not know that (for example) DeLong,
RMST, decision-curve and conformal kernels were available to generated code.

This owner compiles those declarations without upgrading their authority:

* ``host_owned`` means an existing deterministic runner owns the operation;
* ``coder_generated`` means the Coder may implement the declared method and may
  consume the named reviewed resource, but the result remains subject to the
  ordinary value/evidence/scientific gates;
* ``not_available`` is a recognised roadmap item and must fail closed rather
  than be approximated by a nearby estimand.

Package coordinates are declarations, not installation claims.  The runtime
snapshot remains the authority for whether an optional package is present in a
particular runner image.
"""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Literal, Tuple

from ..contracts.method_kernels import CURATED_METHOD_KERNELS
from ..contracts.method_packages import BASELINE_PACKAGES, CURATED_METHOD_PACKAGES
from .analysis_method_suite import AnalysisMethod, get_suite
from .method_adapter_catalog import (
    MethodAdapterContract,
    get_method_adapter_contract,
)
from .analysis_types import (
    CATALOG_DETAIL_LADDER,
    canonical_analysis_family,
    get_analysis_type,
)
from .study_design import study_design_family_for_analysis_type
from .study_design_playbook import StudyDesignFamily

__all__ = [
    "ScientificAction",
    "ScientificActionRuntimeContract",
    "ScientificActionCatalog",
    "ScientificActionGapError",
    "ScientificActionResolution",
    "ReviewedScientificPrimitive",
    "scientific_actions_for_analysis_type",
    "scientific_action_for_id",
    "resolve_scientific_action_request",
    "suggest_scientific_actions",
    "validate_plan_scientific_action_selections",
    "planner_scientific_action_guide",
]

ScientificExecutionMode = Literal[
    "host_owned",
    "coder_generated",
    "not_available",
]


@dataclass(frozen=True)
class ScientificActionRuntimeContract:
    """Exact progressive-plan shape owned by one deterministic adapter.

    The method-suite prose remains useful to reviewers, but it cannot tell the
    compiler which product identity a downstream step may bind.  This small
    contract is the machine-readable boundary: the Planner still chooses the
    action and variables, while the host fixes only its typed result products
    and direct upstream products.
    """

    outputs: Tuple[Tuple[str, str], ...]
    required_product_inputs: Tuple[str, ...] = ()
    article_roles: Tuple[str, ...] = ()
    standard_executor: str = ""


_RUNTIME_CONTRACTS: dict[str, ScientificActionRuntimeContract] = {
    "phenotyping.cluster_solution": ScientificActionRuntimeContract(
        outputs=(
            ("table:phenotype_profiles", "custom"),
            ("table:phenotype_assignments", "custom"),
        ),
        article_roles=("phenotype_structure", "phenotype_profile"),
        standard_executor="cross_sectional_phenotyping",
    ),
    "phenotyping.k_selection": ScientificActionRuntimeContract(
        outputs=(("table:cluster_selection", "custom"),),
        required_product_inputs=("table:phenotype_assignments",),
        article_roles=("cluster_selection",),
        standard_executor="cross_sectional_phenotyping",
    ),
    "phenotyping.cluster_stability": ScientificActionRuntimeContract(
        outputs=(("table:cluster_stability", "custom"),),
        required_product_inputs=("table:phenotype_assignments",),
        article_roles=("stability",),
        standard_executor="cross_sectional_phenotyping",
    ),
    "prediction.discrimination_calibration": ScientificActionRuntimeContract(
        outputs=(
            ("table:prediction_scores", "custom"),
            ("table:model_performance", "custom"),
        ),
        article_roles=("model_performance",),
        standard_executor="prediction_model",
    ),
    "prediction.internal_validation": ScientificActionRuntimeContract(
        outputs=(("table:validation", "custom"),),
        required_product_inputs=("table:prediction_scores",),
        article_roles=("validation",),
        standard_executor="prediction_model",
    ),
    "prediction.calibration_metrics": ScientificActionRuntimeContract(
        outputs=(("table:calibration", "custom"),),
        required_product_inputs=("table:prediction_scores",),
        article_roles=("calibration",),
        standard_executor="prediction_model",
    ),
    "prediction.decision_curve": ScientificActionRuntimeContract(
        outputs=(("table:clinical_utility", "custom"),),
        required_product_inputs=("table:prediction_scores",),
        article_roles=("clinical_utility",),
        standard_executor="prediction_model",
    ),
}


@dataclass(frozen=True)
class ScientificAction:
    """One method the Planner may select, with its honest execution boundary."""

    action_id: str
    analysis_family: StudyDesignFamily
    method_key: str
    name: str
    purpose: str
    tier: str
    execution_mode: ScientificExecutionMode
    produces: str
    runner: str | None
    kernel_imports: Tuple[str, ...] = ()
    software_packages: Tuple[str, ...] = ()
    required_inputs: Tuple[str, ...] = ()
    composition_action_ids: Tuple[str, ...] = ()
    alternative_action_ids: Tuple[str, ...] = ()
    primary_for_analysis_types: Tuple[str, ...] = ()
    notes: str = ""
    method_adapter: MethodAdapterContract | None = None
    runtime_contract: ScientificActionRuntimeContract | None = None

    @property
    def adapter_status(
        self,
    ) -> Literal[
        "full_action",
        "typed_subcontract",
        "supporting_only",
        "none",
    ]:
        """Say whether host code owns the whole action or only supports it."""

        if self.method_adapter is not None:
            return self.method_adapter.scope
        if self.execution_mode == "host_owned":
            return "supporting_only"
        return "none"


@dataclass(frozen=True)
class ReviewedScientificPrimitive:
    """A reviewed Coder resource relevant to the family, not an estimand owner."""

    resource_id: str
    import_name: str
    kind: Literal["kernel", "package"]
    capability: str
    fallback: str
    runtime_verification: Literal[
        "source_digest_bound",
        "runner_snapshot_required",
    ]


@dataclass(frozen=True)
class ScientificActionCatalog:
    """The compiled Planner surface for one canonical analysis type."""

    analysis_type: str
    analysis_family: StudyDesignFamily
    primary_contract_id: str | None
    primary_contract_registered: bool
    actions: Tuple[ScientificAction, ...]
    reviewed_primitives: Tuple[ReviewedScientificPrimitive, ...]
    required_primary_action_ids: Tuple[str, ...]


ScientificActionResolutionStatus = Literal[
    "direct",
    "composed",
    "alternative",
    "unavailable",
]


@dataclass(frozen=True)
class ScientificActionResolution:
    """A deterministic, user-presentable answer for one action request."""

    status: ScientificActionResolutionStatus
    requested_action_id: str
    selected_action_ids: Tuple[str, ...]
    alternative_action_ids: Tuple[str, ...]
    missing_requirements: Tuple[str, ...]
    issue_code: str | None
    requires_user_confirmation: bool
    detail: str

    @property
    def executable(self) -> bool:
        return self.status in {"direct", "composed"}

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": "easyicu.scientific_action_resolution/1",
            "status": self.status,
            "requested_action_id": self.requested_action_id,
            "selected_action_ids": list(self.selected_action_ids),
            "alternative_action_ids": list(self.alternative_action_ids),
            "missing_requirements": list(self.missing_requirements),
            "issue_code": self.issue_code,
            "requires_user_confirmation": self.requires_user_confirmation,
            "detail": self.detail,
        }


class ScientificActionGapError(ValueError):
    """Typed fail-closed gap that downstream UX can render without parsing prose."""

    def __init__(self, resolution: ScientificActionResolution) -> None:
        self.resolution = resolution
        self.issue_code = resolution.issue_code or "scientific_action_gap"
        # ``JobManager`` already understands stable ``code`` values.  The
        # browser-safe payload lets the Web/Pi layer show alternatives and
        # missing inputs as a user decision rather than an opaque traceback.
        self.code = self.issue_code
        self.details = resolution.to_dict()
        self.user_action_required = resolution.to_dict()
        super().__init__(f"{self.issue_code}: {resolution.detail}")


def _execution_mode(method: AnalysisMethod) -> ScientificExecutionMode:
    if method.implementation == "deterministic":
        return "host_owned"
    if method.implementation == "llm_coded":
        return "coder_generated"
    return "not_available"


def _resource_indexes() -> tuple[dict[str, object], dict[str, object]]:
    kernels = {kernel.module: kernel for kernel in CURATED_METHOD_KERNELS}
    packages = {
        **{name: name for name in BASELINE_PACKAGES},
        **{package.import_name: package for package in CURATED_METHOD_PACKAGES},
    }
    return kernels, packages


def _compile_action(
    *,
    family: StudyDesignFamily,
    method: AnalysisMethod,
) -> ScientificAction:
    kernels, packages = _resource_indexes()
    unknown_kernels = sorted(set(method.kernel_modules) - set(kernels))
    unknown_packages = sorted(set(method.software_packages) - set(packages))
    if unknown_kernels or unknown_packages:
        raise ValueError(
            "analysis-method resource binding drift for "
            f"{family}.{method.key}: unknown_kernels={unknown_kernels!r}, "
            f"unknown_packages={unknown_packages!r}"
        )
    action_id = f"{family}.{method.key}"
    runtime_contract = _RUNTIME_CONTRACTS.get(action_id)
    method_adapter = get_method_adapter_contract(action_id)
    return ScientificAction(
        action_id=action_id,
        analysis_family=family,
        method_key=method.key,
        name=method.name,
        purpose=method.purpose,
        tier=method.tier,
        execution_mode=(
            "host_owned"
            if runtime_contract is not None
            or (method_adapter is not None and method_adapter.scope == "full_action")
            else _execution_mode(method)
        ),
        produces=method.produces,
        runner=method.runner,
        kernel_imports=tuple(
            str(kernels[module].import_path)
            for module in method.kernel_modules
        ),
        software_packages=method.software_packages,
        required_inputs=method.required_inputs,
        composition_action_ids=method.composition_action_ids,
        alternative_action_ids=method.alternative_action_ids,
        primary_for_analysis_types=method.primary_for_analysis_types,
        notes=method.notes,
        method_adapter=method_adapter,
        runtime_contract=runtime_contract,
    )


def _reviewed_primitives(
    *,
    analysis_type: str,
    family: StudyDesignFamily,
) -> Tuple[ReviewedScientificPrimitive, ...]:
    relevant_tokens = {analysis_type, family}
    primitives: list[ReviewedScientificPrimitive] = []
    for kernel in CURATED_METHOD_KERNELS:
        if relevant_tokens.isdisjoint(kernel.families):
            continue
        primitives.append(
            ReviewedScientificPrimitive(
                resource_id=f"kernel:{kernel.module}",
                import_name=kernel.import_path,
                kind="kernel",
                capability=kernel.capability,
                fallback=kernel.fallback,
                runtime_verification="source_digest_bound",
            )
        )
    for package in CURATED_METHOD_PACKAGES:
        if relevant_tokens.isdisjoint(package.families):
            continue
        primitives.append(
            ReviewedScientificPrimitive(
                resource_id=f"package:{package.import_name}",
                import_name=package.import_name,
                kind="package",
                capability=package.capability,
                fallback=package.fallback,
                runtime_verification="runner_snapshot_required",
            )
        )
    return tuple(primitives)


def scientific_actions_for_analysis_type(value: str) -> ScientificActionCatalog:
    """Compile one canonical analysis type into its Planner-visible actions.

    An analysis type without a registered primary capability still receives the
    family method inventory so the Planner can describe an honest protocol or
    select an individually owner-backed supporting action.  The catalog states
    that missing primary boundary explicitly; it never borrows the family
    display brief as execution authority.
    """

    analysis_type = canonical_analysis_family(value)
    if analysis_type is None:
        raise ValueError(f"unknown analysis_type {value!r}")
    type_spec = get_analysis_type(analysis_type)
    family = study_design_family_for_analysis_type(analysis_type)
    suite = get_suite(family)
    actions = (
        tuple(_compile_action(family=family, method=method) for method in suite.methods)
        if suite is not None
        else ()
    )
    catalog = ScientificActionCatalog(
        analysis_type=analysis_type,
        analysis_family=family,
        primary_contract_id=type_spec.capability_id,
        primary_contract_registered=bool(type_spec.capability_id),
        actions=actions,
        reviewed_primitives=_reviewed_primitives(
            analysis_type=analysis_type,
            family=family,
        ),
        required_primary_action_ids=tuple(
            action.action_id
            for action in actions
            if analysis_type in action.primary_for_analysis_types
        ),
    )
    action_index = {action.action_id: action for action in catalog.actions}
    for action in catalog.actions:
        related = (*action.composition_action_ids, *action.alternative_action_ids)
        unknown = sorted(set(related) - set(action_index))
        if unknown or action.action_id in related:
            raise ValueError(
                f"scientific action expansion drift for {action.action_id!r}: "
                f"unknown={unknown!r}, self_reference={action.action_id in related}"
            )
    return catalog


def suggest_scientific_actions(
    *,
    analysis_type: str,
    query: str,
    limit: int = 3,
) -> Tuple[str, ...]:
    """Suggest registered ids for UX only; never select or execute them."""

    wanted = str(query or "").strip().lower()
    if not wanted or int(limit) <= 0:
        return ()
    catalog = scientific_actions_for_analysis_type(analysis_type)
    ranked = sorted(
        (
            (
                SequenceMatcher(
                    None,
                    wanted,
                    " ".join(
                        (
                            action.action_id,
                            action.method_key,
                            action.name.lower(),
                            action.purpose.lower(),
                        )
                    ),
                ).ratio(),
                action.action_id,
            )
            for action in catalog.actions
            if action.execution_mode != "not_available"
        ),
        key=lambda item: (-item[0], item[1]),
    )
    return tuple(action_id for _, action_id in ranked[: int(limit)])


def resolve_scientific_action_request(
    *,
    analysis_type: str,
    action_id: str,
) -> ScientificActionResolution:
    """Resolve direct → declared composition → alternatives → explicit gap.

    Alternatives are proposals, not substitutes: the caller must obtain user
    confirmation and create a new Plan action.  No branch infers scientific
    equivalence from names or prose.
    """

    wanted = str(action_id or "").strip()
    catalog = scientific_actions_for_analysis_type(analysis_type)
    action_index = {action.action_id: action for action in catalog.actions}
    action = action_index.get(wanted)
    if action is None:
        alternatives = suggest_scientific_actions(
            analysis_type=analysis_type,
            query=wanted,
        )
        return ScientificActionResolution(
            status="unavailable",
            requested_action_id=wanted,
            selected_action_ids=(),
            alternative_action_ids=alternatives,
            missing_requirements=("registered scientific action descriptor",),
            issue_code="scientific_action_unregistered",
            requires_user_confirmation=bool(alternatives),
            detail=(
                f"scientific_action_id {wanted!r} is not registered for "
                f"analysis_type={catalog.analysis_type!r}. Suggested registered "
                f"actions for review: {list(alternatives)!r}."
            ),
        )
    composition = tuple(
        action_id
        for action_id in action.composition_action_ids
        if action_index[action_id].execution_mode != "not_available"
    )
    if (
        action.execution_mode != "not_available"
        and composition
        and len(composition) == len(action.composition_action_ids)
    ):
        return ScientificActionResolution(
            status="composed",
            requested_action_id=wanted,
            selected_action_ids=(wanted, *composition),
            alternative_action_ids=action.alternative_action_ids,
            missing_requirements=(),
            issue_code=None,
            requires_user_confirmation=False,
            detail=(
                f"{wanted!r} is available as {action.execution_mode} with the "
                f"reviewed supporting composition {list(composition)!r}."
            ),
        )
    if action.execution_mode != "not_available":
        return ScientificActionResolution(
            status="direct",
            requested_action_id=wanted,
            selected_action_ids=(wanted,),
            alternative_action_ids=action.alternative_action_ids,
            missing_requirements=(),
            issue_code=None,
            requires_user_confirmation=False,
            detail=f"{wanted!r} is available as {action.execution_mode}.",
        )

    if composition and len(composition) == len(action.composition_action_ids):
        return ScientificActionResolution(
            status="composed",
            requested_action_id=wanted,
            selected_action_ids=composition,
            alternative_action_ids=action.alternative_action_ids,
            missing_requirements=(),
            issue_code="scientific_action_decomposed",
            requires_user_confirmation=False,
            detail=(
                f"{wanted!r} has an explicitly reviewed equivalent composition: "
                f"{list(composition)!r}."
            ),
        )
    alternatives = tuple(
        action_id
        for action_id in action.alternative_action_ids
        if action_index[action_id].execution_mode != "not_available"
    )
    if alternatives:
        return ScientificActionResolution(
            status="alternative",
            requested_action_id=wanted,
            selected_action_ids=(),
            alternative_action_ids=alternatives,
            missing_requirements=action.required_inputs,
            issue_code="scientific_action_requires_user_choice",
            requires_user_confirmation=True,
            detail=(
                f"{wanted!r} is recognised but unavailable. The following are "
                "different estimands/workflows and require explicit user choice: "
                f"{list(alternatives)!r}."
            ),
        )
    return ScientificActionResolution(
        status="unavailable",
        requested_action_id=wanted,
        selected_action_ids=(),
        alternative_action_ids=(),
        missing_requirements=action.required_inputs
        or ("registered execution owner or reviewed Coder implementation",),
        issue_code="scientific_action_not_available",
        requires_user_confirmation=False,
        detail=(
            f"{wanted!r} is recognised but not available; do not approximate it "
            "with another method or estimand. Missing: "
            f"{list(action.required_inputs) or ['registered execution boundary']!r}."
        ),
    )


def scientific_action_for_id(
    *,
    analysis_type: str,
    action_id: str,
) -> ScientificAction:
    """Resolve an exact action within the declared analysis-type boundary."""

    wanted = str(action_id or "").strip()
    resolution = resolve_scientific_action_request(
        analysis_type=analysis_type,
        action_id=wanted,
    )
    catalog = scientific_actions_for_analysis_type(analysis_type)
    action = next(
        (action for action in catalog.actions if action.action_id == wanted),
        None,
    )
    if action is None or action.execution_mode == "not_available":
        raise ScientificActionGapError(resolution)
    return action


def validate_plan_scientific_action_selections(
    *,
    plan: object,
    inferred_analysis_type: str,
    require_result_actions: bool = False,
) -> None:
    """Fail closed on stale, cross-family or unavailable typed selections.

    Historical plans and non-scientific helper steps may leave the field null.
    The live Planner path sets ``require_result_actions`` so a newly emitted
    result step cannot name a registered canonical method while omitting its
    exact action coordinate.  Free-form methods remain governed by their
    existing capability contracts; this validator never guesses an action from
    approximate prose.

    The family catalog still publishes primary, standard-supporting,
    exploratory, and unavailable methods so each study family can choose its
    appropriate diagnostics. A descriptive study's measurement/data-quality
    support is not interchangeable with survival PH diagnostics, prediction
    calibration, causal overlap, or clustering stability.
    """

    analysis_type = str(getattr(plan, "analysis_type", None) or inferred_analysis_type)
    catalog = scientific_actions_for_analysis_type(analysis_type)
    required_primary = catalog.required_primary_action_ids
    if required_primary:
        primary_steps = tuple(
            step
            for step in getattr(plan, "steps", ())
            if getattr(step, "planned_analysis_role", None) == "primary"
        )
        selected_primary = tuple(
            str(getattr(step, "scientific_action_id", None) or "")
            for step in primary_steps
        )
        if (
            len(primary_steps) != 1
            or not selected_primary
            or selected_primary[0] not in required_primary
        ):
            raise ScientificActionGapError(
                ScientificActionResolution(
                    status="unavailable",
                    requested_action_id=selected_primary[0] if selected_primary else "",
                    selected_action_ids=(),
                    alternative_action_ids=required_primary,
                    missing_requirements=(
                        "one primary step bound to the required typed scientific action",
                    ),
                    issue_code="scientific_action_declaration_required",
                    requires_user_confirmation=False,
                    detail=(
                        f"analysis_type={catalog.analysis_type!r} requires exactly "
                        f"one primary action from {list(required_primary)!r}; "
                        f"received {list(selected_primary)!r}."
                    ),
                )
            )
    for step in getattr(plan, "steps", ()):
        action_id = getattr(step, "scientific_action_id", None)
        if action_id is None:
            outputs = tuple(str(item) for item in getattr(step, "expected_outputs", ()))
            role = str(getattr(step, "planned_analysis_role", "") or "")
            result_outputs = tuple(
                item for item in outputs if not item.startswith(("figure:", "report:"))
            )
            exact_action_ids = tuple(
                action.action_id
                for action in catalog.actions
                if action.method_key == str(getattr(step, "method", "") or "").strip()
            )
            if (
                require_result_actions
                and role in {"primary", "secondary", "sensitivity"}
                and result_outputs
                and exact_action_ids
            ):
                raise ValueError(
                    "scientific_action_required_for_result_step: Planner step "
                    f"{getattr(step, 'step_id', '<unknown>')!r} is a {role!r} "
                    "result-bearing scientific step that selected a registered "
                    f"method and must bind one of {list(exact_action_ids)!r}"
                )
            continue
        selected_action = scientific_action_for_id(
            analysis_type=analysis_type,
            action_id=action_id,
        )
        method_key = str(getattr(step, "method", "") or "").strip()
        exact_method_actions = tuple(
            action.action_id
            for action in catalog.actions
            if action.method_key == method_key
        )
        if (
            exact_method_actions
            and selected_action.action_id not in exact_method_actions
        ):
            raise ValueError(
                "scientific_action_method_mismatch: Planner step "
                f"{getattr(step, 'step_id', '<unknown>')!r} declares method "
                f"{method_key!r} but binds {selected_action.action_id!r}; exact "
                f"method actions are {list(exact_method_actions)!r}"
            )


def planner_scientific_action_guide(
    analysis_type: str,
    *,
    detail: str = "full",
) -> str:
    """Render the inferred family's methods and reviewed resources for Planner.

    ``detail`` follows the existing prompt-budget ladder.  Every rung retains
    every method and its execution status; pressure removes explanation, never
    an unavailable boundary.
    """

    if detail not in CATALOG_DETAIL_LADDER:
        raise ValueError(
            f"unknown scientific-action catalog detail {detail!r}; "
            f"expected one of {CATALOG_DETAIL_LADDER}"
        )
    catalog = scientific_actions_for_analysis_type(analysis_type)
    # This guide is in the fixed portion of every Planner request.  The full
    # method descriptions already live in ``planner_analysis_type_guide``;
    # repeating them here once cost every case ~2 KB.  This projection contains
    # only the typed coordinates needed to write a valid Plan.  The host-side
    # resolver retains the richer purpose/input/composition metadata for gap UX.
    contract = catalog.primary_contract_id or "none"
    if detail == "names_only":
        grouped: dict[tuple[str, str], list[str]] = {}
        for action in catalog.actions:
            grouped.setdefault((action.tier, action.execution_mode), []).append(
                action.action_id
            )
        lines = [
            "ACTIONS exact; gaps fail:",
            f"type={catalog.analysis_type};family={catalog.analysis_family};"
            f"contract={contract}.",
        ]
        if not catalog.primary_contract_registered:
            lines.append("primary execution contract=not_registered.")
        if catalog.required_primary_action_ids:
            lines.append("primary=" + ",".join(catalog.required_primary_action_ids))
        lines.extend(
            f"{tier}/{mode}:{','.join(action_ids)}"
            for (tier, mode), action_ids in grouped.items()
        )
        typed_subcontracts = [
            action.action_id
            for action in catalog.actions
            if action.adapter_status == "typed_subcontract"
        ]
        if typed_subcontracts:
            lines.append("typed_subcontracts:" + ",".join(typed_subcontracts))
        support_only = [
            action.action_id
            for action in catalog.actions
            if action.adapter_status == "supporting_only"
        ]
        if support_only:
            lines.append("host_support_only:" + ",".join(support_only))
        return "\n".join(lines)
    lines = [
        "SCIENTIFIC ACTIONS (inferred family; exact ids):",
        f"type={catalog.analysis_type}; family={catalog.analysis_family}; "
        f"primary_contract={contract}.",
        "scientific_action_id is either null or exactly one of this current "
        "family's ids: "
        + ", ".join(action.action_id for action in catalog.actions)
        + ".",
        "Set scientific_action_id only when a result step selects that exact "
        "action. Cohort-definition, Table 1, raw distribution, and figure-only "
        "support steps do not gain an action id merely because they emit an "
        "artifact; otherwise leave it null. unavailable=fail-closed; alternatives "
        "require user confirmation; never substitute methods, import another family "
        "prefix, or invent ids.",
    ]
    if not catalog.primary_contract_registered:
        lines.append(
            "NO registered primary execution contract; do not claim a completed "
            "family workflow."
        )
    if catalog.required_primary_action_ids:
        lines.append(
            "Required primary action for this analysis type: "
            + ", ".join(catalog.required_primary_action_ids)
            + "."
        )
    grouped: dict[tuple[str, str], list[str]] = {}
    for action in catalog.actions:
        grouped.setdefault((action.tier, action.execution_mode), []).append(
            action.action_id
        )
    for (tier, mode), action_ids in grouped.items():
        lines.append(f"{tier}/{mode}: {','.join(action_ids)}")
    if detail in {"full", "without_guardrails"}:
        for action in catalog.actions:
            entry = f"{action.action_id}: {action.name}"
            if action.adapter_status == "typed_subcontract":
                entry += "; host=partial"
            elif action.adapter_status == "supporting_only":
                entry += "; host=support"
            if detail == "full":
                entry += f"; {action.purpose}"
                resources = (*action.kernel_imports, *action.software_packages)
                if resources:
                    entry += "; resources=" + ",".join(resources)
            lines.append(entry)
    required_actions = tuple(
        action
        for action in catalog.actions
        if action.action_id in catalog.required_primary_action_ids
    )
    if detail == "full":
        for action in required_actions:
            if action.required_inputs:
                lines.append(
                    f"{action.action_id} requires: "
                    + " | ".join(action.required_inputs)
                )
            if action.composition_action_ids:
                lines.append(
                    f"{action.action_id} composes: "
                    + ",".join(action.composition_action_ids)
                )
    return "\n".join(lines)
