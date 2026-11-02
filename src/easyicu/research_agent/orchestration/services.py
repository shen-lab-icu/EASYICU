"""Live collaborators injected into the research-agent pipeline."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Mapping, Optional, Tuple

if TYPE_CHECKING:
    from .runtime_revision import ExecutionRuntimeRevision
    from .scientific_runtime import ScientificRuntimeAuthorities


@dataclass(frozen=True)
class PipelineServices:
    """Non-serializable collaborators used by one pipeline instance.

    Configuration describes what a run is allowed to do. Services are the
    concrete objects that do it: provider clients, review control planes,
    runner factories, and optional plugin registries. Keeping these objects
    outside ``PipelineConfig`` prevents a live connection pool or lock from
    becoming part of a supposedly replayable configuration object.
    """

    llm: Optional[Any] = None
    vlm_client: Optional[Any] = None
    visual_qa_adapter: Optional[Any] = None
    llm_concept_auditor_client: Optional[Any] = None
    human_review_gate: Optional[Any] = None
    runner_factory: Optional[Any] = None
    case_plugin_registry: Optional[Any] = None
    provider_hard_stop: Optional[Any] = None
    deferred_first_observed_consumer: Optional[Any] = None
    deferred_first_observed_late_publisher: Optional[Any] = None
    deferred_first_observed_stage: Optional[Any] = None
    # A host may enable private outline recovery only for a metadata canary.
    metadata_outline_journal_task: Optional[Any] = None
    # Only the trusted host can attest interrupted guarded Provider calls.
    metadata_outline_guard_verifier: Optional[Any] = None
    time_origin_attestor: Optional[Any] = None
    case_duration_attestor: Optional[Any] = None
    case_duration_metadata_attestor: Optional[Any] = None
    approved_case_duration_materializer: Optional[Any] = None
    execution_runtime_revision: Optional[ExecutionRuntimeRevision] = None
    # Host admission control for a fresh repair of a digest-bound reviewed plan.
    plan_revision_guard: Optional[Any] = None

    @classmethod
    def split_legacy_kwargs(
        cls,
        kwargs: Mapping[str, Any],
        *,
        services: Optional["PipelineServices"] = None,
    ) -> Tuple["PipelineServices", Dict[str, Any]]:
        """Separate live collaborators from legacy flat constructor options."""
        remaining = dict(kwargs)
        supplied = {
            field.name: remaining.pop(field.name)
            for field in fields(cls)
            if field.name in remaining
        }
        if services is not None and supplied:
            names = ", ".join(sorted(supplied))
            raise TypeError(
                "Pass live collaborators through PipelineServices only; "
                f"received both services= and legacy option(s): {names}"
            )
        return services or cls(**supplied), remaining

    def model_safe_resume_dir(self, run_dir: Path) -> Path:
        refuse_model_access_to_local_source_flow(run_dir)
        return run_dir

    def deferred_preplan_literature(self, *, context: Any, run_dir: Path, evidence: Any) -> Any:
        """Replay the host-verified prior bundle without retrieval or a model call."""
        from ..literature import LiteratureBundle, manuscript_citable_keys

        bundle = self.deferred_first_observed_stage.reuse_preplan_literature(
            context=context, run_dir=run_dir, evidence=evidence,
        )
        if bundle is None:
            return None, context, [], [], None
        if not isinstance(bundle, LiteratureBundle):
            raise ValueError("deferred_first_observed_literature_invalid")
        comparators = [item.citation_key for item in bundle.screening_decisions
                       if item.disposition == "include" and item.evidence_role == "direct_comparator"]
        return None, context, list(manuscript_citable_keys(bundle)), comparators, bundle

    def compile_deferred_plan(self, *, context: Any, allowed_literature_citation_keys: Any) -> Any:
        """Ask only the host-owned stage to compile its verified saved skeleton."""
        from ..schema import AnalysisPlan
        if self.deferred_first_observed_stage is None:
            raise ValueError("deferred_first_observed_stage_missing")
        plan = self.deferred_first_observed_stage.compile_plan(
            context=context, allowed_literature_citation_keys=allowed_literature_citation_keys,
        )
        if not isinstance(plan, AnalysisPlan):
            raise ValueError("deferred_first_observed_compiled_plan_invalid")
        return plan

    def persist_deferred_progressive_evidence(self, *, context: Any, run_dir: Path,
                                              evidence: Any, prompt_pack_version: str) -> Any:
        if self.deferred_first_observed_stage is not None:
            return self.deferred_first_observed_stage.persist_replayed_progressive_evidence(
                context=context, run_dir=run_dir, evidence=evidence,
                prompt_pack_version=prompt_pack_version,
            )

    def prepare_deferred_review(self, *, context: Any, agent_context: Any, normalized_lineage: Any,
                                run_dir: Path, evidence: Any) -> None:
        if self.deferred_first_observed_stage is not None:
            self.deferred_first_observed_stage.prepare_review(
                context=context, agent_context=agent_context, normalized_lineage=normalized_lineage,
                run_dir=run_dir, evidence=evidence,
            )

    def publish_deferred_source_flow(self, *, run_dir: Path, evidence: Any,
                                     approved_plan_sha256: str, decision_set_sha256: str) -> Any:
        if self.deferred_first_observed_late_publisher is None:
            raise ValueError("deferred_first_observed_late_publisher_missing")
        return self.deferred_first_observed_late_publisher(
            run_dir=run_dir, evidence_store=evidence, approved_plan_sha256=approved_plan_sha256,
            decision_set_sha256=decision_set_sha256,
        )

    def deferred_review_findings(self, *, context: Any, normalized_lineage: Any,
                                 run_dir: Path, evidence: Any) -> list[Any]:
        if self.deferred_first_observed_stage is None:
            return []
        return list(self.deferred_first_observed_stage.review_findings(
            context=context, normalized_lineage=normalized_lineage, run_dir=run_dir, evidence=evidence,
        ))

    def deferred_review_scope(self, *, context: Any) -> Any:
        from contextlib import nullcontext
        if self.deferred_first_observed_stage is None:
            return nullcontext()
        return self.deferred_first_observed_stage.review_scope(context=context)

    def normalize_deferred_review_interval(self, *, plan: Any, context: Any) -> Any:
        if self.deferred_first_observed_stage is None:
            return plan
        return self.deferred_first_observed_stage.normalize_review_interval_text(context=context, plan=plan)

    def bind_plan_dependence(self, *, plan: Any, context: Any) -> Any:
        """Use the stage's verified independence proof only while shaping its plan."""
        from ..planning.dependence_authority import bind_context_dependence_authority

        with self.deferred_review_scope(context=context):
            return bind_context_dependence_authority(plan=plan, context=context)

    def reviewed_input_preparer(self, *, run_dir: Path, binding: Dict[str, Any]) -> Any:
        """Bind the host's verified input consumer to one review lifecycle."""
        from functools import partial
        return partial(prepare_workflow_inputs, run_dir=run_dir,
                       consumer=self.deferred_first_observed_consumer, binding=binding)

    def validate_plan_revision(
        self, *, plan: Any, context: Any,
        before_runtime_binding: Optional[ScientificRuntimeAuthorities] = None,
        run_dir: Path | None = None,
    ) -> None:
        """Compare repairs at the same stage as the reviewed source plan.

        Signed runtime owners compile generic Planner methods and their figure
        bindings. Defer their draft check until that compilation is complete;
        the final call must still check the entire plan before persistence.
        Ordinary drafts retain their early scope check.
        """
        if before_runtime_binding is not None and (
            before_runtime_binding.current_case is not None
            or before_runtime_binding.trajectory is not None
        ):
            return
        receipt = None
        if self.plan_revision_guard is not None:
            receipt = self.plan_revision_guard.validate_plan(plan=plan, context=context)
        from easyicu.extensions.method_attachments import validate_attached_methods

        if before_runtime_binding is None:
            validate_attached_methods(plan=plan, context=context)
        if receipt is not None and run_dir is not None:
            (run_dir / "plan_revision_receipt.json").write_text(receipt.model_dump_json(indent=2) + "\n", encoding="utf-8")

    def canonical_payload(self) -> Dict[str, Optional[str]]:
        """Return type identities only; never serialize live object state."""
        return {
            field.name: (
                None
                if (value := getattr(self, field.name)) is None
                else f"{type(value).__module__}.{type(value).__qualname__}"
            )
            for field in fields(self)
        }


from copy import copy
from dataclasses import dataclass
import hashlib
import json
import re
from typing import Protocol

from ..authority.filesystem import AnchoredDirectory
from ..canonical_json import canonical_sha256
from ..research_context.typed import ResearchContextAuthority, parse_research_context_json

from ..authority.run_input import (
    DEFERRED_RECIPE_NAME as RECIPE_NAME,
    DEFERRED_RECEIPT_NAME as RECEIPT_NAME,
    DEFERRED_ACTUAL_CAPSULE_EVIDENCE_ID as ACTUAL_CAPSULE_EVIDENCE_ID,
    DEFERRED_RECEIPT_EVIDENCE_ID as RECEIPT_EVIDENCE_ID,
)


def refuse_model_access_to_local_source_flow(run_dir: Path) -> None:
    """Refuse model-bearing replay after local-only source-flow publication.

    Check the fixed artifact itself as well as the indexed record: publication
    may have crashed between the atomic file write and the evidence commit.
    """
    import os

    root = Path(run_dir)
    evidence_dir = root / "evidence"
    local_names = ("first_observed_source_flow", "outcome_missing_local")
    targets = tuple(path for name in local_names for path in (
        root / f"{name}.json", evidence_dir / f"{name}__{name}.json"))
    if any(os.path.lexists(path) for path in targets):
        raise RuntimeError("first_observed_source_flow_model_reentry_forbidden")
    if evidence_dir.exists() and any(
        next(evidence_dir.glob(f"{name}_v*__{name}.json"), None) is not None for name in local_names
    ):
        raise RuntimeError("first_observed_source_flow_model_reentry_forbidden")


@dataclass(frozen=True)
class DeferredFirstObservedInput:
    cohort_path: Path
    context: ResearchContextAuthority
    capsule_sha256: str
    receipt_sha256: str


class DeferredFirstObservedConsumer(Protocol):
    def __call__(self, *, run_dir: Path, approved_plan_sha256: str,
                 decision_set_sha256: str, evidence_store: Any) -> DeferredFirstObservedInput: ...


def bind_deferred_first_observed(*, run_dir: Path, plan_result: Any,
        approved_plan_sha256: str, decision_set_sha256: str, recipe_sha256: str,
        consumer: DeferredFirstObservedConsumer | None) -> tuple[Any, Path | None]:
    """Activate only an existing recipe after a digest-bound approval.

    The host verifies its signed receipt and source authorization. Core then
    independently reopens fixed paths and binds the returned input to the
    approved plan and decision. A missing collaborator never authorizes work.
    """
    root = Path(run_dir).absolute()
    with AnchoredDirectory.open(root) as directory:
        absent = directory.is_absent(RECIPE_NAME)
        directory.assert_still_selected()
    if absent and not recipe_sha256:
        return plan_result, None
    if absent or not re.fullmatch(r'[a-f0-9]{64}', recipe_sha256 or ''):
        raise ValueError('deferred_first_observed_recipe_authority_missing')
    with AnchoredDirectory.open(root) as directory:
        directory.read_bytes(RECIPE_NAME, max_bytes=1024 * 1024, expected_sha256=recipe_sha256)
        directory.assert_still_selected()
    if (consumer is None or not re.fullmatch(r'[a-f0-9]{64}', approved_plan_sha256 or '')
            or not re.fullmatch(r'[a-f0-9]{64}', decision_set_sha256 or '')):
        raise ValueError('deferred_first_observed_approval_or_consumer_missing')
    value = consumer(run_dir=root, approved_plan_sha256=approved_plan_sha256,
                     decision_set_sha256=decision_set_sha256, evidence_store=plan_result.evidence)
    if not isinstance(value, DeferredFirstObservedInput):
        raise ValueError('deferred_first_observed_result_invalid')
    expected_cohort = root / 'deferred_input' / 'cohort.parquet'
    if value.cohort_path != expected_cohort:
        raise ValueError('deferred_first_observed_cohort_mismatch')
    with AnchoredDirectory.open(root) as directory:
        receipt = directory.read_bytes(RECEIPT_NAME, max_bytes=1024 * 1024,
                                       expected_sha256=value.receipt_sha256)
        directory.assert_still_selected()
    with AnchoredDirectory.open(root / 'deferred_input') as directory:
        capsule = directory.read_bytes('run_input_capsule.json', max_bytes=16 * 1024 * 1024,
                                       expected_sha256=value.capsule_sha256)
        context_bytes = directory.read_bytes('research_context.json', max_bytes=16 * 1024 * 1024)
        with directory.open_regular('cohort.parquet'):
            pass
        directory.assert_still_selected()
    try:
        body = json.loads(receipt)['body']
        parsed_context = parse_research_context_json(context_bytes)
        context_digest = hashlib.sha256(context_bytes).hexdigest()
        if (body['schema_version'] != 'easyicu.first-observed-deferred-input-receipt/1'
                or body['approved_plan_sha256'] != approved_plan_sha256
                or body['decision_set_sha256'] != decision_set_sha256
                or body['recipe_sha256'] != recipe_sha256
                or body['capsule_sha256'] != value.capsule_sha256
                or body['context_sha256'] != context_digest
                or canonical_sha256(parsed_context.model_dump(mode='json'))
                != canonical_sha256(value.context.model_dump(mode='json'))):
            raise ValueError('deferred_first_observed_receipt_mismatch')
    except (KeyError, TypeError, AttributeError, json.JSONDecodeError) as exc:
        raise ValueError('deferred_first_observed_receipt_invalid') from exc
    del capsule
    # Seal both new authorities alongside, never in place of, the reviewed
    # metadata capsule. An interrupted replay accepts only identical records.
    from ..authority.runtime_artifacts import verified_run_evidence_path

    for evidence_id, source, digest in (
        (ACTUAL_CAPSULE_EVIDENCE_ID, root / 'deferred_input' / 'run_input_capsule.json', value.capsule_sha256),
        (RECEIPT_EVIDENCE_ID, root / RECEIPT_NAME, value.receipt_sha256),
    ):
        existing = plan_result.evidence.get(evidence_id)
        if existing is None:
            plan_result.evidence.register_file(kind='log', description='Approved deferred input authority',
                                               source_path=source, evidence_id=evidence_id)
        elif (str(existing.sha256) != digest
              or verified_run_evidence_path(root, existing) is None):
            raise ValueError('deferred_first_observed_evidence_mismatch')
    result = copy(plan_result)
    result.context = parsed_context
    result.context_path = root / 'deferred_input' / 'research_context.json'
    # Preserve the reviewed and egress-filtered agent context; actual cohort
    # context is available only to the host execution path.
    return result, expected_cohort


def prepare_workflow_inputs(plan_result: Any, plan_sha256: str, decision_sha256: str,
                            recipe_sha256: str, *, run_dir: Path, consumer: Any,
                            binding: Dict[str, Any]) -> Any:
    """Apply the verified host input to this run's shared execute/finalise binding."""
    result, cohort = bind_deferred_first_observed(
        run_dir=run_dir, plan_result=plan_result, approved_plan_sha256=plan_sha256,
        decision_set_sha256=decision_sha256, recipe_sha256=recipe_sha256, consumer=consumer,
    )
    if cohort is not None:
        binding["cohort_path"] = cohort
        binding["deferred_decision_sha256"] = decision_sha256
        binding["deferred_receipt_sha256"] = str(result.evidence.get(RECEIPT_EVIDENCE_ID).sha256)
    return result


def release_reviewed_provider_transport(*, run_dir: Path, binding: Dict[str, Any],
                                        provider_task: Any, evidence: Any) -> None:
    """Open the original ledger only after the reviewed input is durable."""
    decision = binding.get("deferred_decision_sha256")
    if decision is None:
        return
    if provider_task is None:
        raise ValueError("deferred_review_provider_task_missing")
    from .human_review_checkpoint import load_checkpoint
    from ..authority.run_input import verified_deferred_input_capsule_digest
    checkpoint = load_checkpoint(Path(run_dir) / "human_review_checkpoint.json", require_pending=False)
    if checkpoint.state != "executing" or checkpoint.consumed_decision_sha256 != decision:
        raise ValueError("deferred_review_execution_checkpoint_invalid")
    if verified_deferred_input_capsule_digest(run_dir=Path(run_dir), evidence_store=evidence) is None:
        raise ValueError("deferred_review_input_authority_missing")
    receipt = evidence.get(RECEIPT_EVIDENCE_ID)
    if receipt is None or receipt.sha256 != binding.get("deferred_receipt_sha256"):
        raise ValueError("deferred_review_receipt_changed")
    provider_task.release_deferred_review_transport(
        decision_sha256=decision, receipt_sha256=receipt.sha256,
    )


__all__ = ["PipelineServices", "DeferredFirstObservedInput", "DeferredFirstObservedConsumer", "bind_deferred_first_observed"]
