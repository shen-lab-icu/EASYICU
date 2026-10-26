"""One approved deterministic run. No Planner, generated code or report stages."""

from __future__ import annotations

from pathlib import Path
from time import monotonic

from easyicu.research_agent.authority.evidence_store import EvidenceStore
from easyicu.research_agent.methods.descriptive_inputs import DescriptiveInputError
from easyicu.research_agent.methods.table_one import TableOneContractError

from .analysis import descriptive_cells
from .contracts import QuickAnswer, QuickApproval, QuickProposal, QuickStop
from .evidence import project_quick_answer, record_answer
from .planning import card_digest, prepare_quick_answer
from .source import PreparedSource, require_source


def execute_quick_answer(proposal: QuickProposal, *, approval: QuickApproval,
                         source: PreparedSource, provider: object, run_dir: Path) -> QuickAnswer:
    """Called only by the host's user-approval action, never as a model tool.

    Approval is one digest of the full card, including any source substitution.
    The host retains proposal/source objects; browser input supplies only the
    approval fields. A new run directory makes duplicate execution fail closed.
    """
    start = monotonic()
    try:
        card = proposal.card
        if (proposal.status != "approval_required" or card is None
                or approval.card_sha256 != card.sha256 or card_digest(card) != card.sha256):
            raise QuickStop("quick_approval_mismatch")
        require_source(source, provider)
        current = prepare_quick_answer(card.plan.reading, source=source, provider=provider)
        if current != proposal:
            raise QuickStop("quick_plan_changed")
        if card.capability.status == "substitution" and not approval.accept_substitutions:
            raise QuickStop("quick_substitution_approval_required")
        # Exclusive creation is the run's single-execution guard. No partial
        # result is an answer: evidence and all numeric claims must be durable.
        try:
            Path(run_dir).mkdir(parents=True, exist_ok=False)
        except FileExistsError as exc:
            raise QuickStop("quick_run_already_exists") from exc
        cells = descriptive_cells(source, card.plan)
        require_source(source, provider)
        if monotonic() - start > 180:
            raise QuickStop("quick_time_budget_exceeded")
        store = EvidenceStore(run_dir)
        record = record_answer(store, proposal, approval, cells)
        return project_quick_answer(run_dir=run_dir, proposal=proposal,
                                    evidence_id=record.evidence_id, sha256=record.sha256)
    except (DescriptiveInputError, TableOneContractError):
        return QuickAnswer(status="stopped", reason="quick_input_not_testable",
                           source=source.identity.to_dict(), expand=proposal.expand)
    except QuickStop as exc:
        return QuickAnswer(status="stopped", reason=exc.code, source=source.identity.to_dict(),
                           expand=proposal.expand)
    except (OSError, ValueError):
        return QuickAnswer(status="stopped", reason="quick_execution_failed",
                           source=source.identity.to_dict(), expand=proposal.expand)
