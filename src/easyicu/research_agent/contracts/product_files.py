"""Scientific product formats and the host run delivery/export contract.

The fixed download boundary and planning resolve the same typed outputs here;
delivery promises no scientific result, patient export or publication readiness.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Literal, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, model_validator


FIGURE_SUFFIXES = frozenset({".png", ".svg", ".pdf", ".tif", ".tiff"})
KNOWN_FILE_SUFFIXES = frozenset(
    {
        *FIGURE_SUFFIXES,
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

_AUDIT_PHYSICAL_FILE_KINDS = frozenset({"log", "table"})
_REPORT_PHYSICAL_FILE_KINDS = frozenset({"log"})


def file_kinds(value: object) -> frozenset[str]:
    """Project a physical suffix to the closed product-kind vocabulary."""

    suffix = Path(str(value or "").strip()).suffix.lower()
    if suffix in FIGURE_SUFFIXES:
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


def descriptor_path_is_compatible(*, kind: str, path: str) -> bool:
    """Return whether an exact typed descriptor can use ``path``.

    ``audit`` and ``report`` are semantic roles rather than suffixes.  They are
    authorized only by explicit typed descriptors while their physical payloads
    use ordinary table/log formats; suffix inference alone never creates them.
    """

    physical_kinds = file_kinds(path)
    if kind == "audit":
        return bool(physical_kinds & _AUDIT_PHYSICAL_FILE_KINDS)
    if kind == "report":
        return bool(physical_kinds & _REPORT_PHYSICAL_FILE_KINDS)
    return kind in physical_kinds


def contained_regular_output_file(out_dir: Path, value: object) -> Path:
    """Return one link-free regular file lexically contained in ``out_dir``.

    This helper runs inside the macOS execution sandbox.  Calling
    :meth:`Path.resolve` there asks the kernel to inspect ancestors such as
    ``/private`` that are intentionally outside the sandbox allowlist.  The
    host already supplies the output root, so containment is enforced without
    ancestor traversal. Output-local absolute aliases are first reduced
    lexically to a relative path; outside paths and parent components remain
    invalid. Every output-local component is then checked for symlinks, and
    hard-linked files are rejected.
    """

    root = Path(out_dir)
    raw = str(value or "").strip()
    supplied = Path(raw)
    if supplied.is_absolute():
        try:
            relative = supplied.relative_to(root.absolute())
        except ValueError as exc:
            raise ValueError("output path must be contained") from exc
    else:
        relative = supplied
    if not raw or ".." in relative.parts:
        raise ValueError("output path must be contained")
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


RUN_DELIVERY_SCHEMA_VERSION = "easyicu.run_delivery/1"
HOST_DELIVERY_REASON = "question_requirement_covered_by_host_delivery"
HostDeliveryKind = Literal["article_preview", "manuscript_pdf", "references"]
DeliverableKind = Literal[
    "article_preview", "manuscript_pdf", "result_tables", "figures", "references",
]

# The fixed browser document boundary, previously owned by agent_runs. Keeping
# it here lets planning read the same capabilities without importing webserver.
RUN_DOCUMENT_SPECS = MappingProxyType({
    "manuscript_revision.pdf": {"media_type": "application/pdf", "max_bytes": 16 * 1024 * 1024},
    "manuscript_scaffold.pdf": {"media_type": "application/pdf", "max_bytes": 16 * 1024 * 1024},
    "manuscript_scaffold.tex": {"media_type": "text/x-tex; charset=utf-8", "max_bytes": 2 * 1024 * 1024},
    "manuscript_scaffold.bib": {"media_type": "application/x-bibtex; charset=utf-8", "max_bytes": 2 * 1024 * 1024},
    "system_validation_report.html": {"media_type": "text/html; charset=utf-8", "max_bytes": 8 * 1024 * 1024},
    "system_validation_report.pdf": {"media_type": "application/pdf", "max_bytes": 16 * 1024 * 1024},
})


@dataclass(frozen=True)
class RunDeliverable:
    artifact_names: tuple[str, ...]
    # Compatibility reader for stored requirements predating typed kinds.
    # Only a whole delivery-only request is accepted, never a substring of a
    # scientific requirement. The Planner's prose about feasibility is ignored.
    request_names: str


RUN_DELIVERABLES: Mapping[DeliverableKind, RunDeliverable] = MappingProxyType({
    "article_preview": RunDeliverable(
        ("manuscript_draft.json", "manuscript_provenance.json"),
        r"(?:web|html|browser|online|网页|在线|浏览器)\s*"
        r"(?:(?:article|manuscript|report|文章|论文|报告)\s*)?(?:preview|预览)"
        r"|(?:article|manuscript|report|文章|论文|报告)\s*(?:preview|预览)"
        r"|(?:article|manuscript|report)\s+text|正文",
    ),
    "manuscript_pdf": RunDeliverable(
        ("manuscript_scaffold.pdf", "manuscript_revision.pdf", "manuscript_pdf_receipt.json"),
        r"pdf|typeset copy",
    ),
    "result_tables": RunDeliverable(
        ("result_tables.json",), r"(?:result\s+)?tables?|结果表(?:格)?|表格",
    ),
    "figures": RunDeliverable(
        ("figure_gallery.json",), r"figures?|图表|图件|图形|图片",
    ),
    "references": RunDeliverable(
        ("manuscript_scaffold.bib",),
        r"reference\s+list|references|bibliography|citations?"
        r"|参考文献(?:清单|列表)?|引用(?:清单|列表)?|文献列表",
    ),
})
RUN_DELIVERY_ARTIFACT_NAMES = tuple(dict.fromkeys(
    name for spec in RUN_DELIVERABLES.values() for name in spec.artifact_names
    if name not in RUN_DOCUMENT_SPECS
))

# Neutral delivery grammar only. Unknown words (methods, data requests,
# destinations, formats, etc.) leave a residue and keep ordinary gap handling.
_DELIVERY_VERB = (
    r"(?:\b(?:provide|include|deliver|(?:give|send)\s+me|export|generate|supply|share)\b"
    r"|提供|包含|交付|导出|生成)"
)
_DELIVERY_GRAMMAR = re.compile(
    rf"{_DELIVERY_VERB}|\b(?:please|"
    r"a|an|the|and|or|of|with|downloadable|download|manuscript|article|report|files?|readable)\b"
    r"|(?:请|以及|并|和|及|与|或|可下载|下载|文件|论文|文章|报告|信息充分|清晰|且|可读|的)",
    re.IGNORECASE,
)
_SEPARATORS = re.compile(r"[\s,，、;；。.：:()（）/\-]+")
_DELIVERY_NAMES = {
    kind: re.compile(r"(?<![a-z])(?:" + spec.request_names + r")(?![a-z])", re.IGNORECASE)
    for kind, spec in RUN_DELIVERABLES.items()
}


def requested_deliverable_kinds(quote: str) -> tuple[DeliverableKind, ...]:
    """Translate a complete delivery request to typed kinds, including old records.

    Scientific/mixed clauses and unknown deliveries deliberately resolve to
    nothing, so mentioning a PDF cannot hide an unavailable method or variable.
    """
    remainder = quote
    kinds: list[DeliverableKind] = []
    for kind, pattern in _DELIVERY_NAMES.items():
        remainder, count = pattern.subn(" ", remainder)
        if count:
            kinds.append(kind)
    remainder = _DELIVERY_GRAMMAR.sub(" ", remainder)
    return tuple(kinds) if kinds and not _SEPARATORS.sub("", remainder) else ()


HOST_DELIVERY_KINDS: tuple[HostDeliveryKind, ...] = ("article_preview", "manuscript_pdf", "references")

_DELIVERY_SENTENCE_BOUNDARY = re.compile(r"[。？！?!；;\n]+|\.(?!\d)")
_DELIVERY_PLEASE = r"(?:\bplease\s+|请\s*)"
_DIRECT_NEGATION = (
    rf"(?:\b(?:do\s+not|don['’]t|never|no|not)\b|无需|不必|不要|不需要|不用|(?:不|别|勿)(?={_DELIVERY_VERB}))"
)
_DELIVERY_EXCLUSION = re.compile(
    rf"^(?:{_DELIVERY_PLEASE})?(?:{_DIRECT_NEGATION}|\bwithout\b)\s*", re.IGNORECASE,
)
_DELIVERY_CLAUSE_BOUNDARY = re.compile(
    rf"(?:[,，]\s*(?:(?:\b(?:and|but)\s+|并\s*|但(?:是)?\s*|不过\s*))?"
    rf"|\b(?:and|but)\s+|并\s*|但(?:是)?\s*|不过\s*)"
    rf"(?=(?:{_DELIVERY_PLEASE})?(?:{_DIRECT_NEGATION}|{_DELIVERY_VERB}))"
    rf"|(?=\bwithout\s+(?:(?:a|an|the)\s+)?(?:"
    + "|".join(spec.request_names for spec in RUN_DELIVERABLES.values()) + r")(?![a-z]))",
    re.IGNORECASE,
)
_DELIVERY_COMMAND = re.compile(rf"^(?:{_DELIVERY_PLEASE})?{_DELIVERY_VERB}", re.IGNORECASE)
_DELIVERY_RESET = re.compile(_DELIVERY_PLEASE, re.IGNORECASE)
_DELIVERY_ALTERNATIVE = re.compile(r"\bor\b|或者?|二选一", re.IGNORECASE)

HOST_DELIVERY_CLARIFICATION = "host_delivery_request_ambiguous"
HOST_DELIVERY_CLARIFICATION_MESSAGE = "Choose which deliverables to include on the plan review card."


class DeliveryRequest(Enum):
    REQUESTED = "requested"
    NOT_REQUESTED = "not_requested"
    CHECK = "check"


class PlanDeliverables(BaseModel):
    """The editable selection sealed with a reviewed plan."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    requested: tuple[HostDeliveryKind, ...]
    check: tuple[HostDeliveryKind, ...] = ()
    confirmed: bool
    source: Literal["detector_prefill", "user_edit"]

    @model_validator(mode="after")
    def _consistent(self) -> PlanDeliverables:
        if self.confirmed == bool(self.check) or (self.check and self.source == "user_edit"):
            raise ValueError("A confirmed deliverable set cannot have unresolved choices.")
        kinds = (*self.requested, *self.check)
        if len(kinds) != len(set(kinds)):
            raise ValueError("Each deliverable must have one state.")
        return self


def neutral_delivery_kinds(quote: str) -> tuple[DeliverableKind, ...]:
    """Read only product identities from an entirely delivery-only quote."""
    kinds: set[DeliverableKind] = set()
    for sentence in _DELIVERY_SENTENCE_BOUNDARY.split(quote):
        for fragment in _DELIVERY_CLAUSE_BOUNDARY.split(sentence):
            clause = fragment.strip()
            if not _SEPARATORS.sub("", clause):
                continue
            exclusion = _DELIVERY_EXCLUSION.match(clause)
            if exclusion:
                clause = clause[exclusion.end():]
            matched = requested_deliverable_kinds(clause)
            if not matched:
                return ()
            kinds.update(matched)
    return tuple(kind for kind in RUN_DELIVERABLES if kind in kinds)


def _delivery_object_kinds(clause: str) -> set[DeliverableKind]:
    """Read a neutral object at the start of a delivery command."""
    kinds = set(requested_deliverable_kinds(clause))
    if not kinds and _DELIVERY_COMMAND.match(clause):
        for kind, pattern in _DELIVERY_NAMES.items():
            if any(kind in requested_deliverable_kinds(clause[:name.end()]) for name in pattern.finditer(clause)):
                kinds.add(kind)
    return kinds


def delivery_requests(question: str) -> dict[HostDeliveryKind, DeliveryRequest]:
    """Suggest initial card values from local delivery commands and exclusions."""
    decisions: dict[HostDeliveryKind, DeliveryRequest] = {}
    for sentence in _DELIVERY_SENTENCE_BOUNDARY.split(question):
        negated_delivery = False
        for fragment in _DELIVERY_CLAUSE_BOUNDARY.split(sentence):
            clause = fragment.strip()
            exclusion = _DELIVERY_EXCLUSION.match(clause)
            if exclusion or _DELIVERY_RESET.match(clause):
                negated_delivery = False
            object_text = clause[exclusion.end():] if exclusion else clause
            objects = _delivery_object_kinds(object_text)
            alternative = len(objects) > 1 and _DELIVERY_ALTERNATIVE.search(object_text)
            for kind in HOST_DELIVERY_KINDS:
                if not _DELIVERY_NAMES[kind].search(clause):
                    continue
                if kind not in objects or negated_delivery or (alternative and not exclusion):
                    verdict = DeliveryRequest.CHECK
                elif exclusion:
                    verdict = DeliveryRequest.NOT_REQUESTED
                else:
                    verdict = DeliveryRequest.REQUESTED
                previous = decisions.get(kind, verdict)
                decisions[kind] = verdict if previous is verdict else DeliveryRequest.CHECK
            negated_delivery |= bool(exclusion and objects)
    return {kind: decisions.get(kind, DeliveryRequest.NOT_REQUESTED) for kind in HOST_DELIVERY_KINDS}


HOST_DELIVERY_RECEIPT = "host_delivery_receipt.json"
HOST_DELIVERY_RECEIPT_SCHEMA = "easyicu.host_delivery_receipt/1"
_BOUND_MANUSCRIPT = "manuscript_scaffold_bound.md"
HOST_DELIVERY_SOURCES: Mapping[HostDeliveryKind, tuple[str, ...]] = MappingProxyType({
    "article_preview": (_BOUND_MANUSCRIPT, "manuscript_provenance.json"),
    "manuscript_pdf": (_BOUND_MANUSCRIPT, "manuscript_scaffold.pdf", "manuscript_pdf_receipt.json"),
    "references": ("manuscript_scaffold.bib",),
})


def confirmed_host_delivery_kinds(plan: Mapping[str, object]) -> tuple[HostDeliveryKind, ...]:
    """Read the confirmed selection from an authoritative serialized plan."""
    selection = plan.get("deliverables")
    if not isinstance(selection, Mapping) or selection.get("confirmed") is not True:
        return ()
    requested = selection.get("requested")
    if not isinstance(requested, (tuple, list)):
        return ()
    return tuple(kind for kind in HOST_DELIVERY_KINDS if kind in requested)


def host_delivery_owners(plan: Mapping[str, object]) -> dict[HostDeliveryKind, str]:
    """Read terminal delivery owners from a serialized, authoritative plan."""
    steps = plan.get("steps")
    if not isinstance(steps, list):
        return {}
    # Historical runs sealed their selected deliveries as typed owner steps.
    allowed = confirmed_host_delivery_kinds(plan) if "deliverables" in plan else HOST_DELIVERY_KINDS
    return {
        step["host_delivery_kind"]: step["step_id"]
        for step in steps
        if isinstance(step, Mapping) and step.get("host_delivery_kind") in HOST_DELIVERY_KINDS
        and step["host_delivery_kind"] in allowed
        and isinstance(step.get("step_id"), str) and step["step_id"]
    }


def host_delivery_failure_code(kind: HostDeliveryKind) -> str:
    return f"research_pipeline_{kind}_delivery_unverified"


def read_host_delivery_source(run_dir: Path, name: str) -> bytes:
    limit = RUN_DOCUMENT_SPECS.get(name, {}).get("max_bytes", 8 * 1024 * 1024)
    path = contained_regular_output_file(run_dir, name)
    with path.open("rb") as source:
        raw = source.read(limit + 1)
    if len(raw) > limit or not raw.strip():
        raise ValueError("host_delivery_source_empty_or_oversize")
    return raw


def host_delivery_source_digests(sources: Mapping[str, bytes]) -> dict[str, str]:
    return {name: hashlib.sha256(raw).hexdigest() for name, raw in sources.items()}


def _host_delivery_receipt(run_dir: Path, run_id: str) -> dict:
    try:
        receipt = json.loads(read_host_delivery_source(run_dir, HOST_DELIVERY_RECEIPT))
    except (OSError, ValueError):
        return {}
    if (
        isinstance(receipt, dict) and receipt.get("schema_version") == HOST_DELIVERY_RECEIPT_SCHEMA
        and receipt.get("run_id") == run_id and isinstance(receipt.get("deliveries"), dict)
    ):
        return receipt
    return {}


def recorded_host_delivery_owners(run_dir: Path, *, run_id: str) -> dict[HostDeliveryKind, str]:
    """Identify obligations to report as unverified when the plan is unavailable."""
    receipt = _host_delivery_receipt(run_dir, run_id)
    planned = receipt.get("planned_owners")
    if not isinstance(planned, dict):
        planned = {
            kind: entry.get("step_id") for kind, entry in receipt.get("deliveries", {}).items()
            if isinstance(entry, dict)
        }
    return {
        kind: step_id for kind in HOST_DELIVERY_KINDS
        if isinstance(step_id := planned.get(kind), str) and step_id
    }


def verified_host_deliveries(
    run_dir: Path, *, run_id: str, plan_sha256: str, owners: Mapping[HostDeliveryKind, str],
) -> dict[HostDeliveryKind, dict[str, bytes]]:
    """Recheck the producer's receipt against the run, plan and current files."""
    if not owners:
        return {}
    receipt = _host_delivery_receipt(run_dir, run_id)
    if receipt.get("plan_sha256") != plan_sha256:
        return {}
    verified = {}
    for kind, step_id in owners.items():
        try:
            sources = {name: read_host_delivery_source(run_dir, name) for name in HOST_DELIVERY_SOURCES[kind]}
            if receipt["deliveries"].get(kind) == {
                "step_id": step_id, "sha256_by_artifact": host_delivery_source_digests(sources),
            }:
                verified[kind] = sources
        except (OSError, ValueError):
            continue
    return verified


def host_delivery_coverage(owners: Mapping[DeliverableKind, Sequence[str]]) -> dict[str, object] | None:
    """Describe host delivery backed by a plan's typed product owners."""
    if not owners or not all(steps and kind in RUN_DELIVERABLES for kind, steps in owners.items()):
        return None
    return {
        "source": "host_verified",
        "contract": RUN_DELIVERY_SCHEMA_VERSION,
        "deliverable_kinds": list(owners),
        "artifact_names": list(dict.fromkeys(
            name for kind in owners for name in RUN_DELIVERABLES[kind].artifact_names
        )),
        "owner_step_ids_by_kind": {kind: list(steps) for kind, steps in owners.items()},
        "reason": "The host verified a typed owner for each requested deliverable in this plan.",
    }


__all__ = [
    "DeliverableKind",
    "HOST_DELIVERY_REASON",
    "RUN_DELIVERABLES",
    "RUN_DELIVERY_ARTIFACT_NAMES",
    "RUN_DELIVERY_SCHEMA_VERSION",
    "RUN_DOCUMENT_SPECS",
    "host_delivery_coverage",
    "requested_deliverable_kinds",
    "FIGURE_SUFFIXES",
    "KNOWN_FILE_SUFFIXES",
    "contained_regular_output_file",
    "descriptor_path_is_compatible",
    "file_kinds",
]
