"""Project-scoped result actions: offer executable additions, then one review."""
from pathlib import Path
import uuid

from easyicu.research_agent.canonical_json import canonical_sha256
from easyicu.research_agent.planning.family_spec.baseline_association_template import LABELS, compile_result_append
from . import state_paths, study_contexts
from .agent_review_recovery import _locked, _write
from .research_append_run import append_run_for_id, propose_append_run
from .research_append_history import resolve_result_source
from .research_append_source import read_json, registered_source_changed


def _option(kind, *, available=False, detail=""):
    en, zh = LABELS[kind]
    return dict(kind=kind, label_en=en, label_zh=zh, available=available, owner=kind,
                status_en=detail or "Open the analysis settings to prepare this addition.",
                status_zh="当前结果尚不支持此项追加，请进入分析设置完善所需条件。")


def _resolve(service, project_id, run_id):
    row = service._research_run_row(project_id, run_id)
    return resolve_result_source(Path(row["project_dir"]), run_id=run_id, study_id=row["study_id"])


def append_options(service, *, project_id, run_id):
    # Project authorization happens even when the old run has no append seal.
    row = service._research_run_row(project_id, run_id)
    try:
        source, predecessor = resolve_result_source(Path(row["project_dir"]), run_id=run_id, study_id=row["study_id"])
    except (ValueError, OSError, KeyError):
        return dict(run_id=run_id, source_run_sha256=None, options=[_option(kind) for kind in LABELS])
    study = study_contexts.get_context(source.seal["study_id"]) or {}
    options = []
    for kind in LABELS:
        try:
            compiled = compile_result_append(kind=kind, source=source.plan, context=source.context, authority=source.baseline)
            needed = {value for step in compiled.plan.steps for value in step.inputs if ":" in value}
            produced = {value for step in compiled.plan.steps for value in step.expected_outputs}
            if not needed - produced <= source.seal["products"].keys():
                raise ValueError("Required source evidence is unavailable; open the analysis settings.")
            options.append(_option(kind, available=True))
        except ValueError as exc:
            options.append(_option(kind, detail=str(exc)))
    changed = registered_source_changed(source, study)
    return dict(run_id=run_id, source_run_sha256=predecessor["sha256"] if predecessor else source.digest, options=options,
                source_changed=changed, source_notice=source.earlier_export_notice(changed=changed))


def propose_for_conversation(service, *, session_id, project_id, run_id, kind, expected_source_sha256):
    source, predecessor = _resolve(service, project_id, run_id)
    service.host_action_session(session_id, project_id=project_id, study_context_id=source.seal["study_id"])
    if (predecessor["sha256"] if predecessor else source.digest) != expected_source_sha256:
        raise ValueError("The source changed since the action was offered; refresh the results.")
    study = study_contexts.get_context(source.seal["study_id"])
    if not study or study.get("active_job_id"):
        raise ValueError("The study is unavailable or already running an analysis.")
    # A duplicate click or a transport retry returns the same review packet.
    key = canonical_sha256([project_id, session_id, expected_source_sha256, kind])
    root = state_paths.state_root() / "result-appends"
    with _locked(root / (key + ".lock")):
        receipt = root / (key + ".proposal.json")
        child = append_run_for_id(read_json(receipt)["run_id"]) if receipt.exists() else None
        if child is None:
            child = propose_append_run(source=source, kind=kind, wrapper=source.wrapper.parent / ("run_append_" + uuid.uuid4().hex),
                                       study=study, registered_source_changed=registered_source_changed(source, study), predecessor=predecessor)
            _write(receipt, {"run_id": child.checkpoint.run_id})
    service.rebind_session(session_id, project_id=project_id)
    return dict(run_id=child.checkpoint.run_id, study_id=study["id"], human_review_pending=child.checkpoint.state == "pending")
