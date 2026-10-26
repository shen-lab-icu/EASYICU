"""Immutable study lineage and digest-bound copies of earlier aggregate results."""
from pathlib import Path

from easyicu.research_agent.authority.evidence_store import EvidenceStore, sha256_of_file
from easyicu.research_agent.authority.runtime_artifacts import verified_run_evidence_path
from easyicu.research_agent.canonical_json import canonical_sha256
from .agent_review_recovery import _write
from .research_append_source import contained, load_append_source, read_json

COMPLETION = "result_append_completion.json"


def seal_append_completion(child):
    files = {name: sha256_of_file(contained(child.wrapper, name)) for name in (
        "result_append_run.json", "result_tables.json", "figure_gallery.json",
        f"pipeline/{child.checkpoint.run_id}/human_review_checkpoint.json",
        f"pipeline/{child.checkpoint.run_id}/append_result.json",
        f"pipeline/{child.checkpoint.run_id}/manifest.json",
    )}
    gallery = child.run_dir / "figure_gallery.json"
    if gallery.exists():
        files[str(gallery.relative_to(child.wrapper))] = sha256_of_file(gallery)
    records = EvidenceStore(child.run_dir).records()
    body = dict(run_id=child.checkpoint.run_id, study_id=child.record["study"]["id"], files=files,
                evidence=[r.model_dump(mode="json") for r in records])
    body["sha256"] = canonical_sha256(body)
    path = child.wrapper / COMPLETION
    if path.exists() and read_json(path) != body:
        raise ValueError("A completed addition is immutable.")
    if not path.exists():
        _write(path, body)
    return body


def verify_completion(child):
    seal = read_json(child.wrapper / COMPLETION)
    if (child.checkpoint.state != "completed" or seal["run_id"] != child.checkpoint.run_id
            or seal["study_id"] != child.record["study"]["id"]
            or seal["sha256"] != canonical_sha256({k: v for k, v in seal.items() if k != "sha256"})):
        raise ValueError("The earlier addition is not a sealed completed result.")
    for name, digest in seal["files"].items():
        if sha256_of_file(contained(child.wrapper, name)) != digest:
            raise ValueError("An earlier addition's result changed.")
    for record in seal["evidence"]:
        if verified_run_evidence_path(child.run_dir, record) is None:
            raise ValueError("An earlier addition's evidence changed.")
    return seal


def resolve_result_source(wrapper, *, run_id, study_id):
    from .research_append_run import load_append_run

    wrapper = Path(wrapper)
    if run_id.startswith("append-"):
        child = load_append_run(wrapper)
        seal = verify_completion(child)
        if seal["run_id"] != run_id or seal["study_id"] != study_id:
            raise ValueError("The source belongs to another study or run.")
        source = load_append_source(Path(child.record["source_wrapper"]))
        verify_predecessor(child)
        predecessor = dict(wrapper=str(wrapper), run_id=run_id, sha256=seal["sha256"])
    else:
        source = load_append_source(wrapper)
        predecessor = None
        if source.seal["run_id"] != run_id:
            raise ValueError("The source belongs to another study or run.")
    if source.seal["study_id"] != study_id:
        raise ValueError("The source belongs to another study or run.")
    return source, predecessor


def verify_predecessor(child):
    from .research_append_run import load_append_run

    # Bound the walk and reject cycles before following any further link.
    seen = {child.checkpoint.run_id}
    parent_dir = child.wrapper.parent.resolve()
    while prior := child.record.get("predecessor"):
        if len(seen) >= 100 or prior["run_id"] in seen or Path(prior["wrapper"]).parent.resolve() != parent_dir:
            raise ValueError("Invalid study result lineage.")
        seen.add(prior["run_id"])
        parent = load_append_run(Path(prior["wrapper"]))
        seal = verify_completion(parent)
        if (seal["sha256"] != prior["sha256"] or seal["run_id"] != prior["run_id"]
                or seal["study_id"] != child.record["study"]["id"]
                or parent.authority.source_run_sha256 != child.authority.source_run_sha256):
            raise ValueError("The earlier study result changed.")
        child = parent


def adopt_result_tables(source, predecessor, evidence):
    from .agent_pipeline_runs import _table_projection
    from .research_append_run import load_append_run

    if predecessor:
        parent = load_append_run(Path(predecessor["wrapper"]))
        if verify_completion(parent)["sha256"] != predecessor["sha256"]:
            raise ValueError("The earlier study result changed.")
        run_dir, run_id = parent.run_dir, parent.checkpoint.run_id
        tables = read_json(parent.wrapper / "result_tables.json")["tables"]
    else:
        run_dir, run_id = source.run_dir, source.seal["run_id"]
        tables = _table_projection(run_dir, plan=source.plan.model_dump(mode="json"))["tables"]
    registry = {r.evidence_id: r for r in EvidenceStore(run_dir).records()}
    adopted = []
    for table in tables:
        original = registry.get(table["evidence_id"])
        path = verified_run_evidence_path(run_dir, original) if original else None
        if path is None or original.sha256 != table.get("sha256"):
            raise ValueError("The earlier result table changed.")
        origin = table.get("origin") or dict(run_id=run_id, evidence_id=original.evidence_id,
                                            sha256=original.sha256, source_run_sha256=source.digest)
        record = evidence.register_file(kind="table", description=original.description, source_path=path,
            evidence_id="prior_" + canonical_sha256(origin)[:32], produced_by_step=original.produced_by_step,
            metadata={"origin": origin})
        if record.sha256 != original.sha256:
            raise ValueError("The earlier result copy changed.")
        adopted.append({**table, "evidence_id": record.evidence_id, "origin": origin})
    return adopted


def adopt_result_figures(source, predecessor, evidence):
    from .research_append_run import load_append_run

    if predecessor:
        parent = load_append_run(Path(predecessor["wrapper"]))
        seal = verify_completion(parent)
        if seal["sha256"] != predecessor["sha256"]:
            raise ValueError("The earlier study result changed.")
        run_dir, run_id, records = parent.run_dir, parent.checkpoint.run_id, seal["evidence"]
    else:
        run_dir, run_id, records = source.run_dir, source.seal["run_id"], source.seal["products"].values()
    path = run_dir / "figure_gallery.json"
    if not path.exists():
        return []
    registry = {r["relative_path"]: r for r in records if r["kind"] == "figure"}
    figures = []
    for row in read_json(path).get("figures", []):
        if not row.get("relative_path"):
            continue
        original = registry.get(row["relative_path"])
        path = verified_run_evidence_path(run_dir, original) if original else None
        if path is None:
            raise ValueError("The earlier figure is not sealed evidence.")
        origin = (original.get("metadata") or {}).get("origin") or dict(
            run_id=run_id, evidence_id=original["evidence_id"], sha256=original["sha256"])
        identity = "prior_figure_" + canonical_sha256(origin)[:32]
        record = evidence.register_file(kind="figure", description=original["description"], source_path=path,
            evidence_id=identity, produced_by_step=original.get("produced_by_step"), metadata={"origin": origin})
        if record.sha256 != original["sha256"]:
            raise ValueError("The earlier figure copy changed.")
        figures.append(dict(figure_id=identity, label=original["description"], relative_path=record.relative_path,
                            status="available", caption="Previously reported result; the source evidence is unchanged."))
    return figures


def merge_result_figures(child):
    prior = child.record.get("prior_figures", [])
    if not prior:
        return
    path = child.run_dir / "figure_gallery.json"
    gallery = read_json(path) if path.exists() else {}
    inherited = {row["figure_id"] for row in prior}
    current = [row for row in gallery.get("figures", []) if row.get("figure_id") not in inherited]
    _write(path, {**gallery, "figures": [*prior, *current]})
