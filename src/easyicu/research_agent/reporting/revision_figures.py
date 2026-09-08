"""Revision-owned figure exports from unchanged registered aggregate inputs.

This owner can refresh presentation through a known deterministic renderer.
It cannot run analysis code or add evidence to the sealed research run.
"""

from dataclasses import dataclass, replace
import hashlib
import inspect
import json
from pathlib import Path

import pandas as pd

from ..authority.runtime_artifacts import current_step_records, verified_run_evidence_path
from ..authority.evidence_store import evidence_artifact_basename_stem
from ..execution.runners.exposure_outcome_distribution_render import (
    exposure_outcome_distribution_figure_owns_step,
    run_exposure_outcome_distribution_figure,
)
from ..execution.runners.missingness_measurement_figure_executor import (
    missingness_measurement_figure_executor_owns_step, run_missingness_measurement_figure,
)
from ..execution.runners.deterministic_missingness import measurement_audit_product_filename
from ..contracts.figure_plan import resolve_data_quality_figure_inputs
from ..figures.publication import FigureContract
from .manuscript_figures import ManuscriptFigures, build_manuscript_figures
from .manuscript_labels import source_bound_manuscript_labels
from .registered_report_inputs import ReadOnlyReportEvidence
from .writer_only_migration import WriterOnlyMigrationError


@dataclass(frozen=True)
class RevisionFigureBundle:
    root: Path
    pdf: ManuscriptFigures
    png: ManuscriptFigures
    receipt: dict
    receipt_sha256: str


def _fail(message):
    raise WriterOnlyMigrationError(code="REPORT_FIGURE_REVISION_INVALID", detail=message)


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_revision_figure_bundle(bundle: RevisionFigureBundle, *, source_root=None, revision_output=None) -> None:
    """Both consumers must use the same unchanged revision receipt and files."""
    path = bundle.root / "figure_revision_receipt.json"
    if path.is_symlink() or _digest(path) != bundle.receipt_sha256:
        _fail("Figure revision receipt changed")
    if json.loads(path.read_text()) != bundle.receipt:
        _fail("Figure revision receipt differs from its in-memory projection")
    if bundle.pdf.as_receipt() != bundle.receipt["pdf"] or bundle.png.as_receipt() != bundle.receipt["png"]:
        _fail("Figure projections differ from the bound receipt")
    if revision_output is not None and bundle.root != revision_output / "figures":
        _fail("Figure bundle belongs to another report revision")
    if source_root is not None and (
        bundle.receipt["source_run_id"] != source_root.name
        or bundle.receipt["source_plan_sha256"] != _digest(source_root / "analysis_plan.json")
        or bundle.receipt["source_context_sha256"] != _digest(source_root / "research_context.json")
    ):
        _fail("Figure bundle belongs to another research input revision")
    for entry in bundle.receipt["entries"]:
        for name, expected in entry["files"].items():
            path = bundle.root / name
            if path.is_symlink() or path.parent.is_symlink() or _digest(path) != expected:
                _fail("Figure revision export changed")


def _source_owner(figure, records):
    """Follow explicit same-byte promotion links, never a name/role guess."""
    if figure.produced_by_step:
        return figure
    candidates = [r for r in records if r.evidence_id in figure.inputs
                  and r.kind == "figure" and r.sha256 == figure.sha256
                  and r.produced_by_step]
    if len(candidates) > 1:
        _fail("Promoted figure has ambiguous source ownership")
    return candidates[0] if candidates else None


def _render_inputs(step, owner, execution, records, root, *, audit=False):
    """Rebind only the exact aggregates consumed by the completed figure step."""
    rows = [r for r in execution if r.get("step_id") == step.step_id]
    if len(rows) != 1 or rows[0].get("status") != "ok":
        _fail("Figure step has no unique successful execution")
    source_ids = rows[0].get("resolved_input_evidence_ids", [])
    if set(source_ids) != set(owner.inputs):
        _fail("Figure execution and source evidence membership differ")
    sources = [r for r in records if r.evidence_id in source_ids]
    if len(sources) != len(source_ids):
        _fail("Figure input is no longer current")
    inputs = {}
    for key in step.inputs:
        kind, product = key.split(":", 1)
        filename = measurement_audit_product_filename(product) if audit else None
        stems = {product, Path(filename).stem} if filename else {product}
        candidates = [r for r in sources if r.kind == kind == "table"
                      and evidence_artifact_basename_stem(Path(r.relative_path), r.evidence_id) in stems]
        if len(candidates) != 1:
            _fail("Declared figure input has no unique current registered table")
        record = candidates[0]
        path = verified_run_evidence_path(root, record)
        if path is None or path.suffix != ".csv":
            _fail("Registered aggregate table changed")
        contracts = [c for c in step.input_consumption_contracts if c.input_key == key]
        if len(contracts) != 1 or contracts[0].mode != "all_rows":
            _fail("Report rendering cannot change input consumption")
        frame = pd.read_csv(path)
        if _digest(path) != record.sha256:
            _fail("Registered aggregate changed during reading")
        identity = dict(input_key=key, declared_kind=kind, product=product,
                        evidence_id=record.evidence_id, sha256=record.sha256)
        inputs[key] = {
            **identity, "evidence_kind": record.kind,
            "relative_path": path.relative_to(root.resolve()).as_posix(),
            "identity_row": identity,
            "product_contract": {"columns": list(frame.columns), "row_count": len(frame)},
            "consumption_contract": {"input_key": key, "mode": "all_rows",
                                     "artifact_sha256": record.sha256, "verified_row_count": len(frame)},
        }
    if len(inputs) != len(sources):
        _fail("Report renderer would omit a declared input")
    return {"step_id": step.step_id, "inputs": inputs}


def build_revision_figure_bundle(*, prepared, output: Path) -> RevisionFigureBundle:
    """Render once per revision and share these exact exports between Web/PDF."""
    evidence = ReadOnlyReportEvidence(prepared.source_run_dir)
    root = evidence.root
    execution = current_step_records(json.loads((root / "manifest.json").read_text())["per_step_records"])
    records = evidence.current_verified_records(execution)
    projections = [build_manuscript_figures(evidence_records=records, run_dir=root, prefer_png=png)
                   for png in (False, True)]
    if any(p.findings or p.omitted_evidence_ids for p in projections):
        _fail("Source figures are not valid for report projection")
    pdf, png = projections
    if len(pdf.figures) != len(png.figures):
        _fail("Web and PDF figure membership differs")
    labels = source_bound_manuscript_labels(prepared.context, prepared.plan.display_labels, include_unlabeled=True)
    directory = output / "figures"
    directory.mkdir(exist_ok=False)
    projected = [[], []]
    entries = []
    for index, pair in enumerate(zip(pdf.figures, png.figures), 1):
        if (pair[0].contract_evidence_id != pair[1].contract_evidence_id
                or pair[0].contract_sha256 != pair[1].contract_sha256 or pair[0].caption != pair[1].caption):
            _fail("Web and PDF figures do not share one source contract")
        source_record = next(r for r in records if r.evidence_id == pair[0].evidence_id)
        owner = _source_owner(source_record, records)
        step = next((s for s in prepared.plan.steps if owner and s.step_id == owner.produced_by_step), None)
        target = directory / f"figure_{index}"
        target.mkdir()
        entry = {"source_exports": {f.evidence_id: f.figure_sha256 for f in pair},
                 "source_contract_id": pair[0].contract_evidence_id,
                 "source_contract_sha256": pair[0].contract_sha256,
                 "mode": "preserved_export", "exports": {}}
        renderer = None
        options = {}
        if step and exposure_outcome_distribution_figure_owns_step(step):
            binding = _render_inputs(step, owner, execution, records, root)
            renderer = run_exposure_outcome_distribution_figure
        elif step and (roles := resolve_data_quality_figure_inputs(step.inputs, steps=prepared.plan.steps)):
            binding = _render_inputs(step, owner, execution, records, root, audit=True)
            if not missingness_measurement_figure_executor_owns_step(step, plan=prepared.plan, resolved_bindings=binding["inputs"]):
                _fail("Data-quality rendering contract is incomplete")
            renderer = run_missingness_measurement_figure
            options = {"missingness_input": roles["measurement_missingness"],
                       "process_input": roles["measurement_process"]}
        if renderer:
            product = step.expected_outputs[0].split(":", 1)[1]
            renderer(
                out_dir=target, run_dir=root, resolved_inputs=binding, step_id=step.step_id,
                figure_product=product, display_labels=labels, **options,
            )
            contract_path = target / f"{product}.figure_contract.json"
            contract = FigureContract.model_validate_json(contract_path.read_text())
            if not contract.reader_caption:
                _fail("Re-rendered figure has no explanatory caption")
            entry.update(mode="deterministic_presentation", input_bindings=binding,
                         display_labels=labels, renderer_sha256=_digest(Path(inspect.getfile(renderer))))
            for slot, suffix in enumerate(("pdf", "png")):
                path = target / f"{product}.{suffix}"
                figure = replace(pair[slot], evidence_id=f"report_figure_{index}_{suffix}",
                                 relative_path=path.relative_to(directory).as_posix(),
                                 caption=contract.reader_caption, figure_sha256=_digest(path),
                                 contract_evidence_id=f"report_figure_{index}_contract",
                                 contract_sha256=_digest(contract_path))
                projected[slot].append(figure)
                entry["exports"][suffix] = {"evidence_id": figure.evidence_id,
                                            "source_evidence_id": pair[slot].evidence_id}
        else:
            # Unsupported rendering families keep their verified original bytes.
            # They are not silently reinterpreted through another chart family.
            for slot, figure in enumerate(pair):
                source = root / figure.relative_path
                content = source.read_bytes()
                if source.is_symlink() or hashlib.sha256(content).hexdigest() != figure.figure_sha256:
                    _fail("Source figure changed before copying")
                path = target / source.name
                path.write_bytes(content)
                projected[slot].append(replace(figure, relative_path=path.relative_to(directory).as_posix()))
                entry["exports"][source.suffix[1:]] = {"evidence_id": figure.evidence_id,
                                                       "source_evidence_id": figure.evidence_id}
        entry["files"] = {p.relative_to(directory).as_posix(): _digest(p) for p in sorted(target.iterdir()) if p.is_file()}
        entries.append(entry)
    pdf = replace(pdf, figures=tuple(projected[0]))
    png = replace(png, figures=tuple(projected[1]))
    receipt = {"schema_version": "easyicu.report_figure_revision/1", "revision_id": output.name,
               "source_run_id": root.name, "source_plan_sha256": _digest(root / "analysis_plan.json"),
               "source_context_sha256": _digest(root / "research_context.json"),
               "analysis_steps_executed": 0, "entries": entries,
               "pdf": pdf.as_receipt(), "png": png.as_receipt()}
    receipt_path = directory / "figure_revision_receipt.json"
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    return RevisionFigureBundle(directory, pdf, png, receipt, _digest(receipt_path))
