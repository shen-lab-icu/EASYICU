"""Synthetic run and retained-literature artifacts for preview contracts."""
from __future__ import annotations
import hashlib
import json
import struct
import zlib
from pathlib import Path
from typing import Any, Dict, List


LOG_ROW = "stay_id=30000123, glucose=212"

SECRET = "sk-" + "Q" * 24

BASELINE_CODE = (
    "import pandas as pd\n\n"
    "cohort = pd.read_parquet(INPUT)\n"
    "table = cohort.groupby('glucose_group').size()\n"
    "print(table)\n"
)

FIRST_DRAFT = "rate = cohort['death'].mean()\n"

REPAIRED_CODE = (
    "source = '/Users/someone/private/cohort.parquet'\n"
    f"key = '{SECRET}'\n"
    "rate = cohort.groupby('glucose_group')['death'].mean()\n"
)

def _png() -> bytes:
    def chunk(kind: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(kind + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", crc)

    header = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(b"\x00\xff\xff\xff"))
        + chunk(b"IEND", b"")
    )

def _at(seconds: int) -> str:
    return f"2026-01-01T00:{seconds // 60:02d}:{seconds % 60:02d}+00:00"

def _audit(seconds: int, phase: str, event: str, **extra: Any) -> Dict[str, Any]:
    return {
        "timestamp": _at(seconds),
        "phase": phase,
        "event": event,
        "status": extra.pop("status", "running"),
        "step_id": extra.pop("step_id", None),
        "detail": extra,
    }

AUDIT = [
    _audit(0, "runtime", "Execution runtime validated before planning.", status="complete"),
    _audit(1, "context", "Research context built.", status="complete"),
    _audit(2, "context", "Exposure groups staged: glucose_group."),
    _audit(3, "hypothesis", "Building pre-plan literature and hypothesis blueprint."),
    _audit(10, "planning", "Study structure validation attempt 1/3 started."),
    _audit(14, "planning", "Study structure attempt 1/3 did not satisfy its contract; retrying.",
           validation_issues=["internal planner wording"]),
    _audit(15, "planning", "Study structure validation attempt 2/3 started."),
    _audit(19, "planning", "Study structure attempt 2/3 passed validation."),
    _audit(20, "plan", "Analysis plan ready with 4 step(s).", status="complete"),
    _audit(21, "step", "Skipped step already completed by pre-execution: cohort_accounting.",
           status="complete", step_id="cohort_accounting"),
    _audit(22, "step", "Step 2/4 started: baseline_table.", step_id="baseline_table"),
    _audit(23, "coder", "Using planner-specified grouped Table 1 executor for baseline_table.",
           step_id="baseline_table"),
    _audit(24, "runner", "Running standard executor script for baseline_table.",
           step_id="baseline_table"),
    _audit(30, "step", "Step 2/4 complete: baseline_table.", status="complete",
           step_id="baseline_table"),
    _audit(31, "step", "Step 3/4 started: outcome_distribution.", step_id="outcome_distribution"),
    _audit(32, "coder", "Generating analysis script for outcome_distribution.",
           step_id="outcome_distribution"),
    _audit(40, "runner", "Running generated script for outcome_distribution.",
           step_id="outcome_distribution"),
    _audit(41, "runner", f"Script failed: {LOG_ROW}", status="error",
           step_id="outcome_distribution", stderr=LOG_ROW),
    _audit(42, "coder", "Repairing failed script for outcome_distribution.",
           step_id="outcome_distribution", repair_attempts=1),
    _audit(50, "runner", "Running generated script for outcome_distribution.",
           step_id="outcome_distribution", repair_attempts=1),
    _audit(55, "step", "Step 3/4 complete: outcome_distribution.", status="complete",
           step_id="outcome_distribution"),
    _audit(56, "step", "Step 4/4 started: running_step.", step_id="running_step"),
    _audit(57, "coder", "Generating analysis script for running_step.", step_id="running_step"),
]

def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def _write(path: Path, data: bytes | str) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = data.encode("utf-8") if isinstance(data, str) else data
    path.write_bytes(raw)
    return raw

def _synthetic_run(run: Path) -> Path:
    """Same shape as an executed pipeline run; every value is invented."""

    _write(run / "audit_log.jsonl", "\n".join(json.dumps(row) for row in AUDIT) + "\n")
    _write(run / "analysis_plan.json", json.dumps({"steps": [
        {"step_id": step} for step in (
            "cohort_accounting", "baseline_table", "outcome_distribution", "running_step",
            "planned_later",
        )
    ]}))
    literature = _write(run / "preplan_literature_bundle.json", json.dumps({
        "search_provenance": {"search_queries": {
            "pubmed": ["sepsis glucose mortality", "glucose /Users/someone/notes.txt"],
        }},
        "citations": [{"pmid": "1"}, {"pmid": "2"}, {"pmid": "3"}],
    }))
    records: List[Dict[str, Any]] = []

    def register(eid: str, kind: str, step: str | None, name: str, data: bytes, mode: str = "") -> None:
        relative = f"evidence/{eid}__{name}"
        _write(run / relative, data)
        record = {"evidence_id": eid, "kind": kind, "relative_path": relative,
                  "sha256": _sha(data), "produced_by_step": step}
        if mode:
            record["generation_mode"] = mode
        records.append(record)

    register("preplan_literature_bundle", "log", None, "preplan_literature_bundle.json", literature)
    _write(run / "steps/cohort_accounting/outputs/cohort.parquet", b"PAR1 synthetic rows PAR1")

    baseline = run / "steps/baseline_table"
    code = _write(baseline / "analysis.py", BASELINE_CODE)
    _write(baseline / "run.log", f"loaded cohort\n{LOG_ROW}\nTraceback: none\n")
    table = _write(baseline / "outputs/table_one.csv", "variable,group,n\nage,high,12\n")
    figure = _write(baseline / "outputs/figure.png", _png())
    _write(baseline / "outputs/figure.svg", "<svg xmlns='http://www.w3.org/2000/svg'/>")
    _write(baseline / "outputs/cohort.parquet", b"PAR1 synthetic rows PAR1")
    _write(baseline / "outputs/step_summary.json", json.dumps({
        "deterministic_standard_analysis": "grouped_table_one",
        "headline": {"en": "Baseline table built for 3 groups.", "zh": "已生成 3 组的基线特征表。"},
    }))
    register("code_analysis_baseline", "code", "baseline_table", "analysis.py", code,
             "deterministic_standard")
    register("table_baseline", "table", "baseline_table", "table_one.csv", table)
    register("figure_baseline", "figure", "baseline_table", "figure.png", figure)

    outcome = run / "steps/outcome_distribution"
    repaired = _write(outcome / "analysis.py", REPAIRED_CODE)
    _write(outcome / "run.log", f"{LOG_ROW}\nKeyError: death\n")
    _write(outcome / "outputs/step_summary.json", json.dumps({"status": "complete"}))
    register("code_analysis_outcome_1", "code", "outcome_distribution", "analysis.py",
             FIRST_DRAFT.encode(), "llm")
    register("code_analysis_outcome_2", "code", "outcome_distribution", "analysis.py",
             repaired, "repaired")

    running = run / "steps/running_step"
    _write(running / "analysis.py", "rows = 1\n")
    _write(running / "outputs/step_summary.json", json.dumps({
        "headline": {"en": "Not finished yet.", "zh": "尚未完成。"},
    }))
    _write(running / "outputs/partial.csv", "a\n1\n")
    _write(run / "evidence/evidence_index.json", json.dumps(records))
    return run

def _failed_replay_wrapper(root: Path) -> Path:
    wrapper = root / "run_job1"
    _synthetic_run(wrapper / "pipeline" / "run_inner")
    _write(wrapper / "run_context.json", json.dumps({
        "run_id": "run_job1", "study_id": "study1", "engine": "easyicu.research_agent.pipeline",
    }))
    _write(wrapper / "source_run_manifest.json", json.dumps({
        "run_id": "run_job1", "status": "failed", "engine": "easyicu.research_agent.pipeline",
    }))
    return wrapper

def _literature_source(*, filler: int = 0, summary: str = "This source describes an ICU cohort.") -> bytes:
    return json.dumps({
        "research_question": "A synthetic research question",
        "citations": [{
            "key": "paper_1", "title": "Synthetic study", "year": "2025",
            "pmid": "12345", "relevance": "x" * filler,
        }],
        "screening_decisions": [{
            "citation_key": "paper_1", "source": "pubmed", "disposition": "include",
            "evidence_role": "database", "rationale": "A synthetic source candidate.",
        }],
        "design_evidence_cards": [{
            "citation_key": "paper_1", "evidence_role": "database", "access_mode": "abstract",
            "abstract_locator": "https://pubmed.ncbi.nlm.nih.gov/12345/",
            "abstract_sha256": "a" * 64, "supplement_status": "unknown",
            "reviewed_at": "2026-10-29T12:00:00Z",
            "evidence": [{"dimension": "study_population", "review_role": "data_source",
                          "source_backed_summary": summary, "locator": "Abstract"}],
        }],
    }).encode()
