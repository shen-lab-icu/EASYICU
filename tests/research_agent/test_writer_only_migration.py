from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from easyicu.research_agent.literature import CitationRecord, LiteratureBundle
from easyicu.research_agent.reporting.manuscript_literature import (
    audit_manuscript_literature,
)
from easyicu.research_agent.reporting.manuscript_quality import (
    audit_manuscript_quality,
)
from easyicu.research_agent.reporting.manuscript_sections import (
    quality_repair_section_keys,
)
from easyicu.research_agent.reporting.writer_only_migration import (
    PreparedWriterOnlyMigration,
    WriterOnlyMigrationError,
    _normalize_claim_token_sentences,
    _repair_abstract_conclusion_boundary,
    _remove_unresolved_evidence_tokens,
    publish_writer_only_result,
    repair_writer_only,
)


def _manuscript(*, leak: bool = False) -> str:
    discussion = "The result remained host-bound." if leak else "The association does not establish causation [@strobe_2007]."
    return f"""# Sepsis status and mortality in an ICU cohort

**Keywords:** sepsis, intensive care, mortality, cohort, epidemiology

## Abstract

**Background:** Sepsis is an important ICU syndrome.

**Methods:** We conducted a retrospective cohort analysis.

**Results:** Sepsis status was associated with mortality.

**Conclusions:** The association requires external validation.

## Introduction

Transparent observational reporting is important [@strobe_2007].

## Methods

### Study design and cohort
We conducted a retrospective ICU cohort study [@strobe_2007].

### Variables
The exposure was sepsis status and the outcome was in-hospital death.

### Statistical analysis
We used logistic regression.

### Software and reproducibility
Analyses used versioned software and registered artifacts.

## Results

### Cohort characteristics
The cohort included eligible ICU stays.

### Primary outcome
The primary outcome was in-hospital death.

### Primary association
Sepsis status was associated with mortality.

### Sensitivity and subgroup analyses
Sensitivity analyses used the prespecified population.

## Discussion

{discussion}

## Limitations

This single-database study remains susceptible to residual confounding.

## Conclusion

The observed association requires external validation.

## Data and code availability

Data and code availability require author verification before submission.

## Funding

Funding information requires author verification before submission.

## Ethics approval

Ethics information requires author verification before submission.

## Conflicts of interest

Conflict-of-interest information requires author verification before submission.

## Supplementary artifact release

The release inventory requires author verification before submission.
"""


def _literature() -> LiteratureBundle:
    return LiteratureBundle(
        research_question="Sepsis and mortality",
        citations=[
            CitationRecord(
                key="strobe_2007",
                title="The STROBE statement",
                year="2007",
            )
        ],
    )


def _source_run(tmp_path: Path, manuscript: str) -> tuple[Path, dict[str, str]]:
    run = tmp_path / "sealed_run"
    run.mkdir()
    contents = {
        "manuscript_scaffold.md": manuscript,
        "research_context.json": "{}",
        "analysis_plan.json": "{}",
        "preplan_literature_bundle.json": "{}",
        "writer_evidence_digest.md": "digest",
    }
    hashes: dict[str, str] = {}
    for name, text in contents.items():
        path = run / name
        path.write_text(text, encoding="utf-8")
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return run, hashes


def _prepared(tmp_path: Path, manuscript: str) -> PreparedWriterOnlyMigration:
    run, hashes = _source_run(tmp_path, manuscript)
    literature = _literature()
    return PreparedWriterOnlyMigration(
        source_run_dir=run,
        original_source_manuscript=manuscript,
        source_manuscript=manuscript,
        migration_draft_path=None,
        migration_draft_sha256=hashlib.sha256(manuscript.encode()).hexdigest(),
        context=None,  # type: ignore[arg-type]
        plan=None,  # type: ignore[arg-type]
        literature=literature,
        literature_digest="digest",
        evidence_digest="digest",
        evidence_ids=(),
        expected_display_labels=(),
        administrative_authority=None,
        source_hashes=hashes,
        source_quality_audit=audit_manuscript_quality(manuscript),
        source_literature_audit=audit_manuscript_literature(manuscript, literature),
        planned_section_keys=quality_repair_section_keys(manuscript),
        removed_unknown_literature_keys=(),
        removed_unknown_literature_sentences=0,
    )


def test_preflight_owner_projection_uses_manuscript_contract() -> None:
    assert quality_repair_section_keys(_manuscript(leak=True)) == ("discussion",)


def test_writer_only_repair_preserves_source_and_publishes_separately(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _prepared(tmp_path, _manuscript(leak=True))
    monkeypatch.setattr(
        "easyicu.research_agent.reporting.writer_only_migration._claim_policy_projection",
        lambda _run, manuscript: (manuscript, {}),
    )
    monkeypatch.setattr(
        "easyicu.research_agent.reporting.writer_only_migration._bind_and_copy_evidence",
        lambda _prepared, manuscript, output_dir: (manuscript, ()),
    )

    class FakeWriter:
        def repair_existing(self, manuscript: str, **_kwargs: object):
            assert "host-bound" in manuscript
            return _manuscript(), ("discussion",)

    result = repair_writer_only(prepared, writer=FakeWriter())
    receipt = publish_writer_only_result(
        prepared,
        result,
        output_dir=tmp_path / "output",
        provider="fake",
        model="fake-writer",
        provider_summary={"n_calls": 1, "by_role": {"writer": {"n_calls": 1}}},
        provider_ledger="runtime/provider_hard_stop.json",
    )

    assert "host-bound" in (prepared.source_run_dir / "manuscript_scaffold.md").read_text()
    assert "host-bound" not in (tmp_path / "output" / "manuscript_scaffold.md").read_text()
    assert receipt["roles_used"] == ["writer"]
    assert receipt["analysis_steps_executed"] == 0
    assert receipt["publication_authorized"] is False


def test_writer_only_failure_never_publishes_replacement(tmp_path: Path) -> None:
    prepared = _prepared(tmp_path, _manuscript(leak=True))

    class FailingWriter:
        def repair_existing(self, _manuscript: str, **_kwargs: object):
            raise RuntimeError("provider failed")

    with pytest.raises(
        WriterOnlyMigrationError,
        match="WRITER_ONLY_REPAIR_FAILED_PRIOR_PRESERVED",
    ):
        repair_writer_only(prepared, writer=FailingWriter())

    assert not (tmp_path / "output" / "manuscript_scaffold.md").exists()


def test_writer_only_accepts_fail_closed_filter_when_contracts_remain_valid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unsupported = "An unsupported current-study statement."
    manuscript = _manuscript().replace(
        "The association does not establish causation [@strobe_2007].",
        (
            "The association does not establish causation [@strobe_2007]. "
            + unsupported
        ),
    )
    prepared = _prepared(tmp_path, manuscript)
    canonical = manuscript.replace(" " + unsupported, "")
    monkeypatch.setattr(
        "easyicu.research_agent.reporting.writer_only_migration._claim_policy_projection",
        lambda _run, _manuscript: (
            canonical,
            {"discussion": (unsupported,)},
        ),
    )

    class FakeWriter:
        def repair_existing(self, value: str, **_kwargs: object):
            return value, ()

        def repair_sections(self, *_args: object, **_kwargs: object):
            raise AssertionError("valid fail-closed projection should not use Provider")

    result = repair_writer_only(prepared, writer=FakeWriter())

    assert unsupported not in result.manuscript
    assert result.authority_filtered_section_keys == ("discussion",)
    assert result.authority_repaired_section_keys == ()


def test_writer_only_owner_has_no_scientific_execution_imports() -> None:
    source = Path(
        "src/easyicu/research_agent/reporting/writer_only_migration.py"
    ).read_text(encoding="utf-8")

    for forbidden in (
        "research_agent.pipeline",
        "research_agent.execution",
        "research_agent.agents.coder",
        "authority.evidence_store",
    ):
        assert forbidden not in source


def test_unresolved_evidence_tokens_are_removed_before_claim_filter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manuscript = (
        "Prior work supports this statement [@strobe_2007] "
        "{evidence:missing_alias}."
    )
    monkeypatch.setattr(
        "easyicu.research_agent.reporting.writer_only_migration._read_only_authority",
        lambda _run: type(
            "Authority",
            (),
            {"records": (), "aliases": {}, "claims_by_ref": {}},
        )(),
    )

    cleaned, refs, count = _remove_unresolved_evidence_tokens(
        tmp_path,
        manuscript,
    )

    assert cleaned == "Prior work supports this statement [@strobe_2007]."
    assert refs == ("missing_alias",)
    assert count == 1


def test_claim_tokens_are_normalized_to_standalone_paragraphs() -> None:
    raw = (
        "**Results:** The cohort included 10 stays. "
        "{claim:step.first} {claim:step.second}\n\n"
        "**Conclusions:** Validation is required [@record_2015]."
    )

    normalized, count = _normalize_claim_token_sentences(raw)

    assert count == 2
    assert (
        "The cohort included 10 stays.\n\n{claim:step.first}\n\n"
        "{claim:step.second}"
    ) in normalized


def test_abstract_conclusion_fallback_is_cited_and_noncausal() -> None:
    raw = _manuscript().replace(
        "**Conclusions:** The association requires external validation.",
        "**Conclusions:** The treatment improved survival {evidence:unsupported}.",
    )

    repaired, changed = _repair_abstract_conclusion_boundary(raw, _literature())

    assert changed is True
    assert "The treatment improved survival" not in repaired
    assert "do not establish causation [@strobe_2007]" in repaired
    assert "validation in other cohorts" in repaired
