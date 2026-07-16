from __future__ import annotations

import json
from pathlib import Path

import pytest

from easyicu.research_agent.discovery_handoff import (
    assert_discovery_analysis_ready,
    build_handoff_from_row,
    select_discovery_row,
    write_handoff_packet,
)
from easyicu.research_agent.discovery_package import (
    _figure_inventory,
    _valid_figure_artifact,
    validate_discovery_manuscript_package,
)
from easyicu.research_agent.discovery_story_figure import render_discovery_story_figure
from easyicu.research_agent.evidence import EvidenceStore


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _register_text_evidence(
    root: Path,
    *,
    evidence_id: str,
    filename: str,
    kind: str,
    text: str,
) -> None:
    EvidenceStore(root).register_text(
        kind=kind,
        description=f"fixture {evidence_id}",
        text=text,
        filename=filename,
        evidence_id=evidence_id,
    )


def _register_handoff_evidence(root: Path, handoff) -> None:
    handoff_path = write_handoff_packet(handoff, root / "discovery_handoff.json")
    EvidenceStore(root).register_file(
        kind="log",
        description="Frozen discovery handoff fixture.",
        source_path=handoff_path,
        evidence_id="discovery_handoff",
        aliases=["discovery_handoff"],
        producer="discovery_launcher",
        generation_mode="human_confirmed",
        metadata={"artifact_role": "discovery_handoff"},
    )


def _write_valid_figure_exports(root: Path, stem: str) -> dict[str, Path]:
    from PIL import Image

    figure_dir = root / "publication_figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    svg = figure_dir / f"{stem}.svg"
    svg.write_text(
        (
            "<svg xmlns='http://www.w3.org/2000/svg' width='20' height='20'>"
            "<rect x='1' y='1' width='18' height='18'/></svg>"
        ),
        encoding="utf-8",
    )
    png = figure_dir / f"{stem}.png"
    Image.new("RGB", (4, 4), color=(20, 80, 140)).save(png, format="PNG")
    return {"svg": svg, "png": png}


def _register_code_backed_figure_bundle(
    root: Path,
    *,
    stem: str,
    contract_payload: dict,
    source_ids: list[str],
) -> dict[str, str]:
    store = EvidenceStore(root)
    script_path = root / f"{stem}_plot.py"
    script_path.write_text("print('deterministic figure')\n", encoding="utf-8")
    script = store.register_file(
        kind="code",
        description="Deterministic figure script fixture.",
        source_path=script_path,
        evidence_id=f"{stem}_script",
        producer="test",
        generation_mode="deterministic_code",
    )
    figure_paths = _write_valid_figure_exports(root, stem)
    contract_path = root / "publication_figures" / f"{stem}.figure_contract.json"
    _write_json(contract_path, contract_payload)
    contract = store.register_file(
        kind="log",
        description="Structured figure contract fixture.",
        source_path=contract_path,
        inputs=source_ids,
        evidence_id=f"{stem}_contract",
        producer="test",
        generation_mode="deterministic_code",
        metadata={
            "artifact_role": "figure_contract",
            "figure_id": stem,
            "source_evidence_ids": source_ids,
        },
    )
    figure_ids: dict[str, str] = {}
    for suffix, path in figure_paths.items():
        record = store.register_file(
            kind="figure",
            description=f"Code-backed {suffix} fixture.",
            source_path=path,
            inputs=[contract.evidence_id, *source_ids],
            script_evidence_id=script.evidence_id,
            evidence_id=f"{stem}_{suffix}",
            producer="test_figure_renderer",
            generation_mode="deterministic_code",
            metadata={
                "artifact_role": "manuscript_figure",
                "figure_id": stem,
                "contract_evidence_id": contract.evidence_id,
                "source_evidence_ids": source_ids,
            },
        )
        figure_ids[suffix] = record.evidence_id
    return {
        "script": script.evidence_id,
        "contract": contract.evidence_id,
        **figure_ids,
    }


def _build_strict_ready_package(root: Path) -> dict[str, str]:
    handoff = build_handoff_from_row(
        {
            "literature_idea_id": "litidea_ok",
            "executable_candidate_id": "execidea_ok",
            "candidate_topic": "definition sensitivity in adult critical care",
            "go_no_go": "recommend",
            "go_no_go_reason": "feasible and differentiated",
            "literature_source": "Critical Care example",
            "resolved_outcome_concept": "death",
        },
        triage_report_path=root / "triage.json",
        human_confirmed=True,
        human_confirmation_note="Reviewed feasibility and approved analysis.",
    )
    _register_handoff_evidence(root, handoff)
    _write_json(
        root / "run_status.json",
        {
            "status": "publication_ready",
            "gates": {
                "execution_complete": True,
                "manuscript_ready": True,
                "publication_ready": True,
            },
        },
    )
    (root / "manuscript_ready.md").write_text(
        "The evidence-bound discovery package is ready.", encoding="utf-8"
    )
    for evidence_id, filename, kind, text in [
        ("cohort_attrition", "cohort_attrition.csv", "table", "stage,n\nall,10\n"),
        ("primary_result", "primary_result.csv", "statistic", "estimate\n1.2\n"),
        ("claim_ledger", "claim_ledger.csv", "log", "claim,evidence\na,b\n"),
    ]:
        _register_text_evidence(
            root,
            evidence_id=evidence_id,
            filename=filename,
            kind=kind,
            text=text,
        )
    source_ids = [
        "discovery_handoff",
        "cohort_attrition",
        "primary_result",
        "claim_ledger",
    ]
    contract_payload = {
        "figure_id": "easyicu_discovery_story",
        "core_claim": "Evidence-bound discovery story.",
        "panels": [
            {
                "panel_id": "A",
                "title": "Mined idea",
                "role": "overview",
                "claim": "A frozen literature idea seeded the analysis.",
                "evidence_ids": ["discovery_handoff"],
                "metadata": {"story_role": "discovery_provenance"},
            },
            {
                "panel_id": "B",
                "title": "Cohort evaluability",
                "role": "cohort_accounting",
                "claim": "The cohort denominator is explicit.",
                "evidence_ids": ["cohort_attrition"],
                "metadata": {"story_role": "cohort_evaluability"},
            },
            {
                "panel_id": "C",
                "title": "Primary result",
                "role": "primary_estimand",
                "claim": "The primary estimate is source-backed.",
                "evidence_ids": ["primary_result"],
                "metadata": {"story_role": "primary_result"},
            },
            {
                "panel_id": "D",
                "title": "Evidence audit",
                "role": "audit",
                "claim": "The claim ledger preserves reproducibility.",
                "evidence_ids": ["claim_ledger"],
                "metadata": {"story_role": "audit_reproducibility"},
            },
        ],
        "source_data": source_ids,
    }
    return _register_code_backed_figure_bundle(
        root,
        stem="easyicu_discovery_story",
        contract_payload=contract_payload,
        source_ids=source_ids,
    )


def test_select_discovery_row_requires_go_or_recommend_for_analysis() -> None:
    rows = [
        {
            "literature_idea_id": "a",
            "candidate_topic": "blocked",
            "go_no_go": "db-cannot-do",
            "go_no_go_reason": "missing concept",
            "novelty_label": "apparently_gap",
        },
        {
            "literature_idea_id": "b",
            "candidate_topic": "definition sensitivity",
            "go_no_go": "hold",
            "go_no_go_reason": "needs human differentiation",
            "novelty_label": "crowded_but_differentiable",
            "differentiators": ["cohort_definition_sensitivity"],
        },
    ]

    with pytest.raises(ValueError, match="no go/recommend"):
        select_discovery_row(rows, require_analysis_ready=True)

    selected = select_discovery_row(rows)
    assert selected["literature_idea_id"] == "b"


def test_discovery_package_accepts_agent_handoff_and_multi_panel_story(tmp_path: Path):
    _build_strict_ready_package(tmp_path)

    assessment = validate_discovery_manuscript_package(run_dir=tmp_path)

    assert assessment.package_ready is True
    assert assessment.status == "article_ready"
    assert assessment.figure_panel_count == 4
    assert assessment.missing_story_roles == []

    index = json.loads(
        (tmp_path / "evidence" / "evidence_index.json").read_text(encoding="utf-8")
    )
    primary_record = next(
        record for record in index if record["evidence_id"] == "primary_result"
    )
    (tmp_path / primary_record["relative_path"]).write_text(
        "estimate\n9.9\n", encoding="utf-8"
    )
    tampered = validate_discovery_manuscript_package(run_dir=tmp_path)
    assert tampered.package_ready is False
    assert tampered.checks["figure_provenance_valid"] is False


def test_discovery_package_rejects_plain_handoff_hash_mismatch(tmp_path: Path):
    _build_strict_ready_package(tmp_path)
    handoff_path = tmp_path / "discovery_handoff.json"
    payload = json.loads(handoff_path.read_text(encoding="utf-8"))
    payload["selection_rationale"] = "Tampered after EvidenceStore registration."
    _write_json(handoff_path, payload)

    assessment = validate_discovery_manuscript_package(run_dir=tmp_path)

    assert assessment.checks["handoff_present"] is True
    assert assessment.checks["handoff_evidence_registered"] is True
    assert assessment.checks["handoff_evidence_hash_match"] is False
    assert assessment.package_ready is False


def test_discovery_package_rejects_tampered_handoff_evidence_copy(tmp_path: Path):
    _build_strict_ready_package(tmp_path)
    index = json.loads(
        (tmp_path / "evidence" / "evidence_index.json").read_text(encoding="utf-8")
    )
    record = next(item for item in index if item["evidence_id"] == "discovery_handoff")
    (tmp_path / record["relative_path"]).write_text("{}", encoding="utf-8")

    assessment = validate_discovery_manuscript_package(run_dir=tmp_path)

    assert assessment.checks["handoff_evidence_registered"] is False
    assert assessment.checks["handoff_evidence_hash_match"] is False
    assert assessment.package_ready is False


@pytest.mark.parametrize("artifact", ["contract", "svg", "png"])
def test_discovery_package_rejects_publication_artifact_tamper(
    tmp_path: Path,
    artifact: str,
):
    _build_strict_ready_package(tmp_path)
    figure_dir = tmp_path / "publication_figures"
    if artifact == "contract":
        contract = figure_dir / "easyicu_discovery_story.figure_contract.json"
        contract.write_text(
            contract.read_text(encoding="utf-8") + "\n", encoding="utf-8"
        )
    elif artifact == "svg":
        svg = figure_dir / "easyicu_discovery_story.svg"
        svg.write_text(
            "<svg xmlns='http://www.w3.org/2000/svg'><circle r='2'/></svg>",
            encoding="utf-8",
        )
    else:
        from PIL import Image

        Image.new("RGB", (5, 5), color=(180, 20, 20)).save(
            figure_dir / "easyicu_discovery_story.png",
            format="PNG",
        )

    assessment = validate_discovery_manuscript_package(run_dir=tmp_path)

    assert assessment.checks["figure_provenance_valid"] is False
    assert assessment.package_ready is False


def test_discovery_package_rejects_magic_only_and_empty_figure_exports(
    tmp_path: Path,
):
    _build_strict_ready_package(tmp_path)
    figure_dir = tmp_path / "publication_figures"
    (figure_dir / "easyicu_discovery_story.svg").write_text(
        "<svg xmlns='http://www.w3.org/2000/svg'/>", encoding="utf-8"
    )
    (figure_dir / "easyicu_discovery_story.png").write_bytes(b"\x89PNG\r\n\x1a\n")

    assessment = validate_discovery_manuscript_package(run_dir=tmp_path)

    assert assessment.checks["figure_artifacts_present"] is False
    assert sorted(assessment.figure_inventory[0].invalid_figure_paths) == [
        "publication_figures/easyicu_discovery_story.png",
        "publication_figures/easyicu_discovery_story.svg",
    ]
    assert assessment.package_ready is False


def test_discovery_package_rejects_figure_symlink_outside_run(tmp_path: Path):
    _build_strict_ready_package(tmp_path)
    outside = tmp_path.parent / f"{tmp_path.name}_outside.svg"
    outside.write_text(
        "<svg xmlns='http://www.w3.org/2000/svg'><rect width='2' height='2'/></svg>",
        encoding="utf-8",
    )
    figure = tmp_path / "publication_figures" / "easyicu_discovery_story.svg"
    figure.unlink()
    figure.symlink_to(outside)

    assessment = validate_discovery_manuscript_package(run_dir=tmp_path)

    assert "publication_figures/easyicu_discovery_story.svg" in (
        assessment.figure_inventory[0].invalid_figure_paths
    )
    assert assessment.package_ready is False


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("script_evidence_id", None),
        ("producer", None),
        ("generation_mode", None),
    ],
)
def test_discovery_package_requires_code_backed_figure_record(
    tmp_path: Path,
    field: str,
    value,
):
    bundle = _build_strict_ready_package(tmp_path)
    store = EvidenceStore(tmp_path)
    figure = next(item for item in store._records if item.evidence_id == bundle["svg"])
    setattr(figure, field, value)
    store._save()

    assessment = validate_discovery_manuscript_package(run_dir=tmp_path)

    assert assessment.figure_inventory[0].unregistered_figure_paths == [
        "publication_figures/easyicu_discovery_story.svg"
    ]
    assert assessment.package_ready is False


def test_discovery_package_requires_primary_source_coverage_in_figure_record(
    tmp_path: Path,
):
    bundle = _build_strict_ready_package(tmp_path)
    store = EvidenceStore(tmp_path)
    figure = next(item for item in store._records if item.evidence_id == bundle["svg"])
    figure.inputs.remove("primary_result")
    figure.metadata["source_evidence_ids"].remove("primary_result")
    store._save()

    assessment = validate_discovery_manuscript_package(run_dir=tmp_path)

    assert assessment.checks["figure_provenance_valid"] is False
    assert assessment.package_ready is False


def test_discovery_package_blocks_single_panel_publication_figure(tmp_path: Path):
    row = {
        "literature_idea_id": "litidea_bad",
        "candidate_topic": "single panel outcome robustness",
        "go_no_go": "hold",
        "go_no_go_reason": "needs differentiation",
    }
    handoff = build_handoff_from_row(
        row,
        triage_report_path=tmp_path / "triage.json",
        target_outcome="death",
    )
    write_handoff_packet(handoff, tmp_path / "discovery_handoff.json")
    _write_json(
        tmp_path / "run_status.json",
        {
            "status": "publication_ready",
            "gates": {
                "execution_complete": True,
                "manuscript_ready": True,
                "publication_ready": True,
            },
        },
    )
    (tmp_path / "manuscript_ready.md").write_text(
        "The exploratory association with death was near the null.",
        encoding="utf-8",
    )
    _write_json(
        tmp_path / "steps" / "04_outcome_gate" / "outputs" / "step_summary.json",
        {
            "step_id": "04_outcome_gate",
            "primary_analysis_authorized": False,
            "grouped_death_analysis_executed": False,
        },
    )
    _write_json(
        tmp_path
        / "publication_figures"
        / "easyicu_publication_figure.figure_contract.json",
        {
            "figure_id": "easyicu_publication_figure",
            "core_claim": "Primary effect and robustness range.",
            "panels": [
                {
                    "panel_id": "A",
                    "title": "Primary effect and robustness variants",
                    "role": "robustness",
                    "claim": "Primary row and variants are shown.",
                    "evidence_ids": ["robustness_panel"],
                }
            ],
            "source_data": ["robustness_panel"],
        },
    )

    assessment = validate_discovery_manuscript_package(run_dir=tmp_path)

    assert assessment.package_ready is False
    assert assessment.status == "manuscript_only"
    assert "discovery_provenance" in assessment.missing_story_roles
    assert assessment.blocked_outcome_steps == ["04_outcome_gate"]
    assert assessment.manuscript_outcome_leak_terms


def test_discovery_story_figure_writes_four_panel_contract(tmp_path: Path):
    row = {
        "literature_idea_id": "litidea_story",
        "candidate_topic": "ventilation practice heterogeneity",
        "go_no_go": "hold",
        "go_no_go_reason": "needs differentiation",
        "literature_source": "Annals of Intensive Care example",
    }
    handoff = build_handoff_from_row(
        row,
        triage_report_path=tmp_path / "triage.json",
        target_outcome="death",
    )
    _register_handoff_evidence(tmp_path, handoff)
    store = EvidenceStore(tmp_path)
    store.register_text(
        kind="table",
        description="Generic cohort accounting fixture.",
        text="stage,n\nall,20\n",
        filename="arbitrary_sample_flow.csv",
        evidence_id="arbitrary_sample_flow",
        aliases=["sample_flow"],
        metadata={"semantic_role": "cohort_attrition"},
    )
    store.register_text(
        kind="statistic",
        description="Generic evaluability fixture.",
        text="concept,coverage\nexposure,0.8\n",
        filename="arbitrary_measurement_coverage.csv",
        evidence_id="arbitrary_measurement_coverage",
        aliases=["measurement_coverage"],
        metadata={"semantic_role": "evaluability"},
    )
    _write_json(
        tmp_path / "run_status.json",
        {
            "status": "analysis_only",
            "gates": {
                "execution_complete": True,
                "manuscript_ready": False,
                "publication_ready": False,
                "missing_evidence_count": 0,
            },
        },
    )
    _write_json(
        tmp_path / "evidence_audit.json",
        {"evidence_count": 12},
    )
    _write_json(
        tmp_path / "numeric_audit.json",
        {"numeric_error_count": 0},
    )

    paths = render_discovery_story_figure(run_dir=tmp_path, handoff=handoff)
    contract = json.loads(paths["contract"].read_text(encoding="utf-8"))

    assert {"svg", "pdf", "png", "tiff", "contract"} <= set(paths)
    assert _valid_figure_artifact(paths["svg"], root=tmp_path) is True
    assert len(contract["panels"]) == 4
    assert contract["panels"][1]["evidence_ids"] == [
        "arbitrary_sample_flow",
        "arbitrary_measurement_coverage",
    ]
    assert "aki" not in json.dumps(contract).lower()
    assessment = validate_discovery_manuscript_package(
        run_dir=tmp_path, require_handoff=False
    )
    assert assessment.figure_panel_count == 4
    assert "primary_result" in assessment.missing_story_roles
    assert assessment.package_ready is False


def test_svg_validator_allows_only_standard_static_doctype(tmp_path: Path):
    safe = tmp_path / "safe.svg"
    safe.write_text(
        """<?xml version="1.0" encoding="utf-8" standalone="no"?>
<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN"
 "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">
<svg xmlns="http://www.w3.org/2000/svg"><rect width="2" height="2"/></svg>
""",
        encoding="utf-8",
    )
    unsafe_system = tmp_path / "unsafe_system.svg"
    unsafe_system.write_text(
        """<!DOCTYPE svg SYSTEM "https://attacker.invalid/evil.dtd">
<svg xmlns="http://www.w3.org/2000/svg"><rect width="2" height="2"/></svg>
""",
        encoding="utf-8",
    )
    unsafe_entity = tmp_path / "unsafe_entity.svg"
    unsafe_entity.write_text(
        """<!DOCTYPE svg [<!ENTITY xxe SYSTEM "file:///etc/passwd">]>
<svg xmlns="http://www.w3.org/2000/svg"><text>&xxe;</text></svg>
""",
        encoding="utf-8",
    )

    assert _valid_figure_artifact(safe, root=tmp_path) is True
    assert _valid_figure_artifact(unsafe_system, root=tmp_path) is False
    assert _valid_figure_artifact(unsafe_entity, root=tmp_path) is False


def test_discovery_analysis_gate_requires_human_confirmation(tmp_path: Path):
    row = {
        "literature_idea_id": "litidea_gate",
        "candidate_topic": "candidate",
        "go_no_go": "recommend",
        "go_no_go_reason": "feasible",
        "resolved_outcome_concept": "death",
    }
    unconfirmed = build_handoff_from_row(
        row, triage_report_path=tmp_path / "triage.json"
    )
    with pytest.raises(ValueError, match="human confirmation"):
        assert_discovery_analysis_ready(unconfirmed)

    confirmed = build_handoff_from_row(
        row,
        triage_report_path=tmp_path / "triage.json",
        human_confirmed=True,
    )
    assert assert_discovery_analysis_ready(confirmed) is True


def test_discovery_handoff_rejects_endpoint_drift(tmp_path: Path):
    row = {
        "literature_idea_id": "litidea_endpoint",
        "candidate_topic": "candidate",
        "go_no_go": "recommend",
        "go_no_go_reason": "feasible",
        "resolved_outcome_concept": "aki",
    }

    with pytest.raises(ValueError, match="conflicts"):
        build_handoff_from_row(
            row,
            triage_report_path=tmp_path / "triage.json",
            target_outcome="death",
            human_confirmed=True,
        )


def test_discovery_package_rejects_unregistered_source_and_invalid_figure(
    tmp_path: Path,
):
    row = {
        "literature_idea_id": "litidea_contract_only",
        "candidate_topic": "candidate",
        "go_no_go": "recommend",
        "go_no_go_reason": "feasible",
        "resolved_outcome_concept": "death",
    }
    handoff = build_handoff_from_row(
        row,
        triage_report_path=tmp_path / "triage.json",
        human_confirmed=True,
    )
    write_handoff_packet(handoff, tmp_path / "discovery_handoff.json")
    _write_json(
        tmp_path / "run_status.json",
        {
            "status": "publication_ready",
            "gates": {
                "execution_complete": True,
                "manuscript_ready": True,
                "publication_ready": True,
            },
        },
    )
    (tmp_path / "manuscript_ready.md").write_text("Ready.", encoding="utf-8")
    figure_dir = tmp_path / "publication_figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    (figure_dir / "contract_only.svg").write_text(
        "<svg xmlns='http://www.w3.org/2000/svg'/>", encoding="utf-8"
    )
    (figure_dir / "contract_only.png").write_bytes(b"not-a-png")
    (tmp_path / "invented_result.csv").write_text("estimate\n9.9\n", encoding="utf-8")
    _write_json(
        tmp_path / "publication_figures" / "contract_only.figure_contract.json",
        {
            "figure_id": "contract_only",
            "core_claim": "Primary result with audit evidence.",
            "panels": [
                {
                    "panel_id": "A",
                    "title": "Primary result",
                    "role": "relationship",
                    "claim": "Result.",
                    "evidence_ids": ["invented_result.csv"],
                }
            ],
            "source_data": ["invented_result.csv"],
        },
    )

    assessment = validate_discovery_manuscript_package(run_dir=tmp_path)

    assert assessment.package_ready is False
    assert assessment.checks["figure_artifacts_present"] is False
    assert assessment.checks["figure_evidence_bound"] is False
    assert assessment.figure_inventory[0].invalid_figure_paths == [
        "publication_figures/contract_only.png",
        "publication_figures/contract_only.svg",
    ]
    assert assessment.figure_inventory[0].provenance_valid is False


def test_discovery_data_foundation_honours_openrouter_provider(monkeypatch):
    import tools.run_discovery_to_manuscript as launcher

    seen = {}

    class FakeClient:
        def __init__(self, **kwargs):
            seen.update(kwargs)

    monkeypatch.setenv("OPENROUTER_API_KEY", "router-key")
    monkeypatch.setenv("OPENROUTER_BASE_URL", "https://router.example/v1")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://wrong-openai.example/v1")
    monkeypatch.setattr(launcher, "OpenAIClient", FakeClient)

    launcher._build_data_foundation_llm(
        provider="openrouter",
        model="provider/model",
        request_timeout=12.0,
    )

    assert seen["api_key"] == "router-key"
    assert seen["base_url"] == "https://router.example/v1"
    assert seen["model"] == "provider/model"


def test_primary_result_requires_table_or_statistic_on_primary_panel(tmp_path: Path):
    _register_text_evidence(
        tmp_path,
        evidence_id="cohort_table",
        filename="cohort.csv",
        kind="table",
        text="n\n10\n",
    )
    _register_text_evidence(
        tmp_path,
        evidence_id="run_status",
        filename="run_status.json",
        kind="log",
        text="{}",
    )
    figure_dir = tmp_path / "publication_figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    (figure_dir / "misbound.svg").write_text(
        "<svg xmlns='http://www.w3.org/2000/svg'/>", encoding="utf-8"
    )
    _write_json(
        figure_dir / "misbound.figure_contract.json",
        {
            "figure_id": "misbound",
            "core_claim": "Primary result and cohort accounting.",
            "panels": [
                {
                    "panel_id": "A",
                    "title": "Primary result",
                    "role": "relationship",
                    "claim": "Primary estimate.",
                    "evidence_ids": ["run_status"],
                },
                {
                    "panel_id": "B",
                    "title": "Cohort",
                    "role": "cohort_accounting",
                    "claim": "Cohort count.",
                    "evidence_ids": ["cohort_table"],
                },
            ],
            "source_data": ["run_status", "cohort_table"],
        },
    )

    item = _figure_inventory(tmp_path)[0]

    assert item.primary_result_data_bound is False
    assert item.provenance_valid is False


def test_discovery_contract_record_must_bind_all_panel_sources(tmp_path: Path):
    bundle = _build_strict_ready_package(tmp_path)
    store = EvidenceStore(tmp_path)
    contract = next(
        item for item in store._records if item.evidence_id == bundle["contract"]
    )
    contract.inputs.remove("primary_result")
    store._save()

    assessment = validate_discovery_manuscript_package(run_dir=tmp_path)

    assert assessment.figure_inventory[0].contract_registered is False
    assert assessment.package_ready is False


def test_free_text_cannot_self_certify_all_story_roles(tmp_path: Path):
    _register_text_evidence(
        tmp_path,
        evidence_id="source",
        filename="source.csv",
        kind="table",
        text="value\n1\n",
    )
    figure_dir = tmp_path / "publication_figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        figure_dir / "prose_only.figure_contract.json",
        {
            "figure_id": "prose_only",
            "core_claim": "Discovery cohort primary result audit reproducibility.",
            "panels": [
                {
                    "panel_id": "A",
                    "title": (
                        "Literature discovery cohort evaluability primary result audit"
                    ),
                    "role": "mechanism",
                    "claim": "All story keywords appear only in free prose.",
                    "evidence_ids": ["source"],
                }
            ],
            "source_data": ["source"],
        },
    )

    item = _figure_inventory(tmp_path)[0]

    assert item.story_roles == []
    assert item.provenance_valid is False
