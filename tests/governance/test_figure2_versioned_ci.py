"""Current authority and transparent contract succession stay separate from v3."""
from __future__ import annotations

import ast
import copy
import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from benchmarks.figure2_canonical9.evaluator import paper_rubric_v3 as legacy
from benchmarks.figure2_canonical9.evaluator import scoring as legacy_scoring
from benchmarks.figure2_canonical9.evaluator import scoring_inputs as legacy_inputs
from benchmarks.figure2_canonical9.evaluator import safety_issuer as legacy_issuer
from benchmarks.figure2_canonical9.successor_v4 import paper_rubric_v4 as current
from benchmarks.figure2_canonical9.successor_v4 import scoring, scoring_inputs, safety_issuer
from tests.support.figure2_successor import ROOT, policy, projected_ast


class CoordinateNeutral(ast.NodeTransformer):
    """Independent equivalence check: only version coordinates may differ."""
    def __init__(self):
        declaration = policy()
        self.coordinates = {value: key for key, value in declaration["authority_coordinates"].items()}
        self.old = declaration["legacy_namespace"]
        self.new = declaration["current_namespace"]

    def visit_Constant(self, node):
        if isinstance(node.value, str):
            node.value = self.coordinates.get(node.value, node.value)
        return node

    def visit_Name(self, node):
        node.id = node.id.replace("_v4", "_v3")
        return node

    def visit_FunctionDef(self, node):
        node = self.generic_visit(node)
        node.name = node.name.replace("_v4", "_v3")
        return node

    def visit_ImportFrom(self, node):
        node.module = (node.module or "").replace(self.new, self.old).replace("paper_rubric_v4", "paper_rubric_v3")
        # Projection may split a mixed 'from package import a,b'. Compare the
        # original and derived imports in the same canonical one-alias form.
        result = []
        for alias in node.names:
            name = alias.name.replace("paper_rubric_v4", "paper_rubric_v3")
            asname = alias.asname
            if asname == name:
                asname = None
            result.append(ast.ImportFrom(module=node.module, names=[ast.alias(name=name, asname=asname)], level=node.level))
        return result


def test_all_149_original_contracts_have_exact_current_identity_mapping():
    declaration = policy()
    assert len(declaration["expected_original_nodes"]) == 149
    assert len(set(declaration["expected_successor_nodes"])) == 149
    assert len(declaration["test_sources"]) == 4
    for relative, entry in declaration["test_sources"].items():
        payload = (ROOT / relative).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == entry["sha256"]
        source = ast.parse(payload)
        projected = projected_ast(payload.decode())
        assert ast.dump(CoordinateNeutral().visit(copy.deepcopy(source))) == ast.dump(CoordinateNeutral().visit(copy.deepcopy(projected)))
        assert sum(isinstance(n, ast.Assert) for n in ast.walk(source)) == sum(isinstance(n, ast.Assert) for n in ast.walk(projected))
        assert (ROOT / entry["successor_test"]).is_file()


def test_legacy_rubric_is_unchanged_and_rejects_current_source():
    declaration = policy()
    legacy_bytes = (ROOT / declaration["legacy_rubric_file"]).read_bytes()
    assert hashlib.sha256(legacy_bytes).hexdigest() == declaration["legacy_rubric_file_sha256"]
    raw = json.loads(legacy_bytes)
    assert raw["scorer_tree_sha256"] == declaration["legacy_scorer_tree_sha256"]
    assert legacy.scorer_tree_sha256() != declaration["legacy_scorer_tree_sha256"]
    with pytest.raises(ValueError, match="scorer tree digest mismatch"):
        legacy.load_figure2_paper_rubric()


def test_current_manifest_is_engineering_only_and_covers_delegated_owners():
    manifest = current.load_figure2_paper_rubric()
    assert manifest.authority_status == "engineering_candidate_no_formal_authority"
    assert manifest.rubric_ref != legacy.FIGURE2_PAPER_RUBRIC_REF
    assert {
        "src/easyicu/research_agent/contracts/model_terms.py",
        "src/easyicu/research_agent/contracts/diagnostic_specs.py",
        "src/easyicu/research_agent/planning/cohort_contract.py",
        "src/easyicu/research_agent/authority/source_fingerprints.py",
        "src/easyicu/research_agent/authority/scientific_claim_registry.py",
    } <= set(manifest.scorer_core_files)


@pytest.fixture
def source_tree(tmp_path, monkeypatch):
    root = tmp_path / "source"
    for relative in [
        "src/easyicu/__init__.py", "src/easyicu/data/concept-dict.json",
        "src/easyicu/research_agent/contracts/model_terms.py",
        "src/easyicu/research_agent/authority/source_fingerprints.py",
        "src/easyicu/research_agent/authority/scientific_claim_registry.py",
        "benchmarks/figure2_canonical9/evaluator/shared.py",
        "benchmarks/figure2_canonical9/successor_v4/paper_rubric_v4.py",
    ]:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}" if path.suffix == ".json" else "value = 1\n")
    payload = json.loads(current.default_figure2_paper_rubric_path().read_text())
    monkeypatch.setattr(current, "__file__", str(root / "benchmarks/figure2_canonical9/successor_v4/paper_rubric_v4.py"))
    payload["scorer_core_files"] = current.current_core_files()
    payload["scorer_tree_sha256"] = current.scorer_tree_sha256()
    path = root / "manifest.json"
    path.write_text(json.dumps(payload))
    assert current.load_figure2_paper_rubric(path).scorer_tree_sha256 == payload["scorer_tree_sha256"]
    return root, path


@pytest.mark.parametrize("relative", [
    "src/easyicu/research_agent/contracts/model_terms.py",
    "src/easyicu/research_agent/authority/source_fingerprints.py",
    "src/easyicu/research_agent/authority/scientific_claim_registry.py",
])
def test_changed_delegated_owner_is_rejected(source_tree, relative):
    root, manifest = source_tree
    with (root / relative).open("a") as handle:
        handle.write("# changed after the source lock\n")
    with pytest.raises(ValueError, match="scorer tree digest mismatch"):
        current.load_figure2_paper_rubric(manifest)


@pytest.mark.parametrize("change", ["new_owner", "missing_owner"])
def test_source_membership_change_is_rejected(source_tree, change):
    root, manifest = source_tree
    path = root / "src/easyicu/research_agent/contracts/new_owner.py"
    if change == "new_owner":
        path.write_text("new = True\n")
    else:
        (root / "src/easyicu/research_agent/contracts/model_terms.py").unlink()
    with pytest.raises(ValueError, match="core membership drifted"):
        current.load_figure2_paper_rubric(manifest)


@pytest.mark.parametrize("kind", ["missing", "symlink", "file"])
def test_source_enumeration_root_must_be_real_local_directory(source_tree, kind):
    root, _ = source_tree
    target = root / "src/easyicu/data"
    if kind == "missing":
        (target / "concept-dict.json").unlink()
        target.rmdir()
    elif kind == "file":
        (target / "concept-dict.json").unlink()
        target.rmdir()
        target.write_text("not a directory")
    else:
        moved = root / "moved-data"
        target.rename(moved)
        try:
            target.symlink_to(moved, target_is_directory=True)
        except OSError as error:
            pytest.skip(f"symlinks unavailable: {error}")
    with pytest.raises((OSError, ValueError)):
        current.current_core_files()


@pytest.mark.parametrize("new,old", [
    (current.Figure2PaperRubricManifest, legacy.Figure2PaperRubricManifest),
    (current.Figure2PaperScorecard, legacy.Figure2PaperScorecard),
    (scoring.Figure2PaperScorecardEnvelope, legacy_scoring.Figure2PaperScorecardEnvelope),
    (scoring.Figure2EvaluationAttempt, legacy_scoring.Figure2EvaluationAttempt),
    (scoring_inputs.Figure2RunTaskAuthority, legacy_inputs.Figure2RunTaskAuthority),
    (scoring_inputs.Figure2ScoringInputAuthority, legacy_inputs.Figure2ScoringInputAuthority),
    (safety_issuer.Figure2SafetyRequest, legacy_issuer.Figure2SafetyRequest),
    (safety_issuer.Figure2SafetyReceipt, legacy_issuer.Figure2SafetyReceipt),
])
def test_unchanged_legacy_authority_schema_cannot_be_reused_as_v4(new, old):
    old_schema = old.model_fields["schema_version"].annotation.__args__[0]
    new_schema = new.model_fields["schema_version"].annotation.__args__[0]
    assert old_schema != new_schema
    with pytest.raises(ValidationError) as raised:
        new.model_validate({"schema_version": old_schema})
    assert any(row["loc"] == ("schema_version",) and row["type"] == "literal_error" for row in raised.value.errors())


@pytest.mark.parametrize("name", ["scoring_inputs.py", "scoring.py", "safety_issuer.py"])
def test_production_forks_change_only_declared_authority_coordinates(name):
    successor = ROOT / "benchmarks/figure2_canonical9/successor_v4"
    derivation = json.loads((successor / "DERIVATION.json").read_text())
    entry = derivation["source_owner_projection"][name]
    origin = (ROOT / entry["origin"]).read_bytes()
    assert hashlib.sha256(origin).hexdigest() == entry["origin_sha256"]
    assert entry["coordinate_projection_only"] is True

    class ProductionNeutral(CoordinateNeutral):
        def visit_ImportFrom(self, node):
            if node.level == 2 and node.module and node.module.startswith("evaluator."):
                node.level = 1
                node.module = node.module.removeprefix("evaluator.")
            node.module = (node.module or "").replace("paper_rubric_v4", "paper_rubric_v3")
            return node

    old_tree = ProductionNeutral().visit(ast.parse(origin))
    new_tree = ProductionNeutral().visit(ast.parse((successor / name).read_text()))
    assert ast.dump(old_tree) == ast.dump(new_tree)


@pytest.mark.parametrize("lexical_escape", [False, True])
def test_source_enumeration_rejects_escaped_root(source_tree, lexical_escape):
    root, _ = source_tree
    outside = root.parent / "outside"
    outside.mkdir()
    (outside / "outside.py").write_text("outside = True\n")
    target = root / ".." / "outside" if lexical_escape else outside
    with pytest.raises(ValueError, match="escapes the repository"):
        current._source_paths(target, suffixes=(".py",))
