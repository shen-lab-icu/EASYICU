"""Every call that reads from a source or export path is classified (S1540).

B6, B9, B11 and B12 (UX walks, 2026-11-19) were one class: a consumer took a
study's bound source path, which for a raw official-Demo study is the raw
folder, and read tables, modules, a catalog or an identity from it where the
registered official export answers (``bound_source_authority_path``). B12
escaped a sweep keyed on reader names (``describe_source_tables``). This
sweep is keyed on the value instead: it finds every call in
``src/easyicu/webserver`` that is handed a source or export path, whatever
the callee is called, and requires each one to be classified.

Rule (static, AST, per function):
- A *path value* is a name or attribute in ``PATH_NAMES``; ``x.get("path")``
  or ``x["path"]`` where ``x`` is a source mapping (``SOURCE_MAPS``, or a
  ``["data_source"]`` / ``.get("data_source")`` read); a local assigned from a
  path value (one-step intra-function taint); and any of those wrapped in a
  pure conversion (``WRAPS``: ``str``, ``Path``, ``expanduser``, ...), or an
  ``a or b`` / ``a if c else b`` with a path value in it.
- A *site* is a call that receives a path value as a positional or keyword
  argument, unless its callee is in ``PURE`` (string, container, logging and
  stat helpers that cannot read tables, modules or catalogs).

Classes:
- OWNER: the callee is an owner (``bound_source_authority_path`` and peers),
  the path comes from one, or a raw folder is diverted first by design (the
  raw-Demo planning menu, the signed-choice owner).
- FOLDER: describes, checks, hashes, lists or identifies the folder itself;
  its result never becomes a run's scope, catalog or identity.
- REGISTERED: only a registry row, a validated prepared binding, a published
  package or a review of one reaches it.
- REFUSES: a raw folder gets a typed refusal (or reads as "not a package").
- PASS: forwards into a webserver function whose own path parameter is in
  ``PATH_NAMES``, so its body's sites are classified here too.
- BUG: the raw folder is read where its registered export answers. Kept red
  until fixed.

Known limits (static, deliberately bounded): a path stored on an attribute
(``self.folder = export_path``), a closure variable assigned after its inner
function, a record field outside ``PATH_NAMES``, and a container other than
a literal dict, tuple or list are not followed. A new reader behind one of
these shapes is caught by the behaviour tests of its owner, not here.
"""

from __future__ import annotations

import ast
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3] / "src" / "easyicu" / "webserver"
PATH_NAMES = frozenset({
    "export_path", "source_path", "raw_path", "input_root", "data_dir", "export_dir", "package_path", "source_dir",
    "export_root", "source_root", "raw_dir", "raw_folder", "source_folder", "export_folder", "package_dir",
})
SOURCE_MAPS = frozenset({"source", "data_source", "bound_source", "effective_source"})
WRAPS = frozenset({
    "str", "Path", "fspath", "_clean_text", "expanduser", "resolve", "strip", "normpath", "abspath", "realpath",
    # A member path built from the source path (Carbon b14r1), or the path
    # re-spelled as text (Carbon b14r2).
    "join", "joinpath", "rstrip", "lstrip", "replace", "format", "with_name", "with_suffix",
})
PURE = WRAPS | frozenset({
    "len", "bool", "isinstance", "int", "dict", "list", "set", "tuple", "frozenset", "sorted", "append", "extend", "update",
    "add", "join", "format", "replace", "startswith", "endswith", "lower", "upper", "split", "rstrip", "lstrip", "get",
    "setdefault", "debug", "info", "warning", "error", "exception", "log", "hash", "repr", "same", "fromkeys",
    "dumps", "relative_to", "is_relative_to", "samefile", "exists", "is_dir", "is_file", "lexists", "isdir", "isfile",
    "basename", "dirname", "_norm_path", "normalize_path", "next", "any", "all", "getattr", "hasattr", "print",
})


def _source_map(node: ast.AST, maps: frozenset[str] | set[str] = frozenset()) -> bool:
    if isinstance(node, ast.Name):
        return node.id in SOURCE_MAPS or node.id in maps
    if isinstance(node, ast.Call) and node.args and isinstance(node.func, ast.Name) and node.func.id in {"dict", "MappingProxyType"}:
        # ``dict(study.get("data_source") or {})`` (Carbon b14r2).
        return _source_map(node.args[0], maps)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in {"copy", "deepcopy"}:
        return _source_map(node.func.value, maps) or any(_source_map(arg, maps) for arg in node.args)
    if isinstance(node, ast.BoolOp):
        return any(_source_map(value, maps) for value in node.values)
    if isinstance(node, ast.IfExp):
        return _source_map(node.body, maps) or _source_map(node.orelse, maps)
    if isinstance(node, ast.Attribute):
        return node.attr in SOURCE_MAPS
    if isinstance(node, ast.Subscript):
        return isinstance(node.slice, ast.Constant) and node.slice.value == "data_source"
    return (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "get"
            and bool(node.args) and isinstance(node.args[0], ast.Constant) and node.args[0].value == "data_source")


def _path_value(node: ast.AST, tainted: set[str], maps: frozenset[str] | set[str] = frozenset()) -> bool:
    if isinstance(node, ast.Name):
        return node.id in PATH_NAMES or node.id in tainted
    if isinstance(node, ast.Attribute):
        return node.attr in PATH_NAMES
    if isinstance(node, ast.Subscript):
        return isinstance(node.slice, ast.Constant) and node.slice.value == "path" and _source_map(node.value, maps)
    if isinstance(node, ast.BoolOp):
        return any(_path_value(value, tainted, maps) for value in node.values)
    if isinstance(node, ast.IfExp):
        return _path_value(node.body, tainted, maps) or _path_value(node.orelse, tainted, maps)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        # ``Path(export_path) / "member.parquet"`` (Carbon b14r1).
        return _path_value(node.left, tainted, maps)
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Mod)):
        # ``export_path + "/x"``, ``"%s/x" % export_path`` (Carbon b14r2).
        return _path_value(node.left, tainted, maps) or _path_value(node.right, tainted, maps)
    if isinstance(node, ast.JoinedStr):
        return any(isinstance(part, ast.FormattedValue) and _path_value(part.value, tainted, maps) for part in node.values)
    if isinstance(node, ast.NamedExpr):
        return _path_value(node.value, tainted, maps)
    if isinstance(node, ast.Dict):
        # ``reader(**{"export_path": folder})`` or a kwargs local (Carbon b14r2).
        return any(value is not None and _path_value(value, tainted, maps) for value in node.values)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "dict":
        # ``dict(export_path=folder)`` (Carbon b14r3).
        return any(_path_value(value, tainted, maps)
                   for value in (*node.args, *(keyword.value for keyword in node.keywords)))
    if isinstance(node, (ast.Tuple, ast.List)):
        return any(_path_value(element, tainted, maps) for element in node.elts)
    if isinstance(node, ast.Call):
        func = node.func
        if (isinstance(func, ast.Attribute) and func.attr == "get" and node.args
                and isinstance(node.args[0], ast.Constant) and node.args[0].value == "path"):
            return _source_map(func.value, maps)
        name = func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else ""
        if name in WRAPS:
            return any(_path_value(arg, tainted, maps) for arg in node.args) or (
                isinstance(func, ast.Attribute) and _path_value(func.value, tainted, maps))
    return False


def _path_keys(node: ast.AST, tainted: set[str], maps, known: dict[str, set]) -> set | None:
    """The keys of a dict value that carry a path (``None`` for an unknown key).

    ``None`` as a result means the value is not a dict this sweep can read.
    """

    if isinstance(node, ast.Name) and node.id in known:
        return set(known[node.id])
    if isinstance(node, ast.Dict):
        keys: set = set()
        for key, value in zip(node.keys, node.values):
            if key is None:
                inner = _path_keys(value, tainted, maps, known)
                keys |= inner if inner is not None else ({None} if _path_value(value, tainted, maps) else set())
            elif _path_value(value, tainted, maps):
                keys.add(key.value if isinstance(key, ast.Constant) else None)
        return keys
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "dict":
        keys = {keyword.arg for keyword in node.keywords
                if keyword.arg is not None and _path_value(keyword.value, tainted, maps)}
        for arg in (*node.args, *(keyword.value for keyword in node.keywords if keyword.arg is None)):
            inner = _path_keys(arg, tainted, maps, known)
            keys |= inner if inner is not None else ({None} if _path_value(arg, tainted, maps) else set())
        return keys
    return None


def _callee(node: ast.Call) -> str:
    func = node.func
    return func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else "?"


class _Sites(ast.NodeVisitor):
    def __init__(self, rel: str, sites: Counter, bindings: dict | None = None) -> None:
        self.rel, self.sites, self.scope, self.tainted, self.maps = rel, sites, [], [set()], [set()]
        self.kinds: list[str] = []
        self.bindings = bindings if bindings is not None else {}
        self.keys: list[dict[str, set]] = [{}]

    def _enter(self, name: str, kind: str) -> None:
        # A nested function sees its enclosing function's locals (Carbon
        # b14r1); a class body or a method starts clean (Carbon b14r2).
        inherit = kind == "function" and bool(self.kinds) and self.kinds[-1] == "function"
        self.scope.append(name)
        self.kinds.append(kind)
        self.tainted.append(set(self.tainted[-1]) if inherit else set())
        self.maps.append(set(self.maps[-1]) if inherit else set())
        self.keys.append(dict(self.keys[-1]) if inherit else {})

    def _leave(self) -> None:
        self.scope.pop()
        self.kinds.pop()
        self.tainted.pop()
        self.maps.pop()
        self.keys.pop()

    def _function(self, node) -> None:
        self._enter(node.name, "function")
        self.generic_visit(node)
        self._leave()

    visit_FunctionDef = visit_AsyncFunctionDef = _function

    def visit_ClassDef(self, node) -> None:
        self._enter(node.name, "class")
        self.generic_visit(node)
        self._leave()

    def _bind(self, name: str, value: ast.AST) -> None:
        keys = _path_keys(value, self.tainted[-1], self.maps[-1], self.keys[-1])
        carries = _path_value(value, self.tainted[-1], self.maps[-1])
        if keys:
            # ``runner_kwargs = {"export_path": path, ...}`` (Carbon b14r3).
            # Every assignment adds to what the name may carry, so a branch
            # cannot narrow it (Carbon b14r4), nor an opaque value it already
            # carried (Carbon b14r5).
            self.keys[-1][name] = self._prior_keys(name) | keys
        elif keys is None and name in self.keys[-1]:
            # An opaque rebind of a kwargs dict carries unknown keys.
            self.keys[-1][name] = self.keys[-1][name] | {None}
        if carries:
            self.tainted[-1].add(name)
        if _source_map(value, self.maps[-1]):
            # ``source = study.get("data_source") or {}`` (Carbon b14r1).
            self.maps[-1].add(name)

    def _prior_keys(self, name: str) -> set:
        return self.keys[-1].get(name, {None} if name in self.tainted[-1] else set())

    def _store_key(self, target: ast.AST, value: ast.AST) -> None:
        # ``kwargs["label"] = export_path`` / ``kwargs |= {...}`` (Carbon b14r4).
        # A nested mapping (``patch["data_source"] = {"path": ...}``) is not a
        # path argument; it is read through ``["data_source"]`` instead.
        if not (isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name)):
            return
        key = target.slice.value if isinstance(target.slice, ast.Constant) else None
        nested = isinstance(value, ast.Dict) or (
            isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and value.func.id == "dict")
        if not (nested and key == "data_source"):
            if _path_value(value, self.tainted[-1], self.maps[-1]):
                name = target.value.id
                self.keys[-1][name] = self._prior_keys(name) | {key}
                self.tainted[-1].add(name)

    def visit_AugAssign(self, node) -> None:
        if isinstance(node.target, ast.Name) and isinstance(node.op, ast.BitOr):
            self._bind(node.target.id, node.value)
        self.generic_visit(node)

    def visit_Assign(self, node) -> None:
        for target in node.targets:
            if isinstance(target, ast.Name):
                self._bind(target.id, node.value)
            elif isinstance(target, ast.Subscript):
                self._store_key(target, node.value)
            elif isinstance(target, (ast.Tuple, ast.List)) and isinstance(node.value, (ast.Tuple, ast.List)):
                # ``folder, database = export_path, name`` (Carbon b14r2).
                for element, value in zip(target.elts, node.value.elts):
                    if isinstance(element, ast.Name):
                        self._bind(element.id, value)
        self.generic_visit(node)

    def visit_NamedExpr(self, node) -> None:
        if isinstance(node.target, ast.Name):
            self._bind(node.target.id, node.value)
        self.generic_visit(node)

    def visit_AnnAssign(self, node) -> None:
        if node.value is not None and isinstance(node.target, ast.Name):
            self._bind(node.target.id, node.value)
        self.generic_visit(node)

    def visit_Call(self, node) -> None:
        callee = _callee(node)
        tainted, maps = self.tainted[-1], self.maps[-1]
        slots = {("pos", index) for index, value in enumerate(node.args) if _path_value(value, tainted, maps)}
        for keyword in node.keywords:
            if keyword.arg is not None and _path_value(keyword.value, tainted, maps):
                slots.add(("kw", keyword.arg))
            elif keyword.arg is None:
                keys = _path_keys(keyword.value, tainted, maps, self.keys[-1])
                if keys:
                    # An unknown key reads as ``**`` and no PASS row accepts it.
                    slots.update(("kw", "**" if key is None else key) for key in keys)
                elif keys is None and _path_value(keyword.value, tainted, maps):
                    slots.add(("kw", "**"))
        if callee not in PURE and slots:
            key = (self.rel, ".".join(self.scope) or "<module>", callee)
            self.sites[key] += 1
            self.bindings.setdefault(key, set()).update(slots)
        self.generic_visit(node)


def path_consumer_sites(sources: dict[str, str], bindings: dict | None = None) -> Counter:
    sites: Counter = Counter()
    for rel, text in sources.items():
        _Sites(rel, sites, bindings).visit(ast.parse(text, filename=rel))
    return sites


def _webserver_sources() -> dict[str, str]:
    return {str(path.relative_to(ROOT)): path.read_text(encoding="utf-8") for path in sorted(ROOT.rglob("*.py"))}


_CLASSES = frozenset({"OWNER", "FOLDER", "REGISTERED", "REFUSES", "PASS", "BUG"})
_CLASSIFIED = {
    ('agent_outputs.py', '_load_context', 'folder_unpublished_reason'): (1, 'REGISTERED'),
    ('agent_outputs.py', '_load_context', 'NativeExportUnpublishedError'): (1, 'REGISTERED'),
    ('agent_outputs.py', '_load_context', 'load_published_manifest'): (1, 'REGISTERED'),
    ('agent_outputs.py', '_load_context', 'require_listed_members_present'): (1, 'REGISTERED'),
    ('agent_outputs.py', '_load_context', 'verify_member_integrity'): (1, 'REGISTERED'),
    ('agent_outputs.py', 'build_agent_output_artifacts', '_load_context'): (1, 'REGISTERED'),
    ('agent_pipeline_runs.py', '_native_metadata_planning_authority', 'native_planning_metadata'): (1, 'OWNER'),
    ('agent_pipeline_runs.py', '_metadata_only_planning_catalog', '_native_metadata_planning_authority'): (1, 'PASS'),
    ('agent_pipeline_runs.py', '_metadata_only_planning_catalog', 'build_available_catalog'): (1, 'OWNER'),
    ('agent_pipeline_runs.py', '_source_concepts_without_values', 'is_export_package'): (1, 'OWNER'),
    ('agent_pipeline_runs.py', '_source_concepts_without_values', 'concepts_without_values'): (1, 'OWNER'),
    ('agent_pipeline_runs.py', '_metadata_only_planning_acquisition', '_native_metadata_planning_authority'): (1, 'PASS'),
    ('agent_pipeline_runs.py', '_metadata_only_planning_acquisition', '_metadata_only_planning_catalog'): (1, 'PASS'),
    ('agent_pipeline_runs.py', '_metadata_only_planning_acquisition', '_source_concepts_without_values'): (1, 'PASS'),
    ('agent_pipeline_runs.py', '_restore_metadata_only_planning_acquisition', '_metadata_only_planning_catalog'): (1, 'PASS'),
    ('agent_pipeline_runs.py', '_restore_metadata_only_planning_acquisition', '_verified_demo_planning_output_identity'): (1, 'OWNER'),
    ('agent_pipeline_runs.py', 'research_context_declarations', 'bound_export_selection_basis'): (1, 'REGISTERED'),
    ('agent_pipeline_runs.py', 'research_context_declarations', 'verified_first_stay_applied'): (1, 'REGISTERED'),
    ('agent_pipeline_runs.py', 'research_context_declarations', 'bound_export_event_time_semantics'): (1, 'REGISTERED'),
    ('agent_pipeline_runs.py', 'research_context_declarations', 'bound_export_selection_report'): (1, 'REGISTERED'),
    ('agent_pipeline_runs.py', 'acquire_target_trial_universe', 'acquire_universe_for_question'): (1, 'REGISTERED'),
    ('agent_pipeline_runs.py', 'target_trial_research_context', 'registered_source_identity'): (1, 'FOLDER'),
    ('agent_pipeline_runs.py', 'target_trial_research_context', 'research_context_declarations'): (1, 'PASS'),
    ('agent_pipeline_runs.py', '_reviewed_candidate_source_time_axis', 'attest_registered_export_time_origin'): (1, 'REGISTERED'),
    ('agent_pipeline_runs.py', '_load_candidate_plan_materialization_authority', '_reviewed_candidate_source_time_axis'): (1, 'PASS'),
    ('agent_pipeline_runs.py', '_load_candidate_plan_materialization_authority', '_metadata_only_planning_coordinates'): (1, 'PASS'),
    ('agent_pipeline_runs.py', 'make_research_pipeline_run_runner', 'ResearchPipelineLaunchRequest'): (1, 'PASS'),
    ('agent_pipeline_runs.py', 'make_research_pipeline_run_runner.runner', 'public_demo_egress_scope'): (1, 'FOLDER'),
    ('agent_pipeline_runs.py', 'make_research_pipeline_run_runner.runner', 'validate_research_pipeline_source'): (1, 'REGISTERED'),
    ('agent_pipeline_runs.py', 'make_research_pipeline_run_runner.runner', '_load_candidate_plan_materialization_authority'): (1, 'PASS'),
    ('agent_pipeline_runs.py', 'make_research_pipeline_run_runner.runner', '_data_foundation_profile'): (2, 'OWNER'),
    ('agent_pipeline_runs.py', 'make_research_pipeline_run_runner.runner', 'create_seed'): (1, 'PASS'),
    ('agent_pipeline_runs.py', 'make_research_pipeline_run_runner.runner', '_restore_metadata_only_planning_acquisition'): (1, 'PASS'),
    ('agent_pipeline_runs.py', 'make_research_pipeline_run_runner.runner', '_metadata_only_planning_acquisition'): (1, 'PASS'),
    ('agent_pipeline_runs.py', 'make_research_pipeline_run_runner.runner', '_verified_demo_planning_output_identity'): (1, 'OWNER'),
    ('agent_pipeline_runs.py', 'make_research_pipeline_run_runner.runner', 'acquire_target_trial_universe'): (1, 'PASS'),
    ('agent_pipeline_runs.py', 'make_research_pipeline_run_runner.runner', 'acquire_universe_for_question'): (1, 'REGISTERED'),
    ('agent_pipeline_runs.py', 'make_research_pipeline_run_runner.runner', 'materialize_web_time_varying_input'): (1, 'PASS'),
    ('agent_pipeline_runs.py', 'make_research_pipeline_run_runner.runner', 'materialize_web_hospital_followup'): (1, 'PASS'),
    ('agent_pipeline_runs.py', 'make_research_pipeline_run_runner.runner', 'bound_export_selection_basis'): (1, 'REGISTERED'),
    ('agent_pipeline_runs.py', 'make_research_pipeline_run_runner.runner', 'bound_export_selection_report'): (1, 'REGISTERED'),
    ('agent_pipeline_runs.py', 'make_research_pipeline_run_runner.runner', 'research_context_declarations'): (1, 'PASS'),
    ('agent_pipeline_runs.py', 'resume_research_pipeline', 'validate_research_pipeline_source'): (1, 'REGISTERED'),
    ('agent_review_recovery.py', 'recover_pending_review', 'validate_research_pipeline_source'): (1, 'REGISTERED'),
    ('agent_runs.py', 'make_agent_run_runner', 'build_agent_context_binding'): (1, 'FOLDER'),
    ('agent_runs.py', 'make_agent_run_runner.runner', 'describe_export_source'): (1, 'FOLDER'),
    ('agent_runs.py', 'make_agent_run_runner.runner', 'summarize_export_workspace'): (1, 'FOLDER'),
    ('agent_runs.py', 'make_agent_run_runner.runner', 'build_agent_output_artifacts'): (1, 'REGISTERED'),
    ('catalog.py', '_registered_source_concepts', 'read_prepared_export_manifest'): (1, 'REGISTERED'),
    ('cohort_landmark.py', '_verify_eicu_outcomes', 'ParquetFile'): (1, 'REGISTERED'),
    ('cohort_landmark.py', '_verify_eicu_outcomes', 'collect'): (2, 'REGISTERED'),
    ('cohort_landmark.py', '_build_landmark_group', '_verify_eicu_outcomes'): (1, 'REGISTERED'),
    ('cohort_review.py', 'cohort_review_summary', '_summary_cache_key'): (1, 'FOLDER'),
    ('cohort_review.py', 'cohort_review_summary', '_read_module_frame'): (1, 'REGISTERED'),
    ('cohort_review.py', 'cohort_review_summary', '_fallback_entity_frame'): (1, 'REGISTERED'),
    ('cohort_review.py', 'cohort_review_summary', '_selected_feature_profiles'): (1, 'REGISTERED'),
    ('cohort_review.py', 'cohort_review_summary', '_build_custom_group'): (1, 'REGISTERED'),
    ('cohort_review.py', 'cohort_review_summary', 'build_landmark_group'): (1, 'REGISTERED'),
    ('cohort_review.py', 'cohort_review_summary', '_coverage_payload'): (1, 'REGISTERED'),
    ('cohort_review.py', '_source_provenance', '_hash'): (1, 'FOLDER'),
    ('crossdb_review.py', '_source_aggregate', 'describe_export_source'): (1, 'FOLDER'),
    ('crossdb_review.py', '_source_aggregate', '_hash'): (1, 'FOLDER'),
    ('data_package_execution_readiness.py', '_hospital_first_eligibility_review', 'resolve_study_first_icu_stay_per_hospitalization'): (1, 'PASS'),
    ('data_package_execution_readiness.py', '_hospital_first_eligibility_review', '_read_review_concept'): (1, 'REGISTERED'),
    ('data_package_execution_readiness.py', '_first_icu_stay_flags', 'resolve_study_first_icu_stay'): (1, 'PASS'),
    ('data_package_execution_readiness.py', '_age_eligibility_review', '_read_review_concept'): (1, 'REGISTERED'),
    ('data_package_execution_readiness.py', '_patient_grouping_review', 'resolve_study_patient_grouping'): (1, 'PASS'),
    ('data_package_execution_readiness.py', '_first_icu_stay_review', 'resolve_study_first_icu_stay'): (1, 'PASS'),
    ('data_package_execution_readiness.py', '_hospital_first_stay_review', 'resolve_study_first_icu_stay_per_hospitalization'): (1, 'PASS'),
    ('data_package_execution_readiness.py', '_concept_time_capability', '_read_review_concept'): (1, 'REGISTERED'),
    ('data_package_execution_readiness.py', '_read_review_concept', 'read_exported_concept'): (1, 'REGISTERED'),
    ('data_package_execution_readiness.py', '_hospital_mortality_followup_capability', 'resolve_raw_hospital_source_binding'): (1, 'REGISTERED'),
    ('data_package_execution_readiness.py', '_cohort_eligibility_review', '_age_eligibility_review'): (1, 'PASS'),
    ('data_package_execution_readiness.py', '_runtime_readiness_review', '_patient_grouping_review'): (1, 'PASS'),
    ('data_package_execution_readiness.py', '_runtime_readiness_review', '_first_icu_stay_review'): (1, 'PASS'),
    ('data_package_execution_readiness.py', '_runtime_readiness_review', '_hospital_first_stay_review'): (1, 'PASS'),
    ('data_package_execution_readiness.py', '_runtime_readiness_review', '_hospital_mortality_followup_capability'): (1, 'PASS'),
    ('data_package_execution_readiness.py', '_cohort_eligibility_review', '_hospital_first_eligibility_review'): (1, 'PASS'),
    ('data_package_execution_readiness.py', '_runtime_readiness_review', '_concept_time_capability'): (2, 'PASS'),
    ('data_package_execution_readiness.py', '_cohort_eligibility_review', '_first_icu_stay_flags'): (1, 'PASS'),
    ('data_package_execution_readiness.py', '_cohort_eligibility_review', '_read_review_concept'): (1, 'REGISTERED'),
    ('data_package_execution_readiness.py', 'build_data_package_execution_readiness', '_cohort_eligibility_review'): (1, 'PASS'),
    ('data_package_execution_readiness.py', 'build_data_package_execution_readiness', '_runtime_readiness_review'): (1, 'PASS'),
    ('data_package_review.py', '_typed_event_availability_receipt', 'read_exported_concept'): (1, 'REGISTERED'),
    ('data_package_review.py', 'build_registered_data_package_review', 'build_available_catalog'): (1, 'REFUSES'),
    ('data_package_review.py', 'build_registered_data_package_review', '_typed_event_availability_receipt'): (1, 'PASS'),
    ('data_package_review.py', 'build_registered_data_package_review', 'build_data_package_execution_readiness'): (1, 'PASS'),
    ('dataio.py', '_feature_definition_payload', '_shareable_path_reference'): (1, 'FOLDER'),
    ('dataio.py', 'list_dir', 'require_path'): (1, 'FOLDER'),
    ('dataio.py', 'list_dir', '_list_dir'): (2, 'FOLDER'),
    ('dataio.py', '_list_dir', '_safe_dir'): (1, 'FOLDER'),
    ('dataio.py', 'resolve_registered_export_binding', '_read_export_manifest'): (1, 'REGISTERED'),
    ('dataio.py', 'resolve_registered_export_binding', 'resolve_raw_mimic_iv_source_binding'): (1, 'PASS'),
    ('dataio.py', 'preview_registered_export_icd_cohort', 'resolve_registered_export_binding'): (1, 'REGISTERED'),
    ('dataio.py', 'preview_registered_export_icd_cohort', 'preview_export_cohort'): (1, 'REGISTERED'),
    ('dataio.py', 'create_dir', 'require_path'): (1, 'FOLDER'),
    ('dataio.py', 'create_dir', '_safe_dir'): (1, 'FOLDER'),
    ('dataio.py', 'summarize_export_workspace', 'require_path'): (2, 'FOLDER'),
    ('dataio.py', 'summarize_export_workspace', '_safe_dir'): (1, 'FOLDER'),
    ('dataio.py', 'summarize_export_workspace', '_unpublished_export_error'): (1, 'FOLDER'),
    ('dataio.py', 'summarize_export_workspace', '_export_file_inventory'): (1, 'FOLDER'),
    ('dataio.py', 'summarize_export_workspace', '_cohort_total_rows'): (1, 'FOLDER'),
    ('dataio.py', 'describe_export_source', 'require_path'): (1, 'FOLDER'),
    ('dataio.py', 'describe_export_source', '_safe_dir'): (1, 'FOLDER'),
    ('dataio.py', 'describe_export_source', '_unpublished_export_error'): (1, 'FOLDER'),
    ('dataio.py', 'describe_export_source', 'source_description'): (1, 'FOLDER'),
    ('dataio.py', 'describe_export_source', '_read_export_manifest'): (1, 'REGISTERED'),
    ('dataio.py', 'describe_export_source', '_export_file_inventory'): (1, 'FOLDER'),
    ('dataio.py', 'describe_export_source', '_fast_stay_ids'): (1, 'FOLDER'),
    ('dataio.py', 'describe_export_source', 'source_fields'): (1, 'FOLDER'),
    ('dataio.py', 'describe_source_tables', 'require_path'): (1, 'FOLDER'),
    ('dataio.py', 'describe_source_tables', '_safe_dir'): (1, 'FOLDER'),
    ('dataio.py', 'describe_source_tables', '_unpublished_export_error'): (1, 'FOLDER'),
    ('dataio.py', 'describe_source_tables', 'native_planning_metadata'): (1, 'OWNER'),
    ('dataio.py', 'describe_source_tables', 'source_description'): (1, 'FOLDER'),
    ('dataio.py', 'describe_source_tables', '_read_export_manifest'): (1, 'REGISTERED'),
    ('dataio.py', 'describe_source_tables', '_export_file_inventory'): (1, 'FOLDER'),
    ('dataio.py', 'validate_research_pipeline_source', 'prepared_export_manifest_path'): (1, 'REFUSES'),
    ('dataio.py', 'validate_research_pipeline_source', 'inspect_export_layout'): (1, 'REFUSES'),
    ('dataio.py', 'validate_research_pipeline_source', 'open_export_package'): (1, 'REFUSES'),
    ('dataio.py', 'read_prepared_export_manifest', '_read_export_manifest'): (1, 'REGISTERED'),
    ('dataio.py', 'scan_path', 'require_path'): (1, 'FOLDER'),
    ('dataio.py', 'scan_path', '_safe_dir'): (1, 'FOLDER'),
    ('dataio.py', 'scan_path', '_unpublished_export_error'): (1, 'FOLDER'),
    ('dataio.py', 'scan_path', '_detect_database'): (1, 'FOLDER'),
    ('dataio.py', 'scan_path', '_member_refusal'): (2, 'FOLDER'),
    ('dataio.py', 'scan_path', '_check_data_status'): (1, 'FOLDER'),
    ('dataio.py', 'scan_path', 'inspect_export_layout'): (1, 'FOLDER'),
    ('dataio.py', 'scan_path', '_export_source_data_path'): (1, 'FOLDER'),
    ('dataio.py', 'scan_path', '_conversion_quality_for_path'): (1, 'FOLDER'),
    ('dataio.py', 'scan_path', '_estimate_size'): (1, 'FOLDER'),
    ('dataio.py', 'scan_path', 'source_fields'): (1, 'FOLDER'),
    ('dataio.py', 'make_convert_runner', 'require_path'): (1, 'FOLDER'),
    ('dataio.py', 'make_convert_runner.runner', 'require_path'): (1, 'FOLDER'),
    ('dataio.py', 'make_convert_runner.runner', 'DataConverter'): (1, 'FOLDER'),
    ('demo_source_storage.py', 'public_demo_replay_scope', '_verified_synthetic_replay_source'): (1, 'FOLDER'),
    ('demo_source_storage.py', 'public_demo_replay_scope', '_verified_public_demo'): (1, 'FOLDER'),
    ('demo_source_storage.py', 'public_demo_replay_scope', 'egress_source_database'): (1, 'FOLDER'),
    ('demo_source_storage.py', 'public_demo_replay_scope', 'ReplayScope'): (1, 'FOLDER'),
    ('demo_source_storage.py', 'public_demo_replay_scope', 'public_demo_egress_scope'): (1, 'FOLDER'),
    ('demo_source_storage.py', 'egress_source_database', '_verified_public_demo'): (1, 'FOLDER'),
    ('demo_source_storage.py', 'egress_source_database', 'verified_public_demo_raw'): (1, 'FOLDER'),
    ('demo_source_storage.py', 'egress_source_database', 'inspect_export_layout'): (1, 'FOLDER'),
    ('demo_source_storage.py', 'public_demo_egress_scope', '_verified_public_demo'): (1, 'FOLDER'),
    ('demo_source_storage.py', 'public_demo_egress_scope', 'verified_public_demo_raw'): (1, 'FOLDER'),
    ('demo_source_storage.py', 'public_demo_egress_scope', 'synthetic_source_scope'): (1, 'FOLDER'),
    ('eicu_first_observed_source.py', 'derive_from_verified_eicu_demo', 'resolve_raw_source_identity_binding'): (1, 'OWNER'),
    ('eicu_first_observed_source.py', 'derive_from_registered_eicu_export', 'resolve_raw_source_identity_binding'): (1, 'PASS'),
    ('export_download.py', 'prepare_registered_export_bundle', 'require_path'): (2, 'FOLDER'),
    ('export_download.py', 'prepare_registered_export_bundle', 'require_published_native_export'): (1, 'REGISTERED'),
    ('export_download.py', 'prepare_registered_export_bundle', '_manifest_files'): (1, 'REGISTERED'),
    ('export_download.py', 'prepare_registered_export_bundle', 'require_listed_members_present'): (1, 'REGISTERED'),
    ('export_download.py', 'prepare_registered_export_bundle', 'ExportMemberIntegrityError'): (1, 'REGISTERED'),
    ('export_download.py', 'prepare_registered_export_bundle', 'export_relative_path'): (1, 'REGISTERED'),
    ('export_download.py', 'prepare_registered_export_bundle', 'ZipFile'): (1, 'REGISTERED'),
    ('export_download.py', 'prepare_registered_export_bundle', 'ExportDownloadBundle'): (1, 'REGISTERED'),
    ('extraction_filters.py', '_source_provenance', '_hash'): (1, 'FOLDER'),
    ('first_observed_deferred_execution.py', 'consume_approved_first_observed_input', 'resolve_raw_source_identity_binding'): (1, 'PASS'),
    ('first_observed_deferred_execution.py', 'consume_approved_first_observed_input', 'open'): (1, 'REGISTERED'),
    ('first_observed_deferred_execution.py', 'consume_approved_first_observed_input', 'acquire_universe_for_question'): (1, 'REGISTERED'),
    ('first_observed_deferred_recipe.py', 'build_pending_first_observed_review', 'open'): (1, 'REGISTERED'),
    ('first_observed_deferred_recipe.py', 'build_pending_first_observed_review', 'declared_native_export_inventory'): (1, 'REGISTERED'),
    ('first_observed_deferred_recipe.py', 'build_pending_first_observed_review', '_declared_typed_export_columns'): (1, 'REGISTERED'),
    ('first_observed_run_guard.py', 'host_first_observed_review_scope.verify', 'verify_prepared_first_observed_population'): (1, 'REGISTERED'),
    ('first_observed_run_guard.py', 'host_first_observed_review_scope.verify_independence', '_restriction_from_cohort'): (1, 'REGISTERED'),
    ('first_observed_run_guard.py', 'host_first_observed_review_scope.verify_independence', 'verify_prepared_first_observed_population'): (1, 'REGISTERED'),
    ('first_observed_run_guard.py', 'verify_prepared_first_observed_population', 'open'): (1, 'REGISTERED'),
    ('first_observed_run_guard.py', 'verify_prepared_first_observed_population', 'resolve_prepared_first_observed_coordinate'): (1, 'REGISTERED'),
    ('guided_sessions.py', '_trash_guided_project_folder', '_send2trash'): (1, 'FOLDER'),
    ('hospital_outcome_projection.py', 'materialize_web_hospital_status', 'resolve_raw_hospital_source_binding'): (1, 'REGISTERED'),
    ('hosted_warmup.py', 'warm', 'precompute_feature_distributions'): (1, 'REGISTERED'),
    ('ideas/mining.py', 'bounded_sample_feasibility', '_bounded_feature_sample_stat'): (1, 'REGISTERED'),
    ('ideas/mining.py', 'bounded_sample_feasibility', '_bounded_joint_observed_entities'): (1, 'REGISTERED'),
    ('ideas/mining.py', 'bounded_sample_feasibility', '_sha256'): (1, 'FOLDER'),
    ('ideas/mining.py', '_pre_experiment', '_feature_stats'): (1, 'REGISTERED'),
    ('ideas/mining.py', '_pre_experiment', '_sha256'): (1, 'FOLDER'),
    ('ideas/mining.py', 'scan_literature_folder', 'require_path'): (1, 'FOLDER'),
    ('ideas/mining.py', '_active_export', 'describe_export_source'): (1, 'FOLDER'),
    ('ideas/mining.py', 'idea_execution_readiness_binding', '_sha256'): (1, 'FOLDER'),
    ('manuscript_repair.py', 'make_report_only_run_runner', 'validate_research_pipeline_source'): (1, 'REGISTERED'),
    ('patient_drilldown/__init__.py', 'patient_review_sources', 'describe_export_source'): (1, 'FOLDER'),
    ('patient_drilldown/__init__.py', 'patient_review_entity_page', '_read_entity_index_rows'): (1, 'REGISTERED'),
    ('patient_drilldown/__init__.py', 'patient_review_entity_page', 'entity_navigation_payload'): (1, 'REGISTERED'),
    ('patient_drilldown/__init__.py', 'patient_review_entity', '_read_entity_index_rows'): (2, 'REGISTERED'),
    ('patient_drilldown/__init__.py', 'patient_review_entity', '_entity_ref'): (1, 'REGISTERED'),
    ('patient_drilldown/__init__.py', 'patient_review_entity', '_read_module_frame'): (4, 'REGISTERED'),
    ('patient_drilldown/__init__.py', 'patient_review_entity', '_read_review_frames'): (1, 'REGISTERED'),
    ('patient_drilldown/__init__.py', '_source_provenance', '_hash'): (1, 'FOLDER'),
    ('patient_drilldown/__init__.py', 'patient_review_entity', '_entity_option'): (1, 'REGISTERED'),
    ('patient_drilldown/__init__.py', 'patient_review_entity', '_selected_payload'): (1, 'REGISTERED'),
    ('patient_drilldown/__init__.py', 'patient_review_entity', '_trajectory_review_payload'): (1, 'REGISTERED'),
    ('patient_drilldown/__init__.py', 'patient_review_table_preview', '_table_preview_payloads'): (1, 'REGISTERED'),
    ('patient_drilldown/__init__.py', 'patient_review_drilldown', '_read_module_frame'): (5, 'REGISTERED'),
    ('patient_drilldown/__init__.py', 'patient_review_drilldown', '_fallback_entity_frame'): (1, 'REGISTERED'),
    ('patient_drilldown/__init__.py', 'patient_review_drilldown', '_read_review_frames'): (1, 'REGISTERED'),
    ('patient_drilldown/__init__.py', 'patient_review_drilldown', '_entity_ref'): (2, 'REGISTERED'),
    ('patient_drilldown/__init__.py', 'patient_review_drilldown', '_entity_option'): (1, 'REGISTERED'),
    ('patient_drilldown/__init__.py', 'patient_review_drilldown', '_selected_payload'): (1, 'REGISTERED'),
    ('patient_drilldown/__init__.py', 'patient_review_drilldown', 'entity_navigation_payload'): (1, 'REGISTERED'),
    ('patient_drilldown/__init__.py', 'patient_review_drilldown', '_eligibility_flow_payload'): (1, 'REGISTERED'),
    ('patient_drilldown/__init__.py', 'patient_review_drilldown', 'build_feature_coverage'): (1, 'PASS'),
    ('patient_drilldown/__init__.py', 'patient_review_drilldown', '_data_table_review_payload'): (1, 'REGISTERED'),
    ('patient_drilldown/__init__.py', 'patient_review_drilldown', '_trajectory_review_payload'): (1, 'REGISTERED'),
    ('patient_drilldown/coverage.py', 'build_feature_coverage', '_source_signature'): (1, 'FOLDER'),
    ('patient_drilldown/coverage.py', 'build_feature_coverage', 'move_to_end'): (2, 'REGISTERED'),
    ('patient_drilldown/coverage.py', 'build_feature_coverage', '_build_feature_coverage'): (1, 'REGISTERED'),
    ('pi_copilot/demo_planning_authority.py', 'resolve_choice', '_source_facts'): (1, 'OWNER'),
    ('pi_copilot/demo_planning_authority.py', 'publish_choice', '_source_facts'): (2, 'OWNER'),
    ('pi_copilot/demo_planning_source.py', 'resolve_demo_planning_source', 'verified_public_demo_replanning_export'): (1, 'OWNER'),
    ('pi_copilot/demo_planning_source.py', 'resolve_demo_planning_source', 'bound_export_cohort_refusal'): (1, 'OWNER'),
    ('pi_copilot/demo_planning_source.py', 'resolve_demo_planning_source', 'resolve_registered_export_binding'): (1, 'OWNER'),
    ('pi_copilot/demo_planning_source.py', 'resolve_demo_planning_source', 'verified_public_demo_raw'): (1, 'OWNER'),
    ('pi_copilot/demo_planning_source.py', 'resolve_demo_planning_source', 'DemoPlanningSource'): (1, 'OWNER'),
    ('pi_copilot/extraction_handoff.py', '_verify_study_source', 'resolve_registered_export_binding'): (1, 'REGISTERED'),
    ('pi_copilot/install.py', 'install_runtime', 'runtime_manifest'): (1, 'FOLDER'),
    ('pi_copilot/install.py', 'install_runtime', 'runtime_is_installed'): (1, 'FOLDER'),
    ('pi_copilot/install.py', 'install_runtime', '_packaged_source_file'): (3, 'FOLDER'),
    ('pi_copilot/plan_decisions.py', '_requested_endpoint_coordinates', '_metadata_only_planning_coordinates'): (1, 'PASS'),
    ('pi_copilot/projections.py', 'project_study_context', 'path_digest'): (1, 'FOLDER'),
    ('pi_copilot/quick_answers.py', 'QuickAnswerService._submit_quick_answer', 'registered_source_identity'): (1, 'FOLDER'),
    ('pi_copilot/quick_answers.py', 'QuickAnswerService._submit_quick_answer', 'public_demo_egress_scope'): (1, 'FOLDER'),
    ('pi_copilot/quick_answers.py', 'QuickAnswerService._submit_quick_answer', 'prepare_registered_question'): (1, 'PASS'),
    ('pi_copilot/quick_answers.py', 'QuickAnswerService._quick_row', 'registered_source_identity'): (1, 'FOLDER'),
    ('pi_copilot/quick_answers.py', 'QuickAnswerService.approve_quick_answer', 'public_demo_egress_scope'): (1, 'FOLDER'),
    ('pi_copilot/service.py', 'PiCopilotService._session_source_reference', 'registered_source_identity'): (1, 'FOLDER'),
    ('pi_copilot/service.py', 'PiCopilotService.prepare_data_workbench_snapshot', 'describe_export_source'): (1, 'FOLDER'),
    ('pi_copilot/study_context_update.py', '_execution_catalog', 'bound_source_authority_path'): (1, 'OWNER'),
    ('pi_copilot/tools.py', '_review_cohort', 'describe_export_source'): (1, 'FOLDER'),
    ('pi_copilot/tools.py', '_review_cohort', '_eligibility_flow_payload'): (1, 'REGISTERED'),
    ('pi_copilot/tools.py', '_preview_icd_cohort', 'preview_registered_export_icd_cohort'): (1, 'REGISTERED'),
    ('pi_copilot/tools.py', '_bound_registered_export', 'describe_export_source'): (1, 'FOLDER'),
    ('pi_copilot/tools.py', '_assess_idea_feasibility', 'build_available_catalog'): (1, 'REFUSES'),
    ('pi_copilot/tools.py', '_start_extraction', 'path_digest'): (1, 'FOLDER'),
    ('pi_copilot/tools.py', '_list_concept_catalog', 'build_available_catalog'): (1, 'REGISTERED'),
    ('pi_copilot/workflow.py', '_source_groups_stays_at', 'resolve_study_patient_grouping'): (1, 'OWNER'),
    ('pi_copilot/workflow.py', '_study_source_groups_stays', '_source_groups_stays_at'): (1, 'OWNER'),
    ('pi_copilot/workflow.py', 'build_project_workflow_projection', 'bound_export_cohort_refusal'): (1, 'REFUSES'),
    ('quick_answer/preparation.py', 'prepare_registered_question', 'validate_research_pipeline_source'): (1, 'REGISTERED'),
    ('quick_answer/preparation.py', 'prepare_registered_question', '_preparation_budget'): (1, 'FOLDER'),
    ('quick_answer/preparation.py', 'prepare_registered_question', '_verified_identity'): (1, 'PASS'),
    ('quick_answer/preparation.py', 'prepare_registered_question', 'acquire_universe_for_question'): (1, 'REGISTERED'),
    ('quick_answer/preparation.py', 'prepare_registered_question', '_first_stays_orderable'): (1, 'PASS'),
    ('quick_answer/preparation.py', '_verified_identity', 'resolve_study_patient_grouping'): (1, 'PASS'),
    ('quick_answer/preparation.py', '_verified_identity', 'resolve_study_first_icu_stay'): (1, 'PASS'),
    ('quick_answer/preparation.py', '_first_stays_orderable', 'resolve_raw_source_identity_binding'): (1, 'PASS'),
    ('quick_answer/preparation.py', 'prepare_registered_question', 'registered_source_identity'): (1, 'FOLDER'),
    ('raw_case_snapshot.py', 'stage_reviewed_source_files', 'open'): (3, 'REGISTERED'),
    ('raw_preparation_candidate.py', 'create_candidate_for_review', 'candidate_from_scientific_review'): (1, 'REGISTERED'),
    ('raw_preparation_dependencies.py', 'declared_native_export_inventory', 'require_no_in_progress_marker'): (1, 'REGISTERED'),
    ('raw_preparation_dependencies.py', 'declared_native_export_inventory', 'open'): (1, 'REGISTERED'),
    ('raw_preparation_dependencies.py', '_declared_typed_export_columns', 'open'): (1, 'REGISTERED'),
    ('raw_preparation_dependencies.py', 'native_planning_metadata', 'require_no_in_progress_marker'): (1, 'REGISTERED'),
    ('raw_preparation_dependencies.py', 'native_planning_metadata', 'open'): (2, 'REGISTERED'),
    ('raw_preparation_dependencies.py', 'native_planning_metadata', 'declared_native_export_inventory'): (1, 'REGISTERED'),
    ('raw_preparation_dependencies.py', 'native_planning_metadata', '_declared_typed_export_columns'): (1, 'REGISTERED'),
    ('raw_preparation_dependencies.py', 'declaration_from_provenance', 'registered_source_identity'): (2, 'FOLDER'),
    ('raw_preparation_dependencies.py', 'compile_producer_dependencies', '_schema'): (1, 'REGISTERED'),
    ('raw_preparation_dependencies.py', 'candidate_from_scientific_review', 'declaration_from_provenance'): (1, 'REGISTERED'),
    ('raw_preparation_dependencies.py', 'candidate_from_scientific_review', 'compile_producer_dependencies'): (1, 'REGISTERED'),
    ('raw_preparation_projection.py', '_candidate_projection', 'declared_native_export_inventory'): (1, 'REGISTERED'),
    ('raw_preparation_projection.py', '_candidate_projection', '_declared_typed_export_columns'): (1, 'REGISTERED'),
    ('raw_source_authority.py', 'resolve_manifest_raw_source_binding', '_read_export_manifest'): (2, 'REGISTERED'),
    ('raw_source_authority.py', 'resolve_manifest_raw_source_binding', '_regular_descendant'): (1, 'FOLDER'),
    ('raw_source_authority.py', 'resolve_manifest_raw_source_binding', 'RawHospitalSourceBinding'): (1, 'PASS'),
    ('raw_source_authority.py', 'resolve_raw_hospital_source_binding', 'resolve_manifest_raw_source_binding'): (1, 'PASS'),
    ('raw_source_authority.py', 'resolve_raw_hospital_source_binding', 'resolve_raw_mimic_iv_source_binding'): (1, 'PASS'),
    ('raw_source_authority.py', 'resolve_raw_source_identity_binding', 'resolve_raw_hospital_source_binding'): (1, 'REGISTERED'),
    ('raw_source_authority.py', 'resolve_raw_source_identity_binding', 'resolve_manifest_raw_source_binding'): (1, 'PASS'),
    ('raw_source_authority.py', 'resolve_raw_mimic_iv_source_binding', '_regular_descendant'): (1, 'FOLDER'),
    ('raw_source_authority.py', 'resolve_raw_mimic_iv_source_binding', 'RawMimicIVSourceBinding'): (1, 'PASS'),
    ('raw_source_authority.py', '_read_export_manifest', 'unpublished_reason'): (1, 'REGISTERED'),
    ('raw_source_authority.py', '_read_export_manifest', 'open'): (1, 'REGISTERED'),
    ('research_auto_resume.py', '_recover_unstarted_metadata_planner.validate', 'metadata_binding'): (1, 'PASS'),
    ('research_early_resume.py', 'metadata_binding', 'registered_source_identity'): (1, 'FOLDER'),
    ('research_early_resume.py', 'metadata_binding', '_metadata_only_planning_catalog'): (1, 'PASS'),
    ('research_early_resume.py', 'metadata_binding', '_source_concepts_without_values'): (1, 'PASS'),
    ('research_early_resume.py', 'create_seed', 'metadata_binding'): (2, 'PASS'),
    ('research_launch_resume.py', '_development_resume_launch_scope', 'prepared_package_identity'): (1, 'REGISTERED'),
    ('research_launch_resume.py', '_development_resume_launch_scope', 'prepared_package_offer_matches'): (1, 'REGISTERED'),
    ('research_launch_resume.py', '_development_resume_launch_scope', 'validate_research_pipeline_source'): (1, 'REGISTERED'),
    ('research_launch_scientific.py', '_observability_preserving_exposure', 'export_kdigo_strict_derivation_available'): (1, 'PASS'),
    ('research_launch_scientific.py', '_bound_source_column', 'native_planning_metadata'): (1, 'REFUSES'),
    ('research_launch_scientific.py', '_bound_source_column', 'build_available_catalog'): (1, 'REFUSES'),
    ('research_launch_scientific.py', 'bound_source_planning_concepts', 'verified_public_demo_raw'): (1, 'FOLDER'),
    ('research_launch_scientific.py', 'bound_source_planning_concepts', 'is_export_package'): (1, 'OWNER'),
    ('research_launch_scientific.py', 'bound_source_planning_concepts', 'native_planning_metadata'): (1, 'OWNER'),
    ('research_launch_scientific.py', 'bound_source_planning_concepts', 'build_available_catalog'): (1, 'OWNER'),
    # S1540 sweep: these two read the raw Demo folder until routed (fixed in this diff).
    ('research_launch_scientific.py', '_metadata_only_planning_coordinates', 'bound_source_authority_path'): (1, 'OWNER'),
    ('research_launch_scientific.py', '_metadata_only_planning_coordinates', '_observability_preserving_exposure'): (1, 'OWNER'),
    ('research_launch_scientific.py', '_metadata_only_planning_coordinates', '_bound_source_column'): (1, 'OWNER'),
    ('research_launch_scientific.py', '_data_foundation_profile', 'bound_source_authority_path'): (1, 'OWNER'),
    ('research_launch_scientific.py', 'verified_patient_grouping', 'resolve_study_patient_grouping'): (1, 'OWNER'),
    ('research_launch_scientific.py', 'target_trial_launch_inputs', '_data_foundation_profile'): (1, 'PASS'),
    ('research_launch_scientific.py', '_first_icu_stay_for_cohort', 'resolve_study_first_icu_stay'): (1, 'OWNER'),
    ('research_launch_scientific.py', 'bound_export_cohort_refusal', 'read_prepared_export_manifest'): (1, 'REFUSES'),
    ('research_launch_scientific.py', '_require_export_holds_study_cohort', 'bound_export_cohort_refusal'): (1, 'REFUSES'),
    ('research_launch_scientific.py', 'bound_export_selection_basis', 'read_prepared_export_manifest'): (1, 'REGISTERED'),
    ('research_launch_scientific.py', 'bound_export_selection_report', 'read_prepared_export_manifest'): (1, 'REGISTERED'),
    ('research_launch_scientific.py', 'bound_export_event_time_semantics', 'read_prepared_export_manifest'): (1, 'REGISTERED'),
    ('research_launch_scientific.py', '_neutral_materialization_scope', 'bound_source_authority_path'): (1, 'OWNER'),
    ('research_multi_extraction.py', 'prepare_source', 'prepared_export_manifest_path'): (1, 'OWNER'),
    ('research_multi_pipeline.py', 'PipelineDatabaseOwner.execute', 'public_demo_egress_scope'): (1, 'FOLDER'),
    ('research_multi_pipeline.py', 'PipelineDatabaseOwner._provider', 'public_demo_egress_scope'): (1, 'FOLDER'),
    ('research_multi_pipeline.py', 'PipelineDatabaseOwner._provider', 'egress_source_database'): (1, 'FOLDER'),
    ('research_multi_pipeline.py', 'PipelineDatabaseOwner._plan', 'validate_research_pipeline_source'): (1, 'REGISTERED'),
    ('research_multi_pipeline.py', 'PipelineDatabaseOwner._plan', 'make_research_pipeline_run_runner'): (1, 'PASS'),
    ('research_multi_sources.py', 'egress_grant', 'public_demo_egress_scope'): (1, 'FOLDER'),
    ('research_multi_sources.py', 'egress_grant', 'egress_source_database'): (1, 'FOLDER'),
    ('research_pipeline_run_preparation.py', '_prepare_scientific_launch', '_require_export_holds_study_cohort'): (1, 'PASS'),
    ('research_pipeline_run_preparation.py', '_prepare_scientific_launch', '_neutral_materialization_scope'): (1, 'PASS'),
    ('research_pipeline_run_preparation.py', '_prepare_scientific_launch', '_metadata_only_planning_coordinates'): (1, 'PASS'),
    ('research_pipeline_run_preparation.py', '_prepare_scientific_launch', 'target_trial_launch_inputs'): (1, 'PASS'),
    ('research_pipeline_run_preparation.py', '_prepare_scientific_launch', '_data_foundation_profile'): (1, 'PASS'),
    ('research_pipeline_run_preparation.py', '_prepare_scientific_launch', 'validate_research_pipeline_source'): (1, 'REGISTERED'),
    ('research_pipeline_run_preparation.py', 'prepare_target_trial_launch', 'ResearchPipelineLaunchRequest'): (1, 'PASS'),
    ('research_pipeline_run_preparation.py', '_prepare_launch_execution', 'PreparedLaunchExecution'): (1, 'PASS'),
    ('research_pipeline_run_preparation.py', 'prepare_research_pipeline_run', 'registered_source_identity'): (1, 'FOLDER'),
    ('research_plan_revision.py', 'load_prepared_plan_revision', 'validate_research_pipeline_source'): (1, 'REGISTERED'),
    ('research_resume_offer.py', 'offer_key', 'prepared_package_identity'): (1, 'REGISTERED'),
    ('research_resume_offer.py', 'offer_key', '_directory_identity'): (1, 'FOLDER'),
    ('research_resume_package.py', 'prepared_package_identity', 'prepared_export_manifest_path'): (1, 'REGISTERED'),
    ('research_resume_package.py', 'prepared_package_offer_matches', '_package_offer_verdict'): (1, 'PASS'),
    ('research_resume_package.py', '_package_offer_verdict', 'validate_research_pipeline_source'): (1, 'REGISTERED'),
    ('research_resume_package.py', '_package_offer_verdict', 'prepared_package_identity'): (1, 'REGISTERED'),
    ('research_run_submission.py', 'submit_research_run', 'describe_source_tables'): (1, 'FOLDER'),
    ('research_run_submission.py', 'submit_research_run', '_with_repeated_stays_default'): (1, 'OWNER'),
    ('research_run_submission.py', 'submit_research_run', 'prepared_export_manifest_path'): (1, 'OWNER'),
    ('research_run_submission.py', 'submit_research_run', 'validate_research_pipeline_source'): (2, 'REGISTERED'),
    ('research_run_submission.py', 'submit_research_run', 'public_demo_egress_scope'): (1, 'FOLDER'),
    ('research_run_submission.py', 'submit_research_run', 'egress_source_database'): (1, 'FOLDER'),
    ('research_run_submission.py', 'submit_research_run', 'build_agent_context_binding'): (1, 'FOLDER'),
    ('routes/jobs.py', 'submit_extraction', 'registered_source_identity'): (1, 'FOLDER'),
    ('run_population_proof.py', 'verified_first_stay_applied', 'open_export_package'): (1, 'REGISTERED'),
    ('scientific_runtime_projection.py', 'export_kdigo_strict_derivation_available', 'is_export_package'): (1, 'REFUSES'),
    ('scientific_runtime_projection.py', 'export_kdigo_strict_derivation_available', 'open_export_package'): (1, 'REFUSES'),
    ('sic_raw_preparation_grant.py', 'sic_preparation_schema_sha256', 'open'): (2, 'REGISTERED'),
    ('sic_raw_preparation_grant.py', 'verify_sic_raw_preparation_grant', 'registered_source_identity'): (1, 'FOLDER'),
    ('sic_raw_preparation_grant.py', 'verify_sic_raw_preparation_grant', 'prepare_case_duration_metadata_plan'): (1, 'REGISTERED'),
    ('sic_raw_preparation_grant.py', 'verify_sic_raw_preparation_grant', 'sic_preparation_schema_sha256'): (1, 'FOLDER'),
    ('source_identity_authority.py', 'resolve_derived_patient_grouping', 'resolve_raw_source_identity_binding'): (1, 'PASS'),
    ('source_identity_authority.py', 'resolve_study_patient_grouping', 'resolve_patient_grouping_authority'): (1, 'PASS'),
    ('source_identity_authority.py', 'resolve_study_patient_grouping', 'resolve_derived_patient_grouping'): (1, 'PASS'),
    ('source_identity_authority.py', 'resolve_study_first_icu_stay', 'resolve_raw_source_identity_binding'): (1, 'PASS'),
    ('source_identity_authority.py', 'resolve_study_first_icu_stay', 'resolve_study_patient_grouping'): (1, 'PASS'),
    ('source_identity_authority.py', 'resolve_study_first_icu_stay_per_hospitalization', 'resolve_raw_source_identity_binding'): (1, 'PASS'),
    ('source_partition_candidates.py', 'candidate_from_bound_research_context', 'registered_source_identity'): (1, 'FOLDER'),
    ('source_partition_candidates.py', 'discover_source_partition_candidates', '_manifest'): (2, 'REGISTERED'),
    ('sources.py', '_preferred_configured_export', '_path_is_within'): (1, 'FOLDER'),
    ('sources.py', 'resolve_registered_export', 'describe_export_source'): (1, 'FOLDER'),
    ('sources.py', 'save_registry', 'discard'): (1, 'FOLDER'),
    ('target_trial_setup.py', '_compile', 'prepare_target_trial_launch'): (1, 'PASS'),
    ('target_trial_setup.py', '_compile', 'acquire_target_trial_universe'): (1, 'PASS'),
    ('target_trial_setup.py', '_compile', 'target_trial_research_context'): (1, 'PASS'),
    ('target_trial_setup.py', 'submit_target_trial_compile.runner', '_compile'): (1, 'PASS'),
    ('time_origin_source_authority.py', 'attest_registered_export_time_origin', 'registered_source_identity'): (2, 'FOLDER'),
    ('time_origin_source_authority.py', 'attest_registered_export_time_origin', 'open_export_package'): (1, 'REGISTERED'),
    ('time_origin_source_authority.py', 'prepare_case_duration_metadata_plan', 'registered_source_identity'): (2, 'FOLDER'),
    ('time_varying_runtime_projection.py', 'materialize_web_time_varying_input', 'resolve_raw_hospital_source_binding'): (1, 'REGISTERED'),
    ('time_varying_runtime_projection.py', 'materialize_web_hospital_followup', 'materialize_web_hospital_status'): (1, 'PASS'),
    ('time_varying_runtime_projection.py', 'materialize_web_hospital_followup', 'resolve_raw_hospital_source_binding'): (1, 'REGISTERED'),
    # Carbon b14r1 collector gaps (source map behind `or`, `/` and os.path.join
    # members, closures). Two KDIGO remedy reads of the raw folder were BUG and
    # now read source_authority_export_path, so they left the sweep.
    ('agent_outputs.py', '_load_context', '_read_columns'): (1, 'REFUSES'),
    ('agent_outputs.py', '_load_context', '_read_export_frame'): (1, 'REFUSES'),
    ('agent_outputs.py', '_load_context', 'member_unreadable'): (2, 'REFUSES'),
    ('agent_pipeline_runs.py', '_execution_retry_source_cohort', 'load_verified_materialized_cohort_authority'): (1, 'REGISTERED'),
    ('agent_pipeline_runs.py', '_execution_retry_source_trajectory', 'load_verified_materialized_trajectory_authority'): (1, 'REGISTERED'),
    ('cohort_review.py', 'cohort_review_summary', 'source_file_identity'): (2, 'FOLDER'),
    ('data_package_review.py', 'build_registered_data_package_review', '_normalized_path'): (1, 'FOLDER'),
    ('dataio.py', 'summarize_export_workspace', '_quality_row'): (1, 'REFUSES'),
    ('dataio.py', 'summarize_export_workspace', '_read_export_frame'): (1, 'REFUSES'),
    ('dataio.py', 'summarize_export_workspace', '_read_stay_id_frame'): (1, 'REFUSES'),
    ('patient_drilldown/feature_detail.py', 'load_feature_detail', 'read_export_projection'): (1, 'REGISTERED'),
    ('pi_copilot/install.py', 'install_runtime', 'copytree'): (1, 'FOLDER'),
    ('pi_copilot/quick_answers.py', 'QuickAnswerService._transition_quick_question', 'registered_source_identity'): (1, 'REFUSES'),
    ('pi_copilot/session_storage.py', 'SessionStorageMaintenance.restore', 'require_instance_path'): (1, 'FOLDER'),
    ('pi_copilot/tools.py', '_accept_idea_handoff', 'idea_execution_readiness_binding'): (1, 'FOLDER'),
    ('quick_answer/preparation.py', '_preparation_budget', 'ParquetFile'): (1, 'REGISTERED'),
    ('raw_source_authority.py', 'resolve_raw_mimic_iv_source_binding', '_sha256_file'): (1, 'FOLDER'),
    ('research_append_source.py', 'registered_source_changed', 'resolve_registered_export'): (1, 'REFUSES'),
    ('research_append_source.py', 'registered_source_changed', 'validate_research_pipeline_source'): (1, 'REFUSES'),
    ('research_auto_resume.py', 'recover_runner', 'make_research_pipeline_run_runner'): (1, 'PASS'),
    ('research_multi_extraction.py', 'prepare_source', 'validate_research_pipeline_source'): (1, 'REGISTERED'),
    ('research_multi_pipeline.py', 'PipelineDatabaseOwner.reference_target', 'registered_source_identity'): (1, 'REFUSES'),
    ('source_identity_authority.py', 'resolve_patient_grouping_authority', '_regular_file'): (1, 'REGISTERED'),
    ('source_partition_candidates.py', 'discover_source_partition_candidates', 'listdir'): (1, 'REGISTERED'),
    ('source_partition_candidates.py', 'discover_source_partition_candidates', 'open'): (2, 'REGISTERED'),
    ('time_origin_source_authority.py', 'prepare_case_duration_metadata_plan', 'read_schema'): (1, 'REGISTERED'),
    # Carbon b14r2 collector (dict payloads, `+`/f-string/format paths, tuple
    # and walrus binds, class scopes). cohort_review_summary and the patient
    # drilldown resolve a registered source; the checked_* readings go through
    # bound_source_planning_concepts, the raw-Demo capability menu owner (S1506).
    ('agent_runs.py', 'make_agent_run_runner.runner', '_evaluate_gate_with_ledger'): (1, 'FOLDER'),
    ('agent_runs.py', 'make_agent_run_runner.runner', 'safe_artifacts_after_privacy_scan'): (2, 'FOLDER'),
    ('crossdb_review.py', '_crossdb_review_summary_for_sources', 'cohort_review_summary'): (1, 'REGISTERED'),
    ('data_package_review.py', 'build_registered_data_package_review', 'cohort_review_summary'): (1, 'REGISTERED'),
    ('dataio.py', 'make_convert_runner.runner', 'emit'): (1, 'FOLDER'),
    ('hosted_warmup.py', 'warm', 'cohort_review_summary'): (1, 'REGISTERED'),
    ('ideas/mining.py', 'scan_literature_folder', 'IdeaMiningWebError'): (2, 'FOLDER'),
    ('ideas/mining.py', 'scan_literature_folder', '_assert_no_row_payload'): (1, 'FOLDER'),
    ('pi_copilot/demo_planning_source.py', 'bind_demo_planning_source', '_planning_study'): (1, 'OWNER'),
    ('pi_copilot/demo_planning_source.py', 'bind_demo_planning_source', 'publish_choice'): (1, 'OWNER'),
    ('pi_copilot/demo_planning_source.py', 'bind_demo_planning_source', 'upsert_context'): (1, 'OWNER'),
    ('pi_copilot/extraction_handoff.py', 'submit_study_extraction', 'submit'): (1, 'FOLDER'),
    ('pi_copilot/service.py', 'PiCopilotService._bind_registered_source_from_message', 'upsert_context'): (1, 'REGISTERED'),
    ('pi_copilot/service.py', 'PiCopilotService.prepare_data_workbench_snapshot', 'cohort_review_summary'): (1, 'REGISTERED'),
    ('pi_copilot/tools.py', '_review_cohort', 'cohort_review_summary'): (2, 'REGISTERED'),
    ('pi_copilot/tools.py', '_review_patient_timeline', 'patient_review_drilldown'): (1, 'REGISTERED'),
    ('pi_copilot/tools.py', '_review_patient_timeline', 'patient_review_entity_page'): (1, 'REGISTERED'),
    ('pi_copilot/tools.py', '_review_patient_timeline', 'patient_review_feature'): (1, 'REGISTERED'),
    ('pi_copilot/tools.py', '_run', 'submit_agent_run'): (1, 'FOLDER'),
    ('research_launch_scientific.py', '_metadata_only_planning_coordinates', 'checked_exposure_concept'): (1, 'OWNER'),
    ('research_launch_scientific.py', '_metadata_only_planning_coordinates', 'checked_stated_adjustments'): (1, 'OWNER'),
    ('research_launch_scientific.py', '_metadata_only_planning_coordinates', 'checked_stated_subgroups'): (1, 'OWNER'),
    ('research_run_submission.py', 'submit_research_run', 'make_report_only_run_runner'): (1, 'PASS'),
    ('research_run_submission.py', 'submit_research_run', 'make_research_pipeline_run_runner'): (1, 'PASS'),
    ('research_run_submission.py', 'submit_research_run', 'prepare_descriptor'): (1, 'FOLDER'),
    ('sources.py', 'register_source', 'save_registry'): (1, 'REGISTERED'),
}


def test_the_sweep_finds_an_unrouted_reader_by_its_value_not_its_name():
    # B12's shape, under a reader name no list carries, plus a read of the
    # bound source mapping and a one-step local; pure helpers are ignored.
    sites = path_consumer_sites({"synthetic.py": (
        "def scope(study, export_path):\n"
        "    described = folder_modules(export_path, planning_metadata_only=True)\n"
        "    source = study['data_source']\n"
        "    folder = str(source.get('path') or '')\n"
        "    catalog = any_new_catalog_reader(Path(folder).expanduser())\n"
        "    return len(export_path), str(export_path), described, catalog\n"
    )})
    assert sites == Counter({
        ("synthetic.py", "scope", "folder_modules"): 1,
        ("synthetic.py", "scope", "any_new_catalog_reader"): 1,
    })


def test_the_sweep_follows_wrapped_maps_member_paths_and_closures():
    # Carbon b14r1: a source map behind ``or``, a source-map local, a member
    # path built with ``/`` or ``os.path.join``, and an enclosing function's
    # local read inside a closure all still reach the sweep.
    sites = path_consumer_sites({"synthetic.py": (
        "def outer(study, export_path):\n"
        "    reader_a((study.get('data_source') or {}).get('path'))\n"
        "    bound = study.get('data_source') or {}\n"
        "    reader_b(bound['path'])\n"
        "    reader_c(Path(export_path) / 'cohort.parquet')\n"
        "    reader_d(os.path.join(export_path, 'cohort.parquet'))\n"
        "    folder = str(export_path)\n"
        "    def inner():\n"
        "        return reader_e(folder)\n"
        "    return inner\n"
    )})
    assert sites == Counter({
        ("synthetic.py", "outer", "reader_a"): 1,
        ("synthetic.py", "outer", "reader_b"): 1,
        ("synthetic.py", "outer", "reader_c"): 1,
        ("synthetic.py", "outer", "reader_d"): 1,
        ("synthetic.py", "outer.inner", "reader_e"): 1,
    })


def test_the_sweep_follows_text_paths_payloads_and_binds():
    # Carbon b14r2: a path re-spelled as text, a copied source map, a
    # payload dict or ``**`` forward, tuple and walrus binds; a class body's
    # names do not leak into its methods.
    sites = path_consumer_sites({"synthetic.py": (
        "class Owner:\n"
        "    folder = export_path\n"
        "    def method(self):\n"
        "        return reader_x(folder)\n"
        "def outer(study, export_path):\n"
        "    reader_a(f'{export_path}/cohort.parquet')\n"
        "    reader_b(export_path + '/cohort.parquet')\n"
        "    reader_c(export_path.rstrip('/'))\n"
        "    copied = dict(study.get('data_source') or {})\n"
        "    reader_d(copied['path'])\n"
        "    reader_e({'source_path': export_path})\n"
        "    reader_f(**{'export_path': export_path})\n"
        "    folder, name = export_path, 'x'\n"
        "    reader_g(folder)\n"
        "    if (member := Path(export_path)):\n"
        "        reader_h(member)\n"
        "    kwargs = {'export_path': export_path, 'label': 'x'}\n"
        "    reader_i(**kwargs)\n"
        "    reader_j(**{**kwargs, 'label': 'y'})\n"
        "    reader_k(**dict(source_path=export_path))\n"
        "    reader_l(**{name: export_path})\n"
    )})
    assert sites == Counter({("synthetic.py", "outer", f"reader_{letter}"): 1 for letter in "abcdefghijkl"})
    bindings: dict = {}
    path_consumer_sites({"synthetic.py": (
        "def outer(export_path, name):\n"
        "    kwargs = {'export_path': export_path, 'label': 'x'}\n"
        "    reader_i(**kwargs)\n"
        "    reader_l(**{name: export_path})\n"
    )}, bindings)
    assert bindings[("synthetic.py", "outer", "reader_i")] == {("kw", "export_path")}
    assert bindings[("synthetic.py", "outer", "reader_l")] == {("kw", "**")}
    # Carbon b14r4: branches, opaque rebinds and key stores only widen a
    # kwargs dict; ``dict(**kwargs)`` keeps its known keys.
    bindings = {}
    path_consumer_sites({"synthetic.py": (
        "def outer(export_path, cond):\n"
        "    if cond:\n"
        "        branch = {'label': export_path}\n"
        "    else:\n"
        "        branch = {'export_path': export_path}\n"
        "    reader_m(**branch)\n"
        "    opaque = {'export_path': export_path}\n"
        "    opaque = build(export_path)\n"
        "    reader_n(**opaque)\n"
        "    stored = {'export_path': export_path}\n"
        "    stored['label'] = export_path\n"
        "    reader_o(**stored)\n"
        "    merged = {'export_path': export_path}\n"
        "    merged |= {'label': export_path}\n"
        "    reader_p(**merged)\n"
        "    reader_q(**dict(**stored))\n"
        "    if cond:\n"
        "        late = str(export_path)\n"
        "    else:\n"
        "        late = {'export_path': export_path}\n"
        "    reader_r(**late)\n"
        "    payload = {}\n"
        "    payload['options'] = {'path': export_path}\n"
        "    reader_s(payload)\n"
    )}, bindings)
    assert bindings[("synthetic.py", "outer", "reader_m")] == {("kw", "label"), ("kw", "export_path")}
    assert bindings[("synthetic.py", "outer", "reader_n")] == {("kw", "export_path"), ("kw", "**")}
    assert bindings[("synthetic.py", "outer", "reader_o")] == {("kw", "export_path"), ("kw", "label")}
    assert bindings[("synthetic.py", "outer", "reader_p")] == {("kw", "export_path"), ("kw", "label")}
    assert bindings[("synthetic.py", "outer", "reader_q")] == {("kw", "export_path"), ("kw", "label")}
    assert bindings[("synthetic.py", "outer", "reader_r")] == {("kw", "export_path"), ("kw", "**")}
    assert bindings[("synthetic.py", "outer", "reader_s")] == {("pos", 0)}


def test_a_pass_row_must_reach_a_path_parameter_by_its_slot():
    forms = [(["study", "export_path"], {"study", "export_path", "label"})]
    assert _slot_reaches_a_path_parameter(("pos", 1), forms)
    assert _slot_reaches_a_path_parameter(("kw", "export_path"), forms)
    assert not _slot_reaches_a_path_parameter(("kw", "**"), forms)
    assert not _slot_reaches_a_path_parameter(("pos", 0), forms)
    assert not _slot_reaches_a_path_parameter(("kw", "label"), forms)
    assert not _slot_reaches_a_path_parameter(("kw", "export_path"), [])


def test_every_source_path_consumer_is_classified():
    sites = path_consumer_sites(_webserver_sources())
    expected = {key: count for key, (count, _cls) in _CLASSIFIED.items()}
    new = sorted((key, count) for key, count in sites.items() if expected.get(key) != count)
    assert not new, (
        "Classify these calls that receive a source or export path (route a study's bound path through "
        f"bound_source_authority_path, or record why a raw Demo folder cannot reach them): {new}"
    )
    assert not sorted(expected.keys() - sites.keys()), "a classified call is gone; drop its row"
    assert {cls for _count, cls in _CLASSIFIED.values()} <= _CLASSES


def webserver_signatures(sources: dict[str, str]) -> dict[str, list[tuple[list[str], set[str]]]]:
    """Each webserver function or record class: its positional and keyword slots."""

    signatures: dict[str, list[tuple[list[str], set[str]]]] = {}
    for text in sources.values():
        for node in ast.walk(ast.parse(text)):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                arguments = node.args
                positional = [arg.arg for arg in (*arguments.posonlyargs, *arguments.args)]
                if positional and positional[0] in {"self", "cls"}:
                    positional = positional[1:]
                keywords = {*positional, *(arg.arg for arg in arguments.kwonlyargs)}
                signatures.setdefault(node.name, []).append((positional, keywords))
            elif isinstance(node, ast.ClassDef):
                fields = [item.target.id for item in node.body
                          if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name)]
                signatures.setdefault(node.name, []).append((fields, set(fields)))
    return signatures


def _slot_reaches_a_path_parameter(slot: tuple[str, object], forms: list[tuple[list[str], set[str]]]) -> bool:
    kind, value = slot
    for positional, keywords in forms:
        if kind == "kw" and value in keywords and value in PATH_NAMES:
            return True
        if kind == "pos" and isinstance(value, int) and value < len(positional) and positional[value] in PATH_NAMES:
            return True
    return False


def test_every_pass_row_forwards_into_a_classified_webserver_parameter():
    # A PASS row is only sound when its callee's own body is swept here: the
    # callee is a webserver function or record whose parameter receiving the
    # path, by position or keyword, is in PATH_NAMES (Carbon b14r1, b14r2).
    sources = _webserver_sources()
    bindings: dict = {}
    path_consumer_sites(sources, bindings)
    signatures = webserver_signatures(sources)
    unresolved = sorted(
        (key, sorted(bindings.get(key, ()), key=str)) for key, (_count, cls) in _CLASSIFIED.items()
        if cls == "PASS" and not all(_slot_reaches_a_path_parameter(slot, signatures.get(key[2], []))
                                     for slot in bindings.get(key, {("kw", "?")}))
    )
    assert not unresolved, f"classify these by what the callee does with the path: {unresolved}"


def test_no_source_path_consumer_reads_the_raw_folder_for_its_registered_export():
    bugs = sorted(key for key, (_count, cls) in _CLASSIFIED.items() if cls == "BUG")
    assert not bugs, f"route these through bound_source_authority_path: {bugs}"
