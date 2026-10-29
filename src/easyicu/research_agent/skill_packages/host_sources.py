"""Read-only source projection for built-in skills; never import their kernels.

Action adapters and capability owners supply the dispatch links. The explicit
roots below identify runtime entry points called by their generated scripts,
and public helpers for skills whose model is still authored by the Coder.
Excerpts are source evidence, not standalone executable skill scripts.
"""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path
import textwrap
from typing import Any, Literal, NamedTuple, Protocol, Sequence, TypedDict

from ..planning.analysis_method_suite import METHOD_SUITE_REGISTRY
from ..planning.capability_registry import CAPABILITY_REGISTRY
from ..planning.method_adapter_catalog import DIRECT_METHOD_BINDINGS, HIGH_FREQUENCY_METHOD_ADAPTERS


class SourceDefinition(NamedTuple):
    module: str
    qualname: str


class DefinitionLocation(TypedDict):
    qualname: str
    line_start: int
    line_end: int


class HostImplementationFile(TypedDict):
    path: str
    language: Literal["python"]
    size_bytes: int
    sha256: str
    content: str
    role: Literal["host_implementation"]
    source_path: str
    source_sha256: str
    definitions: list[DefinitionLocation]


class SkillSourceSpec(Protocol):
    skill_id: str
    capability_id: str | None
    action_ids: tuple[str, ...]
    included_module_ids: tuple[str, ...]


class MethodSkillSourceError(RuntimeError):
    """A declared host source or dispatch link no longer resolves."""

    code = "method_skill_source_invalid"


_PACKAGE_ROOT = Path(__file__).resolve().parents[2]
_PREFIX = "easyicu.research_agent."
_SELECTION = _PREFIX + "execution.runners.selection"
_DEFINITIONS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
_SUPPORT_FUNCTIONS = frozenset({
    SourceDefinition(_PREFIX + "methods.cluster_proportion", "require_wald_group_variation"),
    SourceDefinition(_PREFIX + "methods.descriptive_inputs", "measurement_provenance_receipt"),
    SourceDefinition(_PREFIX + "methods.source_status", "reconcile_binary_event_presence"),
    SourceDefinition(_PREFIX + "methods.source_status", "reconcile_conditional_event_time"),
})

# These skills' action contracts are narrower than their broad family owner.
_ACTION_ONLY = frozenset({
    "treatment-patterns", "cohort-characterization-table-one",
    "missingness-measurement-audit", "restricted-mean-survival-time",
})
_EXTRA_RUNNERS: dict[str, tuple[str, ...]] = {
    "target-trial-emulation": ("execution.runners.target_trial_executor",),
    "trajectory-phenotyping": ("execution.runners.trajectory_scientific_representation_executor",),
}
_SKILL_ROOTS: dict[str, tuple[SourceDefinition, ...]] = {
    "six-database-cohort-extraction": (SourceDefinition(_PREFIX + "concept_catalog", "extraction_projection"),),
    "clinical-concept-definitions": (SourceDefinition(_PREFIX + "concept_catalog", "clinical_definitions"),),
    "sample-size-and-power": tuple(SourceDefinition(_PREFIX + "skill_packages", name) for name in (
        "two_proportions", "two_means", "survival_events", "prediction_binary",
    )),
    "treatment-patterns": (SourceDefinition(_PREFIX + "skill_packages", "summarize_treatment_patterns"),
                           SourceDefinition(_PREFIX + "methods.competing_risks", "estimate_cif")),
    "dynamic-landmark-prediction": tuple(SourceDefinition(_PREFIX + "methods.dynamic_prediction", name) for name in (
        "build_landmark_feature_matrix", "attach_landmark_outcomes", "evaluate_landmark_probabilities",
    )),
    "trajectory-phenotyping": (SourceDefinition(_PREFIX + "methods.subtype_assignment",
                                               "fit_and_evaluate_early_subtype_assignment"),),
}

# Runtime roots supplement the registry's script-rendering callable. Local
# helper definitions are followed by AST name references, without imports.
_RUNTIME_ROOTS: dict[str, tuple[str, ...]] = {
    "execution.runners.covariate_balance_executor": ("run_covariate_balance",),
    "execution.runners.cluster_sizes_executor": ("run_cluster_sizes",),
    "execution.runners.prediction_subgroup_executor": ("run_prediction_subgroup_step",),
    "execution.runners.prediction_validation_executor": ("run_external_validation_step",),
    "execution.runners.adjusted_association_executor": ("run_adjusted_association_from_env",),
    "execution.runners.landmark_categorical_association_executor": (
        "run_landmark_categorical_cohort", "run_landmark_categorical_primary",
    ),
    "execution.runners.landmark_spline_executor": ("run_landmark_spline_association",),
    "execution.runners.time_varying_executor": ("run_time_varying_association", "run_time_varying_g_method"),
    "execution.runners.survival_primary_executor": ("run_survival_primary",),
    "execution.runners.rmst_executor": ("run_rmst_contrast",),
    "execution.runners.prediction_model_executor": (
        "run_prediction_model", "run_prediction_score_analysis", "run_prediction_benchmark_comparison",
    ),
    "execution.runners.cross_sectional_phenotyping_executor": ("run_primary_phenotyping", "run_phenotyping_diagnostic"),
    "execution.runners.exposure_outcome_distribution_executor": ("run_exposure_outcome_distribution_from_env",),
    "execution.runners.ordered_stratified_executor": ("run_ordered_stratified_from_env",),
    "execution.runners.trajectory_scientific_representation_executor": ("run_trajectory_scientific_representation", "run_repeated_measures"),
    "execution.runners.trajectory_scientific_candidate_executor": ("run_trajectory_scientific_candidate_selection",),
    "execution.runners.trajectory_stability_executor": ("run_trajectory_stability",),
    "execution.runners.target_trial_executor": ("run_target_trial_suite",),
}
_ESTIMATOR_ROOTS: dict[str, tuple[tuple[str, str], ...]] = {
    "execution.runners.covariate_balance_executor": (("methods.covariate_balance", "compute_covariate_balance"),),
    "execution.runners.cluster_sizes_executor": (("methods.cluster_sizes", "compute_cluster_sizes"),),
    "execution.runners.prediction_subgroup_executor": (("methods.prediction_subgroup", "evaluate_prediction_subgroups"), ("methods.prediction_validation", "evaluate_binary_predictions")),
    "execution.runners.trajectory_scientific_representation_executor": (("methods.mixed_effects", "fit_mixed_effects"), ("methods.gee", "fit_gee")),
    "execution.runners.adjusted_association_executor": (("robustness.estimators", "fit_estimator"),),
    "execution.runners.landmark_categorical_association_executor": (
        ("execution.runners.adjusted_association_executor", "run_adjusted_association_from_env"),
        ("robustness.estimators", "fit_estimator"),
    ),
    "execution.runners.landmark_spline_executor": (
        ("execution.runners.landmark_spline_fit", "prepare_landmark_model_population"),
        ("execution.runners.landmark_spline_fit", "fit_binomial_model"),
        ("execution.runners.landmark_spline_fit", "restricted_cubic_spline_basis"),
    ),
    "execution.runners.time_varying_executor": (
        ("methods.time_varying_exposure_cox", "fit_cluster_robust_time_varying_cox"),
        ("methods.msm", "fit_longitudinal_msm"), ("methods.gformula", "gformula_ate"),
        ("contracts.time_varying_exposure", "GMethodSpec"),
    ),
    "execution.runners.survival_primary_executor": (
        ("methods.ph_schoenfeld", "ph_test"),
        ("methods.competing_risks", "estimate_cif"),
        ("methods.competing_risks", "fit_fine_gray"),
        ("methods.gray_test", "gray_test"),
    ),
    "execution.runners.rmst_executor": (("methods.rmst", "rmst"), ("methods.rmst", "rmst_difference")),
    "execution.runners.ordered_stratified_executor": (
        ("methods.ordered_trends", "cochran_armitage_trend"),
        ("methods.ordered_trends", "jonckheere_terpstra_trend"),
        ("execution.runners.adjusted_association_executor", "run_adjusted_association_from_env"),
        ("robustness.estimators", "fit_estimator"),
    ),
    "execution.runners.exposure_outcome_distribution_executor": (
        ("methods.cluster_proportion", "patient_weighted_hoeffding_interval"),
        ("methods.cluster_proportion", "require_wald_group_variation"),
    ),
    "execution.runners.prediction_model_executor": (("methods.prediction_validation", "evaluate_binary_predictions"),),
    "execution.runners.trajectory_scientific_candidate_executor": (("execution.runners.trajectory_stability_executor", "_fit_with_engine"),),
    "execution.runners.trajectory_stability_executor": (("trajectory.mixed_mode_latent_class", "fit_observed_data_mixed_mode_lca"),),
    "execution.runners.target_trial_executor": (
        ("methods.clone_censor_weight", "estimate_clone_censor_weight"),
        ("methods.clone_censor_weight", "bootstrap_clone_censor_weight"),
    ),
}


class _Sources:
    """One projection's consistent byte snapshots, discarded after the read."""

    def __init__(self, package_root: Path) -> None:
        self.root = package_root
        self.modules: dict[str, tuple[Path, bytes, ast.Module]] = {}

    def read(self, module: str) -> tuple[Path, bytes, ast.Module]:
        if module not in self.modules:
            parts = module.split(".")
            if parts[0] != "easyicu" or any(not part.isidentifier() for part in parts):
                raise MethodSkillSourceError(f"invalid source module: {module}")
            path = self.root.joinpath(*parts[1:]).with_suffix(".py")
            if not path.is_file():
                path = self.root.joinpath(*parts[1:], "__init__.py")
            try:
                content = path.read_bytes()
                tree = ast.parse(content.decode("utf-8"), filename=str(path))
            except (OSError, UnicodeError, SyntaxError) as exc:
                raise MethodSkillSourceError(f"unreadable source module: {module}") from exc
            self.modules[module] = (path, content, tree)
        return self.modules[module]

    def definition(self, source: SourceDefinition) -> ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef:
        node: ast.AST = self.read(source.module)[2]
        for part in source.qualname.split("."):
            matches = [child for child in getattr(node, "body", ())
                       if isinstance(child, _DEFINITIONS) and child.name == part]
            if len(matches) != 1:
                raise MethodSkillSourceError(f"definition not unique or missing: {source.module}:{source.qualname}")
            node = matches[0]
        return node


def _registered_render_sources(sources: _Sources) -> tuple[SourceDefinition, ...]:
    """Resolve the actual StepExecutor render lambdas using only their syntax."""
    tree = sources.read(_SELECTION)[2]
    imports: dict[str, SourceDefinition] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            base = ".".join(_SELECTION.split(".")[:-node.level]) if node.level else ""
            module = ".".join(part for part in (base, node.module) if part)
            for alias in node.names:
                imports[alias.asname or alias.name] = SourceDefinition(module, alias.name)
    found = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "StepExecutor"):
            continue
        render = next((arg.value for arg in node.keywords if arg.arg == "render"), None)
        body = render.body if isinstance(render, ast.Lambda) else render
        if isinstance(body, ast.Call) and isinstance(body.func, ast.Name):
            target = imports.get(body.func.id, SourceDefinition(_SELECTION, body.func.id))
            found.append(target)
    return tuple(dict.fromkeys(found))


def _source_roots(specs: Sequence[SkillSourceSpec], sources: _Sources) -> dict[str, tuple[SourceDefinition, ...]]:
    from ..planning.method_adapter_catalog import get_method_adapter_contract

    capabilities = {row.capability_id: row for row in CAPABILITY_REGISTRY}
    adapters = {row.action_id: row for row in HIGH_FREQUENCY_METHOD_ADAPTERS}
    methods = {f"{suite.family}.{method.key}": method for suite in METHOD_SUITE_REGISTRY for method in suite.methods}
    registered = _registered_render_sources(sources)
    result = {}
    for spec in specs:
        roots = list(_SKILL_ROOTS.get(spec.skill_id, ()))
        modules = list(_EXTRA_RUNNERS.get(spec.skill_id, ()))
        if getattr(spec, "execution_mode", "") == "agent_coded_with_host_gates":
            roots.extend((
                SourceDefinition(_PREFIX + "execution.initial_code_policy", "resolve_initial_code_origin"),
                SourceDefinition(_PREFIX + "execution.step_execution", "initial_code_origin"),
            ))
        for action in spec.action_ids:
            adapter = adapters.get(action) or get_method_adapter_contract(action)
            if adapter is not None:
                roots.append(SourceDefinition(adapter.owner_module, adapter.owner_entrypoint))
            elif (method := methods.get(action)) is not None and method.runner is not None:
                modules.extend(cap.primary_runner_module for cap in CAPABILITY_REGISTRY
                               if cap.primary_runner == method.runner and cap.primary_runner_module)
        capability = capabilities.get(spec.capability_id)
        if capability is not None and capability.primary_runner_module and spec.skill_id not in _ACTION_ONLY:
            modules.append(capability.primary_runner_module)
        for module in dict.fromkeys(modules):
            matches = [source for source in registered if source.module == _PREFIX + module]
            if not matches:
                raise MethodSkillSourceError(f"executor render binding missing: {module}")
            roots.extend(matches)
        for module in dict.fromkeys(source.module for source in roots):
            relative = module.removeprefix(_PREFIX)
            roots.extend(SourceDefinition(module, name) for name in _RUNTIME_ROOTS.get(relative, ()))
            roots.extend(SourceDefinition(_PREFIX + owner, name) for owner, name in _ESTIMATOR_ROOTS.get(relative, ()))
            for binding in DIRECT_METHOD_BINDINGS:
                if binding.owner_module == module:
                    roots.extend(SourceDefinition(binding.direct_module, name) for name in binding.direct_symbols)
        if not roots:
            raise MethodSkillSourceError(f"skill has no host sources: {spec.skill_id}")
        result[spec.skill_id] = tuple(dict.fromkeys(roots))
    return result


def method_skill_source_map(
    specs: Sequence[SkillSourceSpec], *, skill_ids: Sequence[str] | None = None,
    package_root: Path = _PACKAGE_ROOT,
) -> dict[str, tuple[SourceDefinition, ...]]:
    """Map skill ids to ordered host definitions, including local helper closure."""
    sources = _Sources(package_root)
    roots = _source_roots(specs, sources)
    by_id = {spec.skill_id: spec for spec in specs}
    result = {}
    for skill_id in roots if skill_ids is None else skill_ids:
        if skill_id not in roots:
            raise MethodSkillSourceError(f"unknown skill source binding: {skill_id}")
        pending = list(roots[skill_id])
        for module_id in by_id[skill_id].included_module_ids:
            pending.extend(roots[module_id])
        selected: dict[SourceDefinition, None] = {}
        while pending:
            source = pending.pop(0)
            if source in selected:
                continue
            node = sources.definition(source)
            selected[source] = None
            local = {child.name for child in sources.read(source.module)[2].body if isinstance(child, _DEFINITIONS)}
            names = {child.id for child in ast.walk(node) if isinstance(child, ast.Name)}
            pending.extend(SourceDefinition(source.module, name) for name in sorted(names & local))
        module_order = {module: index for index, module in enumerate(dict.fromkeys(source.module for source in selected))}
        result[skill_id] = tuple(sorted(selected, key=lambda source: (
            module_order[source.module], sources.definition(source).lineno,
        )))
    return result


def method_skill_implementation_files(
    skill_id: str, specs: Sequence[SkillSourceSpec], *, package_root: Path = _PACKAGE_ROOT,
) -> list[HostImplementationFile]:
    """Read current source bytes and add digest-bound, decorated excerpts."""
    mapping = method_skill_source_map(specs, skill_ids=(skill_id,), package_root=package_root)
    sources = _Sources(package_root)
    return _implementation_files(mapping[skill_id], sources)


def _implementation_files(
    definitions: Sequence[SourceDefinition], sources: _Sources,
) -> list[HostImplementationFile]:
    grouped: dict[str, list[SourceDefinition]] = {}
    for source in definitions:
        grouped.setdefault(source.module, []).append(source)
    files: list[HostImplementationFile] = []
    for module, definitions in grouped.items():
        path, raw, _ = sources.read(module)
        source_path = "src/easyicu/" + path.relative_to(sources.root).as_posix()
        source_sha256 = hashlib.sha256(raw).hexdigest()
        lines = raw.decode("utf-8").splitlines(keepends=True)
        locations: list[DefinitionLocation] = []
        excerpts = []
        for source in definitions:
            node = sources.definition(source)
            start = min([node.lineno, *(decorator.lineno for decorator in node.decorator_list)])
            end = node.end_lineno
            if end is None:
                raise MethodSkillSourceError(f"definition has no end line: {source}")
            locations.append({"qualname": source.qualname, "line_start": start, "line_end": end})
            excerpts.append(f"# {source_path}:{start}-{end} sha256={source_sha256}\n"
                            + "".join(lines[start - 1:end]))
        content = "\n\n".join(excerpts)
        encoded = content.encode("utf-8")
        files.append({
            "path": f"implementation/{module.removeprefix('easyicu.')}.py",
            "language": "python", "size_bytes": len(encoded),
            "sha256": hashlib.sha256(encoded).hexdigest(), "content": content,
            "role": "host_implementation", "source_path": source_path,
            "source_sha256": source_sha256, "definitions": locations,
        })
    return files


def _definition_signature(node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) -> str:
    """Display syntax only: annotations and defaults are never evaluated."""
    if isinstance(node, ast.ClassDef):
        arguments = [*(ast.unparse(base) for base in node.bases),
                     *(ast.unparse(keyword) for keyword in node.keywords)]
        return "class " + node.name + ("(" + ", ".join(arguments) + ")" if arguments else "")
    prefix = "async def " if isinstance(node, ast.AsyncFunctionDef) else "def "
    returns = " -> " + ast.unparse(node.returns) if node.returns is not None else ""
    return prefix + node.name + "(" + ast.unparse(node.args) + ")" + returns


def implementation_metadata(
    *, mode: str, summary_en: str, summary_zh: str,
    bindings: dict[SourceDefinition, str], files: Sequence[HostImplementationFile],
) -> dict[str, Any]:
    """Describe the same byte excerpts shipped in the package, without imports.

    Original-source and excerpt coordinates are distinct. Parsing the shipped
    excerpt also keeps signatures/docstrings consistent with its hash if a
    source file changes during a later request.
    """
    entrypoints = []
    for file in files:
        module = "easyicu." + file["path"].removeprefix("implementation/").removesuffix(".py")
        lines = file["content"].splitlines(keepends=True)
        cursor = 0
        for location in file["definitions"]:
            count = location["line_end"] - location["line_start"] + 1
            # Each segment has a provenance comment, then verbatim source;
            # two joining newlines follow the preceding source segment.
            while cursor < len(lines) and not lines[cursor].startswith("# " + file["source_path"] + ":"):
                cursor += 1
            first = cursor + 1
            source = SourceDefinition(module, location["qualname"])
            role = bindings.get(source)
            if role is not None:
                excerpt = "".join(lines[first:first + count])
                # A selected method can be indented in its original class.
                tree = ast.parse(textwrap.dedent(excerpt))
                node = next(child for child in tree.body if isinstance(child, _DEFINITIONS))
                entrypoints.append({
                    "name": node.name, "module": module, "qualname": source.qualname,
                    "signature": _definition_signature(node),
                    "source_path": file["source_path"], "source_sha256": file["source_sha256"],
                    "line_start": location["line_start"], "line_end": location["line_end"],
                    "file_path": file["path"], "excerpt_line_start": first + 1,
                    "excerpt_line_end": first + count, "role": role,
                    "docstring": ast.get_docstring(node) or "",
                })
            cursor = first + count
    return {"mode": mode, "summary_en": summary_en, "summary_zh": summary_zh,
            "entrypoints": entrypoints, "examples": []}


def method_skill_implementation_metadata(
    spec: SkillSourceSpec, files: Sequence[HostImplementationFile],
    specs: Sequence[SkillSourceSpec],
) -> dict[str, Any]:
    """Expose curated roots; local helper closure stays in the source files."""
    roots = _source_roots(specs, _Sources(_PACKAGE_ROOT))
    selected = list(roots[spec.skill_id])
    for module_id in spec.included_module_ids:
        selected.extend(roots[module_id])
    mode = ("owner_projection" if getattr(spec, "layer", "") == "data_definition" else
            "generated_code" if getattr(spec, "execution_mode", "") == "agent_coded_with_host_gates" else
            "fixed_host")
    summaries = {
        "owner_projection": (
            "Read-only metadata projection from the existing data/definition owner. These functions do not run a study.",
            "从既有数据与定义模块读取元数据。所列函数不执行研究。"),
        "generated_code": (
            "The model writes study-specific code from the reviewed plan. Listed host functions are helpers, code-origin gates or controlled built-in branches; shared runners are not an independent implementation of this whole method.",
            "模型按已审阅的方案编写本次研究代码。所列宿主函数是辅助函数、代码来源检查或受控内置分支；共享运行器不代表整个方法已有独立的固定算法。"),
        "fixed_host": (
            "Existing host functions implement the declared workflow or module. Shared entrypoints dispatch reviewed subcontracts; the source excerpts include supporting helpers, not a separate algorithm for every named method.",
            "既有宿主函数实现所声明的工作流或模块。共享入口按已审阅的具体合同分派；源码片段包含辅助函数，并不表示每个方法名称都有独立算法。"),
    }
    bindings = {}
    for source in selected:
        gate = source.module.endswith(("execution.initial_code_policy", "execution.step_execution"))
        support = source.qualname.endswith(("_code", "_scaffold", "_sha256")) or source in _SUPPORT_FUNCTIONS
        bindings[source] = ("gate" if gate else "helper" if mode == "generated_code" else
                            "helper" if support else
                            "kernel" if ".methods." in source.module or ".robustness." in source.module else
                            "helper" if ".contracts." in source.module else "entrypoint")
    return implementation_metadata(mode=mode, summary_en=summaries[mode][0], summary_zh=summaries[mode][1],
                                   bindings=bindings, files=files)
