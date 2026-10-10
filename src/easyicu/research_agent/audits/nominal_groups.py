"""A nominal exposure grouping's codes enter a script's models only as categories.

Owner
-----
A study's nominal exposure grouping is staged as integer level codes
(``contracts.exposure_group_rules``): ``1`` to ``k`` name its groups and
``k + 1`` its unmeasured stays.  The codes carry neither order nor spacing.
The deterministic code gate (``audits.patterns``) therefore asks this owner
whether a step's script enters them in a linear model as one numeric term --
outside ``C(...)`` in a formula, or in a linear model's design without
indicator coding -- or reads them as the ordered levels of an
ordered-stratified analysis (``contracts.ordered_stratified``).  An ordinal
grouping's codes lie along its scale and pass.

The check reads the script's syntax only.  It follows names bound from
calls (a split, a copy), formulas assembled from pieces, and pipelines that
end in a linear model; a design built where no line names its columns is
not seen.  An outcome is no term -- the left of a formula's ``~``, a fit's
``y``, a model's ``endog`` -- and a column a table drops is not in it.
"""

from __future__ import annotations

import ast
import re
from typing import Dict, List, Optional, Sequence, Set, Tuple

from ..contracts.exposure_group_rules import EXPOSURE_GROUP_TRANSFORM_ID
from ..contracts.ordered_stratified import is_ordered_stratified_analysis_step
from ..schema import AnalysisStep, ResearchContext, ValidationFinding


#: Models whose design takes each feature column as one numeric term.
_LINEAR_TERM_MODELS = frozenset(
    {
        "Logit",
        "OLS",
        "GLM",
        "LogisticRegression",
        "LinearRegression",
        "LogisticRegressionCV",
        "Ridge",
        "RidgeCV",
        "Lasso",
        "LassoCV",
        "ElasticNet",
        "ElasticNetCV",
        "PoissonRegressor",
        "GammaRegressor",
        "TweedieRegressor",
        "GEE",
        "MixedLM",
        "MNLogit",
        "Poisson",
        "NegativeBinomial",
        "Probit",
        "WLS",
        "GLS",
        "QuantReg",
        "CoxPHFitter",
        "WeibullAFTFitter",
        "LogNormalAFTFitter",
        "LogLogisticAFTFitter",
    }
)
#: Arguments naming strata, clusters, groups, weights or a survival
#: outcome's columns, which a model takes out of its design.
_NON_TERM_ARGUMENTS = frozenset(
    {
        "strata",
        "cluster_col",
        "weights_col",
        "entry_col",
        "duration_col",
        "event_col",
        "groups",
        "sample_weight",
        "formula",
    }
)
#: Arguments that are no part of a design: the outcome a model predicts, and
#: a helper's estimator, folds and options.
_NOT_DESIGN_ARGUMENTS = frozenset(
    {"y", "endog", "estimator", "cv", "scoring", "cov_kwds", "family"}
)
#: Helpers that fit the model they are given on the design they are given.
_TRAINING_HELPERS = frozenset(
    {
        "cross_val_score",
        "cross_val_predict",
        "cross_validate",
        "learning_curve",
        "validation_curve",
        "permutation_test_score",
    }
)
#: Patsy designs whose first argument is a formula, with or without ``~``.
_DESIGN_FORMULA_CALLS = frozenset({"dmatrix", "dmatrices"})
#: Types under which a formula reads a column as categories.
_CATEGORY_TYPES = frozenset({"category", "str", "string", "object"})
_CATEGORICAL_TERM = re.compile(r"(?<![\w.])C\(")
#: A piece of a formula the script does not spell, by its index.
_TOKEN = "\x00{}\x00"
_TOKEN_PATTERN = re.compile("\x00(\\d+)\x00")
#: Calls that change a bound list after it is bound.
_LIST_CHANGES = frozenset({"append", "extend", "insert"})


def _without_categorical_terms(formula: str) -> str:
    """``formula`` without its ``C(...)`` terms, nested parentheses included."""

    kept: List[str] = []
    position = 0
    while (match := _CATEGORICAL_TERM.search(formula, position)) is not None:
        kept.append(formula[position : match.start()])
        depth, position = 1, match.end()
        while position < len(formula) and depth:
            depth += {"(": 1, ")": -1}.get(formula[position], 0)
            position += 1
    kept.append(formula[position:])
    return "".join(kept)


def _columns_read_by(node: ast.AST, reads: Dict[str, Set[str]]) -> Set[str]:
    """Every column name ``node`` spells or reaches through a bound name.

    A table's ``drop(...)`` names the columns it leaves out: they are not
    read into what it returns.
    """

    if isinstance(node, ast.Constant):
        return {node.value} if isinstance(node.value, str) else set()
    if isinstance(node, ast.Name):
        return set(reads.get(node.id, set()))
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "drop"
    ):
        dropped: Set[str] = set()
        for argument in _call_arguments(node):
            dropped |= _columns_read_by(argument, reads)
        return _columns_read_by(node.func.value, reads) - dropped
    columns: Set[str] = set()
    for child in ast.iter_child_nodes(node):
        columns |= _columns_read_by(child, reads)
    return columns


def _holds_formula_text(node: ast.AST) -> bool:
    return any(
        isinstance(sub, ast.Constant)
        and isinstance(sub.value, str)
        and "~" in sub.value
        for sub in ast.walk(node)
    )


def _formula_expressions(tree: ast.Module) -> List[ast.AST]:
    """Expressions that spell a model formula, however it is assembled.

    A string with ``~``; a concatenation, f-string or ``format`` call holding
    one; a ``formula=`` argument; and a patsy design's formula.
    """

    found: List[ast.AST] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Constant, ast.BinOp, ast.JoinedStr)):
            if _holds_formula_text(node):
                found.append(node)
        elif isinstance(node, ast.Call):
            receiver = node.func.value if isinstance(node.func, ast.Attribute) else None
            name = _call_target_name(node)
            if (
                name == "format"
                and receiver is not None
                and _holds_formula_text(receiver)
            ):
                found.append(node)
            if name in _DESIGN_FORMULA_CALLS and node.args:
                found.append(node.args[0])
            found.extend(
                keyword.value for keyword in node.keywords if keyword.arg == "formula"
            )
    # A piece of a formula is read as part of the whole formula.
    inner = {id(sub) for node in found for sub in ast.walk(node) if sub is not node}
    return [node for node in found if id(node) not in inner]


class _SpelledFormula:
    """A formula expression spelled out from the strings the script binds.

    A name bound once is spelled as its value; a list joined into terms --
    spelled out, or made by a comprehension over one -- an f-string and a
    ``format`` call as the text they make.  A piece the
    script does not spell -- a name bound otherwise, more than once or
    changed after it was bound, a computed value -- stands as a token for
    the nominal columns it reaches, which count as bare wherever it stands.
    """

    def __init__(
        self,
        bound: Dict[str, List[ast.AST]],
        changed: Set[str],
        reads: Dict[str, Set[str]],
        nominal: Set[str],
    ) -> None:
        self._bound = bound
        self._changed = changed
        self._reads = reads
        self._nominal = nominal
        self._tokens: List[Set[str]] = []
        self._spelling: Set[str] = set()
        #: A comprehension's name, spelled as the item it stands for.
        self._items_at: Dict[str, str] = {}

    def bare_terms(self, expression: ast.AST) -> Set[str]:
        """Nominal columns the formula takes outside any ``C(...)``.

        The left of ``~`` is the outcome the model predicts, not a term.
        """

        spelled = self.spell(expression)
        return self._bare(spelled.split("~", 1)[1] if "~" in spelled else spelled)

    def _bare(self, text: str) -> Set[str]:
        kept = _without_categorical_terms(text)
        bare = {
            name
            for name in self._nominal
            if re.search(rf"(?<![\w.]){re.escape(name)}(?!\w)", kept)
        }
        for match in _TOKEN_PATTERN.finditer(kept):
            bare |= self._tokens[int(match.group(1))]
        return bare

    def _token(self, columns: Set[str]) -> str:
        self._tokens.append(columns & self._nominal)
        return _TOKEN.format(len(self._tokens) - 1)

    def spell(self, node: ast.AST) -> str:
        if isinstance(node, ast.Constant):
            return str(node.value)
        if isinstance(node, ast.JoinedStr):
            return "".join(self.spell(value) for value in node.values)
        if isinstance(node, ast.FormattedValue):
            return self.spell(node.value)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            return self.spell(node.left) + self.spell(node.right)
        if isinstance(node, ast.Name):
            return self._spell_name(node)
        if isinstance(node, ast.Call):
            spelled = self._spell_call(node)
            if spelled is not None:
                return spelled
        return self._token(_columns_read_by(node, self._reads))

    def _spell_name(self, node: ast.Name) -> str:
        if node.id in self._items_at:
            return self._items_at[node.id]
        values = self._bound.get(node.id, [])
        if node.id in self._spelling or node.id in self._changed or not values:
            return self._token(_columns_read_by(node, self._reads))
        self._spelling.add(node.id)
        try:
            if len(values) == 1 and not isinstance(values[0], (ast.List, ast.Tuple)):
                return self.spell(values[0])
            # Bound more than once: any of its values may be the one read.
            reached: Set[str] = set()
            for value in values:
                if isinstance(value, (ast.List, ast.Tuple)):
                    reached |= _columns_read_by(value, self._reads)
                else:
                    reached |= self._bare(self.spell(value))
            return self._token(reached)
        finally:
            self._spelling.discard(node.id)

    def _items(self, node: ast.AST) -> Optional[List[str]]:
        if isinstance(node, ast.Name):
            values = self._bound.get(node.id, [])
            if node.id in self._changed or len(values) != 1:
                return None
            node = values[0]
        if isinstance(node, (ast.List, ast.Tuple)):
            return [self.spell(element) for element in node.elts]
        if isinstance(node, (ast.GeneratorExp, ast.ListComp)):
            return self._comprehension_items(node)
        return None

    def _comprehension_items(
        self, node: "ast.GeneratorExp | ast.ListComp"
    ) -> Optional[List[str]]:
        """``f"C({c})" for c in names``, spelled item by item; unfiltered only."""

        if len(node.generators) != 1:
            return None
        loop = node.generators[0]
        if loop.ifs or loop.is_async or not isinstance(loop.target, ast.Name):
            return None
        items = self._items(loop.iter)
        if items is None:
            return None
        name = loop.target.id
        outer = self._items_at.get(name)
        spelled: List[str] = []
        try:
            for item in items:
                self._items_at[name] = item
                spelled.append(self.spell(node.elt))
        finally:
            if outer is None:
                self._items_at.pop(name, None)
            else:
                self._items_at[name] = outer
        return spelled

    def _spell_call(self, node: ast.Call) -> Optional[str]:
        func = node.func
        if not isinstance(func, ast.Attribute):
            return None
        if func.attr == "join" and len(node.args) == 1 and not node.keywords:
            items = self._items(node.args[0])
            return None if items is None else self.spell(func.value).join(items)
        if func.attr != "format":
            return None
        positional = list(node.args)
        named = {keyword.arg: keyword.value for keyword in node.keywords if keyword.arg}
        following = iter(positional)

        def fill(match: "re.Match[str]") -> str:
            field = match.group(1)
            if not field:
                value = next(following, None)
            elif field.isdigit():
                index = int(field)
                value = positional[index] if index < len(positional) else None
            else:
                value = named.get(field)
            return self._token(set()) if value is None else self.spell(value)

        return re.sub(r"\{(\w*)\}", fill, self.spell(func.value))


def _bindings_by_name(
    bindings: Sequence[ast.Assign], tree: ast.Module
) -> Tuple[Dict[str, List[ast.AST]], Dict[str, List[ast.AST]]]:
    """Each name's bound values, and what changes a name after it was bound.

    A list grows by ``append``, ``extend`` or ``insert``, a name by ``+=``;
    a loop or an annotated or ``:=`` binding is a change too.
    """

    bound: Dict[str, List[ast.AST]] = {}
    for node in bindings:
        for target in node.targets:
            if isinstance(target, ast.Name):
                bound.setdefault(target.id, []).append(node.value)
    changes: Dict[str, List[ast.AST]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
            changes.setdefault(node.target.id, []).append(node.value)
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in _LIST_CHANGES
            and isinstance(node.func.value, ast.Name)
        ):
            changes.setdefault(node.func.value.id, []).extend(_call_arguments(node))
        elif isinstance(node, (ast.For, ast.comprehension)):
            for name in ast.walk(node.target):
                if isinstance(name, ast.Name):
                    changes.setdefault(name.id, []).append(node.iter)
        elif isinstance(node, (ast.AnnAssign, ast.NamedExpr)) and isinstance(
            node.target, ast.Name
        ):
            if node.value is not None:
                changes.setdefault(node.target.id, []).append(node.value)
    return bound, changes


def _names_category(node: ast.AST) -> bool:
    return (isinstance(node, ast.Constant) and str(node.value) in _CATEGORY_TYPES) or (
        isinstance(node, ast.Name) and node.id == "str"
    )


def _categorical_columns(tree: ast.Module, nominal: Set[str]) -> Set[str]:
    """Nominal columns cast to categories or text, which a formula reads as groups."""

    cast: Set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _call_target_name(node)
        if name == "Categorical":
            cast |= _columns_read_by(node, {}) & nominal
        elif name == "astype" and isinstance(node.func, ast.Attribute):
            for argument in node.args:
                if isinstance(argument, ast.Dict):
                    cast |= {
                        str(key.value)
                        for key, value in zip(argument.keys, argument.values)
                        if isinstance(key, ast.Constant) and _names_category(value)
                    } & nominal
                elif _names_category(argument):
                    cast |= _columns_read_by(node.func.value, {}) & nominal
    return cast


def _call_arguments(node: ast.Call) -> List[ast.AST]:
    return [*node.args, *(keyword.value for keyword in node.keywords)]


def _indicator_coded_columns(
    tree: ast.Module, reads: Dict[str, Set[str]], nominal: Set[str]
) -> Set[str]:
    """Nominal columns given one indicator column per group.

    ``get_dummies`` codes the columns its ``columns=`` names, or the one
    column it is given; given a table alone it codes no integer column.  A
    one-hot encoder codes the columns of its column-transformer entry, or
    those it is fitted on.
    """

    parents = {
        child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)
    }
    encoders = {
        target.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Call)
        and _call_target_name(node.value) == "OneHotEncoder"
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    coded: Set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _call_target_name(node)
        receiver = node.func.value if isinstance(node.func, ast.Attribute) else None
        if name == "get_dummies":
            named = [
                keyword.value for keyword in node.keywords if keyword.arg == "columns"
            ]
            if named:
                coded |= _columns_read_by(named[0], reads)
            elif node.args and (
                isinstance(node.args[0], ast.Attribute)
                or (
                    isinstance(node.args[0], ast.Subscript)
                    and isinstance(node.args[0].slice, ast.Constant)
                )
            ):
                coded |= _columns_read_by(node.args[0], {})
        elif name == "OneHotEncoder":
            parent = parents.get(node)
            if isinstance(parent, ast.Tuple):
                coded |= {
                    column
                    for element in parent.elts
                    if element is not node
                    for column in _columns_read_by(element, reads)
                }
            elif isinstance(parent, ast.Attribute) and isinstance(
                parents.get(parent), ast.Call
            ):
                for argument in _call_arguments(parents[parent]):
                    coded |= _columns_read_by(argument, reads)
        elif (
            name in {"fit", "fit_transform", "transform"}
            and isinstance(receiver, ast.Name)
            and receiver.id in encoders
        ):
            for argument in _call_arguments(node):
                coded |= _columns_read_by(argument, reads)
    return coded & nominal


def _builds_linear_model(node: ast.Call) -> bool:
    """A linear model's constructor, or a pipeline that ends in one."""

    return any(
        isinstance(sub, ast.Call) and _call_target_name(sub) in _LINEAR_TERM_MODELS
        for sub in ast.walk(node)
    )


def _nominal_columns_in_linear_designs(
    tree: ast.Module,
    reads: Dict[str, Set[str]],
    bindings: Sequence[ast.Assign],
    nominal: Set[str],
) -> Set[str]:
    """Nominal columns a linear model reads as given, with no indicator coding."""

    models = {
        target.id
        for node in bindings
        if isinstance(node.value, ast.Call) and _builds_linear_model(node.value)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    read: Set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or any(
            keyword.arg == "formula" for keyword in node.keywords
        ):
            continue
        receiver = node.func.value if isinstance(node.func, ast.Attribute) else None
        fitted = _call_target_name(node) == "fit" and (
            (isinstance(receiver, ast.Name) and receiver.id in models)
            or (isinstance(receiver, ast.Call) and _builds_linear_model(receiver))
        )
        trained = _call_target_name(node) in _TRAINING_HELPERS and any(
            (isinstance(argument, ast.Name) and argument.id in models)
            or (isinstance(argument, ast.Call) and _builds_linear_model(argument))
            for argument in node.args
        )
        if (
            not (fitted or trained)
            and _call_target_name(node) not in _LINEAR_TERM_MODELS
        ):
            continue
        if fitted:
            # fit(X, y): the second position is the outcome.
            design = node.args[:1]
        elif trained:
            # cross_val_score(estimator, X, y): the design is the second.
            design = node.args[1:2]
        else:
            # A statsmodels model(endog, exog, ...): the first is the outcome.
            design = node.args[1:]
        # Strata, clusters, groups and weights are named, not modelled.
        named: Set[str] = set()
        terms: Set[str] = set()
        for argument in design:
            terms |= _columns_read_by(argument, reads)
        for keyword in node.keywords:
            if keyword.arg in _NOT_DESIGN_ARGUMENTS:
                continue
            target = named if keyword.arg in _NON_TERM_ARGUMENTS else terms
            target |= _columns_read_by(keyword.value, reads)
        read |= (terms - named) & nominal
    return read - _indicator_coded_columns(tree, reads, nominal)


def nominal_group_findings(
    *,
    context: ResearchContext,
    step: Optional[AnalysisStep],
    tree: ast.Module,
    alias_map: Dict[str, Set[str]],
    validator: str,
) -> List[ValidationFinding]:
    """A nominal grouping's codes name groups: no model or trend reads their order.

    A study's nominal exposure grouping is staged as integer level codes
    (``contracts.exposure_group_rules``).  The codes carry neither order nor
    spacing, so a linear model takes them only as categories against a
    reference group, and an ordered-stratified analysis does not read them.
    The check is static: a design built by a pipeline it cannot follow, or a
    whole table whose columns no line names, is not seen.
    """

    nominal = {
        str(variable.name)
        for variable in context.variables
        if str(variable.unit_normalization or "") == EXPOSURE_GROUP_TRANSFORM_ID
    }
    if not nominal:
        return []
    # A name bound from a call (a split, a copy) or a selection reads what
    # it reads, and a list grown after it was bound reads what it grew by.
    reads = {name: set(columns) for name, columns in alias_map.items()}
    bindings = [node for node in ast.walk(tree) if isinstance(node, ast.Assign)]
    bound, changes = _bindings_by_name(bindings, tree)
    for _ in range(2):
        for name, values in changes.items():
            for value in values:
                reads[name] = reads.get(name, set()) | _columns_read_by(value, reads)
        for node in bindings:
            if not isinstance(node.value, (ast.Call, ast.Subscript)):
                continue
            for target, read in _bound_parts(node, reads):
                for name in ast.walk(target):
                    if isinstance(name, ast.Name) and read:
                        reads[name.id] = reads.get(name.id, set()) | read
    changed = set(changes)
    step_id = step.step_id if step is not None else None
    findings: List[ValidationFinding] = []
    in_formulas: Set[str] = set()
    for expression in _formula_expressions(tree):
        in_formulas |= _SpelledFormula(bound, changed, reads, nominal).bare_terms(
            expression
        )
    linear = (
        in_formulas - _categorical_columns(tree, nominal)
    ) | _nominal_columns_in_linear_designs(tree, reads, bindings, nominal)
    if linear:
        findings.append(
            ValidationFinding(
                validator=validator,
                severity="error",
                message=(
                    f"Nominal exposure-group column(s) {sorted(linear)} enter a "
                    "model as one numeric term. Their level codes name groups and "
                    "carry no order or spacing: enter them as categories against "
                    "the study's reference group, as C(column, "
                    "Treatment(reference=code)) written out in the formula or as "
                    "indicator columns that leave the reference group out."
                ),
                detail={
                    "kind": "nominal_group_used_as_linear_term",
                    "columns": sorted(linear),
                    "step_id": step_id,
                },
            )
        )
    if step is not None and is_ordered_stratified_analysis_step(step):
        ordered = sorted(nominal & {str(name) for name in step.inputs})
        if ordered:
            findings.append(
                ValidationFinding(
                    validator=validator,
                    severity="error",
                    message=(
                        f"The ordered-stratified analysis reads nominal "
                        f"exposure-group column(s) {ordered} as ordered levels. "
                        "Their codes name groups in no order: compare the groups "
                        "as categories, or state an ordinal grouping."
                    ),
                    detail={
                        "kind": "nominal_group_read_as_ordered",
                        "columns": ordered,
                        "step_id": step_id,
                    },
                )
            )
    return findings


def _bound_parts(
    node: ast.Assign, reads: Dict[str, Set[str]]
) -> List[Tuple[ast.AST, Set[str]]]:
    """Each target a call or a selection binds, with the columns it reads.

    A split returns a training and a test part of each array it is given, in
    order, so each part reads its own array; any other binding's targets read
    all that its value reads.
    """

    call = node.value
    if (
        isinstance(call, ast.Call)
        and _call_target_name(call) == "train_test_split"
        and len(node.targets) == 1
        and isinstance(node.targets[0], (ast.Tuple, ast.List))
        and call.args
        and not any(isinstance(argument, ast.Starred) for argument in call.args)
        and len(node.targets[0].elts) == 2 * len(call.args)
    ):
        return [
            (part, _columns_read_by(call.args[index // 2], reads))
            for index, part in enumerate(node.targets[0].elts)
        ]
    read = _columns_read_by(call, reads)
    return [(target, read) for target in node.targets]


def _call_target_name(node: ast.Call) -> Optional[str]:
    """The called callable's short name: its last attribute, or its name."""

    func = node.func
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return None


__all__ = ["nominal_group_findings"]
