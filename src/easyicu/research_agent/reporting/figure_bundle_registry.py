"""One routing table from a parent step's declared family to its renderer.

Six deterministic figure renderers already shared one clean interface. What had
no owner was *choosing* between them: selection was split across four
mechanisms in two modules — a dict in `pipeline`, an if-chain beside it, the
same dict restated 1,900 lines further down, and a third table in
`publication_bundles` — plus name-token guesses inside the renderers
themselves. Ownership was signalled by returning ``None``, which is how figure
policy ended up living in an 8,000-line pipeline module.

The renderers are deep and stay exactly where they are. This module owns only
the routing: each renderer *declares* what it owns, in one place, and the
pipeline asks.

Every renderer is bound lazily by its owning module, so importing this registry
does not drag in matplotlib or require pipeline initialization. Optional
pipeline compatibility providers preserve existing substitution seams.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Sequence, Tuple

Renderer = Callable[..., Optional[str]]
RendererLoader = Callable[[], Renderer]

#: Sentinels for a parent whose figure-data contract cannot pick one renderer.
AMBIGUOUS_FIGURE_DATA_FAMILY = "__ambiguous_figure_data_family__"
INCOMPATIBLE_FIGURE_DATA_FAMILY = "__incompatible_figure_data_family__"

__all__ = [
    "AMBIGUOUS_FIGURE_DATA_FAMILY",
    "INCOMPATIBLE_FIGURE_DATA_FAMILY",
    "FIGURE_BUNDLES",
    "FigureBundleRegistry",
    "FigureBundleRoute",
]


@dataclass
class FigureBundleRoute:
    """What one renderer owns, declared rather than inferred."""

    key: str
    analysis_families: Tuple[str, ...] = ()
    methods: Tuple[str, ...] = ()
    figure_data_families: Tuple[str, ...] = ()
    loader: Optional[RendererLoader] = None
    provider: Optional[RendererLoader] = field(default=None, repr=False)
    renderer: Optional[Renderer] = field(default=None, repr=False)

    def resolve(self) -> Optional[Renderer]:
        # A provider is re-read on every call. The renderers reachable as
        # `pipeline` module attributes were previously looked up there at call
        # time, so substituting one — which the figure rescue tests do —
        # changed what ran. Caching them here would silently take that away.
        if self.provider is not None:
            return self.provider()
        if self.renderer is None and self.loader is not None:
            self.renderer = self.loader()
        return self.renderer


class FigureBundleRegistry:
    """The single place a declared family is turned into a renderer."""

    def __init__(self) -> None:
        self._routes: Dict[str, FigureBundleRoute] = {}
        self._fallbacks: Dict[str, Tuple[str, ...]] = {}

    # -- declaration ----------------------------------------------------

    def declare(self, route: FigureBundleRoute) -> None:
        if route.key in self._routes:
            raise ValueError(f"figure bundle route already declared: {route.key}")
        for label, values, lookup in (
            ("analysis family", route.analysis_families, self._owner_of_family),
            ("method", route.methods, self._owner_of_method),
            (
                "figure-data family",
                route.figure_data_families,
                self._owner_of_figure_data_family,
            ),
        ):
            for value in values:
                owner = lookup(value)
                if owner is not None:
                    # Without this, two routes claiming the same key resolved
                    # by dict order — which is not ownership, it is luck.
                    raise ValueError(
                        f"{label} {value!r} is already owned by {owner!r}"
                    )
        self._routes[route.key] = route

    def bind(self, key: str, renderer: Renderer) -> None:
        """Attach a renderer this module cannot import for itself."""
        self._route(key).renderer = renderer

    def bind_provider(self, key: str, provider: RendererLoader) -> None:
        """Attach a renderer that must be re-read on every resolution."""
        self._route(key).provider = provider

    def _route(self, key: str) -> FigureBundleRoute:
        route = self._routes.get(key)
        if route is None:
            raise KeyError(f"no figure bundle route declared for {key!r}")
        return route

    def declare_family_fallback(self, family: str, keys: Sequence[str]) -> None:
        """Sibling renderings of one closed family, tried in order.

        Cohort sensitivity, overlap, and attrition/flow are three displays of
        the same cohort-definition family. A renderer that rejects a parent of
        the wrong shape must hand off to its sibling rather than fall through
        to the coder — but only within the family the parent declared.
        """
        unknown = [key for key in keys if key not in self._routes]
        if unknown:
            raise KeyError(f"undeclared figure bundle routes: {unknown}")
        self._fallbacks[family.strip().lower()] = tuple(keys)

    # -- resolution -----------------------------------------------------

    def _owner_of_family(self, family: str) -> Optional[str]:
        for key, route in self._routes.items():
            if family in route.analysis_families:
                return key
        return None

    def _owner_of_method(self, method: str) -> Optional[str]:
        for key, route in self._routes.items():
            if method in route.methods:
                return key
        return None

    def _owner_of_figure_data_family(self, family: str) -> Optional[str]:
        for key, route in self._routes.items():
            if family in route.figure_data_families:
                return key
        return None

    def _resolve(self, key: Optional[str]) -> Optional[Renderer]:
        route = self._routes.get(key or "")
        return route.resolve() if route is not None else None

    def renderer_for_analysis_family(self, family: Optional[str]) -> Optional[Renderer]:
        """Map a parent's recorded ``analysis_family`` to its renderer."""
        return self._resolve(self._owner_of_family(str(family or "").strip().lower()))

    def renderer_for_method(self, method: Optional[str]) -> Optional[Renderer]:
        """Map an exact controlled parent ``method`` to its renderer."""
        wanted = str(method or "").strip().lower()
        return self._resolve(self._owner_of_method(wanted) if wanted else None)

    def renderer_for_figure_data_family(
        self, family: Optional[str]
    ) -> Optional[Renderer]:
        """Map an explicit step-level figure-data contract to its renderer."""
        wanted = str(family or "").strip().lower()
        return self._resolve(
            self._owner_of_figure_data_family(wanted) if wanted else None
        )

    def fallback_renderers_for_family(
        self, family: Optional[str]
    ) -> Tuple[Renderer, ...]:
        """The declared in-family renderer chain, or just the family's own."""
        wanted = str(family or "").strip().lower()
        keys = self._fallbacks.get(wanted)
        if keys is not None:
            resolved = tuple(
                renderer
                for renderer in (self._resolve(key) for key in keys)
                if renderer is not None
            )
            if resolved:
                return resolved
        renderer = self.renderer_for_analysis_family(wanted)
        return (renderer,) if renderer is not None else ()


def _load(module: str, name: str) -> RendererLoader:
    def _loader() -> Renderer:
        from importlib import import_module

        return getattr(import_module(module, __package__), name)

    return _loader


FIGURE_BUNDLES = FigureBundleRegistry()

# Every route has an owner loader; pipeline compatibility providers are optional.
for _route in (
    FigureBundleRoute(
        key="association",
        analysis_families=("association", "dose_response"),
        loader=_load("..figures.association_prior_outputs", "_render_association_publication_bundle_from_prior_outputs"),
    ),
    FigureBundleRoute(
        key="sensitivity",
        analysis_families=("cohort_definition_sensitivity", "sensitivity_analysis"),
        methods=("cohort_definition_sensitivity",),
        loader=_load("..figures.sensitivity_prior_outputs", "_render_sensitivity_publication_bundle_from_prior_outputs"),
    ),
    FigureBundleRoute(
        key="prediction",
        analysis_families=("prediction", "prediction_model"),
        loader=_load(
            ".publication_bundles",
            "_render_prediction_publication_bundle_from_prior_outputs",
        ),
    ),
    FigureBundleRoute(
        key="survival",
        analysis_families=("survival", "survival_analysis"),
        loader=_load("..figures.survival", "render_survival_bundle_from_prior_outputs"),
    ),
    FigureBundleRoute(
        key="cohort",
        analysis_families=("cohort_definition",),
        loader=_load(
            ".publication_bundles",
            "_render_cohort_overlap_publication_bundle_from_prior_outputs",
        ),
    ),
    FigureBundleRoute(
        key="cohort_flow",
        loader=_load(
            ".publication_bundles",
            "_render_cohort_flow_publication_bundle_from_prior_outputs",
        ),
    ),
    FigureBundleRoute(
        key="missingness",
        analysis_families=("missingness", "measurement", "data_quality"),
        methods=("missingness", "missingness_audit", "missingness_measurement_audit"),
        loader=_load(
            "..figures.missingness_publication",
            "render_missingness_publication_bundle_from_prior_outputs",
        ),
    ),
    FigureBundleRoute(
        key="absolute_risk",
        analysis_families=("absolute_risk_context",),
        loader=_load(
            ".publication_bundles",
            "_render_absolute_risk_publication_bundle_from_prior_outputs",
        ),
    ),
    FigureBundleRoute(
        key="phenotype",
        analysis_families=("phenotyping", "clustering"),
        loader=_load(
            ".publication_bundles",
            "_render_phenotype_publication_bundle_from_prior_outputs",
        ),
    ),
    FigureBundleRoute(
        key="descriptive",
        analysis_families=("descriptive", "table_one", "baseline"),
        loader=_load(
            ".publication_bundles",
            "_render_descriptive_publication_bundle_from_prior_outputs",
        ),
    ),
    FigureBundleRoute(
        key="ordered_distribution",
        methods=("ordinal_exposure_derivation_and_quality_control",),
        figure_data_families=("ordered_category_distribution",),
        loader=_load(
            "..figures.ordered_distribution",
            "render_ordered_distribution_bundle_from_prior_outputs",
        ),
    ),
    FigureBundleRoute(
        key="distribution_availability",
        methods=("exposure_distribution_and_missingness_audit",),
        loader=_load(
            "..figures.distribution_availability",
            "render_distribution_availability_bundle_from_prior_outputs",
        ),
    ),
):
    FIGURE_BUNDLES.declare(_route)
del _route

# `cohort_definition_sensitivity` may hand off across its siblings; a bare
# `cohort_definition` renders the exact closed product its parent declared and
# never probes an overlap renderer whose schema might coincide.
FIGURE_BUNDLES.declare_family_fallback(
    "cohort_definition_sensitivity", ("sensitivity", "cohort", "cohort_flow")
)
FIGURE_BUNDLES.declare_family_fallback("cohort_definition", ("cohort_flow",))
