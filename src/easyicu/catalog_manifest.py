"""Generated reviewer summary for catalog and trust-contract counts."""

from __future__ import annotations

import json
from pathlib import Path

from .clinical_contracts import load_clinical_contracts
from .research_agent.planning.capability_registry import CAPABILITY_REGISTRY
from .resources import load_data_sources, load_dictionary


def catalog_manifest() -> dict[str, object]:
    data_root = Path(__file__).resolve().parent / "data"
    base = json.loads((data_root / "concept-dict.json").read_text(encoding="utf-8"))
    sofa2 = json.loads((data_root / "sofa2-dict.json").read_text(encoding="utf-8"))
    merged = load_dictionary(include_sofa2=True)
    capabilities = tuple(CAPABILITY_REGISTRY)
    public_databases = tuple(
        source.name
        for source in load_data_sources()
        if not source.name.endswith("_demo")
    )
    return {
        "base_dictionary_concepts": len(base),
        "sofa2_overlay_concepts": len(sofa2),
        "merged_dictionary_concepts": len(tuple(merged.keys())),
        "supported_databases": public_databases,
        "clinical_contracts": len(load_clinical_contracts()),
        "scientific_capabilities": len(capabilities),
        "reportable_capabilities": sum(
            item.scientific_validation == "reportable" for item in capabilities
        ),
    }


def render_catalog_summary_markdown() -> str:
    manifest = catalog_manifest()
    databases = ", ".join(f"`{name}`" for name in manifest["supported_databases"])
    return "\n".join(
        [
            "# EasyICU generated catalog summary",
            "",
            "_Generated from the shipped dictionaries, data-source registry, clinical contracts, and scientific capability registry. Do not edit counts by hand._",
            "",
            "| Surface | Current count / value |",
            "| --- | --- |",
            f"| Base concept dictionary | {manifest['base_dictionary_concepts']} |",
            f"| SOFA-2 overlay entries | {manifest['sofa2_overlay_concepts']} |",
            f"| Unique merged dictionary concepts | {manifest['merged_dictionary_concepts']} |",
            f"| Supported public ICU databases | {len(manifest['supported_databases'])}: {databases} |",
            f"| Clinical definition contracts | {manifest['clinical_contracts']} |",
            f"| Scientific capabilities | {manifest['scientific_capabilities']} |",
            f"| Capabilities with an explicit reportable validator owner | {manifest['reportable_capabilities']} |",
            "",
        ]
    )


if __name__ == "__main__":  # pragma: no cover - documentation generator
    print(render_catalog_summary_markdown(), end="")


__all__ = ["catalog_manifest", "render_catalog_summary_markdown"]
