from pathlib import Path

from easyicu.catalog_manifest import catalog_manifest, render_catalog_summary_markdown


def test_catalog_summary_is_generated_and_counts_live_registries() -> None:
    root = Path(__file__).resolve().parents[2]
    manifest = catalog_manifest()
    committed = (root / "docs/catalog_summary.md").read_text(encoding="utf-8")

    assert committed == render_catalog_summary_markdown()
    assert manifest["merged_dictionary_concepts"] >= manifest["base_dictionary_concepts"]
    assert len(manifest["supported_databases"]) == 6
    assert manifest["reportable_capabilities"] == 8
