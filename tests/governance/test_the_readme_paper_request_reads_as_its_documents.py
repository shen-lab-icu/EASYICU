"""The README's "ask for a paper" examples are read by the host as the PDF and references.

The README tells a researcher how to get a paper. If its example wording
drifts from what the deliverable reader understands, the advertised path
stops at plan review (UX rc_70e).
"""
import re
from pathlib import Path

import pytest

from easyicu.research_agent.contracts.product_files import neutral_delivery_kinds

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = (
    ("README.md", r'for example "…, ([^"]+)"'),
    ("README_zh.md", r"例如「……，([^」]+)」"),
    ("docs/images/easyicu_workflow.svg", r'for example "…, ([^"]+)"'),
)


@pytest.mark.parametrize(("name", "pattern"), EXAMPLES)
def test_the_readme_paper_example_names_the_pdf_and_the_references(name, pattern):
    text = " ".join((ROOT / name).read_text(encoding="utf-8").split())
    text = text.replace("&quot;", '"')
    examples = re.findall(pattern, text)

    assert examples, f"{name} no longer shows a paper request example"
    for example in examples:
        assert neutral_delivery_kinds(example) == ("manuscript_pdf", "references"), example
