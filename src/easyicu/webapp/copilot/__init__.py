"""Research Copilot submodules (incremental split of llm_chat.py).

Phase 6 of `easyicu美化/copilot_接线施工计划.md`: peel cohesive, low-coupling
helpers out of the ~13k-line `llm_chat.py` into focused modules, one verifiable
step at a time. `llm_chat.py` re-imports the moved symbols so all existing
references keep working.
"""
