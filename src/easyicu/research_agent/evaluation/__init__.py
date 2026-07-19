"""Evaluation adapters over already-produced research artifacts.

The package is intentionally lazy: importing it must not construct LLM
clients, load rubrics, or pull the research-agent control plane into memory.
"""
