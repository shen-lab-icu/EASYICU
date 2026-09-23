"""Executable reference workflow packages ("skill packages") over the research-agent kernels.

Not to be confused with ``research_agent.skills`` (the ClinicalSkill registry):
this package holds runnable, tested analysis workflows.

A skill is one method family packaged the way a reviewer can run it: a
``SKILL.md`` that fixes the standard workflow, a ``scripts/`` package with a
few typed entry functions that print verification tokens, ``references/`` for
method notes, and a synthetic example cohort so the whole package can be
smoke-tested in under a minute.

Skills compose existing owners -- the landmark eligibility mask, the adjusted
association kernel, the ordered-trend and spline primitives -- instead of
re-deriving them.  They do not select the exposure, outcome, cohort or estimand:
those arrive in a typed specification that the caller (a reviewed plan, or a
user running the script directly) supplies.  Every number a skill writes into
``report.md`` is copied from the tables it exported, and ``export_all`` refuses
to print its completion token until the export consistency gate has passed.

This package must stay free of webserver and orchestration imports; the host
pipeline may call a skill, a skill never calls the host.
"""

from __future__ import annotations

__all__: list[str] = []
