"""A generated script split into host-owned and agent-owned regions.

The host writes authority-bearing code directly into the sandbox source --
deliberately, so the pre-execution static gate can verify the exact code that
will run rather than trust an imported helper. Four standard executors do this
(``cohort_summary_executor``, ``deterministic_robustness``,
``deterministic_missingness``, ``table_one_executor``), all through
``render_standard_plausibility_receipt_code``.

The script that carries those regions is then handed to the model to repair,
with no declared boundary between what the host owns and what the model may
write. fresh17 step ``07_standard_robustness_sensitivity`` is what that costs.
The executed script was entirely host-generated; it could not satisfy the
step's declared outputs, so the host asked the model to repair *its own*
script, and the returned draft:

* replaced the sealed scope ``plausibility_expected_columns = ('age',)`` with
  ``None`` plus a runtime re-derivation, and
* deleted the pin ``declared_contracts_sha256 != '4d8bd1f3...'`` that binds the
  resolved contracts to the step authority, keeping only a self-satisfiable
  ``declared == computed``.

The mechanical preflight caught the first and blocked, correctly. Nothing
catches the second: ``source_contracts_sha256`` appears in the gates only as a
finding-detail field, never read back out of the code. A draft that kept the
scope tuple and dropped the pin would have executed with the authority binding
silently gone.

This type makes the boundary explicit. The host keeps its regions; the model
is given the body and only the body; the assembled script -- what executes and
what the static gate verifies -- is unchanged. It is pure data: it renders no
code, calls no gate and knows nothing about plausibility.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Optional

_JOIN = "\n\n"


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class HostScaffoldedScript:
    """Host prologue, agent body, host epilogue.

    ``assembled()`` is the only thing that executes, so a scaffold is never a
    second source of truth about the script -- it is the same bytes with a
    boundary drawn through them.
    """

    prologue: str = ""
    body: str = ""
    epilogue: str = ""

    def assembled(self) -> str:
        parts = [part for part in (self.prologue, self.body, self.epilogue) if part]
        if not parts:
            return ""
        return _JOIN.join(parts) + "\n"

    @property
    def prologue_sha256(self) -> str:
        return _digest(self.prologue)

    @property
    def epilogue_sha256(self) -> str:
        return _digest(self.epilogue)

    @property
    def has_host_regions(self) -> bool:
        return bool(self.prologue or self.epilogue)

    def with_body(self, body: str) -> "HostScaffoldedScript":
        """The only supported edit: replace the agent-owned region."""

        return HostScaffoldedScript(
            prologue=self.prologue,
            body=str(body or "").strip(),
            epilogue=self.epilogue,
        )

    def body_of(self, script: str) -> Optional[str]:
        """The agent region of ``script``, or ``None`` if it is not this scaffold.

        ``None`` is the answer that matters: it means the returned draft did
        not keep the host's regions byte-for-byte, so it cannot be treated as a
        body and must not be wrapped in them -- wrapping a rewritten prologue
        would run the host's audit and the model's rewrite of it side by side.
        """

        text = str(script or "")
        if not self.has_host_regions:
            return text.strip() or None
        head = self.prologue + _JOIN if self.prologue else ""
        tail = _JOIN + self.epilogue + "\n" if self.epilogue else "\n"
        if not text.startswith(head) or not text.endswith(tail):
            return None
        return text[len(head) : len(text) - len(tail)]

    def host_regions_intact(self, script: str) -> bool:
        """Whether ``script`` still carries this scaffold's host regions."""

        return self.body_of(script) is not None


__all__ = ["HostScaffoldedScript"]
