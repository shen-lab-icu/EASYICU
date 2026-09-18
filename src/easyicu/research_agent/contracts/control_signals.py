"""Leaf contract for host-owned cancellation and orchestration control."""


class ProgressControlSignal(RuntimeError):
    """A control decision, never an advisory observer or schema failure."""
