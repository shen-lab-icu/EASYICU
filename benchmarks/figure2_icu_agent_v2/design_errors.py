"""Shared typed failures for the Figure 2 v2.1 design boundary."""


class DesignContractError(ValueError):
    """Stable design-contract failure with a machine-readable reason code."""

    def __init__(self, reason_code: str, detail: str) -> None:
        self.reason_code = reason_code
        self.detail = detail
        super().__init__(f"{reason_code}: {detail}")


__all__ = ["DesignContractError"]
