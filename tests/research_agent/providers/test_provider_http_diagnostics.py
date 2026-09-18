"""Response-free transport diagnostics, independent of SDK exception names."""

from types import SimpleNamespace

import pytest

from easyicu.research_agent.providers import clients
from easyicu.research_agent.providers.structured_retry import safe_provider_error_category


@pytest.mark.parametrize("status", [400, 401, 403, 408, 429, 500, 502, 503, 504, 599])
@pytest.mark.parametrize("on_response", [False, True])
def test_http_diagnostic_reads_only_typed_error_status(status, on_response):
    error = RuntimeError("secret response body and status code 418")
    target = error
    if on_response:
        error.response = SimpleNamespace()
        target = error.response
    target.status_code = status
    assert clients.safe_provider_http_status_code(error) == status
    assert safe_provider_error_category(error) == "provider_http"


@pytest.mark.parametrize("status", [None, True, False, "503", 503.0, 99, 200, 399, 600])
def test_http_diagnostic_never_promotes_text_or_invalid_status(status):
    error = RuntimeError("HTTP status 503; token=secret-not-a-diagnostic")
    error.status_code = status
    assert clients.safe_provider_http_status_code(error) is None
    assert safe_provider_error_category(error) == "error"


def test_http_diagnostic_tolerates_broken_response_property():
    class BrokenResponse(RuntimeError):
        @property
        def response(self):
            raise ValueError("secret")

    assert clients.safe_provider_http_status_code(BrokenResponse()) is None


def test_named_transport_categories_keep_their_existing_meaning():
    for name, expected in [("RateLimitError", "rate_limit"), ("APITimeoutError", "timeout")]:
        error = type(name, (RuntimeError,), {})("secret")
        error.status_code = 429
        assert safe_provider_error_category(error) == expected
