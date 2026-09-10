"""Only registered mock identities may bypass external-provider opt-in."""

import pytest

from easyicu.ai_optin import AIOptInError, check_external_llm_opt_in


@pytest.mark.parametrize(
    "name", ["remote-offline-model", "NotMockLLMClient", "offline/api", None]
)
def test_external_names_cannot_impersonate_offline_by_substring(name):
    with pytest.raises(AIOptInError):
        check_external_llm_opt_in(name, ai_enabled=False)


@pytest.mark.parametrize(
    "name",
    ["MockLLMClient", "MockLLMClient (offline, deterministic)", "mock", "OFFLINE"],
)
def test_known_offline_identities_keep_working(name):
    check_external_llm_opt_in(name, ai_enabled=False)
