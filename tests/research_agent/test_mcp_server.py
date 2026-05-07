"""MCP server protocol surface."""

from __future__ import annotations

import json


def test_mcp_initialize_and_tools_list(ra):
    from easyicu.research_agent.mcp_server import handle_jsonrpc

    init = handle_jsonrpc({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {},
    })
    assert init["result"]["capabilities"]["tools"] == {}
    assert init["result"]["serverInfo"]["name"] == "easyicu-research-agent"

    listed = handle_jsonrpc({
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/list",
        "params": {},
    })
    names = {tool["name"] for tool in listed["result"]["tools"]}
    assert {
        "research_agent.run",
        "research_agent.list_skills",
        "research_agent.read_manifest",
    } <= names


def test_mcp_tools_call_wraps_tool_result_as_content(ra):
    from easyicu.research_agent.mcp_server import handle_jsonrpc

    resp = handle_jsonrpc({
        "jsonrpc": "2.0",
        "id": "skills",
        "method": "tools/call",
        "params": {
            "name": "research_agent.list_skills",
            "arguments": {},
        },
    })
    assert resp["id"] == "skills"
    assert resp["result"]["isError"] is False
    text = resp["result"]["content"][0]["text"]
    data = json.loads(text)
    assert any(skill["key"] == "sofa_mortality" for skill in data["skills"])


def test_mcp_legacy_tool_shape_still_dispatches(ra):
    from easyicu.research_agent.mcp_server import handle_jsonrpc

    resp = handle_jsonrpc({
        "id": 7,
        "tool": "research_agent.list_skills",
        "arguments": {},
    })
    assert resp["id"] == 7
    assert "skills" in resp["result"]

