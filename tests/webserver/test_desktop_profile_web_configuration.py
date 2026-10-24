"""Real profile options must bind once at the Web configuration boundary."""
from __future__ import annotations

import ast
import inspect

import pytest

from easyicu.research_agent.orchestration.config import PipelineConfig
from easyicu.research_agent.orchestration.profiles import get_submission_profile
from easyicu.webserver import agent_pipeline_runs
from easyicu.webserver.research_launch_runtime import _submission_profile_ref


@pytest.mark.parametrize("runner_kind", ["host", "docker"])
@pytest.mark.parametrize("live_pubmed", [False, True])
def test_web_profile_options_bind_once_and_keep_complete_plan_review(
    tmp_path, runner_kind, live_pubmed,
):
    profile = get_submission_profile(_submission_profile_ref(
        budget_mode="full_reviewed", live_pubmed=live_pubmed,
        runner_kind=runner_kind,
    ))
    tree = ast.parse(inspect.getsource(agent_pipeline_runs.make_research_pipeline_run_runner))
    options = profile.pipeline_options()
    # Even a profile default cannot waive the Web's complete-plan review.
    options["require_human_plan_review"] = False
    update, = [node for node in ast.walk(tree) if (
        isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "profile_options" and node.func.attr == "update"
        and node.args and isinstance(node.args[0], ast.Dict)
        and any(isinstance(key, ast.Constant) and key.value == "enable_memory"
                for key in node.args[0].keys)
    )]
    options.update(ast.literal_eval(update.args[0]))
    call, = [node for node in ast.walk(tree) if (
        isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        and node.func.id == "PipelineConfig"
        and any(keyword.arg is None and isinstance(keyword.value, ast.Name)
                and keyword.value.id == "profile_options" for keyword in node.keywords)
    )]
    # Exercise Python's actual keyword binding from the production call. Other
    # values depend on a live study, so replace those values, not its key shape.
    explicit = {keyword.arg: None for keyword in call.keywords if keyword.arg is not None}
    merged = dict(**explicit, **options)
    assert merged["require_human_plan_review"] is True
    config = PipelineConfig(workdir=tmp_path, runner_kind=runner_kind, **options)
    assert config.require_human_plan_review is True
    if runner_kind == "host":
        assert config.generate_manuscript is False
        assert config.execution_tier == "first_result"
