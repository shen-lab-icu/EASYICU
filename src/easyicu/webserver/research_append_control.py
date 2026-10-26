"""Append stop projection and the existing job cancellation/progress boundary."""
from .agent_review_recovery import _write


def check_cancel(job):
    from .research_pipeline_run_errors import ResearchPipelineRunError

    checker = getattr(job, "is_cancel_requested", None)
    if (callable(checker) and checker()) or getattr(job, "cancel_requested", False):
        raise ResearchPipelineRunError("research_pipeline_cancelled", "The addition was stopped at your request.")


def append_progress(job):
    def progress(event):
        if job is not None:
            from .agent_pipeline_runs import _pipeline_progress

            check_cancel(job)
            _pipeline_progress(job, event)
            check_cancel(job)
    return progress


def stop_append(child, *, code="result_append_source_changed"):
    from easyicu.research_agent.orchestration.human_review_checkpoint import write_checkpoint
    from .research_append_run import load_append_run
    from .research_append_projection import publish_append_projection

    cancelled = code == "research_pipeline_cancelled"
    source = code == "result_append_source_changed"
    detail = dict(
        failure_code=code, owner="data_preparation" if source else "analysis_settings",
        message_en="The data or results this addition builds on have changed, so it stopped. Check the source in data preparation."
        if source else "The addition was stopped at your request; earlier results are unchanged." if cancelled
        else "The addition did not finish. Check its analysis settings before preparing a new addition.",
        message_zh="这次结果所用的数据或结果文件已被改动，追加已停止。请到数据准备里核对数据来源，再准备新的分析。"
        if source else "已按你的要求停止追加分析，之前的结果不受影响。" if cancelled
        else "追加分析没有完成。请到分析设置检查后，重新准备追加分析。",
        next_action_en="Open data preparation" if source else "Open analysis settings",
        next_action_zh="进入数据准备" if source else "进入分析设置",
    )
    _write(child.run_dir / "append_stop.json", detail)
    write_checkpoint(child.run_dir / "human_review_checkpoint.json", child.checkpoint.transitioned("failed"))
    publish_append_projection(load_append_run(child.wrapper))
