/* Paginated Pi conversation replay loader.
   Owner: loading browser-safe transcript/lifecycle pages only. */
(function () {
  'use strict';

  function rows(value) { return Array.isArray(value) ? value : []; }
  function page(value) { return value && typeof value === 'object' ? value : {}; }

  function sessionHasHistory(session) {
    if (!session || typeof session !== 'object') return false;
    return Number(session.message_count || 0) > 0
      || Boolean(String(session.last_message_job_id || '').trim())
      || Boolean(String(session.active_message_job_id || '').trim());
  }

  function preferredSessionId(sessions, rememberedSessionId, requestedAgentMode, requestedLanguage) {
    const requestedMode = requestedAgentMode === 'workspace'
      ? 'workspace'
      : (requestedAgentMode === 'research' ? 'research' : '');
    const language = requestedLanguage === 'zh'
      ? 'zh'
      : (requestedLanguage === 'en' ? 'en' : '');
    const saved = rows(sessions).filter(session => {
      const sessionMode = session && session.agent_mode === 'workspace' ? 'workspace' : 'research';
      const sessionLanguage = session && session.language === 'zh' ? 'zh' : 'en';
      return (!requestedMode || sessionMode === requestedMode)
        && (!language || sessionLanguage === language);
    });
    const remembered = String(rememberedSessionId || '').trim();
    const rememberedRow = saved.find(row => String(row && row.session_id || '') === remembered);
    const historicalRow = saved.find(sessionHasHistory);
    if (rememberedRow && (sessionHasHistory(rememberedRow) || !historicalRow)) return remembered;
    if (historicalRow) return String(historicalRow.session_id || '');
    return rememberedRow
      ? String(rememberedRow.session_id || '')
      : String(saved[0] && saved[0].session_id || '');
  }

  async function hydrate(api, session, projectId) {
    if (!api || typeof api.loadPiCopilotSession !== 'function' || !session) return session;
    const sessionId = String(session.session_id || '').trim();
    const project = String(projectId || '').trim();
    if (!sessionId || !project) return session;

    let transcript = rows(session.transcript).slice();
    let turns = rows(session.conversation_replay && session.conversation_replay.turns).slice();
    let transcriptCursor = page(session.transcript_page).next_cursor;
    let replayCursor = page(session.conversation_replay && session.conversation_replay.turn_page).next_cursor;
    const seenTranscript = new Set();
    const seenReplay = new Set();

    for (let request = 0; request < 100 && (transcriptCursor || replayCursor); request += 1) {
      const transcriptKey = transcriptCursor == null ? '' : String(transcriptCursor);
      const replayKey = replayCursor == null ? '' : String(replayCursor);
      if ((transcriptKey && seenTranscript.has(transcriptKey)) || (replayKey && seenReplay.has(replayKey))) break;
      if (transcriptKey) seenTranscript.add(transcriptKey);
      if (replayKey) seenReplay.add(replayKey);
      const payload = await api.loadPiCopilotSession(sessionId, project, {
        transcriptCursor: transcriptKey || '0', transcriptLimit: 200,
        replayCursor: replayKey || '0', replayLimit: 100,
      });
      const older = payload && payload.session ? payload.session : {};
      if (transcriptKey) transcript = rows(older.transcript).concat(transcript);
      if (replayKey) {
        const olderReplay = older.conversation_replay || {};
        turns = rows(olderReplay.turns).concat(turns);
      }
      transcriptCursor = transcriptKey ? page(older.transcript_page).next_cursor : null;
      replayCursor = replayKey
        ? page(older.conversation_replay && older.conversation_replay.turn_page).next_cursor
        : null;
    }

    const replay = Object.assign({}, session.conversation_replay || {}, {
      turns,
      turn_page: {
        items: turns, start: 0, end: turns.length, total: turns.length,
        has_more: false, next_cursor: null,
      },
    });
    return Object.assign({}, session, {
      transcript,
      transcript_page: {
        items: transcript, start: 0, end: transcript.length, total: transcript.length,
        has_more: false, next_cursor: null,
      },
      conversation_replay: replay,
      last_turn_events: turns.length ? rows(turns[turns.length - 1].events) : rows(session.last_turn_events),
    });
  }

  function lifecycleTurns(session) {
    const replay = session && session.conversation_replay;
    const turns = rows(replay && replay.turns);
    if (turns.length) return turns;
    const events = rows(session && session.last_turn_events);
    return events.length ? [{
      job_id: session.last_message_job_id || 'latest-turn',
      status: session.last_turn_status || 'done',
      allowed_actions: rows(session.last_turn_allowed_actions),
      events,
    }] : [];
  }

  function childJobPresentation(job, tr) {
    const translate = typeof tr === 'function' ? tr : (en => en);
    const reviewPending = Boolean(job && job.human_review_pending);
    const gateBlocked = !reviewPending && String(job && job.gate_status || '') === 'blocked';
    const gateReason = String(job && job.gate_reason_code || '');
    const errorCode = String(job && job.error_code || '');
    // Only a genuine historical Planner efficiency-budget stop is presented
    // as resumable. Contract/compiler failures are failures, not a normal
    // pause, and must never produce a misleading "continue" affordance.
    const plannerCheckpointSaved = errorCode
      === 'research_pipeline_planner_efficiency_budget_exhausted';
    const planFoundationBlocked = gateBlocked && gateReason === 'data_foundation_blocked';
    const created = Number(job && job.created_at_epoch);
    const finished = Number(job && job.finished_at_epoch);
    return {
      // A pending review does not need its build log unfolded. The attention
      // signal is the review card below the activity -- which states the
      // decision and carries the buttons -- while the expanded body is 22
      // lines of "Plan draft 1/3 passed contract validation" that pushed that
      // card off screen. The summary keeps the title, step count and duration,
      // and a failed turn still opens because there the detail is the answer.
      expanded: false,
      durationKnown: Number.isFinite(created) && Number.isFinite(finished) && finished >= created,
      startedAt: Number.isFinite(created) ? created * 1000 : null,
      endedAt: Number.isFinite(finished) ? finished * 1000 : null,
      title: reviewPending
        ? translate('Analysis plan ready for review', '分析计划已就绪，等待审阅')
        : plannerCheckpointSaved
          ? translate('Planner saved a validated checkpoint', '规划器已保存验证检查点')
        : planFoundationBlocked
          ? translate('Research plan was not generated', '研究计划未生成')
          : gateBlocked
            ? translate('EasyICU task did not pass its scientific gate', 'EasyICU 科研任务未通过')
            : '',
      terminalLabel: reviewPending
        ? translate('Plan contract passed; analysis is paused for human review', '计划合同已通过；分析已暂停，等待人工审阅')
        : plannerCheckpointSaved
          ? translate('A validated checkpoint was saved; continue to finish the plan', '已保存验证检查点；可继续完成研究计划')
        : planFoundationBlocked
          ? translate('Research plan was not generated because data preparation did not pass', '研究计划未生成：数据准备未通过')
          : gateBlocked
            ? translate('The scientific gate blocked this task', '科学闸门已阻止本次任务')
            : '',
      blocked: gateBlocked || plannerCheckpointSaved,
    };
  }

  window.EU_GUIDED_PI_REPLAY = {
    hydrate,
    lifecycleTurns,
    childJobPresentation,
    preferredSessionId,
  };
})();
