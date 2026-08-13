/* Paginated Pi conversation replay loader.
   Owner: loading browser-safe transcript/lifecycle pages only. */
(function () {
  'use strict';

  function rows(value) { return Array.isArray(value) ? value : []; }
  function page(value) { return value && typeof value === 'object' ? value : {}; }

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
    const created = Number(job && job.created_at_epoch);
    const finished = Number(job && job.finished_at_epoch);
    return {
      expanded: reviewPending,
      durationKnown: Number.isFinite(created) && Number.isFinite(finished) && finished >= created,
      startedAt: Number.isFinite(created) ? created * 1000 : null,
      endedAt: Number.isFinite(finished) ? finished * 1000 : null,
      title: reviewPending
        ? translate('Analysis plan ready for review', '分析计划已就绪，等待审阅')
        : '',
      terminalLabel: reviewPending
        ? translate('Plan contract passed; analysis is paused for human review', '计划合同已通过；分析已暂停，等待人工审阅')
        : '',
    };
  }

  window.EU_GUIDED_PI_REPLAY = { hydrate, lifecycleTurns, childJobPresentation };
})();
