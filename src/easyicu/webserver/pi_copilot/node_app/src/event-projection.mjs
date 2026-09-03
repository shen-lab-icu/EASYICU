/* Safe, dependency-neutral projection from Pi SDK events/session messages to
   EasyICU's browser contract. Raw reasoning, tool arguments, and partial tool
   output intentionally never cross this boundary. */

const MAX_TEXT_CHARS = 12000;
const WORKSPACE_FILE_TOOLS = new Set([
  "easyicu_read_project_file",
  "easyicu_write_project_file",
  "easyicu_edit_project_file",
  "easyicu_check_project_file",
  "easyicu_preview_project_file",
]);
const START_TIMESTAMP_EVENTS = new Set([
  "agent_start", "turn_start", "message_start", "tool_execution_start",
  "compaction_start", "auto_retry_start",
]);

function boundedText(value, limit = MAX_TEXT_CHARS) {
  return String(value ?? "").slice(0, limit);
}

function modelErrorCode(message) {
  if (!message || message.stopReason !== "error") return "";
  const detail = String(message.errorMessage || "").toLowerCase();
  if (detail.includes("pi_shell_token_budget_exhausted")) {
    return "pi_shell_token_budget_exhausted";
  }
  if (detail.includes("pi_shell_session_provider_call_budget_exhausted")) {
    return "pi_shell_session_provider_call_budget_exhausted";
  }
  if (/context|token.*limit|maximum.*length/.test(detail)) return "pi_model_context_limit";
  if (/rate.?limit|too many requests|quota/.test(detail)) return "pi_model_rate_limited";
  if (/timeout|timed out|connection reset|server_error|internal_server_error|\b5\d\d\b/.test(detail)) {
    return "pi_model_provider_unavailable";
  }
  return "pi_model_provider_error";
}

export function pairTranscriptMessages(messages, contextMessages) {
  const source = Array.isArray(messages) ? messages : [];
  const context = Array.isArray(contextMessages) ? contextMessages : [];
  const aligned = source.length === context.length && source.every((message, index) => (
    String(message?.role || "") === String(context[index]?.message?.role || "")
  ));
  if (!aligned) return source.map((message) => ({ message, entryId: "" }));
  return source.map((message, index) => ({
    message,
    // Pi session entry ids are opaque SDK values, not EasyICU stable ids. In
    // particular they may begin with a digit, so only bound the value here.
    entryId: boundedText(context[index]?.entryId, 160).trim(),
  }));
}

function eventTimestamp(event, { useMessageTimestamp = false } = {}) {
  const value = Number(
    event?.timestamp ?? (useMessageTimestamp ? event?.message?.timestamp : undefined),
  );
  return Number.isFinite(value) && value > 0
    ? new Date(value).toISOString()
    : new Date().toISOString();
}

function safeRelativeFile(value) {
  const file = boundedText(value, 240).trim().replaceAll("\\", "/");
  if (!file || file.startsWith("/") || file.includes("\0")) return "";
  const parts = file.split("/");
  if (parts.some((part) => !part || part === "." || part === "..")) return "";
  return file;
}

function safeStableId(value, limit = 160) {
  const text = boundedText(value, limit).trim();
  return /^[A-Za-z][A-Za-z0-9_.-]{0,159}$/.test(text) ? text : "";
}

function safeJobId(value) {
  const text = boundedText(value, 160).trim();
  return /^[A-Za-z0-9][A-Za-z0-9_.-]{0,159}$/.test(text) ? text : "";
}

function safeArtifactName(value) {
  const name = boundedText(value, 160).trim();
  if (!name || name.includes("/") || name.includes("\\") || !name.endsWith(".json")) return "";
  return /^[A-Za-z0-9_.-]+$/.test(name) ? name : "";
}

function safeResearchDocumentName(value) {
  const name = boundedText(value, 80).trim();
  return /^manuscript_scaffold\.(pdf|tex|bib)$/.test(name) ? name : "";
}

function safeSystemValidationDocumentName(value) {
  const name = boundedText(value, 80).trim();
  return /^system_validation_report\.(html|pdf)$/.test(name) ? name : "";
}

function safeSha256(value) {
  const text = boundedText(value, 64).trim().toLowerCase();
  return /^[a-f0-9]{64}$/.test(text) ? text : "";
}

function safeLiteratureUrl(value) {
  const text = boundedText(value, 500).trim();
  return /^https:\/\/pubmed\.ncbi\.nlm\.nih\.gov\/[0-9]{1,12}\/$/.test(text)
    || /^https:\/\/doi\.org\/10\.[0-9]{4,9}\/[A-Za-z0-9._;()/:+-]+$/.test(text)
    ? text : "";
}

function projectedResource(value) {
  if (!value || typeof value !== "object" || Array.isArray(value)) return undefined;
  if (value.kind === "literature_source") {
    const url = safeLiteratureUrl(value.url);
    const title = boundedText(value.title || value.label, 500).trim();
    if (!url || !title) return undefined;
    const authorityClass = value.authority_class === "literature_method"
      ? "literature_method" : "literature_retrieval_candidate";
    const retrievalFit = ["direct_retrieval_fit", "adjacent_retrieval_fit", "unclassified"]
      .includes(value.retrieval_fit) ? value.retrieval_fit : "";
    return {
      kind: "literature_source",
      url,
      label: boundedText(value.label || title, 160),
      title,
      year: boundedText(value.year, 16),
      venue: boundedText(value.venue, 240),
      relevance: boundedText(value.relevance, 1200),
      doi: boundedText(value.doi, 240),
      pmid: boundedText(value.pmid, 32),
      media_type: "text/html",
      authority_class: authorityClass,
      ...(retrievalFit ? {
        retrieval_fit: retrievalFit,
        retrieval_rationale: boundedText(value.retrieval_rationale, 600),
      } : {}),
    };
  }
  if (value.kind === "research_artifact") {
    const runId = safeStableId(value.run_id);
    const artifact = safeArtifactName(value.artifact);
    if (!runId || !artifact) return undefined;
    const sha256 = safeSha256(value.sha256);
    return {
      kind: "research_artifact",
      run_id: runId,
      artifact,
      label: boundedText(value.label || artifact, 160),
      media_type: "application/json",
      ...(sha256 ? { sha256 } : {}),
    };
  }
  if (value.kind === "idea_plan") {
    const runId = safeStableId(value.run_id);
    if (!runId || value.artifact !== "idea_plan.json") return undefined;
    return {
      kind: "idea_plan",
      run_id: runId,
      artifact: "idea_plan.json",
      label: boundedText(value.label || "Idea Mining plan preview", 160),
      media_type: "application/json",
      authority_class: "idea_mining_planning_only",
    };
  }
  if (value.kind === "research_document" || value.kind === "system_validation_document") {
    const runId = safeStableId(value.run_id);
    const systemValidation = value.kind === "system_validation_document";
    const artifact = systemValidation
      ? safeSystemValidationDocumentName(value.artifact)
      : safeResearchDocumentName(value.artifact);
    if (!runId || !artifact) return undefined;
    const sha256 = safeSha256(value.sha256);
    return {
      kind: systemValidation ? "system_validation_document" : "research_document",
      run_id: runId,
      artifact,
      label: boundedText(value.label || artifact, 160),
      media_type: boundedText(value.media_type || "application/octet-stream", 120),
      ...(sha256 ? { sha256 } : {}),
    };
  }
  if (value.kind === "data_package_review") {
    const studyContextId = safeStableId(value.study_context_id);
    const reviewSha256 = safeSha256(value.review_sha256);
    const studyRevision = Number(value.study_revision);
    if (
      !studyContextId
      || !reviewSha256
      || !Number.isInteger(studyRevision)
      || studyRevision < 0
    ) return undefined;
    return {
      kind: "data_package_review",
      study_context_id: studyContextId,
      study_revision: studyRevision,
      review_sha256: reviewSha256,
      label: boundedText(value.label || "Data package review", 160),
      media_type: "application/json",
    };
  }
  if (value.kind === "data_workbench_snapshot") {
    const view = boundedText(value.view, 80).trim();
    const snapshotSha256 = safeSha256(value.snapshot_sha256);
    if (![
      "cohort_summary", "feature_distribution", "icd_cohort_preview", "patient_timeline", "crossdb_comparison",
    ].includes(view) || !snapshotSha256) return undefined;
    return {
      kind: "data_workbench_snapshot",
      view,
      snapshot_sha256: snapshotSha256,
      label: boundedText(value.label || "Data Workbench", 160),
      media_type: "application/json",
    };
  }
  if (value.kind === "native_workspace") {
    const route = boundedText(value.route, 80).trim();
    const studyContextId = safeStableId(value.study_context_id);
    const studyRevision = Number(value.study_revision);
    const state = boundedText(value.state, 40).trim();
    const jobId = safeJobId(value.job_id);
    const sourceId = safeStableId(value.source_id);
    const expectedDatabase = boundedText(value.expected_database, 40).trim().toLowerCase();
    const entryMode = boundedText(value.entry_mode, 40).trim();
    const extractionScope = boundedText(value.extraction_scope, 40).trim();
    const allowedDatabases = new Set(["miiv", "mimic", "eicu", "aumc", "hirid", "sic"]);
    const allowedScopes = new Set(["study_required", "all_supported", "reuse_prepared_full"]);
    if (
      route !== "extraction"
      || !studyContextId
      || !Number.isInteger(studyRevision)
      || studyRevision < 0
      || !["setup", "running", "review"].includes(state)
      || (sourceId && !/^src_[a-f0-9]{12}$/.test(sourceId))
      || (expectedDatabase && !allowedDatabases.has(expectedDatabase))
      || (entryMode && entryMode !== "source_binding")
      || (extractionScope && !allowedScopes.has(extractionScope))
    ) return undefined;
    return {
      kind: "native_workspace",
      route,
      state,
      study_context_id: studyContextId,
      study_revision: studyRevision,
      label: boundedText(value.label || "Data Extraction", 160),
      media_type: "application/vnd.easyicu.native-workspace",
      ...(jobId ? { job_id: jobId } : {}),
      ...(sourceId ? { source_id: sourceId } : {}),
      ...(expectedDatabase ? { expected_database: expectedDatabase } : {}),
      ...(entryMode ? { entry_mode: entryMode } : {}),
      ...(extractionScope ? { extraction_scope: extractionScope } : {}),
    };
  }
  const file = safeRelativeFile(value.file);
  if (!file) return undefined;
  const kind = value.kind === "webpage" ? "webpage" : "file";
  const fallbackLabel = file.split("/").at(-1) || file;
  const sha256 = safeSha256(value.sha256);
  const checkedSha256 = kind === "webpage" ? safeSha256(value.checked_sha256) : "";
  return {
    kind,
    file,
    label: boundedText(value.label || fallbackLabel, 160),
    media_type: boundedText(value.media_type || "text/plain", 120),
    ...(sha256 ? { sha256 } : {}),
    ...(checkedSha256 ? { checked_sha256: checkedSha256 } : {}),
  };
}

function projectedResources(value) {
  if (!Array.isArray(value)) return [];
  return value.slice(0, 80).map(projectedResource).filter(Boolean);
}

function projectedIdeaMining(value) {
  if (!value || typeof value !== "object") return undefined;
  const idea = value.idea && typeof value.idea === "object" ? value.idea : {};
  const feasibility = value.feasibility && typeof value.feasibility === "object"
    ? value.feasibility : {};
  const mappedConcepts = Array.isArray(idea.mapped_concepts)
    ? idea.mapped_concepts.slice(0, 24).map(row => ({
      concept_id: boundedText(row?.concept_id, 120),
      label: boundedText(row?.label, 180),
      module: boundedText(row?.module, 120),
      role: boundedText(row?.role, 120),
      status: boundedText(row?.status, 120),
      ...(typeof row?.available === "boolean" ? { available: row.available } : {}),
    })).filter(row => row.concept_id || row.label)
    : [];
  const support = idea.design_support && typeof idea.design_support === "object"
    ? idea.design_support : {};
  const citations = Array.isArray(support.citations)
    ? support.citations.slice(0, 4).map(row => ({
      citation_id: boundedText(row?.citation_id, 120),
      title: boundedText(row?.title, 400),
      year: Number(row?.year) || undefined,
      url: safeLiteratureUrl(row?.url),
      supports: Array.isArray(row?.supports)
        ? row.supports.slice(0, 4).map(item => boundedText(item, 240)) : [],
    })).filter(row => row.citation_id && row.title)
    : [];
  const projected = {
    run_id: boundedText(value.run_id, 160),
    selected_idea_id: boundedText(value.selected_idea_id, 160),
    idea: {
      idea_title: boundedText(idea.idea_title, 500),
      population: boundedText(idea.population, 500),
      exposure_or_predictor: boundedText(idea.exposure_or_predictor, 500),
      outcome: boundedText(idea.outcome, 500),
      analysis_family: boundedText(idea.analysis_family, 120),
      rationale: boundedText(idea.rationale, 1600),
      go_no_go: boundedText(idea.go_no_go, 80),
      go_no_go_reason: boundedText(idea.go_no_go_reason, 1600),
      next_action: boundedText(idea.next_action, 1000),
      plan_status: boundedText(idea.plan_status, 120),
      mapped_concepts: mappedConcepts,
      unresolved_slots: Array.isArray(idea.unresolved_slots)
        ? idea.unresolved_slots.slice(0, 8).map(item => boundedText(item, 120)) : [],
      design_support: {
        card_id: boundedText(support.card_id, 120),
        version: boundedText(support.version, 40),
        file_sha256: safeSha256(support.file_sha256),
        trust_level: boundedText(support.trust_level, 80),
        review_status: boundedText(support.review_status, 80),
        summary: boundedText(support.summary, 1200),
        population: boundedText(support.population, 800),
        time_zero: boundedText(support.time_zero, 800),
        outcome_family: boundedText(support.outcome_family, 1200),
        requires_confirmation: Array.isArray(support.requires_confirmation)
          ? support.requires_confirmation.slice(0, 6).map(item => boundedText(item, 500)) : [],
        stop_conditions: Array.isArray(support.stop_conditions)
          ? support.stop_conditions.slice(0, 6).map(item => boundedText(item, 500)) : [],
        citations,
        authority: boundedText(support.authority, 120),
      },
    },
    feasibility: {
      status: boundedText(feasibility.status, 120),
      reason: boundedText(feasibility.reason, 1600),
      reportable: feasibility.reportable === true,
    },
  };
  return projected.idea.idea_title || projected.selected_idea_id ? projected : undefined;
}

function toolResource(toolName, args) {
  const name = boundedText(toolName, 160);
  if (!WORKSPACE_FILE_TOOLS.has(name)) return undefined;
  let params = args;
  if (typeof params === "string") {
    try { params = JSON.parse(params); } catch { return undefined; }
  }
  return projectedResource({
    kind: name === "easyicu_preview_project_file" ? "webpage" : "file",
    file: params?.file,
    media_type: name === "easyicu_preview_project_file" ? "text/html" : "text/plain",
    checked_sha256: name === "easyicu_preview_project_file"
      ? params?.checked_sha256 : undefined,
  });
}

function toolReceipt(result) {
  const details = result?.details && typeof result.details === "object"
    ? result.details
    : {};
  const ownerDetails = details.details && typeof details.details === "object"
    ? details.details
    : {};
  const content = Array.isArray(result?.content) ? result.content : [];
  const fallback = content.find((item) => item && item.type === "text")?.text || "";
  const resource = projectedResource(details.resource || ownerDetails.resource);
  const resources = projectedResources(details.resources || ownerDetails.resources);
  const ideaMining = projectedIdeaMining(
    ownerDetails.idea_mining || details.idea_mining,
  );
  const jobId = safeJobId(ownerDetails.job_id || details.job_id);
  const hostRebindAfterTurn = ownerDetails.host_rebind_after_turn === true;
  return {
    status: boundedText(details.status || "", 40),
    code: boundedText(details.code || "", 160),
    summary: boundedText(details.summary || fallback, 2000),
    owner: boundedText(details.owner || "", 240),
    ...(jobId ? { job_id: jobId } : {}),
    ...(hostRebindAfterTurn ? { host_rebind_after_turn: true } : {}),
    ...(resource ? { resource } : {}),
    ...(resources.length ? { resources } : {}),
    ...(ideaMining ? { idea_mining: ideaMining } : {}),
  };
}

export function normalizePiEvent(event) {
  if (!event || typeof event !== "object") return undefined;
  const startEvent = START_TIMESTAMP_EVENTS.has(event.type);
  const at = eventTimestamp(startEvent ? event : {}, {
    // Pi message timestamps identify when generation started. Reusing them on
    // message_end/turn_end made multi-second provider calls look like 0 ms.
    useMessageTimestamp: event.type === "message_start",
  });
  if (event.type === "agent_start") return { type: "run_start", at };
  if (event.type === "turn_start") {
    return { type: "turn_start", at, turn_index: Number(event.turnIndex || 0) };
  }
  if (event.type === "message_start" && event.message?.role === "assistant") {
    return { type: "assistant_start", at };
  }
  if (event.type === "message_update") {
    const update = event.assistantMessageEvent || {};
    if (update.type === "text_delta") {
      return { type: "text_delta", at, delta: boundedText(update.delta, 8000) };
    }
    return undefined;
  }
  if (event.type === "tool_execution_start") {
    const resource = toolResource(event.toolName, event.args);
    return {
      type: "tool_start",
      at,
      tool_call_id: boundedText(event.toolCallId, 160),
      tool_name: boundedText(event.toolName, 160),
      ...(resource ? { resource } : {}),
    };
  }
  if (event.type === "tool_execution_update") {
    return {
      type: "tool_progress",
      at,
      tool_call_id: boundedText(event.toolCallId, 160),
      tool_name: boundedText(event.toolName, 160),
    };
  }
  if (event.type === "tool_execution_end") {
    const receipt = toolReceipt(event.result);
    return {
      type: "tool_end",
      at,
      tool_call_id: boundedText(event.toolCallId, 160),
      tool_name: boundedText(event.toolName, 160),
      is_error: Boolean(
        event.isError || receipt.status === "blocked" || receipt.status === "failed",
      ),
      ...receipt,
    };
  }
  if (event.type === "message_end" && event.message?.role === "assistant") {
    const errorCode = modelErrorCode(event.message);
    return {
      type: "message_end",
      at,
      stop_reason: boundedText(event.message.stopReason || "complete", 80),
      ...(errorCode ? { error_code: errorCode } : {}),
    };
  }
  if (event.type === "turn_end") {
    return { type: "turn_end", at, turn_index: Number(event.turnIndex || 0) };
  }
  if (event.type === "agent_end") {
    return { type: "agent_cycle_end", at, will_retry: Boolean(event.willRetry) };
  }
  if (event.type === "agent_settled") return { type: "run_end", at };
  if (event.type === "compaction_start") {
    return { type: "compaction_start", at, reason: boundedText(event.reason, 80) };
  }
  if (event.type === "compaction_end") {
    return {
      type: "compaction_end",
      at,
      reason: boundedText(event.reason, 80),
      aborted: Boolean(event.aborted),
    };
  }
  if (event.type === "auto_retry_start") {
    return {
      type: "retry",
      at,
      attempt: Number(event.attempt || 0),
      max_attempts: Number(event.maxAttempts || 0),
    };
  }
  return undefined;
}

export function projectTranscriptMessage(message) {
  if (!message || typeof message !== "object") return undefined;
  const role = boundedText(message.role, 40);
  if (!new Set(["user", "assistant", "toolResult"]).has(role)) return undefined;
  const content = Array.isArray(message.content)
    ? message.content
    : [{ type: "text", text: message.content }];
  const timestamp = eventTimestamp({ message }, { useMessageTimestamp: true });

  if (role === "toolResult") {
    const receipt = toolReceipt(message);
    return {
      role: "tool",
      timestamp,
      content: [{
        type: "tool_result",
        tool_call_id: boundedText(message.toolCallId, 160),
        tool_name: boundedText(message.toolName, 160),
        is_error: Boolean(
          message.isError || receipt.status === "blocked" || receipt.status === "failed",
        ),
        ...receipt,
      }],
    };
  }

  const parts = [];
  for (const item of content.slice(0, 80)) {
    if (!item || typeof item !== "object") continue;
    if (item.type === "text") {
      parts.push({ type: "text", text: boundedText(item.text, MAX_TEXT_CHARS) });
    } else if (item.type === "toolCall") {
      const resource = toolResource(item.name, item.arguments);
      parts.push({
        type: "tool_call",
        tool_call_id: boundedText(item.id, 160),
        tool_name: boundedText(item.name, 160),
        ...(resource ? { resource } : {}),
      });
    }
  }
  return {
    role,
    timestamp,
    content: parts,
    stop_reason: role === "assistant"
      ? boundedText(message.stopReason || "", 80)
      : undefined,
    ...(role === "assistant" && modelErrorCode(message)
      ? { error_code: modelErrorCode(message) }
      : {}),
  };
}
