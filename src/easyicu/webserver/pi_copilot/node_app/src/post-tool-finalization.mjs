import { createAssistantMessageEventStream } from "@earendil-works/pi-ai";

const ZERO_USAGE = Object.freeze({
  input: 0,
  output: 0,
  cacheRead: 0,
  cacheWrite: 0,
  totalTokens: 0,
  cost: Object.freeze({ input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 }),
});

function latestStudyContextUpdate(context) {
  const messages = Array.isArray(context?.messages) ? context.messages : [];
  const result = messages.at(-1);
  const assistant = messages.at(-2);
  if (
    result?.role !== "toolResult"
    || result.toolName !== "easyicu_update_study_context"
    || result.isError === true
    || assistant?.role !== "assistant"
  ) return null;
  const call = Array.isArray(assistant.content)
    ? assistant.content.find((item) => (
      item?.type === "toolCall"
      && item.name === "easyicu_update_study_context"
      && item.id === result.toolCallId
    ))
    : null;
  const receipt = result.details && typeof result.details === "object"
    ? result.details
    : {};
  if (!call || receipt.status !== "ok" || receipt.code !== "study_context_updated") return null;
  return { call, receipt };
}

function confirmedDataSource(workflow) {
  const configuration = workflow?.study_setup_receipt?.configuration;
  const source = configuration?.data_source;
  return Boolean(source && typeof source === "object" && (source.database || source.label));
}

function latestDataSourceCatalog(context) {
  const messages = Array.isArray(context?.messages) ? context.messages : [];
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const result = messages[index];
    if (
      result?.role !== "toolResult"
      || result.toolName !== "easyicu_list_data_sources"
      || result.isError === true
    ) continue;
    const receipt = result.details && typeof result.details === "object"
      ? result.details
      : {};
    if (receipt.status === "ok" && receipt.code === "easyicu_data_sources_listed") {
      return receipt.details && typeof receipt.details === "object"
        ? receipt.details
        : {};
    }
  }
  return {};
}

function initialQuestionSaveNeedsDataSourceSelection(update) {
  const args = update?.call?.arguments;
  const workflow = update?.receipt?.details?.workflow;
  const missing = workflow?.missing_setup_fields;
  if (!args || typeof args !== "object" || !Array.isArray(missing)) return false;
  return Boolean(String(args.question || "").trim())
    && !confirmedDataSource(workflow)
    && missing.includes("data_source");
}

function studyUpdateIsReadyForPlanning(update) {
  const workflow = update?.receipt?.details?.workflow;
  // Unconfirmed design proposals may have been omitted by the update owner.
  // Those are execution requirements, not reasons to reopen initial setup.
  // Read the returned workflow, never infer readiness from the model's args.
  return workflow?.next_action_code === "provider_ready_to_generate_plan";
}

function finalizedMessage(model, text) {
  return {
    role: "assistant",
    content: [{ type: "text", text }],
    api: model.api,
    provider: model.provider,
    model: model.id,
    usage: ZERO_USAGE,
    stopReason: "stop",
    timestamp: Date.now(),
  };
}

function completedStream(message) {
  const stream = createAssistantMessageEventStream();
  stream.push({ type: "start", partial: message });
  stream.push({ type: "text_start", contentIndex: 0, partial: message });
  stream.push({ type: "text_delta", contentIndex: 0, delta: message.content[0].text, partial: message });
  stream.push({ type: "text_end", contentIndex: 0, content: message.content[0].text, partial: message });
  stream.push({ type: "done", reason: "stop", message });
  return stream;
}

function messageText(message) {
  if (typeof message?.content === "string") return message.content;
  if (!Array.isArray(message?.content)) return "";
  return message.content
    .filter((item) => item?.type === "text")
    .map((item) => String(item.text || ""))
    .join("");
}

function latestUserPrompt(context) {
  const messages = Array.isArray(context?.messages) ? context.messages : [];
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    if (messages[index]?.role === "user") return messageText(messages[index]);
  }
  return "";
}

function zeroDirectionEntryText(context, language) {
  if (!latestUserPrompt(context).includes("[EASYICU_ZERO_DIRECTION_ENTRY_V1]")) return "";
  return language === "zh"
    ? "你现在只需要选择一个最容易开始的入口，不必先写出完整研究问题。\n\n选择现有 ICU 数据时，EasyICU 仍会先确认数据源；本轮不会读取数据或生成研究方案。\n\n**下一步：**\n- 从临床困惑开始\n- 从已有文章或 PDF 开始\n- 从现有 ICU 数据开始"
    : "Choose the easiest available starting point; you do not need a complete research question yet.\n\nIf you start from existing ICU data, EasyICU will still confirm the source first; this turn will not read data or create a study plan.\n\n**Next step:**\n- Start from a clinical uncertainty\n- Start from an article or PDF\n- Start from existing ICU data";
}

function mandatoryIdeaLiteratureSearch(context) {
  const messages = Array.isArray(context?.messages) ? context.messages : [];
  const result = messages.at(-1);
  if (
    result?.role !== "toolResult"
    || result.toolName !== "easyicu_mine_ideas"
    || result.isError === true
  ) return null;
  const receipt = result.details && typeof result.details === "object"
    ? result.details
    : {};
  if (receipt.status !== "ok" || receipt.code !== "easyicu_idea_mined") return null;
  const ownerDetails = receipt.details && typeof receipt.details === "object"
    ? receipt.details
    : {};
  const mining = ownerDetails.idea_mining && typeof ownerDetails.idea_mining === "object"
    ? ownerDetails.idea_mining
    : {};
  const runId = boundedLabel(mining.run_id);
  const ideaId = boundedLabel(mining.selected_idea_id);
  const prompt = latestUserPrompt(context);
  const internalMarker = "\n\n[EASYICU_INTERNAL_RESPONSE_LANGUAGE_V1]\n";
  const topic = boundedLabel(
    prompt.includes(internalMarker)
      ? prompt.slice(0, prompt.indexOf(internalMarker))
      : prompt,
  ) || boundedLabel(mining?.idea?.idea_title);
  return runId && ideaId && topic
    ? { topic, run_id: runId, idea_id: ideaId }
    : null;
}

function toolCallStream(model, name, arguments_) {
  const toolCall = {
    type: "toolCall",
    id: `call_easyicu_host_${Date.now().toString(36)}`,
    name,
    arguments: arguments_,
  };
  const message = {
    role: "assistant",
    content: [toolCall],
    api: model.api,
    provider: model.provider,
    model: model.id,
    usage: ZERO_USAGE,
    stopReason: "toolUse",
    timestamp: Date.now(),
  };
  const stream = createAssistantMessageEventStream();
  stream.push({ type: "start", partial: message });
  stream.push({ type: "toolcall_start", contentIndex: 0, partial: message });
  stream.push({
    type: "toolcall_delta",
    contentIndex: 0,
    delta: JSON.stringify(arguments_),
    partial: message,
  });
  stream.push({ type: "toolcall_end", contentIndex: 0, toolCall, partial: message });
  stream.push({ type: "done", reason: "toolUse", message });
  return stream;
}

function boundedLabel(value) {
  return String(value || "").replace(/\s+/g, " ").trim().slice(0, 120);
}

function dataSourceSelectionText(language, catalog) {
  const rows = Array.isArray(catalog?.supported_databases)
    ? catalog.supported_databases.filter((row) => row && typeof row === "object")
    : [];
  const choices = rows
    .map((row) => boundedLabel(row.display_label || row.label))
    .filter(Boolean)
    .slice(0, 6)
    .map((label) => language === "zh" ? `- 使用 ${label}` : `- Use ${label}`);
  const missingReleaseLabels = rows
    .filter((row) => row.reference_release === null)
    .map((row) => boundedLabel(row.display_label || row.label))
    .filter(Boolean);
  if (!choices.length) {
    choices.push(...(language === "zh"
      ? ["- 查看并选择 EasyICU 支持的数据库", "- 选择并绑定其他本地 ICU 数据源"]
      : ["- View and choose a supported EasyICU database", "- Choose and bind another local ICU data source"]));
  }
  const releaseNote = missingReleaseLabels.length
    ? (language === "zh"
      ? `目录没有为 ${missingReleaseLabels.join("、")} 声明单一参考版本，因此按规范名称显示，EasyICU 不会猜测版本。\n\n`
      : `The catalog does not declare a single reference release for ${missingReleaseLabels.join(", ")}, so EasyICU shows the canonical name and does not invent a version.\n\n`)
    : "";
  return language === "zh"
    ? `研究问题已保存。当前项目尚未选择本次会话的数据源；在确认具体数据源前，EasyICU 不会继续定义研究设计或生成正式研究计划。\n\n${releaseNote}**下一步：**请先选择数据库：\n${choices.join("\n")}`
    : `The research question is saved. This project has not selected a data source for the conversation; EasyICU will not continue defining the study design or generate the formal research plan until a specific source is confirmed.\n\n${releaseNote}**Next step:** Choose the database first:\n${choices.join("\n")}`;
}

/**
 * Finalize the narrow initial-question save path from the typed EasyICU receipt.
 * The first provider call still interprets the user's question and invokes the
 * owner tool. A second provider call is unnecessary here because the browser
 * already owns source selection and the candidate-plan confirmation.
 */
export function hostPostToolFinalization(model, context, language) {
  const zeroDirection = zeroDirectionEntryText(context, language);
  if (zeroDirection) {
    return completedStream(finalizedMessage(model, zeroDirection));
  }
  const literatureSearch = mandatoryIdeaLiteratureSearch(context);
  if (literatureSearch) {
    return toolCallStream(model, "easyicu_search_literature", literatureSearch);
  }
  const update = latestStudyContextUpdate(context);
  if (initialQuestionSaveNeedsDataSourceSelection(update)) {
    return completedStream(finalizedMessage(
      model,
      dataSourceSelectionText(language, latestDataSourceCatalog(context)),
    ));
  }
  if (!studyUpdateIsReadyForPlanning(update)) return null;
  const text = language === "zh"
    ? "研究问题和数据源已就绪，可以生成候选研究计划，供你审阅。尚未开始数据提取或分析。"
    : "The research question and data source are ready for a candidate research plan for your review. Data extraction and analysis have not started.";
  return completedStream(finalizedMessage(model, text));
}
