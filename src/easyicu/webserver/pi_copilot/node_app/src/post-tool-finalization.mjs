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
  if (receipt.status !== "ok" || receipt.code !== "study_context_updated") return null;
  return { call, receipt };
}

function confirmedDataSource(workflow) {
  const configuration = workflow?.study_setup_receipt?.configuration;
  const source = configuration?.data_source;
  return Boolean(source && typeof source === "object" && (source.database || source.label));
}

function initialQuestionSaveNeedsDataPreparation(update) {
  const args = update?.call?.arguments;
  const workflow = update?.receipt?.details?.workflow;
  const missing = workflow?.missing_setup_fields;
  if (!args || typeof args !== "object" || !Array.isArray(missing)) return false;
  if (!String(args.question || "").trim() || !confirmedDataSource(workflow)) return false;
  if (args.outcome || args.primary_exposure || args.time_window) return false;
  return workflow.next_action_code === "study_setup_incomplete"
    && ["outcome", "primary_exposure", "time_window"].some((field) => missing.includes(field));
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

/**
 * Finalize the narrow initial-question save path from the typed EasyICU receipt.
 * The first provider call still interprets the user's question and invokes the
 * owner tool. A second provider call is unnecessary here because the browser
 * already owns the confirmed-source transition and data-preparation review.
 */
export function hostPostToolFinalization(model, context, language) {
  const update = latestStudyContextUpdate(context);
  if (!initialQuestionSaveNeedsDataPreparation(update)) return null;
  const text = language === "zh"
    ? "研究问题和当前研究设置已保存；尚未开始数据提取或分析。\n\n**下一步：**请选择数据来源操作：\n- 使用当前已确认的数据来源"
    : "The research question and current study setup are saved; extraction and analysis have not started.\n\n**Next step:** Choose a data-source action:\n- Use the currently confirmed data source";
  return completedStream(finalizedMessage(model, text));
}
