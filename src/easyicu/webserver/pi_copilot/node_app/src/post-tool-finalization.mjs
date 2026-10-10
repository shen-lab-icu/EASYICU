import { createAssistantMessageEventStream } from "@earendil-works/pi-ai";

const ZERO_USAGE = Object.freeze({
  input: 0,
  output: 0,
  cacheRead: 0,
  cacheWrite: 0,
  totalTokens: 0,
  cost: Object.freeze({ input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 }),
});
// main.mjs appends host sections to the researcher's text in this order:
// the language requirement, then the current-turn owner receipts.
export const OWNER_CONTEXT_MARKER = "\n\n[EASYICU_CURRENT_TURN_OWNER_CONTEXT_V1]\n";
const HOST_SECTION_PREFIX = "\n\n[EASYICU_";
const DEMO_WORD = "(?:demo|演示|示例)";

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

// The researcher's own words: the prompt up to the first host section.
function researcherText(context) {
  const prompt = latestUserPrompt(context);
  const at = prompt.indexOf(HOST_SECTION_PREFIX);
  return at >= 0 ? prompt.slice(0, at) : prompt;
}

// Receipts the host preloaded for this turn; the owner-context rule tells the
// model to treat them exactly as the corresponding tool results.
function ownerContextReceipts(context) {
  const prompt = latestUserPrompt(context);
  const at = prompt.indexOf(OWNER_CONTEXT_MARKER);
  if (at < 0) return [];
  const tail = prompt.slice(at + OWNER_CONTEXT_MARKER.length);
  const end = tail.indexOf(HOST_SECTION_PREFIX);
  try {
    const receipts = JSON.parse(end >= 0 ? tail.slice(0, end) : tail);
    return Array.isArray(receipts) ? receipts : [];
  } catch {
    return [];
  }
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
  const preloaded = ownerContextReceipts(context).find((receipt) => (
    receipt?.status === "ok" && receipt.code === "easyicu_data_sources_listed"
  ));
  return preloaded?.details && typeof preloaded.details === "object" ? preloaded.details : {};
}

// The one official demo the researcher's own text names: the demo title's
// product token ("eICU", "MIMIC-IV") next to a demo word. The browser's
// data-source card applies the same rule to offer that demo directly.
function namedOfficialDemo(text, catalog) {
  const value = String(text || "").normalize("NFKC").toLowerCase();
  const demos = Array.isArray(catalog?.official_demos) ? catalog.official_demos : [];
  if (!value || !demos.length) return null;
  const hits = demos.filter((demo) => {
    const head = String(demo?.label || "").trim().split(/\s+/)[0] || "";
    const name = head.toLowerCase().replace(/[^a-z0-9]+/g, "[\\s-]*");
    if (!name) return false;
    const pattern = new RegExp(`${name}[^。.!！?？;；\\n]{0,16}${DEMO_WORD}|${DEMO_WORD}[^。.!！?？;；\\n]{0,8}${name}`);
    return pattern.test(value);
  });
  return hits.length === 1 ? hits[0] : null;
}

function namedDemoText(language, demo) {
  const label = [boundedLabel(demo.label), demo.version ? `v${boundedLabel(demo.version)}` : ""].filter(Boolean).join(" ");
  return language === "zh"
    ? `研究问题已保存。你的问题指定了 ${label}（仅官方 Demo 数据）；在确认具体数据源前，EasyICU 不会继续定义研究设计或生成正式研究计划。确认数据源不等于批准分析。\n\n**下一步：**在下方数据源卡片点击「用于本次会话」，EasyICU 会注册并确认这份数据，然后拟定研究计划。`
    : `The research question is saved. Your question names ${label} (official demo data only); EasyICU will not continue defining the study design or generate the formal research plan until a specific source is confirmed. Confirming a source does not approve analysis.\n\n**Next step:** Click "Use it for this conversation" on the data-source card below; EasyICU registers and confirms this data, then proposes the research plan.`;
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

// The design the update owner withheld while saving the rest
// (details.unsaved_design): which one, why, and who supplies it.  A reply the
// host finalizes states it, so the researcher is told even when no second
// provider call is made.
function withheldDesignText(update, language) {
  const withheld = update?.receipt?.details?.unsaved_design;
  if (!withheld || typeof withheld !== "object") return "";
  const zh = language === "zh";
  if (withheld.code === "web_trajectory_design_required") {
    return zh
      ? "分析设计这次没有保存：轨迹聚类需要先有经审阅的轨迹设计（建模哪些指标、固定窗口和网格、可选的类别数、什么算稳定），EasyICU 不替研究者选定。候选研究计划会提出轨迹设计，你在审阅中批准后，EasyICU 会把它连同分析设计一起写入研究配置。"
      : "The analysis design was not saved: trajectory clustering needs a reviewed trajectory design first (which measures are modelled, the fixed window and grid, the admissible numbers of classes, and what counts as stable), and EasyICU does not choose these for the study. The candidate research plan proposes one; once you approve it in the review, EasyICU records it together with the analysis design.";
  }
  const code = boundedLabel(withheld.code);
  return zh
    ? `分析设计这次没有保存：EasyICU 的执行检查没有接受它（代码：${code}）。候选研究计划会提出设计，供你审阅。`
    : `The analysis design was not saved: EasyICU's execution check did not accept it (code: ${code}). The candidate research plan proposes a design for your review.`;
}

// The fields the update owner did not save this turn
// (details.unconfirmed_omissions), one line per reason.  A reply the host
// finalizes states them, so the researcher is told although the model never
// reads the owner's summary.
const OMISSION_LINES = {
  study_cohort_population_requires_plan: [
    "人群限定没有保存为研究设置（人群预设和为它写的名称）：按疾病或暴露限定的人群由候选研究计划提出，你在审阅中确认。",
    "The population restriction was not saved to the study (its preset and the name written for it): a population restricted by a condition is proposed in the candidate research plan for your review.",
  ],
  study_cohort_all_stays_confirmation_required: [
    "入住选择没有保存：还没有确定是纳入全部符合条件的 ICU 入住，还是每位患者只取一次入住。",
    "The stay selection was not saved: it is not yet chosen whether every eligible ICU stay counts or one stay per patient.",
  ],
  study_cohort_first_stay_confirmation_required: [
    "“只取首次 ICU 入住”没有保存：它会改变分析单位，需要你明确选择。",
    "The first-ICU-stay restriction was not saved: it changes the analysis unit and needs your explicit choice.",
  ],
  study_primary_outcome_confirmation_required: [
    "主要结局没有保存：问题里提到的结局只记作候选意向，候选研究计划会提出具体定义，供你审阅。",
    "The primary outcome was not saved: the question names it only as candidate intent, and the candidate research plan proposes its definition for your review.",
  ],
  study_primary_exposure_confirmation_required: [
    "主要暴露没有保存：问题里提到的暴露只记作候选意向，候选研究计划会提出具体定义，供你审阅。",
    "The primary exposure was not saved: the question names it only as candidate intent, and the candidate research plan proposes its definition for your review.",
  ],
  study_analysis_goal_confirmation_required: [
    "分析目标没有保存：问题里提到的目标只记作候选意向，候选研究计划会提出分析方案，供你审阅。",
    "The analysis goal was not saved: the question names it only as candidate intent, and the candidate research plan proposes the analysis for your review.",
  ],
};

function unsavedFieldsText(update, language) {
  const omissions = update?.receipt?.details?.unconfirmed_omissions;
  if (!Array.isArray(omissions) || !omissions.length) return "";
  const zh = language === "zh";
  const byCode = new Map();
  for (const item of omissions) {
    const code = String(item?.code || "");
    if (!byCode.has(code)) byCode.set(code, []);
    byCode.get(code).push(String(item?.field || ""));
  }
  const lines = [...byCode].map(([code, fields]) => {
    const known = OMISSION_LINES[code];
    if (!known) {
      const named = fields.map(boundedLabel).join(zh ? "、" : ", ");
      return zh
        ? `- ${named} 没有保存（代码：${boundedLabel(code)}）。`
        : `- ${named} was not saved (code: ${boundedLabel(code)}).`;
    }
    const modules = fields.includes("modules")
      ? (zh ? "本轮一起提议的特征模块也没有保存。" : " The feature modules proposed with it were not saved either.")
      : "";
    return `- ${known[zh ? 0 : 1]}${modules}`;
  });
  return `${zh ? "这次没有保存的设置：" : "Not saved this time:"}\n${lines.join("\n")}`;
}

// The withheld setup goes before the reply's next-step list, or at its end.
function withWithheldDesign(text, withheld) {
  if (!withheld) return text;
  const at = text.indexOf("\n\n**");
  return at >= 0
    ? `${text.slice(0, at)}\n\n${withheld}${text.slice(at)}`
    : `${text}\n\n${withheld}`;
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

// The one database the researcher's own words chose. The host's source
// catalog resolved it from the same text (selected_database) and recommends
// its most complete EasyICU export, so the reply offers that export instead
// of asking which database to use. A bare "MIMIC" stays ambiguous.
// Same database words as the host's source catalog (tools.py
// _database_named_in_message): the reply credits the researcher only with a
// database their own text names.
const DATABASE_WORDS = [
  ["miiv", /\bmimic[\s_-]*(?:iv|4)\b/],
  ["mimic", /\bmimic[\s_-]*(?:iii|3)\b/],
  ["eicu", /\beicu\b/],
  ["aumc", /\b(?:amsterdamumcdb|aumc)\b/],
  ["hirid", /\bhirid\b/],
  ["sic", /\b(?:sicdb|sic)\b/],
];
function databaseNamedIn(text) {
  const value = String(text || "").normalize("NFKC").toLowerCase();
  const hit = DATABASE_WORDS.find(([, pattern]) => pattern.test(value));
  return hit ? hit[0] : "";
}

function namedDatabaseText(language, catalog, text) {
  const selected = catalog?.selected_database;
  if (!selected || typeof selected !== "object" || catalog?.database_selection_deferred === true) return "";
  if (!selected.database || databaseNamedIn(text) !== String(selected.database)) return "";
  const label = [boundedLabel(selected.label), selected.reference_release ? `v${boundedLabel(selected.reference_release)}` : ""]
    .filter(Boolean).join(" ");
  if (!label) return "";
  const zh = language === "zh";
  const recommended = catalog?.recommended_source && typeof catalog.recommended_source === "object"
    ? catalog.recommended_source : null;
  if (recommended && recommended.availability === "available_in_easyicu") {
    const stays = Number(recommended.aggregate?.stays);
    const modules = Number(recommended.module_count);
    const facts = [
      Number.isFinite(stays) && stays > 0 ? (zh ? `${stays.toLocaleString("en-US")} 个 ICU 入住记录` : `${stays.toLocaleString("en-US")} ICU stays`) : "",
      Number.isFinite(modules) && modules > 0 ? (zh ? `${modules} 个数据模块` : `${modules} data modules`) : "",
    ].filter(Boolean).join(zh ? "，" : ", ");
    return zh
      ? `研究问题已保存。你的问题指定了 ${label}，EasyICU 中已有可直接使用的 ${label} 数据${facts ? `（${facts}）` : ""}。确认数据源不等于批准分析。\n\n**下一步：**\n- 使用 EasyICU 中已准备好的 ${label} 数据导出（推荐）`
      : `The research question is saved. Your question names ${label}, and EasyICU already has ${label} data ready to use${facts ? ` (${facts})` : ""}. Confirming a source does not approve analysis.\n\n**Next step:**\n- Use the prepared ${label} EasyICU data export (recommended)`;
  }
  // Registered exports EasyICU cannot choose between: the researcher picks
  // one by its numbered, path-free facts.
  const choices = Array.isArray(catalog?.registered_source_choices)
    ? catalog.registered_source_choices.filter((row) => row && typeof row === "object").slice(0, 6)
    : [];
  if (choices.length) {
    const lines = choices.map((row, index) => {
      const choice = Number.isInteger(row.choice) ? row.choice : index + 1;
      const stays = Number(row.stays);
      const modules = Number(row.module_count);
      const date = boundedLabel(row.generated_date);
      const facts = [
        zh ? `第 ${choice} 份` : `export ${choice}`,
        date ? (zh ? `生成于 ${date}` : `generated ${date}`) : "",
        Number.isFinite(stays) && stays > 0 ? (zh ? `${stays.toLocaleString("en-US")} 个 ICU 入住记录` : `${stays.toLocaleString("en-US")} ICU stays`) : "",
        Number.isFinite(modules) && modules > 0 ? (zh ? `${modules} 个数据模块` : `${modules} data modules`) : "",
      ].filter(Boolean).join(zh ? "，" : ", ");
      return zh
        ? `- 使用 EasyICU 中已准备好的 ${label} 数据导出（${facts}）`
        : `- Use the prepared ${label} EasyICU data export (${facts})`;
    });
    return zh
      ? `研究问题已保存。你的问题指定了 ${label}，EasyICU 中已登记 ${choices.length} 份 ${label} 数据导出，无法自动判断用哪一份。确认数据源不等于批准分析。\n\n**下一步：**选择其中一份：\n${lines.join("\n")}`
      : `The research question is saved. Your question names ${label}, and EasyICU has ${choices.length} registered ${label} data exports it cannot choose between. Confirming a source does not approve analysis.\n\n**Next step:** choose one of them:\n${lines.join("\n")}`;
  }
  return zh
    ? `研究问题已保存。你的问题指定了 ${label}，但 EasyICU 里还没有登记这份数据。\n\n**下一步：**在下方数据源卡片选择本机的 ${label} 数据目录；EasyICU 登记后会拟定研究计划。`
    : `The research question is saved. Your question names ${label}, but EasyICU has no registered copy of it yet.\n\n**Next step:** Choose your local ${label} folder in the data-source card below; EasyICU registers it and then proposes the research plan.`;
}

function dataSourceSelectionText(language, catalog, text = "") {
  const demo = namedOfficialDemo(text, catalog);
  if (demo) return namedDemoText(language, demo);
  const named = namedDatabaseText(language, catalog, text);
  if (named) return named;
  const rows = Array.isArray(catalog?.supported_databases)
    ? catalog.supported_databases.filter((row) => row && typeof row === "object")
    : [];
  const choices = rows
    .map((row) => boundedLabel(row.display_label || row.label))
    .filter(Boolean)
    .slice(0, 6)
    .map((label) => language === "zh" ? `- 使用 ${label}` : `- Use ${label}`);
  if (!choices.length) {
    choices.push(...(language === "zh"
      ? ["- 查看并选择 EasyICU 支持的数据库", "- 选择并绑定其他本地 ICU 数据源"]
      : ["- View and choose a supported EasyICU database", "- Choose and bind another local ICU data source"]));
  }
  return language === "zh"
    ? `研究问题已保存。请先选择这项研究使用的数据库，EasyICU 随后拟定研究计划。\n\n**下一步：**\n${choices.join("\n")}`
    : `The research question is saved. Choose the database for this study first; EasyICU then proposes the research plan.\n\n**Next step:**\n${choices.join("\n")}`;
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
  const withheld = [withheldDesignText(update, language), unsavedFieldsText(update, language)]
    .filter(Boolean)
    .join("\n\n");
  if (initialQuestionSaveNeedsDataSourceSelection(update)) {
    return completedStream(finalizedMessage(
      model,
      withWithheldDesign(
        dataSourceSelectionText(language, latestDataSourceCatalog(context), researcherText(context)),
        withheld,
      ),
    ));
  }
  if (!studyUpdateIsReadyForPlanning(update)) return null;
  const text = language === "zh"
    ? "研究问题和数据源已就绪，可以生成候选研究计划，供你审阅。尚未开始数据提取或分析。"
    : "The research question and data source are ready for a candidate research plan for your review. Data extraction and analysis have not started.";
  return completedStream(finalizedMessage(model, withWithheldDesign(text, withheld)));
}
