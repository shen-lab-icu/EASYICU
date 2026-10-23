import { setTimeout as sleep } from "node:timers/promises";

import { modelErrorCode } from "./event-projection.mjs";

const RETRY_DELAY_MS = 1500;

function hasContent(event) {
  // Opening a content block commits this attempt even before its first delta.
  if (/^(text|thinking|toolcall)_(start|delta|end)$/.test(event?.type || "")) return true;
  return [event?.partial, event?.message, event?.error].some((message) => (
    Array.isArray(message?.content) && message.content.some((part) => (
      (part.type === "text" && Boolean(part.text))
      || (part.type === "thinking" && Boolean(part.thinking))
      || part.type === "toolCall"
    ))
  ));
}

function isTransient(error) {
  return modelErrorCode({
    stopReason: error?.stopReason ?? "error",
    errorMessage: error?.errorMessage || error?.message || String(error),
  }) === "pi_model_provider_unavailable";
}

function replayPrefix(stream, iterator, prefix) {
  return {
    async *[Symbol.asyncIterator]() {
      try {
        yield* prefix;
        while (true) {
          const next = await iterator.next();
          if (next.done) return;
          yield next.value;
        }
      } finally {
        await iterator.return?.();
      }
    },
    result: () => stream.result(),
  };
}

/** Inspect only the first attempt's empty prefix; every open is host-guarded. */
export async function retryTransientProviderStream(
  openAttempt,
  { signal, sleep: wait = sleep } = {},
) {
  let iterator;
  const prefix = [];
  try {
    const stream = await openAttempt();
    iterator = stream[Symbol.asyncIterator]();
    while (true) {
      const next = await iterator.next();
      if (next.done) return replayPrefix(stream, iterator, prefix);
      const event = next.value;
      prefix.push(event);
      if (hasContent(event) || event.type === "done") {
        return replayPrefix(stream, iterator, prefix);
      }
      if (event.type === "error") {
        if (!signal?.aborted && isTransient(event.error) && !prefix.some(hasContent)) break;
        return replayPrefix(stream, iterator, prefix);
      }
    }
  } catch (error) {
    if (signal?.aborted || !isTransient(error) || prefix.some(hasContent)) throw error;
  }
  // The failed prefix stays private. The second stream passes through unchanged,
  // including its terminal error or a refusal from the next authorization.
  await iterator?.return?.();
  await wait(RETRY_DELAY_MS, undefined, { signal });
  return await openAttempt();
}
