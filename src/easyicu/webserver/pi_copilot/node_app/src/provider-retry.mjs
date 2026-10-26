import { setTimeout as sleep } from "node:timers/promises";

import { modelErrorCode } from "./event-projection.mjs";

const RETRY_DELAYS_MS = [1500, 4000, 10000];

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

/** Retry only empty attempts; every open is host-guarded, every wait abortable. */
export async function retryTransientProviderStream(
  openAttempt,
  { signal, sleep: wait = sleep, onRetry } = {},
) {
  for (let attempt = 0; ; attempt += 1) {
    let iterator;
    const prefix = [];
    const canRetry = () => attempt < RETRY_DELAYS_MS.length && !signal?.aborted;
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
          if (canRetry() && isTransient(event.error) && !prefix.some(hasContent)) break;
          return replayPrefix(stream, iterator, prefix);
        }
      }
    } catch (error) {
      if (!canRetry() || !isTransient(error) || prefix.some(hasContent)) throw error;
    }
    // Discard this failed prefix before announcing a retry. Counts describe
    // retries (1..3), not total calls; no provider error detail crosses here.
    await iterator?.return?.();
    signal?.throwIfAborted();
    onRetry?.({ type: "auto_retry_start", attempt: attempt + 1, maxAttempts: RETRY_DELAYS_MS.length });
    await wait(RETRY_DELAYS_MS[attempt], undefined, { signal });
    signal?.throwIfAborted();
  }
}
