/* Host-admitted model selection on the existing runtime's connection. */
export function conversationModel({ runtime, selected, config }, requested) {
  if (requested === undefined) return selected;
  if (typeof requested !== 'string' || !requested.trim() || requested.length > 256) {
    throw Object.assign(new Error('Invalid conversation model'), { code: 'pi_session_model_unavailable' });
  }
  // The host checks discovery and data handling. API catalogues may contain
  // custom names absent from the SDK registry; preserve the verified wire route.
  if (!config.codexAuthFile) return { ...selected, id: requested, name: requested };
  const model = runtime.getModel(config.provider, requested);
  if (!model) throw Object.assign(new Error('Account model unavailable'), { code: 'pi_session_model_unavailable' });
  return model;
}

export async function switchConversationModel(record, activeRequests, runtime, model) {
  if (activeRequests.has(record.externalId) || record.session.isStreaming) {
    throw Object.assign(new Error('A conversation turn is running'), { code: 'pi_session_busy' });
  }
  await record.session.setModel(conversationModel(runtime, model));
}
