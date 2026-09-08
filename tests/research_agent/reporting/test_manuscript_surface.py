from types import SimpleNamespace

from easyicu.research_agent.reporting.manuscript_labels import source_bound_manuscript_labels
from easyicu.research_agent.reporting.manuscript_surface import (
    deduplicate_claim_paragraphs, repair_filtered_section_openers, repeated_reader_paragraphs,
)


def test_duplicate_claim_cleanup_is_exact_block_scoped_and_idempotent():
    source = "## Abstract\n\n**Results:**\n\n{claim:a}\n\n**Conclusions:**\n\n{claim:a}\n\n{claim:b}\n\n{claim:a}\n\n## Conclusion\n\n{claim:a}\n"
    result = deduplicate_claim_paragraphs(source)
    assert result.count("{claim:a}") == 3
    assert result.count("{claim:b}") == 1
    assert deduplicate_claim_paragraphs(result) == result
    assert deduplicate_claim_paragraphs("A claim {claim:a} supports this. Again {claim:a}.") == "A claim {claim:a} supports this. Again {claim:a}."


def test_filtered_opener_loses_only_dangling_connective():
    sentence = "The current findings therefore provide a bounded description [@source]."
    before = "## Discussion\n\nAn unsupported claim. " + sentence + "\n\nThe results therefore remain uncertain.\n"
    after = before.replace("An unsupported claim. ", "")
    repaired = repair_filtered_section_openers(after, before_filter=before)
    assert "findings provide" in repaired and "results therefore" in repaired
    assert repair_filtered_section_openers(before, before_filter=before) == before
    assert repair_filtered_section_openers(repaired, before_filter=before) == repaired


def test_repetition_is_scoped_to_abstract_block_not_repeated_across_sections():
    paragraph = "Recorded source data supported this descriptive estimate; independent clinical validation remained pending."
    assert repeated_reader_paragraphs("**Results:**\n\n" + paragraph + "\n\n**Conclusions:**\n\n" + paragraph) == ()
    assert repeated_reader_paragraphs(paragraph + "\n\n" + paragraph) == (paragraph,)


def test_source_metadata_labels_preserve_event_recording_not_clinical_absence():
    exposure = SimpleNamespace(name="event", description="Recorded syndrome definition",
        observation_semantics=SimpleNamespace(kind="positive_only_event"),
        clinical_definition=SimpleNamespace(definition="Syndrome-X"),
        observed_domain={"is_binary": True, "levels": [0, 1]})
    outcome = SimpleNamespace(name="outcome", description="in hospital mortality")
    context = SimpleNamespace(variables=[exposure, outcome])
    labels = {"event=0": "未记录综合征", "event=1": "记录综合征", "outcome": "院内死亡"}
    result = source_bound_manuscript_labels(context, labels)
    assert result == {"event=0": "No recorded Syndrome-X", "event=1": "Recorded Syndrome-X", "outcome": "in hospital mortality"}
    assert source_bound_manuscript_labels(context, labels, language="zh") == labels
    assert source_bound_manuscript_labels(None, labels) == labels
    exposure.observation_semantics.kind = "measurement"
    assert source_bound_manuscript_labels(context, labels)["event=0"] == labels["event=0"]
    exposure.observation_semantics.kind = "positive_only_event"
    exposure.observed_domain["levels"] = [1, 2]
    assert source_bound_manuscript_labels(context, labels)["event=0"] == labels["event=0"]


def test_recorded_status_repair_does_not_redefine_measurements_or_background_literature():
    from easyicu.research_agent.reporting.manuscript_labels import recorded_definition_section_errors
    context = SimpleNamespace(variables=[SimpleNamespace(observation_semantics=SimpleNamespace(kind='positive_only_event'))])
    text = '## Introduction\n\nIn this study, the exposure was represented as a diagnosis status observed in the first day.\n\n## Discussion\n\nThe findings are anchored to that clinical definition.\n'
    assert set(recorded_definition_section_errors(text, context)) == {'introduction', 'discussion'}
    safe = text.replace('diagnosis status', 'recorded diagnosis status').replace('that clinical definition', 'the operational definition')
    assert recorded_definition_section_errors(safe, context) == {}
    context.variables[0].observation_semantics.kind = 'measurement'
    assert recorded_definition_section_errors(text, context) == {}


def test_event_representative_supplies_recorded_status_not_diagnosis_label():
    variable = SimpleNamespace(name='signal_max', source_concept='signal', description='Syndrome diagnosis',
        observation_semantics=SimpleNamespace(kind='positive_only_event', representative_column='signal_max'),
        clinical_definition=SimpleNamespace(definition='Syndrome-X'), observed_domain={'is_binary':True,'levels':[0,1]})
    context = SimpleNamespace(variables=[variable])
    labels = source_bound_manuscript_labels(context, {'signal_max':'诊断'}, include_unlabeled=True)
    assert labels['signal_max'] == labels['signal'] == 'Recorded Syndrome-X status'
    variable.observation_semantics.representative_column = 'other'
    assert 'signal' not in source_bound_manuscript_labels(context, {}, include_unlabeled=True)


def test_source_label_expansion_deduplicates_only_its_exact_multiword_prefix():
    from easyicu.research_agent.reporting.manuscript_quality import repair_reader_internal_phrases
    text = 'In-hospital outcome proportions were reported {evidence:outcome}. In-hospital in hospital mortality remained descriptive.'
    result, changes = repair_reader_internal_phrases(text, reader_display_labels={'outcome':'in hospital mortality'})
    assert 'in-hospital in hospital' not in result.lower()
    assert result.count('in hospital mortality') == 2
    assert '{evidence:outcome}' in result
    assert any(r['code']=='MANUSCRIPT_REPEATED_LABEL_PREFIX_REMOVED' for r in changes)
    assert repair_reader_internal_phrases(result, reader_display_labels={'outcome':'in hospital mortality'})[0] == result
    unchanged = 'Hospital mortality and in hospital mortality used different denominators: 10 and 10.'
    assert repair_reader_internal_phrases(unchanged, reader_display_labels={'outcome':'in hospital mortality'})[0] == unchanged
