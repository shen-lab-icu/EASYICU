"""Genuine scientific reviewer for an invented pending first-ICU proposal.

All source, search and preparation proof data are synthetic; this is neither
an approval decision nor evidence about real studies, data or model output.
"""
import json
from tests.support.first_observed_synthetic_literature import KEY, synthetic_first_observed_literature
from tests.research_agent.planning.descriptive_interval_fixtures import descriptive_context
from tests.research_agent.planning.progressive_planner_fixtures import _payload
from easyicu.research_agent.canonical_json import canonical_sha256
from easyicu.research_agent.literature import LiteratureSearchProvenance
from easyicu.research_agent.planning.dependence_authority import host_independent_units_scope
from easyicu.research_agent.planning.population_compile import host_first_observed_recipe_scope, compile_population
from easyicu.research_agent.planning.population_spec import PopulationSpec
from easyicu.research_agent.planning.progressive_compiler import compile_progressive_plan
from easyicu.research_agent.planning.progressive_contract import ProgressivePlanSkeleton
from easyicu.research_agent.planning.scientific_review import build_plan_scientific_review
from easyicu.research_agent.research_context.export_selection import SourceIdentity


def positive_case(tmp_path, confidence=0.95):
    c = descriptive_context(repeats=False)
    question = (f'Describe exposure and outcome proportions with {int(confidence*100)}% confidence intervals '
                'among adult patients with one visible hospitalization at their first observed ICU admission.')
    source = {'database':'eicu_demo', 'registration_id':'synthetic', 'archive_sha256':'a'*64}
    constraints = json.loads(c.user_preferences.data_constraints)
    constraints['source_identity'] = source
    c = c.model_copy(update={'research_question':question,
        'cohort':c.cohort.model_copy(update={'database':'eicu_demo',
            'provenance':{'analysis_unit':'icu_stay','evidence_stage':'metadata_only_planning'}}),
        'user_preferences':c.user_preferences.model_copy(update={'data_constraints':json.dumps(constraints)})})
    raw = _payload()
    raw.update(analysis_type='descriptive_epidemiology', robustness_intents=[])
    raw['steps'] = [raw['steps'][0], raw['steps'][2]]
    raw['steps'][0]['raw_inputs'] = ['stay_id']
    raw['steps'][1].update(planned_analysis_role='primary', confidence_level=confidence,
        literature_bindings=[{'citation_key':KEY, 'design_elements':['population','time_zero','outcome','estimand','reporting'],
                              'application':'Use the synthetic source to frame the observed cohort and proportions.'}])
    raw['cohort']['population_spec'] = {'criteria':[{'id':'c1','kind':'first_observed_icu_single_hospital_adult',
        'role':'include','source':'question',
        'quote':'adult patients with one visible hospitalization at their first observed ICU admission'}]}
    skeleton = ProgressivePlanSkeleton.model_validate(raw)
    def proof(context, criterion):
        assert context is c
        return {'coordinate_sha256':'a'*64,'preparation_checkpoint_sha256':'b'*64,
            'source_identity_sha256':canonical_sha256(SourceIdentity.from_record(source).to_dict()),
            'criterion_sha256':canonical_sha256(criterion.model_dump(mode='json')),
            'export_manifest_sha256':'c'*64,'preparation_run_id':'synthetic_run'}
    literature = synthetic_first_observed_literature(question, tmp_path)
    literature.search_provenance = LiteratureSearchProvenance(curated_seed_count=0,
        sources_enabled=['synthetic_fixture'], sources_returning=['synthetic_fixture'],
        search_conducted=True, searched_at='2026-01-01T00:00:00+00:00',
        search_queries={'synthetic_fixture':['invented synthetic comparator']},
        record_queries={KEY:['invented synthetic comparator']},
        note='Synthetic test provenance; no external retrieval occurred.')
    with host_first_observed_recipe_scope(proof), host_independent_units_scope(lambda candidate: candidate is c):
        population = compile_population(PopulationSpec.model_validate(skeleton.cohort.population_spec), c)
        plan, receipt = compile_progressive_plan(skeleton=skeleton, context=c, allowed_literature_citation_keys=[KEY])
        review = build_plan_scientific_review(context=c, plan=plan, literature=literature)
    return c, skeleton, population, plan, receipt, literature, review


