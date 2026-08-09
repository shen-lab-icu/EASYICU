# EasyICU clinical conformance matrix

_Generated from `easyicu/data/clinical-contracts.json`. `mapping_only` means extraction wiring is covered; it does not claim that a database-specific clinical result has an independent gold-standard validation._

| Contract | Concepts | Definition/version | Source | Status | Validation | Golden vector | mimic | miiv | eicu | aumc | hirid | sic |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `sepsis3_2016` | `sep3` | Sepsis-3 / 2016 | PMID:26903338 | source_bound_golden | automated_golden; independent_clinical_review_pending | `tests/clinical_specs/sepsis3_2016.json` | mapping_only | mapping_only | mapping_only | mapping_only | mapping_only | mapping_only |
| `sepsis3_sofa2_sensitivity_2025` | `sep3_sofa2` | SOFA-2 Sepsis sensitivity phenotype / 2025-sensitivity-v1 | DOI:10.1001/jama.2025.20516 + PMID:26903338 (SOFA-2 Table 2) | experimental | automated_golden; experimental; independent_clinical_review_pending | `tests/clinical_specs/sepsis3_sofa2_sensitivity_2025.json` | mapping_only | mapping_only | mapping_only | mapping_only | mapping_only | mapping_only |
| `sofa2_aggregate_2025` | `sofa2` | SOFA-2 aggregate / 2025 | DOI:10.1001/jama.2025.20516 (Table 2) | source_bound_golden | automated_golden; independent_clinical_review_pending | `tests/clinical_specs/sofa2_aggregate_2025.json` | mapping_only | mapping_only | mapping_only | mapping_only | mapping_only | mapping_only |
| `sofa2_cns_2025` | `sofa2_cns` | SOFA-2 brain component / 2025 | DOI:10.1001/jama.2025.20516 (Table 2) | source_bound_golden | automated_golden; independent_clinical_review_pending | `tests/clinical_specs/sofa2_cns_2025.json` | mapping_only | mapping_only | mapping_only | mapping_only | mapping_only | mapping_only |
| `kdigo_aki_2012` | `kdigo_aki`, `kdigo_creat`, `kdigo_uo` | KDIGO acute kidney injury staging / 2012 | KDIGO 2012 AKI guideline (Chapter 2.1 staging criteria) | source_bound_golden | automated_golden; independent_clinical_review_pending | `tests/clinical_specs/kdigo_aki_2012.json` | mapping_only | algorithm_golden | mapping_only | mapping_only | mapping_only | mapping_only |
