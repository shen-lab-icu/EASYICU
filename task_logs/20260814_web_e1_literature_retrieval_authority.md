# Web E1 literature retrieval and screening authority — 2026-08-14

## Scope

- Development-only Web E1 iteration; no formal Canonical9 Provider batch.
- Fix the generic PubMed retrieval/screening owner exposed by the ordinary
  Copilot conversation. No benchmark question, shared prompt, rubric, or
  manuscript answer was changed.
- Exact implementation commit: `45566f0`.

## Confirmed failure

The rev6 E1 literature receipt `lit_523ab9b3309797435293d8f8` contained six
completed PubMed queries but zero records. Every query required the literal
Title/Abstract phrase `SOFA-2 sepsis`, an internal execution-label projection
that the relevant literature does not use. The bounded excerpt selector also
kept the first five exposure/design sentences, allowing exposure synonyms to
crowd a later outcome sentence out of the screening receipt.

## Owner-level repair

1. `literature_concepts.py` now owns conventional retrieval identities and
   stricter direct-screening terms for owner-issued concepts.
2. Retrieval uses bounded alternatives such as SOFA-2 or Sepsis-3 + SOFA;
   query aliases only improve recall and never confer comparator authority.
3. A separate all-year concept definition/development stratum retains the
   construct's source literature alongside direct, recent, review and
   database-specific strata.
4. Excluded exact-fallback records no longer erase the prespecified stratum
   sample; only a source-screened direct comparator receives priority.
5. `literature_excerpt.py` retains one extractive source sentence for each
   declared focus axis before filling the bounded excerpt with design context.
6. Both Web retrieval and sealed Research Agent screening consume the same
   composite concept identity. SOFA-2 alone cannot satisfy the experimental
   Sepsis-3/SOFA-2 exposure; both terms plus the exact outcome/population/design
   evidence remain required.

## Real PubMed probe

Using the exact current E1 exposure/outcome/population/database slots, the
updated owner retained an eight-record bounded set containing:

- PMID 26903338 — Sepsis-3 consensus (2016);
- PMID 28114553 — JAMA SOFA/SIRS/qSOFA in-hospital mortality validation (2017);
- PMID 41159833 — JAMA SOFA-2 development and validation (2025);
- PMID 41159829 — SOFA-2 rationale/method consensus (2025);
- current 2026 SOFA-2/sepsis cohort studies, including at least one record that
  passed all adult ICU, composite exposure, in-hospital mortality, extractive
  design and publication-type gates as a direct-comparator candidate.

The source set remains metadata/bounded abstract only. Human review must still
confirm time zero, estimand and adjustment comparability.

## Focused verification

- Web PubMed/Idea Mining: 16 passed.
- Research Agent comparator/excerpt: 7 passed.
- Web literature authority + Copilot handoff: 19 passed.
- Package dependency directions: 7 passed.
- Ruff, diff check and architecture ratchet: passed.
- Full exact-head CI: intentionally not run during E1 development iteration.

## Next gate

Restart Web on `45566f0`, repeat the authorized PubMed tool call in the same
ordinary E1 conversation, verify the new digest-bound receipt and then request
one fresh full run only through the human Plan-review pause. Do not approve or
execute the Plan before scientific review.
