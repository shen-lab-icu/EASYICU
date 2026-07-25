# Research Know-How clinical and methods review protocol

This protocol upgrades a card from `curated_mvp` to `clinical_reviewed`. Changing
the string alone is invalid.

## Review unit

Review every `claim_id`, not merely the card title. For each claim confirm:

1. the text is appropriate for the declared topic and study family;
2. each cited source supports that exact claim and evidence scope;
3. eligibility and time-zero language does not silently become a universal
   disease rule;
4. stop conditions and user-confirmation requirements are complete;
5. advice is case-neutral enough for a shared card and does not encode a
   benchmark answer.

## Required attestation

Record reviewer/owner, review date, scope, literature-search cutoff, card
version, and whether both clinical and methods review were completed. Compute
`reviewed_content_sha256` over canonical JSON after removing the
`review_attestation` field:

```python
import json
from pathlib import Path
from easyicu.research_agent.know_how import reviewable_card_content_sha256

payload = json.loads(Path("card.json").read_text())
print(reviewable_card_content_sha256(payload))
```

Set `review_status="clinical_reviewed"` only after inserting a matching
attestation. Loader verification fails when content, version, or digest changes.

## Governance

- One person may hold both roles only when their clinical and quantitative
  methods expertise is documented; otherwise use separate reviewers.
- A new material citation or claim text requires a new card version and review.
- Literature cutoff is explicit; scheduled refreshes create additive versions.
- User-supplied cards remain `user_supplied_unreviewed` unless moved through a
  project review registry outside the user-controlled path.
