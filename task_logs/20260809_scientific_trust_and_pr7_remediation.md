# Scientific Trust Gate v1 + PR #7 remediation

- Date: 2026-08-09
- Branch: `fix/pi-workspace-review-20260809`
- Implementation commit: `961a6c7`
- Inputs: PR #7 review at `3234f4f`; repository-wide scientific trust review

## Closed review findings

### PR #7 security and provenance

- Research artifact HTML attributes now escape quotes as well as markup; embedded figures accept only bounded PNG data URLs. A hostile-renderer Node regression covers labels, paths, event-handler payloads and non-image data URLs.
- Human signoff verifies the complete artifact-name set and the selected artifact's name, SHA-256 and byte count. New, removed or changed artifacts make the signoff stale.
- All whitelisted artifact reads reject symlinks/non-regular files, enforce containment and a 2 MiB bound, open without following links where supported, and derive JSON, bytes and digest from one byte snapshot.
- Workspace reads/previews now return text, size and digest from one locked snapshot. Ancestor retargeting fails before filesystem mutation; file-occupied directory boundaries return stable Pi owner codes.
- Browser artifact projection retains only explicit `relative_path` path fields. Future `artifact_path`, `output_dir`, `cache_file`, `cwd` and similar host-path fields fail closed.
- The scoped security workflow now watches `agent_runs.py`, the shared renderer and its hostile-input regression. The unreachable `reportable` preview wording was removed.

### Clinical correctness and scientific authority

- The incomplete public `apache_ii_score()` no longer emits a partial value under the APACHE II name; it raises stable code `apache_ii_not_implemented`.
- KDIGO creatinine Stage 3 now evaluates the patient-specific acute increase (`current >= low_48h + 0.3`) instead of the erroneous fixed `low_48h <= 3.7` threshold.
- Canonical and compatibility SOFA-2 CNS implementations no longer score CAM-ICU positivity alone; GCS 15 receives one point only when delirium treatment is present.
- Generic missing score dependencies and failed blood-cell ratio derivations return missing + an attributable reason instead of zero or a numerator with percentage identity. SaFi exposes observed/imputed FiO2 provenance.
- Generic sepsis requests resolve to canonical 2016 `sep3`; the SOFA-2 variant is explicitly experimental, noncanonical and opt-in.
- A typed clinical-contract registry binds source/version/status, golden-vector version, validation scope, reviewer state and per-database conformance. Independent fixtures execute KDIGO, SOFA-2 CNS/aggregate, canonical Sepsis-3 and the SOFA-2 sensitivity phenotype.
- The registry intentionally says `independent_clinical_review_pending`; automated tests are not represented as clinician signoff.

### Agent and release trust

- Scientific capabilities now default to `analysis_only`. Only survival and exact adjusted association declare an explicit validator owner + typed receipt and remain reportable.
- Catalog counts, clinical conformance and the capability matrix are generated from live registries rather than hand-maintained numbers.
- The production runner base is digest-pinned with a lock record. A Linux workflow builds it, runs a networkless/read-only/cap-drop smoke test and creates a CycloneDX SBOM with commit-pinned actions.

## Verification

- Clinical/concept/capability/catalog suite: `206 passed`.
- Workspace/Web/research artifact/runner suite: `361 passed` after fixing one generated-catalog demo-count regression; focused rerun then passed.
- Earlier clinical marker gate: `11 passed`.
- Ruff on all changed Python owners and tests: passed.
- `git diff --check`, JSON parsing and workflow YAML parsing: passed.
- No Provider call, patient-data read, paid canary or manuscript result was produced.

## Explicit remaining boundaries

- This machine had no usable Docker daemon, so the real image build/smoke/SBOM is delegated to exact-head Linux CI; local tests validate the workflow and lock contract only.
- Database rows marked `mapping_only` in `docs/clinical_conformance_matrix.md` are not clinically validated. Independent clinician review is still pending.
- Splitting the two ~11k-line orchestrators, retiring the 1.x compatibility API and running measured six-database clinical conformance are separate P3/release programs, not represented as closed by this patch.
