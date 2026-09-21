# EasyICU desktop workspace design QA

QA date: 2026-09-19
Target: `http://127.0.0.1:8770/` on the active EasyICU project and task
Reference: Biomni project workspace, task conversation, trace/source viewer, Results shelf, skill catalogue and skill detail

## Reference evidence

- Biomni task workspace: `outputs/ui-audit/20260919-biomni-easyicu-full-controls/01-biomni-task-workspace-current.jpg`
- Biomni trace/source panel: `outputs/ui-audit/20260919-biomni-easyicu-full-controls/04-biomni-trace-source.jpg`
- Biomni Results shelf: `outputs/ui-audit/20260919-biomni-easyicu-full-controls/05-biomni-results-list.jpg`
- Biomni skill catalogue/detail: `outputs/ui-audit/20260919-biomni-easyicu-full-controls/06-biomni-skill-hub.jpg` and `07-biomni-skill-detail.jpg`
- EasyICU comparison captures: `08-easyicu-workspace-current.jpg` through `12-easyicu-skills.jpg`
- Detailed control audit: `outputs/ui-audit/20260919-biomni-easyicu-full-controls/audit-report.md`
- Biomni signed-in entry references supplied during review: `/var/folders/68/cz0swdq52vx1_rh5m4gql6v00000gn/T/codex-clipboard-c62c7af7-7e54-4bfa-bb2f-4586cb062d28.png` and `/var/folders/68/cz0swdq52vx1_rh5m4gql6v00000gn/T/codex-clipboard-66e9dfd8-3de7-4baa-9d89-582614fd66a4.png`

## Implemented result

### Persistent desktop workspace

- Project, task, materials and current results remain in one desktop frame while the center content changes.
- Data, Skills and Settings now open inside the same project context instead of replacing it with a separate full-page product shell.
- The Data rail keeps one active global state and one active subsection for extraction, patient review, cohort statistics and database comparison.
- Returning to the research workspace restores the exact project/task selection.

### Conversation, trace and source

- The conversation follows the Biomni reading order: prompt, concise response, expandable execution trace, final result, answer actions, follow-up questions and scientific review.
- Execution rows are clickable operation records with status and duration instead of static phase labels.
- Clicking an operation or a result source opens a center workbench tab with Input, focused code, Output, tool metadata and linked products.
- Figure source lookup resolves gallery records that omit `figure_id` through renderer metadata and opens the producing figure block rather than the top of a generic source file.
- The tested publication figure opens the focused range `1033–1069 / 3772` and still exposes the complete source on demand.

### Results and follow-ups

- Result files keep their content-type readers and link back to source evidence.
- Follow-up questions use compact rows. Each row supports continuing in the current task, preparing a new task, or dismissing the suggestion.
- Redundant helper phrases such as “点击后可编辑再发送” were removed from the visible hierarchy.

### Skills and methods

- The catalogue clearly distinguishes 17 launchable research workflows from 38 reusable method components: 55 catalogued capabilities in total.
- Search/category filtering, skill detail, method detail, enable/use actions and task starters were checked in the shared shell.
- Method components remain grouped by scientific role instead of being flattened into a copied 82-item list.

### Signed-in research entry

- Home opens a Biomni-like research entry inside the existing EasyICU project shell: project, task and materials remain visible while the unused result aside is hidden for the empty task.
- The question composer is the dominant object. Three reviewed Method Skill cards sit below it with a compact search and a direct route to the full catalogue.
- Selecting a Method Skill inserts both the method context and its editable scientific question into the composer. It does not send the question or start analysis.
- The entry loads the reviewed 15-workflow host catalogue and exposes Biomni's compact shuffle control beside search; each activation replaces the three visible cards without changing or sending the current draft.
- Entry and Skills use the same `330px` desktop project boundary, warm-paper palette and teal active state, so the content origin does not move when switching surfaces.

### Visual system

- Conversation, Data, Skills, Settings and the connection screen use the same neutral white/warm-gray/ink palette with teal reserved for selected navigation and active links.
- The Skills and Methods surfaces now use the exact desktop workspace palette: `#f7f6f2` canvas, `#fbfaf7` bounded surfaces, `#d7d4cc` hairlines, `#8f8b82` structural dividers and `#246366` active teal.
- Global module navigation matches the research workspace interaction: the active icon receives the teal fill while the full navigation row stays transparent.
- Guided, Data, Skills and Settings share one desktop rail geometry: a `59px` global rail inside a responsive `330–380px` project/context rail, so switching modules no longer moves the content origin.
- Desktop module changes use a short `120–210ms` view transition with the global navigation held in place. The fallback animates only the working surface, and reduced-motion preferences disable the effect.
- Broad fluorescent yellow fills and yellow link text were removed.
- Light hover, focus and selected states use dark teal text (`#174547`); white text is reserved for solid dark controls. The previously unreadable follow-up state now measures `9.21:1` against its light surface.
- Layout uses thin dividers, compact rows and cards only for bounded objects.
- Desktop is the acceptance target for this pass; mobile adaptation was intentionally excluded at the user's request.

## Live interaction verification

- Project/task context survived navigation across conversation, Data, Skills and Settings.
- Data subsection selection updated the center panel without losing the project rail.
- Skills showed 17 workflows and 38 method components; a method detail opened and its tabs/actions remained usable.
- Settings section navigation scrolled to the requested section and returned to the exact research task.
- Four follow-up rows exposed current-task, new-task and dismiss actions; dismissing one reduced the list, and reloading restored the persisted run result.
- A current-task follow-up populated the composer with the selected prompt.
- A result source opened the tabbed workbench at the focused producing-code range with Input and Output visible.
- The Skills catalogue, Methods catalogue and a method detail were reloaded in the live in-app browser after the palette correction. All three retained the warm paper canvas and the same teal active states; the browser-reported computed colors matched the workspace tokens.
- At the `1893px` verification viewport, Guided measured `59px / 330px` for global/project rails; Skills and Data both measured `59px + 271px = 330px`. Guided → Skills → Data → Guided transitions entered and completed without changing that boundary.
- The second follow-up prompt was checked in the live page with the shared hover/focus rule active: computed foreground `rgb(23, 69, 71)`, background `rgb(240, 239, 233)`, contrast `9.21:1`; the temporary composer draft used for interaction testing was cleared.
- Browser console warnings/errors after the final navigation: 0.
- The signed-in entry was exercised in the live in-app browser: selecting `队列描述与 Table 1` inserted its full scientific question and `方法 · 队列描述与 Table 1 / 仅分析` context without creating a message.
- Searching for `生存` reduced the entry grid to `生存与时间结局分析`; clearing the query restored all three cards.
- `浏览全部技能与方法` opened the 17-workflow / 38-method catalogue, and `返回研究工作区` restored the same empty task and prepared method question.
- The entry view hid the dormant right results rail and retained the same left boundary as the Skills catalogue; the selected method chip and visible controls remained dark on light surfaces.
- A stale result-preview state was reproduced and fixed: the empty-task layout now resolves to `330px + 1563px` at the `1893px` viewport, and the entry canvas is centered at its intended `1060px` maximum width instead of remaining trapped in a historical preview column.
- Biomni's signed-in empty-task entry was exercised directly: its Drive filename is a static row, and neither clicking the filename nor the temporary `Preview file` control opens a reader. EasyICU now follows that boundary: the empty task shows seven searchable static material rows with no resource buttons, and clicking a row leaves the right reader closed.
- A completed EasyICU task restores all seven material buttons and opens the selected resource in the right reader. Returning from that task to the empty task closes the previous reader, clears preview/source state, and restores the static material rows.
- The workflow shuffle changed `队列描述与 Table 1 / 研究因素与结局分布 / 生存与时间结局分析` to `缺失与测量审计 / 调整后关联模型 / 有序趋势与剂量反应`. Selecting `调整后关联模型` populated its reviewed question and method chip while the message count remained zero.
- The composer `+` now exposes one compact menu for PDF, article links, project resources, and skills/methods. Resources and skills open the same docked right drawer pattern, with working search, category filters, catalogue tabs, close control, and `Escape` behavior.
- Selecting `Cox 比例风险模型` from the 38-method catalogue inserted its editable scientific prompt and method context, closed the drawer, and did not send a message. Project resources likewise insert a version-bound result reference and close the drawer.
- The composer menu and drawers were checked without the old yellow treatment or instructional microcopy; every hover state keeps dark readable text on a light surface.

## Automated verification

- JavaScript syntax checks passed for the changed workspace, event and outcome modules.
- Focused backend/static/shell tests cover source-range resolution, source workbench wiring, persistent module navigation, follow-up actions, and static route reliability.
- Palette follow-up: desktop shell, static routes and Copilot static contracts passed (`246 passed, 5 warnings in 7.01s`); `git diff --check` also passed.
- Rail and transition follow-up: the same focused contracts passed (`247 passed, 5 warnings in 6.87s`), including shared geometry, route-transition wiring and reduced-motion handling.
- Interaction contrast follow-up: accessibility, Copilot static and desktop shell contracts passed (`173 passed in 6.31s`); the contract rejects white `accent-ink` overrides on the light research canvas.
- Final combined result: `263 passed, 5 warnings in 5.77s`; JavaScript syntax checks and `git diff --check` passed.
- Signed-in entry follow-up: the complete Copilot static contract file passed (`164 passed in 5.77s`); the Method Skill hub and Guided study-results JavaScript contracts passed; JavaScript syntax checks and `git diff --check` passed.
- Materials/shuffle follow-up: the complete Copilot static contract file passed (`164 passed in 5.94s`); both JavaScript contracts and syntax checks passed. The live empty-task → completed-task → empty-task path confirmed the resource interaction boundary and stale-reader cleanup.
- Unified composer picker follow-up: the focused JavaScript and Copilot static contracts passed; the live menu, resource drawer, 15-workflow/38-method drawer, search, category, selection and keyboard-close paths were exercised without sending a message.

## Scope and remaining product decisions

- Share/transfer are not copied because EasyICU has no approved external session-sharing policy.
- Destructive result deletion and account-oriented controls are not introduced as cosmetic parity features.
- EasyICU keeps its stronger scientific authority, audit and method taxonomy while reproducing Biomni's workspace behavior.

final result: passed

## 2026-09-20 Unified Skill catalogue pass

Source reference: Biomni presents its reusable packages under one top-level Skills catalogue, with package type expressed inside the catalogue rather than by hiding most skills behind a separate primary tab.
Implementation URL: `http://127.0.0.1:8770/?pi_project=draft_b0b76fc60ab1&pi_session=pi_87e47764dfe0f221e08f&ui_check=skillcatalog#skills`

- The default catalogue now exposes all 66 real, loadable built-in packages in one view.
- The same view preserves scientific scope through visible groups: 10 complete research workflows, 9 reusable analysis modules, six families containing 45 method packages, and 2 writing or figure modules.
- The Method library remains a focused 45-component view; the same packages are no longer hidden from the primary Skills catalogue.
- The catalogue summary states the full 66-package inventory before the first group. Seven planned methods remain excluded instead of inflating the total.
- Live browser verification opened the package catalogue, switched to the 45-item Method library, opened Decision-curve analysis, rendered its package-backed Overview, and exposed its three-document `SKILL.md` / `references` tree.
- At the 1893px viewport the document body and viewport were both 1893px wide; no horizontal overflow was present. Browser console warnings/errors: 0.
- Package rule: the main document and references are always present; `scripts/` appears only for a real capability-specific implementation. Shared host execution is not copied into each package.

final result: passed

## 2026-09-20 Composed research workflow pass

- Four new complete ICU EHR workflows compose registered modules around one primary scientific owner: adjusted exposure-outcome association, ordinal dose-response, fixed-landmark categorical association, and time-varying exposure survival.
- Each workflow declares an ordered six-phase project path, two or three reusable modules, a primary capability, execution mode, diagnostics, artifacts, and its claim ceiling. The cards therefore represent executable project structures rather than aliases for individual methods.
- Each workflow generates a four-document read-only package: `SKILL.md`, `references/composition.md`, `references/workflow_contract.md`, and `references/validation_framework.md`. A `scripts/` group is added only for real package-specific implementation files.
- Live browser verification showed 66 packages and 10 complete workflows, opened the adjusted exposure-outcome workflow, verified all six phases in Overview and the package tree, and found no horizontal overflow at the 1893px viewport.
- Automated verification: 48 JavaScript contracts and 211 focused Python capability, Planner, kernel, Web/static/accessibility tests passed; Ruff, compile, JavaScript syntax, and `git diff --check` passed.

final result: passed

## 2026-09-20 Skill overview and package-document pass

Source visual truth path: `/var/folders/68/cz0swdq52vx1_rh5m4gql6v00000gn/T/codex-clipboard-14e49d5c-3bc0-4854-abea-35f69e4a5b31.png`
Implementation screenshot: current-turn Codex in-app-browser captures for the trajectory Skill Overview and Files states; the browser API exposed the capture inline but did not expose a filesystem path.
Implementation URL: `http://127.0.0.1:8770/?pi_project=draft_b0b76fc60ab1&pi_session=pi_87e47764dfe0f221e08f&ui_check=skilloverview#skills`

- Source pixels: `2560 × 1318`.
- Implementation viewport and pixels: `1893 × 1324`, CSS viewport `1893 × 1324`, device pixel ratio `1`.
- Density normalization: both images were evaluated at their native density. The review compared the Skill detail content region because the overall application frames and viewport widths intentionally differ.
- State: built-in `轨迹表型发现与早期识别`, first in Overview and then in Files with `SKILL.md` selected.

### Full-view comparison evidence

- The reference treats Overview as the rendered main Markdown file and reserves Files for the package tree and selected-file reader. EasyICU now uses the same information architecture.
- The EasyICU project and Skill navigation remain in their established rails; the detail header, tabs, document surface, and package tree align on one content grid without horizontal overflow.
- Overview loads the package on entry and renders the complete `SKILL.md`; it no longer shows a second, field-composed summary.

### Focused region comparison evidence

- In Files, the visible tree is grouped as main file, `references`, and `scripts`, matching the reference package hierarchy. The trajectory package exposes seven real files.
- The active `SKILL.md` preview and Overview were compared from live rendered text after switching tabs. Their normalized document bodies were exactly equal.
- The rendered document exposes the same hierarchy in both states: title, use conditions, required data, workflow, implementation, validation, outputs, evidence boundary, failure behavior, and task starter.

### Comparison history

- Earlier P1: Overview was a shallow set of cards assembled separately from `SKILL.md`, while the complete file existed only under Files. This created duplicate truth and materially weaker content than the reference.
- Fix: built-in skills now load their package when opened; Overview renders the package's actual `SKILL.md`, while Files reuses the same loaded package for its tree and reader.
- Earlier P1: generated `SKILL.md` files only listed a prompt, host identifiers, a few inputs, diagnostics, outputs, and a brief boundary.
- Fix: the generator now writes operational instructions for use conditions, required decisions, ordered execution, host methods, composition, validation, evidence limits, failure behavior, and launch. It also distinguishes complete workflows, reusable modules, and method components.
- Post-fix evidence: live Overview and Files captures at `1893 × 1324`; exact text equality was true, package layout `clientWidth=972` and `scrollWidth=972`, browser console warnings/errors `0`.

### Required fidelity surfaces

- Fonts and typography: serif document headings and sans-serif body copy reproduce the reference's editorial reading hierarchy; code identifiers retain monospace styling. No heading markers or YAML frontmatter leak into the reading view.
- Spacing and layout rhythm: tabs, one bordered document surface, and the two-column file reader use consistent 8–16px controls and 22–34px document padding. Long content scrolls in the appropriate container.
- Colors and visual tokens: warm paper surfaces, restrained gray dividers, dark readable text, and teal active states remain consistent with the rest of the EasyICU workspace.
- Image quality and asset fidelity: this screen contains no source imagery to reproduce; existing product icons remain vector assets from the application.
- Copy and content: the visible Overview is now the package's actual instruction document. The additional governance language reflects EasyICU's host-owned execution and scientific authority rather than ornamental UI copy.

### Findings

- No actionable P0, P1, or P2 differences remain for the requested Overview/Files behavior.
- The different global navigation and application identity are intentional EasyICU product constraints.
- English package prose is retained because the reference package documents are English and the package is an auditable technical artifact; surrounding controls remain localized.

### Primary interactions tested

- Opened the six-workflow catalogue and selected the trajectory workflow.
- Confirmed the complete Overview loaded automatically from `SKILL.md`.
- Switched to Files and verified all seven files and three groups.
- Compared Overview and file-preview document text for equality.
- Returned to Overview and left it open for handoff.
- Checked the browser console and horizontal overflow.

final result: passed

## 2026-09-20 Data extraction module-first pass

Source comparison: Biomni's project entry and resource drawer keep the primary task surface concise, present a small set of useful choices first, and move detailed configuration behind an explicit interaction. Biomni does not expose an ICU extraction builder, so this pass adopts that hierarchy without copying a nonexistent scientific flow.
Implementation URL: `http://127.0.0.1:8770/?pi_project=draft_b0b76fc60ab1&pi_session=pi_87e47764dfe0f221e08f&ui_check=modulefirst#extraction`

- Removed the large recommended-extraction hero from the rendered setup path. All 19 modules are visible on first load and none is selected until the user acts; the six core modules remain first in the list, and “Use recommended modules” is an explicit shortcut.
- Replaced six competing cards with a two-column flat checklist. The collapsed state shows only the module name and feature count; the clinical description, feature list, units, and select-all controls appear only after that module is opened.
- Kept the other 13 modules behind one plain disclosure. The select-all/clear toolbar appears only in the full 19-module view, where it is relevant.
- Split setup into three in-place pages: Features, Cohort, and Export. Confirming features replaces the first page with cohort/time-window controls; confirming the cohort replaces it with export settings and the final extraction summary. Previous-page controls restore the earlier page without dropping any selection.
- Kept the former summary rail as one compact “Ready to extract” row on the final Export page only. Demo extraction remains runnable without an export folder; real extraction still requires a local destination.
- Removed the separate terminology strip; Data dictionary is now a secondary header link.
- Sepsis definition audit details appear only while a Sepsis module is open, instead of occupying the default setup.
- Responsive verification: two module columns at the normal 1280px preview, one column at 820px with the desktop rail present, and no horizontal overflow. The narrow-layout regression that squeezed Chinese headings vertically was found in live review and fixed.
- Interaction verification: the first page contains no cohort or export controls and opens with 0 modules / 0 features selected; six flat core candidates are visible and 19 after expanding additional modules. Confirm Features stays disabled until a selection exists. The explicit recommended shortcut selects six modules / 92 features; Confirm Features then opens the eight-preset cohort page, Confirm Cohort opens the three-format export page, and both Back controls preserve the selection. The demo Start extraction button appears only on the Export page.
- Browser console warnings/errors: 0. Automated verification: 48 JavaScript contracts and focused Python 96 passed / 5 warnings, including Web/static shell and accessibility contracts.

final result: passed
