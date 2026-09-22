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

## 2026-09-21 Workbench and reader coexistence pass

Source reference: Biomni keeps every secondary view as a center tab beside a persistent global rail, so a result file and the operation that produced it are read side by side; opening one never closes the other. Round-4 evidence: `outputs/ui-audit/20260921-biomni-easyicu-round4/audit-report.md`.
Implementation URL: `http://127.0.0.1:8770/?pi_project=draft_b0b76fc60ab1&pi_session=pi_87e47764dfe0f221e08f`

- A workbench tab (operation or figure source) and the right-side reader now stay open together. `sourceView.open()` no longer closes the reader and `preview.open()` no longer closes the workbench; only the session owner closes the workbench on conversation change. At 1600px and above the grid is `59px | conversation | workbench | reader`; below 1600px, or in focused reading, the conversation yields to `59px | workbench | reader`. The results shelf yields while the reader is open, as before.
- Between 1181px and 1599px an open workbench tab compacts the project panel to the 59px global rail instead of hiding it, so Home, Projects, Skills, Data and Settings stay reachable; closing the tab restores the full project panel.
- The composer action row wraps as a whole: toggle labels never fold and the model/send group moves under them in a narrow conversation column instead of overlapping.
- Shelf sections size to content within caps (progress 30%, compute 20%, notes 22%) and the results list takes the rest with a 220px floor: nine files visible at 1000px height where 3.5 were before.
- The compact rail heading keeps its search and new-task actions on one row with icon-only buttons below 1181px.
- Trace summaries show the step count and total elapsed time on the row, not only in the accessible name.
- Verification (focused): `test_pi_copilot_static.py`, `test_webserver_static_routes.py`, `test_coverage_intersection.py` 238 passed; headless Chrome layout measurements at 1024/1280/1366/1440/1542/1600/1893 with zero console errors and no horizontal overflow; the figure → source → focus → back path exercised in the live preview pane at 1796px.
- Not changed: per-step durations and linked products on pipeline rows (transcript rows carry synthetic timestamps and `projectPipelineSteps` strips resources by design), the three status signals for one retried run, file dates and source-in-trace on the shelf, the empty task created on each Home visit, and the reader as a center tab.

final result: passed

## 2026-09-21 Control walk pass (entry search, compact collapse, settings anchors)

Source reference: Biomni's entry search covers the whole capability list, collapsing the project panel keeps the global rail, and settings sections each open at the top. Round-5 walk notes: `outputs/ui-audit/20260921-biomni-easyicu-round4/round5-control-walk-notes.md`.

- The signed-in entry search now matches the full reviewed method catalogue (up to six cards) instead of the three cards on screen; clearing the query restores the current shuffle set.
- Collapsing the project panel at 1181px and above compacts the rail to the 59px global navigation; the floating restore chevron is gone and the nav's Projects button restores the panel before it opens the project picker.
- The settings page keeps `min(60vh,640px)` of room below the last section so every rail anchor reaches the top.
- The composer reference chip reads "已引用 · 研究总览 · 当前版本" without the instruction line.
- Verification (focused): static suites 239 passed; headless Chrome at 1542px confirmed the search, collapse/restore and anchor positions with zero console errors.
- Recorded, not changed: the in-rail project list (51 rows) should become a compact switcher plus a project page; task rows lack move/copy; no runtime indicator; the data-visualization error string is English and far from its button; the model menu shows the raw HTTP "Not Found" from a stale server process; skills have no per-skill deep link; Home still creates an empty task per visit.

final result: passed

## 2026-09-21 Draft reuse, project switcher, skill deep links, error wording

- Home, 新会话 and the rail's + 任务 open an empty draft the project already has before creating a session; the studyWorkspace owner exposes `isEmptyConversation` for that check.
- The global rail's Projects button toggles the in-rail project switcher, whose expanded list is capped at `min(34vh,420px)` so tasks and materials keep the lower half of the panel.
- Skill details are addressable as `#skills/<encoded id>`; the router accepts `screen/sub` hashes and the skills owner syncs its selection with the hash, including Back and cold loads.
- Model-menu transport failures read as sentences with the status instead of the bare HTTP reason; the data-visualization failure carries a localized prefix and the composer error banner can be dismissed.
- Verification (focused): static suites 246 passed; headless Chrome at 1542px confirmed draft reuse (session count unchanged), switcher toggle and cap, deep-link round trip and cold load, and the new wording with zero JS errors.

final result: passed

## 2026-09-21 — Live conversation walk on a fresh review instance (round 6)

- Scope: real end-to-end conversation on a second `EASYICU_HOME` instance
  (port 8517, current tree) with the cached eICU demo, compared with the
  Biomni reference flow. Notes and timeline:
  `outputs/ui-audit/20260921-biomni-easyicu-round4/round6-live-conversation-walk.md`.
- Regressions found and repaired: (A) the source-binding reader lost its
  「确认数据来源并继续」 control after the extraction page moved to
  `.eudata-head`/`.eudata-workspace` (`screens-extraction-embedded.js`; the
  desktop module shell was also nested inside the reader; scan placeholder
  "Unknown" no longer becomes the persisted source label); (B) metadata-only
  planning failed on demo databases with
  `research_pipeline_planning_identity_unavailable` because the identity
  registry has no `_demo` keys (`agent_pipeline_runs.py`).
- Verification: `node --check` on both JS owners; ruff on the Python owner;
  static suites 255 passed (`test_pi_copilot_static.py`,
  `test_pi_copilot_extraction_workspace.py`, `test_webserver_static_routes.py`,
  `test_webserver_desktop_module_shell.py`); planning suites 8 + 271 passed
  (`test_planning_source_menu.py`, `test_plan_change_requirements.py`,
  `test_plan_revision_references.py`, `test_pi_copilot_research_workflow.py`);
  live re-run on the review instance: the reader shows 绑定数据来源 → 确认
  数据来源并继续, the session's `data_source_authorization` became
  `confirmed`, and the regenerated plan run passed the identity gate and
  reached the progressive planner.
- Fix batch after the walk (same day): demo-prep job follow-through with a
  one-click source confirmation (`screens-guided-pi-host-jobs.js`, new owner
  wired into the shell, session view and events); planner progress tally on
  the running card, plan row and (then) compute shelf (`screens-guided-pi-childjob.js`,
  `screens-guided-pi-activity.js`, `screens-guided-pi-aside.js`); failure
  codes rendered as sentences (`screens-guided-pi-error-text.js`,
  `screens-guided-pi-confirmation.js`); streamed replies patched in place with
  a caret (`screens-guided-pi.js`, `screens-guided-pi-session-view.js`,
  `guided-pi.css`). Contract tests:
  `test_planning_progress_and_run_failures_read_as_progress_and_reasons`,
  `test_demo_source_preparation_reports_back_and_offers_one_confirmation`,
  `test_streamed_reply_grows_in_place_instead_of_rebuilding_the_conversation`;
  static suites 258 passed; live re-run on a fresh project confirmed the
  notice → one click → plan generation path and the tally across a reload.
- Fix batch 2: a refused automatic plan configuration
  (`agent_plan_patient_grouping_unavailable`) now renders a decision card
  with the plan preview, localized reason and the cohort admission options
  (`screens-guided-pi-plan-actions.js`, `screens-guided-pi-confirmation.js`,
  `screens-guided-pi-cohort-eligibility.js`, `screens-guided-pi-session-view.js`,
  `screens-guided-pi-error-text.js`); a passive reload offers an explicit
  「应用执行设置」card; the candidate-plan summary opens by default. Contract:
  `test_refused_plan_configuration_becomes_a_decision_card_not_a_banner`;
  `test_pi_copilot_plan_first_handoff.py` and
  `test_pi_copilot_cohort_selection_ui.py` updated to the open summary and the
  combined-card helper. Copilot package 1455 passed.
- Fix batch 3: reference-matched skin as one presentational owner
  (`css/guided-pi-skin.css`, route-scoped tokens: paper `#f7f6f2`, ink
  borders, lime primary; bordered trace rows, list-style follow-ups,
  reference composer/send/model pill/switch, ink-bordered cards, 40px section
  headers with a lime underline, lime rail disc) loaded after
  `guided-pi-desktop.css`; `guided-pi-cohort-eligibility.css` restyled in
  place; the cohort card recommends the first-admission rule when the
  compiler refused an all-stay configuration; the compute shelf reads the
  workflow `latest_run.gate_reason_code`. Contract:
  `test_reference_skin_is_route_scoped_and_loads_after_the_layout_owners`.
  Verified with headless captures at 1024 / 1280 / 1542 px and the preview
  pane at 1796 px (no horizontal overflow, rail width unchanged at 59px);
  copilot package 1456 passed.
- Fix batch 4 (user: 「对话的 UI 和思维链的展示没改」「思维链这么短，
  不能滚动多展示一些细节吗」): the reply turn now reads as the reference
  lays it out — the model's opening sentence above the traces, the trace
  list with the model's interim narration placed after the tool call it
  follows, the answer with its right-aligned action bar last; a running turn
  groups the same way (`screens-guided-pi-activity.js::renderTimeline`
  with a `renderSegment` callback from `screens-guided-pi-session-view.js`;
  `afterSteps` recorded on live and persisted assistant segments in
  `screens-guided-pi.js` / `screens-guided-pi-transcript.js`). Trace rows
  are one button each (16px icon · 12px/500 label · clock + 10px mono
  duration · chevron), the list scrolls at 300px with 「展开全部」, the
  live row shows a spinner and live elapsed. The model's reasoning summary
  (Pi `thinking` content, what the provider returns for display) now crosses
  the bridge bounded to 4,000 chars as `thinking_start/delta/end` events and
  `thinking` transcript parts (`event-projection.mjs`), is sanitized like
  reply text (`projections.py::project_transcript`), streams into a
  「正在思考…」 row patched in place, and persists as rows whose duration
  comes from the lifecycle phase (`transcript.js`). Tool arguments, partial
  output and thinking signatures still never cross. Lifecycle turns are
  matched to user messages by time (15 s window) instead of list position,
  which had attached a host-advanced turn's trace to the next user question.
  Contracts: `test_reply_turn_reads_as_intro_traces_with_reasoning_and_answer`,
  `test_sidecar_projects_bounded_reasoning_summaries_for_the_trace`,
  `test_reasoning_summaries_are_sanitized_like_replies_and_only_for_the_model`;
  the former "reasoning never crosses" pins in
  `test_sidecar_contract_hides_reasoning_and_enforces_token_budget`,
  `test_sidecar_projects_safe_agent_activity_and_tool_receipts` and the
  activity privacy notice were updated to the new boundary (product decision
  by the user, 2026-09-21). Live: a real turn on the review instance showed
  已读取科研流程 → Planning to inspect candidate plan (7.0 s) → 已读取科学计划
  → Balancing three-sentence constraint (18.6 s) → answer, identical after a
  reload. Copilot + static + planning suites 1490 passed
  (`test_repository_contract.py::test_static_build_metadata_does_not_require_setuptools_scm`
  fails on the committed `pyproject.toml` build requirements, unrelated).
- Fix batch 5 (user: the 待办 panel 「展示的也不清楚」, compare Biomni's): the
  reference 待办 panel is a task checklist (rows with a status mark; its empty
  state says 多步骤任务会显示待办项). Ours was a "current stage" head with a
  gate sentence (「科学计划审阅要求先形成新的研究/计划版本，当前不能继续分析」),
  a 「后续阶段」 line and a collapsed 1/7 stage list. It is now one checklist:
  every stage is a row (✓ done · lime disc = current · empty circle = open ·
  dashed lock = later · dotted = optional), the current row states what is
  needed now in the decision card's own words (`pendingDecisionTitle` from
  the confirmation owner / data-consent / cohort question) or the running
  task with a spinner (`activeTaskTitle`), and carries the one action
  (「查看待确认事项」 or the review action); the stage reason is only the
  fallback; the 1/7 count moves to the section header
  (`screens-guided-pi-aside.js`, `screens-guided-pi.js`, `guided-pi-skin.css`).
  To stay under the shell's size ratchet, the live turn stream
  (`handlePiEvent` and the in-place streaming patchers) moved to a new owner
  `screens-guided-pi-live-stream.js` (shell 1902 → 1774 lines); test pins
  that read those functions from the shell now read the owner. Contracts:
  `test_workflow_to_dos_read_as_a_checklist_with_the_current_decision`
  (replaces `test_workflow_stage_list_preserves_expansion_below_current_results`),
  `guided_workflow_copy.test.js` and `guided_pi_study_results.test.js`
  unchanged and passing. Screenshot: `skin-1542-todos.png`. Copilot + static
  + ownership + planning suites 1480 passed; the one failure,
  `test_owner_js_files_do_not_grow_past_their_ratchet` for
  `screens-guided.js` (4221 > 4177 + 40), comes from that file's own
  uncommitted +11/-2 change, not from this work.
- Fix batch 6 (user, with two Biomni crops: 「他们的 trace 比较丰富」 and
  「他们还会提供追问的内容」): Biomni's rows are titled by the model's own
  one-line description of each step and interleaved with its narration; ours
  carried host labels only because the model went straight from reasoning to
  tool calls. Two prompt rules in the bridge (`node_app/src/main.mjs`): a
  trace-narration rule (one sentence in the user's language before the first
  tool call saying what it is about to check and why; one more when a result
  changes the plan) — those sentences are what the turn layout from batch 4
  places above and between the trace rows; and a follow-up-questions rule (a
  completed answer ends with a '**可以继续问：**' / 'Follow-up questions:'
  block of 2–3 questions, not at gates or where the Next step already offers
  choices). New owner `screens-guided-pi-follow-ups.js` lifts that block out
  of the reply (`split`) and renders it as the reference suggestion list
  (lightbulb heading, ↳ rows on a 5% lime tint) under the latest reply only;
  a click sends the question as the researcher's own message with no grant
  (`screens-guided-pi-session-view.js`, `screens-guided-pi-events.js`; the
  live-stream patcher strips the block while streaming). Contract:
  `test_model_follow_up_questions_are_lifted_from_the_reply_and_offered_as_suggestions`.
  Two more repairs surfaced by the live check: a finished host receipt
  recorded after the latest reply (a review click, a plan job) no longer
  demotes that reply, so its choices and suggestions stay interactive
  (`latestAssistant` in the session view), and reply markdown now renders
  GitHub tables instead of raw pipes (`screens-guided-pi-markdown.js`,
  `test_reply_markdown_renders_github_tables`). Live (review instance run
  from a source snapshot because the checkout's untracked
  `research_agent/skills/` package shadows `skills.py` and breaks the
  webserver import; the bridge runs from the packaged source dir when no
  installed runtime matches): a tool-using question now opens with the
  model's own sentence above the trace (「我先列出当前运行对应的受治理产物
  及其可打开资源，再按用途说明。」), the answer's table renders, and three
  「继续追问」 suggestions follow the action bar; clicking one sends it.
  Screenshots: `skin-1542-narrated-turn.png`, `skin-1542-followups.png`.
  Copilot + static + ownership + planning suites 1482 passed (the
  `screens-guided.js` ratchet failure is the other session's).
- Fix batch 7 (user: 「把这套 skin 也铺到数据工作台和技能页，但不建议用这个
  黄色」): the accent is now one token pair in `guided-pi-skin.css`
  (`--skin-primary:#a3dbd6`, `--skin-primary-rgb`) and every tint derives
  from it; the reference lime is gone. The chosen replacement is a soft
  clinical teal from EasyICU's own accent hue (`tokens.css` `--accent`,
  hue 205) so the conversation route matches the rest of the product; ink
  stays on every fill (send disc, rail disc, primary buttons, current to-do).
  New presentational owner `css/module-shell-skin.css` (loaded after
  `skills-hub.css` / `extraction.css` / `guided-pi-skin.css`) remaps the
  module shell's `--sk-*` and generic tokens to the same paper / ink /
  accent and reshapes controls: serif page titles, 1px ink cards at 8px,
  6px controls, ink-bordered search and segmented tabs, 40px section headers
  with the accent underline, the conversation route's rail (ink brand disc,
  accent active disc) and context aside, the data route's module bar with an
  accent underline, composite fields with one border. Skills, Data (source
  binding, demo modules, cohort statistics) and Settings were captured at
  1542 px: `skin-1542-skills.png`, `skin-1542-data.png`,
  `skin-1542-data-modules.png`, `skin-1542-cohort.png`,
  `skin-1542-settings.png`, `skin-1542-conversation-teal.png`. Contract added
  to `test_reference_skin_is_route_scoped_and_loads_after_the_layout_owners`
  (single accent token, no lime literal, module skin load order and scope,
  no `.gpi-` selectors in the module skin).
- Fix batch 8 (user: 「借鉴 biomni 的时候会不会想一下我们的适配性……
  他们是不是有云服务计算，我们这个模块直接抄但是却没有对应功能」): a fit
  audit of every control the conversation route borrowed from the
  reference, judged by whether an EasyICU owner stands behind it.
  Backed and kept: 待办 (the seven governed stages + the current decision),
  成果 (the run's artifacts with the run id under 成果来源), 笔记 (per-project,
  browser-local — the caption says so), 布局, the trace disclosure and
  source tabs, both follow-up lists, 科学审阅, the composer + menu (上传
  PDF / 粘贴文章链接 / 项目资料 / 技能与方法), the access-level menu
  (`turnGrants()`), the model menu, copy / edit-and-resend / regenerate,
  Skills upload and Skill Builder. Copied without a function, now removed:
  (1) the 计算 aside panel — the reference lists cloud machines, background
  workers, jobs and pipelines; EasyICU executes locally and
  `remote_compute_enabled` is a disabled capability with no adapter. Its
  two real facts moved onto the current to-do row (`runFacts()` in
  `screens-guided-pi-aside.js`: the running job's stage sentence under the
  task title; after a failed run, the cause with its run id in error ink);
  the run id of a finished run stays under 成果来源. (2) the composer's
  「自动推进」 switch — `state.autoMode` had no consumer, while its tooltip
  promised auto-continued confirmations; the access-level menu beside it is
  the real approval control. (3) 好评 / 差评 on replies — a class toggle
  lost on the next render with no feedback ledger behind it. Dead CSS for
  all three removed (`guided-pi-workspace.css`, `guided-pi-desktop.css`,
  `guided-pi-skin.css`, `workspace-canvas.css`); the 布局 menu offers the
  three panels the owners fill. Contract:
  `test_reference_controls_without_an_easyicu_owner_are_not_rendered`; the
  checklist contract now covers the stage line and the failure line.
  Verified on the review instance (三面板 aside, no switch, reply actions =
  复制 / 重新生成; headless 1542 px capture `skin-1542-fit.png`).
  Follow-ups if wanted later, each needing a backend owner first: a
  feedback ledger before ratings return; project notes stored in the
  project folder instead of the browser; a run-history list (only
  `latest_run` is projected today).
- Fix batch 9 (user: 「这里的项目和下面的任务有什么区别，为什么要有两个，
  biomni 就只用了一种」): the reference has both levels too — a "PROJECT /
  name ▾" combobox (popover: current project · 所有最近任务 · 创建新项目)
  over the task list — but its project is a light container of tasks and
  files, so one level is all a user notices. In EasyICU the project *is*
  the study: question, data binding and consent, cohort rule, plan
  approval, runs, evidence and results, one local folder; the rows under
  it are conversations about that study (待办 / 成果 / 资料 are per
  project). What made ours read as two lists of the same thing: the rail
  printed the project name twice (a brand line, then — under the
  new-project button — a picker summary that expanded inline into a
  second list styled like the rows below) and called the second level
  「任务」, the word the running research job already uses. Now the picker
  summary is the heading's only control (kicker 项目 + name + chevron, the
  reference combobox; `screens-guided-projects.js` no longer renders a
  brand line), its list opens as a bordered popover anchored under the
  top block so 「新建 / 打开研究项目」 stays reachable and the conversations
  never move (`guided-pi-desktop.css`, `guided-pi-skin.css`), the
  one-line definition 「研究配置、运行、证据和对话历史都保存在这里」 stays
  visible inside it, and the second level is named 对话 everywhere it is
  one (rail heading and + button, search, empty group, row menu, header
  新对话, rename/remove prompts, 「在新对话中追问」; 「科研任务」 remains
  the running job). Five labels marked `sr-only` (a class no stylesheet
  defined) rendered as visible text over the search inputs and the
  results toolbar; they now use the shell's `shell-sr-only`. Contract:
  `test_rail_reads_as_one_project_switcher_over_its_conversations`;
  `tests/js/guided_project_handoff.test.js` and the rail pins in
  `test_webserver_static_routes.py` repointed. Capture:
  `skin-1542-project-switcher.png` (closed / open). Suites 1485 passed.
- Fix batch 10 (user: 「biomni 还有努力的程度，+ 里面的内容也比我们丰富」):
  measured the reference's + menu (240px ink-bordered 6px panel, 4px pad,
  32px rows 14px/20px with 16px icons, 1px separators, "Type @" / "Type /"
  rows whose hint is a 10px mono kbd chip). Its length comes from cloud
  features EasyICU does not have (file/folder upload into a sandbox, cloud
  import, GitHub/Linear connectors, integrations, alerts); the fit rule
  from batch 8 says those are not copied. What EasyICU does have behind the
  menu — a literature PDF or article link for idea mining, the project
  resource picker, the Skill picker — now uses the reference anatomy
  (`guided-pi-skin.css`), and the two drawers open from the keyboard as
  the rows promise: "@" opens 项目资料 and "/" opens 技能与方法 when typed
  at the start of the message or after a space, the trigger character is
  removed, IME composition is ignored (`composerPickerTrigger` in
  `screens-guided-pi-events.js`; hints in `screens-guided-pi-idea-source.js`).
  Effort ("努力程度"): EasyICU's runtime supports Pi thinking levels
  (off / minimal / low / medium / high, clamped per model), the session
  route and record already carry `thinking_level`, but the service and the
  bridge force "off" since the 2026-08-08 boundary hardening ("raw provider
  reasoning is neither streamed nor persisted"). Batch 4 replaced that
  boundary with bounded, sanitized reasoning summaries, so the original
  reason no longer applies as stated; enabling a per-session effort menu is
  a governance and cost decision (longer reasoning, more tokens) and needs
  `pi_copilot/service.py` (`resolved_thinking = "off"`), which another
  session is editing — left as a decision for the user rather than shipped
  as a dead control. Capture: `skin-1542-plus-menu.png`. Contract extended
  in `test_copilot_plus_menu_keeps_article_url_entry_in_the_conversation` (kbd hints, trigger helper). Suites passed.
- Fix batch 11 (user: 「默认一般应该是中吧」 — decision to ship the effort
  control with medium as the default): the per-session effort level is now
  real end to end. Bridge (`node_app/src/main.mjs`): `session.create`
  honours `thinking_level` (validated, default medium; Pi clamps to the
  model) instead of forcing "off", and a new `session.set_thinking_level`
  operation switches a session between turns and reports the effective
  level. Service (`pi_copilot/service.py`): `resolve_thinking_level`
  (default `DEFAULT_THINKING_LEVEL = "medium"`) replaces the forced "off"
  in `create_session`, `_ensure_open` reopens with the record's own level,
  and `set_thinking_level` (busy → 409, unknown level → 422) writes the
  bridge's effective level back to the record. Route:
  `PiSessionCreateRequest.thinking_level` defaults to medium and
  `POST /api/copilot/pi/sessions/{id}/thinking-level` takes
  `{project_id, thinking_level}`. Frontend: new owner
  `screens-guided-pi-effort-menu.js` renders 关闭 / 低 / 中 / 高 beside the
  model chip with one-line descriptions and the footnote that the trace
  shows only the bounded reasoning summary at every level; a new
  conversation starts at the remembered choice (medium by default), a pick
  calls the route and updates the session in place; `api.js`
  `setPiCopilotThinkingLevel`. Contracts: sidecar smoke test (create at
  medium, set to high, reject "max"), service
  `test_effort_level_is_a_per_session_choice_that_defaults_to_medium`,
  route defaults/422s, and the static owner test
  `test_effort_level_is_a_per_conversation_menu_on_the_composer`. Live on
  the review instance (restarted from the synced snapshot): an existing
  session showed 关闭 (its stored level), switching to 中 returned
  `thinking_level: "medium"` from the route, and the next turn answered in
  7.0 s with a reasoning row in its trace. Capture:
  `skin-1542-effort-menu.png`. Sessions created before this change kept
  their stored "off" until switched (batch 15 reads them at medium). Note:
  `service.py` and
  `routes/pi_copilot.py` are being edited concurrently by another session
  (its `message_origin` hunks); the effort edits sit in separate regions.
  Wire-level proof (user: 「这些功能是真实存在的吗，会怎么影响我们的对话」):
  `effort-probe/` runs the real sidecar against a recording fake
  `/v1/chat/completions` endpoint. Captured request bodies: a session
  created at medium sends `reasoning_effort: "medium"`; at "off" the field
  is absent (the model service applies its own default — the state of
  every conversation before the menu, which is why reasoning rows already
  appeared at "off"); at low, `"low"`; after `session.set_thinking_level`
  high, the same session's next prompt sends `"high"`. For this
  deployment (provider `cliproxyapi`, transport `openai-completions`) the
  level is the OpenAI-style `reasoning_effort` on every conversation reply
  and tool-choosing turn; it does not touch the Research Agent's own
  provider calls (`provider_adapter.py` fixes its reasoning effort per
  account profile). The menu therefore offers only 低 / 中 / 高 — the
  explicit values every OpenAI-style endpoint accepts — and shows a
  pre-menu session as 「未指定」 (and a model-clamped "minimal" as 最低)
  without offering either; the footnote names what the level affects.
- Fix batch 12 (user: 「biomni 的 + 里有 connector、skills 各种，我们是没有
  这些吗」): mapped the reference's nine + rows to EasyICU owners. Backed
  elsewhere already: Resources (@) → 项目资料; Skills (/) → 技能与方法 (66
  built-in packages + user SKILL.md); Connectors → Settings 连接器 (PubMed,
  Zotero) and MCP 工具 (Streamable HTTP servers with explicit tool
  allowlists, frozen into new sessions, callable through
  `easyicu_call_mcp_tool`); Select from uploaded files → 项目资料. Not
  present by design: arbitrary file/folder upload into a sandbox (patient
  rows must not enter the model; data enters through the governed source
  binding), cloud import, GitHub/Linear connectors, external integrations,
  alerts. The one honest addition: a 「连接器与工具 ›」 group in the + menu
  (`screens-guided-pi-idea-source.js` `extensionsHtml`) showing this
  conversation's frozen MCP servers with allowlist counts, its user Skills,
  and the PubMed / Zotero switches, each row opening the matching Settings
  capability tab (`screens-guided-pi-events.js` sets the existing
  `easyicu.settings.openCapabilityTab` key, which `screens-settings.js` now
  honours for any known tab); the data comes from the session record and
  page settings (`composerExtensions` in the session view), no request.
  Verified live: rows read MCP 服务 · 本对话未固化任何服务 / 用户 Skill ·
  未安装 / PubMed 已启用 / Zotero 未启用, and the Zotero row opened Settings
  on the 连接器 tab. Capture: `skin-1542-plus-connectors.png`. Contract in
  `test_copilot_plus_menu_keeps_article_url_entry_in_the_conversation`.
- Fix batch 13 (user: 「请帮我修复」 — the first item of the recommended
  list, the project run record): the workflow route already read the last
  ten runs bound to the study (`list_bound_run_history`) but projected only
  the authoritative one. `ProjectWorkflowProjection` now carries `runs`
  (`project_run_history` in `pi_copilot/workflow.py`: run id, type, engine,
  run/gate status, gate reason code, readiness, artifact count and names,
  plan availability, updated time, `authoritative`; capped at 10 rows and
  40 names, passed through `ensure_safe_projection` — no run directory,
  gate checks or payloads cross). The shell keeps it on `state.workflow.runs`
  and the 成果 section renders 「运行记录 · N」 under the current run's
  files (`runHistoryHtml` in `screens-guided-pi-aside.js`): newest first,
  type · outcome (待审阅 / 失败 / 已取消 / 运行中 / 阻断 / 已通过 /
  已有记录), 当前 pill on the authoritative run, time · file count, the
  failure cause as a sentence via `runFailureText`, and the run id; open by
  default only when the shelf itself is empty, disclosure state kept
  across re-renders. Rows are facts, not links: historical artifacts need
  their digests to open and the route serves only the authoritative run's
  refs. Contracts:
  `test_run_record_projects_every_bound_run_without_paths` (backend) and
  `test_results_section_lists_the_project_run_record` (owner). Live on the
  review instance: the two-run project lists both failed planning attempts
  with their causes (「候选计划未通过编译校验…」, 「所选数据库缺少可用于规划的
  ICU 住院身份定义…」) and marks the current one; capture
  `skin-1542-run-record.png`. Suites 1489 passed.
- Fix batch 14 (user: 「继续」 — the next recommended item, project notes as
  project memory): 笔记 now lives as `project_notes.md` in the project's
  local folder instead of the browser. Owner: `guided_sessions.py` (which
  already owns the folder's `guided_draft.json` /
  `guided_copilot_session.json`) gained `read_project_notes` /
  `write_project_notes` — draft-registered folder only (`_safe_project_dir`),
  20,000-character cap, CRLF normalised, atomic replace, no file created for
  an empty note, `available: false` for a project without a folder;
  routes `GET`/`POST /api/guided/projects/{project_id}/notes` in
  `routes/guided.py` (the shared Copilot route file is untouched). The
  aside's notes panel (`screens-guided-pi-aside.js`) reads the folder when
  the project changes, saves 800 ms after the last keystroke, reports
  「已保存到项目文件夹 · time」 / 「有未保存的改动」 / 「正在保存…」 / the
  error, keeps the browser copy only as the fallback for a project without
  a folder (「此项目没有本地文件夹，笔记仅保存在此浏览器」), migrates a
  pre-existing browser-only note into the folder once, and never rewrites
  the textarea while it has focus. Contracts:
  `tests/webserver/test_webserver_project_notes.py` (round trip, limits,
  fallback, routes), the owner test
  `test_project_notes_are_project_memory_in_the_project_folder`, and the
  route inventory (`test_webserver_route_contracts.py`, which now also
  lists the effort route). Live: typing 「先看乳酸分布，再决定分组阈值。」
  created `project_notes.md` in the project folder within a second, the
  browser copy was cleared, and a reload read the note back from the
  folder. Capture: `skin-1542-project-notes.png`. Whole webserver test
  package: 2816 passed (only the other session's `screens-guided.js`
  ratchet deselected).
- Fix batch 15 (user: 「继续做迁移」 — the last item I could do alone, the
  pre-menu conversations' effort level): every conversation created before
  batch 11 carries the forced `"off"` in the session metadata store (the
  user's real store: 100 of 100 rows), so its composer read 「努力 · 未指定」
  and had to be switched by hand. The store owner (`service.py`
  `_read_records` / `_write_records`) now versions the store:
  `easyicu.pi-copilot-store/1` predates the menu, so `"off"` (or a missing
  field) there is the absence of a choice and is read as the default
  `medium` — the level a reopen sends to the bridge — and the first write
  persists the result under `easyicu.pi-copilot-store/2`, where `"off"` is
  a real state (a model clamp, or an explicit request) and is preserved.
  No record is touched until something writes, so the migration is one
  metadata rewrite per store and nothing else changes. The menu's "off"
  wording no longer describes it as the pre-menu state. Contract:
  `test_session_store_reads_pre_effort_menu_sessions_at_the_default_level`
  (read at medium, reopen at medium, first write bumps the schema with the
  migrated levels, `"off"` preserved in a /2 store). Live on :8517 (its
  store was /1 with two `"off"` rows): after the restart the sessions API
  listed all three at medium while the file was still /1; one
  content-preserving write (`thinking-level` medium on an already-medium
  session) rewrote the file as /2 with all three at medium; the untouched
  pre-menu session `pi_1fdc…` in 「乳酸与 ICU 死亡率」 renders 「努力 · 中」
  with 中 checked and no fourth row. Capture:
  `skin-1542-effort-migrated.png`. The user's :8770 server was not
  restarted; its store migrates on its next start. Whole webserver test
  package: 2817 passed (same deselection).
- Fix batch 16 (user: 「不管点哪个按钮弹出来的窗口，点其他地方都不会还原，必须
  重新点那个按钮才能关闭」 and 「高中低下面那一大段文字没必要，完全访问下面也
  是」): every floating menu is a native `<details>`, which only toggles
  from its own summary, and only three of them had ad hoc outside-click
  code in the conversation's events owner — so the access, effort, model,
  conversation ••• and project-switcher menus stayed open until their
  button was clicked again, and two could be open at once (the user's
  capture: 自动审批 and 努力 open together). One shell owner,
  `js/popover-menus.js` (loaded after `composer-keyboard.js`), now gives
  every `<details data-popover-menu>` the missing behaviour: opening one
  closes the others (`toggle`, capture phase), a press or a focus move
  outside closes it (`pointerdown` / `focusin`, capture), Escape closes it
  and returns focus to its summary (skipped when an inner picker already
  handled the key). Marked: the composer's + / 自动审批 / 努力 / model
  menus, the header's 布局 and 更多 menus, the rail's conversation ••• menu,
  the project switcher (only while it is a switcher — with no project
  chosen or in manage mode the list is the surface), and the skills page's
  新建技能 and ··· menus. Inline disclosures (run record, traces, evidence,
  the + menu's 连接器与工具 group) are not menus and stay put. The events
  owner keeps only its close-after-action for the 更多 and + menus; its
  outside-click and Escape branches are gone. Both footnotes were removed:
  the effort popover and the access popover are rows only (the chip's
  tooltip already explains the current level). Contract:
  `test_floating_menus_share_one_dismissal_owner` (markup marks, no
  reconstructed policy in the events owner, and the owner's behaviour
  driven through a fake document). Live on :8517: 自动审批 → 努力 closes
  the first; an outside click closes 努力; Escape closes + and focuses its
  summary; 更多 closes the model menu; outside clicks close 更多 and the
  project switcher; the same summary still toggles; choosing 请求访问
  inside applies and closes; Tab inside keeps the menu, focusing the
  composer closes it; the skills page's 新建技能 menu closes on an outside
  press. Capture: `skin-1542-popover-dismissal.png`. Whole webserver test
  package: 2820 passed (same deselection; two batch-8 pins loosened to
  prefixes because the other session is extending `runFailureText` with a
  detail argument).
- Fix batch 17 (user: 「点击模型切换，窗口直接在下面，我都看不到」 and 「你需要
  细致地排查每个 webui 的元素、模块」): instead of fixing the one menu, two
  repo tools now press everything. `tools/audit_web_popovers.py` opens every
  `details>summary` / `[aria-haspopup]` / `[aria-expanded]` on every route
  at 1542×1000, 1280×760 and 1180×680 with a real mouse press and, for the
  floating content it reveals, checks inside-viewport, not clipped by an
  overflow ancestor, not covered at its centre, and closes on Escape /
  outside press / its own opener (116 openers, 58 floating menus per run).
  `tools/audit_web_fit.py` checks the page at rest: controls covered at
  their centre, floating elements outside the viewport, silently clipped
  text, controls under 24 px, horizontal page overflow. First run found:
  (1) the model menu kept the header-era `top:calc(100% + 7px)` after
  moving to the composer, so it opened below the composer, off-screen and
  clipped by `.gpi-panel` at every size — the user's report; (2) at
  1280×760 the conversation panel is 520 px wide and the effort menu,
  right-aligned to its chip, ran past the panel's left edge; (3) the rail's
  folder menu (270 px, right-aligned) overhung the rail's left edge at
  ≤1180 px and was clipped; (4) the context strip's 「已确认数据源」
  disclosure floats but did not dismiss; (5) hit areas under 24 px: the run
  record summary (18 px), message action buttons (22 px), activity
  summaries (23 px), breadcrumb links (20 px). Fixes: a new small owner
  `css/guided-pi-composer-menus.css` (guided-pi.css was at its 600-line
  budget) anchors every composer menu to the compose card — above it,
  inset from its edges, leading menus left and trailing menus right, the
  model list scrolling inside a `max-height` — and, for the entry
  composer high on an empty page, opens them below the card with a
  shorter list on windows under 760 px tall; the folder menu fills its
  button (`left:0;right:0`); the data-source disclosure carries
  `data-popover-menu`; the four hit areas are 24–26 px. After the fixes
  the popover audit passes every floating menu except the folder menu's
  Escape (its open state lives in `screens-guided.js`, the other
  session's file — outside press already closes it), and the fit audit
  flags nothing. Contract:
  `test_composer_menus_open_inside_the_conversation_panel`. Captures:
  `skin-1280-model-menu.png` (conversation, 520 px panel),
  `skin-1280-entry-model-menu.png` (entry composer); reports
  `audit-web-popovers.json`, `audit-web-fit.json`. Whole webserver test
  package: 2830 passed (same deselection).
- Fix batch 18 (user: 「请帮我全部完成」 — checkpoint, rules, audits in the
  suite): three other Claude sessions share this checkout (Demo workflow,
  论文实验进度, 项目修稿). Asked over SendMessage before staging: Demo owns
  `gate_detail` (workflow / projections / aside / error-text), the retired
  `data-gpi-legacy` entry and pins `reader-notes1` / `demo-entry2` /
  `no-legacy1`; 论文 owns `trajectory_design` and the family-spec planner;
  `message_origin` / `easyicu_resume` belong to neither and are not this
  session's either. The checkpoint therefore stages only this session's
  hunks: each file's version is HEAD plus its own hunks, written with
  `git hash-object` + `update-index`; the working tree is untouched and every
  other session's change stays unstaged. The staged tree was exported next to
  a pure-HEAD export and both ran `tests/webserver` + `tests/governance`:
  the only failure the staged tree added over HEAD's 21 environment-bound
  failures was `test_owner_js_files_do_not_grow_past_their_ratchet` —
  `screens-guided.js` 4221 lines against 4177 + 40, from this session's
  nine-line Projects-button handler (earlier batches wrongly attributed that
  failure to another session). The handler moved to the project-rail owner
  (`EU_GUIDED_PROJECTS.showProjects()`); the guided screen now delegates and
  is 4202 lines, below HEAD's 4212. The same pass fixed the folder menu's
  Escape (open item from batch 17): its toggle re-renders, which dropped
  focus to `<body>`, so the shell-scoped keydown branch never ran.
  `popover-menus.js` gained `register({ isOpen, contains, close })` for a
  menu whose open state lives in its owner; the guided screen registers the
  folder menu and its own outside-click and Escape branches are gone, and
  Escape returns focus to the toggle. `tools/audit_web_popovers.py` now
  passes every floating menu at 1280×760 and 1180×680. Rules: the workspace
  `CLAUDE.md` frontend section gained two bullets (borrow anatomy, not
  controls; floating menus are `details[data-popover-menu]`, anchored to the
  container they must stay inside, audited after layout changes). Audits:
  both tools read `tools/web_ui_audit_routes.py` (which adds patient,
  cohort, crossdb and dictionary to the popover audit and dictionary to the
  fit audit); `tests/webserver/test_web_ui_audits.py` fails when a screen on
  `window.SCREENS` is neither audited nor excluded with a reason (`entry`
  redirects, `states` is the design reference), and runs both tools live
  under the new `requires_web_server` marker (declared in `pytest.ini`,
  skipped and counted by `tests/conftest.py` without
  `EASYICU_WEB_AUDIT_BASE`, like the corpus / node / docker gates).
- Open (owner decisions): typed authorizations bounce to a model turn,
  first-turn database question repeats the question's own facts, planner
  latency (≥ 11 serial provider calls), compile failure on the lactate
  levels, `_DB_LABELS` demo keys. `tools/lint_progress.py`: OK.
- final result: passed
