# Legacy Streamlit WebApp

The Streamlit WebApp under `src/easyicu/webapp/` is deprecated and frozen.
The maintained user-facing WebApp path is the native FastAPI implementation
under `src/easyicu/webserver/`.

## Current policy

- Streamlit is no longer an active visual fallback.
- Streamlit CSS visual parity is no longer maintained.
- Only `tokens.css` and `shell_overrides.css` stay in the default Streamlit CSS
  path as a minimal base layer.
- Stage24B removed the legacy route split CSS files after Stage24A made them
  inactive by default.
- Stage26B moved the default `easyicu-webapp` command and one-click launchers to
  the native FastAPI server. The Streamlit command is now
  `easyicu-webapp-legacy` and requires the `easyicu[webapp-legacy]` extra.

## Recovery

The old route split CSS is no longer present in the working tree. If archive
forensics require it, recover the files from Git history at `63bba1c` or an
earlier commit before Stage24B, then run with
`EASYICU_ENABLE_LEGACY_STREAMLIT_CSS=1`.

The environment switch is a historical rollback/debug aid only. Do not add new
selector-level fixes to the legacy split CSS. FastAPI native remains the main
WebApp path.

## Do not use for new work

Do not add new Streamlit UI polish, route-specific CSS, visual parity tests, or
browser guard work for this legacy app unless explicitly requested for archive
forensics. New WebApp functionality and user-facing QA belong in the FastAPI
native implementation.
