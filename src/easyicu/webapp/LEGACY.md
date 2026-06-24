# Legacy Streamlit WebApp

The Streamlit WebApp under `src/easyicu/webapp/` is deprecated and frozen.
The maintained user-facing WebApp path is the native FastAPI implementation
under `src/easyicu/webserver/`.

## Current policy

- Streamlit is no longer an active visual fallback.
- Streamlit CSS visual parity is no longer maintained.
- The legacy route split CSS files remain in the repository during Stage24A,
  but they are not loaded by default.
- Only `tokens.css` and `shell_overrides.css` stay in the default Streamlit CSS
  path as a minimal base layer.
- Stage24B may remove the inactive split CSS files after default-runtime
  validation.

## Temporary opt-in

To temporarily restore the old Streamlit route CSS while it still exists:

```bash
EASYICU_ENABLE_LEGACY_STREAMLIT_CSS=1 easyicu-webapp
```

This opt-in is a rollback/debug aid only. Do not add new selector-level fixes to
the legacy split CSS. FastAPI native remains the main WebApp path.

## Do not use for new work

Do not add new Streamlit UI polish, route-specific CSS, visual parity tests, or
browser guard work for this legacy app unless explicitly requested for archive
forensics. New WebApp functionality and user-facing QA belong in the FastAPI
native implementation.
