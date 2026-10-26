"""Personal display/advisory settings in the desktop or hosted user's home.

Hosted deployments give each user an isolated EASYICU_HOME, as for sessions
and project bindings. No browser-supplied user ID or filesystem coordinate is
accepted. These values never enter Planner or execution contracts.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from easyicu.webserver import state_paths
from easyicu.webserver.hosted import require_instance_path

MAX_INSTRUCTIONS_CHARS = 4000
MAX_PREFERENCES_BYTES = 32_768
# The current publication owner implements one Nature-compatible style.
# Do not advertise other journals until that owner supplies their renderers.
JOURNAL_FIGURE_STYLES = ("nature",)


class PersonalPreferences(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    figure_export_format: Literal["png", "svg", "pdf"] = "svg"
    journal_figure_style: Literal["nature"] = "nature"
    interface_language: Literal["en", "zh"] = "en"
    instructions: str = Field(default="", max_length=MAX_INSTRUCTIONS_CHARS)

    @field_validator("instructions")
    @classmethod
    def printable_instructions(cls, value: str) -> str:
        if any(ord(char) < 32 and char not in "\n\r\t" for char in value):
            raise ValueError("instructions contain control characters")
        # Reject unpaired Unicode surrogates before persistence/transport.
        value.encode("utf-8")
        return value


class PreferencesError(ValueError):
    def __init__(self, code: str, status_code: int = 422):
        super().__init__(code)
        self.code = code
        self.status_code = status_code


class PersonalPreferencesStore:
    def __init__(self, path: Path | None = None):
        self.path = Path(path) if path is not None else state_paths.state_root() / "personal_preferences.json"

    def _check_path(self) -> None:
        require_instance_path(self.path)
        if any(path.is_symlink() for path in (self.path, *self.path.parents)):
            raise PreferencesError("preferences_store_invalid", 409)

    def read(self) -> PersonalPreferences:
        self._check_path()
        try:
            with self.path.open("rb") as handle:
                raw = handle.read(MAX_PREFERENCES_BYTES + 1)
            if len(raw) > MAX_PREFERENCES_BYTES:
                raise ValueError("oversized preferences")
            return PersonalPreferences.model_validate(json.loads(raw))
        except FileNotFoundError:
            return PersonalPreferences()
        except (OSError, ValueError, UnicodeError) as exc:
            raise PreferencesError("preferences_store_invalid", 409) from exc

    def update(self, patch: dict) -> PersonalPreferences:
        from easyicu.webserver.pi_copilot.locking import exclusive_file_lock

        if not isinstance(patch, dict):
            raise PreferencesError("preferences_payload_invalid")
        if set(patch) - PersonalPreferences.model_fields.keys():
            raise PreferencesError("preferences_field_unknown")
        self._check_path()
        with exclusive_file_lock(self.path.with_suffix(".lock"), code="preferences_store_busy"):
            try:
                value = PersonalPreferences.model_validate({**self.read().model_dump(), **patch})
            except ValidationError as exc:
                field = exc.errors()[0]["loc"][0]
                raise PreferencesError(f"preferences_{field}_invalid") from exc
            # Apply the same sensitive-input boundary as ordinary Copilot text.
            from easyicu.webserver.pi_copilot.projections import reject_sensitive_message

            reject_sensitive_message(value.instructions)
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=self.path.parent, delete=False) as handle:
                    temporary = Path(handle.name)
                    handle.write(value.model_dump_json() + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                temporary.replace(self.path)
            except OSError as exc:
                raise PreferencesError("preferences_store_unavailable", 503) from exc
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
            return value

    def projection(self) -> dict:
        return {
            "schema_version": "easyicu.personal-preferences/1",
            "preferences": self.read().model_dump(),
            "applies_to": "new_conversations",
            "options": {
                "figure_export_format": ["png", "svg", "pdf"],
                "journal_figure_style": list(JOURNAL_FIGURE_STYLES),
                "interface_language": ["en", "zh"],
                "max_instructions_chars": MAX_INSTRUCTIONS_CHARS,
            },
        }
