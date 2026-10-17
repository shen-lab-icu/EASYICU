"""One disposable home even when pytest loads conftest under two names."""

import os
import tempfile

# This regular package module has one import identity. Keep the directory alive
# for the process; importing a conftest alias must not replace its environment.
PYTEST_EASYICU_HOME = tempfile.TemporaryDirectory(prefix="easyicu-pytest-state-")
os.environ["EASYICU_HOME"] = PYTEST_EASYICU_HOME.name
