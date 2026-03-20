"""Version consistency tests for release metadata and runtime surfaces."""

from pathlib import Path
import re

from app_version import APP_VERSION
from bridge import __version__ as bridge_version
from bridge.moltis_bridge import app
from core import __version__ as core_version


def test_runtime_versions_match_pyproject():
    pyproject_text = Path("pyproject.toml").read_text()
    match = re.search(r'^version = "([^"]+)"$', pyproject_text, re.MULTILINE)
    assert match is not None
    package_version = match.group(1)

    assert APP_VERSION == "1.4.0"
    assert package_version == APP_VERSION
    assert core_version == APP_VERSION
    assert bridge_version == APP_VERSION
    assert app.version == APP_VERSION
