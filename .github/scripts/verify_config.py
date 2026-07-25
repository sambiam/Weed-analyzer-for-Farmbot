#!/usr/bin/env python3
"""Verify the Home Assistant app configuration schema is internally consistent.

Checks that config.yaml, the translations file and the Settings model agree on
the analysis_resolution option, and that the default is the documented
960x720. Also checks that config.yaml, pyproject.toml and the package
`__version__` agree, so a release that bumps one and forgets the others fails
the build instead of shipping a version Home Assistant won't recognise as an
update. Run in CI so drift of either kind fails the build.
"""

from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "farmbot_vision" / "src"))

from farmbot_vision import __version__  # noqa: E402
from farmbot_vision.resolution import AnalysisResolution  # noqa: E402
from farmbot_vision.settings import Settings  # noqa: E402

ALLOWED = {item.value for item in AnalysisResolution}


def fail(message: str) -> None:
    print(f"config schema check FAILED: {message}")
    raise SystemExit(1)


def main() -> None:
    config = yaml.safe_load((ROOT / "farmbot_vision" / "config.yaml").read_text())

    options = config.get("options", {})
    schema = config.get("schema", {})

    if options.get("analysis_resolution") != "960x720":
        fail("options.analysis_resolution default must be 960x720")

    schema_line = schema.get("analysis_resolution", "")
    presets = set(schema_line.removeprefix("list(").removesuffix(")").split("|"))
    if presets != ALLOWED:
        fail(f"schema presets {presets} must equal {ALLOWED}")

    # The Settings model must accept every allowed preset and reject others.
    for value in ALLOWED:
        Settings(analysis_resolution=value)
    try:
        Settings(analysis_resolution="2592x1944")
    except ValueError:
        pass
    else:
        fail("Settings accepted a disallowed resolution")

    if Settings().analysis_resolution.value != "960x720":
        fail("default Settings resolution must be 960x720")

    translations = yaml.safe_load(
        (ROOT / "farmbot_vision" / "translations" / "en.yaml").read_text()
    )
    if "analysis_resolution" not in translations.get("configuration", {}):
        fail("translations missing analysis_resolution description")

    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    pyproject_version = pyproject.get("project", {}).get("version")
    if pyproject_version != __version__:
        fail(f"pyproject.toml version {pyproject_version!r} must match farmbot_vision.__version__ {__version__!r}")

    if config.get("version") != __version__:
        fail(f"config.yaml version {config.get('version')!r} must match farmbot_vision.__version__ {__version__!r}")

    readme = (ROOT / "farmbot_vision" / "README.md").read_text()
    if not re.search(rf"\b{re.escape(__version__)}\b", readme):
        fail(f"farmbot_vision/README.md must mention current version {__version__!r}")

    changelog = (ROOT / "farmbot_vision" / "CHANGELOG.md").read_text()
    if f"## {__version__} " not in changelog and not changelog.startswith(f"# Changelog\n\n## {__version__}"):
        fail(f"farmbot_vision/CHANGELOG.md must have a heading for {__version__!r}")

    print(
        f"config schema check passed: analysis_resolution consistent, default 960x720, "
        f"version {__version__} consistent across config.yaml, pyproject.toml, README and CHANGELOG"
    )


if __name__ == "__main__":
    main()
