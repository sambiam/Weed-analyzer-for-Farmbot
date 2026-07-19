#!/usr/bin/env python3
"""Verify the Home Assistant app configuration schema is internally consistent.

Checks that config.yaml, the translations file and the Settings model agree on
the analysis_resolution option, and that the default is the documented
960x720. Run in CI so a schema drift fails the build.
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "farmbot_vision" / "src"))

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

    if config.get("version") != "0.3.0":
        fail("config.yaml version must be 0.3.0")

    print("config schema check passed: analysis_resolution consistent, default 960x720")


if __name__ == "__main__":
    main()
