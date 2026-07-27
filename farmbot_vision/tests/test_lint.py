"""Guardrail so lint/format regressions are caught by ``pytest``, not just CI.

Running the full suite locally before pushing is common practice; folding the
same checks CI runs (``ruff format --check`` / ``ruff check``) into the suite
means a push-blocking regression shows up wherever tests are run, not only
after a round-trip through GitHub Actions.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _run_ruff(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "ruff", *args, "."],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_ruff_format_check() -> None:
    result = _run_ruff("format", "--check")
    assert result.returncode == 0, result.stdout + result.stderr


def test_ruff_check() -> None:
    result = _run_ruff("check")
    assert result.returncode == 0, result.stdout + result.stderr
