# Development and validation

Use Python 3.12 and Docker BuildKit.

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e '.[dev]'
ruff check .
ruff format --check .
pytest
docker buildx build --platform linux/amd64 --build-arg BUILD_ARCH=amd64 farmbot_vision
docker buildx build --platform linux/arm64 --build-arg BUILD_ARCH=aarch64 farmbot_vision
```

Dependencies are version-pinned and selected because PyPI supplies aarch64 CPython 3.12 wheels for NumPy, OpenCV headless, Pydantic, and the pure-Python/asynchronous packages. The Debian slim base avoids compiling these on the Pi. Release tags such as `v0.1.0` trigger a signed-provenance/SBOM multi-platform GHCR build. The retired `home-assistant/builder` is not used.

Set `FARMV_DATA_DIR` to a temporary directory for local execution. Set a dummy `SUPERVISOR_TOKEN` only when mocking the Home Assistant endpoints. Never put a real token in a repository, test fixture, command transcript, or log.

## Releasing a new version

`farmbot_vision/src/farmbot_vision/__version__` is the single source of truth
for the app version. Every release that changes app behaviour must bump it,
because Home Assistant only offers an update when `config.yaml`'s `version`
changes, and a stale version silently hides real releases from users. To cut
a release:

1. Bump `__version__` in [`farmbot_vision/src/farmbot_vision/__init__.py`](../farmbot_vision/src/farmbot_vision/__init__.py).
2. Copy that same value into `version` in [`farmbot_vision/config.yaml`](../farmbot_vision/config.yaml) and into `project.version` in [`pyproject.toml`](../pyproject.toml).
3. Add a dated entry at the top of [`farmbot_vision/CHANGELOG.md`](../farmbot_vision/CHANGELOG.md) describing what changed, folding in anything still under an `## Unreleased` heading.
4. Update the version mentioned in [`farmbot_vision/README.md`](../farmbot_vision/README.md) (the add-on's in-app documentation panel) and, for a user-visible change, the `Version:` line and **Status** section of the top-level [`README.md`](../README.md).
5. Run `python .github/scripts/verify_config.py`, which fails the build if `config.yaml`, `pyproject.toml`, `farmbot_vision/README.md` and `farmbot_vision/CHANGELOG.md` don't all agree with `__version__` — treat it as the release checklist, not just a CI gate.
6. Tag the release `vX.Y.Z` to trigger the multi-platform GHCR build.

Follow semantic versioning: patch for fixes, minor for backward-compatible
features, major for breaking changes to the companion-integration contract or
stored data.

The published release image is always tagged from `config.yaml`'s `version`
(read and applied automatically by [`.github/workflows/release.yml`](../.github/workflows/release.yml)),
so a correct `config.yaml` is what actually matters for Supervisor update
detection. The `BUILD_VERSION` defaults in [`farmbot_vision/Dockerfile`](../farmbot_vision/Dockerfile)
and [`.github/workflows/ci.yml`](../.github/workflows/ci.yml) only label
locally-built and CI smoke-test images — update them to match for consistency,
but they don't affect what Supervisor sees.
