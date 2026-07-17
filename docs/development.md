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
