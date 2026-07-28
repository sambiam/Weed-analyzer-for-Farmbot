# Codex instructions

## FarmBot Vision app versioning is mandatory

Whenever Codex changes app code, behavior, dependencies, or user-visible UI
under `farmbot_vision`, it must bump the app version before declaring the task
complete. Documentation-only and formatting-only changes are the only
exceptions.

- Use a patch bump for fixes/internal changes, a minor bump for features or
  visible behavior, and a major bump for breaking changes requiring user
  action.
- Keep these version strings identical:
  `farmbot_vision/config.yaml`, `farmbot_vision/src/farmbot_vision/__init__.py`,
  `farmbot_vision/Dockerfile`, `farmbot_vision/README.md`, root `README.md`,
  `.github/workflows/ci.yml`, and root `pyproject.toml`.
- Add a dated entry to `farmbot_vision/CHANGELOG.md`.
- Run `rg` for the old version to catch missed release metadata.
- In the final response, state the exact app version and release tag
  (`VX.Y.Z`). Do not create the tag or release unless the user explicitly asks.

If the companion integration is changed, also follow its own `CLAUDE.md`
versioning instructions in that repository.
