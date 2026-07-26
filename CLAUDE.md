# FarmBot Vision — versioning

Home Assistant Supervisor matches the add-on's Docker image tag to
`farmbot_vision/config.yaml`'s `version` field **exactly**. If the version
isn't bumped, Supervisor has nothing to compare against and never shows the
new build as an update — this has already happened twice (see the `1.1.5`
and `1.4.0`-era CHANGELOG entries) and burned a release each time.

**Whenever a code, behavior, or dependency change is made to `farmbot_vision`
(not for docs-only or formatting-only changes), bump the version before
considering the change done:**

1. Pick the new version:
   - **Patch** (`1.1.5` → `1.1.6`): bug fixes, internal refactors, small
     behavior tweaks that don't change what a user configures or sees.
   - **Minor** (`1.1.7` → `1.2.0`): new features, new settings, or any
     user-visible behavior change (e.g. moving config between the app and
     the companion integration, as happened in `1.1.6`).
   - **Major** (`x.0.0`): breaking changes that require the user to take
     action (e.g. a config migration, a removed feature with no
     replacement).
2. Update the version string in **all** of these locations (all must match):
   - `farmbot_vision/config.yaml` → `version:`
   - `farmbot_vision/src/farmbot_vision/__init__.py` → `__version__`
   - `farmbot_vision/Dockerfile` → `ARG BUILD_VERSION`
   - `farmbot_vision/README.md` → the `# FarmBot Vision X.Y.Z` heading and
     the `Version: **X.Y.Z**` badge line
   - `.github/workflows/ci.yml` → `BUILD_VERSION=` build arg
   - `pyproject.toml` (repo root) → `version =`
   - Root `README.md` → the `Version: **X.Y.Z**` badge line, if present
3. Add a dated entry to `farmbot_vision/CHANGELOG.md`. If there's already an
   `## Unreleased` section, rename it to `## X.Y.Z - YYYY-MM-DD` rather than
   adding a duplicate heading.
4. Tell the user the exact version and tag name they need to create as a
   GitHub release (tags in this repo follow `VX.Y.Z`, e.g. `V1.1.6`) — do
   not push, tag, or create the release yourself; that's a user decision
   (see the repo-wide git safety rules).
