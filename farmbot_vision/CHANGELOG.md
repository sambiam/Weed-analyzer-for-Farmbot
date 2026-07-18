# Changelog

## 0.1.3 - 2026-07-18

- Run the app container as root so it can read Home Assistant's root-only `/data/options.json`.

## 0.1.2 - 2026-07-18

- Fixed AppArmor access for the complete Python shared-library tree and its native dependencies on Home Assistant OS and Supervised installations.

## 0.1.1 - 2026-07-18

- Fixed a startup crash (`libpython3.12.so.1.0: cannot open shared object file`) caused by the AppArmor profile not covering `/usr/local/lib`, where the official Python image installs its shared library; also added the `m` (mmap-exec) permission required to load compiled extensions such as NumPy and OpenCV.

## 0.1.0 - 2026-07-17

- Initial Home Assistant app for aarch64 and amd64.
- Added strict companion-integration contracts over the Supervisor Core REST/WebSocket proxy.
- Added sequential classical canopy analysis, maximum-leaf protection, overlap uncertainty, and temporal evidence hooks.
- Added no-shrink radius safety, optimistic-concurrency handling, manual calibration, monotonic curve learning, SQLite migrations, retention, scheduling, and ingress UI.
- Added synthetic safety tests and BuildKit multi-architecture CI/release workflows.
