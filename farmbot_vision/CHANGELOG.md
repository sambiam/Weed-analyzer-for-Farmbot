# Changelog

## 0.2.0 - 2026-07-18

Configurable analysis resolution and the revised high-resolution image
contract (contract **farmbot-vision-v2**).

- Added the `analysis_resolution` app option (`640x480`, `960x720`, `1280x960`)
  with a new default of **960x720**. Existing installations migrate to the
  default automatically. Changing it requires an app restart.
- Added a typed `Resolution` model (width, height, pixel count, label,
  relative workload) and rejected any non-allowlisted dimensions.
- Raised the image request/response ceiling to 1280x960 and extended the
  `VisionImage` contract with source/oriented/processed dimensions, resize
  scales, optional `source_sha256` and optional `processed_calibration`.
- Validated returned images fully: checksum over the returned JPEG, decoded
  dimensions, JPEG format, resize-scale consistency, aspect ratio, no
  upscaling, and size limits. Base64 image data is never logged.
- Calibration now always corresponds to the exact processed pixels, selected
  in preference order: processed calibration → reference calibration scaled to
  the resolution → compatible manual calibration → none. A native 2592x1944
  scale is never applied to a resized frame.
- Manual calibration is now tied to config entry, image, processed resolution,
  pixel points, separation and version, with an interactive point-and-overlay
  calibration page (no external tools, no frontend build toolchain).
- Without valid calibration the app produces pixel-only diagnostics, marks the
  result uncalibrated, and refuses every write and approval.
- Made morphology kernels and area thresholds resolution-aware so the physical
  plant radius stays stable across all three presets; historical masks from a
  different resolution are safely rescaled or rejected.
- Preserved single-job / single-image / single-thread processing and the
  CPU/memory gates; health now reports the selected resolution, pixel count and
  contract version. Dashboard shows full resolution/calibration provenance.
- Added database migration 2 (additive columns only; existing data preserved).
- Declared minimum companion integration version **1.2.0**.

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
