# Changelog

## 0.5.0 - 2026-07-20

Calibration is rebuilt around FarmBot's own coordinate model.

- **FarmBot calibration is now the only method.** The legacy "measure two
  points" method is removed. Enter FarmBot's `Pixel coordinate scale`, the
  resolution it was measured at, camera rotation, origin location and any
  residual offset.
- **Composite photo-row view.** The calibration page stitches one photo row
  (images sharing an X coordinate) into a single FarmBot-style image in
  coordinate space, with plant **and weed** centres overlaid and labelled
  (plant name · crop; weeds in red). A row selector switches between rows, and
  only one row is loaded at a time to bound memory. The composite updates live
  as the calibration values are edited, replacing the old flick-through-and-
  click-overlay-per-image workflow.
- **Alignment fixed to match FarmBot.** `garden_to_pixel` now inverts FarmBot's
  own pixel↔coordinate model: the metric offset from the image centre is scaled
  and reflected by the origin, then rotated **in pixel space about the image
  centre** (mirroring FarmBot physically rotating each photo to align it). With
  no camera rotation the map is unchanged, so existing calibrations keep their
  behaviour; rotated cameras now overlay correctly instead of drifting.
- **Calibration persists across restarts.** Saved FarmBot values are written to
  a JSON store in the persistent `/data` volume and restore the active
  calibration automatically at startup — no re-entry after a restart — while
  staying fully editable in the app.
- **Weed points added to the inventory contract.** `get_vision_inventory` may
  now include a `weeds` list; it defaults to empty so older companion
  integrations still validate.

## 0.4.0 - 2026-07-20

- New-photo integration events now target and analyse only the completed image.
- Automatic image jobs queue behind an active job instead of being discarded.
- The only loaded FarmBot config entry is selected automatically when the app
  option is blank.
- Heartbeats are sent immediately at startup and at most five minutes apart,
  including for installations retaining the former 15-minute option, so Home
  Assistant vision entities remain available within the integration's
  ten-minute timeout.

## 0.3.0 - 2026-07-19

Manual calibration can now mirror FarmBot's own camera calibration.

- Added a **FarmBot calibration values** method to the calibration page: enter
  FarmBot's `Pixel coordinate scale` (mm/pixel) and the resolution it was
  measured at, plus camera rotation and origin location. The scale is inverted
  to pixels-per-millimetre and rescaled to the analysis resolution through the
  same path as reference calibration, so a native scale is never applied
  directly to a resized frame and the numbers can be copied verbatim.
- Added **Origin location in image** (`top_left`/`top_right`/`bottom_left`/
  `bottom_right`) to the calibration model and `garden_to_pixel`. This encodes
  the garden↔pixel axis reflection FarmBot expresses, which a pure rotation
  cannot; `top_left` is the identity and default, so every existing calibration
  is unchanged. Available to both calibration methods.
- Offset fields now carry guidance that FarmBot's camera offset is already
  folded into the image-centre coordinate and should stay 0 unless the overlay
  shows a residual shift.
- Non-destructive migration 3 adds the `origin_location` column; migrated rows
  read back as `top_left`.

## 0.2.1 - 2026-07-18

Runtime fixes for Home Assistant Ingress and FarmBot Vision events.

- Removed the explicit root `ingress_entry` so Home Assistant uses its default
  Ingress entry path.
- Added ASGI middleware that rewrites duplicate leading slashes internally,
  including `//` and `///settings`, without redirects or query-string changes.
- Kept dashboard, calibration, image, artifact, and recommendation links
  relative so they remain inside a dynamic Ingress session.
- Accepted the companion event's optional `device_id` and validated every
  requested plant ID as a positive integer while continuing to reject unknown
  fields.
- Skipped malformed JSON and invalid individual events in place so they do not
  close the active WebSocket subscription; connection, authentication, and
  subscription failures retain bounded reconnect handling.
- Added sanitized event observability and job-lock rejection logging.

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
