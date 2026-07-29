# Changelog

## 2.3.1 - 2026-07-29

- Fixed the photo grid worker requeuing an entire failed batch, including
  targets whose frames had already been verified and merged during polling —
  it now requeues only targets that still lack a verified frame.
- Fixed a `rejected` batch-start response caused by a transient "busy"
  condition (the previous batch's task still unwinding, or FarmBot's own busy
  flag not yet clear) being treated as a hard failure. It is now retried with
  backoff (up to 6 attempts, ~5s growing to ~30s); any other rejection reason
  still fails immediately.
- Increased the photo-grid batch size from 12 to 25 coordinates per batch,
  now that the integration isolates per-target failures within a batch —
  cutting a 77-cell grid from 7 batches to 4 and reducing exposure to the
  busy-race window at each batch boundary.
- The worker now runs up to 3 verification passes (previously 2) but stops
  early if a pass verifies zero new frames, since a repeat pass would only
  move the bot pointlessly.
- Added per-batch and end-of-run diagnostic logging (verified counts per
  batch/pass, and the coordinates of any cells left unverified) and extended
  the terminal failure message to list example missing coordinates.

## 2.3.0 - 2026-07-28

- Replaced the **Automatic decision threshold** analysis card with a
  calibration-aware **Photo grid** card containing separate **Start photo
  grid** and **View most recent grid** actions.
- Whole-bed grid coordinates are now calculated from the saved FarmBot camera
  scale, reference resolution, rotation and optical offsets together with the
  bot's live X/Y axis limits. Captures follow a serpentine path with calibrated
  overlap.
- Reliable capture uses the integration's acknowledged movement and processed
  image checks in bounded batches. The app independently matches returned
  X/Y/Z coordinates within 25 mm, persists every verified frame, retries only
  missing/mismatched locations, and never reports completion with a hole.
- Added a birds-eye canvas mosaic of the most recent verified grid, aligned in
  garden coordinates and overlaid with current FarmBot plants and weed points.
- The existing photo-grid repair controls remain available for older or
  externally created grids.
- Added root `AGENTS.md` instructions that require Codex to bump every app
  version location and changelog for future non-documentation app changes.
- Requires companion integration 2.2.0, which preserves the existing light
  state, illuminates verified grid captures, and restores the light afterward.

## 2.2.0 - 2026-07-28

- Weed leaf and stem fragments separated by small segmentation gaps are now
  grouped into one detection. Its centre and radius enclose the complete
  grouped weed rather than marking each leaf with a separate small circle.
- Crop protection now excludes the complete vegetation cluster when part of a
  leaf crosses a known crop's protection boundary, preventing exposed crop
  edges from being reported as weeds. As before, weeds overlapping a crop may
  be conservatively excluded.
- Added an **Unknown** button to the weed review dialog. Detections clipped by
  the edge of the frame, or only a few pixels across, often cannot honestly be
  called either way, but until now the only way to clear one from the
  recommendations was to accept or reject it — which also fed a guess into the
  visual verifier's training data. Unknown discards the detection instead: it
  is neither accepted nor rejected, no training label is recorded, and the
  position is suppressed so the same ambiguous candidate is not offered again.
- Reviewing a weed no longer closes the dialog. Choosing a label now advances
  to the next weed in the same image, then to the next image, then back to the
  previous image — so a queue can be worked through without clicking **View**
  again for every detection. The dialog closes only on Close, Escape, or a
  click outside it, and shows an empty state once nothing is left to review.
  Focus stays on the button that was pressed, so the same key can be used
  repeatedly on a run of similar detections.
- Added a **Close-up** view alongside *Without overlay* and *With overlay*,
  showing the weed magnified on the original photo with no marker ring or
  analysis overlay drawn over it, plus a zoom slider (2×–12×). The chosen view
  persists as the review advances from one weed to the next.

## 2.1.3 - 2026-07-28

- Fixed consolidated recommendations showing the same unrecognisable picture
  for every plant. The composite stitched *every* analysed photo, including the
  ones where the plant's protection circle was entirely outside the frame —
  those views hold no pixels of the plant and only stretched the canvas across
  the whole photo grid, so each plant ended up with a near-identical
  garden-wide mosaic in which it was an invisible speck. Composites now use
  only the photos that actually saw the plant and are cropped to its
  neighbourhood, so each recommendation shows that plant filling the frame with
  its current and recommended radii drawn on top.
- The consolidation note no longer counts photos that never contained the
  plant, so "Consolidated from 5 images" now reads, for example, "Consolidated
  from 2 of 5 images (3 did not show the plant)".

## 2.1.2 - 2026-07-28

- Fixed "Save tag" on the verifier's labeled training images returning
  `{"detail":"Not Found"}`. The new tag was written to the database, but the
  route at `/weed-model/samples/{detection_id}` redirected one directory level
  up instead of two, landing on `/weed-model/weed-settings` — which does not
  exist. It now redirects back to the weed settings page and shows the
  confirmation notice.

## 2.1.1 - 2026-07-27

- Fixed bulk clearing of weeds to properly handle multiple selections and update
  state correctly.

## 2.1.0 - 2026-07-27

- Fixed photo-grid repair endlessly re-photographing the same cells. A repair
  photo is taken hours after the grid run it belongs to, so it never joined
  that run's one-hour time cluster and the cell stayed "missing" forever. The
  app now remembers which photos it captured for which run and credits them to
  the right cells, so every successful repair raises "grid cells found" by one
  and lowers the missing count by one immediately.
- A gantry-obscured photo is now deleted from FarmBot once a usable photo of
  the same cell has replaced it (requires companion integration 2.1.0; older
  integrations simply leave the old photo in place).
- A cell that still has no usable photo after two fresh captures is now
  reported and skipped instead of being retried indefinitely.
- The "grid cells found" count now counts filled cells rather than images, so
  a cell photographed more than once is no longer counted twice.

## 2.0.5 - 2026-07-27

- Photo-grid repair now runs automatically a configurable number of minutes
  (5 by default) after the latest photo grid finishes, instead of at a fixed
  time of day. Enabled by default; the delay and the on/off toggle are set
  from the "Repair photo grid" card.

## 2.0.4 - 2026-07-27

- Now requires companion integration 2.0.2 and its
  `position_verified_photo_grid_repair` capability. This prevents repair from
  running against the earlier movement payload that FarmBot accepted as a
  no-op.
- Repair movement is confirmed from live FarmBot status before the camera is
  triggered; failures distinguish the requested cell from the last observed
  position.
- Camera capture now allows six failed attempts per cell. If all six fail, the
  session records that cell and continues repairing the remaining cells
  instead of repeatedly blocking on the same location.

## 2.0.3 - 2026-07-27

- Replaced twelve-image repair batches with a background, one-cell-at-a-time
  queue. Each next cell is submitted only after the companion integration has
  verified a newly processed image at the previous target coordinates.
- Repairs now remain queued while FarmBot is busy and automatically continue
  after ordinary tasks, Home Assistant interruptions, or retryable camera
  failures.
- Added a gantry-classifier debug viewer showing every positive photo with its
  X/Y/Z coordinates and whether it is an interior repair target or an ignored
  perimeter positive.
- Reduced gantry false positives by requiring a brighter, more continuous
  low-saturation rail with stronger parallel edges. Positives on any outer
  row or column are treated as garden-bed edging and excluded from repair.
- Now requires companion integration 2.0.1 and its
  `verified_photo_grid_repair` capability so an older unverified repair
  implementation cannot silently reproduce stale-position photos.

## 2.0.2 - 2026-07-27

- Fixed photo-grid repair always failing with a bare HTTP 400 on any grid with
  more than 12 missing/gantry cells. The companion integration's
  `start_vision_grid_repair` service accepts at most twelve targets per call
  (a limit the app never enforced); requests over that limit were rejected by
  the integration's schema validation before ever reaching the handler, which
  Home Assistant reports as an empty, detail-free 400 response. Repair now
  sends the first 12 cells (missing cells before gantry-obstructed ones) and
  reports how many remain for a follow-up run.

## 2.0.1 - 2026-07-27

- Fixed a 2.0.0 build that shipped with unresolved merge-conflict markers
  still committed in several source files, which made the app fail to start.
- Fixed the photo-grid repair capability check falsely reporting "integration
  does not provide photo-grid repair" for a correctly loaded V2.0.0
  integration; a 400 response from an already-capability-checked repair
  request now surfaces the integration's actual rejection reason instead of
  guessing it is unsupported.
- Added a **Recheck grid** action so the photo-grid status can be refreshed
  on demand instead of only on the next scheduled inspection or repair
  attempt.

## 2.0.0 - 2026-07-27

- Added repair detection for missing photo-grid cells and gantry images, a
  persisted automatic repair time, and a manual Repair now action.
- Gantry content is treated as expected on the minimum and maximum Y rows of
  the garden bed; only gantry content at an interior Y coordinate is marked
  for replacement.
- Fixed the generic HTTP 400 failure by requiring an advertised
  `photo_grid_repair` capability before calling Home Assistant. The app now
  reports that integration V2.0.0 must be installed and Home Assistant
  restarted when the loaded integration cannot repair grids.
- Photo-grid retakes use the V2 companion service, which validates axis bounds,
  waits after each move, takes only replacement photos, and restores the
  FarmBot's starting position.

## 1.8.0 - 2026-07-26

- Added calibrated multi-image canopy fusion. Plant ownership is segmented in
  each original image, aligned on a plant-centred metric canvas, and measured
  from the fused mask when partial views cross image boundaries.
- Added app-managed fusion controls for activation, view and pixel evidence,
  radial statistics, angular coverage, disagreement tolerance, diagnostics,
  and automatic-action reliability requirements.
- Added a locally trained weed visual verifier with in-app hard-negative
  labelling, manual or automatic retraining, validation metrics, shadow mode,
  enforcement thresholds, and automation gates.
- Added configurable colour/shape filtering, complete known-crop protection,
  temporal weed confirmation, candidate-crop storage, and conservative
  automatic weed creation, radius maintenance, and disappearance handling.
- Fusion provenance and diagnostics are persisted with each measurement; an
  unreliable partial-view fusion stays reviewable and cannot trigger an
  automatic plant-radius change while its guardrail is enabled.

## 1.7.0 - 2026-07-26

- Plants whose protection area falls entirely outside the analysed image are
  no longer silently skipped. They now get a low-confidence, uncertain
  measurement so they still appear for manual review; no automatic change is
  ever applied to them.

## 1.6.0 - 2026-07-26

- Plant-radius review now opens one stitched composite containing every source
  image that identified the plant, aligned with the saved calibration's scale,
  rotation, and coordinate-origin settings.
- The review composite marks the original radius in cyan, the new radius in
  red, and the plant centre with a white dot.
- Reviewers can switch between the original stitched photos and the same
  composite with the plant ownership mask overlaid. Raw masks and per-frame
  diagnostic images are no longer included in the plant-radius viewer.

## 1.5.1 - 2026-07-26

- Fixed plant-protection-radius zone checks: a boundary now only requires the
  plant's centre to stay inside it, matching weeds and plant centres. A
  protection radius may extend past a boundary's edge; it still must not
  overlap a forbidding exclusion zone, since those mark real hazards.

## 1.5.0 - 2026-07-26

- Rejecting a weed recommendation now permanently suppresses that position
  instead of just changing its detection status, so a future analysis pass
  doesn't recreate the same rejected weed. Any other pending detections
  within the same tolerance are also marked rejected.
- Weed proximity suppression is now scoped per config entry (bot) and also
  treats previously rejected positions as occupied.
- Removal approval now refreshes the plant's current radius from the
  integration immediately before applying, instead of using the possibly
  stale radius captured earlier, so an older vision measurement no longer
  makes an otherwise valid, already-approved removal fail as stale.

## 1.4.0 - 2026-07-26

- Replaced direct soil-point capture targets with calculated clear-soil sites
  that avoid current FarmBot plants and weeds, the latest detected plant
  canopies, Vision weeds, and excluded garden zones.
- Only points last updated more than 14 days ago are eligible, and each clear
  replacement must be less than 200 mm from its assigned point.
- Guided calibration uses the same plant, weed, canopy, zone, and motion
  clearance checks and recomputes the selected site immediately before capture.
- Review now shows old and proposed X/Y coordinates. Approval relocates the
  existing soil GenericPointer and updates its measured Z; nothing is applied
  automatically.
- Requires companion FarmBot integration **1.8.0** for stale-timestamp and
  relocation validation.

## 1.3.1 - 2026-07-26

- Added hover tooltips to the Soil height guided calibration form explaining
  what Capture Z and Baseline (mm) mean, since neither was self-explanatory
  from the label alone.

## 1.3.0 - 2026-07-26

- Added a **Soil height** tab that inventories recognized FarmBot soil points,
  orders selected points by nearest neighbour, and supports selected/all runs,
  retry, stop-after-current, diagnostics, and individual or bulk review.
- Added required per-bot guided virtual-stereo calibration and a three-view
  StereoSGBM pipeline with correspondence rectification, vegetation and
  consistency masks, RANSAC soil-plane fitting, cross-pair validation, and
  fail-closed quality gates.
- Added durable soil calibration, measurement, decision and job audit records.
  In-progress soil jobs are marked interrupted on restart and never resume
  movement or apply a result automatically.
- Requires companion FarmBot integration **1.7.0** for acknowledged safe-motion
  captures and human-approved, optimistic-concurrency-protected point updates.

## 1.2.0 - 2026-07-26

- Consolidated repeated views of each plant into one confidence-, visibility-,
  and recency-weighted recommendation, using a robust median to prevent one
  anomalous image from producing a large radius increase.
- Added plant coordinates, concise review text, and garden-scale composite
  image stitching with bold original and planned radius circles.
- Prevented disconnected vegetation and hairline-connected weeds from
  inflating crop measurements while retaining bounded historical leaf evidence.
- Added known-weed matching, increasing-only radius tracking, disappearance
  tracking, and separate automatic radius/removal confidence controls.
- Replaced the queue timeframe dropdown with explicit from/to date-time inputs.

## 1.1.7 - 2026-07-25

- Re-cut the release: the `V1.1.6` GitHub release/tag was created against a
  commit that predated the 1.1.6 version bump (same failure mode as 1.1.5),
  so its build published the app image tagged `1.1.5` instead of `1.1.6`.
  Since Home Assistant Supervisor matches the image tag to `config.yaml`'s
  `version` field exactly, it could not find a `1.1.6` image and the update
  failed with `manifest unknown`. This release is cut from the current
  `main`, so its published image is correctly tagged `1.1.7` to match.
- Added a release-workflow check that fails the publish job if the release
  tag doesn't match the version baked into `config.yaml` at that commit,
  instead of silently publishing a mismatched image.

## 1.1.6 - 2026-07-25

- Added a **Boundaries & zones** tab. Boundaries enclose where things may be
  placed and exclusion zones mark areas to keep clear; each zone separately
  permits or forbids weed placement, plant-centre moves, and protection-radius
  growth. Rectangles, circles, and polygons are supported, zones can be
  disabled without deleting them, and a garden map draws them alongside the
  bot's plants and weeds. Zones persist in `/data/zones.json` and gate both
  automatic writes and manual approvals; with no zones configured nothing is
  restricted.

## 1.1.5 - 2026-07-25

- Re-cut the release: the `V1.1.4` GitHub release/tag was created against a
  commit that predated the 1.1.4 version bump, so its build published the
  app image tagged `0.5.0`/`V1.1.4` instead of `1.1.4`. Since Home Assistant
  Supervisor matches the image tag to `config.yaml`'s `version` field exactly,
  it could not find a `1.1.4` image and kept reporting `0.5.0` as both
  installed and latest. This release is cut from the current `main`, so its
  published image is correctly tagged `1.1.5` to match.

## 1.1.4 - 2026-07-25

- Made all analysis outcomes manually approvable/rejectable; certainty now gates automation only.
- Fixed new spread-curve HTTP 400s, creates curves for plants without one, and verifies radius
  and curve writes against FarmBot before reporting success.
- Analyses partially visible plants at reduced confidence instead of skipping their centres.
- Added known-plant soft ownership before weed classification and centre-move suggestions for
  offset but otherwise healthy canopies.
- Replaced the queue card with an Analysis workflow and a timeframe-based image picker.
- Added persistent, opt-in weed detection settings with recommendation and automatic-creation modes.
- Improved pale-leaf segmentation, overlapping-canopy ownership, empty-centre removal detection,
  and reviewability of large human-approved growth changes.
- Added removal review alternatives to keep a plant or move its recorded centre.

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
- Added conservative radius-change safety: small decreases can be automatic while larger decreases remain lower-confidence review items, alongside optimistic-concurrency handling, manual calibration, monotonic curve learning, SQLite migrations, retention, scheduling, and ingress UI.
- Added synthetic safety tests and BuildKit multi-architecture CI/release workflows.
