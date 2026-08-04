# Changelog

## 3.28.0 - 2026-08-04

- Added independent, opt-in tool verification controls for rotary-tool loading
  and unloading; unreliable tool sensing no longer blocks either operation by
  default.
- Changed adaptive weeding to carry the current X/Y through every safe-height
  retract and travel directly along the nearest-neighbour route between weeds.
- Made fresh soil-height checks opt-in, added a configurable maximum age, and
  added an option to continue with an older nearby height after measurement
  failure.
- Added one-sided fallback cuts that approach from open ground and stop at the
  weed centre when a complete through-cut would violate plant clearance.

## 3.27.1 - 2026-08-04

- Fixed selected soil-height application by removing legacy v2 results from the
  ordinary pending-results table and safely skipping results that were already
  resolved between a live refresh and batch submission.
- Made live soil-height updates preserve selected coordinates, active checkbox
  elements, unchanged table rows, and the exact scroll position. Polling now
  updates only changed status regions and no longer reloads the whole page when
  a legacy repair scan completes.

## 3.27.0 - 2026-08-04

- Added a one-time repair workflow for valid or previously applied
  `soil-stereo-v2` height measurements. It re-downloads the retained calibration
  and measurement images, reconstructs their recorded X/Y/Z positions, and
  stages v3 results without changing FarmBot.
- Added an automatically shown comparison modal listing each detected legacy
  height change, its signed difference, confidence, and whether the old result
  had already been applied. Corrections require explicit selection and human
  confirmation; users can instead keep the existing values.
- Persist each legacy repair outcome so unchanged, unavailable, applied,
  rejected, or conflicted rows cannot be processed again. Once every eligible
  v2 row is resolved, the workflow and its controls disappear permanently.
- Blocked legacy v2 results from the ordinary soil-result apply routes so the
  confirmation modal is the only application path for a staged correction.

## 3.26.1 - 2026-08-04

- Fixed a systematic soil-height scale error caused by normalizing disparity
  with the requested camera baseline. Stereo pairs now use the X/Y positions
  recorded on each exposure, so a real 16.15 mm move is no longer treated as
  the nominal 15 mm move (the supplied 505 mm example otherwise reads about
  469 mm while all three pairs misleadingly agree).
- Guided calibration now derives each known depth from the recorded Z position
  instead of the requested 0/25/50 mm movement. Calibration and measurement
  also fail closed if the three exposures drift by more than 1 mm in Z.
- Bumped the soil stereo algorithm to v3. Existing soil calibrations require a
  one-time recalibration so v2 baseline normalization cannot be reused.

## 3.26.0 - 2026-08-04

- Added daily automated runs for all available soil-height measurement sites,
  configured by local run time from the Soil height tab.
- Added optional automatic acceptance using both a minimum confidence and a
  maximum change from the point's original height.
- Added a minutes-or-hours delay for one automatic retry of failed points after
  each manual or scheduled run.
- Added Select all and Select all failed controls to the soil-point table.
- Made the stereo-pair disagreement failure limit configurable instead of
  fixing it at 8 mm.

## 3.25.2 - 2026-08-04

- Stopped the Soil height page from reloading the whole page every few
  seconds while a calibration or measurement job runs. It now patches just
  the job status, review table and pending-results table in place over
  AJAX, so scroll position, open dropdowns and in-progress form input are no
  longer reset out from under the user.

## 3.25.1 - 2026-08-03

- Fixed every batched soil status response failing strict validation because
  the new `batch_id` field was missing from the app's response model.
- Poll asynchronous batch restoration instead of holding one Supervisor HTTP
  request open during the return move.
- Stop a measurement run after a fatal companion contract or transport error,
  clean up its batch, and avoid rapidly queuing all remaining points behind an
  unobserved capture.
- Require FarmBot integration 2.9.1 for asynchronous batch restoration.

## 3.25.0 - 2026-08-03

- Group all captures in a multi-point soil measurement into one companion
  integration batch. The FarmBot now travels directly from one measurement to
  the next and restores the run's original X/Y/Z only once at the end.
- Retry transient `FarmBot is busy` capture responses for up to ten minutes
  instead of immediately recording a failed soil measurement. Status and logs
  now explain that the job is waiting for the current operation to finish.
- Require FarmBot integration 2.9.0 for serialized soil batches and explicit
  end-of-run position restoration.

## 3.24.1 - 2026-08-03

- Made soil-height stereo matching robust to the supplied textured-mulch
  portrait and landscape captures by guiding SGBM from feature flow, using the
  adjacent views to guide the wider outer pair, and measuring coverage only
  over clear overlap where matching is geometrically possible.
- Accept consistent dense soil planes with at least 3% usable coverage. At the
  supported resolutions this still requires thousands of left/right-checked
  pixels, while avoiding the old false rejection of reliable 4-6% captures.
- Include the soil algorithm version in calibration compatibility, so existing
  v1 calibrations fail safely with a recalibration prompt instead of mixing old
  calibration math with the revised measurement algorithm.
- Fixed concurrent image-cache trimming raising `FileNotFoundError` and
  returning HTTP 500 when another request had already evicted the same JPEG.

## 3.24.0 - 2026-08-03

- Soil-calibration quality-gate failures now report which gate failed on
  which pair, at which Z level (coverage, plane support, L/R error, or plane
  MAD, with the measured value and threshold), instead of a bare "failed
  quality gates" message. The same diagnostics are logged for measurement
  gate failures.
- The Soil height tab shows the failing calibration job's full diagnostics
  and, when the failure was a numeric quality gate rather than a broken
  capture, offers an "Accept calibration anyway" override that recomputes
  and saves the calibration from the already-captured images without another
  50 mm capture movement. Overridden calibrations are flagged as such
  wherever the active calibration is shown.

## 3.23.2 - 2026-08-03

- Removed the hard-coded `1280x960` soil-image assertion. Guided calibration
  now accepts and records the camera's actual processed and source dimensions,
  including the observed `1280x720` format.
- Subsequent soil measurements request the saved soil-specific processed size
  and require their processed and source geometry to match that calibration,
  independently of the general Vision analysis resolution.
- The Soil height tab now labels the active calibration's dedicated soil image
  format, and geometry failures report both expected and received dimensions.

## 3.23.1 - 2026-08-03

- Soil calibration and measurement captures now surface per-frame movement,
  retry, acceptance, and failure messages from the companion integration.
- Added final processed-image checks for blur, washed-out exposure, and a
  maximum 5 mm coordinate error before stereo analysis begins.
- Allow sufficient job time for five independently verified capture attempts
  at every stereo frame instead of timing out a valid retry sequence.
- Require FarmBot integration 2.8.1 for the verified per-frame soil capture
  contract.

## 3.23.0 - 2026-08-03

- Restrict the Weeding tab and execution pipeline to inventory records
  explicitly identified as FarmBot Weed points; plants, soil points,
  ToolSlots, and generic points cannot become weeding candidates.
- Add radius-range filtering, a select-all-shown checkbox, clear-selection
  control, and live shown/selected counts for efficient batch selection.

## 3.22.1 - 2026-08-02

- Removed the FarmBot soil-point anchor requirement from custom-coordinate
  soil calibration while retaining axis and motion safety validation.
- Added a persistent clear-soil margin control to the Soil height tab so users
  can relax site selection when conservative margins find no usable soil.
- Fixed slow soil planning repeatedly showing an unavailable/loading state by
  retaining the last successful plan and refreshing it in the background.
- Fixed anchor-free custom calibration jobs incorrectly trying to resolve a
  missing soil point before capture.

## 3.22.0 - 2026-08-02

- Added a Weeding tab that previews and runs risk-scored straight rotary-tool
  cuts for selected FarmBot weeds.
- Use only soil heights recorded within 500 mm and 30 days, inverse-distance
  interpolate multiple samples, and automatically measure a plant- and
  weed-clear soil patch when recent evidence is unavailable. New measurements
  are rejected when they differ unreasonably from nearby history.
- Choose each cut against every plant protection circle, size it beyond the
  weed extent, keep it inside configured axis bounds, and order weeds to reduce
  non-cutting travel.
- Added overload recovery that reverses at half speed, raises after repeated or
  pre-cut contact, continues past individual failures, and always issues a
  final motor-off command.
- Added an illuminated post-run scan. Verification analyses only the attempted
  weed points, cannot discover new weeds, removes points confirmed absent, and
  retains present, inconclusive, or uncaptured weeds.
- Added an optional rotary-tool mount/dismount stage. The Weeding tab uses a
  detected FarmBot ToolSlot when available and provides editable fallback slot
  coordinates (defaulting to X 4.2, Y 576.8, Z -386) and pullout direction.
- Added mounted-tool, plant-height-aware routing. It is enabled by default with
  a configurable 300 mm threshold, routes around protected plant canopies, and
  conservatively protects plants whose height is unknown.

## 3.21.0 - 2026-08-02

- Replaced missing-detection weed removal confidence with explicit per-known-weed
  visual results: present, absent, or inconclusive. Automatic removal now
  requires consecutive fully visible absent observations from an enforcing
  learned-verifier pipeline; uncertain and crop-obscured evidence cannot remove
  a weed.
- Require repeated learned-verifier acceptance before automatically widening a
  known weed, while retaining the strict extent and rolling 24-hour growth
  limits. Heuristic confidence and verifier shadow mode can no longer authorise
  a radius write.
- Prevent reprocessing the same image from advancing known-weed presence or
  absence streaks.

- Added custom-coordinate soil calibration with a FarmBot soil-point safety
  anchor as an alternative to a calculated clear-soil site.
- Added custom-coordinate soil measurement for manually verified locations when
  no calculated clear-soil site is available.
- Removed the redundant removal-confidence and radius-adjustment-confidence
  settings, and added a consecutive confirmed-image control for weed radius
  widening.

## 3.20.1 - 2026-08-02

- Reapply minimum radius-change thresholds after multi-image consolidation so below-threshold recommendations do not reappear in the review queue.

## 3.20.0 - 2026-08-02

- Added live plant and weed review-queue reconciliation so reviewed or
  superseded recommendations no longer remain stale in the dashboard.
- Automatically refresh the review page when new recommendations arrive and
  handle last-second already-reviewed actions without showing a stale-item
  error.

## 3.19.0 - 2026-08-02

- Optimistically advance plant-radius reviews while approvals complete in the background.
- Add configurable minimum increase/reduction thresholds and preserve curve maxima after reductions.
- Rename the Weed settings tab to Settings and add evidence View buttons for automatic changes.

## 3.18.0 - 2026-08-01

- Removed the remaining strong-green pre-verifier veto, so pale seedlings and
  multi-leaf rosettes reach the learned verifier instead of disappearing as a
  colour/shape rejection.
- Let shadow-mode verifier results rescue review candidates, including
  high-confidence weeds provisionally owned by a crop, without changing crop
  geometry or granting authority for automatic creation or radius growth.
- Replaced permanent rejected/dismissed coordinate dead zones with same-image
  deduplication, and limited stale created-record suppression to the 24-hour
  FarmBot inventory-sync window. Later photos can surface newly emerged
  vegetation or correct an earlier classification.
- Added explicit candidate rescue statistics and field-like regressions using
  the reported 0.79 review threshold and pale-foliage settings.
- Made an enforcing trained verifier mandatory for every automatic weed
  creation. Heuristic confidence is now review-only, and the settings page
  clearly separates fallback review, verifier decisions and automatic-creation
  controls.

## 3.17.0 - 2026-08-01

- Split weed discovery from weed extent measurement: pale and shaded leaves
  now reach the verifier through a permissive mask, while stricter adjustable
  saturation/excess-green controls keep soil out of measured area and radius.
- Group nearby leaves into a span-bounded whole-weed proposal, infer the weed
  centre from the complete leaf layout, and replace farthest-pixel radius with
  a supported configurable radial percentile.
- Added separate verifier rejection and acceptance thresholds for weeds and
  clearly labelled plant-growth boundary thresholds. Low-confidence plant
  radius results are retained for diagnostics but omitted from review.
- Capped automatic known-weed radius increases against both millimetre and
  percentage limits over a rolling 24-hour baseline, preventing repeated
  same-day views from compounding an oversized mask.
- Increased the maximum-weed-area slider range from 10,000 to 40,000 mm² while
  retaining the 100,000 mm² typed-input limit.

## 3.16.0 - 2026-08-01

- Added a focused measurements review modal with accessible approve/reject
  actions, previous/next navigation, automatic advance after review, and a
  compact plant summary.
- Removed measurement-table review controls and improved action contrast for
  mobile and desktop use.

## 3.15.1 - 2026-08-01

- Fixed photo-grid startup when FarmBot returns an explicit `null` radius for
  a weed point. These points now use the existing zero-radius fallback.

## 3.15.0 - 2026-08-01

- Made weed discovery recall-first before verification: the candidate mask now
  honours the configured hue, saturation, and excess-green controls and keeps
  narrow foliage that the crop mask's 3 mm opening previously erased.
- Oversized blobs are now sent through verification and review with their area
  retained as evidence instead of being silently discarded. The default soft
  maximum rises from 2,500 to 10,000 mm²; exceeding it still blocks automatic
  FarmBot weed creation.
- Unowned vegetation inside crop-protection zones can now reach the enabled
  verifier for review and training, while crop ownership remains protected and
  the persisted overlap flag prevents automatic creation near a known crop.
- Bounded nearby-leaf grouping now prevents a chain of adjacent weeds from
  collapsing into one bed-spanning candidate that the verifier cannot classify
  usefully.
- Added a lightweight learned second opinion for plant-radius growth. Only new
  vegetation beyond the previous canopy edge is scored; crop-confirmed evidence
  may extend the geometric radius, verifier-confirmed weeds and non-crop
  material are removed, and uncertain evidence holds the old boundary with a
  confidence cap that prevents automatic writes.
- Reused the existing locally trained weed/crop/category model instead of
  adding a full-frame neural network. Boundary checks run only on a few new
  connected regions, honour verifier shadow mode, and fall back to the
  72-sector geometric estimator when the model or a trained crop head is not
  available.
- Mapped known FarmBot weed centres and radii into each processed image before
  final plant measurement. Their complete outward radial sectors are removed
  from new plant ownership, while established canopy inside the previous edge
  remains protected.
- Verifier-confirmed boundary weeds now bypass the old circular crop-exclusion
  stage and enter the normal weed review/confirmation workflow instead of being
  hidden by the plant mask.
- Added boundary-verifier settings, per-image statistics, measurement
  diagnostics, and regression coverage for crop acceptance, uncertainty,
  known-weed exclusion, and weed-workflow handoff.

## 3.14.0 - 2026-07-31

- Added an experimental **Draw shape** tab that traces a shape on the XY plane
  by sending raw firmware G-code to FarmBot. Choose a circle or a regular
  polygon (3-24 sides), give it a centre, a circumradius and a rotation, and
  the app plans the path, previews it over the bed, and writes the G-code
  program out in an editable box. What is sent is exactly the text in that box,
  hand edits included.
- **This is the only feature in the app that moves FarmBot outside FarmBot OS's
  motion planning.** The program reaches the Farmduino through FarmBot OS's Lua
  `gcode()` function, which validates nothing, so the companion integration
  (2.6.0+, capability `experimental_raw_gcode`) re-checks the whole program
  against live axis bounds and firmware config and refuses it as a unit if any
  point leaves the bed. A "Validate only" button runs exactly that check
  without moving anything.
- Circles are drawn as many short `G00` chords rather than as an arc: the
  FarmBot firmware does not implement `G01`, and `G00` is explicitly not
  guaranteed to travel in a straight line. The segment count is chosen from a
  chord-tolerance (sagitta) budget, defaulting to 0.5 mm, so "how round" is
  stated in millimetres instead of guessed at. Expect approximate results --
  this measures how well a bot tracks a path, it is not a plotter.
- The path always approaches at the travel Z, descends to the draw Z only once
  it is over the start point, and retracts at the end, so a mistyped centre
  drags through air rather than through the bed. Nothing is actuated; only the
  gantry moves.

## 3.13.0 - 2026-07-31

- Replaced farthest-owned-pixel plant sizing with a Pi-friendly 72-sector
  supported boundary estimator. It uses the existing radius as a temporal
  prior, rejects broad implausible expansion from soil, moss and attached
  weeds, preserves narrow genuine leaves, and confidence-caps clipped results
  so they cannot be applied automatically.
- Trimmed rejected outer evidence from each plant ownership mask before weed
  candidate verification and calibrated multi-image fusion.
- Kept radius measurements relative to FarmBot's stored centre so a slightly
  offset centre still encloses the far leaf edge, while clean masks can propose
  reductions for previously oversized radii.
- Plant-radius review images now draw the current and proposed circles only for
  the plant being reviewed; neighbouring plants remain visible in the photo
  context without competing labels or circles.

## 3.12.7 - 2026-07-31

- Prevented tiny plant footprints from producing negative confidence values
  that abort an entire multi-image analysis job.
- Isolated vision-engine failures to the affected image and continued the
  remaining batch, with failed-image details in the job result and logs.
- Show source photos and per-frame overlays for partial runs before a plant
  composite has been built.

## 3.12.6 - 2026-07-31

- Fixed large measurement writes failing with `SQLITE_CANTOPEN` when SQLite
  selected an inaccessible auxiliary temporary-file directory. SQLite now
  uses in-memory temporary storage and the container explicitly pins its temp
  directory to the AppArmor-writable `/tmp`.
- Log SQLite error codes and names when a measurement write needs recovery.

## 3.12.5 - 2026-07-31

- Reconnect and retry plant measurement persistence after a transient SQLite
  database-open failure, with storage diagnostics in the warning log.
- Persist plant evidence before per-image weed writes so a later storage error
  cannot leave a partial weed result while hiding all plant recommendations.

## 3.12.4 - 2026-07-31

- Start the normal analysis pipeline as soon as quality-cleared grid photos
  are available, instead of waiting for slow optional quality repairs to finish.
  Added explicit job and handoff summaries so missing recommendations identify
  the stage that stopped them.

## 3.12.3 - 2026-07-31

- Fixed completed photo grids being left out of the normal analysis pipeline
  when their delayed per-image events were held or lost. Verified grid photos
  now have an explicit completion handoff, and late duplicate events are
  ignored after that handoff succeeds.

## 3.12.2 - 2026-07-31

- Fixed the most-recent photo grid losing all plant and weed circles whenever
  Home Assistant's live inventory was temporarily unavailable. The grid now
  falls back to the map points snapshotted when capture began, including each
  weed's stored radius and centre dot.
- The missing live weed inventory was traced to the companion integration;
  FarmBot integration 2.5.1 now supplies those Weed map points to both the
  calibration and most-recent grid APIs.

## 3.12.1 - 2026-07-31

- Removed the "Health JSON" link from the app's navigation bar. The
  `/api/health` and `/health` endpoints still work as before; this only
  removes the visible tab.

## 3.12.0 - 2026-07-31

- The photo grid now assesses each photo **while the grid is still running**,
  in the time the bot spends driving to the next coordinate, rather than only
  after the whole route finishes. The blur / washed-out / close-leaf check is
  the same single inspection the post-grid pass always did — it has simply
  moved earlier, and its result is cached and reused, so no image is fetched or
  decoded twice. The OpenCV work runs on a worker thread so it can never delay
  a capture, and any frame the scout cannot reach mid-run falls back to the
  post-grid pass unchanged.
- While it is there, the scout reports two further signals at no extra cost:
  **possible weeds** (vegetation components the quality check had already found,
  mapped into garden coordinates and filtered against the plants FarmBot knows
  about) and **fully framed plants** (pure calibration geometry — which cells
  can measure a whole plant on their own). Confirmed weed detection and plant
  radius measurement are untouched and still run in the post-grid analysis
  pipeline, because a plant spanning several cells needs the composite.
- Grid status now distinguishes a cell with **no photo** (solid red) from a cell
  whose photo has a **quality problem** (green interior, red border). A flagged
  cell turns blue inside while it is being retried but keeps its red border
  until a repair actually completes. The card also summarises how many photos
  were checked during capture, how many need a retake, and how many possible
  weeds and fully framed plants were seen; each cell's tooltip gives its own
  detail.

## 3.11.0 - 2026-07-31

- Consolidated the Analysis page overview into full-width Info and Photo
  analysis cards, with clearer FarmBot, resolution, timing, grid and queue
  status details. Moved review-queue clear actions beside their table headers.

## 3.10.1 - 2026-07-31

- Fixed settings-page labels rendering one row out of alignment with their
  inputs (e.g. calibration page). `<label>` elements had no explicit
  `display`, so the browser's default inline flow let each label's caption
  and following `<input>` bleed onto the same visual line as the previous
  field, shifting every caption up by one row. Labels are now `display:block`
  site-wide, which fixes this across every settings form, not just
  calibration.

## 3.10.0 - 2026-07-30

- Fixed weed candidate generation starving the learned verifier. Weeds plainly
  visible in a photo were being discarded before the verifier ever scored them,
  and because a discarded candidate is never stored, reviewed or labelled, the
  loss was invisible. Four independent causes, each measured on a reference
  scene:
  - Crop protection grouped vegetation islands transitively. One weed near a
    crop protected every weed reachable from it through a chain of
    within-12 mm hops — about 70% of all vegetation in a weedy bed. Protection
    is now bounded to 30 mm from the crop itself, which still covers a leaf
    straddling the exclusion boundary.
  - The heuristic confidence terms were summed unnormalised, so a genuine weed
    could not score above roughly 0.70 while the default review threshold was
    also 0.70. Terms are now normalised against what real foliage scores.
  - The colour, shape and size thresholds could veto a candidate outright.
    They are now clamped to absolute recall floors, so no saved value —
    including one inherited from an earlier build whose defaults real foliage
    could not meet — can stop a candidate from being scored.
  - Crop padding blanked a 120 mm radius around a 60 mm plant. While a trained
    verifier is scoring, padding beyond the canopy is capped at 12 mm; the
    verifier receives each candidate's distance to the nearest crop and judges
    it better than a fixed circle can.
- Applied the candidate recall boost whenever a trained verifier is scoring,
  including shadow mode, which is the stage meant to be gathering examples to
  label.
- Retuned the candidate defaults to values real photographs can meet: minimum
  area 75 → 20 mm², strong-green fraction 0.45 → 0.10, solidity 0.25 → 0.08,
  circularity 0.03 → 0.01, maximum aspect ratio 7 → 12, review confidence
  0.70 → 0.45.
- Logged per-image candidate counts and the reason each was dropped, so a
  starved verifier is visible in the add-on log.
- Rebuilt the weed settings page. Every setting now has a paired slider and
  number box, and a `?` tooltip saying what it does, what high and low values
  mean, and how to calibrate it. Added visual aids: a hue band marking the
  accepted colour range, a colour sampler that centres the range on a colour
  picked from your own photo, area limits shown as the physical width of a
  blob, and shape diagrams for solidity and circularity. Grouped the settings
  under plain-language headings.

## 3.9.0 - 2026-07-30

- Replaced the dashboard's approval and rollback history with a normalized
  change log showing applied plant and weed changes, locations, radius before
  and after, decision method, and confidence.

## 3.8.3 - 2026-07-30

- Reconciled photo-grid batches with FarmBot's durable processed-image
  inventory when Home Assistant loses or delays a repair-status response, so
  already-captured rows are not re-requested or marked missing.
- Treated Home Assistant startup responses as temporary during a grid capture
  and retained the existing batch while the service recovers.
- Suppressed routine successful `/api/photo-grid/status` access-log entries
  while retaining access logs for errors and other routes.

## 3.8.2 - 2026-07-30

- Recovered pale, washed-out leaf sections when they are anchored to detected
  vegetation, while keeping isolated pale material out of the vegetation mask.
- Allowed elongated crop-leaf components that reach a known plant centre while
  continuing to reject isolated irrigation-line and weed-shaped components.
- Added regression coverage for a pale outer leaf and an isolated weed.

## 3.8.1 - 2026-07-30

- Fixed add-on startup under AppArmor by allowing the `nice` process-priority
  wrapper used by the application launch script.

## 3.8.0 - 2026-07-30

- The weed verifier's automatic retraining now retrains after every N new
  labels (user-configurable, default 1) instead of after each single new
  label once the minimum dataset exists. The settings page shows how many
  new labels have accumulated since the last trained run.

## 3.7.1 - 2026-07-30

- Fixed plant evidence consolidation so the dashboard limit applies after
  grouping all photo-grid images by plant. Useful earlier tiles are no longer
  discarded when a run contains more than 100 raw plant/image measurements.
- Removed the unsafe raw-frame fallback from plant review buttons, which could
  show the same unrelated grid photo for every no-evidence recommendation.
- Added correctly centred, labelled garden-grid placeholders for plants whose
  local neighbourhood has not been photographed yet.

## 3.7.0 - 2026-07-30

- Added global-blur detection to the canonical photo-grid quality pass using
  strong-edge density, Laplacian detail, image contrast, and adjacent grid
  photos as a local sharpness baseline.
- Blurry originals are excluded from analysis, deleted, and retaken once at
  the same coordinate. A retake that remains blurry or fails another quality
  rule is also discarded without starting a repair loop.
- Added regression coverage for severe blur, neighbour-relative moderate blur,
  successful clear retakes, and still-blurry retakes.

- Coalesced bursts of new-photo events into one analysis job, eliminating the
  repeated whole-grid evidence, fusion, and composite work that grew
  quadratically during a photo-grid upload.
- Kept Home Assistant responsive during analysis by lowering the add-on's CPU
  scheduling priority, limiting OpenCV to one worker, yielding between images
  and while system load is high, and moving fusion/composite work off the web
  event loop. Full-resolution analysis and output algorithms are unchanged.
- Cached processed UI images on disk and in the browser, deduplicating
  concurrent requests and avoiding repeated Home Assistant image RPCs.
- Capped the soil tab's cold live lookup at 750 ms, cached its safe-site plan
  for repeat navigation, and added database indexes for pending-analysis views.
- Replaced multi-kilobyte per-plant INFO records with compact summaries while
  retaining the complete evidence diagnostics at DEBUG level.

## 3.6.1 - 2026-07-30

- Removed the separate legacy photo-grid repair controls and scheduler. Missing
  frames, washed-out retakes, leaf-free alternate views, and large-plant
  follow-ups now remain part of the app's canonical photo-grid sequence.
- Added a live Grid status card whose squares come directly from the persisted
  target plan, with verified/current/retrying state, completion percentage,
  and current bot instruction. A 6 by 9 plan now displays exactly 54 cells.
- Replaced the approximate FarmBot display crop with the exact maximum-area
  axis-aligned rectangle contained inside each rotated camera image. Mosaic
  cells can no longer request pixels beyond a rotated source edge.
- Photo-grid planning now spaces captures from that blank-free footprint, so
  increasing camera rotation automatically adds the rows and columns needed
  to keep calibration, full-grid, and plant-measurement mosaics gap-free.

## 3.6.0 - 2026-07-30

- The trained visual verifier is now the weed score outright rather than being
  blended with the shape heuristic. Blending compressed the verifier's
  calibrated range into roughly 0.25-0.95 and lifted every rejected candidate,
  so the displayed confidence disagreed with the gate that was actually
  deciding. `Verifier weight in final score` is removed; the verifier threshold
  is the single gate whenever enforcement is on.
- The heuristic score is now what it always measured — plant-ness, not
  weed-ness — and is used only as an ordering before the verifier is trained.
  Every one of its terms rises for moss, fallen leaves and crop foliage exactly
  as it does for weeds, which is why reweighting it could never separate them.
- Added a `Candidate recall boost` setting that relaxes the colour/shape gates
  while a trained verifier is enforcing, so borderline weeds reach the model
  instead of being dropped by rules that cannot classify them. Behaviour is
  unchanged when the verifier is off or in shadow mode.
- Three spatial features feed the verifier: distance to the nearest known
  plant, surrounding vegetation density, and proximity to the frame edge.
  Samples labelled by earlier versions lack them, so training falls back to the
  original sixteen features until every sample carries the full set.
- Verifier scores are corrected back to the observed class balance. The fit is
  class-weighted, so its 0.5 was a balanced-prior answer and far too permissive
  for a real candidate stream.
- Training now holds out whole images rather than individual candidates.
  Two crops from one photo are usually the same weed under the same light, and
  splitting between them reported a precision the model did not have.
- Training sweeps the threshold curve and suggests the highest-recall operating
  point whose precision is confidently at or above 95%, judged on the Wilson
  lower bound so a small lucky validation fold cannot justify a permissive
  threshold. One click applies the suggestion.
- The weed review dialog shows the verifier's best guess at what the object
  actually is ("moss 71%"), using per-category heads trained from the same
  labels. It is descriptive only and never changes the accept/reject decision.
- The weed settings page lists the unlabelled candidates closest to the
  decision boundary, which are worth far more per label than another obvious
  weed, and can label them without changing their review status.
- Added export and import of the training bundle (labels plus features,
  optionally the crop images). An exported bundle can be shipped with the
  add-on as `bundled_weed_model.json`, which a fresh install uses until it has
  trained a model of its own.

## 3.5.1 - 2026-07-30

- Fixed per-photo grid events so plant evidence, confidence, fusion, and review
  composites are rebuilt from every pending measurement in the current
  verified grid instead of whichever individual upload ran last.
- Plant review now shows the target plant's surrounding grid cells even when a
  complete single photo supplies the measurement or no usable evidence was
  selected. Zero-evidence rows no longer fall back to a shared unrelated
  diagnostic image.

## 3.5.0 - 2026-07-30

- Calibration, latest full-grid, and plant-measurement composites now crop
  every photo to a deterministic rectangular grid cell. Cell boundaries meet
  at exact camera-centre midpoints and the outside cells terminate at the
  configured garden-bed border, eliminating overlap-shaped and rotated seams.
- Added subtle cell borders and a stronger garden/window outline to all grid
  views, matching the FarmBot map's legible tiled presentation.
- Plant review images now restrict base and quality-repair photos, masks, and
  priority selection to their assigned cell. Standard and diagnostic views
  retain identical tessellated geometry.

## 3.4.0 - 2026-07-30

- Added a persisted second-pass quality gate for newly captured photo grids,
  detecting washed-out frames and close leaves that obscure the camera.
- Washed-out originals are excluded from analysis, deleted through the
  companion integration, and given one same-coordinate retake. A still-washed
  retake is also discarded without triggering another repair.
- Leaf-obstructed cells keep their original background, capture four offset
  views in one repair attempt, rank them by unobscured plant content, and paint
  the selected view as the top layer in grid and plant composites.
- Delayed new-image analysis until the grid quality pass finishes and persisted
  discarded image IDs, preventing rejected or unselected frames from producing
  measurements even when remote deletion is unavailable.

## 3.3.0 - 2026-07-30

- Calibration now renders every loaded photo even when its reported dimensions
  differ from the selected Camera settings resolution. The actual resolutions
  remain visible as a warning so the user can diagnose and correct the setting
  without losing the calibration grid.
- Photo-grid capture planning now reproduces FarmBot's post-rotation rectangle
  crop instead of treating the larger rotated bounding box as usable coverage.
  New grids use closer coordinates where necessary and retain the configured
  overlap after FarmBot crops the displayed map photos.
- The calibration and analysis-page grid renderers now apply the same FarmBot
  crop geometry and use each downloaded photo's reported dimensions. Both
  views therefore agree with the capture planner and FarmBot map more closely.

## 3.2.0 - 2026-07-30

- Selected complete single-image plant evidence exclusively and separated
  candidate, useful, selected, and excluded images so irrelevant grid tiles no
  longer lower confidence.
- Added calibrated 72-sector outer-boundary coverage, bed-space mask radius
  measurement, partial/large-plant fusion, noise removal, and structured
  evidence diagnostics.
- Rebuilt plant review imagery from one shared standard/target-mask renderer,
  including tight calibrated crops, expanded tile windows, deterministic seams,
  current/proposed radii, centre crosses, crop labels, and neighbouring plants.
- Made the review modal responsive and geometry-stable between standard and
  diagnostic modes, removing the independent live-grid canvas pipeline.
- Added deduplicated centred follow-up photo scheduling through the existing
  FarmBot grid-repair movement service when a fit-sized plant lacks 50% grid
  coverage; plants too large for one frame remain composite-only.
- Added regression coverage for single/partial/large/no-evidence selection,
  3x4 expanded composites, calibrated transforms, mask isolation, targeted
  capture deduplication, and the mobile review layout.

## 3.1.0 - 2026-07-29

- Matched FarmBot Web App photo sizing by rebuilding each processed preview
  JPEG's physical footprint from its actual pre-resize dimensions instead of
  assuming every capture has the resolution typed into the form.
- Mixed-resolution photos in a selected date range are now hidden using the
  same dimension check as FarmBot, with the actual dimensions and hidden count
  reported below the grid.
- Split Farm Designer orientation into its real independent controls: Map
  origin reflects the selected axes, while Rotate map swaps X and Y. Existing
  saved orientations are migrated to retain their previous on-screen layout.
- The preview now prefers live FarmBot axis bounds over potentially stale
  photo-grid metadata and reports both the bounds and their source.
- Photos are painted oldest-to-newest like FarmBot so the newest neighbouring
  tile remains visible where captures overlap.

## 3.0.0 - 2026-07-29

- Corrected camera calibration to match the current FarmBot Web App source:
  copied camera rotation values now use FarmBot's negative display direction,
  and copied camera offsets place the optical centre at photo coordinate plus
  offset without requiring users to invert either value.
- Photo-grid planning now subtracts the FarmBot camera offset from desired
  optical centres when calculating gantry capture coordinates.
- Clarified that Pixel coordinate scale is copied verbatim in millimetres per
  capture pixel, while the FarmBot Camera settings resolution is used only to
  rescale analysis and preview copies.
- Added calibration-grid zoom in, zoom out, and reset controls. The full-bed
  canvas can be enlarged to 600% and scrolled to inspect individual seams and
  overlay centres.
- Existing installations that stored inverted workaround rotation or offset
  values must replace them with the values shown verbatim in FarmBot after
  upgrading.

## 2.6.0 - 2026-07-29

- The whole-bed photo grid is now captured as one continuous serpentine route.
  With companion integration 2.5.0 the entire 77-cell plan is sent in a single
  `start_vision_grid_repair` call, so the bot no longer drives back out to its
  staging position and cycles the lighting every twelve cells, and rows are no
  longer cut in half and resumed later.
- Where the loaded integration still requires chunking, batches are strictly
  consecutive slices of the same canonical route: the first cell of a batch is
  the cell that follows the last cell of the one before it, with no renumbering,
  no overlap and no regenerated coordinates. The app logs a warning naming the
  capability to update for.
- Uploaded frames are credited to a cell by the stable target index the
  integration returns, instead of by 25 mm coordinate proximity. A verified cell
  can no longer look unverified and get photographed a second time.
- Grid coordinates are rounded to micrometres so a cell's planned position is
  identical however often it is generated, stored or resent.

## 2.5.0 - 2026-07-29

- Replaced the calibration row composite with a date-filtered, full-bed photo
  grid that matches the analysis page's birds-eye mosaic and updates live as
  camera calibration values change.
- The calibration grid keeps only the newest image at each FarmBot coordinate
  in the selected range, preventing repeated captures from being painted over
  one another.
- Plant and weed circles remain overlaid across the complete bed. A persistent
  Map origin setting now rotates the photos, markers, and labels together to
  match the FarmBot web app's map orientation.
- Corrected calibration-preview offset handling so its inverse image transform
  matches the analysis pipeline's garden-coordinate transform.

## 2.4.1 - 2026-07-29

- **Fixed:** The photo grid captured almost nothing (2 of 77 coordinates in
  the observed run). `PHOTO_GRID_CHUNK_SIZE` had been raised to 25 in 2.3.1,
  but the companion integration's `start_vision_grid_repair` service schema
  caps `targets` at twelve. Home Assistant therefore rejected every full-size
  batch with HTTP 400 during schema validation — before the handler ran, so
  no coordinate in the batch was visited or photographed. Only the short
  final remainder was small enough to be accepted. The chunk size is now
  derived from `PHOTO_GRID_MAX_TARGETS_PER_CALL` (12), which documents the
  cap as a contract limit rather than a tuning knob, and a test asserts the
  relationship so it cannot regress silently.
- **Added:** A batch rejected outright is now halved and retried in smaller
  calls instead of stranding every coordinate in it. A per-call cap mismatch
  between the add-on and the companion integration — which are versioned and
  updated independently — now degrades into slower capture rather than a
  near-empty grid.

## 2.4.0 - 2026-07-29

- Added a **Fallen leaf** choice to the weed review modal for isolated leaves
  that have detached from a plant. Selecting it rejects the FarmBot weed
  recommendation, suppresses the position, and records a dedicated hard
  negative so the local verifier can learn not to classify similar fallen
  leaves as weeds.
- Fallen-leaf samples now appear as a separate count and editable tag on the
  verifier training page.

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
