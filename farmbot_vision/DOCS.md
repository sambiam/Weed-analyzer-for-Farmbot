# FarmBot Vision documentation

## Upgrading

Install the new app version from the Home Assistant App store, then restart
the app. Close the old FarmBot Vision browser tab and reopen the Web UI so
Home Assistant creates a fresh Ingress session. See
[`CHANGELOG.md`](CHANGELOG.md) for what changed in each release; the
companion integration only needs to change when a release raises the minimum
compatible companion integration version (currently **1.8.0**).

Since 0.2.1 the app removes the explicit root Ingress entry and normalizes
duplicate leading slashes at the ASGI boundary. Dashboard and calibration URLs
are relative to the current `X-Ingress-Path`, so they remain inside temporary
Ingress sessions. The app never logs the complete `X-Ingress-Path` value.

Manual `farmbot_vision_request` data is:

```json
{"config_entry_id":"string","device_id":"string","plant_ids":[],"mode":"recommend"}
```

`device_id` may be omitted, `plant_ids` must contain positive integers, and an
empty list means all eligible plants. Unknown fields remain rejected. A single
malformed JSON or invalid event is skipped in place; it does not reconnect the
WebSocket, and subsequent valid events continue to be processed.

For each newly processed FarmBot photo, integration 1.4.0 or newer emits a
targeted request containing `image_id` and omitting `mode`. The app then uses
its configured mode and analyses only that image. Requests wait behind a
running job rather than being dropped.

## Before enabling it

FarmBot Vision requires Home Assistant Core 2026.7 or newer and companion FarmBot integration 1.8.0 or newer. Start in **Observe** mode. Do not use early experimental output as the sole input to destructive weeding.

## Modes

- **Observe** stores measurements and diagnostic overlays without writes.
- **Recommend** proposes individual increases and exposes approve/reject controls. Nothing is written until approved.
- **Auto radius** writes only high-confidence increases that pass all configured limits, plus small high-confidence decreases within `maximum_automatic_radius_reduction_percent` (10% by default). Larger decreases remain reviewable with reduced confidence and are never automatic.
- **Auto curve** is a future advanced mode and is intentionally unavailable in 0.1.0. Curve proposals never modify or replace a user-created curve.

## Analysis resolution

The `analysis_resolution` option selects the processed image size: `640x480`,
`960x720` (default) or `1280x960`. Arbitrary dimensions are rejected. Higher
resolutions resolve finer canopy detail at a proportional CPU/memory cost —
relative pixel work is 1× / 2.25× / 4× (a native 2592 × 1944 frame would be
~16.4×, and is deliberately not a selectable mode). **960 × 720 is the
recommended default for a 4 GB Raspberry Pi 4.** Settings load once at startup,
so changing the resolution takes effect after the app is restarted.

## Calibration

Calibration always corresponds to the exact pixels analysed. It is selected in
preference order: (1) `processed_calibration` returned with the image;
(2) reference calibration scaled to the processed resolution using oriented
dimensions; (3) a compatible manual calibration; (4) none. A native-resolution
scale is never applied directly to a resized frame.

If no integration calibration is available, open **Calibration** and copy
FarmBot's own camera calibration (Photos → Camera calibration):

- **Pixel coordinate scale** (mm/pixel) and the **resolution it was measured
  at** (its native frame, e.g. 2592 × 1944). FarmBot's scale is stated for that
  native frame, so it is inverted to pixels-per-millimetre and rescaled to the
  analysis resolution exactly as reference calibration is — a native scale is
  never applied directly to a resized frame, so the numbers can be copied
  verbatim.
- **Camera rotation** and **Origin location in image**, plus any residual
  offset correction.

**Verify against a whole photo row.** Click *Load bot inventory*, then pick a
**photo row** — the images FarmBot captured at the same X coordinate. The app
stitches that row into a single composite in coordinate space (each photo
rotated and placed exactly as FarmBot's own map does), and overlays every known
plant centre (green, labelled with the plant name and crop) and every FarmBot
weed point (red). The composite updates live as you edit the values, so you can
confirm the circles sit on their plants and weeds across the whole row before
saving. Only one row is loaded at a time to keep memory bounded. Use the *Row X
tolerance* to control how close two X coordinates must be to count as one row.

**Origin location** encodes the garden↔pixel axis reflection (`top_left` is the
identity and default), letting a rotated or mirrored camera mount map correctly
— something rotation alone cannot express. FarmBot's camera offset is already
folded into the image-centre coordinate, so the offset fields default to 0 and
should only be used to correct a residual shift seen in the overlay.

**Persistence.** Saved calibration values are written to the app's persistent
`/data` volume and restored automatically after a restart — you never re-enter
them — and remain editable in the app at any time. The derived
processed-resolution calibration becomes the active one used by analysis.
Automatic writes and approvals are refused without valid calibration.

The transformation assumes image metadata `x,y` is the image's ground centre.
`garden_to_pixel` inverts FarmBot's own pixel↔coordinate model: the metric
offset from the image centre is scaled and reflected by the origin, then rotated
in pixel space about the image centre (mirroring FarmBot physically rotating
each photo to align it to the garden axes). Offsets are in millimetres. Every
observation retains the exact transform, resolution, resize scales, calibration
source and version used.

## Supplemental soil height

Open **Soil height** to measure the existing FarmBot soil-height
`GenericPointer` records. The app does not infer soil points from names: the
companion must recognize FarmBot's `measure-soil-height` or `at_soil_level`
metadata. Fewer than three usable points triggers a warning because FarmBot's
`soil_height(x, y)` interpolation needs at least three measurements.

Calibration is required once per bot/camera arrangement. Select clear textured
soil, enter the manually measured camera-to-soil distance at capture Z, and
confirm that moving 50 mm toward the soil is safe. The bot takes three lateral
views at each of three Z levels. Calibration is activated only when the
inverse-depth fit is monotonic and its maximum residual is at most 5 mm.
Recalibrate after any camera move, rotation, refocus, remount, resolution,
baseline, source-geometry, Z-direction, or declared camera-setting change.

Measurements capture three views 15 mm apart, or a validated one-sided triplet
near a Y-axis edge. The pipeline rectifies roll and vertical offset, computes
StereoSGBM disparities for both adjacent pairs and the outer pair, masks green
vegetation and inconsistent pixels, and fits the dominant soil plane with
RANSAC. A result remains diagnostic-only unless coverage, plane support,
left/right consistency, plane residual, cross-pair agreement, and propagated
uncertainty all pass their quality gates.

Use **Measure selected**, **Measure all**, or **Retry failed**. Points are
visited in nearest-neighbour order. **Stop after current point** allows the
companion's current atomic capture to finish and never sends an emergency stop.
All valid results require explicit individual or selected approval. Apply
re-fetches the same point and refuses stale coordinates; only its Z coordinate
is updated. Restarted jobs are recorded as interrupted and never resume bot
motion or apply results automatically.

## Image selection and analysis

### Reliable photo grid

The Analysis page's **Photo grid** card provides separate **Start photo grid**
and **View most recent grid** actions. Starting a grid reads the selected bot's
saved camera calibration and live motion limits. The app converts the native
pixel scale and reference dimensions to the camera's garden footprint,
includes camera rotation and X/Y optical offsets, and lays out an overlapping
serpentine path over the complete X/Y bed bounds.

Companion integration 2.2.0 or newer switches the standard lighting peripheral
on while capturing and restores its previous state afterward. It acknowledges
each safe movement, waits for live X/Y/Z position confirmation, takes the
photo, and waits for the processed image inventory. The app also checks each
returned frame against its requested X/Y/Z coordinate within 25 mm. Only
verified frames count. Missing or mismatched targets are retried within the
same grid sequence; a grid with any remaining hole is marked failed rather
than complete. After capture, that same sequence quality-checks every frame,
retakes washed-out or blurry images, tries alternate views for leaf-obstructed
images, and captures clear centred views of large plants when needed. Blur is
judged from whole-frame detail and strong edges, with adjacent grid cells used
as the local sharpness baseline when available.

The latest-grid view draws verified frames into one calibrated birds-eye
garden mosaic and overlays current FarmBot plants and weeds. The plan and
verified frame IDs are saved in `/data`, so completed progress survives an app
restart. The **Grid status** card renders one square per canonical target, so
its dimensions and count always match the current bot's plan. It shows verified
photos in green, the current/failed/retrying target in blue, completion
percentage, and a live description of the capture or quality-repair phase.

The app asks the integration for inventory in the configured lookback window,
then retrieves processed JPEGs one at a time at the configured resolution (at
most 1280 × 960). It validates content type, checksum over the returned JPEG,
base64, JPEG format, decoded dimensions, resize-scale consistency, aspect ratio,
absence of upscaling, and payload/dimension limits. Base64 image data is never
logged or persisted.

The classical pipeline combines HSV and Excess Green, morphology, components, known-centre seeds, nearest-centre ownership, historical-mask evidence, maximum accepted distance, and confidence. Ambiguous overlap prevents writes. The protection radius is the largest accepted leaf distance plus safety and calibration margins; a separate 90th-percentile value is retained only as the typical canopy measurement.

## Weed detection and the learned verifier

Weed detection runs in two stages, and it matters which one is doing what.

The **heuristic** turns vegetation that no known plant owns into candidates,
using area, colour purity and shape. Everything it measures rises for moss,
fallen leaves and crop foliage exactly as it does for a weed, so it is a
candidate generator, not a classifier — tightening its thresholds mostly
removes small weeds rather than false positives.

The **learned verifier** is a small logistic model trained on your own labels.
It sees sixteen visual features the heuristic ignores (hue, saturation,
texture, edge density, and what surrounds the candidate: mulch orange, bare
neutral soil, neighbouring canopy) plus the distance to the nearest known
plant. Once it is trained and enforcing, its score *is* the weed confidence and
the **verifier confidence threshold** is the gate.

The intended path:

1. Enable weed detection with the verifier in **shadow mode**. Candidates are
   scored but the heuristic still decides, so nothing changes yet.
2. Review detections. Accepting records a weed; the category buttons record
   hard negatives (moss, mushroom, fallen leaf, mulch/soil, hardware). The
   **Most informative to label next** list on the weed settings page shows the
   candidates sitting closest to the decision boundary — those teach the model
   far more than another obvious weed.
3. Train. Validation holds out whole images, and the reported precision comes
   with a 90% lower confidence bound so a small label set is not flattered.
4. Apply the **suggested threshold**, which is the highest-recall operating
   point whose precision is confidently at or above 95%.
5. Turn shadow mode off. The **candidate recall boost** then relaxes the
   colour/shape gates so borderline weeds reach the verifier rather than being
   dropped by rules that cannot classify them.
6. Only then consider automatic creation, which can additionally require
   verifier approval and several independent looks.

During review the dialog shows the verifier's **best guess** at what the object
is — "moss 71% · fallen leaf 18%" — from per-category heads trained on the same
labels. It is there to make a borderline detection easier to judge and never
affects the accept/reject decision.

Multi-image confirmation filters flicker from shadows, wind and exposure. It
does not help against moss, stones or hardware, which are perfectly persistent
and will be seen the same way on every pass; those need labels, not more looks.

**Export and import.** *Export labels and model* produces a JSON bundle of every
label with its features, which is what training consumes. Use it to back up
your labels, move them to another FarmBot, or seed a fresh install: import the
bundle and retrain, or place it next to the source as
`bundled_weed_model.json`, where a new install will use it until it has trained
a model of its own. Crop images are only included if you ask for them, since
they are needed only for re-checking labels by eye.

## Boundaries and exclusion zones

The **Boundaries & zones** tab defines where the app may place things, in FarmBot
garden millimetres. A **boundary** encloses the area where placement is allowed;
an **exclusion zone** marks an area to keep clear. Zones can be rectangles,
circles, or polygons, and each one can be switched off without deleting it.

Every zone states independently whether, inside it:

- **weeds** may be created,
- a **plant centre** may be moved there,
- a plant's **protection radius** may extend into it.

Overlaps resolve in a fixed order: an exclusion zone that *allows* an aspect is
an explicit exception and wins; otherwise any zone that *forbids* the aspect and
is touched by the position denies it; otherwise, if at least one boundary allows
that aspect, the position must fall inside one of them. With no zones configured
nothing is restricted, so existing installations behave exactly as before.
Clearing a tick box on a boundary carves a hole in it for that aspect only.

Weeds and plant centres are tested as points. A protection radius is tested as a
disc: it must fit entirely inside an allowing boundary and must not overlap a
forbidding zone at all.

Zones gate automatic writes and manual approvals alike. A weed detected in a
forbidden position is discarded rather than stored, so it is never created and
never appears for review; the last-job card counts how many were dropped. A
blocked automatic radius increase stays a recommendation whose reason names the
zone, and manual approval of a blocked radius or centre move is refused with the
same explanation. Measurements recorded before the app stored plant positions
cannot be radius-checked and keep their previous behaviour.

Zones are stored in `/data/zones.json` and survive restarts and upgrades. The
tab's garden map draws every zone and can overlay the bot's plants (with their
protection radii) and existing FarmBot weeds so a zone can be checked before it
starts gating writes.

## Scheduling and resources

Automatic new-photo, manual, and integration-event runs are available. Daily scheduling is disabled by default and does not run until a FarmBot and calibration exist. Only one job and one image run at a time. OpenCV and common numerical thread pools are limited to one thread. Jobs pause when CPU or free-memory gates fail.

The design targets under 200 MB idle and under 600 MB peak RSS on a Pi 4 at the
default 960 × 720. Image arrays are released after each image, masks are stored
compressed (PNG) rather than as raw arrays, and only one decoded image is held
at a time. `cv2.setNumThreads(1)` is called at startup and the numerical thread
pools are pinned to one. Approximate relative CPU/memory cost by preset: 640 ×
480 ≈ 1×, 960 × 720 ≈ 2.25×, 1280 × 960 ≈ 4× (native 2592 × 1944 ≈ 16.4×, not
selectable). On a 4 GB Pi 4 use 960 × 720, keep the free-memory gate at 512 MB
and the CPU gate at 80%; 1280 × 960 is viable but leaves less headroom. The
health page reports version, algorithm and contract versions, selected
resolution and pixel count, job timing, peak RSS, OpenCV threads, database size,
and artifact size.

## Curves and rollback

Raw measurements remain separate from learned curves. Crop radii are grouped into age bins, an upper quantile is taken, and Pool Adjacent Violators produces a monotonic curve with at most ten points. Displayed FarmBot values are diameters. First assignment always requires approval. Version 0.1.0 does not send curve writes; ownership and rollback tables are reserved for the explicitly opted-in future workflow.

Approved individual changes are auditable in the decisions table and protected by the integration's `expected_current_radius_mm` optimistic concurrency check. A stale response causes an inventory refresh, never a forced update.

## Retention, privacy, and export

SQLite and artifacts live only in `/data`. The database uses WAL, normal synchronous mode, foreign keys, and a busy timeout. Successful masks default to 7 days and overlays to 14 days. Original FarmBot images and base64 payloads are not stored. Logs exclude image URLs, image data, tokens, and credentials.

For future labelled-model work, back up the app and export selected overlay/mask files together with matching measurement rows from `farmbot_vision.db`. Remove garden-identifying metadata before sharing. The UI intentionally does not expose a bulk public export endpoint.

## Troubleshooting

- **Ingress dashboard shows `{"detail":"Not Found"}` or logs show `GET // HTTP/1.1 404 Not Found`:** verify the installed app is at least **0.2.1**, restart it, close the old browser tab, and reopen the Web UI for a fresh Ingress session.
- **Logs show `Vision event connection interrupted: ValidationError`:** verify the app is at least **0.2.1** so `VisionRequestEvent` accepts the optional `device_id` and invalid individual events are handled locally.
- **No bots or inventory:** upgrade the companion integration and reauthenticate it in Home Assistant.
- **Calibration required:** complete manual calibration or supply it from the integration.
- **No vegetation connected:** verify image coordinates, rotation, lighting, and HSV suitability.
- **Uncertain:** inspect overlap, a newly disconnected green region, edge clipping, or excessive growth.
- **Stale radius:** the plant changed after measurement; rerun analysis.
- **Paused:** reduce other workload or lower the image volume; do not disable memory protection casually.
- **Corrupt database:** stop the app, restore `/data` from a Home Assistant backup, and retain the corrupt file for diagnosis.

## Future strategy: hybrid plant crop analysis

A future release may add "hybrid plant crop analysis": one resized full-frame
context image plus one higher-resolution crop around an individual plant, with
mappings from crop pixels back to full-image pixels and from full-image pixels
back to FarmBot coordinates. Only the interfaces are shaped for this now; no
heavyweight full native-resolution pipeline is included, and no user setting is
exposed for it until it works.

## Current limitations

Uncalibrated runs produce pixel-only diagnostics with no millimetre radius and
no writes. A fallback such as "1 pixel per millimetre" is never treated as valid
calibration. Temporal registration remains translation-only, and masks from a
different resolution are rescaled by dimension ratio or rejected. Green material
underneath a crop canopy cannot be seen. Dense touching canopies may remain deliberately unresolved. Classical colour segmentation is sensitive to unusual lighting and non-plant green objects. Temporal registration is translation-only. Manual point selection uses entered pixel coordinates rather than an interactive canvas in 0.1.0. The application cannot validate FarmBot writes beyond the response contract; the companion integration remains the final authority.
