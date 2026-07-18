# FarmBot Vision documentation

## Upgrading from 0.2.0

Install app version **0.2.1**, then restart the app. Close the old FarmBot
Vision browser tab and reopen the Web UI so Home Assistant creates a fresh
Ingress session. The companion integration does not need to be changed.

This release removes the explicit root Ingress entry and normalizes duplicate
leading slashes at the ASGI boundary. Dashboard and calibration URLs are
relative to the current `X-Ingress-Path`, so they remain inside temporary
Ingress sessions. The app never logs the complete `X-Ingress-Path` value.

The accepted `farmbot_vision_request` data is:

```json
{"config_entry_id":"string","device_id":"string","plant_ids":[],"mode":"recommend"}
```

`device_id` may be omitted, `plant_ids` must contain positive integers, and an
empty list means all eligible plants. Unknown fields remain rejected. A single
malformed JSON or invalid event is skipped in place; it does not reconnect the
WebSocket, and subsequent valid events continue to be processed.

## Before enabling it

FarmBot Vision requires Home Assistant Core 2026.7 or newer and a companion FarmBot integration that provides the actions in `docs/integration-contract.md`. Start in **Observe** mode. Do not use early experimental output as the sole input to destructive weeding.

## Modes

- **Observe** stores measurements and diagnostic overlays without writes.
- **Recommend** proposes individual increases and exposes approve/reject controls. Nothing is written until approved.
- **Auto radius** writes only high-confidence increases that pass all configured limits. It never shrinks a radius.
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

If no integration calibration is available, open **Calibration**, pick a recent
FarmBot image (shown at the configured resolution), click point A and point B on
two features a known distance apart, enter that separation, preview the
resulting pixels-per-millimetre, adjust rotation and offsets, then overlay the
known plant centres and confirm several align before saving. No external tools
are needed. Manual calibration records the config entry, image, processed
resolution, pixel points, separation and version; when the resolution changes it
is only reused if the scaling relationship is fully known (and then a transformed
calibration is recorded), otherwise recalibration is required. Automatic writes
and approvals are refused without valid calibration.

The transformation assumes image metadata `x,y` is the image's ground centre.
Rotation is applied in the image plane; offsets are in millimetres. Every
observation retains the exact transform, resolution, resize scales, calibration
source and version used.

## Image selection and analysis

The app asks the integration for inventory in the configured lookback window,
then retrieves processed JPEGs one at a time at the configured resolution (at
most 1280 × 960). It validates content type, checksum over the returned JPEG,
base64, JPEG format, decoded dimensions, resize-scale consistency, aspect ratio,
absence of upscaling, and payload/dimension limits. Base64 image data is never
logged or persisted.

The classical pipeline combines HSV and Excess Green, morphology, components, known-centre seeds, nearest-centre ownership, historical-mask evidence, maximum accepted distance, and confidence. Ambiguous overlap prevents writes. The protection radius is the largest accepted leaf distance plus safety and calibration margins; a separate 90th-percentile value is retained only as the typical canopy measurement.

## Scheduling and resources

Manual and integration-event runs are always available. Daily scheduling is disabled by default and does not run until a FarmBot and calibration exist. Only one job and one image run at a time. OpenCV and common numerical thread pools are limited to one thread. Jobs pause when CPU or free-memory gates fail.

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
