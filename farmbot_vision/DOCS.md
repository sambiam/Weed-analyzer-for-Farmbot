# FarmBot Vision documentation

## Before enabling it

FarmBot Vision requires Home Assistant Core 2026.7 or newer and a companion FarmBot integration that provides the actions in `docs/integration-contract.md`. Start in **Observe** mode. Do not use early experimental output as the sole input to destructive weeding.

## Modes

- **Observe** stores measurements and diagnostic overlays without writes.
- **Recommend** proposes individual increases and exposes approve/reject controls. Nothing is written until approved.
- **Auto radius** writes only high-confidence increases that pass all configured limits. It never shrinks a radius.
- **Auto curve** is a future advanced mode and is intentionally unavailable in 0.1.0. Curve proposals never modify or replace a user-created curve.

## Calibration

Integration-provided calibration takes precedence. If unavailable, open **Calibration**, choose two points from a representative overhead photograph, measure their real separation, and enter the pixel coordinates, distance, rotation, and FarmBot-coordinate offsets. Check the resulting projected plant centres against multiple known plants before enabling recommendations. Automatic writes are refused without calibration.

The transformation assumes image metadata `x,y` is the image's ground centre. Rotation is applied in the image plane; offsets are in millimetres. Every observation retains the exact transform and calibration version used.

## Image selection and analysis

The app asks the integration for inventory in the configured lookback window, then retrieves processed JPEGs one at a time at no more than 640 × 480. It validates type, dimensions, base64 size, decoded size, and SHA-256. Original base64 is never persisted.

The classical pipeline combines HSV and Excess Green, morphology, components, known-centre seeds, nearest-centre ownership, historical-mask evidence, maximum accepted distance, and confidence. Ambiguous overlap prevents writes. The protection radius is the largest accepted leaf distance plus safety and calibration margins; a separate 90th-percentile value is retained only as the typical canopy measurement.

## Scheduling and resources

Manual and integration-event runs are always available. Daily scheduling is disabled by default and does not run until a FarmBot and calibration exist. Only one job and one image run at a time. OpenCV and common numerical thread pools are limited to one thread. Jobs pause when CPU or free-memory gates fail.

The design targets under 200 MB idle and under 600 MB peak RSS on a Pi 4. The synthetic 640 × 480 benchmark is enforced at under three seconds on CI-class hardware; actual Pi timings vary with image content. The health page reports version, algorithm, job timing, peak RSS, OpenCV threads, database size, and artifact size.

## Curves and rollback

Raw measurements remain separate from learned curves. Crop radii are grouped into age bins, an upper quantile is taken, and Pool Adjacent Violators produces a monotonic curve with at most ten points. Displayed FarmBot values are diameters. First assignment always requires approval. Version 0.1.0 does not send curve writes; ownership and rollback tables are reserved for the explicitly opted-in future workflow.

Approved individual changes are auditable in the decisions table and protected by the integration's `expected_current_radius_mm` optimistic concurrency check. A stale response causes an inventory refresh, never a forced update.

## Retention, privacy, and export

SQLite and artifacts live only in `/data`. The database uses WAL, normal synchronous mode, foreign keys, and a busy timeout. Successful masks default to 7 days and overlays to 14 days. Original FarmBot images and base64 payloads are not stored. Logs exclude image URLs, image data, tokens, and credentials.

For future labelled-model work, back up the app and export selected overlay/mask files together with matching measurement rows from `farmbot_vision.db`. Remove garden-identifying metadata before sharing. The UI intentionally does not expose a bulk public export endpoint.

## Troubleshooting

- **No bots or inventory:** upgrade the companion integration and reauthenticate it in Home Assistant.
- **Calibration required:** complete manual calibration or supply it from the integration.
- **No vegetation connected:** verify image coordinates, rotation, lighting, and HSV suitability.
- **Uncertain:** inspect overlap, a newly disconnected green region, edge clipping, or excessive growth.
- **Stale radius:** the plant changed after measurement; rerun analysis.
- **Paused:** reduce other workload or lower the image volume; do not disable memory protection casually.
- **Corrupt database:** stop the app, restore `/data` from a Home Assistant backup, and retain the corrupt file for diagnosis.

## Current limitations

Green material underneath a crop canopy cannot be seen. Dense touching canopies may remain deliberately unresolved. Classical colour segmentation is sensitive to unusual lighting and non-plant green objects. Temporal registration is translation-only. Manual point selection uses entered pixel coordinates rather than an interactive canvas in 0.1.0. The application cannot validate FarmBot writes beyond the response contract; the companion integration remains the final authority.
