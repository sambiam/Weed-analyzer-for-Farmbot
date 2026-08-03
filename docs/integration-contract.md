# Companion FarmBot integration contract

Contract version: **farmbot-vision-v2**. Minimum compatible companion FarmBot
integration release: **2.2.0** (the release that advertises and implements
photo-grid repair in addition to clear-site soil-height capture, the
returned-JPEG contract and known-weed writes). Version 2.2.0 of the companion integration in the sibling
`Farmbot-for-Home-Assistant` repository implements this contract.

All actions are in the `farmbot` domain. Response actions must support Home Assistant service response data. Unknown, invalid, or unauthorised fields must fail rather than be coerced. Timestamps are ISO-8601. The integration remains the only component that talks to FarmBot APIs.

## `farmbot.list_vision_bots`

No input. Response:

```json
{"bots":[{"config_entry_id":"string","device_id":"string","name":"string","integration_version":"2.2.0","capabilities":["photo_grid_repair","verified_photo_grid_repair","position_verified_photo_grid_repair","illuminated_photo_grid_capture","experimental_raw_gcode"]}]}
```

## `farmbot.get_vision_inventory`

Input: `{"config_entry_id":"string","image_lookback_hours":72}`.

Response fields:

- `device_id`, `generated_at`
- `plants[]`: `id`, `name`, `openfarm_slug`, `x`, `y`, `z`, `radius`, `plant_stage`, nullable `planted_at`, nullable `spread_curve_id`
- `images[]`: `id`, `created_at`, `processed`, and `meta` containing `x`, `y`, `z`, optional `name`.
  The app also tolerates a non-conforming shape observed from at least one companion
  integration build in production: `x`/`y`/`z`/`name` sent flat on the image object
  instead of nested under `meta`, and `processed` omitted entirely (treated as `true`).
  This is a compatibility shim in `InventoryImage._normalize`, not the target contract —
  new integration work should still emit the nested/complete shape above.
- `curves[]`: `id`, `name`, `type` (must be `spread`), and day-string to diameter mapping `data`
- `weeds[]` (optional): FarmBot `Weed` map points as `id`, nullable `name`, `x`, `y`, `z`, `radius`. Omit the key entirely (or send `[]`) when the integration does not expose weeds; the app treats a missing list as empty. Used by the calibration composite overlay to distinguish weeds from known plants.
- `camera_calibration`: `available`, nullable positive `pixels_per_mm_x/y`, nullable `rotation_degrees`, nullable `offset_x_mm/y`, and (v2) `reference_width`, `reference_height`, `basis` (`reference_image` or `native_frame`)

When `available` is true both pixel scales are required. This is the **reference** (normalized) calibration: `pixels_per_mm_*` are stated relative to `reference_width` x `reference_height`. The app scales them to the processed resolution — a native scale is never applied directly to a resized frame. Image metadata coordinates are defined as the ground coordinate at image centre.

## `farmbot.get_vision_image`

Input: `{"config_entry_id":"string","image_id":456,"max_width":960,"max_height":720}`. `max_width`/`max_height` are the app's configured analysis resolution and are at most 1280 x 960.

Response (contract v2):

- `image_id`, `content_type` (only `image/jpeg`)
- lowercase hex `sha256` **over the returned JPEG bytes** (the app verifies the bytes it receives)
- optional `source_sha256` over the original download (format-checked only; never verified because the original is not shipped)
- `source_width`, `source_height` (before EXIF orientation)
- `oriented_width`, `oriented_height` (after EXIF orientation)
- `width`, `height` (processed, ≤ requested and ≤ 1280 x 960)
- `resize_scale_x` = `width / oriented_width`, `resize_scale_y` = `height / oriented_height`
- `image_base64`, and `meta` containing `x`, `y`, `z`, `created_at`
- optional `processed_calibration`: `{available, pixels_per_mm_x, pixels_per_mm_y, rotation_degrees, offset_x_mm, offset_y_mm, basis:"processed_image", width, height}` where `width`/`height` equal the returned image

The integration must resize before base64 encoding and must not return signed URLs. The app fetches sequentially and independently validates the checksum, base64, JPEG format, decoded dimensions, resize-scale consistency, aspect ratio, absence of upscaling, and payload/dimension limits. Older v1 responses (no `source_*`/`oriented_*`/`resize_scale_*`) are accepted as a legacy path but yield pixel-only diagnostics with no metric writes.

## Photo-grid repair services

`farmbot.start_vision_grid_repair` accepts `config_entry_id` and `targets`,
each containing finite `x`, `y`, and `z` coordinates plus an optional
caller-owned `index`. Integration 2.5.0 accepts up to 256 targets in one call
and advertises this as the `continuous_photo_grid_capture` capability, so a
whole bed grid is one run: the lighting, the drive in from the parked position
and the drive back to it happen once around the entire route rather than once
per call. Integrations before 2.5.0 cap a call at twelve targets and reject an
oversized call during Home Assistant's schema validation, with no status record
and no partial capture; the app falls back to twelve-target chunking whenever
the capability is absent, and to sending no `index` unless
`indexed_photo_grid_targets` is advertised (older schemas reject unknown target
keys). Cell-to-cell moves inside the grid skip the `safe_z` retract only when
every target shares one Z that is already within 25 mm of the top of the Z
axis; the move into the grid and the move back out always retract. It validates
connection, emergency-stop, busy state, and all axis bounds before queuing
each acknowledged safe-Z move separately from its settle/photo request. A
photo request must not be sent until a fresh FarmBot status report places all
three axes within 5 mm of the requested target. A target is complete only when
the REST image inventory contains a new,
fully-processed image within 25 mm of its requested X/Y/Z coordinates.
`take_photo` may be retried six times because FarmBot OS reports its camera
errors asynchronously. After six failed attempts the app records that cell as
failed for the current session and continues repairing the remaining cells.
It returns a `repair_id` immediately and restores the initial position when
the command finishes or fails. Integration 2.2.0 and the
`illuminated_photo_grid_capture` capability additionally preserve the current
state of the standard FarmBot lighting peripheral (pin 7), turn it on before
the first movement/photo, and restore its prior state during cleanup. Failure
to enable lighting stops the request before a photo is taken.

`farmbot.get_vision_grid_repair` accepts `config_entry_id` and `repair_id`.
It returns `queued|running|waiting_images|complete|failed`, a sanitized
message, the validated target coordinates, verified frames, per-target
completion/failure details, and the current photo-attempt number. From 2.5.0
each frame carries the `target_index` it was captured for, `failures` carries
`{index, reason, code}` with `code` one of `movement`, `camera`,
`upload_timeout` or `error`, and an aborted run reports the cells it never
attempted as `unattempted_targets`. An `upload_timeout` is an unknown
completion state and never counts as a captured cell.

`farmbot.delete_vision_image` accepts `config_entry_id` and `image_id`, and
returns `deleted|rejected` with the image ID and a sanitized message. It
deletes the image only when it belongs to the config entry's own FarmBot, and
reports an already-absent image as `deleted` so a retry is safe. The app calls
it only to retire a gantry-obscured grid cell's photo once a usable photo of
the same cell has replaced it. It is advertised as the `vision_image_deletion`
capability; an integration without it keeps the replaced photo, which is a
cosmetic loss only, because the app credits the replacement to the cell
regardless.

Each verified repair frame's `image_id` is what lets the app attribute a
repair photo to the grid run that needed it. A repair happens well outside the
one-hour window that defines a run, so without those IDs the repaired cell
would keep reading as missing and would be re-photographed forever.

## Experimental raw G-code services

Advertised as the `experimental_raw_gcode` capability by integration 2.6.0 and
newer. This is the only part of the contract that moves FarmBot outside FarmBot
OS's motion planning: FarmBot OS accepts only CeleryScript on its normal path,
so the program is delivered through FarmBot OS v15's Lua `gcode()` function,
which forwards a command to the Farmduino verbatim and validates nothing. The
integration is therefore the whole of the safety story on this path, and the
app must not treat its own checks as sufficient.

`gcode()` requires **FarmBot OS v15 or newer**. Nothing in the contract reports
the Lua API's version, so an older FarmBot OS is not detected up front: the run
is accepted and then fails when its first chunk executes. Advertising
`experimental_raw_gcode` says the *integration* supports this, not that the bot
does.

`farmbot.start_vision_gcode` accepts `config_entry_id`, `lines` (the program,
one line per entry, at most 2000), `feed_mm_per_min` (1-3000, default 400),
`return_to_start`, `dry_run`, and a required `acknowledge_experimental` that
must be `true`. It accepts only `G21`, `G90`, `G91`, `G00` (with X/Y/Z/F/A/B/C)
and a standalone `F`; everything else is rejected by name, including `G01`,
which the FarmBot firmware does not implement. `Q` is rejected because FarmBot
OS appends its own and setting it crashes FarmBot OS.

The integration resolves every move to an absolute XYZ target, refuses the
whole program if any resolved point leaves the axis bounds derived from
firmware config, and converts the feed rate into the per-axis `A`/`B`/`C`
speeds in steps/second the firmware takes, scaled so all axes finish together
(`G00` does not interpolate) and clamped to `movement_max_spd_*`. Explicit
A/B/C on a line pass through, still clamped. It refuses to start unless the bot
is connected, unlocked, idle and reporting a position on all three axes, aborts
between chunks on disconnect or emergency stop, and returns to the starting
position through FarmBot OS's own supervised movement with `safe_z`.

With `dry_run: true` it returns `{"status":"validated", "moves", "total_distance_mm",
"feed_mm_per_min", "extent", "warnings", "message"}` and moves nothing.
Otherwise it returns `{"status":"queued", "run_id", "message"}`, or
`{"status":"rejected", "message"}` naming the offending line.

`farmbot.get_vision_gcode` accepts `config_entry_id` and `run_id` and returns
`queued|running|complete|failed` with `message`, `moves`, `chunks_sent`,
`chunks_total`, `total_distance_mm`, `feed_mm_per_min`, `start_position`,
`extent` and `warnings`.

## Soil-height services

`farmbot.get_vision_soil_points` accepts `config_entry_id`. It returns active
FarmBot `GenericPointer` records recognized only by
`meta.created_by == "measure-soil-height"` or `meta.at_soil_level == true`,
plus the current position, connected/busy/emergency-stop state, Z direction and
axis bounds. A matching display name alone is never sufficient.

`farmbot.start_vision_soil_capture` accepts `config_entry_id`, `point_id`,
optional paired `capture_x`/`capture_y`, `capture_z`, `baseline_mm`, and
`z_offsets_mm`, plus an optional measurement `batch_id` UUID. Use `[0]` for a
measurement and `[0,25,50]` for calibration.
A relocated capture must be less than 200 mm from a point last updated more
than 14 days ago. It returns a `capture_id` immediately. The
integration validates the bot and bounds, uses acknowledged safe-Z movement,
captures `-baseline, 0, +baseline` along Y (or an actual-coordinate one-sided
triplet at an edge), waits for processed images, claims those image IDs from
ordinary photo analysis, and restores the initial position when possible.
Captures sharing a `batch_id` serialize behind one another and defer that
restoration.

`farmbot.get_vision_soil_capture` accepts `config_entry_id` and `capture_id`.
It returns `queued|running|waiting_images|complete|failed`, a sanitized
message, and completed frames as `image_id`, `x`, `y`, `z`,
`lateral_offset_mm`, and `z_offset_mm`.

`farmbot.finish_vision_soil_capture_batch` accepts `config_entry_id` and
`batch_id`. It waits for the batch's last atomic capture, restores the position
saved when its first capture started, and returns `complete` only after the
acknowledged safe-Z move finishes. Repeating the finish call is idempotent.

`farmbot.apply_vision_soil_height` accepts `config_entry_id`, `point_id`,
`measurement_id`, expected `x/y/z` and `updated_at`, recommended `x/y/z`,
`confidence`, `apply`, and `human_approved`. Writes require both booleans,
re-fetch the point, enforce a 0.5 mm coordinate tolerance, unchanged timestamp,
axis bounds, age over 14 days, and relocation under 200 mm. The existing point
is relocated and its Z is updated without replacing its metadata. Missing,
discarded, wrong-type, fresh, changed, non-finite, and unrecognized points fail
closed.

## `farmbot.apply_vision_radius`

Input:

```json
{"config_entry_id":"string","plant_id":123,"measurement_id":"UUID","expected_current_radius_mm":120.0,"recommended_radius_mm":185.0,"confidence":0.94,"apply":false}
```

The integration must validate bounds, authorization, and optimistic concurrency. A stale current radius must return HTTP-equivalent conflict semantics (409 or 412 through the service response path) or a structured failure that can be mapped to that condition. `apply:false` validates without mutation. It may return a response. Automatic writes may include only small radius decreases (10% by default in the companion integration); larger decreases are for explicit human approval.

## `farmbot.upsert_vision_spread_curve`

Input:

```json
{"config_entry_id":"string","crop_slug":"lettuce","curve_id":null,"name":"[FarmBot Vision] Lettuce protection spread","data":{"1":30,"14":100,"30":300},"assign_to_plant_ids":[123,124],"apply":false}
```

Values are diameters in millimetres. A new curve name must use `[FarmBot Vision]`. The integration must reject modification of non-adopted user curves and validate app-owned curve IDs. `apply:false` validates only. FarmBot Vision 0.1.0 does not call this advanced action automatically.

## Known weed tracking

`farmbot.update_vision_weed_radius` accepts `config_entry_id`, `weed_id`,
`expected_current_radius_mm`, `recommended_radius_mm`, `confidence`, `apply`,
and `human_approved`. It verifies that the point is a Weed, enforces optimistic
concurrency, and never reduces the existing radius.

`farmbot.remove_vision_weed` accepts `config_entry_id`, `weed_id`, `confidence`,
`apply`, and `human_approved`. It verifies that the point is a Weed before
removal. The app only calls it after the user enables automatic weed removal
and consecutive fully visible images produce explicit verifier-backed absence
results. A missing final weed detection by itself never authorises removal.

## `farmbot.report_vision_status`

Input fields: `config_entry_id`, `available`, `status` (`idle|running|warning|error`), nullable `job_id`, nullable `last_completed_at`, integer `plants_analysed`, `recommendations`, `automatically_applied`, `uncertain`, and a non-sensitive `message` no longer than 240 characters. The integration should avoid creating recorder churn when the payload is unchanged.

## `farmbot_vision_request` event

Event data:

```json
{
  "config_entry_id": "string",
  "device_id": "string",
  "plant_ids": [],
  "mode": "recommend"
}
```

`device_id` is optional for backward compatibility. `mode` must be one of
`observe`, `recommend`, or `auto_radius`; every `plant_ids` value must be a
positive integer. An empty plant list means all eligible plants. The app
rejects unknown event fields. A malformed event is logged with sanitized field
and error-type details and skipped without reconnecting the active subscription,
so a later valid event is still processed. The integration should not emit
overlapping requests repeatedly.

Automatic new-photo requests may instead omit `mode` and include a positive
integer `image_id`. In that form the app uses its configured operating mode and
processes only the named image. Manual requests retain the payload above.

## Companion implementation status

1. Register all documented actions with response support where specified.
2. Resolve `config_entry_id` only against loaded FarmBot entries and authorise every resource/write.
3. Add bounded image download/resize/JPEG encoding and SHA-256 response generation.
4. Expose plant, image, spread-curve, and calibration serializers with the exact field semantics above.
5. Implement optimistic radius concurrency and map stale writes distinctly.
6. Track adopted/FarmBot-Vision curve ownership and reject arbitrary user-curve modification.
7. Add status entities or diagnostics with update de-duplication.
8. Emit the request event from integration services/UI controls.
9. Add contract, malformed-response, authentication, reauthentication, stale-write, and permission tests.
10. Declare the minimum companion version once that integration release exists.
    **Done: companion 1.8.0 implements the image, known-weed and clear-site soil-height
    contracts.**

## Contract v2 summary of required integration capabilities

1. `sha256` computed over the returned (re-encoded) JPEG bytes.
2. `source_width/height`, `oriented_width/height`, processed `width/height`.
3. `resize_scale_x/y` equal to processed÷oriented in each axis.
4. `processed_calibration` (basis `processed_image`) when calibration is known, plus reference dimensions on `camera_calibration`.

The minimum compatible companion integration version is **2.2.0**.
