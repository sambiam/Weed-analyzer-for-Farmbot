# Companion FarmBot integration contract

Required companion integration version: a future release implementing contract **farmbot-vision-v1**. The existing integration is not changed in this repository.

All actions are in the `farmbot` domain. Response actions must support Home Assistant service response data. Unknown, invalid, or unauthorised fields must fail rather than be coerced. Timestamps are ISO-8601. The integration remains the only component that talks to FarmBot APIs.

## `farmbot.list_vision_bots`

No input. Response:

```json
{"bots":[{"config_entry_id":"string","device_id":"string","name":"string"}]}
```

## `farmbot.get_vision_inventory`

Input: `{"config_entry_id":"string","image_lookback_hours":72}`.

Response fields:

- `device_id`, `generated_at`
- `plants[]`: `id`, `name`, `openfarm_slug`, `x`, `y`, `z`, `radius`, `plant_stage`, nullable `planted_at`, nullable `spread_curve_id`
- `images[]`: `id`, `created_at`, `processed`, and `meta` containing `x`, `y`, `z`, optional `name`
- `curves[]`: `id`, `name`, `type` (must be `spread`), and day-string to diameter mapping `data`
- `camera_calibration`: `available`, nullable positive `pixels_per_mm_x/y`, nullable `rotation_degrees`, nullable `offset_x_mm/y`

When `available` is true both pixel scales are required. Image metadata coordinates are defined as the ground coordinate at image centre.

## `farmbot.get_vision_image`

Input: `{"config_entry_id":"string","image_id":456,"max_width":640,"max_height":480}`.

Response: `image_id`, `content_type` (only `image/jpeg` in v1), lowercase hex `sha256`, `width`, `height`, `image_base64`, and `meta` containing `x`, `y`, `z`, `created_at`. The integration must resize before base64 encoding and must not return signed URLs. The app fetches sequentially and validates hash, encoding, dimensions, and payload size.

## `farmbot.apply_vision_radius`

Input:

```json
{"config_entry_id":"string","plant_id":123,"measurement_id":"UUID","expected_current_radius_mm":120.0,"recommended_radius_mm":185.0,"confidence":0.94,"apply":false}
```

The integration must validate bounds, authorization, and optimistic concurrency. A stale current radius must return HTTP-equivalent conflict semantics (409 or 412 through the service response path) or a structured failure that can be mapped to that condition. `apply:false` validates without mutation. It may return a response.

## `farmbot.upsert_vision_spread_curve`

Input:

```json
{"config_entry_id":"string","crop_slug":"lettuce","curve_id":null,"name":"[FarmBot Vision] Lettuce protection spread","data":{"1":30,"14":100,"30":300},"assign_to_plant_ids":[123,124],"apply":false}
```

Values are diameters in millimetres. A new curve name must use `[FarmBot Vision]`. The integration must reject modification of non-adopted user curves and validate app-owned curve IDs. `apply:false` validates only. FarmBot Vision 0.1.0 does not call this advanced action automatically.

## `farmbot.report_vision_status`

Input fields: `config_entry_id`, `available`, `status` (`idle|running|warning|error`), nullable `job_id`, nullable `last_completed_at`, integer `plants_analysed`, `recommendations`, `automatically_applied`, `uncertain`, and a non-sensitive `message` no longer than 240 characters. The integration should avoid creating recorder churn when the payload is unchanged.

## `farmbot_vision_request` event

Event data: `{"config_entry_id":"string","plant_ids":[],"mode":"observe|recommend|auto_radius"}`. An empty plant list means all eligible plants. The integration should not emit overlapping requests repeatedly.

## Exact companion-integration work still required

1. Register the five actions above with response support where specified.
2. Resolve `config_entry_id` only against loaded FarmBot entries and authorise every resource/write.
3. Add bounded image download/resize/JPEG encoding and SHA-256 response generation.
4. Expose plant, image, spread-curve, and calibration serializers with the exact field semantics above.
5. Implement optimistic radius concurrency and map stale writes distinctly.
6. Track adopted/FarmBot-Vision curve ownership and reject arbitrary user-curve modification.
7. Add status entities or diagnostics with update de-duplication.
8. Emit the request event from integration services/UI controls.
9. Add contract, malformed-response, authentication, reauthentication, stale-write, and permission tests.
10. Declare the minimum companion version once that integration release exists.
