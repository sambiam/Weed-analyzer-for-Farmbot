# Architecture and security

```text
FarmBot API ← companion Home Assistant integration ← Core service actions/events
                                                      ↑ Supervisor proxy
                                               FarmBot Vision app
                       Ingress UI ← FastAPI ← job lock ← CV engine → SQLite + artifacts
```

The companion integration owns authentication, token refresh, resource retrieval, image download, write validation, and FarmBot mutations. FarmBot Vision owns analysis, local observations, proposals, and operator review. The boundary is deliberately typed and versionable.

FastAPI serves a relative-link, `X-Ingress-Path`-aware interface on internal
port 8099. `NormalizeIngressPathMiddleware` rewrites only duplicate leading
slashes in HTTP and WebSocket ASGI scopes before route matching; it does not
redirect, alter query strings, or log the temporary Ingress path. An
asynchronous event listener and optional low-frequency scheduler feed a single
job lock. Images are fetched and released sequentially. CPU-heavy OpenCV work
runs outside the event loop, with a 60-second timeout and resource checks
between images.

The `ImageAnalysisEngine` abstraction isolates the classical implementation so a small ONNX or TFLite engine can be added later without changing jobs or persistence. No inference runtime ships.

Analysis resolution is configurable (640×480 / 960×720 default / 1280×960) via the `Resolution` model; the job manager requests exactly that size and images are validated and calibrated against the exact processed pixels. Calibration is resolved per image in preference order — processed calibration, reference calibration scaled to the resolution, compatible manual calibration, none — so a native-resolution scale is never applied to a resized frame. Morphology kernels and area thresholds are derived from the effective pixels-per-millimetre (or, uncalibrated, from image dimensions relative to the 640×480 baseline) so physical results stay comparable across presets. The engine interfaces are shaped to allow a future "hybrid plant crop analysis" strategy (a resized full-frame context image plus a higher-resolution per-plant crop) without committing to a native-resolution pipeline now.

Security posture: protection stays enabled; the app runs as root inside the container because Home Assistant protects `/data/options.json` with mode `0600`, but it requests no host network, privileged capability, full access, Docker API, hardware, or Home Assistant configuration mount. AppArmor permits only required runtime libraries, networking, read-only resource counters, `/tmp`, and `/data`. The Supervisor token exists only in process memory and is never logged.

## Uncertain assumptions

- Integration service responses use Home Assistant's `?return_response` envelope.
- Image metadata `x,y` denotes the ground coordinate at image centre.
- Positive FarmBot X/Y map to image axes after the configured rotation.
- The integration returns JPEG, not PNG/HEIC, and limits its response to requested dimensions (≤ 1280×960).
- The integration reports source, oriented and processed dimensions and resize scales, and computes `sha256` over the returned JPEG (contract farmbot-vision-v2). Reference calibration pixel scales are stated at a declared reference resolution.
- Home Assistant supplies the local timezone to the app environment; otherwise the schedule follows container local time.

These assumptions are explicit so companion integration tests can lock them down.
