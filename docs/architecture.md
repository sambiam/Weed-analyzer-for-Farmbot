# Architecture and security

```text
FarmBot API ← companion Home Assistant integration ← Core service actions/events
                                                      ↑ Supervisor proxy
                                               FarmBot Vision app
                       Ingress UI ← FastAPI ← job lock ← CV engine → SQLite + artifacts
```

The companion integration owns authentication, token refresh, resource retrieval, image download, write validation, and FarmBot mutations. FarmBot Vision owns analysis, local observations, proposals, and operator review. The boundary is deliberately typed and versionable.

FastAPI serves a relative-link, `X-Ingress-Path`-aware interface on internal port 8099. An asynchronous event listener and optional low-frequency scheduler feed a single job lock. Images are fetched and released sequentially. CPU-heavy OpenCV work runs outside the event loop, with a 60-second timeout and resource checks between images.

The `ImageAnalysisEngine` abstraction isolates the classical implementation so a small ONNX or TFLite engine can be added later without changing jobs or persistence. No inference runtime ships in 0.1.0.

Security posture: protection stays enabled; the app is non-root and requests no host network, privileged capability, full access, Docker API, hardware, or Home Assistant configuration mount. AppArmor permits only required runtime libraries, networking, read-only resource counters, `/tmp`, and `/data`. The Supervisor token exists only in process memory and is never logged.

## Uncertain assumptions

- Integration service responses use Home Assistant's `?return_response` envelope.
- Image metadata `x,y` denotes the ground coordinate at image centre.
- Positive FarmBot X/Y map to image axes after the configured rotation.
- The integration returns JPEG, not PNG/HEIC, and limits its response to requested dimensions.
- Home Assistant supplies the local timezone to the app environment; otherwise the schedule follows container local time.

These assumptions are explicit so companion integration tests can lock them down.
