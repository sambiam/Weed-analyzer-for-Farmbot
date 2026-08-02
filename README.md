# FarmBot Vision

FarmBot Vision is an experimental Home Assistant app that measures the real canopy of known FarmBot plants and recommends protection radii that include the furthest accepted genuine leaf. It uses classical computer vision and is designed for a Raspberry Pi 4 without PyTorch, TensorFlow, or a frontend build toolchain.

Version: **3.22.1** · Architectures: **aarch64, amd64** · Home Assistant Core: **2026.7+** · Analysis resolution: **640x480 / 960x720 (default) / 1280x960** · Minimum companion integration: **2.2.0** (contract `farmbot-vision-v2`)

> Automatic destructive weeding must not rely solely on early experimental vision results. Review diagnostic overlays and keep FarmBot's other safety controls enabled.

## Install

1. In Home Assistant, open **Settings -> Apps -> App store -> Repositories**.
2. Add `https://github.com/sambiam/Weed-analyzer-for-Farmbot`.
3. Install **FarmBot Vision**.
4. Install a companion FarmBot custom integration version that implements the contract in [`docs/integration-contract.md`](docs/integration-contract.md).
5. Select its config entry, complete or verify calibration, and start in **Observe** mode.

FarmBot credentials remain in the companion integration. This app receives only short-lived access to Home Assistant through the documented Supervisor proxy and never stores `SUPERVISOR_TOKEN`.

See [app documentation](farmbot_vision/DOCS.md), [architecture](docs/architecture.md), and [development](docs/development.md).

## Status

Version 1.1.4 adds a review workflow for every analysis outcome (approve/reject, with certainty only gating automation), fixes spread-curve writes, and adds persistent opt-in weed detection with recommendation and automatic-creation modes. Version 0.5.0 rebuilt calibration around FarmBot's own coordinate model, added the composite photo-row calibration view, and made calibration persist across restarts in `/data`. Version 0.2.1 fixes duplicate-slash Home Assistant Ingress requests and keeps malformed individual vision events from reconnecting the WebSocket subscription. Version 0.2.0 added configurable analysis resolution (640x480 / 960x720 / 1280x960, default 960x720) and the revised **farmbot-vision-v2** image contract: returned-JPEG checksum, source/oriented/processed dimensions, resize scales, and processed-image calibration. Calibration always corresponds to the exact processed pixels; without valid calibration the app produces pixel-only diagnostics and refuses all writes. On top of 0.1.0's sequential analysis, conservative multi-plant ownership, temporal-mask hooks, manual calibration, safe individual-radius recommendations, monotonic crop curve proposals, SQLite history, ingress UI, triggers and resource gates. Automatic curve writes remain deliberately unavailable.

See [`farmbot_vision/CHANGELOG.md`](farmbot_vision/CHANGELOG.md) for the full version history.

On a 4 GB Raspberry Pi 4 use the default 960x720 with the free-memory gate at 512 MB and CPU gate at 80%. Relative pixel workload: 640x480 = 1x, 960x720 = 2.25x, 1280x960 = 4x.

License: MIT.
