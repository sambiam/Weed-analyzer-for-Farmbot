# FarmBot Vision

FarmBot Vision is an experimental Home Assistant app that measures the real canopy of known FarmBot plants and recommends protection radii that include the furthest accepted genuine leaf. It uses classical computer vision and is designed for a Raspberry Pi 4 without PyTorch, TensorFlow, or a frontend build toolchain.

Version: **0.1.0** · Architectures: **aarch64, amd64** · Home Assistant Core: **2026.7+**

> Automatic destructive weeding must not rely solely on early experimental vision results. Review diagnostic overlays and keep FarmBot's other safety controls enabled.

## Install

1. In Home Assistant, open **Settings → Apps → App store → Repositories**.
2. Add `https://github.com/sambiam/Weed-analyzer-for-Farmbot`.
3. Install **FarmBot Vision**.
4. Install a companion FarmBot custom integration version that implements the contract in [`docs/integration-contract.md`](docs/integration-contract.md).
5. Select its config entry, complete or verify calibration, and start in **Observe** mode.

FarmBot credentials remain in the companion integration. This app receives only short-lived access to Home Assistant through the documented Supervisor proxy and never stores `SUPERVISOR_TOKEN`.

See [app documentation](farmbot_vision/DOCS.md), [architecture](docs/architecture.md), and [development](docs/development.md).

## Status

Version 0.1.0 provides sequential image analysis, conservative multi-plant ownership, temporal-mask hooks, manual calibration, safe individual-radius recommendations, monotonic crop curve proposals, SQLite history, ingress UI, event/manual/daily triggers, and resource gates. Automatic curve writes are deliberately unavailable.

License: MIT.
