# FarmBot Vision 4.3.2

Lightweight, experimental canopy measurement and safe FarmBot plant-radius recommendations. Open the app through Home Assistant Ingress after installation.

Set **Analysis resolution** (`640x480`, `960x720` default, or `1280x960`) in the app options; 960x720 is recommended for a 4 GB Raspberry Pi 4. Changing it requires an app restart.

The companion FarmBot integration must implement the service and event contract (`farmbot-vision-v2`) documented at repository level; the minimum compatible companion integration version is **2.13.0**. No FarmBot credential is accepted or stored by this app.

Upgrading between app versions only requires installing the new version and restarting the app. See [`CHANGELOG.md`](CHANGELOG.md) for what changed in each release, and close/reopen the Web UI tab after upgrading so Home Assistant creates a fresh Ingress session.

## Clear-soil-aware soil-height grid

The **Soil height** page can calculate a full measurement grid from a standard
spacing and maximum allowed deviation. Nominal centres start at half the
spacing from X=0 and Y=0. The preview lists unchanged, relocated, and skipped
points, including the nearest clear location or smaller clear-soil margin that
would make a skipped point usable. Accepting the preview recalculates it from
live FarmBot and Vision plants, weeds, zones, and point timestamps before any
movement begins. An optional age filter skips recent measurements for a chosen
number of days; leaving it off remeasures the complete grid. Plants and weeds
smaller than 15 mm, plus plants younger than 10 days, do not block a clear-soil
site. Custom coordinates update a nearby point automatically or create a new
soil-height point when no candidate is within 200 mm.

## Weed detection and local verifier training

The **Weed settings** page controls each stage independently:

- separate recall-first discovery and strict weed-extent colour controls;
- whole-weed leaf grouping, centre recovery, robust radial percentile and maximum span;
- the number of position-matched image observations required for review and automatic creation;
- independent automatic rejection and acceptance confidence thresholds;
- learned verifier shadow mode, enforcement, weighting and automatic-action requirements; and
- automatic FarmBot creation, verifier-confirmed repeated evidence for known-weed radius growth/removal, and rolling 24-hour radius-growth caps.

The Analysis page stores reviews as local training labels. Accepting a weed supplies a positive
example; **Crop**, **Reject as mulch/soil**, **Fungus/moss**, and **Hardware/other** supply hard
negative examples. Candidate crops, extracted visual features, labels and the trained JSON model
stay under the app's `/data` directory. Training runs locally from **Weed settings -> Train verifier
now** and can optionally run after every new label once the configured minimum number of positive
and negative examples has been reached.

Start with automatic creation disabled and the verifier in shadow mode. Validation precision and
recall are displayed in the app after training. Enforcing the verifier and lowering temporal or
confidence gates are explicit settings, so experienced users can progressively automate the
workflow without editing files or using a command line.

## Multi-image canopy radius

The app segments each source image independently, then aligns each plant's
ownership mask in calibrated, plant-centred garden coordinates. When a canopy
is clipped by an image boundary, the fused mask supplies one radial
measurement across all usable views. This is more reliable than measuring
each partial image independently or segmenting a stitched RGB panorama,
because segmentation retains the original pixels while fusion avoids
double-counted overlap and visible stitching seams.

Radius measurement uses a 72-sector supported boundary instead of the single
farthest mask pixel. The current FarmBot protection radius supplies the prior
canopy edge after configured margins are removed. Broad expansion beyond a
small growth band is clipped and cannot be written automatically, while a
narrow supported leaf can still establish the true outer edge. The resulting
protection radius slightly overestimates the accepted foliage edge by the
configured safety and calibration margins, which is useful for watering and
crop protection without turning nearby soil into plant canopy.

When the learned verifier is enabled, it also checks only newly extending
boundary regions. A trained crop category confirms growth; confident weed,
soil, moss and other non-crop results are removed; and uncertainty holds the
previous boundary for another observation. Known FarmBot weed points are
removed from new plant ownership before this check. This reuses the small local
logistic model and falls back to sector geometry when no suitable model exists,
so it remains practical on a Raspberry Pi 4.

The **Canopy fusion** page controls activation, minimum views, time window,
source-edge weighting, corroboration, radial percentile, angular coverage,
disagreement tolerance, diagnostics, and whether reliable fusion is required
for automatic radius changes. Conservative defaults can be relaxed from the
interface after field validation; manual review remains available throughout.
