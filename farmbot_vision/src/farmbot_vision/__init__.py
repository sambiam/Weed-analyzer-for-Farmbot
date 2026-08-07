"""FarmBot Vision application."""

__version__ = "4.2.2"
ALGORITHM_VERSION = "classical-boundary-recall-0.11.0"

# Version of the typed companion-integration contract this app speaks. v2 adds
# the returned-JPEG checksum, source/oriented/processed dimensions, resize
# scales and processed-image calibration.
CONTRACT_VERSION = "farmbot-vision-v2"

# Minimum companion FarmBot integration release that implements contract v2
# (returned-JPEG checksum, source/oriented/processed dimensions, resize scales
# and processed-image calibration). Matches the companion's semantic version.
MINIMUM_INTEGRATION_VERSION = "2.13.0"
