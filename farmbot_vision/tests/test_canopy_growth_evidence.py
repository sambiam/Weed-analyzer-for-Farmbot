"""Growth must be backed by real vegetation, not by the fixed margins alone."""

from __future__ import annotations

import math

import numpy as np

from farmbot_vision.canopy_radius import (
    MINIMUM_GROWTH_AGE_DAYS,
    MINIMUM_GROWTH_EVIDENCE_AREA_MM2,
    estimate_canopy_radius,
    growth_evidence_hold_reason,
    recommended_protection_radius_mm,
)


def test_sparse_speck_cannot_support_an_increase() -> None:
    reason = growth_evidence_hold_reason(12.0, 40)
    assert reason is not None
    assert "150 mm2" in reason


def test_established_canopy_area_supports_an_increase() -> None:
    assert growth_evidence_hold_reason(MINIMUM_GROWTH_EVIDENCE_AREA_MM2, 40) is None


def test_narrow_leaf_area_is_accepted_despite_low_circle_fill() -> None:
    # A 60 mm x 8 mm leaf fills under 5% of its own bounding circle but is
    # unambiguously a plant, so an area gate must let it through.
    assert growth_evidence_hold_reason(60.0 * 8.0, 40) is None


def test_recently_sown_plant_is_held_regardless_of_area() -> None:
    reason = growth_evidence_hold_reason(5000.0, MINIMUM_GROWTH_AGE_DAYS - 1)
    assert reason is not None
    assert "germination guard" in reason


def test_unknown_age_falls_through_to_the_area_gate() -> None:
    assert growth_evidence_hold_reason(5000.0, None) is None


def test_margin_alone_would_otherwise_propose_a_large_increase() -> None:
    """Reproduce the seedling case: a tiny mask, a 25 mm radius, 30 mm margins."""

    rng = np.random.default_rng(7)
    angles = rng.uniform(0, 2 * math.pi, size=600)
    distances = rng.uniform(24.0, 27.0, size=600)
    estimate = estimate_canopy_radius(
        distances,
        angles,
        current_radius_mm=25.0,
        protection_margin_mm=30.0,
    )
    assert estimate is not None
    proposed = recommended_protection_radius_mm(
        estimate,
        current_radius_mm=25.0,
        protection_margin_mm=30.0,
    )
    # Unguarded, the fixed margins push a ~26 mm mask well past the stored 25 mm.
    assert proposed > 50.0
    # Sixty pixels of vegetation is nowhere near enough to justify that.
    assert growth_evidence_hold_reason(20.0, 3) is not None
