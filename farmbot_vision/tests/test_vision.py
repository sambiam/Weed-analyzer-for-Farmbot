from __future__ import annotations

from datetime import UTC, datetime

import cv2
import numpy as np
from conftest import encode_jpeg, jpeg

from farmbot_vision.models import Calibration, Decision, KnownWeedSeed, PlantSeed
from farmbot_vision.vision import (
    ClassicalVisionEngine,
    _absence_confidence,
    decode_jpeg,
    register_translation,
)

NOW = datetime(2026, 2, 1, tzinfo=UTC)


def analyse(shapes, seed, calibration, previous=None, seeds=None):
    return ClassicalVisionEngine().analyse(
        jpeg(shapes), 9, NOW, seeds or [seed], calibration, previous or {}
    )


class _BoundaryVerifier:
    """Small tri-state verifier fixture with a trained crop category head."""

    available = True
    model = {"class_heads": {"crop": {"weights": [1]}}}

    def __init__(self, weed_probability: float, explanations: list[tuple[str, float]]):
        self.weed_probability = weed_probability
        self.explanations = explanations

    def predict(self, _features: dict[str, float]) -> float:
        return self.weed_probability

    def explain(self, _features: dict[str, float]) -> list[tuple[str, float]]:
        return self.explanations


def test_circular_plant_without_weeds(seed, calibration):
    result = analyse([("circle", ((160, 120), 35))], seed, calibration)
    measurement = result.measurements[0]
    assert 33 <= measurement.maximum_accepted_canopy_radius_mm <= 37
    assert measurement.recommended_protection_radius_mm >= 63
    assert measurement.confidence > 0.7


def test_absence_confidence_is_bounded_for_tiny_fully_covered_cores():
    assert _absence_confidence(1.0) == 0.05
    assert 0.05 <= _absence_confidence(0.0) <= 0.98


def test_radius_is_measured_in_bed_mm_after_anisotropic_rotation():
    image = np.zeros((180, 180, 3), np.uint8)
    cv2.ellipse(image, (90, 90), (30, 60), 0, 0, 360, (20, 210, 30), -1)
    calibration = Calibration(
        source="manual",
        pixels_per_mm_x=2,
        pixels_per_mm_y=1,
        rotation_degrees=90,
        uncertainty_mm=0,
    )
    seed = PlantSeed(
        plant_id=1,
        crop_slug="lettuce",
        center_px=(90, 90),
        current_radius_mm=25,
    )

    result = ClassicalVisionEngine(
        safety_margin_mm=0,
        calibration_uncertainty_mm=0,
    ).analyse(encode_jpeg(image), 9, NOW, [seed], calibration)

    assert 27 <= result.measurements[0].maximum_accepted_canopy_radius_mm <= 33


def test_largest_genuine_leaf_is_not_excluded(seed, calibration):
    result = analyse(
        [("circle", ((160, 120), 30)), ("line", ((160, 120), (250, 120), 10))], seed, calibration
    )
    measurement = result.measurements[0]
    assert measurement.maximum_accepted_canopy_radius_mm > 85
    assert measurement.maximum_accepted_canopy_radius_mm > measurement.typical_canopy_radius_mm


def test_washed_out_leaf_is_recovered_without_claiming_an_isolated_weed(seed, calibration):
    image = np.zeros((240, 320, 3), np.uint8)
    cv2.circle(image, (160, 120), 25, (20, 210, 30), -1)
    # This pale, low-saturation leaf fails the established strong-green
    # classifier but remains green-tinted and attached to the crop canopy.
    cv2.line(image, (160, 120), (260, 120), (190, 205, 195), 8)
    cv2.circle(image, (285, 40), 12, (20, 210, 30), -1)

    result = ClassicalVisionEngine().analyse(encode_jpeg(image), 9, NOW, [seed], calibration, {})
    measurement = result.measurements[0]
    assert measurement.maximum_accepted_canopy_radius_mm > 90

    ownership = cv2.imdecode(np.frombuffer(result.ownership_mask, np.uint8), cv2.IMREAD_UNCHANGED)
    assert ownership[120, 250] == 1
    assert ownership[40, 285] == 0


def test_isolated_weed_does_not_inflate_radius(seed, calibration):
    result = analyse([("circle", ((160, 120), 30)), ("circle", ((285, 40), 10))], seed, calibration)
    assert result.measurements[0].maximum_accepted_canopy_radius_mm < 35


def test_weed_close_to_crop_is_conservative(seed, calibration):
    result = analyse([("circle", ((160, 120), 25)), ("circle", ((235, 120), 9))], seed, calibration)
    assert result.measurements[0].maximum_accepted_canopy_radius_mm < 35
    assert result.measurements[0].decision == Decision.UNCERTAIN


def test_hairline_connected_weed_does_not_inflate_crop_radius(seed, calibration):
    result = analyse(
        [
            ("circle", ((160, 120), 25)),
            ("line", ((185, 120), (264, 120), 1)),
            ("circle", ((275, 120), 11)),
        ],
        seed,
        calibration,
    )
    assert result.measurements[0].maximum_accepted_canopy_radius_mm < 40


def test_broad_attached_green_background_cannot_create_a_radius_jump(seed, calibration):
    """A soil/moss block touching the plant must not become its outer edge."""

    result = analyse(
        [
            ("circle", ((160, 120), 35)),
            ("rect", ((185, 35), (315, 205))),
        ],
        seed,
        calibration,
    )
    measurement = result.measurements[0]
    ownership = cv2.imdecode(np.frombuffer(result.ownership_mask, np.uint8), cv2.IMREAD_UNCHANGED)

    assert measurement.maximum_accepted_canopy_radius_mm < 55
    assert measurement.confidence <= 0.74
    assert "broad outer-mask expansion was rejected" in measurement.reason
    assert ownership[120, 160] == 1
    assert ownership[120, 300] == 0


def test_small_growth_is_measured_from_previous_protection_radius(seed, calibration):
    result = analyse([("circle", ((160, 120), 40))], seed, calibration)
    measurement = result.measurements[0]

    # The stored 60 mm radius represents the earlier ~30 mm canopy plus the
    # configured 20 + 10 mm margins.  A new 40 mm edge is a small real growth.
    assert 38 <= measurement.maximum_accepted_canopy_radius_mm <= 42
    assert 68 <= measurement.recommended_protection_radius_mm <= 72


def test_radius_encloses_canopy_when_recorded_center_is_slightly_offset(calibration):
    seed = PlantSeed(
        plant_id=1,
        crop_slug="lettuce",
        center_px=(160, 120),
        current_radius_mm=60,
    )
    result = analyse([("circle", ((175, 120), 30))], seed, calibration)

    # Radius remains relative to FarmBot's stored centre, so the far leaf edge
    # is about 15 + 30 mm away even before a centre correction is approved.
    assert 43 <= result.measurements[0].maximum_accepted_canopy_radius_mm <= 47


def test_clean_mask_can_reduce_a_previously_overestimated_radius(calibration):
    seed = PlantSeed(
        plant_id=1,
        crop_slug="lettuce",
        center_px=(160, 120),
        current_radius_mm=180,
    )
    result = analyse([("circle", ((160, 120), 35))], seed, calibration)
    measurement = result.measurements[0]

    assert 33 <= measurement.maximum_accepted_canopy_radius_mm <= 37
    assert measurement.recommended_protection_radius_mm < 70


def test_radius_reduction_never_cuts_inside_a_supported_leaf_mask(calibration):
    seed = PlantSeed(
        plant_id=1,
        crop_slug="lettuce",
        center_px=(160, 120),
        current_radius_mm=180,
    )
    result = analyse(
        [
            ("circle", ((160, 120), 35)),
            ("line", ((190, 120), (300, 120), 9)),
        ],
        seed,
        calibration,
    )
    measurement = result.measurements[0]

    assert measurement.maximum_accepted_canopy_radius_mm > 135
    assert measurement.recommended_protection_radius_mm >= 165
    assert measurement.recommended_protection_radius_mm < seed.current_radius_mm


def test_known_weed_is_removed_from_new_plant_boundary(calibration):
    from farmbot_vision.weed_settings import WeedSettings

    seed = PlantSeed(
        plant_id=1,
        crop_slug="lettuce",
        center_px=(160, 120),
        current_radius_mm=60,
    )
    image = jpeg(
        [
            ("circle", ((160, 120), 25)),
            ("line", ((160, 120), (250, 120), 10)),
            ("circle", ((250, 120), 12)),
        ]
    )
    result = ClassicalVisionEngine().analyse(
        image,
        9,
        NOW,
        [seed],
        calibration,
        {},
        WeedSettings(enabled=False),
        [KnownWeedSeed(weed_id=91, center_px=(250, 120), radius_mm=15)],
    )
    ownership = cv2.imdecode(np.frombuffer(result.ownership_mask, np.uint8), cv2.IMREAD_UNCHANGED)

    assert result.measurements[0].maximum_accepted_canopy_radius_mm < 35
    assert ownership[120, 250] == 0
    assert result.boundary_verifier_stats["known_weed_regions"] == 1


def test_boundary_verifier_accepts_confirmed_crop_growth(seed, calibration):
    from farmbot_vision.weed_settings import WeedSettings

    verifier = _BoundaryVerifier(0.05, [("crop", 0.9), ("weed", 0.1)])
    settings = WeedSettings(
        enabled=True,
        visual_verifier_enabled=True,
        visual_verifier_shadow_mode=False,
    )
    result = ClassicalVisionEngine(weed_verifier=verifier).analyse(
        jpeg([("circle", ((160, 120), 40))]),
        9,
        NOW,
        [seed],
        calibration,
        {},
        settings,
    )

    assert 38 <= result.measurements[0].maximum_accepted_canopy_radius_mm <= 42
    assert result.boundary_verifier_stats["crop_accepted"] == 1


def test_crop_context_preserves_uncertain_growth_inside_existing_radius(seed, calibration):
    from farmbot_vision.weed_settings import WeedSettings

    verifier = _BoundaryVerifier(0.4, [("crop", 0.55), ("soil", 0.45)])
    settings = WeedSettings(
        enabled=True,
        visual_verifier_enabled=True,
        visual_verifier_shadow_mode=False,
    )
    result = ClassicalVisionEngine(weed_verifier=verifier).analyse(
        jpeg([("circle", ((160, 120), 40))]),
        9,
        NOW,
        [seed],
        calibration,
        {},
        settings,
    )
    measurement = result.measurements[0]

    assert 38 <= measurement.maximum_accepted_canopy_radius_mm <= 42
    assert result.boundary_verifier_stats["crop_context_accepted"] == 1
    assert result.boundary_verifier_stats["uncertain_held"] == 0


def test_boundary_weed_reaches_weed_workflow_despite_crop_exclusion(seed, calibration):
    from farmbot_vision.weed_settings import WeedSettings

    verifier = _BoundaryVerifier(0.97, [("weed", 0.95), ("crop", 0.05)])
    settings = WeedSettings(
        enabled=True,
        maximum_area_mm2=10_000,
        minimum_confidence=0.4,
        visual_verifier_enabled=True,
        visual_verifier_shadow_mode=False,
    )
    result = ClassicalVisionEngine(weed_verifier=verifier).analyse(
        jpeg(
            [
                ("circle", ((160, 120), 25)),
                ("line", ((160, 120), (250, 120), 10)),
                ("circle", ((250, 120), 12)),
            ]
        ),
        9,
        NOW,
        [seed],
        calibration,
        {},
        settings,
    )

    assert result.measurements[0].maximum_accepted_canopy_radius_mm < 35
    assert result.boundary_verifier_stats["weed_rejected"] == 1
    assert result.weeds, "a verifier-confirmed boundary weed must remain reviewable"


def test_shadow_boundary_verifier_surfaces_weed_without_changing_crop(seed, calibration):
    """Shadow mode may rescue review evidence while leaving crop geometry untouched."""

    from farmbot_vision.weed_settings import WeedSettings

    verifier = _BoundaryVerifier(0.97, [("weed", 0.95), ("crop", 0.05)])
    settings = WeedSettings(
        enabled=True,
        minimum_confidence=0.79,
        visual_verifier_enabled=True,
        visual_verifier_shadow_mode=True,
    )
    result = ClassicalVisionEngine(weed_verifier=verifier).analyse(
        jpeg(
            [
                ("circle", ((160, 120), 25)),
                ("line", ((160, 120), (250, 120), 10)),
                ("circle", ((250, 120), 12)),
            ]
        ),
        9,
        NOW,
        [seed],
        calibration,
        {},
        settings,
    )

    assert result.boundary_verifier_stats["shadow_scored"] == 1
    assert result.measurements[0].maximum_accepted_canopy_radius_mm > 80
    assert result.weeds, "a high-confidence crop-owned weed must reach review in shadow mode"


def test_overlapping_crops_keep_reviewable_nearest_seed_ownership(calibration):
    seeds = [
        PlantSeed(plant_id=1, crop_slug="lettuce", center_px=(135, 120), current_radius_mm=50),
        PlantSeed(plant_id=2, crop_slug="lettuce", center_px=(185, 120), current_radius_mm=50),
    ]
    result = analyse(
        [("circle", ((135, 120), 40)), ("circle", ((185, 120), 40))],
        seeds[0],
        calibration,
        seeds=seeds,
    )
    assert len(result.measurements) == 2
    assert all(not item.ambiguous for item in result.measurements)
    assert all(item.confidence >= 0.6 for item in result.measurements)


def test_disconnected_leaf_accepted_from_previous_mask(seed, calibration):
    previous = np.zeros((240, 320), np.uint8)
    cv2.circle(previous, (235, 120), 12, 255, -1)
    result = analyse(
        [("circle", ((160, 120), 25)), ("circle", ((235, 120), 12))],
        seed,
        calibration,
        {1: previous},
    )
    assert result.measurements[0].maximum_accepted_canopy_radius_mm > 80
    assert not result.measurements[0].ambiguous


def test_sudden_disconnected_region_is_uncertain(seed, calibration):
    result = analyse(
        [("circle", ((160, 120), 25)), ("circle", ((235, 120), 12))], seed, calibration
    )
    assert result.measurements[0].ambiguous
    assert result.measurements[0].maximum_accepted_canopy_radius_mm < 35


def test_green_irrigation_line_and_noise_are_rejected(seed, calibration):
    shapes = [("circle", ((160, 120), 25)), ("rect", ((5, 20), (310, 24)))]
    result = analyse(shapes, seed, calibration)
    assert result.measurements[0].maximum_accepted_canopy_radius_mm < 35


def test_empty_in_frame_centre_is_an_absence_measurement(seed, calibration):
    result = analyse([], seed, calibration)

    assert result.skipped == {}
    assert len(result.measurements) == 1
    measurement = result.measurements[0]
    assert measurement.vegetation_absent is True
    assert measurement.typical_canopy_radius_mm == 0
    assert measurement.maximum_accepted_canopy_radius_mm == 0
    assert measurement.recommended_protection_radius_mm == 0


def test_empty_center_with_outer_vegetation_recommends_removal_and_center_move(seed, calibration):
    result = analyse([("circle", ((220, 120), 16))], seed, calibration)
    measurement = result.measurements[0]
    assert measurement.vegetation_absent is True
    assert measurement.center_misaligned is True
    assert measurement.recommended_center_px is not None


def test_unowned_component_becomes_weed_only_when_enabled(seed, calibration):
    from farmbot_vision.weed_settings import WeedSettings

    engine = ClassicalVisionEngine()
    result = engine.analyse(
        jpeg([("circle", ((160, 120), 25)), ("circle", ((285, 40), 10))]),
        9,
        NOW,
        [seed],
        calibration,
        {},
        WeedSettings(
            enabled=True,
            plant_exclusion_margin_mm=10,
            minimum_area_mm2=25,
            minimum_confidence=0.7,
        ),
    )
    assert len(result.weeds) == 1
    assert result.weeds[0].center_px[0] > 270
    assert result.weed_review_jpeg is not None


def test_nearby_leaf_fragments_become_one_full_weed(seed, calibration):
    from farmbot_vision.weed_settings import WeedSettings

    result = ClassicalVisionEngine().analyse(
        jpeg(
            [
                ("circle", ((160, 120), 25)),
                # Four separate segmented leaves belonging to one rosette.
                ("circle", ((268, 45), 7)),
                ("circle", ((285, 45), 7)),
                ("circle", ((276, 30), 7)),
                ("circle", ((276, 60), 7)),
            ]
        ),
        9,
        NOW,
        [seed],
        calibration,
        {},
        WeedSettings(
            enabled=True,
            plant_exclusion_margin_mm=10,
            minimum_area_mm2=25,
            minimum_confidence=0.65,
        ),
    )

    assert len(result.weeds) == 1
    weed = result.weeds[0]
    assert weed.area_mm2 > 450
    assert 270 < weed.center_px[0] < 282
    assert 40 < weed.center_px[1] < 50
    assert weed.radius_mm >= 22


def test_separate_weeds_beyond_joining_gap_remain_separate(seed, calibration):
    from farmbot_vision.weed_settings import WeedSettings

    result = ClassicalVisionEngine().analyse(
        jpeg(
            [
                ("circle", ((160, 120), 25)),
                ("circle", ((250, 35), 8)),
                ("circle", ((290, 35), 8)),
            ]
        ),
        9,
        NOW,
        [seed],
        calibration,
        {},
        WeedSettings(
            enabled=True,
            plant_exclusion_margin_mm=10,
            minimum_area_mm2=25,
            minimum_confidence=0.65,
        ),
    )

    assert len(result.weeds) == 2


def test_crop_exclusion_protects_entire_outer_leaf_not_just_circle_interior(seed, calibration):
    from farmbot_vision.weed_settings import WeedSettings

    result = ClassicalVisionEngine().analyse(
        jpeg(
            [
                ("circle", ((160, 120), 25)),
                # A segmented crop leaf straddles the circular exclusion
                # boundary. Its outer edge must not become a tiny weed.
                ("circle", ((230, 120), 15)),
            ]
        ),
        9,
        NOW,
        [seed],
        calibration,
        {},
        WeedSettings(
            enabled=True,
            crop_support_radius_multiplier=0.5,
            crop_support_extra_mm=0,
            plant_exclusion_margin_mm=40,
            minimum_area_mm2=12,
            minimum_confidence=0.65,
        ),
    )

    assert result.weeds == []


def test_known_neighbouring_plant_is_not_a_weed_but_small_bottom_weed_is(calibration):
    from farmbot_vision.weed_settings import WeedSettings

    seeds = [
        PlantSeed(
            plant_id=1,
            crop_slug="spinach",
            center_px=(110, 115),
            current_radius_mm=70,
        ),
        PlantSeed(
            plant_id=2,
            crop_slug="marjoram",
            center_px=(245, 90),
            current_radius_mm=30,
        ),
    ]
    result = ClassicalVisionEngine().analyse(
        jpeg(
            [
                ("circle", ((110, 115), 48)),
                ("circle", ((245, 90), 24)),
                ("circle", ((260, 210), 8)),
            ]
        ),
        9,
        NOW,
        seeds,
        calibration,
        {},
        WeedSettings(
            enabled=True,
            plant_exclusion_margin_mm=10,
            minimum_area_mm2=12,
            minimum_confidence=0.7,
        ),
    )
    assert len(result.weeds) == 1
    assert result.weeds[0].center_px[1] > 195
    ownership = cv2.imdecode(np.frombuffer(result.ownership_mask, np.uint8), cv2.IMREAD_UNCHANGED)
    assert ownership[90, 245] == 2


def test_edge_plant_is_low_confidence_reviewable_not_skipped_or_removed(calibration):
    edge_seed = PlantSeed(
        plant_id=7,
        crop_slug="lettuce",
        center_px=(1, 120),
        current_radius_mm=60,
    )
    result = analyse([], edge_seed, calibration)

    assert result.skipped == {}
    assert len(result.measurements) == 1
    assert result.measurements[0].vegetation_absent is False
    assert result.measurements[0].decision == Decision.UNCERTAIN
    assert result.measurements[0].confidence <= 0.2


def test_fully_out_of_frame_plant_is_reviewable_not_skipped(calibration):
    far_seed = PlantSeed(
        plant_id=11,
        crop_slug="lettuce",
        center_px=(-5000, -5000),
        current_radius_mm=60,
    )
    result = analyse([], far_seed, calibration)

    assert result.skipped == {}
    assert len(result.measurements) == 1
    measurement = result.measurements[0]
    assert measurement.plant_id == 11
    assert measurement.decision == Decision.UNCERTAIN
    assert measurement.confidence <= 0.1
    assert measurement.applied is False
    assert measurement.current_radius_mm == measurement.recommended_protection_radius_mm


def test_overlay_and_binary_masks_explain_vegetation_ownership(calibration):
    seeds = [
        PlantSeed(plant_id=1, crop_slug="lettuce", center_px=(100, 120), current_radius_mm=30),
        PlantSeed(plant_id=2, crop_slug="lettuce", center_px=(220, 120), current_radius_mm=30),
    ]
    result = analyse(
        [("circle", ((100, 120), 20)), ("circle", ((220, 120), 20))],
        seeds[0],
        calibration,
        seeds=seeds,
    )

    assert result.mask is not None
    assert result.ownership_mask is not None
    assert result.overlay_jpeg is not None
    vegetation = cv2.imdecode(np.frombuffer(result.mask, np.uint8), cv2.IMREAD_GRAYSCALE)
    ownership = cv2.imdecode(np.frombuffer(result.ownership_mask, np.uint8), cv2.IMREAD_UNCHANGED)
    overlay = cv2.imdecode(np.frombuffer(result.overlay_jpeg, np.uint8), cv2.IMREAD_COLOR)
    assert vegetation is not None and ownership is not None and overlay is not None
    assert vegetation[120, 100] > 0 and vegetation[120, 220] > 0
    assert ownership[120, 100] == 1
    assert ownership[120, 220] == 2
    # The two plants receive distinct ownership tints in the composite, rather
    # than the overlay being only geometry drawn on the original green pixels.
    assert np.linalg.norm(overlay[120, 100].astype(int) - overlay[120, 220].astype(int)) > 25


def test_camera_translation_registration():
    previous = np.zeros((120, 160), np.uint8)
    cv2.circle(previous, (70, 60), 15, 255, -1)
    transform = np.float32([[1, 0, 6], [0, 1, -4]])
    current = cv2.warpAffine(previous, transform, (160, 120))
    dx, dy, response = register_translation(previous, current)
    assert abs(dx - 6) < 0.5
    assert abs(dy + 4) < 0.5
    assert response > 0.5


def test_decode_keeps_processed_resolution_and_caps_at_ceiling():
    # The integration already resized to the processed size; the app keeps it.
    processed = np.zeros((960, 1280, 3), np.uint8)
    cv2.circle(processed, (640, 480), 200, (0, 255, 0), -1)
    ok, encoded = cv2.imencode(".jpg", processed)
    assert ok
    assert decode_jpeg(encoded.tobytes()).shape[:2] == (960, 1280)

    # Anything above the 1280x960 ceiling is defensively downscaled.
    oversized = np.zeros((1200, 1600, 3), np.uint8)
    ok, encoded = cv2.imencode(".jpg", oversized)
    assert ok
    decoded = decode_jpeg(encoded.tobytes())
    assert decoded.shape[1] <= 1280 and decoded.shape[0] <= 960


class _StubVerifier:
    """Stands in for a trained WeedVisualVerifier with a fixed score."""

    def __init__(self, score: float | None, available: bool = True):
        self.score = score
        self.available = available
        self.seen: list[dict[str, float]] = []

    def predict(self, features: dict[str, float]) -> float | None:
        self.seen.append(features)
        return self.score


def _weed_scene(verifier, **overrides):
    from farmbot_vision.weed_settings import WeedSettings

    settings = {
        "enabled": True,
        "plant_exclusion_margin_mm": 10,
        "minimum_area_mm2": 25,
        "minimum_confidence": 0.7,
        "visual_verifier_enabled": True,
        "visual_verifier_shadow_mode": False,
        "visual_verifier_minimum_confidence": 0.6,
    }
    settings.update(overrides)
    calibration = Calibration(
        source="manual", pixels_per_mm_x=1, pixels_per_mm_y=1, uncertainty_mm=10
    )
    seed = PlantSeed(
        plant_id=1,
        crop_slug="lettuce",
        center_px=(160, 120),
        current_radius_mm=60,
        planted_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    return ClassicalVisionEngine(weed_verifier=verifier).analyse(
        jpeg([("circle", ((160, 120), 25)), ("circle", ((285, 40), 10))]),
        9,
        NOW,
        [seed],
        calibration,
        {},
        WeedSettings(**settings),
    )


def test_trained_verifier_replaces_rather_than_dilutes_the_heuristic_score():
    verifier = _StubVerifier(0.72)

    result = _weed_scene(verifier)

    assert len(result.weeds) == 1
    weed = result.weeds[0]
    # Exactly the verifier's number: blending it with the heuristic used to
    # compress and decalibrate the score every downstream threshold reads.
    assert weed.confidence == 0.72
    assert weed.verifier_confidence == 0.72
    # The heuristic is still recorded so the two can be compared in review.
    assert weed.heuristic_confidence > 0


def test_verifier_rejection_drops_the_candidate():
    result = _weed_scene(_StubVerifier(0.4), visual_verifier_minimum_confidence=0.6)

    assert result.weeds == []


def _known_weed_scene(score: float, *, shadow: bool = False, vegetation: bool = True):
    from farmbot_vision.weed_settings import WeedSettings

    image = np.zeros((220, 320, 3), np.uint8)
    if vegetation:
        cv2.circle(image, (250, 105), 12, (20, 210, 30), -1)
    calibration = Calibration(
        source="manual", pixels_per_mm_x=1, pixels_per_mm_y=1, uncertainty_mm=0
    )
    return ClassicalVisionEngine(weed_verifier=_StubVerifier(score)).analyse(
        encode_jpeg(image),
        9,
        NOW,
        [],
        calibration,
        {},
        WeedSettings(
            enabled=True,
            minimum_confidence=0.3,
            visual_verifier_enabled=True,
            visual_verifier_shadow_mode=shadow,
            visual_verifier_rejection_confidence=0.45,
            visual_verifier_acceptance_confidence=0.85,
        ),
        [KnownWeedSeed(weed_id=91, center_px=(250, 105), radius_mm=20)],
    )


def test_known_weed_presence_comes_from_the_enforcing_verifier():
    result = _known_weed_scene(0.91)

    assert len(result.weeds) == 1
    assert result.weeds[0].features["known_weed_id"] == 91
    observation = result.known_weed_observations[0]
    assert observation.status == "present"
    assert observation.confidence == 0.91
    assert observation.verifier_evaluated is True


def test_rejected_known_weed_candidate_is_explicit_absence_evidence():
    result = _known_weed_scene(0.12)

    assert result.weeds == []
    observation = result.known_weed_observations[0]
    assert observation.status == "absent"
    assert observation.confidence == 0.88
    assert observation.verifier_evaluated is True


def test_shadow_verifier_cannot_declare_a_known_weed_absent():
    observation = _known_weed_scene(0.12, shadow=True).known_weed_observations[0]

    assert observation.status == "inconclusive"


def test_empty_unobscured_known_weed_region_is_explicit_visual_absence():
    observation = _known_weed_scene(0.91, vegetation=False).known_weed_observations[0]

    assert observation.status == "absent"
    assert observation.confidence == 0.95
    assert observation.verifier_evaluated is False


def test_shadow_mode_scores_without_deciding():
    verifier = _StubVerifier(0.05)

    result = _weed_scene(verifier, visual_verifier_shadow_mode=True)

    assert len(result.weeds) == 1
    weed = result.weeds[0]
    # Scored and recorded, but the heuristic still owns the decision.
    assert weed.verifier_confidence == 0.05
    assert weed.confidence == weed.heuristic_confidence


def test_shadow_verifier_rescues_pale_seedling_hidden_by_high_review_threshold(calibration):
    """A review-band verifier result must not be discarded by the heuristic."""

    from farmbot_vision.weed_settings import WeedSettings

    image = np.zeros((220, 320, 3), np.uint8)
    pale = tuple(
        int(value) for value in cv2.cvtColor(np.uint8([[[60, 15, 130]]]), cv2.COLOR_HSV2BGR)[0, 0]
    )
    cv2.circle(image, (245, 105), 5, pale, -1)
    verifier = _StubVerifier(0.70)
    settings = WeedSettings(
        enabled=True,
        minimum_confidence=0.79,
        visual_verifier_enabled=True,
        visual_verifier_shadow_mode=True,
        visual_verifier_rejection_confidence=0.45,
        visual_verifier_acceptance_confidence=0.85,
        strong_green_minimum_saturation=20,
        strong_green_minimum_excess_green=15,
        candidate_minimum_saturation=15,
        candidate_minimum_excess_green=2,
    )

    result = ClassicalVisionEngine(weed_verifier=verifier).analyse(
        encode_jpeg(image), 9, NOW, [], calibration, {}, settings
    )

    assert len(result.weeds) == 1
    assert verifier.seen, "the pale seedling must reach the verifier"
    assert result.weeds[0].heuristic_confidence < settings.minimum_confidence
    assert result.weeds[0].confidence == 0.70
    assert result.weed_candidate_stats["verifier_rescued"] == 1


def test_verifier_shape_gates_do_not_veto_uniformly_pale_rosette(calibration):
    """Strong-green purity is a verifier feature, never a pre-verifier veto."""

    from farmbot_vision.weed_settings import WeedSettings

    image = np.zeros((220, 320, 3), np.uint8)
    pale = tuple(
        int(value) for value in cv2.cvtColor(np.uint8([[[60, 15, 130]]]), cv2.COLOR_HSV2BGR)[0, 0]
    )
    for center in ((245, 105), (263, 105), (254, 90), (254, 120)):
        cv2.circle(image, center, 7, pale, -1)
    verifier = _StubVerifier(0.90)

    result = ClassicalVisionEngine(weed_verifier=verifier).analyse(
        encode_jpeg(image),
        9,
        NOW,
        [],
        calibration,
        {},
        WeedSettings(
            enabled=True,
            minimum_confidence=0.79,
            visual_verifier_enabled=True,
            visual_verifier_shadow_mode=False,
            strong_green_minimum_saturation=45,
            strong_green_minimum_excess_green=20,
            minimum_green_purity=0.10,
        ),
    )

    assert len(result.weeds) == 1
    assert verifier.seen, "pale leaves must be scored instead of silently shape-rejected"
    assert result.weed_candidate_stats["shape"] == 0


def test_untrained_verifier_falls_back_to_the_heuristic():
    result = _weed_scene(_StubVerifier(None, available=False))

    assert len(result.weeds) == 1
    weed = result.weeds[0]
    assert weed.verifier_confidence is None
    assert weed.confidence == weed.heuristic_confidence


def test_distance_to_the_nearest_plant_reaches_the_verifier():
    verifier = _StubVerifier(0.9)

    _weed_scene(verifier)

    assert verifier.seen
    features = verifier.seen[0]
    # The weed sits ~160mm from the single seed at 1 px/mm, well inside the
    # 300mm saturation distance, so the feature must be informative.
    assert 0.3 < features["distance_to_plant"] < 0.8


def test_no_shape_setting_can_veto_a_candidate_before_the_verifier_scores_it():
    """Rejecting a candidate here is invisible, so it must not be possible.

    A candidate dropped by the colour/shape gates is never scored, stored,
    reviewed or labelled, so the mistake cannot be noticed or corrected. Every
    gate is clamped to a recall ceiling; the verifier makes the decision.
    """
    impossible = {
        "minimum_circularity": 1.0,
        "minimum_solidity": 1.0,
        "minimum_green_purity": 1.0,
        "maximum_aspect_ratio": 1.0,
        "minimum_area_mm2": 9_000,
        "candidate_recall_boost": 1.0,
    }
    verifier = _StubVerifier(0.9)

    result = _weed_scene(verifier, **impossible)

    assert len(result.weeds) == 1
    assert verifier.seen, "the verifier must still be asked about the candidate"


def test_shadow_mode_also_hands_every_candidate_to_the_verifier():
    """Shadow mode exists to collect candidates worth labelling.

    Holding the gates at the user's values there starved the very stage that is
    meant to be gathering training examples.
    """
    verifier = _StubVerifier(0.9)

    result = _weed_scene(
        verifier,
        minimum_circularity=1.0,
        minimum_solidity=1.0,
        visual_verifier_shadow_mode=True,
    )

    assert len(verifier.seen) == 1
    # Shadow mode still lets the heuristic own the accept/reject decision.
    assert len(result.weeds) == 1
    assert result.weeds[0].confidence == result.weeds[0].heuristic_confidence


def test_candidate_above_configured_maximum_area_is_still_reviewable():
    """Area is evidence and an auto guard, never a pre-review veto."""
    verifier = _StubVerifier(0.9)

    result = _weed_scene(verifier, maximum_area_mm2=10)

    assert len(result.weeds) == 1
    assert verifier.seen
    assert result.weed_candidate_stats["oversized_scored"] >= 1
    assert result.weeds[0].features["configured_maximum_area_exceeded"] == 1


def test_oversized_candidate_is_reviewable_before_verifier_is_trained():
    result = _weed_scene(_StubVerifier(None, available=False), maximum_area_mm2=10)

    assert len(result.weeds) == 1
    assert result.weeds[0].features["configured_maximum_area_exceeded"] == 1


def test_configured_colour_mask_recovers_muted_weed_for_verifier(calibration):
    """The UI colour settings must act before candidate verification."""
    from farmbot_vision.weed_settings import WeedSettings

    image = np.zeros((240, 320, 3), np.uint8)
    # HSV(60, 20, 100): green-biased, but too desaturated for the established
    # crop mask's fixed saturation >22 rule.  It represents shaded/washed leaf
    # pixels like those in the field photos.
    muted_green = tuple(
        int(value) for value in cv2.cvtColor(np.uint8([[[60, 20, 100]]]), cv2.COLOR_HSV2BGR)[0, 0]
    )
    cv2.circle(image, (250, 100), 12, muted_green, -1)
    verifier = _StubVerifier(0.9)
    settings = WeedSettings(
        enabled=True,
        minimum_confidence=0.3,
        visual_verifier_enabled=True,
        visual_verifier_shadow_mode=False,
        visual_verifier_minimum_confidence=0.6,
        strong_green_minimum_saturation=18,
        strong_green_minimum_excess_green=10,
    )

    result = ClassicalVisionEngine(weed_verifier=verifier).analyse(
        encode_jpeg(image), 9, NOW, [], calibration, {}, settings
    )

    assert len(result.weeds) == 1
    assert verifier.seen, "muted configured foliage must reach verification"


def test_default_discovery_recovers_pale_rosette_and_centres_from_all_leaves(calibration):
    """One strong leaf should pull the rest of a pale rosette into one candidate."""
    from farmbot_vision.weed_settings import WeedSettings

    image = np.zeros((220, 320, 3), np.uint8)
    strong = tuple(
        int(value) for value in cv2.cvtColor(np.uint8([[[60, 130, 150]]]), cv2.COLOR_HSV2BGR)[0, 0]
    )
    pale = tuple(
        int(value) for value in cv2.cvtColor(np.uint8([[[60, 20, 130]]]), cv2.COLOR_HSV2BGR)[0, 0]
    )
    cv2.circle(image, (245, 105), 7, strong, -1)
    for center in ((263, 105), (254, 90), (254, 120)):
        cv2.circle(image, center, 7, pale, -1)
    settings = WeedSettings(
        enabled=True,
        minimum_confidence=0.3,
        visual_verifier_enabled=True,
        visual_verifier_shadow_mode=False,
        visual_verifier_rejection_confidence=0.4,
        visual_verifier_acceptance_confidence=0.8,
    )

    result = ClassicalVisionEngine(weed_verifier=_StubVerifier(0.9)).analyse(
        encode_jpeg(image), 9, NOW, [], calibration, {}, settings
    )

    assert len(result.weeds) == 1
    weed = result.weeds[0]
    assert 251 <= weed.center_px[0] <= 257
    assert 101 <= weed.center_px[1] <= 109
    assert weed.radius_mm >= 15


def test_single_remote_green_speck_does_not_define_weed_radius(calibration):
    """The supported radial envelope must ignore one grouped soil-like outlier."""
    from farmbot_vision.weed_settings import WeedSettings

    image = np.zeros((220, 320, 3), np.uint8)
    cv2.circle(image, (160, 110), 12, (20, 210, 30), -1)
    # Within the grouping gap, so this exercises radius robustness rather than
    # merely proving that connected-component labelling keeps it separate.
    cv2.circle(image, (188, 110), 2, (20, 210, 30), -1)
    settings = WeedSettings(enabled=True, minimum_confidence=0.3)

    result = ClassicalVisionEngine().analyse(
        encode_jpeg(image), 9, NOW, [], calibration, {}, settings
    )

    assert len(result.weeds) == 1
    assert result.weeds[0].radius_mm < 18


def test_verifier_keeps_only_the_review_band_and_acceptance_band():
    shared = {
        "visual_verifier_rejection_confidence": 0.45,
        "visual_verifier_acceptance_confidence": 0.85,
    }

    assert _weed_scene(_StubVerifier(0.3), **shared).weeds == []
    assert len(_weed_scene(_StubVerifier(0.6), **shared).weeds) == 1
    assert len(_weed_scene(_StubVerifier(0.9), **shared).weeds) == 1


def test_verifier_scores_unclaimed_weed_inside_crop_protection(seed, calibration):
    """Crop proximity is model context, not an invisible candidate veto."""
    from farmbot_vision.weed_settings import WeedSettings

    verifier = _StubVerifier(0.9)
    settings = WeedSettings(
        enabled=True,
        plant_exclusion_margin_mm=35,
        minimum_confidence=0.3,
        visual_verifier_enabled=True,
        visual_verifier_shadow_mode=False,
        visual_verifier_minimum_confidence=0.6,
    )
    # The separate weed is 70 mm from a crop recorded with a 60 mm radius: it
    # lies inside the old compounded no-candidate circle but is not crop-owned.
    result = ClassicalVisionEngine(weed_verifier=verifier).analyse(
        jpeg([("circle", ((160, 120), 25)), ("circle", ((230, 120), 9))]),
        9,
        NOW,
        [seed],
        calibration,
        {},
        settings,
    )

    assert len(result.weeds) == 1
    assert result.weed_candidate_stats["protected_scored"] >= 1
    assert result.weeds[0].features["crop_protection_overlap"] > 0


def test_crop_mask_support_suppresses_a_crop_fragment_from_weed_review(seed, calibration):
    """A verifier cannot overrule crop-centre and connected-mask evidence."""
    from farmbot_vision.weed_settings import WeedSettings

    verifier = _StubVerifier(0.99)
    settings = WeedSettings(
        enabled=True,
        minimum_confidence=0.3,
        visual_verifier_enabled=True,
        visual_verifier_shadow_mode=False,
        visual_verifier_rejection_confidence=0.45,
        visual_verifier_acceptance_confidence=0.85,
    )
    result = ClassicalVisionEngine(weed_verifier=verifier).analyse(
        jpeg([("circle", ((160, 120), 25)), ("circle", ((210, 120), 8))]),
        9,
        NOW,
        [seed],
        calibration,
        {},
        settings,
    )

    assert verifier.seen, "the verifier should still produce diagnostic evidence"
    assert result.weeds == []
    context = verifier.seen[-1]
    assert context["crop_center_proximity_multiplier"] < 1
    assert context["plant_mask_support_overlap"] > 0
    assert context["crop_context_confidence_multiplier"] < 0.1


def test_crop_support_does_not_chain_across_the_frame(calibration):
    """A weed near a crop must not protect every weed reachable from it.

    Grouping joins vegetation islands up to WEED_GROUP_MAX_GAP_MM apart. Without
    a distance limit those joins chained, so one leaf touching the crop's
    exclusion zone protected a whole connected network of weeds.
    """
    from farmbot_vision.weed_settings import WeedSettings

    seed = PlantSeed(plant_id=1, crop_slug="lettuce", center_px=(40, 120), current_radius_mm=20)
    # A chain of blobs stepping away from the crop, each within the joining gap
    # of the last, ending far enough away to be an unmistakable interrow weed.
    chain = [("circle", ((40, 120), 18))] + [("circle", ((x, 120), 5)) for x in range(70, 290, 14)]

    result = ClassicalVisionEngine().analyse(
        jpeg(chain),
        9,
        NOW,
        [seed],
        calibration,
        {},
        WeedSettings(enabled=True, plant_exclusion_margin_mm=10),
    )

    assert result.weeds, "the far end of the chain must still be reported"
    assert max(weed.center_px[0] for weed in result.weeds) > 200


def test_candidate_stats_explain_oversized_candidates_without_hiding_them(seed, calibration):
    """Oversize safety must be observable without starving review."""
    from farmbot_vision.weed_settings import WeedSettings

    result = ClassicalVisionEngine().analyse(
        jpeg([("circle", ((160, 120), 25)), ("circle", ((285, 40), 10))]),
        9,
        NOW,
        [seed],
        calibration,
        {},
        WeedSettings(enabled=True, plant_exclusion_margin_mm=10, maximum_area_mm2=11),
    )

    assert len(result.weeds) == 1
    assert result.weed_candidate_stats["blobs"] >= 1
    assert result.weed_candidate_stats["oversized_scored"] >= 1


def test_candidate_stats_are_empty_when_weed_detection_is_off(seed, calibration):
    assert analyse([("circle", ((160, 120), 25))], seed, calibration).weed_candidate_stats == {}


def test_a_realistic_weed_clears_the_default_review_threshold(seed, calibration):
    """The heuristic has to be able to reach its own default threshold.

    Its terms used to be summed unnormalised, so a real weed topped out near
    0.70 -- which was also the default review threshold, leaving effectively
    nothing able to pass it.
    """
    from farmbot_vision.weed_settings import WeedSettings

    defaults = WeedSettings(enabled=True)
    result = ClassicalVisionEngine().analyse(
        jpeg([("circle", ((160, 120), 25)), ("circle", ((285, 40), 9))]),
        9,
        NOW,
        [seed],
        calibration,
        {},
        defaults,
    )

    assert len(result.weeds) == 1
    assert result.weeds[0].heuristic_confidence > defaults.minimum_confidence
