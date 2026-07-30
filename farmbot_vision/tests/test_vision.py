from __future__ import annotations

from datetime import UTC, datetime

import cv2
import numpy as np
from conftest import encode_jpeg, jpeg

from farmbot_vision.models import Calibration, Decision, PlantSeed
from farmbot_vision.vision import ClassicalVisionEngine, decode_jpeg, register_translation

NOW = datetime(2026, 2, 1, tzinfo=UTC)


def analyse(shapes, seed, calibration, previous=None, seeds=None):
    return ClassicalVisionEngine().analyse(
        jpeg(shapes), 9, NOW, seeds or [seed], calibration, previous or {}
    )


def test_circular_plant_without_weeds(seed, calibration):
    result = analyse([("circle", ((160, 120), 35))], seed, calibration)
    measurement = result.measurements[0]
    assert 33 <= measurement.maximum_accepted_canopy_radius_mm <= 37
    assert measurement.recommended_protection_radius_mm >= 63
    assert measurement.confidence > 0.7


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


def test_shadow_mode_scores_without_deciding():
    verifier = _StubVerifier(0.05)

    result = _weed_scene(verifier, visual_verifier_shadow_mode=True)

    assert len(result.weeds) == 1
    weed = result.weeds[0]
    # Scored and recorded, but the heuristic still owns the decision.
    assert weed.verifier_confidence == 0.05
    assert weed.confidence == weed.heuristic_confidence


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


def test_a_strict_maximum_area_is_still_honoured():
    """The one size judgement a verifier cannot make from a cropped image."""
    result = _weed_scene(_StubVerifier(0.9), maximum_area_mm2=10)

    assert result.weeds == []


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


def test_candidate_stats_explain_why_candidates_were_dropped(seed, calibration):
    """Starvation has to be observable, not inferred from absent weeds."""
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

    assert result.weeds == []
    assert result.weed_candidate_stats["blobs"] >= 1
    assert result.weed_candidate_stats["size"] >= 1


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
