from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from farmbot_vision.weed_verifier import (
    ALL_LABELS,
    CORE_FEATURE_NAMES,
    FEATURE_NAMES,
    NEGATIVE_LABELS,
    SPATIAL_FEATURE_NAMES,
    WeedVisualVerifier,
    encode_candidate_crop,
    extract_visual_features,
)


def _features(green: float, texture: float) -> dict[str, float]:
    values = {name: 0.2 for name in FEATURE_NAMES}
    values["strong_green_fraction"] = green
    values["mean_excess_green"] = green
    values["mean_saturation"] = green
    values["texture"] = texture
    return values


def _labelled(label: str, green: float, texture: float, index: int, image_id: int) -> dict:
    return {
        "detection_id": f"{label}-{index}",
        "label": label,
        "features": _features(green, texture),
        "created_at": f"2026-01-{index % 28 + 1:02d}",
        "image_id": image_id,
    }


def test_fallen_leaf_is_a_supported_hard_negative_label():
    assert "fallen_leaf" in ALL_LABELS
    assert "fallen_leaf" in NEGATIVE_LABELS


def test_visual_features_and_candidate_crop_are_extracted():
    image = np.zeros((100, 120, 3), np.uint8)
    cv2.ellipse(image, (60, 50), (20, 12), 20, 0, 360, (20, 210, 30), -1)
    component = np.any(image > 0, axis=2)

    features = extract_visual_features(image, component, 200)
    crop = encode_candidate_crop(image, component)

    assert set(features) == set(FEATURE_NAMES)
    assert features["strong_green_fraction"] > 0.9
    assert features["solidity"] > 0.8
    assert crop is not None
    decoded = cv2.imdecode(np.frombuffer(crop, np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape == (96, 96, 3)


def test_local_verifier_trains_persists_and_separates_examples(tmp_path):
    samples = []
    for index in range(12):
        samples.append(
            {
                "detection_id": f"weed-{index}",
                "label": "weed",
                "features": _features(0.9 - index * 0.005, 0.35),
                "created_at": f"2026-01-{index + 1:02d}",
            }
        )
        samples.append(
            {
                "detection_id": f"mulch-{index}",
                "label": "mulch_soil",
                "features": _features(0.1 + index * 0.005, 0.8),
                "created_at": f"2026-02-{index + 1:02d}",
            }
        )

    path = tmp_path / "weed_model.json"
    verifier = WeedVisualVerifier(path)
    model = verifier.train(samples, minimum_per_class=10)

    assert model["sample_count"] == 24
    assert model["metrics"]["precision"] >= 0.8
    assert verifier.predict(_features(0.95, 0.3)) > 0.8
    assert verifier.predict(_features(0.05, 0.9)) < 0.2
    assert WeedVisualVerifier(path).predict(_features(0.95, 0.3)) > 0.8
    assert json.loads(path.read_text())["feature_names"] == list(FEATURE_NAMES)


def test_validation_holds_out_whole_images(tmp_path):
    # Six images, four candidates each: the split must never place two
    # candidates from the same image on both sides.
    samples = []
    for image in range(6):
        for index in range(2):
            samples.append(_labelled("weed", 0.9, 0.35, image * 2 + index, 100 + image))
            samples.append(_labelled("moss", 0.15, 0.8, image * 2 + index, 100 + image))

    model = WeedVisualVerifier(tmp_path / "weed_model.json").train(samples, minimum_per_class=10)

    assert model["grouped_validation"] is True
    assert 0 < model["validation_groups"] < 6


def test_single_image_dataset_reports_that_it_could_not_hold_data_out(tmp_path):
    samples = [_labelled("weed", 0.9, 0.35, index, 7) for index in range(10)]
    samples += [_labelled("moss", 0.1, 0.8, index, 7) for index in range(10)]

    model = WeedVisualVerifier(tmp_path / "weed_model.json").train(samples, minimum_per_class=10)

    assert model["grouped_validation"] is False


def test_threshold_is_suggested_and_metrics_are_reported_at_it(tmp_path):
    samples = []
    for image in range(6):
        samples.append(_labelled("weed", 0.9, 0.3, image, 200 + image))
        samples.append(_labelled("weed", 0.85, 0.35, image + 20, 200 + image))
        for index in range(3):
            samples.append(_labelled("moss", 0.12, 0.85, image * 3 + index, 200 + image))

    verifier = WeedVisualVerifier(tmp_path / "weed_model.json")
    model = verifier.train(samples, minimum_per_class=10)

    suggested = verifier.suggested_threshold
    assert suggested is not None
    assert 0.0 <= suggested <= 1.0
    assert model["metrics"]["at_threshold"] == suggested
    assert model["threshold_curve"]
    assert all("precision" in point and "recall" in point for point in model["threshold_curve"])


def test_a_thin_but_perfect_validation_fold_does_not_justify_a_permissive_threshold(tmp_path):
    # Cleanly separable data with only a few held-out positives. Judged on the
    # point estimate every threshold looks 100% precise, and the sweep used to
    # recommend the very bottom of the curve.
    samples = [_labelled("weed", 0.95, 0.2, index, 1000 + index) for index in range(10)]
    samples += [_labelled("moss", 0.05, 0.9, index, 1100 + index) for index in range(10)]

    verifier = WeedVisualVerifier(tmp_path / "weed_model.json")
    verifier.train(samples, minimum_per_class=10)

    assert verifier.suggested_threshold > 0.1


def test_scores_are_corrected_to_the_observed_class_balance(tmp_path):
    # Four times as many negatives as positives. A class-balanced fit answers
    # "which of an equal pair is this?", which is far too generous for a real
    # candidate stream; the stored model must be shifted back to the prior.
    samples = [_labelled("weed", 0.9, 0.3, index, 300 + index) for index in range(10)]
    samples += [_labelled("moss", 0.5, 0.5, index, 400 + index) for index in range(40)]

    verifier = WeedVisualVerifier(tmp_path / "weed_model.json")
    model = verifier.train(samples, minimum_per_class=10)

    assert model["prior_corrected"] is True
    # An ambiguous candidate sitting between the two clusters must not be
    # called a weed just because the fit was balanced.
    assert verifier.predict(_features(0.7, 0.4)) < 0.5


def test_best_guess_names_the_most_likely_category(tmp_path):
    samples = []
    for index in range(10):
        samples.append(_labelled("weed", 0.9, 0.3, index, 500 + index))
        samples.append(_labelled("moss", 0.95, 0.1, index, 600 + index))
        samples.append(_labelled("hardware", 0.05, 0.9, index, 700 + index))

    verifier = WeedVisualVerifier(tmp_path / "weed_model.json")
    verifier.train(samples, minimum_per_class=10)

    guess = verifier.explain(_features(0.95, 0.1))
    assert guess
    assert guess[0][0] == "moss"
    assert abs(sum(probability for _, probability in guess) - 1.0) < 1e-6
    # Description only: the accept/reject score is still the binary one.
    assert verifier.predict(_features(0.95, 0.1)) is not None


def test_legacy_samples_without_spatial_features_train_on_the_core_set(tmp_path):
    def legacy(label: str, green: float, index: int) -> dict:
        sample = _labelled(label, green, 0.5, index, 800 + index)
        for name in SPATIAL_FEATURE_NAMES:
            sample["features"].pop(name)
        return sample

    samples = [legacy("weed", 0.9, index) for index in range(10)]
    samples += [legacy("moss", 0.1, index + 50) for index in range(10)]

    verifier = WeedVisualVerifier(tmp_path / "weed_model.json")
    model = verifier.train(samples, minimum_per_class=10)

    assert model["feature_names"] == list(CORE_FEATURE_NAMES)
    # A core-only model still scores a full modern feature vector.
    assert verifier.predict(_features(0.95, 0.3)) is not None


def test_spatial_features_are_extracted_and_default_to_far_from_any_crop():
    image = np.zeros((200, 240, 3), np.uint8)
    cv2.ellipse(image, (120, 100), (20, 12), 20, 0, 360, (20, 210, 30), -1)
    component = np.any(image > 0, axis=2)

    far = extract_visual_features(image, component, 200)
    near = extract_visual_features(image, component, 200, distance_to_plant_mm=15)

    assert far["distance_to_plant"] == 1.0
    assert near["distance_to_plant"] < 0.1
    # Nothing else green nearby, and the blob is nowhere near a frame edge.
    assert far["local_vegetation_density"] < 0.05
    assert far["edge_proximity"] == 0.0


def test_edge_proximity_rises_for_a_blob_entering_the_frame():
    image = np.zeros((200, 240, 3), np.uint8)
    cv2.ellipse(image, (2, 100), (20, 12), 0, 0, 360, (20, 210, 30), -1)
    component = np.any(image > 0, axis=2)

    assert extract_visual_features(image, component, 200)["edge_proximity"] == 1.0


def test_bundled_model_is_used_until_a_local_one_exists(tmp_path):
    samples = [_labelled("weed", 0.9, 0.3, index, 900 + index) for index in range(10)]
    samples += [_labelled("moss", 0.1, 0.8, index, 950 + index) for index in range(10)]
    bundled_path = tmp_path / "bundled.json"
    WeedVisualVerifier(bundled_path).train(samples, minimum_per_class=10)

    local_path = tmp_path / "local.json"
    verifier = WeedVisualVerifier(local_path, bundled_path=bundled_path)
    assert verifier.available
    assert verifier.is_bundled

    verifier.train(samples, minimum_per_class=10)
    assert not verifier.is_bundled

    # Clearing local training falls back to the bundled model rather than
    # leaving the add-on with no filtering at all.
    verifier.clear()
    assert verifier.available
    assert verifier.is_bundled


def test_a_model_naming_unknown_features_is_rejected(tmp_path):
    path = tmp_path / "weed_model.json"
    path.write_text(
        json.dumps(
            {
                "feature_names": ["not_a_feature"],
                "mean": [0.0],
                "scale": [1.0],
                "weights": [1.0],
                "bias": 0.0,
            }
        ),
        encoding="utf-8",
    )

    assert not WeedVisualVerifier(path).available


def test_training_requires_both_classes(tmp_path):
    verifier = WeedVisualVerifier(tmp_path / "weed_model.json")
    with pytest.raises(ValueError, match="Need at least 3 weed and 3 non-weed"):
        verifier.train(
            [{"detection_id": "one", "label": "weed", "features": _features(1, 0)}],
            minimum_per_class=3,
        )
