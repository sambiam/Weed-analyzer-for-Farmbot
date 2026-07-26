from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from farmbot_vision.weed_verifier import (
    FEATURE_NAMES,
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


def test_training_requires_both_classes(tmp_path):
    verifier = WeedVisualVerifier(tmp_path / "weed_model.json")
    with pytest.raises(ValueError, match="Need at least 3 weed and 3 non-weed"):
        verifier.train(
            [{"detection_id": "one", "label": "weed", "features": _features(1, 0)}],
            minimum_per_class=3,
        )
