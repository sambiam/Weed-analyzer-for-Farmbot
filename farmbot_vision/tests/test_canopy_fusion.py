from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import cv2
import numpy as np

from farmbot_vision.canopy_fusion import fuse_canopy_masks
from farmbot_vision.canopy_settings import CanopyFusionSettings, CanopyFusionSettingsStore
from farmbot_vision.models import Decision, Measurement


def _measurement(
    mask_path: str,
    *,
    image_id: int,
    timestamp: datetime,
    sectors: list[int] | None = None,
    center_visible: bool = True,
) -> Measurement:
    sectors = list(range(36)) if sectors is None else sectors
    return Measurement(
        measurement_id=uuid4(),
        plant_id=17,
        crop_slug="lettuce",
        image_id=image_id,
        image_timestamp=timestamp,
        current_radius_mm=30,
        typical_canopy_radius_mm=42,
        maximum_accepted_canopy_radius_mm=50,
        recommended_protection_radius_mm=70,
        confidence=0.95,
        decision=Decision.RECOMMENDED,
        reason="partial view",
        algorithm_version="test",
        calibrated=True,
        visible_fraction=0.6,
        boundary_coverage=len(sectors) / 72,
        boundary_sectors=sectors,
        center_visible=center_visible,
        has_plant_evidence=True,
        plant_center_px=(80, 80),
        mask_path=mask_path,
        transform_json=json.dumps(
            {
                "pixels_per_mm_x": 1,
                "pixels_per_mm_y": 1,
                "rotation_degrees": 0,
                "origin_location": "top_left",
            }
        ),
    )


def test_fuses_plant_masks_before_measuring_radius(tmp_path):
    mask = np.zeros((160, 160), dtype=np.uint8)
    cv2.circle(mask, (80, 80), 50, 255, -1)
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    cv2.imwrite(str(first), mask)
    cv2.imwrite(str(second), mask)
    now = datetime.now(UTC)

    result = fuse_canopy_masks(
        [
            _measurement(
                str(first),
                image_id=1,
                timestamp=now,
                sectors=list(range(18)),
            ),
            _measurement(
                str(second),
                image_id=2,
                timestamp=now + timedelta(minutes=2),
                sectors=list(range(18, 36)),
                center_visible=False,
            ),
        ],
        CanopyFusionSettings(always_fuse_when_available=True),
    )

    assert result is not None
    assert result.view_count == 2
    assert 47 <= result.maximum_radius_mm <= 52
    assert result.angular_coverage >= 0.95
    assert result.corroborated_fraction >= 0.99
    assert result.diagnostic_jpeg


def test_fusion_includes_useful_tiles_after_minimum_boundary_coverage(tmp_path):
    base = np.zeros((160, 160), dtype=np.uint8)
    cv2.circle(base, (80, 80), 30, 255, -1)
    outer_leaf = base.copy()
    cv2.line(outer_leaf, (105, 80), (148, 80), 255, 7)
    paths = [tmp_path / f"view-{index}.png" for index in range(3)]
    cv2.imwrite(str(paths[0]), base)
    cv2.imwrite(str(paths[1]), base)
    cv2.imwrite(str(paths[2]), outer_leaf)
    now = datetime.now(UTC)

    result = fuse_canopy_masks(
        [
            _measurement(str(paths[0]), image_id=1, timestamp=now, sectors=list(range(18))),
            _measurement(
                str(paths[1]),
                image_id=2,
                timestamp=now + timedelta(minutes=1),
                sectors=list(range(18, 36)),
                center_visible=False,
            ),
            # This view adds no sectors needed to cross the old 50% selection
            # threshold, but it contains real canopy pixels needed by fusion.
            _measurement(
                str(paths[2]),
                image_id=3,
                timestamp=now + timedelta(minutes=2),
                sectors=list(range(18)),
                center_visible=False,
            ),
        ],
        CanopyFusionSettings(always_fuse_when_available=True),
    )

    assert result is not None
    assert result.view_count == 3
    assert result.maximum_radius_mm > 60


def test_fusion_excludes_stale_view_without_lowering_current_partial_estimate(tmp_path):
    mask = np.zeros((160, 160), dtype=np.uint8)
    cv2.circle(mask, (80, 80), 50, 255, -1)
    path = tmp_path / "mask.png"
    cv2.imwrite(str(path), mask)
    now = datetime.now(UTC)

    result = fuse_canopy_masks(
        [
            _measurement(str(path), image_id=1, timestamp=now),
            _measurement(str(path), image_id=2, timestamp=now - timedelta(hours=8)),
        ],
        CanopyFusionSettings(
            always_fuse_when_available=True,
            maximum_time_gap_hours=6,
        ),
    )

    assert result is not None
    assert result.view_count == 1
    assert result.angular_coverage >= 0.5


def test_fusion_also_rejects_a_broad_implausible_radius_jump(tmp_path):
    mask = np.zeros((160, 160), dtype=np.uint8)
    cv2.circle(mask, (80, 80), 75, 255, -1)
    first = tmp_path / "broad-first.png"
    second = tmp_path / "broad-second.png"
    cv2.imwrite(str(first), mask)
    cv2.imwrite(str(second), mask)
    now = datetime.now(UTC)

    result = fuse_canopy_masks(
        [
            _measurement(str(first), image_id=1, timestamp=now),
            _measurement(
                str(second),
                image_id=2,
                timestamp=now + timedelta(minutes=1),
                sectors=list(range(36, 72)),
                center_visible=False,
            ),
        ],
        CanopyFusionSettings(always_fuse_when_available=True),
    )

    assert result is not None
    assert result.maximum_radius_mm < 40
    assert result.confidence <= 0.74


def test_canopy_settings_round_trip(tmp_path):
    store = CanopyFusionSettingsStore(tmp_path / "canopy.json")
    values = CanopyFusionSettings(
        always_fuse_when_available=True,
        minimum_views=3,
        minimum_angular_coverage=0.8,
        automatic_requires_reliable_fusion=False,
    )

    store.save(values)

    assert store.load() == values
