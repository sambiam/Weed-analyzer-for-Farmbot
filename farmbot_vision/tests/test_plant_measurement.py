import json
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import cv2
import numpy as np

from farmbot_vision.database import Database
from farmbot_vision.models import Calibration, Decision, Measurement
from farmbot_vision.plant_measurement import (
    relative_pixel_to_plant_mm_transform,
    select_measurement_evidence,
)
from farmbot_vision.vision import pixel_to_garden


def _measurement(
    image_id: int,
    *,
    timestamp: datetime,
    confidence: float = 0.9,
    sectors: list[int] | None = None,
    has_evidence: bool = True,
    center_visible: bool = True,
    fits: bool = True,
    maximum: float = 50,
    recommendation: float = 80,
) -> Measurement:
    sectors = list(range(72)) if sectors is None else sectors
    return Measurement(
        measurement_id=uuid4(),
        config_entry_id="bot",
        plant_id=17,
        crop_slug="lettuce",
        image_id=image_id,
        image_timestamp=timestamp,
        current_radius_mm=40,
        typical_canopy_radius_mm=45 if has_evidence else 0,
        maximum_accepted_canopy_radius_mm=maximum if has_evidence else 0,
        recommended_protection_radius_mm=recommendation if has_evidence else 40,
        confidence=confidence,
        decision=Decision.RECOMMENDED if has_evidence else Decision.UNCERTAIN,
        reason="fixture",
        algorithm_version="test",
        plant_center_px=(50, 50),
        visible_fraction=len(sectors) / 72,
        boundary_coverage=len(sectors) / 72,
        boundary_sectors=sectors,
        center_visible=center_visible,
        has_plant_evidence=has_evidence,
        plant_fits_single_frame=fits,
        segmentation_quality=0.9 if has_evidence else 0,
        image_quality=0.9,
        exclusion_reason=None if has_evidence else "no target-owned canopy evidence",
        transform_json=json.dumps(
            {
                "pixels_per_mm_x": 1,
                "pixels_per_mm_y": 1,
                "rotation_degrees": 0,
                "origin_location": "top_left",
            }
        ),
    )


def test_complete_view_alone_sets_confidence_and_empty_tiles_do_not_reduce_it():
    now = datetime(2026, 7, 30, tzinfo=UTC)
    complete = _measurement(1, timestamp=now, confidence=0.94)
    empty = [
        _measurement(
            image_id,
            timestamp=now + timedelta(minutes=image_id),
            confidence=0.05,
            sectors=[],
            has_evidence=False,
            center_visible=False,
        )
        for image_id in (2, 3, 4)
    ]

    selection = select_measurement_evidence([*empty, complete])
    consolidated = Database._consolidate_measurement_rows(
        [item.model_dump(mode="json") for item in [*empty, complete]]
    )

    assert selection.mode == "single_complete"
    assert selection.used_ids == (1,)
    assert consolidated["confidence"] == 0.94
    assert consolidated["used_measurement_count"] == 1
    assert consolidated["useful_measurement_count"] == 1
    assert consolidated["measurement_count"] == 4
    assert "excluded images do not reduce confidence" in consolidated["reason"]
    assert {item["image_id"] for item in consolidated["excluded_images"]} == {2, 3, 4}


def test_partial_boundary_is_union_of_72_bed_space_sectors():
    now = datetime(2026, 7, 30, tzinfo=UTC)
    first = _measurement(10, timestamp=now, sectors=list(range(18)))
    second = _measurement(
        11,
        timestamp=now + timedelta(minutes=1),
        sectors=list(range(18, 36)),
        center_visible=False,
    )
    irrelevant = _measurement(
        12,
        timestamp=now + timedelta(minutes=2),
        sectors=[],
        has_evidence=False,
    )

    selection = select_measurement_evidence([first, second, irrelevant])

    assert selection.mode == "partial_composite"
    assert set(selection.used_ids) == {10, 11}
    assert selection.boundary_coverage == 0.5
    assert selection.exclusion_reason(irrelevant) == "no target-owned canopy evidence"


def test_large_plant_uses_composite_and_retains_all_useful_tiles():
    now = datetime(2026, 7, 30, tzinfo=UTC)
    measurements = [
        _measurement(
            image_id,
            timestamp=now + timedelta(minutes=image_id),
            sectors=list(range(image_id * 12, (image_id + 1) * 12)),
            center_visible=image_id == 0,
            fits=False,
        )
        for image_id in range(3)
    ]

    selection = select_measurement_evidence(measurements)

    assert selection.mode == "large_composite"
    assert set(selection.used_ids) == {0, 1, 2}


def test_no_usable_evidence_returns_uncertain_low_confidence():
    now = datetime(2026, 7, 30, tzinfo=UTC)
    rows = [
        _measurement(
            image_id,
            timestamp=now + timedelta(minutes=image_id),
            confidence=0.9,
            sectors=[],
            has_evidence=False,
            center_visible=False,
        ).model_dump(mode="json")
        for image_id in (20, 21)
    ]

    result = Database._consolidate_measurement_rows(rows)

    assert result["decision"] == "uncertain"
    assert result["confidence"] <= 0.2
    assert result["used_measurement_count"] == 0
    assert result["selected_image_ids"] == []
    assert "No usable target-plant evidence" in result["reason"]


def test_exclusion_reasons_are_persisted_without_entering_confidence_denominator(tmp_path):
    now = datetime(2026, 7, 30, tzinfo=UTC)
    complete = _measurement(1, timestamp=now, confidence=0.92)
    empty = _measurement(
        2,
        timestamp=now + timedelta(minutes=1),
        confidence=0.99,
        sectors=[],
        has_evidence=False,
    )
    database = Database(tmp_path / "db.sqlite")
    database.save_measurements([complete, empty])

    diagnostics = database.set_evidence_selection([complete, empty])
    rows = {row["image_id"]: row for row in database.recent_measurements()}

    assert diagnostics["selected_image_ids"] == [1]
    assert rows[1]["evidence_status"] == "used"
    assert rows[2]["evidence_status"] == "excluded"
    assert rows[2]["exclusion_reason"] == "no target-owned canopy evidence"


def test_pending_grid_context_restores_one_measurement_per_source_image(tmp_path):
    now = datetime(2026, 7, 30, tzinfo=UTC)
    database = Database(tmp_path / "db.sqlite")
    first = _measurement(101, timestamp=now, confidence=0.88)
    second = _measurement(
        102,
        timestamp=now + timedelta(minutes=1),
        confidence=0.05,
        sectors=[],
        has_evidence=False,
        center_visible=False,
    )
    database.save_measurements([first, second])

    restored = database.pending_plant_measurements("bot", 17, [101, 102, 999])

    assert {item.image_id for item in restored} == {101, 102}
    restored_first = next(item for item in restored if item.image_id == 101)
    assert restored_first.measurement_id == first.measurement_id
    assert restored_first.boundary_sectors == list(range(72))
    assert restored_first.has_plant_evidence is True


def test_relative_transform_matches_main_rotation_scale_and_translation_once(tmp_path):
    photo = np.zeros((100, 100, 3), np.uint8)
    mask = np.zeros((100, 100), np.uint8)
    cv2.circle(mask, (50, 50), 10, 255, -1)
    photo_path = tmp_path / "photo.jpg"
    mask_path = tmp_path / "mask.png"
    cv2.imwrite(str(photo_path), photo)
    cv2.imwrite(str(mask_path), mask)
    item = _measurement(30, timestamp=datetime(2026, 7, 30, tzinfo=UTC))
    item = item.model_copy(
        update={
            "source_image_path": str(photo_path),
            "mask_path": str(mask_path),
            "transform_json": json.dumps(
                {
                    "pixels_per_mm_x": 2,
                    "pixels_per_mm_y": 1,
                    "rotation_degrees": 90,
                    "origin_location": "top_left",
                }
            ),
        }
    )
    relative, _ = relative_pixel_to_plant_mm_transform(item)
    mapped = cv2.transform(np.float64([[[60, 50]]]), relative)[0, 0]
    calibration = Calibration(
        source="manual",
        pixels_per_mm_x=2,
        pixels_per_mm_y=1,
        rotation_degrees=90,
        uncertainty_mm=0,
    )
    expected = pixel_to_garden(60, 50, 0, 0, 100, 100, calibration)

    assert np.allclose(mapped, expected, atol=1e-9)
