from farmbot_vision.calibration_store import FarmbotCalibrationInput
from farmbot_vision.photo_grid import match_verified_frames, plan_photo_grid


def _calibration(**updates):
    values = {
        "coordinate_scale": 0.25,
        "reference_width": 1000,
        "reference_height": 800,
        "offset_x_mm": 10,
        "offset_y_mm": -5,
    }
    values.update(updates)
    return FarmbotCalibrationInput(**values)


def test_grid_plan_covers_bounds_in_serpentine_order():
    targets, width, height = plan_photo_grid(
        _calibration(), x_bounds=(0, 1000), y_bounds=(0, 600), z=0
    )
    assert width == 250
    assert height == 200
    assert len(targets) == 20
    assert [(item.row, item.column) for item in targets[:10]] == [
        (0, 0),
        (0, 1),
        (0, 2),
        (0, 3),
        (0, 4),
        (1, 4),
        (1, 3),
        (1, 2),
        (1, 1),
        (1, 0),
    ]
    assert all(0 <= item.x <= 1000 and 0 <= item.y <= 600 for item in targets)


def test_rotation_changes_projected_camera_footprint():
    _, width, height = plan_photo_grid(
        _calibration(rotation_degrees=90),
        x_bounds=(0, 1000),
        y_bounds=(0, 600),
        z=0,
    )
    assert round(width) == 200
    assert round(height) == 250


def test_verified_frames_require_matching_xyz_coordinates():
    targets, _, _ = plan_photo_grid(_calibration(), x_bounds=(0, 1000), y_bounds=(0, 600), z=0)
    result = match_verified_frames(
        targets[:2],
        [
            {
                "image_id": 41,
                "x": targets[0].x + 3,
                "y": targets[0].y - 4,
                "z": 0,
            },
            {"image_id": 42, "x": targets[1].x + 30, "y": targets[1].y, "z": 0},
        ],
    )
    assert [(item.target_index, item.image_id) for item in result] == [(0, 41)]
