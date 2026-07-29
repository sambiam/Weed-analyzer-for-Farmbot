import math
from datetime import UTC, datetime
from types import SimpleNamespace

from farmbot_vision.calibration_store import FarmbotCalibrationInput
from farmbot_vision.photo_grid import (
    PHOTO_GRID_CHUNK_SIZE,
    PHOTO_GRID_CONTINUOUS_CAPABILITY,
    PHOTO_GRID_CONTINUOUS_MAX_TARGETS,
    PhotoGridFrame,
    PhotoGridRecord,
    PhotoGridTarget,
    match_verified_frames,
    photo_grid_chunk_size,
    plan_photo_grid,
    plan_targeted_plant_captures,
)

# Geometry chosen to produce the real bed's 11 x 7 grid: an 1800 x 1600 native
# frame at 0.25 mm/px is a 450 x 400 mm footprint, which at the planner's 15%
# overlap lays 11 columns across 4180 mm and 7 rows across 2164 mm.
BED_X_BOUNDS = (0.0, 4180.0)
BED_Y_BOUNDS = (0.0, 2164.0)
BED_COLUMNS = 11
BED_ROWS = 7


def _bed_calibration():
    return _calibration(reference_width=1800, reference_height=1600)


def _bed_grid(z=0.0):
    return plan_photo_grid(_bed_calibration(), x_bounds=BED_X_BOUNDS, y_bounds=BED_Y_BOUNDS, z=z)[0]


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


def test_bed_grid_generates_seventy_seven_unique_cells():
    targets = _bed_grid()

    assert len(targets) == BED_COLUMNS * BED_ROWS == 77
    assert len({(item.row, item.column) for item in targets}) == 77
    assert len({(item.x, item.y) for item in targets}) == 77
    assert [item.index for item in targets] == list(range(77))
    assert {item.row for item in targets} == set(range(BED_ROWS))
    assert {item.column for item in targets} == set(range(BED_COLUMNS))

    # Coordinates come from the configured bounds, footprint and offsets, not
    # from anything the bot reported.
    calibration = _bed_calibration()
    footprint_x = calibration.coordinate_scale * calibration.reference_width
    footprint_y = calibration.coordinate_scale * calibration.reference_height
    # FarmBot's optical centre is gantry + camera offset, so capture gantry
    # coordinates subtract the copied offsets from the desired bed positions.
    first_x = BED_X_BOUNDS[0] + footprint_x / 2 - calibration.offset_x_mm
    last_x = BED_X_BOUNDS[1] - footprint_x / 2 - calibration.offset_x_mm
    first_y = BED_Y_BOUNDS[0] + footprint_y / 2 - calibration.offset_y_mm
    last_y = BED_Y_BOUNDS[1] - footprint_y / 2 - calibration.offset_y_mm

    first, last = targets[0], targets[-1]
    assert (first.row, first.column) == (0, 0)
    assert (first.x, first.y) == (first_x, first_y)
    # Seven rows means row 7 runs left to right again, so the route ends on the
    # far column of the far row.
    assert (last.row, last.column) == (BED_ROWS - 1, BED_COLUMNS - 1)
    assert (last.x, last.y) == (last_x, last_y)
    assert max(item.x for item in targets) == last_x
    assert all(item.z == 0.0 for item in targets)


def test_bed_grid_uses_even_spacing_from_the_configured_geometry():
    targets = _bed_grid()
    xs = sorted({item.x for item in targets})
    ys = sorted({item.y for item in targets})
    x_spacing = {round(b - a, 6) for a, b in zip(xs, xs[1:], strict=False)}
    y_spacing = {round(b - a, 6) for a, b in zip(ys, ys[1:], strict=False)}

    assert len(x_spacing) == 1
    assert len(y_spacing) == 1
    # The real bed's spacing, derived rather than hard-coded.
    assert x_spacing == {373.0}
    assert y_spacing == {294.0}


def test_bed_grid_is_one_continuous_serpentine_route():
    targets = _bed_grid()
    rows = [targets[row * BED_COLUMNS : (row + 1) * BED_COLUMNS] for row in range(BED_ROWS)]

    for number, row in enumerate(rows):
        # A row transition only ever happens after all 11 of a row's cells.
        assert {item.row for item in row} == {number}
        assert len({item.y for item in row}) == 1
        assert len(row) == BED_COLUMNS
        steps = [b.x - a.x for a, b in zip(row, row[1:], strict=False)]
        direction = 1 if number % 2 == 0 else -1
        assert all(math.copysign(1, step) == direction for step in steps)

    ys = [row[0].y for row in rows]
    y_steps = {round(b - a, 6) for a, b in zip(ys, ys[1:], strict=False)}
    assert len(y_steps) == 1

    for previous, following in zip(rows, rows[1:], strict=False):
        # The transition stays at the end of the grid the row finished on and
        # only steps to the adjacent row.
        assert previous[-1].x == following[0].x
        assert previous[-1].column == following[0].column


def test_bed_grid_route_travels_only_rows_and_row_transitions():
    targets = _bed_grid()
    xs = sorted({item.x for item in targets})
    ys = sorted({item.y for item in targets})
    x_spacing = xs[1] - xs[0]
    y_spacing = ys[1] - ys[0]

    legs = list(zip(targets, targets[1:], strict=False))
    lateral = [leg for leg in legs if leg[0].row == leg[1].row]
    transitions = [leg for leg in legs if leg[0].row != leg[1].row]

    assert len(lateral) == BED_ROWS * (BED_COLUMNS - 1)
    assert len(transitions) == BED_ROWS - 1
    assert all(round(abs(b.x - a.x), 6) == round(x_spacing, 6) and a.y == b.y for a, b in lateral)
    # Each transition is one adjacent row step with no travel back across the row.
    assert all(
        a.x == b.x and round(abs(b.y - a.y), 6) == round(y_spacing, 6) for a, b in transitions
    )

    route = sum(math.dist((a.x, a.y), (b.x, b.y)) for a, b in legs)
    expected = BED_ROWS * (BED_COLUMNS - 1) * x_spacing + (BED_ROWS - 1) * y_spacing
    assert round(route, 6) == round(expected, 6)


def test_batch_slices_reproduce_the_canonical_route_exactly():
    """A deliberately small batch limit must not change the route at all."""
    targets = _bed_grid()
    limit = 5
    batches = [targets[start : start + limit] for start in range(0, len(targets), limit)]

    rejoined = [target for batch in batches for target in batch]
    assert [item.index for item in rejoined] == [item.index for item in targets]
    assert len({item.index for item in rejoined}) == len(targets)
    assert sum(len(batch) for batch in batches) == len(targets)

    for previous, following in zip(batches, batches[1:], strict=False):
        # The first cell of a batch is exactly the cell that follows the last
        # cell of the one before it -- including when the boundary lands in the
        # middle of a row.
        assert following[0].index == previous[-1].index + 1
        assert following[0] is targets[previous[-1].index + 1]
    # A five-cell limit slices an eleven-column row mid-row; row and column
    # numbering is unaffected because it comes from the canonical plan.
    assert batches[1][0].row == 0
    assert batches[1][0].column == 5


def test_chunk_size_follows_the_integrations_advertised_capability():
    assert photo_grid_chunk_size([PHOTO_GRID_CONTINUOUS_CAPABILITY]) == (
        PHOTO_GRID_CONTINUOUS_MAX_TARGETS
    )
    assert PHOTO_GRID_CONTINUOUS_MAX_TARGETS >= 77
    # Anything that does not advertise continuous capture keeps the legacy cap,
    # because Home Assistant refuses an oversized call before the handler runs.
    assert photo_grid_chunk_size(["position_verified_photo_grid_repair"]) == PHOTO_GRID_CHUNK_SIZE
    assert photo_grid_chunk_size([]) == PHOTO_GRID_CHUNK_SIZE
    assert photo_grid_chunk_size(None) == PHOTO_GRID_CHUNK_SIZE


def test_frames_are_credited_by_target_index_when_the_integration_supplies_one():
    targets = _bed_grid()
    neighbour = targets[1]
    result = match_verified_frames(
        targets,
        [
            # Several millimetres off after a long move -- and nearer to the
            # neighbouring cell than a naive proximity match would like -- but
            # the integration already tied it to cell 0.
            {"image_id": 41, "target_index": 0, "x": neighbour.x, "y": neighbour.y, "z": 0},
            {"image_id": 42, "target_index": 999, "x": targets[2].x, "y": targets[2].y, "z": 0},
        ],
    )

    assert [(item.target_index, item.image_id) for item in result] == [(0, 41)]


def test_planned_coordinates_are_stable_across_repeated_planning():
    assert [(item.x, item.y, item.z) for item in _bed_grid()] == [
        (item.x, item.y, item.z) for item in _bed_grid()
    ]
    assert all(item.x == round(item.x, 3) and item.y == round(item.y, 3) for item in _bed_grid())


def _targeted_record() -> PhotoGridRecord:
    calibration = FarmbotCalibrationInput(
        coordinate_scale=1,
        reference_width=100,
        reference_height=100,
    )
    target = PhotoGridTarget(index=0, row=0, column=0, x=50, y=50, z=0)
    return PhotoGridRecord(
        config_entry_id="bot",
        started_at=datetime(2026, 7, 30, tzinfo=UTC),
        completed_at=datetime(2026, 7, 30, 0, 5, tzinfo=UTC),
        status="complete",
        bed_bounds={"x": (0, 500), "y": (0, 400)},
        footprint_width_mm=100,
        footprint_height_mm=100,
        calibration=calibration,
        targets=[target],
        frames=[PhotoGridFrame(target_index=0, image_id=101, x=50, y=50, z=0)],
    )


def test_fit_sized_plant_without_half_coverage_queues_exactly_one_targeted_capture():
    record = _targeted_record()
    plant = SimpleNamespace(id=7, name="Lettuce", openfarm_slug="lettuce", x=250, y=200, radius=20)

    planned, diagnostics = plan_targeted_plant_captures(
        record,
        [plant],
        safety_margin_mm=10,
    )
    record.targeted_captures.extend(planned)
    duplicate, second_diagnostics = plan_targeted_plant_captures(
        record,
        [plant],
        safety_margin_mm=10,
    )

    assert len(planned) == 1
    assert (planned[0].x, planned[0].y) == (250, 200)
    assert diagnostics[0]["targeted_photo_scheduled"] is True
    assert duplicate == []
    assert second_diagnostics[0]["targeted_photo_scheduled"] is False
    assert "already queued or completed" in second_diagnostics[0]["reason"]


def test_plant_too_large_for_one_photo_uses_grid_without_targeted_capture():
    record = _targeted_record()
    plant = SimpleNamespace(id=8, name="Pumpkin", openfarm_slug="pumpkin", x=250, y=200, radius=100)

    planned, diagnostics = plan_targeted_plant_captures(
        record,
        [plant],
        safety_margin_mm=10,
    )

    assert planned == []
    assert diagnostics[0]["targeted_photo_scheduled"] is False
    assert "exceeds the usable single-photo footprint" in diagnostics[0]["reason"]
