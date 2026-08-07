"""Shape planning and G-code generation for the experimental Draw shape tab.

The geometry here decides where the gantry goes on a path that FarmBot OS does
not supervise, so these pin the two things that matter: that the path is the
shape it claims to be, and that the emitted program only ever uses codes the
companion integration will accept.
"""

import math

import pytest

from farmbot_vision import shape_gcode
from farmbot_vision.shape_gcode import (
    MAX_SEGMENTS,
    MIN_SEGMENTS,
    ShapeError,
    generate_program,
    program_lines,
    segments_for_tolerance,
    shape_points,
)


def _points(**kwargs):
    defaults = {
        "shape": "circle",
        "center_x": 500.0,
        "center_y": 400.0,
        "circumradius_mm": 100.0,
    }
    return shape_points(**{**defaults, **kwargs})


def test_a_closed_path_repeats_its_first_point():
    """The last move must return to the start or the shape is left open."""
    points = _points(shape="square")

    assert len(points) == 5
    assert points[0] == points[-1]


@pytest.mark.parametrize(
    "shape, expected_sides",
    [("triangle", 3), ("square", 4), ("pentagon", 5), ("hexagon", 6), ("octagon", 8)],
)
def test_named_polygons_have_their_side_count(shape, expected_sides):
    assert len(_points(shape=shape)) - 1 == expected_sides


def test_every_vertex_sits_on_the_circumradius():
    """'Centre plus circumradius' is one control for all shapes only if this holds."""
    for shape in ("circle", "triangle", "square", "hexagon"):
        for x, y in _points(shape=shape):
            assert math.hypot(x - 500.0, y - 400.0) == pytest.approx(100.0)


def test_rotation_turns_the_shape_about_its_centre():
    square = _points(shape="square", rotation_deg=0)
    turned = _points(shape="square", rotation_deg=90)

    # A square turned by 90 degrees maps onto itself, one vertex along.
    assert turned[0] == pytest.approx(square[1])


def test_a_polygon_needs_an_explicit_side_count():
    with pytest.raises(ShapeError, match="needs a side count"):
        _points(shape="polygon")


@pytest.mark.parametrize("sides", [2, 25])
def test_an_out_of_range_side_count_is_refused(sides):
    with pytest.raises(ShapeError, match="Sides must be"):
        _points(shape="polygon", sides=sides)


def test_an_unknown_shape_is_refused():
    with pytest.raises(ShapeError, match="Unknown shape"):
        _points(shape="dodecahedron")


@pytest.mark.parametrize("radius", [4.9, 2001.0])
def test_an_out_of_range_circumradius_is_refused(radius):
    with pytest.raises(ShapeError, match="Circumradius"):
        _points(circumradius_mm=radius)


@pytest.mark.parametrize("value", [math.inf, math.nan])
def test_non_finite_inputs_are_refused(value):
    with pytest.raises(ShapeError, match="finite"):
        _points(center_x=value)


def test_segment_count_honours_the_chord_tolerance():
    """Every chord's sagitta must stay inside the budget the user asked for."""
    radius, tolerance = 200.0, 0.5
    count = segments_for_tolerance(radius, tolerance)
    sagitta = radius * (1 - math.cos(math.pi / count))

    assert sagitta <= tolerance
    # ...and one fewer segment would exceed it, so the count is not wasteful.
    assert radius * (1 - math.cos(math.pi / (count - 1))) > tolerance


def test_a_bigger_circle_needs_more_segments_for_the_same_tolerance():
    assert segments_for_tolerance(500.0, 0.5) > segments_for_tolerance(50.0, 0.5)


def test_segment_count_stays_within_its_bounds():
    assert segments_for_tolerance(2000.0, 0.001) == MAX_SEGMENTS
    assert segments_for_tolerance(20.0, 15.0) == MIN_SEGMENTS


def test_the_generated_program_only_uses_codes_the_integration_accepts():
    """The integration's allowlist is G21, G90, G91, G00 and a standalone F."""
    plan = generate_program(shape="circle", center_x=500, center_y=400, circumradius_mm=100)

    executable = [line for line in plan.lines if not line.startswith(";")]

    assert {line.split()[0] for line in executable} <= {"G21", "G90", "G00"}
    # G01 appears only in the comment explaining why it is not used.
    assert not any("G01" in line for line in executable)


def test_the_program_approaches_and_retracts_at_the_travel_height():
    """A wrong centre should drag the tool through air, not through the bed."""
    plan = generate_program(
        shape="square",
        center_x=500,
        center_y=400,
        circumradius_mm=100,
        draw_z=-120,
        travel_z=-10,
    )
    moves = [line for line in plan.lines if line.startswith("G00")]

    assert moves[0] == "G00 Z-10.000 F400"  # retract before travelling
    assert moves[1].startswith("G00 X")  # travel over the start point
    assert moves[2] == "G00 Z-120.000 F400"  # only then descend
    assert moves[-1] == "G00 Z-10.000 F400"  # retract at the end


def test_the_first_drawn_point_is_where_the_path_starts():
    plan = generate_program(shape="square", center_x=500, center_y=400, circumradius_mm=100)
    first_x, first_y = plan.points[0]

    assert f"G00 X{first_x:.3f} Y{first_y:.3f}" in plan.lines


def test_every_path_segment_is_its_own_g00():
    plan = generate_program(shape="circle", center_x=500, center_y=400, circumradius_mm=100)
    xy_moves = [line for line in plan.lines if line.startswith("G00 X")]

    # One move onto the start point, then one per segment back around to it.
    assert len(xy_moves) == len(plan.points)


def test_perimeter_matches_the_polygon_it_draws():
    plan = generate_program(shape="square", center_x=0, center_y=0, circumradius_mm=100)

    # A square inscribed in r=100 has sides of r * sqrt(2).
    assert plan.perimeter_mm == pytest.approx(4 * 100 * math.sqrt(2))


def test_extent_is_the_shapes_bounding_box():
    plan = generate_program(shape="circle", center_x=500, center_y=400, circumradius_mm=100)
    min_x, min_y, max_x, max_y = plan.extent()

    assert min_x == pytest.approx(400.0, abs=1.0)
    assert max_x == pytest.approx(600.0, abs=1.0)
    assert min_y == pytest.approx(300.0, abs=1.0)
    assert max_y == pytest.approx(500.0, abs=1.0)


@pytest.mark.parametrize("feed", [0.5, 3001.0])
def test_an_out_of_range_feed_rate_is_refused(feed):
    with pytest.raises(ShapeError, match="[Ff]eed rate"):
        generate_program(
            shape="circle",
            center_x=500,
            center_y=400,
            circumradius_mm=100,
            feed_mm_per_min=feed,
        )


def test_the_header_records_what_was_asked_for():
    """The program is editable text; it has to say what produced it."""
    plan = generate_program(shape="hexagon", center_x=512.5, center_y=400, circumradius_mm=150)
    header = "\n".join(line for line in plan.lines if line.startswith(";"))

    assert "shape=hexagon" in header
    assert "512.5" in header
    assert "segments=6" in header
    # The G00-is-not-straight caveat travels with the program, not just the UI.
    assert "straight line" in header


def test_program_lines_drops_blanks_but_keeps_edits():
    text = "G21\n\n  G90  \r\n\nG00 X10 Y10\n\n"

    assert program_lines(text) == ["G21", "G90", "G00 X10 Y10"]


def test_every_shape_in_the_picker_can_actually_be_planned():
    """A dropdown entry that cannot be generated is a dead option."""
    for shape in shape_gcode.SHAPES:
        sides = 5 if shape == "polygon" else None
        plan = generate_program(
            shape=shape, center_x=500, center_y=400, circumradius_mm=100, sides=sides
        )
        assert len(plan.points) >= 4
