"""3D geometry tests."""

import pytest

from pygeosolve3d.geometry import Line3D


@pytest.mark.parametrize(
    "end, expected",
    (
        ((0, -1, 0), 90),
        ((0, 1, 0), -90),
    ),
)
def test_line3d_angle_to_preserves_clockwise_convention(end, expected):
    line = Line3D("line", (0, 0, 0), (1, 0, 0))
    other = Line3D("other", (0, 0, 0), end)

    assert line.angle_to(other) == pytest.approx(expected)


def test_line3d_angle_to_projects_directions_onto_normal_plane():
    line = Line3D("line", (0, 0, 0), (1, 0, 2))
    other = Line3D("other", (0, 0, 0), (0, 1, 3))

    assert line.angle_to(other) == pytest.approx(-90)


def test_line3d_angle_to_rejects_direction_parallel_to_normal():
    line = Line3D("line", (0, 0, 0), (0, 0, 1))
    other = Line3D("other", (0, 0, 0), (1, 0, 0))

    with pytest.raises(ValueError, match="parallel to the normal"):
        line.angle_to(other)
