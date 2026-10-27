"""Plane geometry and line-plane angle constraint tests."""

import pytest

from pygeosolve3d import Problem


@pytest.mark.parametrize(
    "start,end,expected",
    [
        ((0, 0, 1), (1, 0, 1), 0),
        ((0, 0, 0), (0, 0, 1), 90),
        ((0, 0, 0), (1, 0, 1), 45),
    ],
)
def test_line_plane_angle(start, end, expected):
    problem = Problem()
    problem.add_line("line", start, end)
    problem.add_plane("plane", (0, 0, 0), (1, 0, 0), (0, 1, 0))

    constraint = problem.constrain_line_plane_angle("line", "plane", expected)

    assert constraint.value() == pytest.approx(expected)
    assert constraint.error() == pytest.approx(0)


def test_collinear_points_do_not_define_a_plane():
    problem = Problem()
    problem.add_plane("plane", (0, 0, 0), (1, 1, 1), (2, 2, 2))

    with pytest.raises(ValueError, match="do not define a plane"):
        problem.validate()


@pytest.mark.parametrize("angle", [-1, 91])
def test_line_plane_angle_rejects_out_of_range_targets(angle):
    problem = Problem()
    problem.add_line("line", (0, 0, 0), (1, 0, 0))
    problem.add_plane("plane", (0, 0, 0), (1, 0, 0), (0, 1, 0))

    with pytest.raises(ValueError, match="between 0 and 90"):
        problem.constrain_line_plane_angle("line", "plane", angle)


def test_least_squares_solves_line_plane_angle():
    problem = Problem()
    problem.add_line("line", (0, 0, 0), (1, 0, 1))
    problem.add_plane("ground", (0, 0, 0), (1, 0, 0), (0, 1, 0))
    problem.constrain_position("ground")
    constraint = problem.constrain_line_plane_angle("line", "ground", 0)

    result = problem.solve(solver="ls")

    assert result.success
    assert constraint.value() == pytest.approx(0, abs=1e-4)
