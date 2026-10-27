"""Plotting tests."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pygeosolve3d import Problem


def test_problem_plot_uses_three_dimensions():
    problem = Problem()
    problem.add_line("line", (1, 2, 3), (4, 6, 8))

    problem.plot(show=False)

    ax = plt.gcf().axes[0]
    assert ax.name == "3d"
    assert ax.get_xlabel() == "X"
    assert ax.get_ylabel() == "Y"
    assert ax.get_zlabel() == "Z"

    x, y, z = ax.lines[-1].get_data_3d()
    assert list(x) == [1, 4]
    assert list(y) == [2, 6]
    assert list(z) == [3, 8]

    plt.close("all")
