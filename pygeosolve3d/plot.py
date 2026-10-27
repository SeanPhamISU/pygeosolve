"""Plotting."""

import numpy as np
import matplotlib.pyplot as plt


try:
    from itertools import pairwise
except ImportError:
    # Must be Python < 3.10.
    from itertools import tee

    def pairwise(iterable):
        a, b = tee(iterable)
        next(b, None)
        return zip(a, b)


def plot_problem(problem):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    for primitive in problem.primitives.values():
        for point in primitive.points:
            ax.plot([point.x], [point.y], [point.z], marker="x", color="red")

        for p1, p2 in pairwise(primitive.points):
            ax.plot(
                [p1.x, p2.x],
                [p1.y, p2.y],
                [p1.z, p2.z],
                color="blue",
            )
            ax.text(
                (p1.x + p2.x) / 2,
                (p1.y + p2.y) / 2,
                (p1.z + p2.z) / 2,
                primitive.name,
            )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_box_aspect(
        (
            np.ptp(ax.get_xlim3d()),
            np.ptp(ax.get_ylim3d()),
            np.ptp(ax.get_zlim3d()),
        )
    )


def show():
    plt.show()
