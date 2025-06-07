import matplotlib.pyplot as plt
import numpy as np


def plot_curve_with_tangents(curve, title, num_points=50):
    """
    Plot a curve and its tangent vectors on a 2D plane.

    Args:
        curve (Curve): An instance of a Curve object.
        title (str): Title of the plot.
        num_points (int): Number of points to sample along the curve.
    """
    t_vals = np.linspace(-1, 1, num_points)
    points = curve.evaluate(t_vals)
    tangents = curve.tangent(t_vals)

    plt.plot(points[:, 0], points[:, 1], label="Curve", color="blue")

    for i in range(0, num_points, max(1, num_points // 10)):
        p = points[i]
        t = tangents[i]
        plt.quiver(
            p[0],
            p[1],
            t[0],
            t[1],
            angles="xy",
            scale_units="xy",
            scale=1.0,
            color="red",
            width=0.003,
            label="Tangent" if i == 0 else "",
        )

    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")
    plt.legend()
    plt.grid()
