from abc import ABC, abstractmethod
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

from ttfemesh.domain.curve import Curve, Line2D


class Subdomain(ABC):
    """
    Abstract base class for a subdomain in the domain.
    The subdomains are intended to "glue" more complex domains together.
    """

    @abstractmethod
    def _validate(self):  # pragma: no cover
        """Ensure that the subdomain is valid."""
        pass

    @abstractmethod
    def plot(self):  # pragma: no cover
        """Plot the subdomain and its boundaries."""
        pass


class Subdomain2D(Subdomain):
    def __init__(self, curves: Sequence[Curve]):
        """
        Initialize a 2D subdomain defined by 4 boundary curves.
        The curves must connect properly to form a closed subdomain.
        The start and end points of the curves must be ordered counter-clockwise.
        Note that the curves are considered to be ordered as follows:
        bottom (curve 0), right (curve 1), top (curve 2), left (curve 3).
        This is important for the boundary condition to work correctly.
        It may lead to confusion if, e.g., your curve 0 is visually
        the right edge of the domain.

        Args:
            curves (Sequence[Curve]): List of 4 boundary curves.
        """
        if len(curves) != 4:
            raise ValueError("A 2D subdomain must be defined by exactly 4 curves.")
        self.curves = curves
        self._validate()

    def get_curve(self, index: int) -> Curve:  # noqa
        """
        Get the curve at the specified index.

        Args:
            index (int): Index of the curve.

        Returns:
            Curve: The curve at the specified index.
        """

        if index < 0 or index >= 4:
            raise ValueError("Curve index must be in the range [0, 3].")

        return self.curves[index]

    def _check_connect(self, tol: float = 1e-6):
        """
        Ensure that the curves connect properly.

        Args:
            tol (float): Tolerance for point-wise comparison. Default is 1e-6.
        """

        for i in range(4):
            end = self.curves[i].get_end()
            next_start = self.curves[(i + 1) % 4].get_start()
            if not np.allclose(end, next_start, atol=tol):
                raise ValueError(f"Curves {i} and {(i + 1) % 4} do not connect properly.")

    def _check_orientation(self):
        """Check if the curves defining the domain are ordered counter-clockwise."""

        points = [curve.get_start() for curve in self.curves]
        signed_area = 0.0
        for i in range(len(points) - 1):
            x1, y1 = points[i]
            x2, y2 = points[i + 1]
            signed_area += x1 * y2 - x2 * y1

        if signed_area < 0:
            raise ValueError("The start points of curves are not ordered counter-clockwise.")

    def _validate(self):
        self._check_connect()
        self._check_orientation()

    def plot(self, num_points: int = 100):
        for i, curve in enumerate(self.curves):
            t_vals = np.linspace(-1, 1, num_points)
            points = curve.evaluate(t_vals)
            plt.plot(points[:, 0], points[:, 1], label=f"curve {i}")

        plt.title("Subdomain")
        plt.axis("equal")
        plt.show()

    def __repr__(self):
        points = [curve.get_start() for curve in self.curves]
        return f"Subdomain2D(points={points})"


class Quad(Subdomain2D):
    """
    Quadrilateral subdomain defined by 4 boundary lines.
    Note that the lines are considered to be ordered as follows:
    bottom (line 0), right (line 1), top (line 2), left (line 3).
    This is important for the boundary condition to work correctly.
    It may lead to confusion if, e.g., your line 0 is visually
    the right edge of the domain.
    """

    def __init__(self, curves: Sequence[Line2D]):
        super().__init__(curves)

    def __repr__(self):
        points = [curve.get_start() for curve in self.curves]
        return f"Quad(points={points})"
