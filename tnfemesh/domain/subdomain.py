from abc import ABC, abstractmethod
from typing import List
import numpy as np
import matplotlib.pyplot as plt
from tnfemesh.domain.curve import Curve, Line2D


class Subdomain(ABC):
    """
    Abstract base class for a subdomain in the domain.
    The subdomains are intended to "glue" more complex domains together.
    """

    @abstractmethod
    def validate(self):
        """Ensure that the subdomain is valid."""
        pass

    @abstractmethod
    def plot(self):
        """Plot the subdomain and its boundaries."""
        pass


class Subdomain2D(Subdomain):
    def __init__(self, curves: List[Curve]):
        """
        Initialize a 2D subdomain defined by 4 boundary curves.
        The curves must connect properly to form a closed subdomain.
        The start and end points of the curves must be ordered counter-clockwise.

        Args:
            curves (List[Curve]): List of 4 boundary curves.
        """
        if len(curves) != 4:
            raise ValueError("A 2D subdomain must be defined by exactly 4 curves.")
        self.curves = curves
        self.validate()

    def get_curve(self, index: int) -> Curve:
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

    def validate(self, tol: float = 1e-6):
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

    def plot(self, num_points: int = 100):
        for curve in self.curves:
            t_vals = np.linspace(-1, 1, num_points)
            points = curve.evaluate(t_vals)
            plt.plot(points[:, 0], points[:, 1], label=f"{type(curve).__name__}")

        plt.title("Subdomain")
        plt.axis("equal")
        plt.show()

    def __repr__(self):
        points = [curve.get_start() for curve in self.curves]
        return f"Subdomain2D(points={points})"


class Quad(Subdomain2D):
    """Quadrilateral subdomain defined by 4 boundary lines."""
    def __init__(self, curves: List[Line2D]):
        super().__init__(curves)

    def __repr__(self):
        points = [curve.get_start() for curve in self.curves]
        return f"Quad(points={points})"
