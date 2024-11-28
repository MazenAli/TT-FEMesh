from abc import ABC, abstractmethod
from enum import Enum
from typing import List
import numpy as np
import matplotlib.pyplot as plt
from tnfemesh.domain.curve import Curve


class SubdomainType(Enum):
    GENERAL = "general"
    QUADRILATERAL = "quadrilateral"


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
    def __init__(self, curves: List[Curve],
                 subdomain_type: SubdomainType = SubdomainType.GENERAL):
        """
        Initialize a 2D subdomain defined by 4 boundary curves.
        The curves must connect properly to form a closed subdomain.
        The start and end points of the curves must be ordered counter-clockwise.

        Args:
            curves (List[Curve]): List of 4 boundary curves.
            subdomain_type (SubdomainType): Type of the subdomain.
                Default is SubdomainType.GENERAL.
        """
        if len(curves) != 4:
            raise ValueError("A 2D subdomain must be defined by exactly 4 curves.")
        self.curves = curves
        self.validate()
        self.subdomain_type = subdomain_type

    def validate(self, tol: float = 1e-6):
        """
        Ensure that the curves connect properly.

        Args:
            tol (float): Tolerance for point-wise comparison. Default is 1e-6.
        """

        for i in range(4):
            end = self.curves[i].evaluate(np.array([1]))[0]
            next_start = self.curves[(i + 1) % 4].evaluate(np.array([0]))[0]
            if not np.allclose(end, next_start, atol=tol):
                raise ValueError(f"Curves {i} and {(i + 1) % 4} do not connect properly.")

    def plot(self, num_points: int = 100):
        for curve in self.curves:
            t_vals = np.linspace(0, 1, num_points)
            points = curve.evaluate(t_vals)
            plt.plot(points[:, 0], points[:, 1], label=f"{type(curve).__name__}")

        plt.title("Subdomain")
        plt.axis("equal")
        plt.show()

    def __repr__(self):
        return f"Subdomain2D(type={self.subdomain_type.value}, num_curves={len(self.curves)})"
