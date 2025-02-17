from abc import ABC, abstractmethod
from typing import Tuple

from ttfemesh.domain.curve import Line2D
from ttfemesh.domain.subdomain import Quad


class SubdomainFactory(ABC):
    """
    Abstract base class for a subdomain factory.
    """

    @staticmethod
    @abstractmethod
    def create():
        pass


class RectangleFactory(SubdomainFactory):
    """
    Factory class for creating rectangle subdomains.
    """

    @staticmethod
    def create(bottom_left: Tuple[float, float], top_right: Tuple[float, float]) -> Quad:
        """
        Create a rectangle subdomain defined by the bottom-left and top-right corners.

        Args:
            bottom_left (Tuple[float, float]): Coordinates of the bottom-left corner.
            top_right (Tuple[float, float]): Coordinates of the top-right corner.
        Returns:
            Quad: A rectangle subdomain.
        """

        x0, y0 = bottom_left
        x1, y1 = top_right
        points = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        return Quad([Line2D(points[i], points[(i + 1) % 4]) for i in range(4)])


class QuadFactory(SubdomainFactory):
    """
    Factory class for creating quadrilateral subdomains.
    """

    @staticmethod
    def create(
        p1: Tuple[float, float],
        p2: Tuple[float, float],
        p3: Tuple[float, float],
        p4: Tuple[float, float],
    ) -> Quad:
        """
        Create a trapezoid subdomain defined by the four corner points.
        Points must be ordered counter-clockwise.

        Args:
            p1 (Tuple[float, float]): Coordinates of the first corner.
            p2 (Tuple[float, float]): Coordinates of the second corner.
            p3 (Tuple[float, float]): Coordinates of the third corner.
            p4 (Tuple[float, float]): Coordinates of the fourth corner.
        Returns:
            Subdomain2D: A trapezoid subdomain.
        """

        points = (p1, p2, p3, p4)
        return Quad([Line2D(points[i], points[(i + 1) % 4]) for i in range(4)])
