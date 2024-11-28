from abc import ABC, abstractmethod
from typing import List, Tuple, Literal
import numpy as np
from tnfemesh.domain.subdomain import Subdomain
from tnfemesh.domain.curve import Curve

CurvePosition = Literal["start", "end"]

class SubdomainConnection(ABC):
    @abstractmethod
    def validate(self, subdomains: List[Subdomain]):
        """
        Validates that the connection is consistent with the provided subdomains.

        Args:
            subdomains (List[Subdomain]): List of subdomains in the domain.
        """
        pass


class VertexConnection2D(SubdomainConnection):
    def __init__(self, connection: List[Tuple[int, int, CurvePosition]]):
        """
        Initialize a 2D vertex connection.
        The subdomain indexes reference into the list of subdomains
        passed to the Domain constructor.

        Args:
            connection (List[Tuple[int, int, CurvePosition]]):
                List of subdomains sharing this vertex.
                Each connection is a tuple of (subdomain index, curve index, position).
                Curve position is either "start" or "end".
        """
        self.connection = connection

    def validate(self, subdomains: List[Subdomain], tol: float = 1e-6):
        """
        Validate that all specified subdomains, curves, and positions have the given vertex.

        Args:
            subdomains (List[Subdomain]): List of subdomains in the domain.
            tol (float): Tolerance for point-wise comparison
        """

        curve0 = subdomains[self.connection[0][0]].curves[self.connection[0][1]]
        point0 = curve0.get_start() if self.connection[0][2] == "start" else curve0.get_end()
        for subdomain_idx, curve_idx, position in self.connection:
            curve = subdomains[subdomain_idx].curves[curve_idx]
            point = curve.get_start() if position == "start" else curve.get_end()

            if not np.allclose(point, point0, atol=tol):
                raise ValueError(
                    f"Subdomain {subdomain_idx}, curve {curve_idx}, {position} point {point} "
                    f"does not match the vertex {point0}."
                )

    def get_shared_vertex(self, subdomains: List[Subdomain]) -> np.ndarray:
        """
        Get the shared vertex between the connected subdomains.

        Args:
            subdomains (List[Subdomain]): List of subdomains in the domain.

        Returns:
            np.ndarray: Shared vertex coordinates.
        """
        curve0 = subdomains[self.connection[0][0]].curves[self.connection[0][1]]
        return curve0.get_start() if self.connection[0][2] == "start" else curve0.get_end()


class CurveConnection(SubdomainConnection):
    def __init__(self, subdomains_indices: Tuple[int, int], curve_indices: Tuple[int, int]):
        """
        Initialize a curve connection between two subdomains.
        Only two subdomains can be connected by a curve.

        Args:
            subdomains_indices (Tuple[int, int]):
                A tuple of two subdomain indices that share a curve.
            curve_indices (Tuple[int, int]):
                A tuple of two curve indices in the respective subdomains.
        """

        self.subdomains_indices = subdomains_indices
        self.curve_indices = curve_indices

    def validate(self,
                 subdomains: List[Subdomain],
                 num_points: int = 100,
                 tol: float = 1e-6):
        """
        Validate that the curves are approximately equal.

        Args:
            subdomains (List[Subdomain]): List of subdomains in the domain.
            num_points (int): Number of points to sample along the curve.
            tol (float): Tolerance for point-wise comparison.
        """

        sub1_idx, sub2_idx = self.subdomains_indices
        curve1_idx, curve2_idx = self.curve_indices

        curve1 = subdomains[sub1_idx].curves[curve1_idx]
        curve2 = subdomains[sub2_idx].curves[curve2_idx]

        if not curve1.equals(curve2, num_points=num_points, tol=tol):
            raise ValueError(
                f"Curves {curve1_idx} of subdomain {sub1_idx} and {curve2_idx} of subdomain {sub2_idx} are not equal."
            )

    def get_shared_curve(self, subdomains: List[Subdomain]) -> Curve:
        """
        Get the shared curve between the connected subdomains.

        Args:
            subdomains (List[Subdomain]): List of subdomains in the domain.

        Returns:
            Curve: Shared curve.
        """
        sub1_idx, sub2_idx = self.subdomains_indices
        curve1_idx, curve2_idx = self.curve_indices

        return subdomains[sub1_idx].curves[curve1_idx]