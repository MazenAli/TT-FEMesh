from abc import ABC, abstractmethod
from typing import List, Tuple, Literal
import numpy as np
from ttfemesh.domain.subdomain import Subdomain
from ttfemesh.domain.curve import Curve

CurvePosition = Literal['start', 'end']


class SubdomainConnection(ABC):
    """Generic subdomain connection class."""
    @abstractmethod
    def validate(self, subdomains: List[Subdomain]):
        """
        Validates that the connection is consistent with the provided subdomains.

        Args:
            subdomains (List[Subdomain]): List of subdomains in the domain.
        """
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Dimension of the subdomain connection."""
        pass

    @property
    @abstractmethod
    def num_connected_subdomains(self) -> int:
        """Number of connected subdomains."""
        pass


class SubdomainConnection2D(SubdomainConnection):
    """Generic 2D subdomain connection class."""
    @property
    def dimension(self) -> int:
        """Dimension of the subdomain connection."""
        return 2


class VertexConnection2D(SubdomainConnection2D):
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

    @property
    def num_connected_subdomains(self) -> int:
        """Number of connected subdomains."""
        return len(self.connection)

    def validate(self, subdomains: List[Subdomain], tol: float = 1e-6):
        """
        Validate that all specified subdomains, curves, and positions share the given vertex.

        Args:
            subdomains (List[Subdomain]): List of subdomains in the domain.
            tol (float): Tolerance for point-wise comparison

        Raises:
            ValueError: If the vertex connections are not consistent.
        """
        self._validate_idxs(subdomains)
        if self.num_connected_subdomains < 2:
            raise ValueError("Vertex connection must have at least two connected subdomains.")

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

    def get_connection_pairs(self) -> List[Tuple[Tuple[int, int],
                                                          Tuple[int, int],
                                                          Tuple[CurvePosition, CurvePosition]]]:
        """
        Get all unique pairs of connected subdomains with their curve indices and positions.

        Returns:
            List[Tuple[Tuple[int, int], Tuple[int, int], Tuple[CurvePosition, CurvePosition]]]:
                List of connected subdomain pairs with the indexing
                [(subdomain0, subdomain1), (curve0, curve1), (position0, position1)].
        """
        pairs = []
        n = len(self.connection)

        for i in range(n):
            for j in range(i + 1, n):
                subd1, curve_idx1, curve_pos1 = self.connection[i]
                subd2, curve_idx2, curve_pos2 = self.connection[j]
                pairs.append(((subd1, subd2), (curve_idx1, curve_idx2), (curve_pos1, curve_pos2)))

        return pairs

    def get_shared_vertex(self, subdomains: List[Subdomain]) -> np.ndarray:
        """
        Get the shared vertex between the connected subdomains.

        Args:
            subdomains (List[Subdomain]): List of subdomains in the domain.

        Returns:
            np.ndarray: Shared vertex coordinates.
        """
        self._validate_idxs(subdomains)

        curve0 = subdomains[self.connection[0][0]].curves[self.connection[0][1]]
        return curve0.get_start() if self.connection[0][2] == "start" else curve0.get_end()

    def _validate_idxs(self, subdomains: List[Subdomain]):
        """ Validate that the subdomain and curve indices are within bounds."""
        for subdomain_idx, curve_idx, position in self.connection:
            if subdomain_idx >= len(subdomains):
                raise ValueError(f"Subdomain index {subdomain_idx} is out of bounds.")
            if curve_idx >= len(subdomains[subdomain_idx].curves):
                raise ValueError(f"Curve index {curve_idx} is out of bounds.")

    def __repr__(self):
        return f"VertexConnection2D({self.connection})"


class CurveConnection2D(SubdomainConnection2D):
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

    @property
    def num_connected_subdomains(self) -> int:
        """Number of connected subdomains."""
        return 2

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
        self._validate_idxs(subdomains)

        sub1_idx, sub2_idx = self.subdomains_indices
        curve1_idx, curve2_idx = self.curve_indices

        curve1 = subdomains[sub1_idx].curves[curve1_idx]
        curve2 = subdomains[sub2_idx].curves[curve2_idx]

        if not curve1.equals(curve2, num_points=num_points, tol=tol):
            raise ValueError(
                f"Curves {curve1_idx} of subdomain {sub1_idx}"
                f" and {curve2_idx} of subdomain {sub2_idx} are not equal."
            )

    def get_shared_curve(self, subdomains: List[Subdomain]) -> Curve:
        """
        Get the shared curve between the connected subdomains.

        Args:
            subdomains (List[Subdomain]): List of subdomains in the domain.

        Returns:
            Curve: Shared curve.
        """
        self._validate_idxs(subdomains)

        sub1_idx, sub2_idx = self.subdomains_indices
        curve1_idx, curve2_idx = self.curve_indices

        return subdomains[sub1_idx].curves[curve1_idx]

    def _validate_idxs(self, subdomains: List[Subdomain]):
        """ Validate that the subdomain and curve indices are within bounds."""
        sub1_idx, sub2_idx = self.subdomains_indices
        curve1_idx, curve2_idx = self.curve_indices

        if sub1_idx >= len(subdomains):
            raise ValueError(f"Subdomain index {sub1_idx} is out of bounds.")
        if sub2_idx >= len(subdomains):
            raise ValueError(f"Subdomain index {sub2_idx} is out of bounds.")
        if curve1_idx >= len(subdomains[sub1_idx].curves):
            raise ValueError(f"Curve index {curve1_idx} is out of bounds.")
        if curve2_idx >= len(subdomains[sub2_idx].curves):
            raise ValueError(f"Curve index {curve2_idx} is out of bounds.")

    def __repr__(self):
        return f"CurveConnection({self.subdomains_indices}, {self.curve_indices})"