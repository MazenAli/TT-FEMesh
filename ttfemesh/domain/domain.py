from abc import ABC, abstractmethod
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

from ttfemesh.domain.boundary_condition import BoundaryCondition, DirichletBoundary2D
from ttfemesh.domain.subdomain import Subdomain, Subdomain2D
from ttfemesh.domain.subdomain_connection import (
    CurveConnection2D,
    SubdomainConnection,
    SubdomainConnection2D,
    VertexConnection2D,
)


class Domain(ABC):
    """Domain class that contains subdomains and their connections."""

    def __init__(
        self,
        subdomains: Sequence[Subdomain],
        connections: Sequence[SubdomainConnection],
        boundary_condition: Optional[BoundaryCondition] = None,
    ):
        """
        Initialize a domain with subdomains and their connections.

        Args:
            subdomains (Sequence[Subdomain]): List of subdomains in the domain.
            connections (Sequence[SubdomainConnection]): List of connections between subdomains.
            boundary_condition (Optional[BoundaryCondition]): Optional boundary condition.
        """
        self.subdomains = subdomains
        self.connections = connections
        self.boundary_condition = boundary_condition
        self.validate()

    @property
    def num_subdomains(self) -> int:
        """Number of subdomains in the domain."""
        return len(self.subdomains)

    @property  # noqa
    def num_connections(self) -> int:
        """Number of connections in the domain."""
        return len(self.connections)

    def get_connections(self) -> Sequence[SubdomainConnection]:
        """Get the list of connections."""
        return self.connections

    def get_subdomain(self, idx) -> Subdomain:
        """
        Get a subdomain by index.

        Args:
            idx (int): Index of the subdomain to get.

        Returns:
            Subdomain: The subdomain at the given index.

        Raises:
            ValueError: If the index is out of bounds.
        """
        if idx < 0 or idx >= len(self.subdomains):
            raise ValueError(
                f"Invalid subdomain index: {idx}. " f"Must be in [0, {len(self.subdomains)})."
            )

        return self.subdomains[idx]

    def validate(self):
        """Validates that the connections are consistent with the subdomains."""
        for connection in self.connections:
            connection.validate(self.subdomains)
        if self.boundary_condition:
            self.boundary_condition.validate(self.subdomains)

    def __repr__(self):
        return f"Domain({len(self.subdomains)} subdomains, {len(self.connections)} connections)"

    def plot(self):
        "Plot the subdomains with connections."
        for subdomain in self.subdomains:
            subdomain.plot()

    @property
    @abstractmethod
    def dimension(self):  # pragma: no cover
        """The dimension of the domain."""
        pass


class Domain2D(Domain):
    """A 2D domain with subdomains and their connections.

    Example:
    >>> from ttfemesh.domain import RectangleFactory, CurveConnection2D, VertexConnection2D
    >>> from ttfemesh.domain import DirichletBoundary2D, Domain2D
    >>> lower_left = (0, 0)
    >>> upper_right = (2, 1)
    >>> rectangle1 = RectangleFactory.create(lower_left, upper_right)

    >>> lower_left = (2, 0)
    >>> upper_right = (3, 1)
    >>> rectangle2 = RectangleFactory.create(lower_left, upper_right)

    >>> lower_left = (-2, 1)
    >>> upper_right = (0, 2)
    >>> rectangle3 = RectangleFactory.create(lower_left, upper_right)

    >>> domain_idxs = [0, 1]
    >>> curve_idxs = [1, 3]
    >>> edge = CurveConnection2D(domain_idxs, curve_idxs)

    >>> vertex_idxs = [(0, 3, "start"), (2, 0, "end")]
    >>> vertex = VertexConnection2D(vertex_idxs)

    >>> bc = DirichletBoundary2D([(1, 1), (2, 3)])

    >>> domain = Domain2D([rectangle1, rectangle2, rectangle3], [edge, vertex], bc)
    >>> domain.plot()
    """

    def __init__(
        self,
        subdomains: Sequence[Subdomain2D],
        connections: Sequence[SubdomainConnection2D],
        boundary_condition: Optional[DirichletBoundary2D] = None,
    ):
        """
        Initialize a 2D domain with subdomains and their connections.

        Args:
            subdomains (Sequence[Subdomain2D]): List of 2D subdomains in the domain.
            connections (Sequence[SubdomainConnection2D]): List of connections between subdomains.
            boundary_condition (Optional[DirichletBoundary2D]): Optional 2D boundary condition.
        """
        super().__init__(subdomains, connections, boundary_condition)

    @property
    def dimension(self) -> int:
        """The dimension of the domain."""
        return 2

    def plot(self, num_points=100):
        """
        Plot the domain and its subdomains with connections.

        Args:
            num_points (int): Number of points to sample.
        """

        for subdomain in self.subdomains:
            for curve in subdomain.curves:
                t = np.linspace(-1, 1, 100)
                points = np.array(curve.evaluate(t))
                plt.plot(points[:, 0], points[:, 1], label="Subdomain")

        for connection in self.connections:
            if isinstance(connection, VertexConnection2D):
                shared_vertex = connection.get_shared_vertex(self.subdomains)
                plt.plot(shared_vertex[0], shared_vertex[1], "ro", label="Shared Vertex")
            elif isinstance(connection, CurveConnection2D):
                curve = connection.get_shared_curve(self.subdomains)
                curve_points = curve.evaluate(np.linspace(-1, 1, num_points))
                plt.plot(curve_points[:, 0], curve_points[:, 1], "g--", label="Shared Curve")

        if self.boundary_condition:
            for subdomain_idx, curve_idx in self.boundary_condition.boundary:
                curve = self.subdomains[subdomain_idx].curves[curve_idx]
                curve_points = curve.evaluate(np.linspace(-1, 1, num_points))
                plt.plot(curve_points[:, 0], curve_points[:, 1], "k--", label="Boundary Condition")

        plt.axis("equal")
        plt.title("Domain Plot")
        plt.show()

    def __repr__(self):
        return f"Domain2D({len(self.subdomains)} subdomains, {len(self.connections)} connections)"
