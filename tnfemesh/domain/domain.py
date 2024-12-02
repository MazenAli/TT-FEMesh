from typing import List, Optional
import matplotlib.pyplot as plt
import numpy as np
from tnfemesh.domain.subdomain import Subdomain
from tnfemesh.domain.subdomain_connection import (SubdomainConnection,
                                                  VertexConnection2D,
                                                  CurveConnection)
from tnfemesh.domain.boundary_condition import BoundaryCondition


class Domain:
    def __init__(self,
                 subdomains: List[Subdomain],
                 connections: List[SubdomainConnection],
                 boundary_condition: Optional[BoundaryCondition] = None):
        """
        Initialize a domain with subdomains and their connections.

        Args:
            subdomains (List[Subdomain]): List of subdomains in the domain.
            connections (List[SubdomainConnection]): List of connections between subdomains.
            boundary_condition (Optional[BoundaryCondition]): Optional boundary condition.
        """
        self.subdomains = subdomains
        self.connections = connections
        self.boundary_condition = boundary_condition
        self.validate()

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


class Domain2D(Domain):
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
                plt.plot(shared_vertex[0], shared_vertex[1], 'ro', label="Shared Vertex")
            elif isinstance(connection, CurveConnection):
                curve = connection.get_shared_curve(self.subdomains)
                curve_points = curve.evaluate(np.linspace(-1, 1, num_points))
                plt.plot(curve_points[:, 0], curve_points[:, 1], 'g--', label="Shared Curve")

        if self.boundary_condition:
            for subdomain_idx, curve_idx in self.boundary_condition.boundary:
                curve = self.subdomains[subdomain_idx].curves[curve_idx]
                curve_points = curve.evaluate(np.linspace(-1, 1, num_points))
                plt.plot(curve_points[:, 0], curve_points[:, 1], 'k--', label="Boundary Condition")

        plt.axis("equal")
        plt.title("Domain Plot")
        plt.show()