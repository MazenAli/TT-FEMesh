from abc import ABC, abstractmethod
from typing import List, Tuple
from tnfemesh.domain.subdomain import Subdomain


class BoundaryCondition(ABC):
    """
    Abstract base class for boundary conditions.
    """

    @abstractmethod
    def validate(self, subdomains: List[Subdomain]):
        """
        Validate the boundary condition with respect to the subdomains.

        Args:
            subdomains (List[Subdomain]): List of subdomains in the domain.
        """
        pass


class DirichletBoundary2D(BoundaryCondition):
    """
    Implements a Dirichlet boundary condition for a 2D curve.
    Boundary values are implicitly assumed to be zero.
    """

    def __init__(self, boundary: List[Tuple[int, int]]):
        """
        Initialize the Dirichlet boundary condition.

        Args:
            boundary (List[Tuple[int, int]]):
                List of subdomain and curve indices for the Dirichlet boundary.
        """
        self.boundary = boundary

    def validate(self, subdomains: List[Subdomain]):
        """
        Validate the boundary condition. Ensure that the specified curve exists.

        Args:
            subdomains (List[Subdomain]): List of subdomains in the domain.
        """
        for subdomain_idx, curve_idx in self.boundary:
            if subdomain_idx >= len(subdomains):
                raise ValueError(f"Subdomain index {subdomain_idx} out of range.")
            if curve_idx >= len(subdomains[subdomain_idx].curves):
                raise ValueError(f"Curve index {curve_idx} out of range"
                                 f" for subdomain {subdomain_idx}.")
