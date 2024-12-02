from abc import ABC, abstractmethod
from typing import Any
from tnfemesh.domain import Subdomain, Subdomain2D
from tnfemesh.quadrature import QuadratureRule
import numpy as np


class SubdomainMesh(ABC):
    """Subdomain mesh for a finite element problem."""

    def __init__(self,
                 subdomain: Subdomain,
                 quadrature_rule: QuadratureRule,
                 mesh_size_exponent: int):
        """
        Initialize a subdomain mesh.

        Args:
            subdomain (Subdomain): The subdomain to mesh.
            quadrature_rule (QuadratureRule): The quadrature rule to use.
            mesh_size_exponent (int): The exponent of the discretization size.
                The discretization size is 2**(mesh_size_exponent) per dimension.
        """
        self.subdomain = subdomain
        self.quadrature_rule = quadrature_rule
        self.mesh_size_exponent = mesh_size_exponent

    @abstractmethod
    def element_jacobian(self) -> Any:
        """Return the Jacobian function for the element transformation."""
        pass


class SubdomainMesh2D(SubdomainMesh):

    def __init__(self,
                 subdomain: Subdomain2D,
                 quadrature_rule: QuadratureRule,
                 mesh_size_exponent: int):
        super().__init__(subdomain, quadrature_rule, mesh_size_exponent)

    def element_jacobian(self,
                         index_x: int,
                         index_y: int,
                         xi: float,
                         eta: float) -> np.ndarray:

        side0 = self.subdomain.curves[0]
        side1 = self.subdomain.curves[1]
        side2 = self.subdomain.curves[2]
        side3 = self.subdomain.curves[3]
