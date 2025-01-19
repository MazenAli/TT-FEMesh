from typing import Union, List
from ttfemesh.domain.domain import Domain
from ttfemesh.quadrature.quadrature import QuadratureRule
from ttfemesh.basis.basis import TensorProductBasis


class DomainMesh:
    """
    A class that ties together the domain, subdomain meshes, and basis functions.
    It provides an interface for the element jacobians of the subdomains,
    the element to global index maps, the boundary masks and the concatenation maps.
    """

    def __init__(self,
                 domain: Domain,
                 quadrature_rule: Union[QuadratureRule, List[QuadratureRule]],
                 mesh_size_exponent: Union[int, List[int]],
                 basis: TensorProductBasis):
        """
        Initialize a DomainMesh.

        Args:
            domain (Domain): The domain containing subdomains and their connections.
            quadrature_rule (Union[QuadratureRule, List[QuadratureRule]]):
                Quadrature rule(s) for integration.
                If a single quadrature rule is provided, it is used for all subdomains.
                If a list is provided, it must have the same length as the number of subdomains.
                The indexing of the quadrature rules must match the indexing of the subdomains.
            mesh_size_exponent (Union[int, List[int]]): Discretization size exponent(s).
                If a single value is provided, it is used for all subdomains.
                If a list is provided, it must have the same length as the number of subdomains.
                The indexing of the mesh size exponents must match the indexing of the subdomains.
            basis (TensorProductBasis): The basis functions for the domain.

        Raises:
            ValueError: If the number of quadrature rules or mesh size exponents is invalid.
        """
        num_subdomains = domain.num_subdomains

        # Ensure quadrature_rule is a list with one per subdomain
        if isinstance(quadrature_rule, QuadratureRule):
            self.quadrature_rules = [quadrature_rule] * num_subdomains
        elif isinstance(quadrature_rule, list) and len(quadrature_rule) == num_subdomains:
            self.quadrature_rules = quadrature_rule
        else:
            raise ValueError(
                f"Invalid number of quadrature rules: expected 1 or {num_subdomains}, "
                f"got {len(quadrature_rule) if isinstance(quadrature_rule, list) else 'scalar'}."
            )

        if isinstance(mesh_size_exponent, int):
            self.mesh_size_exponents = [mesh_size_exponent] * num_subdomains
        elif isinstance(mesh_size_exponent, list) and len(mesh_size_exponent) == num_subdomains:
            self.mesh_size_exponents = mesh_size_exponent
        else:
            raise ValueError(
                f"Invalid number of mesh size exponents: expected 1 or {num_subdomains}, "
                f"got {len(mesh_size_exponent) if isinstance(mesh_size_exponent, list) else 'scalar'}."
            )

        self.domain = domain
        self.basis = basis
        self.subdomain_meshes = self._create_subdomain_meshes()

    def _create_subdomain_meshes(self):
        pass

    def __repr__(self) -> str:
        return (f"DomainMesh(domain={self.domain}, "
                f"mesh_size_exponents={self.mesh_size_exponents}, "
                f"quadrature_rules={self.quadrature_rules}, "
                f"basis={self.basis})")