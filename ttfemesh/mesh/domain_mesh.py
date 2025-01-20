from abc import ABC, abstractmethod
from typing import Union, List, Optional, Dict
import numpy as np
from ttfemesh.types import TensorTrain
from ttfemesh.domain.domain import Domain
from ttfemesh.quadrature.quadrature import QuadratureRule
from ttfemesh.basis.basis import TensorProductBasis, Num2Side
from ttfemesh.mesh.subdomain_mesh import SubdomainMesh, SubdomainMesh2D
from ttfemesh.tn_tools.tensor_cross import TTCrossConfig


class DomainMesh(ABC):
    """
    DomainMesh base class that ties together the domain, subdomain meshes, and basis functions.
    It provides an interface for the element jacobians of the subdomains,
    the element to global index maps, the boundary masks and the concatenation maps.
    """

    def __init__(self,
                 domain: Domain,
                 quadrature_rule: Union[QuadratureRule, List[QuadratureRule]],
                 mesh_size_exponent: Union[int, List[int]],
                 basis: TensorProductBasis,
                 tt_cross_config: Optional[TTCrossConfig] = None):
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
            cross_config (Optional[TTCrossConfig]):
                Optional configuration for tensor cross approximation.
                If None, the default configuration is used.

        Raises:
            ValueError: If the number of quadrature rules or mesh size exponents is invalid.
        """

        num_subdomains = domain.num_subdomains

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
        self._tt_cross_config = tt_cross_config
        self.subdomain_meshes = self._create_subdomain_meshes()

    #TODO: add concatenation maps
    @abstractmethod
    def _create_subdomain_meshes(self):
        num_subdomains = self.domain.num_subdomains
        subdomain_meshes = []
        for i in range(num_subdomains):
            subdomain = self.domain.get_subdomain(i)
            mesh_size_exponent = self.mesh_size_exponents[i]
            quadrature_rule = self.quadrature_rules[i]
            tt_cross_config = self._tt_cross_config
            subdomain_mesh = SubdomainMesh(subdomain,
                                           quadrature_rule,
                                           mesh_size_exponent,
                                           tt_cross_config)
            subdomain_meshes.append(subdomain_mesh)

        return subdomain_meshes

    def get_subdomain_mesh(self, subdomain_index: int) -> SubdomainMesh:
        """
        Get the SubdomainMesh for a subdomain.

        Args:
            subdomain_index (int): The index of the subdomain.

        Returns:
            SubdomainMesh: The SubdomainMesh for the subdomain.

        Raises:
            ValueError: If the subdomain index is invalid.
        """
        self._validate_subdomain_index(subdomain_index)
        return self.subdomain_meshes[subdomain_index]

    def get_element2global_index_maps(self, subdomain_index: int) -> np.ndarray:
        """
        Get the TT-representation of transformations mapping from element index to global basis
        function index for all reference basis functions on a reference element in a subdomain.
        This map depends on the type of basis functions used and
        the discretization size of the subdomain.

        Args:
            subdomain_index (int): The index of the subdomain.

        Returns:
            np.ndarray: A matrix of TT-representations, i.e.,
                each element of the matrix is a TT-vector.
                Indexing of the matrix corresponds to the
                reference basis function indexing.
                For example, for a bilinear basis in 2D, the indexing is:
                (i, j) where i and j are the indices of the basis functions in x and y direction
                Specifically, (0, 0), (1, 0), (0, 1) and (1, 1),
                representing the four basis functions
                corresponding to the four vertices of the reference element:
                lower left, lower right, upper right and upper left, respectively.
                See also the documentation for the chosen Basis class.

        Raises:
            ValueError: If the subdomain index is invalid.
        """
        self._validate_subdomain_index(subdomain_index)

        mesh_size_exponent = self.mesh_size_exponents[subdomain_index]
        ttmaps = self.basis.get_all_element2global_ttmaps(mesh_size_exponent)

        return ttmaps

    def get_dirichlet_masks(self) -> Dict[int, TensorTrain]:
        """
        Get the dirichlet boundary masks.

        Returns:
            Dict[TensorTrain]: A dictionary where the keys are subdomain indices,
                and the values are TT-representations of the boundary masks.
        """

        boundary_condition = self.domain.boundary_condition
        if boundary_condition is None:
            print("No boundary condition specified.")
            return None

        grouped = boundary_condition.group_by_subdomain()
        boundary_masks = {}
        for subdomain_index, curve_indices in grouped.items():
            mesh_size_exponent = self.mesh_size_exponents[subdomain_index]
            sides = [Num2Side[i] for i in curve_indices]
            boundary_mask = self.basis.get_dirichlet_mask(mesh_size_exponent, *sides)
            boundary_masks[subdomain_index] = boundary_mask

        return boundary_masks


    def _validate_subdomain_index(self, subdomain_index: int):
        if subdomain_index < 0 or subdomain_index >= self.domain.num_subdomains:
            raise ValueError(f"Invalid subdomain index: {subdomain_index}. "
                             f"Valid indices are in the range [0, {self.domain.num_subdomains}).")

    def __repr__(self) -> str:
        return (f"DomainMesh(domain={self.domain}, "
                f"mesh_size_exponents={self.mesh_size_exponents}, "
                f"quadrature_rules={self.quadrature_rules}, "
                f"basis={self.basis})")


class DomainMesh2D(DomainMesh):
    """Mesh for 2D domains."""
    def _create_subdomain_meshes(self):
        num_subdomains = self.domain.num_subdomains
        subdomain_meshes = []
        for i in range(num_subdomains):
            subdomain = self.domain.get_subdomain(i)
            mesh_size_exponent = self.mesh_size_exponents[i]
            quadrature_rule = self.quadrature_rules[i]
            tt_cross_config = self._tt_cross_config
            subdomain_mesh = SubdomainMesh2D(subdomain,
                                             quadrature_rule,
                                             mesh_size_exponent,
                                             tt_cross_config)
            subdomain_meshes.append(subdomain_mesh)

        return subdomain_meshes

    def __repr__(self) -> str:
        return (f"DomainMesh2D(domain={self.domain}, "
                f"mesh_size_exponents={self.mesh_size_exponents}, "
                f"quadrature_rules={self.quadrature_rules}, "
                f"basis={self.basis})")