from abc import ABC, abstractmethod
from typing import Tuple, List, Callable, Optional
import warnings
import matplotlib.pyplot as plt
import numpy as np
import torch
from tnfemesh.domain import Subdomain, Subdomain2D, Quad
from tnfemesh.quadrature import QuadratureRule
from tnfemesh.mesh.mesh_utils import qindex2dtuple as index_map2d
from tnfemesh.tn_tools.tensor_cross import (gen_teneva_indices,
                                            anova_init_tensor_train,
                                            tensor_train_cross_approximation,
                                            TTCrossConfig)
from tnfemesh.utils.array import ensure_2d
from tnfemesh.tn_tools import interpolate_linear2d
from tnfemesh.types import TensorTrain


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
        self._index_map = None


    @abstractmethod
    def ref2domain_map(self):
        """Return the reference to domain map."""
        pass

    @abstractmethod
    def ref2element_map(self):
        """Return the element transformation function."""
        pass

    @abstractmethod
    def ref2domain_jacobian(self):
        """Return the Jacobian function for the domain transformation."""
        pass

    @abstractmethod
    def get_jacobian_tensor_networks(self):
        """
        Compute the tensor network approximating the Jacobian evaluated on all elements.
        The tensor index corresponds to the element index.
        This is done for each Jacobian component and each quadrature point within the element.
        The output is thus a total of 4*(num_quadrature_points_per_element) tensor networks.
        """
        pass

    @abstractmethod
    def plot(self):
        """Plot the subdomain mesh."""
        pass

    @abstractmethod
    def _validate_idxs(self):
        """Validate indices."""
        pass

    @abstractmethod
    def _validate_ref_coords(self):
        """Validate reference element coordinates."""
        pass


class SubdomainMesh2D(SubdomainMesh):
    """
    Subdomain mesh for a 2D finite element problem.

    Args:
        subdomain (Subdomain2D): The 2D subdomain to mesh.
        quadrature_rule (QuadratureRule): The quadrature rule to use.
        mesh_size_exponent (int): The exponent of the discretization size.
            The discretization size is 2**(mesh_size_exponent) per dimension.
        tt_cross_config (TTCrossConfig, optional):
            Configuration for the tensor train cross approximation.
            Defaults to using an info dict and rest are default settings.
    """

    def __init__(self,
                 subdomain: Subdomain2D,
                 quadrature_rule: QuadratureRule,
                 mesh_size_exponent: int,
                 tt_cross_config: Optional[TTCrossConfig] = None):
        super().__init__(subdomain, quadrature_rule, mesh_size_exponent)

        self._num_points1d = 2**mesh_size_exponent
        self._grid_step1d = 2.0 / (self._num_points1d - 1)
        self._index_map = index_map2d

        self._tt_cross_config = tt_cross_config
        if self._tt_cross_config is None:
            self._tt_cross_config = TTCrossConfig(info={})

        self._tca_strategy = self.__tca_default

    @property
    def num_points1d(self):
        """Number of points per dimension."""
        return self._num_points1d

    @property
    def num_points(self):
        """Total number of points."""
        return self.num_points1d**2

    @property
    def num_elements1d(self):
        """Number of elements per dimension."""
        return self.num_points1d - 1

    @property
    def num_elements(self):
        """Total number of elements."""
        return (self.num_points1d - 1)**2

    @property
    def index_map(self):
        """Index map for binary index to 2D tuple."""
        return self._index_map

    @property
    def tt_cross_config(self):
        """Configuration for the tensor train cross approximation."""
        return self._tt_cross_config

    @tt_cross_config.setter
    def tt_cross_config(self, config: TTCrossConfig):
        """Set the configuration for the tensor train cross approximation."""
        self._tt_cross_config = config

    def ref2domain_map(self, xi_eta: np.ndarray) -> np.ndarray:
        """
        Map from reference quadrilateral [-1, 1]^2 to domain.

        Args:
            xi_eta (np.ndarray): The reference coordinates in the quadrilateral.
                Of shape (num_points, 2) or (2,).

        Returns:
            np.ndarray: The physical coordinates in the domain.
                Of shape (num_points, 2).
        """

        xi_eta_ = ensure_2d(xi_eta)
        self._validate_ref_coords(xi_eta_)
        xi, eta = xi_eta_[:, 0], xi_eta_[:, 1]

        side0 = self.subdomain.curves[0]
        side1 = self.subdomain.curves[1]
        side2 = self.subdomain.curves[2]
        side3 = self.subdomain.curves[3]

        side0_vals = side0(xi)
        side1_vals = side1(eta)
        side2_vals = side2(-xi)
        side3_vals = side3(-eta)

        side0_x, side0_y = side0_vals[:, 0], side0_vals[:, 1]
        side1_x, side1_y = side1_vals[:, 0], side1_vals[:, 1]
        side2_x, side2_y = side2_vals[:, 0], side2_vals[:, 1]
        side3_x, side3_y = side3_vals[:, 0], side3_vals[:, 1]

        side0_start = side0.get_start()
        side1_start = side1.get_start()
        side2_start = side2.get_start()
        side3_start = side3.get_start()

        side0_x_start, side0_y_start = side0_start[0], side0_start[1]
        side1_x_start, side1_y_start = side1_start[0], side1_start[1]
        side2_x_start, side2_y_start = side2_start[0], side2_start[1]
        side3_x_start, side3_y_start = side3_start[0], side3_start[1]

        N_xi_eta_x = (0.5 * (1. - eta) * side0_x +
                      0.5 * (1. + xi) * side1_x +
                      0.5 * (1. + eta) * side2_x +
                      0.5 * (1. - xi) * side3_x -
                      0.25 * (1. - xi) * (1. - eta) * side0_x_start -
                      0.25 * (1. + xi) * (1. - eta) * side1_x_start -
                      0.25 * (1. + xi) * (1. + eta) * side2_x_start -
                      0.25 * (1. - xi) * (1. + eta) * side3_x_start)

        N_xi_eta_y = (0.5 * (1. - eta) * side0_y +
                      0.5 * (1. + xi) * side1_y +
                      0.5 * (1. + eta) * side2_y +
                      0.5 * (1. - xi) * side3_y -
                      0.25 * (1. - xi) * (1. - eta) * side0_y_start -
                      0.25 * (1. + xi) * (1. - eta) * side1_y_start -
                      0.25 * (1. + xi) * (1. + eta) * side2_y_start -
                      0.25 * (1. - xi) * (1. + eta) * side3_y_start)

        N_xi_eta = np.stack([N_xi_eta_x, N_xi_eta_y], axis=-1)

        return N_xi_eta

    def ref2element_map(self, index: Tuple[int, int], xi_eta: np.ndarray) -> np.ndarray:
        """
        Map from reference quadrilateral [-1, 1]^2 to element indexed by (index_x, index_y).

        Args:
            index (Tuple[int, int]): The 2D index of the element.
            xi_eta (np.ndarray): The reference coordinates in the quadrilateral.
                Of shape (num_points, 2) or (2,).

        Returns:
            np.ndarray: The physical coordinates in the element.
                Of shape (num_points, 2).
        """

        xi_eta_ = ensure_2d(xi_eta)
        self._validate_idxs(*index)
        self._validate_ref_coords(xi_eta_)

        index_x, index_y = index
        xi, eta = xi_eta_[:, 0], xi_eta_[:, 1]
        offset_xi = -1. + index_x * self._grid_step1d
        offset_eta = -1. + index_y * self._grid_step1d
        xi_rescaled = offset_xi + 0.5 * (1. + xi) * self._grid_step1d
        eta_rescaled = offset_eta + 0.5 * (1. + eta) * self._grid_step1d

        return self.ref2domain_map(np.column_stack((xi_rescaled, eta_rescaled)))

    def ref2domain_jacobian(self, xi_eta: np.ndarray) -> np.ndarray:
        """
        Compute the Jacobian of the reference to domain map.

        Args:
            xi_eta (np.ndarray): The reference coordinates in the quadrilateral.
                Of shape (num_points, 2) or (2,).

        Returns:
            np.ndarray: The Jacobian of the reference to domain map.
                Of shape (num_points, 2, 2).
        """

        xi_eta_ = ensure_2d(xi_eta)
        self._validate_ref_coords(xi_eta_)

        xi, eta = xi_eta_[:, 0], xi_eta_[:, 1]

        side0 = self.subdomain.curves[0]
        side1 = self.subdomain.curves[1]
        side2 = self.subdomain.curves[2]
        side3 = self.subdomain.curves[3]

        side0_vals = side0(xi)
        side1_vals = side1(eta)
        side2_vals = side2(-xi)
        side3_vals = side3(-eta)

        side0_tangent = side0.tangent(xi)
        side1_tangent = side1.tangent(eta)
        side2_tangent = -side2.tangent(-xi)
        side3_tangent = -side3.tangent(-eta)

        dxi_N = (
                0.5 * (1. - eta)[:, None] * side0_tangent
                + 0.5 * side1_vals
                + 0.5 * (1. + eta)[:, None] * side2_tangent
                - 0.5 * side3_vals
                + 0.25 * (1. - eta)[:, None] * side0.get_start()
                - 0.25 * (1. - eta)[:, None] * side1.get_start()
                - 0.25 * (1. + eta)[:, None] * side2.get_start()
                + 0.25 * (1. + eta)[:, None] * side3.get_start()
        )

        deta_N = (
                -0.5 * side0_vals
                + 0.5 * (1. + xi)[:, None] * side1_tangent
                + 0.5 * side2_vals
                + 0.5 * (1. - xi)[:, None] * side3_tangent
                + 0.25 * (1. - xi)[:, None] * side0.get_start()
                + 0.25 * (1. + xi)[:, None] * side1.get_start()
                - 0.25 * (1. + xi)[:, None] * side2.get_start()
                - 0.25 * (1. - xi)[:, None] * side3.get_start()
        )

        jacobian = np.stack([dxi_N[:, 0], deta_N[:, 0], dxi_N[:, 1], deta_N[:, 1]], axis=-1)
        jacobian = jacobian.reshape(-1, 2, 2)

        return jacobian

    def ref2element_jacobian(self, index: Tuple[int, int], xi_eta: np.ndarray) -> np.ndarray:
        """
        Compute the Jacobian of the reference to element map.

        Args:
            index (Tuple[int, int]): The 2D index of the element.
            xi_eta (np.ndarray): The reference coordinates in the quadrilateral.
                Of shape (num_points, 2) or (2,).

        Returns:
            np.ndarray: The Jacobian of the reference to element map.
                Of shape (num_points, 2, 2).
        """

        xi_eta_ = ensure_2d(xi_eta)
        self._validate_idxs(*index)
        self._validate_ref_coords(xi_eta_)

        index_x, index_y = index
        xi, eta = xi_eta_[:, 0], xi_eta_[:, 1]
        offset_xi = -1. + index_x * self._grid_step1d
        offset_eta = -1. + index_y * self._grid_step1d
        xi_rescaled = offset_xi + 0.5 * (1. + xi) * self._grid_step1d
        eta_rescaled = offset_eta + 0.5 * (1. + eta) * self._grid_step1d

        jacobian_domain = self.ref2domain_jacobian(np.column_stack((xi_rescaled, eta_rescaled)))
        scaling = 0.5 * self._grid_step1d
        jacobian_rescaled = jacobian_domain*scaling

        return jacobian_rescaled

    def get_jacobian_tensor_networks(self) -> np.ndarray:
        """
        Compute the tensor network approximating the Jacobian evaluated on all elements.
        The tensor index corresponds to the element index.
        This is done for each Jacobian component and each quadrature point within the element.
        The output is thus a total of 4*(num_quadrature_points_per_element) tensor networks.

        Returns:
            np.ndarray: 3D array of tensor networks for the Jacobian components.
               Indexing: [quadrature_point_index, component_index_i, component_index_j].
        """

        quadrature_points, _ = self.quadrature_rule.get_points_weights()

        jacobian_tensor_networks = np.ndarray((len(quadrature_points), 2, 2), dtype=object)
        for q, quad_point in enumerate(quadrature_points):
            for i in range(2):
                for j in range(2):
                    eval_point = np.array([quad_point])
                    cross_func = lambda idx: self._cross_func(idx, eval_point)[:, i, j]

                    # out of bounds intentional
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", UserWarning)
                        jac_ij = self._tca_strategy(cross_func)

                    jacobian_tensor_networks[q, i, j] = jac_ij
        return jacobian_tensor_networks

    def get_jacobian_det_tensor_networks(self) -> np.ndarray:
        """
        Compute the tensor network approximating
        the Jacobian determinants evaluated on all elements.

        Returns:
            np.ndarray: 1D array of tensor trains for the Jacobian determinants.
                Indexing: [quadrature_point_index].
        """
        quadrature_points, _ = self.quadrature_rule.get_points_weights()
        det_tensor_networks = np.ndarray((len(quadrature_points)), dtype=object)
        for q, quad_point in enumerate(quadrature_points):
            eval_point = np.array([quad_point])
            cross_func = lambda idx: np.linalg.det(self._cross_func(idx, eval_point))

            # out of bounds intentional
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                jac_det = self._tca_strategy(cross_func)

            det_tensor_networks[q] = jac_det

        return det_tensor_networks

    def get_jacobian_invdet_tensor_networks(self) -> np.ndarray:
        """
        Compute the tensor network approximating
        the inverse of Jacobian determinants evaluated on all elements.

        Returns:
            np.ndarray: 1D array of tensor trains for the inverse Jacobian determinants.
                Indexing: [quadrature_point_index].
        """
        quadrature_points, _ = self.quadrature_rule.get_points_weights()

        invdet_tensor_networks = []
        for quad_point in quadrature_points:
            eval_point = np.array([quad_point])
            cross_func = lambda idx: 1. / np.linalg.det(self._cross_func(idx, eval_point))

            # out of bounds intentional
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                jac_invdet = self.__tca_default(cross_func) # linear interpolation never works here

            invdet_tensor_networks.append(jac_invdet)

        return invdet_tensor_networks

    def get_jacobian_tensors(self) -> np.ndarray:
        """
        Compute the Jacobians evaluated on all elements and all quadrature points.

        Returns:
            np.ndarray: Jacobians.
                Of shape (num_elements_x+1, num_elements_y+1, num_quadrature_points, 2, 2).

        Note:
            This method is not efficient for large meshes.
            Intended only for small meshes for testing.
        """

        quadrature_points, _ = self.quadrature_rule.get_points_weights()
        num_elements_x = self.num_elements1d
        num_elements_y = self.num_elements1d
        num_quadrature_points = quadrature_points.shape[0]
        jacobians = np.empty((num_elements_x + 1, num_elements_y + 1, num_quadrature_points, 2, 2))

        for index_x in range(num_elements_x + 1):
            for index_y in range(num_elements_y + 1):
                index = (index_x, index_y)

                # out of bounds intentional
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    jacobian = self.ref2element_jacobian(index, quadrature_points)

                jacobians[index_x, index_y] = jacobian

        return jacobians

    def get_jacobian_dets(self) -> np.ndarray:
        """
        Compute the determinants of the Jacobians evaluated on
        all elements and all quadrature points.

        Returns:
            np.ndarray: Jacobian determinants.
                Of shape (num_elements_x+1, num_elements_y+1, num_quadrature_points).
        """

        jacobians = self.get_jacobian_tensors()
        jacobian_dets = np.linalg.det(jacobians)

        return jacobian_dets

    def get_jacobian_invdets(self) -> np.ndarray:
        """
        Compute the inverse of the determinants of the Jacobians
        evaluated on all elements and all quadrature points.

        Returns:
            np.ndarray: Inverse Jacobian determinants.
                Of shape (num_elements_x+1, num_elements_y+1, num_quadrature_points).
        """

        jacobian_dets = self.get_jacobian_dets()
        jacobian_invdets = 1. / jacobian_dets

        return jacobian_invdets

    def __tca_default(self, oracle: Callable[[np.ndarray], np.ndarray]) -> TensorTrain:
        """
        Perform tensor train cross approximation for a given oracle function.

        Args:
            oracle (Callable[[np.ndarray], np.ndarray]): The oracle function to approximate.

        Returns:
            TensorTrain: The tensor train cross approximation of the oracle.
        """

        kwargs = self.tt_cross_config.to_dict()
        num_indices = kwargs.pop("num_anova_init")
        order = kwargs.pop("anova_order")
        tensor_shape = [4]*self.mesh_size_exponent
        train_indices = gen_teneva_indices(num_indices, tensor_shape)

        tt_init = anova_init_tensor_train(oracle, train_indices, order)
        tt_cross_cores = tensor_train_cross_approximation(oracle, tt_init, **kwargs)

        return TensorTrain([torch.tensor(core) for core in tt_cross_cores])

    def _cross_func(self,
                    qindex: np.ndarray,
                    xi_eta: np.ndarray) -> np.ndarray:
        """
        Compute the Jacobian for a given element index given in binary
        and reference coordinates.

        Args:
            qindex (np.ndarray): The quaternary index of the element.
                Of shape (num_indices, index_length) or (index_length,).
            xi_eta (np.ndarray): The reference coordinates in the quadrilateral.
                Of shape (1, 2) or (2,).

        Returns:
            np.ndarray: The Jacobian. Of shape (num_indices, 2, 2).

        Raises:
            ValueError: If the index map is not defined, if the reference coordinates are invalid
                or if the number of evaluation points is more than 1.
        """

        if self.index_map is None:
            raise ValueError("Index map is not defined.")

        xi_eta_ = ensure_2d(xi_eta)
        qindex_ = ensure_2d(qindex)
        self._validate_ref_coords(xi_eta_)

        if xi_eta_.shape[0] > 1:
            raise ValueError("Only one evaluation point is supported for TCA.")

        jacobians = []
        for idx in range(qindex_.shape[0]):
            single_bindex = np.array(qindex_[idx, :])
            index = self.index_map(single_bindex)
            jacobian = self.ref2element_jacobian(index, xi_eta_)[0]
            jacobians.append(jacobian)

        jac = np.stack(jacobians)
        return jac

    def plot_element(self, index: Tuple[int, int], num_points: int = 100) -> None:
        """
        Plot the 2D points generated by the ref2element_map for a given index.

        Args:
            index (Tuple[int, int]): The 2D index of the element.
            num_points (int): The resolution of the grid for evaluation.
        """

        xi = np.linspace(-1, 1, num_points)
        eta = np.linspace(-1, 1, num_points)
        XI, ETA = np.meshgrid(xi, eta)

        # Flatten the grid for batch evaluation
        ref_coords = np.vstack([XI.ravel(), ETA.ravel()]).T

        # Evaluate the transformation map
        transformed_points = self.ref2element_map(index, ref_coords)

        # Extract X and Y coordinates
        X = transformed_points[:, 0]
        Y = transformed_points[:, 1]

        # Plot the 2D scatter plot of the mapped points
        plt.scatter(X, Y, c='red', marker='o', alpha=0.6)

        plt.title(f"2D Mapped Points for Index {index}")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.show()

    def plot(self, num_points: int = 100) -> None:
        """
        Plot the boundaries of all elements in the mesh with interpolated curves.

        Args:
            num_points (int): Number of points to sample along each curve.

        Warning:
            This method is not efficient for large meshes,
            call it with a small number of points for visualization purposes only.
        """
        xi_eta_edges = [
            np.column_stack((np.linspace(-1, 1, num_points), -1 * np.ones(num_points))),
            np.column_stack((np.ones(num_points), np.linspace(-1, 1, num_points))),
            np.column_stack((np.linspace(1, -1, num_points), np.ones(num_points))),
            np.column_stack((-1 * np.ones(num_points), np.linspace(1, -1, num_points)))
        ]

        for index_x in range(self.num_elements1d):
            for index_y in range(self.num_elements1d):
                for edge in xi_eta_edges:
                    physical_edge = np.array(
                        [self.ref2element_map((index_x, index_y), xi_eta[np.newaxis, :])[0]
                         for xi_eta in edge])
                    plt.plot(physical_edge[:, 0], physical_edge[:, 1], 'b-',
                             label="Element Boundary" if (index_x, index_y) == (0, 0) else "")

        plt.axis("equal")
        plt.title("Mesh Plot")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()

    def __repr__(self):
        return (f"SubdomainMesh2D(subdomain={self.subdomain},"
                f" quadrature_rule={self.quadrature_rule},"
                f" mesh_size_exponent={self.mesh_size_exponent},"
                f" num_points={self.num_points},"
                f" num_elements={self.num_elements})")

    def _validate_idxs(self, index_x: int, index_y: int):
        """
        Validate the element indices.

        Args:
            index_x (int): The x-index of the element.
            index_y (int): The y-index of the element.

        Raises:
            ValueError: If the indices are out of bounds.

        Note:
            The indices are assumed to be zero-based.
            Indices are valid if 0 <= index_x, index_y <= num_elements1d.
            We include the upper bound to allow for the last "padded" element
            such that the total number of elements is a power of 2: (num_elements1d)**2.
        """
        if index_x < 0 or index_x >= self.num_elements1d:
            warnings.warn(f"Index x={index_x} is out of bounds [0, {self.num_elements1d})."
                          f" The last element is a padded element.")
        if index_y < 0 or index_y >= self.num_elements1d:
            warnings.warn(f"Index y={index_y} is out of bounds [0, {self.num_elements1d})."
                          f" The last element is a padded element.")

    def _validate_ref_coords(self, xi_eta: np.ndarray, tol: float = 1e-6):
        """
        Validate the reference coordinates.

        Args:
            xi_eta (np.ndarray): The reference coordinates in the quadrilateral.
                Of shape (num_points, 2).
            tol (float): The tolerance for the range check.

        Raises:
            ValueError: If the reference coordinates are out of bounds.
        """
        if not xi_eta.shape[1] == 2:
            raise ValueError("Reference coordinates must have shape (num_points, 2).")

        if not np.all(-1 - tol <= xi_eta) or not np.all(xi_eta <= 1 + tol):
            warnings.warn(f"Reference coordinates are not in the range [-1, 1]"
                          f" within tolerance {tol}."
                          " This behavior may be intentional when using tensorized Jacobians.")


class QuadMesh(SubdomainMesh2D):
    """
    Mesh for a quadrilateral subdomain.
    The Jacobians of quadrilateral meshes depend linearly on the element index.
    Hence, instead of using the tensor train cross approximation for a generic subdomain,
    we can represent the Jacobians for all elements and all quadrature points exactly with a
    low-rank Tensor Train.

    Args:
        quad (Quad): The quadrilateral subdomain to mesh.
        quadrature_rule (QuadratureRule): The quadrature rule to use.
        mesh_size_exponent (int): The exponent of the discretization size.
            The discretization size is 2**(mesh_size_exponent) per dimension.
        tt_cross_config (TTCrossConfig, optional): Configuration for
            the tensor train cross approximation. Defaults to None (all default).
    """

    def __init__(self,
                 quad: Quad,
                 quadrature_rule: QuadratureRule,
                 mesh_size_exponent: int,
                 tt_cross_config: Optional[TTCrossConfig] = None):
        super().__init__(quad, quadrature_rule, mesh_size_exponent, tt_cross_config)
        self._tca_strategy = self.__linear_interpolation

    def __linear_interpolation(self,
        oracle: Callable[[np.ndarray], np.ndarray]) -> TensorTrain:
        """
        Perform linear interpolation for a given oracle function.

        Args:
            oracle (Callable[[np.ndarray], np.ndarray]): The oracle function to approximate.

        Returns:
            List[np.ndarray]: The tensor train cores of the linear interpolation.
        """
        func = lambda idx: oracle(idx)[0]
        tt_interpolant = interpolate_linear2d(func, self.mesh_size_exponent)

        return tt_interpolant

    def __repr__(self):
        return (f"QuadMesh(subdomain={self.subdomain},"
                f" quadrature_rule={self.quadrature_rule},"
                f" mesh_size_exponent={self.mesh_size_exponent},"
                f" num_points={self.num_points},"
                f" num_elements={self.num_elements})")
