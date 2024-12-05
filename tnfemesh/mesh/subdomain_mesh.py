from abc import ABC, abstractmethod
from typing import Any, Tuple
import matplotlib.pyplot as plt
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
    def ref2element_map(self):
        """Return the reference to element map."""
        pass

    @abstractmethod
    def ref2element_jacobian(self) -> Any:
        """Return the Jacobian function for the element transformation."""
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

    def __init__(self,
                 subdomain: Subdomain2D,
                 quadrature_rule: QuadratureRule,
                 mesh_size_exponent: int):
        super().__init__(subdomain, quadrature_rule, mesh_size_exponent)

        self._num_points1d = 2**mesh_size_exponent
        self._grid_step1d = 1.0 / (self._num_points1d - 1)

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

    def ref2element_map(self,
                        index: Tuple[int, int],
                        xi_eta: np.ndarray) -> np.ndarray:

        index_x, index_y = index
        self._validate_idxs(index_x, index_y)
        self._validate_ref_coords(xi_eta)
        xi, eta = xi_eta[:, 0], xi_eta[:, 1]

        side0 = self.subdomain.curves[0]
        side1 = self.subdomain.curves[1]
        side2 = self.subdomain.curves[2]
        side3 = self.subdomain.curves[3]

        offset_xi = -1. + index_x * self._grid_step1d
        offset_eta = -1. + index_y * self._grid_step1d
        next_xi = -1. + (index_x+1) * self._grid_step1d
        next_eta = -1. + (index_y+1) * self._grid_step1d

        xi_rescaled = offset_xi + 0.5*self._grid_step1d*(xi+1)
        eta_rescaled = offset_eta + 0.5*self._grid_step1d*(eta+1)

        side0_vals = side0(xi_rescaled)
        side1_vals = side1(eta_rescaled)
        side2_vals = side2(xi_rescaled)
        side3_vals = side3(eta_rescaled)

        side0_x, side0_y = side0_vals[:, 0], side0_vals[:, 1]
        side1_x, side1_y = side1_vals[:, 0], side1_vals[:, 1]
        side2_x, side2_y = side2_vals[:, 0], side2_vals[:, 1]
        side3_x, side3_y = side3_vals[:, 0], side3_vals[:, 1]

        side0_vals_next = side0(next_xi)
        side1_vals_next = side1(next_eta)
        side2_vals_next = side2(next_xi)
        side3_vals_next = side3(next_eta)

        side0_x_next, side0_y_next = side0_vals_next[:, 0], side0_vals_next[:, 1]
        side1_x_next, side1_y_next = side1_vals_next[:, 0], side1_vals_next[:, 1]
        side2_x_next, side2_y_next = side2_vals_next[:, 0], side2_vals_next[:, 1]
        side3_x_next, side3_y_next = side3_vals_next[:, 0], side3_vals_next[:, 1]

        N_xi_eta_x = (0.5 * (1. - eta) * side0_x +
                      0.5 * (1. + xi) * side1_x +
                      0.5 * (1. + eta) * side2_x +
                      0.5 * (1. - xi) * side3_x -
                      0.25 * (1. - xi) * (1. - eta) * side0_x_next -
                      0.25 * (1. + xi) * (1. - eta) * side1_x_next -
                      0.25 * (1. + xi) * (1. + eta) * side2_x_next -
                      0.25 * (1. - xi) * (1. + eta) * side3_x_next)

        N_xi_eta_y = (0.5 * (1. - eta) * side0_y +
                      0.5 * (1. + xi) * side1_y +
                      0.5 * (1. + eta) * side2_y +
                      0.5 * (1. - xi) * side3_y -
                      0.25 * (1. - xi) * (1. - eta) * side0_y_next -
                      0.25 * (1. + xi) * (1. - eta) * side1_y_next -
                      0.25 * (1. + xi) * (1. + eta) * side2_y_next -
                      0.25 * (1. - xi) * (1. + eta) * side3_y_next)

        N_xi_eta = np.stack([N_xi_eta_x, N_xi_eta_y], axis=-1)

        return N_xi_eta

    def plot_element(self, index: Tuple[int, int], num_points: int = 100):
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

    def ref2element_jacobian(self,
                         index_x: int,
                         index_y: int,
                         xi: float,
                         eta: float) -> np.ndarray:

        side0 = self.subdomain.curves[0]
        side1 = self.subdomain.curves[1]
        side2 = self.subdomain.curves[2]
        side3 = self.subdomain.curves[3]

    def plot(self, num_points: int = 100):
        """
        Plot the boundaries of all elements in the mesh with interpolated curves.

        Args:
            num_points (int): Number of points to sample along each curve.
        """
        xi_eta_edges = [
            np.column_stack((np.linspace(-1, 1, num_points), -1 * np.ones(num_points))),
            # Bottom edge
            np.column_stack((np.ones(num_points), np.linspace(-1, 1, num_points))),  # Right edge
            np.column_stack((np.linspace(1, -1, num_points), np.ones(num_points))),  # Top edge
            np.column_stack((-1 * np.ones(num_points), np.linspace(1, -1, num_points)))  # Left edge
        ]

        for index_x in range(self.num_elements_x):
            for index_y in range(self.num_elements_y):
                for edge in xi_eta_edges:
                    # Map the reference edge to the element's physical space
                    physical_edge = np.array(
                        [self.ref2element_map((index_x, index_y), xi_eta[np.newaxis, :])[0]
                         for xi_eta in edge])
                    # Plot the edge
                    plt.plot(physical_edge[:, 0], physical_edge[:, 1], 'b-',
                             label="Element Boundary" if (index_x, index_y) == (0, 0) else "")

        plt.axis("equal")
        plt.title("Mesh Plot")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()

    def _validate_idxs(self, index_x: int, index_y: int):
        pass

    def _validate_ref_coords(self, xi_eta: np.ndarray):
        pass
