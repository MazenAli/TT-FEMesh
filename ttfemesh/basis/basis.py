from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torchtt as tntt

from ttfemesh.basis.basis_utils import left_corner2index_ttmap, right_corner2index_ttmap
from ttfemesh.tt_tools.numeric import unit_vector_binary_tt
from ttfemesh.tt_tools.operations import zorder_kron
from ttfemesh.types import BoundarySide2D, TensorTrain


class Basis(ABC):
    """Abstract base class for basis functions."""

    @property
    @abstractmethod
    def index_range(self):  # pragma: no cover
        """Range of valid indices for the basis functions."""
        pass

    @abstractmethod
    def evaluate(self, *args, **kwargs) -> Any:  # pragma: no cover
        """Evaluate the basis function indexed with idx at a given point."""
        pass

    @abstractmethod
    def derivative(self, *args, **kwargs) -> Any:  # pragma: no cover
        """Evaluate the derivative of the basis function indexed with idx at a given point."""
        pass

    @abstractmethod
    def _validate(self, *args, **kwargs):  # pragma: no cover
        """Validate the basis function index."""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:  # pragma: no cover
        """The number of dimensions of the basis functions."""
        pass


class Basis1D(Basis):
    """Abstract base class for 1D basis functions on the reference element [-1, 1]."""

    def plot(self, idx: int, num_points: int = 100):
        """
        Plot the basis function indexed with idx.

        Args:
            idx (int): Index of the basis function.
            num_points (int): Number of points to plot.

        Returns:
            matplotlib.figure.Figure: The figure object containing the plot.
        """
        self._validate(idx)

        fig = plt.figure()
        x_vals = np.linspace(-1, 1, num_points)
        y_vals = np.array([self.evaluate(idx, x) for x in x_vals])
        plt.plot(x_vals, y_vals, label=f"Basis Function {idx}")

        plt.title("1D Basis Function on [-1, 1]")
        plt.xlabel("x")
        plt.ylabel("Basis Function Value")
        return fig

    @property
    def dimension(self) -> int:
        return 1

    @abstractmethod
    def get_element2global_ttmap(self, *args, **kwargs) -> TensorTrain:  # pragma: no cover
        """
        Get the TT-representation of a corner element index to global basis index map.

        Returns:
            TensorTrain: TT-representation of the corner to global index map.
        """
        pass

    @abstractmethod
    def get_all_element2global_ttmaps(
        self, *args, **kwargs
    ) -> Tuple[TensorTrain, ...]:  # pragma: no cover
        """
        Get the TT-representation for all corner elements in `index_range`
        to global basis index maps.

        Returns:
            Tuple[TensorTrain, ...]:  TT-representations of all corner-to-global index maps.
                First element is the map for index 0, second element is the map for index 1.
        """
        pass

    @abstractmethod
    def get_dirichlet_mask_left(self, *args, **kwargs) -> TensorTrain:  # pragma: no cover
        """
        Get the mask for the left Dirichlet boundary condition.

        Returns:
            TensorTrain: TT-representation of the Dirichlet mask.
        """
        pass

    @abstractmethod
    def get_dirichlet_mask_right(self, *args, **kwargs) -> TensorTrain:  # pragma: no cover
        """
        Get the mask for the right Dirichlet boundary condition.

        Returns:
            TensorTrain: TT-representation of the Dirichlet mask.
        """
        pass

    @abstractmethod
    def get_dirichlet_mask_left_right(self, *args, **kwargs) -> TensorTrain:  # pragma: no cover
        """
        Get the mask for the left and right Dirichlet boundary conditions.

        Returns:
            TensorTrain: TT-representation of the Dirichlet mask.
        """
        pass


class LinearBasis(Basis1D):
    """
    Linear basis functions on the reference element [-1, 1].
    The basis functions are defined as:
    - 0.5 * (1 - x) for idx = 0 (left basis function)
    - 0.5 * (1 + x) for idx = 1 (right basis function)

    Example:
        >>> from ttfemesh.basis import LinearBasis
        >>> basis1d = LinearBasis()
        >>> fig = basis1d.plot(0)
        >>> fig.show()
        >>> fig = basis1d.plot(1)
        >>> fig.show()
    """

    def evaluate(self, idx: int, x: float) -> float:
        """
        Evaluate the basis function at a given point.

        Args:
            idx (int): Index of the basis function.
            x (float): Point in [-1, 1] to evaluate the basis function.

        Returns:
            float: Value of the basis function at x.
        """
        self._validate(idx)

        if idx == 0:
            return 0.5 * (1 - x)

        return 0.5 * (1 + x)

    def derivative(self, idx: int, _: Optional[float] = None) -> float:
        """
        Evaluate the derivative of the basis function at a given point.
        Derivative is constant, so the point x is not used.

        Args:
            idx (int): Index of the basis function.
            _ (Optional[float]): Ignored.

        Returns:
            float: Derivative of the basis function at x.
        """
        self._validate(idx)
        return -0.5 if idx == 0 else 0.5

    @property
    def index_range(self):
        """
        Range of valid indices for the basis functions.
        0 for the left basis function, 1 for the right basis function.
        """
        return range(2)

    def get_element2global_ttmap(self, index: int, mesh_size_exponent: int) -> TensorTrain:
        """
        Get the TT-representation of a corner element index
        to global basis index map.

        Args:
            index (int): Index of the corner element (0 for left, 1 for right).
            mesh_size_exponent (int): Exponent of the 1D mesh size (length of TT).

        Returns:
            TensorTrain: TT-representation of the corner to global index map.

        Raises:
            ValueError: If the index is invalid.
        """
        self._validate(index)

        if index == 0:
            return left_corner2index_ttmap(mesh_size_exponent)
        elif index == 1:
            return right_corner2index_ttmap(mesh_size_exponent)

    def get_all_element2global_ttmaps(self, mesh_size_exponent: int) -> Tuple[TensorTrain, ...]:
        """
        Get TT-representations for all indices in `index_range`.

        Args:
            mesh_size_exponent (int): Exponent of the 1D mesh size (length of TT).

        Returns:
            Tuple[TensorTrain, ...]: TT-representations of all corner-to-global index maps.
                First element is the map for index 0, second element is the map for index 1.
        """
        return tuple(
            self.get_element2global_ttmap(index, mesh_size_exponent) for index in self.index_range
        )

    def get_dirichlet_mask_left(self, mesh_size_exponent: int) -> TensorTrain:
        """
        Get the mask for the left Dirichlet boundary condition.
        The mask is 1 everywhere except at the left boundary.

        Args:
            mesh_size_exponent (int): Exponent of the 1D mesh size (length of TT).

        Returns:
            TensorTrain: TT-representation of the Dirichlet mask.
        """

        mask = tntt.ones([2] * mesh_size_exponent) - unit_vector_binary_tt(mesh_size_exponent, 0)
        return mask

    def get_dirichlet_mask_right(self, mesh_size_exponent: int) -> TensorTrain:
        """
        Get the mask for the right Dirichlet boundary condition.
        The mask is 1 everywhere except at the right boundary.

        Args:
            mesh_size_exponent (int): Exponent of the 1D mesh size (length of TT).

        Returns:
            TensorTrain: TT-representation of the Dirichlet mask.
        """

        mask = tntt.ones([2] * mesh_size_exponent) - unit_vector_binary_tt(
            mesh_size_exponent, 2**mesh_size_exponent - 1
        )
        return mask

    def get_dirichlet_mask_left_right(self, mesh_size_exponent: int) -> TensorTrain:
        """
        Get the mask for the left and right Dirichlet boundary conditions.
        The mask is 1 everywhere except at the left and right boundaries.

        Args:
            mesh_size_exponent (int): Exponent of the 1D mesh size (length of TT).

        Returns:
            TensorTrain: TT-representation of the Dirichlet mask.
        """

        mask = (
            tntt.ones([2] * mesh_size_exponent)
            - unit_vector_binary_tt(mesh_size_exponent, 0)
            - unit_vector_binary_tt(mesh_size_exponent, 2**mesh_size_exponent - 1)
        )

        return mask

    def _validate(self, idx: int):
        if idx not in self.index_range:
            raise ValueError(
                f"Invalid basis function index: {idx}."
                f" Expected one of {list(self.index_range)}."
            )

    def __repr__(self):
        return "LinearBasis"


class TensorProductBasis(Basis):
    """
    Abstract base class for tensor product basis functions for arbitrary dimensions.
    Combines 1D basis functions to define basis functions in higher dimensions.
    """

    def __init__(self, basis_functions: List[Basis1D]):
        """
        Initialize the tensor product basis function.

        Args:
            basis_functions (List[BasisFunction1D]): List of 1D basis functions for each dimension.
        """
        self.basis_functions = basis_functions
        self._dimension = len(basis_functions)

    @property
    def dimension(self) -> int:
        return self._dimension

    @abstractmethod
    def get_element2global_ttmap(self, *args, **kwargs) -> TensorTrain:  # pragma: no cover
        """
        Get the TT-representation of a corner element index to global basis index map.

        Returns:
            TensorTrain: TT-representation of the corner to global index map.
        """
        pass

    @abstractmethod
    def get_all_element2global_ttmaps(self, *args, **kwargs) -> np.ndarray:  # pragma: no cover
        """
        Get the TT-representation for all corner elements in `index_range`
        to global basis index maps.

        Returns:
            np.ndarray: A 2D matrix of TT-representations, indexed by (i, j)
                where i and j are the indices of the basis functions in each dimension.
        """
        pass

    @abstractmethod
    def get_dirichlet_mask(self, *args, **kwargs) -> TensorTrain:  # pragma: no cover
        """
        Get the mask for the Dirichlet boundary condition on the specified sides.

        Returns:
            TensorTrain: TT-representation of the Dirichlet mask.
        """
        pass

    @property
    def index_range(self):
        return [bf.index_range for bf in self.basis_functions]

    def evaluate(self, idx: List[int], x: List[float]) -> float:
        """
        Evaluate the tensor product basis function at a given point.

        Args:
            idx (List[int]): Indices of the basis functions in each dimension.
            x (List[float]): Coordinates in the reference element [-1, 1]^d.

        Returns:
            float: Value of the tensor product basis function at x.
        """
        self._validate(idx)
        return np.prod([bf.evaluate(i, xi) for bf, i, xi in zip(self.basis_functions, idx, x)])

    def derivative(self, idx: List[int], x: List[float], dim: int) -> float:
        """
        Evaluate the partial derivative with respect to a given dimension.

        Args:
            idx (List[int]): Indices of the basis functions in each dimension.
            x (List[float]): Coordinates in the reference element [-1, 1]^d.
            dim (int): Dimension index (0-based) to differentiate.

        Returns:
            float: Value of the derivative at x.
        """
        self._validate(idx)
        if dim < 0 or dim >= self.dimension:
            raise ValueError(
                f"Invalid dimension index: {dim}, expected 0 <= dim < {self.dimension}"
            )

        result = 1.0
        for i, (bf, idx_xi, xi) in enumerate(zip(self.basis_functions, idx, x)):
            if i == dim:
                result *= bf.derivative(idx_xi, xi)
            else:
                result *= bf.evaluate(idx_xi, xi)
        return result

    def _validate(self, idx: List[int]):
        """Validate the basis function indices."""
        if len(idx) != self.dimension:
            raise ValueError(
                f"Invalid number of indices: expected {self.dimension}, got {len(idx)}"
            )
        for i, idx_i in enumerate(idx):
            self.basis_functions[i]._validate(idx_i)

    def __repr__(self):
        return f"TensorProductBasis(dim={self.dimension})"

    def plot(self, idx: List[int], num_points: int = 100):
        """
        Plot the tensor product basis function indexed with idx.

        Args:
            idx (List[int]): Indices of the basis functions in each dimension.
            num_points (int): Number of points to plot.

        Raises:
            NotImplementedError: If the dimension is not 2.
        """
        self._validate(idx)

        if self.dimension == 2:
            return self._plot2d(idx, num_points)
        else:
            raise NotImplementedError(
                "Tensor product plotting" " is only supported for 2D basis functions."
            )

    def _plot2d(self, idx: List[int], num_points: int = 100):
        """
        Plot the tensor product basis function indexed with idx in 2D.

        Args:
            idx (List[int]): Indices of the basis functions in each dimension.
            num_points (int): Number of points to plot.

        Returns:
            matplotlib.figure.Figure: The figure object containing the plot.
        """
        fig = plt.figure()
        x_vals = np.linspace(-1, 1, num_points)
        y_vals = np.linspace(-1, 1, num_points)
        X, Y = np.meshgrid(x_vals, y_vals)
        Z = np.zeros_like(X)
        for i in range(num_points):
            for j in range(num_points):
                Z[i, j] = self.evaluate(idx, [X[i, j], Y[i, j]])
        plt.contourf(X, Y, Z, levels=20)
        plt.colorbar()
        plt.title("2D Tensor Product Basis Function")
        plt.xlabel("x")
        plt.ylabel("y")
        return fig


class BilinearBasis(TensorProductBasis):
    """
    Bilinear basis functions on the reference element [-1, 1]^2.
    The basis functions are defined as:
    - 0.25 * (1 - x) * (1 - y) for idx = (0, 0)
    - 0.25 * (1 + x) * (1 - y) for idx = (1, 0)
    - 0.25 * (1 - x) * (1 + y) for idx = (0, 1)
    - 0.25 * (1 + x) * (1 + y) for idx = (1, 1)

    Example:
        >>> from ttfemesh.basis import BilinearBasis
        >>> basis2d = BilinearBasis()
        >>> fig = basis2d.plot([0, 0])
        >>> fig.show()
        >>> fig = basis2d.plot([0, 1])
        >>> fig.show()
        >>> fig = basis2d.plot([1, 0])
        >>> fig.show()
        >>> fig = basis2d.plot([1, 1])
        >>> fig.show()
    """

    def __init__(self):
        super().__init__([LinearBasis(), LinearBasis()])

    def get_element2global_ttmap(self, index: List[int], mesh_size_exponent: int) -> TensorTrain:
        """
        Get the TT-representation of a corner element index to global basis index map.

        Args:
            index (List[int]): Indices of the corner element
                ((0, 0) for lower left,
                (1, 0) for lower right,
                (0, 1) for upper left,
                (1, 1) for upper right).
            mesh_size_exponent (int): Exponent of the 1D mesh size.

        Returns:
            TensorTrain: TT-representation of the corner to global index map.

        Raises:
            ValueError: If the index is invalid.
        """

        self._validate(index)
        return zorder_kron(
            self.basis_functions[0].get_element2global_ttmap(index[0], mesh_size_exponent),
            self.basis_functions[1].get_element2global_ttmap(index[1], mesh_size_exponent),
        )

    def get_all_element2global_ttmaps(self, mesh_size_exponent: int) -> np.ndarray:
        """
        Get the TT-representation for all corner elements in `index_range`
        to global basis index maps.

        Args:
            mesh_size_exponent (int): Exponent of the 1D mesh size.

        Returns:
            np.ndarray: A 2D matrix of TT-representations, indexed by (i, j)
                where i and j are the indices of the basis functions in each dimension.
        """
        return np.array(
            [
                [
                    zorder_kron(
                        self.basis_functions[0].get_element2global_ttmap(i, mesh_size_exponent),
                        self.basis_functions[1].get_element2global_ttmap(j, mesh_size_exponent),
                    )
                    for j in self.index_range[1]
                ]
                for i in self.index_range[0]
            ]
        )

    def get_dirichlet_mask(
        self, mesh_size_exponent: int, *sides: Union[BoundarySide2D, int]
    ) -> TensorTrain:
        """
        Get the mask for the Dirichlet 2D boundary condition on the specified sides.

        Note that the sides are considered to be ordered as follows:
        bottom (side 0), right (side 1), top (side 2), left (side 3).
        This is important for the boundary condition to work correctly.
        It may lead to confusion if, e.g., your side 0 is visually
        the right edge of the domain.

        Args:
            mesh_size_exponent (int): Exponent of the 1D mesh size.
            *sides (Union[BoundarySide2D, int]): Boundary sides to apply the Dirichlet condition.
                Specify either as integers (0 for bottom, 1 for right, 2 for top, 3 for left)
                or as BoundarySide2D enums.

        Returns:
            TensorTrain: TT-representation of the Dirichlet mask.

        Raises:
            ValueError: If no sides are specified or if an invalid side is given.
        """

        if not sides:
            raise ValueError("At least one boundary side must be specified.")

        sides_ = [BoundarySide2D(side) if isinstance(side, int) else side for side in sides]
        xmask = tntt.ones([2] * mesh_size_exponent)
        if BoundarySide2D.LEFT in sides_ and BoundarySide2D.RIGHT in sides_:
            xmask = self.basis_functions[0].get_dirichlet_mask_left_right(mesh_size_exponent)
        elif BoundarySide2D.LEFT in sides_:
            xmask = self.basis_functions[0].get_dirichlet_mask_left(mesh_size_exponent)
        elif BoundarySide2D.RIGHT in sides_:
            xmask = self.basis_functions[0].get_dirichlet_mask_right(mesh_size_exponent)

        ymask = tntt.ones([2] * mesh_size_exponent)
        if BoundarySide2D.BOTTOM in sides_ and BoundarySide2D.TOP in sides_:
            ymask = self.basis_functions[1].get_dirichlet_mask_left_right(mesh_size_exponent)
        elif BoundarySide2D.BOTTOM in sides_:
            ymask = self.basis_functions[1].get_dirichlet_mask_left(mesh_size_exponent)
        elif BoundarySide2D.TOP in sides_:
            ymask = self.basis_functions[1].get_dirichlet_mask_right(mesh_size_exponent)

        return zorder_kron(xmask, ymask)

    def __repr__(self):
        return super().__repr__() + "::" + "BilinearBasis"
