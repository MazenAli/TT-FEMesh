from abc import ABC, abstractmethod
from typing import List, Iterable, Optional, Tuple, Any
import matplotlib.pyplot as plt
import numpy as np
from tnfemesh.types import TensorTrain
from tnfemesh.basis.basis_utils import left_corner2index_ttmap, right_corner2index_ttmap
from tnfemesh.tn_tools.operations import zorder_kron


class Basis(ABC):
    """Abstract base class for basis functions."""
    @property
    @abstractmethod
    def index_range(self):
        """Range of valid indices for the basis functions."""
        pass

    @abstractmethod
    def evaluate(self, idx: Any, x: Any) -> Any:
        """Evaluate the basis function indexed with idx at a given point."""
        pass

    @abstractmethod
    def _validate(self, idx: Any):
        """Validate the basis function index."""
        pass


class Basis1D(Basis):
    """Abstract base class for 1D basis functions on the reference element [-1, 1]."""

    def plot(self, idx: int, num_points: int = 100):
        """
        Plot the basis function indexed with idx.

        Args:
            idx (int): Index of the basis function.
            num_points (int): Number of points to plot.
        """
        self._validate(idx)

        x_vals = np.linspace(-1, 1, num_points)
        y_vals = np.array([self.evaluate(idx, x) for x in x_vals])
        plt.plot(x_vals, y_vals, label=f"Basis Function {idx}")

        plt.title("1D Basis Function on [-1, 1]")
        plt.xlabel("x")
        plt.ylabel("Basis Function Value")
        plt.show()


class LinearBasis1D(Basis1D):
    """Linear basis functions on the reference element [-1, 1]."""

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
        elif idx == 1:
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
        return -0.5 if self.idx == 0 else 0.5


    @property
    def index_range(self):
        """
        Range of valid indices for the basis functions.
        0 for the left basis function, 1 for the right basis function.
        """
        return range(2)

    def get_index_ttmap(self, index: int, d: int) -> TensorTrain:
        """
        Get the TT-representation of a corner element index
        to global basis index map.

        Args:
            index (int): Index of the corner element (e.g., 0 for left, 1 for right).
            d (int): Exponent of the 1D mesh size (length of TT).

        Returns:
            TensorTrain: TT-representation of the corner to global index map.

        Raises:
            ValueError: If the index is invalid.
        """
        self._validate(index)

        if index == 0:
            return left_corner2index_ttmap(d)
        elif index == 1:
            return right_corner2index_ttmap(d)

    def get_all_index_ttmaps(self, d: int) -> Tuple[TensorTrain, ...]:
        """
        Get TT-representations for all indices in `index_range`.

        Args:
            d (int): Exponent of the 1D mesh size (length of TT).

        Returns:
            Tuple[TensorTrain, ...]: TT-representations of all corner-to-global index maps.
        """
        return tuple(self.get_index_ttmap(index, d) for index in self.index_range)

    def _validate(self, idx: int):
        if idx not in self.index_range:
            raise ValueError(f"Invalid basis function index: {idx}."
                             f" Expected one of {list(self.index_range)}.")

    def __repr__(self):
        return "LinearBasis1D"


class TensorProductBasis(Basis):
    """
    Tensor product basis functions for arbitrary dimensions.
    Combines 1D basis functions to define basis functions in higher dimensions.
    """

    def __init__(self, basis_functions: List[Basis1D]):
        """
        Initialize the tensor product basis function.

        Args:
            basis_functions (List[BasisFunction1D]): List of 1D basis functions for each dimension.
        """
        self.basis_functions = basis_functions
        self.dim = len(basis_functions)

    @property
    def index_range(self):
        return [bf.index_range for bf in self.basis_functions]

    def evaluate(self, idx: Iterable[int], x: Iterable[float]) -> float:
        """
        Evaluate the tensor product basis function at a given point.

        Args:
            idx (Iterable[int]): Indices of the basis functions in each dimension.
            x (Iterable[float]): Coordinates in the reference element [-1, 1]^d.

        Returns:
            float: Value of the tensor product basis function at x.
        """
        self._validate(idx)
        return np.prod([bf.evaluate(i, xi) for bf, i, xi in zip(self.basis_functions, idx, x)])

    def derivative(self, idx: Iterable[int], x: Iterable[float], dim: int) -> float:
        """
        Evaluate the partial derivative with respect to a given dimension.

        Args:
            idx (Iterable[int]): Indices of the basis functions in each dimension.
            x (Iterable[float]): Coordinates in the reference element [-1, 1]^d.
            dim (int): Dimension index (0-based) to differentiate.

        Returns:
            float: Value of the derivative at x.
        """
        self._validate(idx)
        if dim < 0 or dim >= self.dim:
            raise ValueError(f"Invalid dimension index: {dim}, expected 0 <= dim < {self.dim}")

        result = 1.0
        for i, (bf, xi) in enumerate(zip(self.basis_functions, x)):
            if i == dim:
                result *= bf.derivative(xi)
            else:
                result *= bf.evaluate(xi)
        return result

    def _validate(self, idx: Iterable[int]):
        """Validate the basis function indices."""
        if len(idx) != self.dim:
            raise ValueError(f"Invalid number of indices: expected {self.dim}, got {len(idx)}")
        for i, idx_i in enumerate(idx):
            self.basis_functions[i]._validate(idx_i)

    def __repr__(self):
        return f"TensorProductBasis(dim={self.dim})"

    def plot(self, idx: Iterable[int], num_points: int = 100):
        """
        Plot the tensor product basis function indexed with idx.

        Args:
            idx (Iterable[int]): Indices of the basis functions in each dimension.
            num_points (int): Number of points to plot.
        """
        self._validate(idx)

        if self.dim == 2:
            self._plot2d(idx, num_points)
        elif self.dim == 3:
            self._plot3d(idx, num_points)
        else:
            raise NotImplementedError("Plotting is only supported for 2D and 3D basis functions.")

    def _plot2d(self, idx: Iterable[int], num_points: int = 100):
        """
        Plot the tensor product basis function indexed with idx in 2D.

        Args:
            idx (Iterable[int]): Indices of the basis functions in each dimension.
            num_points (int): Number of points to plot.
        """
        x_vals = np.linspace(-1, 1, num_points)
        y_vals = np.linspace(-1, 1, num_points)
        X, Y = np.meshgrid(x_vals, y_vals)
        Z = np.zeros_like(X)
        for i in range(num_points):
            for j in range(num_points):
                Z[i, j] = self.evaluate(idx, (X[i, j], Y[i, j]))
        plt.contourf(X, Y, Z, levels=20)
        plt.colorbar()
        plt.title("2D Tensor Product Basis Function")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()

    def _plot3d(self, idx: Iterable[int], num_points: int = 30):
        """
        Plot the tensor product basis function as a 3D heatmap.

        Args:
            idx (Iterable[int]): Indices of the basis functions in each dimension.
            num_points (int): Number of points per dimension for the plot.
        """

        x = np.linspace(-1, 1, num_points)
        y = np.linspace(-1, 1, num_points)
        z = np.linspace(-1, 1, num_points)
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

        points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
        values = np.array([self.evaluate(idx, point) for point in points])

        values_normalized = (values - values.min()) / (values.max() - values.min())

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        scatter = ax.scatter(
            X.ravel(), Y.ravel(), Z.ravel(),
            c=values_normalized, cmap='viridis', s=5, alpha=0.8
        )

        cbar = fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=10)
        cbar.set_label('Basis function value')

        # Set plot labels
        plt.title("3D Tensor Product Basis Function")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

        plt.show()


class LinearBasis2D(TensorProductBasis):
    """Linear basis functions on the reference element [-1, 1]^2."""

    def __init__(self):
        super().__init__([LinearBasis1D(), LinearBasis1D()])

    def get_index_ttmap(self, index: Tuple[int, int], d: int) -> TensorTrain:
        """
        Get the TT-representation of a corner element index to global basis index map.

        Args:
            index (Tuple[int, int]): Indices of the corner element
                (e.g., (0, 0) for lower left, (1, 1) for upper right).
            d (int): Exponent of the 1D mesh size (length of TT).

        Returns:
            TensorTrain: TT-representation of the corner to global index map.

        Raises:
            ValueError: If the index is invalid.
        """

        self._validate(index)
        return zorder_kron(
            self.basis_functions[0].get_index_ttmap(index[0], d),
            self.basis_functions[1].get_index_ttmap(index[1], d)
        )

    def get_all_index_ttmaps(self, d: int) -> np.ndarray:
        """
        Get TT-representations for all indices in `index_range`.

        Args:
            d (int): Exponent of the 1D mesh size (length of TT).

        Returns:
            np.ndarray: A 2D matrix of TT-representations, indexed by (i, j)
                where i and j are the indices of the basis functions in each dimension.
        """
        return np.array([
            [
                zorder_kron(
                    self.basis_functions[0].get_index_ttmap(i, d),
                    self.basis_functions[1].get_index_ttmap(j, d)
                )
                for j in self.index_range[1]
            ]
            for i in self.index_range[0]
        ])

    def __repr__(self):
        return "LinearBasis2D"
