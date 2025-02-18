from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np
import numpy.polynomial.legendre as leg


class QuadratureRule(ABC):
    """Abstract base class for a quadrature rule."""

    @abstractmethod
    def get_points_weights(self) -> Tuple[np.ndarray, np.ndarray]:
        """Retrieve the quadrature points and weights."""
        pass

    @staticmethod
    @abstractmethod
    def compute_points_weights(self) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the quadrature points and weights."""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """The number of dimensions of the quadrature rule."""
        pass


class QuadratureRule2D(QuadratureRule):
    """Abstract base class for a 2D quadrature rule."""

    @property
    def dimension(self) -> int:
        """The number of dimensions of the quadrature rule."""
        return 2


class GaussLegendre(QuadratureRule):
    """Implements Gauss-Legendre quadrature on [-1, 1]^(dimension)."""

    def __init__(self, order: int, dimension: int = 1):
        """
        Initialize the Gauss-Legendre quadrature rule.

        Args:
            order (int): The order of the quadrature rule. Must be greater than 0.
            dimension (int): The number of dimensions (default is 1).
        """
        self.order = order
        self._dimension = dimension
        self.points: Optional[np.ndarray] = None
        self.weights: Optional[np.ndarray] = None

    @property
    def dimension(self) -> int:
        """The number of dimensions of the quadrature rule."""
        return self._dimension

    def get_points_weights(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the quadrature points and weights.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Quadrature points and weights.
        """
        if self.points is None or self.weights is None:
            self.points, self.weights = self.compute_points_weights(self.order, self.dimension)
        return self.points, self.weights

    @staticmethod
    def compute_points_weights(order: int, dimension: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the Gauss-Legendre quadrature points and weights.

        Args:
            order (int): The order of the quadrature rule. Must be greater than 0.
            dimension (int): The number of dimensions (default is 1).

        Returns:
            Tuple[np.ndarray, np.ndarray]: Quadrature points and weights.
        """
        points_1d, weights_1d = leg.leggauss(order)

        if dimension == 1:
            return points_1d[:, None], weights_1d

        # Tensor product for multi-dimensional quadrature
        grid = np.meshgrid(*[points_1d] * dimension, indexing="ij")
        points = np.stack([g.flatten() for g in grid], axis=-1)

        weights = np.prod(np.meshgrid(*[weights_1d] * dimension, indexing="ij"), axis=0)
        return points, weights.flatten()

    def __repr__(self) -> str:
        """String representation of the Gauss-Legendre quadrature rule."""
        return f"Gauss-Legendre Quadrature Rule (order={self.order}, dimension={self.dimension})"


class GaussLegendre2D(GaussLegendre):
    """Implements Gauss-Legendre quadrature on [-1, 1]^2."""

    def __init__(self, order: int):
        """
        Initialize the 2D Gauss-Legendre quadrature rule.

        Args:
            order (int): The order of the quadrature rule. Must be greater than 0.
        """
        super().__init__(order, dimension=2)

    def __repr__(self) -> str:
        """String representation of the 2D Gauss-Legendre quadrature rule."""
        return f"2D Gauss-Legendre Quadrature Rule (order={self.order})"
