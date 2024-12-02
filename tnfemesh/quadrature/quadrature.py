from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np
import numpy.polynomial.legendre as leg


class QuadratureRule(ABC):
    """Abstract base class for a quadrature rule."""

    @abstractmethod
    def compute(self) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the quadrature points and weights."""
        pass


class GaussLegendre(QuadratureRule):
    """Implements Gauss-Legendre quadrature on [-1, 1]^(dimension)."""

    @staticmethod
    def compute(order: int, dimension: int = 1) -> Tuple[np.ndarray, np.ndarray]:
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
