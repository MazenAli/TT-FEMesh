from itertools import product

import numpy as np
import numpy.polynomial.legendre as leg
import pytest

from ttfemesh.quadrature.quadrature import (
    GaussLegendre,
    GaussLegendre2D,
    QuadratureRule,
    QuadratureRule2D,
)


class TestQuadratureRule:
    def test_abstract_base_class_cannot_be_instantiated(self):
        with pytest.raises(TypeError):
            QuadratureRule()

    def test_abstract_methods_must_be_implemented(self):
        class IncompleteQuadratureRule(QuadratureRule):
            pass

        with pytest.raises(TypeError):
            IncompleteQuadratureRule()


class TestQuadratureRule2D:
    def test_abstract_base_class_cannot_be_instantiated(self):
        with pytest.raises(TypeError):
            QuadratureRule2D()

    def test_dimension_property(self):
        class TestQuadratureRule2D(QuadratureRule2D):
            def get_points_weights(self):
                return np.array([]), np.array([])

            @staticmethod
            def compute_points_weights():
                return np.array([]), np.array([])

        quadrature_rule = TestQuadratureRule2D()
        assert quadrature_rule.dimension == 2


class TestGaussLegendre:
    def test_initialization(self):
        quadrature = GaussLegendre(order=2)
        assert quadrature.order == 2
        assert quadrature.dimension == 1
        assert quadrature.points is None
        assert quadrature.weights is None

    def test_initialization_with_dimension(self):
        quadrature = GaussLegendre(order=2, dimension=3)
        assert quadrature.order == 2
        assert quadrature.dimension == 3
        assert quadrature.points is None
        assert quadrature.weights is None

    def test_get_points_weights_1d(self):
        quadrature = GaussLegendre(order=2)
        points, weights = quadrature.get_points_weights()

        expected_points, expected_weights = leg.leggauss(2)
        assert np.allclose(points.flatten(), expected_points)
        assert np.allclose(weights, expected_weights)

    def test_get_points_weights_2d(self):
        quadrature = GaussLegendre(order=2, dimension=2)
        points, weights = quadrature.get_points_weights()

        points_1d, weights_1d = leg.leggauss(2)
        expected_points = np.array(list(product(points_1d, points_1d)))
        expected_weights = np.array([w1 * w2 for w1 in weights_1d for w2 in weights_1d])

        assert points.shape == (4, 2)
        assert weights.shape == (4,)
        assert np.allclose(points, expected_points)
        assert np.allclose(weights, expected_weights)

    def test_compute_points_weights_1d(self):
        points, weights = GaussLegendre.compute_points_weights(order=2)
        expected_points, expected_weights = leg.leggauss(2)
        assert np.allclose(points.flatten(), expected_points)
        assert np.allclose(weights, expected_weights)

    def test_compute_points_weights_2d(self):
        points, weights = GaussLegendre.compute_points_weights(order=2, dimension=2)
        points_1d, weights_1d = leg.leggauss(2)
        expected_points = np.array(list(product(points_1d, points_1d)))
        expected_weights = np.array([w1 * w2 for w1 in weights_1d for w2 in weights_1d])

        assert points.shape == (4, 2)
        assert weights.shape == (4,)
        assert np.allclose(points, expected_points)
        assert np.allclose(weights, expected_weights)

    def test_repr(self):
        quadrature = GaussLegendre(order=2, dimension=3)
        repr_str = repr(quadrature)
        assert "Gauss-Legendre Quadrature Rule" in repr_str
        assert "order=2" in repr_str
        assert "dimension=3" in repr_str


class TestGaussLegendre2D:
    def test_initialization(self):
        quadrature = GaussLegendre2D(order=2)
        assert quadrature.order == 2
        assert quadrature.dimension == 2
        assert quadrature.points is None
        assert quadrature.weights is None

    def test_get_points_weights(self):
        quadrature = GaussLegendre2D(order=2)
        points, weights = quadrature.get_points_weights()

        points_1d, weights_1d = leg.leggauss(2)
        expected_points = np.array(list(product(points_1d, points_1d)))
        expected_weights = np.array([w1 * w2 for w1 in weights_1d for w2 in weights_1d])

        assert points.shape == (4, 2)
        assert weights.shape == (4,)
        assert np.allclose(points, expected_points)
        assert np.allclose(weights, expected_weights)

    def test_repr(self):
        quadrature = GaussLegendre2D(order=2)
        repr_str = repr(quadrature)
        assert "2D Gauss-Legendre Quadrature Rule" in repr_str
        assert "order=2" in repr_str
