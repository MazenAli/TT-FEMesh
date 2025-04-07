import pytest
import numpy as np
from ttfemesh.domain.subdomain_factory import SubdomainFactory, RectangleFactory, QuadFactory
from ttfemesh.domain.subdomain import Quad
from ttfemesh.domain.curve import Line2D


class TestSubdomainFactory:
    def test_abstract_base_class_cannot_be_instantiated(self):
        with pytest.raises(TypeError):
            SubdomainFactory()

    def test_abstract_methods_must_be_implemented(self):
        class IncompleteFactory(SubdomainFactory):
            pass

        with pytest.raises(TypeError):
            IncompleteFactory()


class TestRectangleFactory:
    def test_create_with_valid_points(self):
        bottom_left = (0.0, 0.0)
        top_right = (1.0, 1.0)
        quad = RectangleFactory.create(bottom_left, top_right)

        assert isinstance(quad, Quad)
        assert len(quad.curves) == 4

        curves = quad.curves
        assert isinstance(curves[0], Line2D)
        assert isinstance(curves[1], Line2D)
        assert isinstance(curves[2], Line2D)
        assert isinstance(curves[3], Line2D)

        assert np.allclose(curves[0].start, np.array([0.0, 0.0]))
        assert np.allclose(curves[0].end, np.array([1.0, 0.0]))

        assert np.allclose(curves[1].start, np.array([1.0, 0.0]))
        assert np.allclose(curves[1].end, np.array([1.0, 1.0]))

        assert np.allclose(curves[2].start, np.array([1.0, 1.0]))
        assert np.allclose(curves[2].end, np.array([0.0, 1.0]))

        assert np.allclose(curves[3].start, np.array([0.0, 1.0]))
        assert np.allclose(curves[3].end, np.array([0.0, 0.0]))

    def test_create_with_negative_coordinates(self):
        bottom_left = (-1.0, -1.0)
        top_right = (1.0, 1.0)
        quad = RectangleFactory.create(bottom_left, top_right)

        assert isinstance(quad, Quad)
        assert len(quad.curves) == 4

        curves = quad.curves
        assert np.allclose(curves[0].start, np.array([-1.0, -1.0]))
        assert np.allclose(curves[0].end, np.array([1.0, -1.0]))
        assert np.allclose(curves[1].start, np.array([1.0, -1.0]))
        assert np.allclose(curves[1].end, np.array([1.0, 1.0]))
        assert np.allclose(curves[2].start, np.array([1.0, 1.0]))
        assert np.allclose(curves[2].end, np.array([-1.0, 1.0]))
        assert np.allclose(curves[3].start, np.array([-1.0, 1.0]))
        assert np.allclose(curves[3].end, np.array([-1.0, -1.0]))

    def test_create_with_non_square_rectangle(self):
        bottom_left = (0.0, 0.0)
        top_right = (2.0, 1.0)
        quad = RectangleFactory.create(bottom_left, top_right)

        assert isinstance(quad, Quad)
        assert len(quad.curves) == 4

        curves = quad.curves
        assert np.allclose(curves[0].start, np.array([0.0, 0.0]))
        assert np.allclose(curves[0].end, np.array([2.0, 0.0]))
        assert np.allclose(curves[1].start, np.array([2.0, 0.0]))
        assert np.allclose(curves[1].end, np.array([2.0, 1.0]))
        assert np.allclose(curves[2].start, np.array([2.0, 1.0]))
        assert np.allclose(curves[2].end, np.array([0.0, 1.0]))
        assert np.allclose(curves[3].start, np.array([0.0, 1.0]))
        assert np.allclose(curves[3].end, np.array([0.0, 0.0]))

    def test_create_with_invalid_points(self):
        bottom_left = (1.0, 1.0)
        top_right = (0.0, 0.0)
        quad = RectangleFactory.create(bottom_left, top_right)

        assert isinstance(quad, Quad)
        assert len(quad.curves) == 4

        curves = quad.curves
        assert np.allclose(curves[0].start, np.array([1.0, 1.0]))
        assert np.allclose(curves[0].end, np.array([0.0, 1.0]))
        assert np.allclose(curves[1].start, np.array([0.0, 1.0]))
        assert np.allclose(curves[1].end, np.array([0.0, 0.0]))
        assert np.allclose(curves[2].start, np.array([0.0, 0.0]))
        assert np.allclose(curves[2].end, np.array([1.0, 0.0]))
        assert np.allclose(curves[3].start, np.array([1.0, 0.0]))
        assert np.allclose(curves[3].end, np.array([1.0, 1.0]))


class TestQuadFactory:
    def test_create_with_valid_points(self):
        p1 = (0.0, 0.0)
        p2 = (1.0, 0.0)
        p3 = (1.0, 1.0)
        p4 = (0.0, 1.0)
        quad = QuadFactory.create(p1, p2, p3, p4)

        assert isinstance(quad, Quad)
        assert len(quad.curves) == 4

        curves = quad.curves
        assert isinstance(curves[0], Line2D)
        assert isinstance(curves[1], Line2D)
        assert isinstance(curves[2], Line2D)
        assert isinstance(curves[3], Line2D)

        assert np.allclose(curves[0].start, np.array([0.0, 0.0]))
        assert np.allclose(curves[0].end, np.array([1.0, 0.0]))
        assert np.allclose(curves[1].start, np.array([1.0, 0.0]))
        assert np.allclose(curves[1].end, np.array([1.0, 1.0]))
        assert np.allclose(curves[2].start, np.array([1.0, 1.0]))
        assert np.allclose(curves[2].end, np.array([0.0, 1.0]))
        assert np.allclose(curves[3].start, np.array([0.0, 1.0]))
        assert np.allclose(curves[3].end, np.array([0.0, 0.0]))

    def test_create_with_trapezoid(self):
        p1 = (0.0, 0.0)
        p2 = (2.0, 0.0)
        p3 = (1.5, 1.0)
        p4 = (0.5, 1.0)
        quad = QuadFactory.create(p1, p2, p3, p4)

        assert isinstance(quad, Quad)
        assert len(quad.curves) == 4

        curves = quad.curves
        assert np.allclose(curves[0].start, np.array([0.0, 0.0]))
        assert np.allclose(curves[0].end, np.array([2.0, 0.0]))
        assert np.allclose(curves[1].start, np.array([2.0, 0.0]))
        assert np.allclose(curves[1].end, np.array([1.5, 1.0]))
        assert np.allclose(curves[2].start, np.array([1.5, 1.0]))
        assert np.allclose(curves[2].end, np.array([0.5, 1.0]))
        assert np.allclose(curves[3].start, np.array([0.5, 1.0]))
        assert np.allclose(curves[3].end, np.array([0.0, 0.0]))

    def test_create_with_concave_quad(self):
        p1 = (0.0, 0.0)
        p2 = (2.0, 0.0)
        p3 = (1.0, 1.0)
        p4 = (1.0, -1.0)
        quad = QuadFactory.create(p1, p2, p3, p4)

        assert isinstance(quad, Quad)
        assert len(quad.curves) == 4

        curves = quad.curves
        assert np.allclose(curves[0].start, np.array([0.0, 0.0]))
        assert np.allclose(curves[0].end, np.array([2.0, 0.0]))
        assert np.allclose(curves[1].start, np.array([2.0, 0.0]))
        assert np.allclose(curves[1].end, np.array([1.0, 1.0]))
        assert np.allclose(curves[2].start, np.array([1.0, 1.0]))
        assert np.allclose(curves[2].end, np.array([1.0, -1.0]))
        assert np.allclose(curves[3].start, np.array([1.0, -1.0]))
        assert np.allclose(curves[3].end, np.array([0.0, 0.0])) 
