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
            @staticmethod
            def create(*args, **kwargs):
                pass

        with pytest.raises(TypeError):
            IncompleteFactory()


class TestRectangleFactory:
    def test_create_with_valid_points(self):
        # Test with a simple unit square
        bottom_left = (0.0, 0.0)
        top_right = (1.0, 1.0)
        quad = RectangleFactory.create(bottom_left, top_right)

        assert isinstance(quad, Quad)
        assert len(quad.curves) == 4

        # Verify the curves are in the correct order and have correct points
        curves = quad.curves
        assert isinstance(curves[0], Line2D)
        assert isinstance(curves[1], Line2D)
        assert isinstance(curves[2], Line2D)
        assert isinstance(curves[3], Line2D)

        # Check bottom curve
        assert np.allclose(curves[0].start, np.array([0.0, 0.0]))
        assert np.allclose(curves[0].end, np.array([1.0, 0.0]))

        # Check right curve
        assert np.allclose(curves[1].start, np.array([1.0, 0.0]))
        assert np.allclose(curves[1].end, np.array([1.0, 1.0]))

        # Check top curve
        assert np.allclose(curves[2].start, np.array([1.0, 1.0]))
        assert np.allclose(curves[2].end, np.array([0.0, 1.0]))

        # Check left curve
        assert np.allclose(curves[3].start, np.array([0.0, 1.0]))
        assert np.allclose(curves[3].end, np.array([0.0, 0.0]))

    def test_create_with_negative_coordinates(self):
        # Test with negative coordinates
        bottom_left = (-1.0, -1.0)
        top_right = (1.0, 1.0)
        quad = RectangleFactory.create(bottom_left, top_right)

        assert isinstance(quad, Quad)
        assert len(quad.curves) == 4

        # Verify the curves are in the correct order and have correct points
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
        # Test with a non-square rectangle
        bottom_left = (0.0, 0.0)
        top_right = (2.0, 1.0)
        quad = RectangleFactory.create(bottom_left, top_right)

        assert isinstance(quad, Quad)
        assert len(quad.curves) == 4

        # Verify the curves are in the correct order and have correct points
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
        # Test with invalid points (top_right not above and to the right of bottom_left)
        bottom_left = (1.0, 1.0)
        top_right = (0.0, 0.0)
        quad = RectangleFactory.create(bottom_left, top_right)

        # The factory should still create a valid quad, just with the points swapped
        assert isinstance(quad, Quad)
        assert len(quad.curves) == 4

        # Verify the curves are in the correct order and have correct points
        curves = quad.curves
        assert np.allclose(curves[0].start, np.array([0.0, 0.0]))
        assert np.allclose(curves[0].end, np.array([1.0, 0.0]))
        assert np.allclose(curves[1].start, np.array([1.0, 0.0]))
        assert np.allclose(curves[1].end, np.array([1.0, 1.0]))
        assert np.allclose(curves[2].start, np.array([1.0, 1.0]))
        assert np.allclose(curves[2].end, np.array([0.0, 1.0]))
        assert np.allclose(curves[3].start, np.array([0.0, 1.0]))
        assert np.allclose(curves[3].end, np.array([0.0, 0.0]))


class TestQuadFactory:
    def test_create_with_valid_points(self):
        # Test with a simple quadrilateral
        p1 = (0.0, 0.0)
        p2 = (1.0, 0.0)
        p3 = (1.0, 1.0)
        p4 = (0.0, 1.0)
        quad = QuadFactory.create(p1, p2, p3, p4)

        assert isinstance(quad, Quad)
        assert len(quad.curves) == 4

        # Verify the curves are in the correct order and have correct points
        curves = quad.curves
        assert isinstance(curves[0], Line2D)
        assert isinstance(curves[1], Line2D)
        assert isinstance(curves[2], Line2D)
        assert isinstance(curves[3], Line2D)

        # Check each curve
        assert np.allclose(curves[0].start, np.array([0.0, 0.0]))
        assert np.allclose(curves[0].end, np.array([1.0, 0.0]))
        assert np.allclose(curves[1].start, np.array([1.0, 0.0]))
        assert np.allclose(curves[1].end, np.array([1.0, 1.0]))
        assert np.allclose(curves[2].start, np.array([1.0, 1.0]))
        assert np.allclose(curves[2].end, np.array([0.0, 1.0]))
        assert np.allclose(curves[3].start, np.array([0.0, 1.0]))
        assert np.allclose(curves[3].end, np.array([0.0, 0.0]))

    def test_create_with_trapezoid(self):
        # Test with a trapezoid
        p1 = (0.0, 0.0)
        p2 = (2.0, 0.0)
        p3 = (1.5, 1.0)
        p4 = (0.5, 1.0)
        quad = QuadFactory.create(p1, p2, p3, p4)

        assert isinstance(quad, Quad)
        assert len(quad.curves) == 4

        # Verify the curves are in the correct order and have correct points
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
        # Test with a concave quadrilateral
        p1 = (0.0, 0.0)
        p2 = (2.0, 0.0)
        p3 = (1.0, 1.0)
        p4 = (1.0, -1.0)
        quad = QuadFactory.create(p1, p2, p3, p4)

        assert isinstance(quad, Quad)
        assert len(quad.curves) == 4

        # Verify the curves are in the correct order and have correct points
        curves = quad.curves
        assert np.allclose(curves[0].start, np.array([0.0, 0.0]))
        assert np.allclose(curves[0].end, np.array([2.0, 0.0]))
        assert np.allclose(curves[1].start, np.array([2.0, 0.0]))
        assert np.allclose(curves[1].end, np.array([1.0, 1.0]))
        assert np.allclose(curves[2].start, np.array([1.0, 1.0]))
        assert np.allclose(curves[2].end, np.array([1.0, -1.0]))
        assert np.allclose(curves[3].start, np.array([1.0, -1.0]))
        assert np.allclose(curves[3].end, np.array([0.0, 0.0])) 