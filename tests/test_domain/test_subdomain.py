import pytest
import numpy as np
from ttfemesh.domain.subdomain import Subdomain, Subdomain2D, Quad
from ttfemesh.domain.curve import Curve, Line2D
import matplotlib.pyplot as plt


class TestSubdomain:
    def test_abstract_base_class_cannot_be_instantiated(self):
        with pytest.raises(TypeError):
            Subdomain()

    def test_abstract_methods_must_be_implemented(self):
        class IncompleteSubdomain(Subdomain):
            pass

        with pytest.raises(TypeError):
            IncompleteSubdomain()


class TestSubdomain2D:
    def test_initialization_with_valid_curves(self):
        p1 = np.array([0.0, 0.0])
        p2 = np.array([1.0, 0.0])
        p3 = np.array([1.0, 1.0])
        p4 = np.array([0.0, 1.0])
        
        curves = [
            Line2D(p1, p2),
            Line2D(p2, p3),
            Line2D(p3, p4),
            Line2D(p4, p1)
        ]
        
        subdomain = Subdomain2D(curves)
        assert len(subdomain.curves) == 4
        assert all(isinstance(curve, Curve) for curve in subdomain.curves)

    def test_initialization_with_wrong_number_of_curves(self):
        p1 = np.array([0.0, 0.0])
        p2 = np.array([1.0, 0.0])
        p3 = np.array([1.0, 1.0])
        
        curves = [
            Line2D(p1, p2),
            Line2D(p2, p3),
            Line2D(p3, p1)
        ]
        
        with pytest.raises(ValueError, match="A 2D subdomain must be defined by exactly 4 curves."):
            Subdomain2D(curves)

    def test_get_curve_with_valid_index(self):
        p1 = np.array([0.0, 0.0])
        p2 = np.array([1.0, 0.0])
        p3 = np.array([1.0, 1.0])
        p4 = np.array([0.0, 1.0])
        
        curves = [
            Line2D(p1, p2),
            Line2D(p2, p3),
            Line2D(p3, p4),
            Line2D(p4, p1)
        ]
        
        subdomain = Subdomain2D(curves)
        curve = subdomain.get_curve(0)
        assert isinstance(curve, Curve)
        assert np.allclose(curve.get_start(), p1)
        assert np.allclose(curve.get_end(), p2)

    def test_get_curve_with_invalid_index(self):
        p1 = np.array([0.0, 0.0])
        p2 = np.array([1.0, 0.0])
        p3 = np.array([1.0, 1.0])
        p4 = np.array([0.0, 1.0])
        
        curves = [
            Line2D(p1, p2),
            Line2D(p2, p3),
            Line2D(p3, p4),
            Line2D(p4, p1)
        ]
        
        subdomain = Subdomain2D(curves)
        with pytest.raises(ValueError, match="Curve index must be in the range \\[0, 3\\]."):
            subdomain.get_curve(4)
        
        with pytest.raises(ValueError, match="Curve index must be in the range \\[0, 3\\]."):
            subdomain.get_curve(-1)

    def test_check_connect_with_disconnected_curves(self):
        p1 = np.array([0.0, 0.0])
        p2 = np.array([1.0, 0.0])
        p3 = np.array([1.0, 1.0])
        p4 = np.array([0.0, 1.0])
        p5 = np.array([0.1, 0.1]) 
        
        curves = [
            Line2D(p1, p2),
            Line2D(p2, p3),
            Line2D(p3, p4),
            Line2D(p5, p1)
        ]
        
        with pytest.raises(ValueError, match="Curves 2 and 3 do not connect properly."):
            Subdomain2D(curves)

    def test_check_orientation_with_clockwise_curves(self):
        p1 = np.array([0.0, 0.0])
        p2 = np.array([0.0, 1.0])
        p3 = np.array([1.0, 1.0])
        p4 = np.array([1.0, 0.0])
        
        curves = [
            Line2D(p1, p2),
            Line2D(p2, p3),
            Line2D(p3, p4),
            Line2D(p4, p1)
        ]
        
        with pytest.raises(ValueError,
                           match="The start points of curves are not ordered counter-clockwise."):
            Subdomain2D(curves)

    def test_repr(self):
        p1 = np.array([0., 0.])
        p2 = np.array([1., 0.])
        p3 = np.array([1., 1.])
        p4 = np.array([0., 1.])
        
        curves = [
            Line2D(p1, p2),
            Line2D(p2, p3),
            Line2D(p3, p4),
            Line2D(p4, p1)
        ]
        
        subdomain = Subdomain2D(curves)
        repr_str = repr(subdomain)
        assert "Subdomain2D(points=" in repr_str

    def test_plot(self):
        p1 = np.array([0.0, 0.0])
        p2 = np.array([1.0, 0.0])
        p3 = np.array([1.0, 1.0])
        p4 = np.array([0.0, 1.0])
        
        curves = [
            Line2D(p1, p2),
            Line2D(p2, p3),
            Line2D(p3, p4),
            Line2D(p4, p1)
        ]

        subdomain = Subdomain2D(curves)
        try:
            subdomain.plot()
            plt.close()
        except Exception as e:
            pytest.fail(f"Plotting failed with error: {e}")
        


class TestQuad:
    def test_initialization_with_valid_lines(self):
        p1 = np.array([0.0, 0.0])
        p2 = np.array([1.0, 0.0])
        p3 = np.array([1.0, 1.0])
        p4 = np.array([0.0, 1.0])
        
        lines = [
            Line2D(p1, p2),
            Line2D(p2, p3),
            Line2D(p3, p4),
            Line2D(p4, p1)
        ]
        
        quad = Quad(lines)
        assert len(quad.curves) == 4
        assert all(isinstance(curve, Line2D) for curve in quad.curves)

    def test_repr(self):
        p1 = np.array([0.0, 0.0])
        p2 = np.array([1.0, 0.0])
        p3 = np.array([1.0, 1.0])
        p4 = np.array([0.0, 1.0])
        
        lines = [
            Line2D(p1, p2),
            Line2D(p2, p3),
            Line2D(p3, p4),
            Line2D(p4, p1)
        ]
        
        quad = Quad(lines)
        repr_str = repr(quad)
        assert "Quad(points=" in repr_str
