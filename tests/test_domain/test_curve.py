import numpy as np
import pytest

from ttfemesh.domain.curve import CircularArc2D, Curve, Line2D, ParametricCurve2D


class TestCurve:
    def test_abstract_base_class_cannot_be_instantiated(self):
        with pytest.raises(TypeError):
            Curve()

    def test_abstract_methods_must_be_implemented(self):
        class IncompleteCurve(Curve):
            def evaluate(self, t):
                pass

        with pytest.raises(TypeError):
            IncompleteCurve()

    def test_validate_parameter_range(self):
        class TestCurveImpl(Curve):
            def evaluate(self, t):
                return np.array([0, 0])

            def tangent(self, t):
                return np.array([0, 0])

        curve = TestCurveImpl()

        # Test valid parameter
        curve._validate(0.0)

        # Test parameter with warning
        with pytest.warns(UserWarning):
            curve._validate(1.1)

        # Test array of parameters
        curve._validate(np.array([-0.5, 0.0, 0.5]))

        # Test array with warning
        with pytest.warns(UserWarning):
            curve._validate(np.array([-1.1, 0.0, 1.1]))

    def test_get_start_end_points(self):
        class TestCurveImpl(Curve):
            def evaluate(self, t):
                return np.array([[t, t**2]])

            def tangent(self, t):
                return np.array([[1, 2 * t]])

        curve = TestCurveImpl()
        assert np.allclose(curve.get_start(), np.array([-1, 1]))
        assert np.allclose(curve.get_end(), np.array([1, 1]))

    def test_call_method(self):
        class TestCurveImpl(Curve):
            def evaluate(self, t):
                return np.array([[t, t**2]])

            def tangent(self, t):
                return np.array([[1, 2 * t]])

        curve = TestCurveImpl()
        assert np.allclose(curve(0.0), np.array([0, 0]))
        assert np.allclose(curve(1.0), np.array([1, 1]))

    def test_equals_method(self):
        class TestCurveImpl(Curve):
            def __init__(self, offset=0):
                self.offset = offset

            def evaluate(self, t):
                return np.array([t + self.offset, t**2])

            def tangent(self, t):
                return np.array([1, 2 * t])

        curve1 = TestCurveImpl()
        curve2 = TestCurveImpl()
        curve3 = TestCurveImpl(offset=1)

        assert curve1.equals(curve2)
        assert not curve1.equals(curve3)

    def test_equals_reverse_direction(self):
        class TestCurveImpl(Curve):
            def __init__(self, reverse=False):
                self.reverse = reverse

            def evaluate(self, t):
                t_ = t if not self.reverse else -t
                return np.array([[t_, t_**2]])

            def tangent(self, t):
                t_ = t if not self.reverse else -t
                return np.array([[1, 2 * t_]]) if not self.reverse else np.array([[1, -2 * t_]])

        curve1 = TestCurveImpl()
        curve2 = TestCurveImpl(reverse=True)
        curve3 = Line2D((-1, 1), (1, 1))

        assert curve1.equals(curve2)
        assert not curve1.equals(curve3)


class TestLine2D:
    @pytest.fixture
    def sample_line(self):
        return Line2D((0.0, 0.0), (1.0, 1.0))

    def test_initialization(self):
        line = Line2D((0.0, 0.0), (1.0, 1.0))
        assert np.allclose(line.start, np.array([0.0, 0.0]))
        assert np.allclose(line.end, np.array([1.0, 1.0]))

    def test_evaluate(self, sample_line):
        # Test scalar input
        assert np.allclose(sample_line.evaluate(-1.0), np.array([0.0, 0.0]))
        assert np.allclose(sample_line.evaluate(0.0), np.array([0.5, 0.5]))
        assert np.allclose(sample_line.evaluate(1.0), np.array([1.0, 1.0]))

        # Test array input
        t = np.array([-1.0, 0.0, 1.0])
        expected = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
        assert np.allclose(sample_line.evaluate(t), expected)

    def test_tangent(self, sample_line):
        # Test scalar input
        assert np.allclose(sample_line.tangent(-1.0), np.array([0.5, 0.5]))
        assert np.allclose(sample_line.tangent(0.0), np.array([0.5, 0.5]))
        assert np.allclose(sample_line.tangent(1.0), np.array([0.5, 0.5]))

        # Test array input
        t = np.array([-1.0, 0.0, 1.0])
        expected = np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])
        assert np.allclose(sample_line.tangent(t), expected)

    def test_repr(self, sample_line):
        assert repr(sample_line) == "Line2D(start=(0.0, 0.0), end=(1.0, 1.0))"


class TestCircularArc2D:
    @pytest.fixture
    def sample_arc(self):
        return CircularArc2D((0.0, 0.0), 1.0, 0.0, np.pi)

    def test_initialization(self):
        arc = CircularArc2D((0.0, 0.0), 1.0, 0.0, np.pi)
        assert np.allclose(arc.center, np.array([0.0, 0.0]))
        assert arc.radius == 1.0
        assert arc.start_angle == 0.0
        assert arc.angle_sweep == np.pi

    def test_evaluate(self, sample_arc):
        # Test scalar input
        assert np.allclose(sample_arc.evaluate(-1.0), np.array([1.0, 0.0]))
        assert np.allclose(sample_arc.evaluate(0.0), np.array([0.0, 1.0]))
        assert np.allclose(sample_arc.evaluate(1.0), np.array([-1.0, 0.0]))

        # Test array input
        t = np.array([-1.0, 0.0, 1.0])
        expected = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
        assert np.allclose(sample_arc.evaluate(t), expected)

    def test_tangent(self, sample_arc):
        # Test scalar input
        assert np.allclose(sample_arc.tangent(-1.0), np.array([0.0, np.pi / 2]))
        assert np.allclose(sample_arc.tangent(0.0), np.array([-np.pi / 2, 0.0]))
        assert np.allclose(sample_arc.tangent(1.0), np.array([0.0, -np.pi / 2]))

        # Test array input
        t = np.array([-1.0, 0.0, 1.0])
        expected = np.array([[0.0, np.pi / 2], [-np.pi / 2, 0.0], [0.0, -np.pi / 2]])
        assert np.allclose(sample_arc.tangent(t), expected)

    def test_repr(self, sample_arc):
        assert (
            repr(sample_arc) == "CircularArc2D(center=(0.0, 0.0), "
            "radius=1.0, start_angle=0.0, angle_sweep=3.141592653589793)"  # noqa
        )


class TestParametricCurve2D:
    @pytest.fixture
    def sample_parametric(self):
        def x_func(t):
            return t

        def y_func(t):
            return t**2

        return ParametricCurve2D(x_func, y_func)

    def test_initialization(self):
        def x_func(t):
            return t

        def y_func(t):
            return t**2

        curve = ParametricCurve2D(x_func, y_func)
        assert curve.x_func == x_func
        assert curve.y_func == y_func

    def test_evaluate(self, sample_parametric):
        # Test scalar input
        assert np.allclose(sample_parametric.evaluate(-1.0), np.array([-1.0, 1.0]))
        assert np.allclose(sample_parametric.evaluate(0.0), np.array([0.0, 0.0]))
        assert np.allclose(sample_parametric.evaluate(1.0), np.array([1.0, 1.0]))

        # Test array input
        t = np.array([-1.0, 0.0, 1.0])
        expected = np.array([[-1.0, 1.0], [0.0, 0.0], [1.0, 1.0]])
        assert np.allclose(sample_parametric.evaluate(t), expected)

    def test_tangent(self, sample_parametric):
        # Test scalar input
        t = 0.0
        tangent = sample_parametric.tangent(t)
        assert tangent.shape == (1, 2)
        assert abs(tangent[0, 0] - 1.0) < 1e-4  # dx/dt ≈ 1
        assert abs(tangent[0, 1] - 0.0) < 1e-4  # dy/dt ≈ 0

        # Test array input
        t = np.array([-1.0, 0.0, 1.0])
        tangent = sample_parametric.tangent(t)
        assert tangent.shape == (3, 2)
        assert np.allclose(tangent[:, 0], np.array([1.0, 1.0, 1.0]), atol=1e-4)
        assert np.allclose(tangent[:, 1], np.array([-2.0, 0.0, 2.0]), atol=1e-4)

    def test_repr(self, sample_parametric):
        assert repr(sample_parametric).startswith("ParametricCurve2D")
