import matplotlib.pyplot as plt
import numpy as np

from ttfemesh.utils.plot_helpers import plot_curve_with_tangents


class MockCurve:
    def __init__(self, points, tangents):
        self.points = points
        self.tangents = tangents

    def evaluate(self, _):
        return self.points

    def tangent(self, _):
        return self.tangents


class TestPlotCurveWithTangents:
    def test_basic_plotting(self):
        points = np.array([[0, 0], [1, 1], [2, 2]])
        tangents = np.array([[1, 1], [1, 1], [1, 1]])
        curve = MockCurve(points, tangents)

        plot_curve_with_tangents(curve, "Test Curve", num_points=3)

        fig = plt.gcf()
        axes = fig.gca()

        assert axes.get_title() == "Test Curve"
        assert axes.get_xlabel() == "x"
        assert axes.get_ylabel() == "y"
        assert axes.get_legend() is not None

        plt.close(fig)

    def test_different_num_points(self):
        points = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]])
        tangents = np.array([[1, 1], [1, 1], [1, 1], [1, 1], [1, 1]])
        curve = MockCurve(points, tangents)

        for num_points in [2, 3, 4]:
            plot_curve_with_tangents(curve, f"Test Curve {num_points}", num_points=num_points)
            fig = plt.gcf()
            plt.close(fig)

    def test_plot_components(self):
        points = np.array([[0, 0], [1, 1], [2, 2]])
        tangents = np.array([[1, 0], [0, 1], [-1, 0]])
        curve = MockCurve(points, tangents)

        plot_curve_with_tangents(curve, "Test Components", num_points=3)
        fig = plt.gcf()
        axes = fig.gca()

        assert len(axes.lines) > 0
        assert len(axes.collections) > 0
        assert axes.get_xlim()[0] <= 0
        assert axes.get_ylim()[0] <= 0

        plt.close(fig)
