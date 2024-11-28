from abc import ABC, abstractmethod
import numpy as np


class Curve(ABC):
    """
    Abstract base class for a curve in the domain.
    Defines the interface for all curve implementations.
    """

    @abstractmethod
    def evaluate(self, t: np.ndarray) -> np.ndarray:
        """
        Evaluate the curve at parameter values t.

        Args:
            t (np.ndarray): Array of parameter values in [0, 1].

        Returns:
            np.ndarray: Array of shape (len(t), 2) with (x, y) coordinates.
        """
        pass

    @abstractmethod
    def tangent(self, t: np.ndarray) -> np.ndarray:
        """
        Compute the tangent vector (not normalized) to the curve at parameter values t.

        Args:
            t (np.ndarray): Array of parameter values in [0, 1].

        Returns:
            np.ndarray: Array of shape (len(t), 2) with tangent vectors.
        """
        pass

    def equals(self, other: "Curve", num_points: int = 100, tol: float = 1e-6) -> bool:
        """
        Check if two curves are approximately equal by sampling.

        Args:
            other (Curve): Another curve to compare.
            num_points (int): Number of points to sample along the curve.
            tol (float): Tolerance for point-wise comparison.

        Returns:
            bool: True if the curves are approximately equal, False otherwise.
        """

        ts = np.linspace(0, 1, num_points)
        for t in ts:
            if not np.allclose(self.evaluate(t), other.evaluate(t), atol=tol):
                return False
        return True


class Line2D(Curve):
    def __init__(self, start: tuple[float, float], end: tuple[float, float]):
        """
        Initialize a line segment from `start` to `end`.

        Args:
            start (tuple): Coordinates of the start point (x, y).
            end (tuple): Coordinates of the end point (x, y).
        """
        self.start = np.array(start)
        self.end = np.array(end)

    def evaluate(self, t: np.ndarray) -> np.ndarray:
        return np.outer(1 - t, self.start) + np.outer(t, self.end)

    def tangent(self, t: np.ndarray) -> np.ndarray:
        return np.tile(self.end - self.start, (len(t), 1))

    def __repr__(self):
        return f"Line2D(start={tuple(self.start)}, end={tuple(self.end)})"


class CircularArc2D(Curve):
    def __init__(self,
                 center: tuple[float, float],
                 radius: float,
                 start_angle: float = 0.,
                 angle_sweep: float = np.pi):
        """
        Initialize a circular arc defined by a center, radius, and angle sweep.

        Args:
            center (tuple): Coordinates of the center (x, y).
            radius (float): Radius of the half-circle.
            start_angle (float): Starting angle in radians. Default is 0.
            angle_sweep (float): Angle sweep in radians. Default is π.
        """
        self.center = np.array(center)
        self.radius = radius
        self.start_angle = start_angle
        self.angle_sweep = angle_sweep

    def evaluate(self, t: np.ndarray) -> np.ndarray:
        angle = self.start_angle + t * self.angle_sweep
        x = self.center[0] + self.radius * np.cos(angle)
        y = self.center[1] + self.radius * np.sin(angle)
        return np.stack((x, y), axis=-1)

    def tangent(self, t: np.ndarray) -> np.ndarray:
        angle = self.start_angle + t * self.angle_sweep
        mul_factor = self.angle_sweep * self.radius
        dx = -np.sin(angle)*mul_factor
        dy = np.cos(angle)*mul_factor
        tangent = np.stack((dx, dy), axis=-1)
        return tangent

    def __repr__(self):
        return (f"CircularArc2D(center={tuple(self.center)}, "
                f"radius={self.radius}, start_angle={self.start_angle}, "
                f"angle_sweep={self.angle_sweep})")


class ParametricCurve2D(Curve):
    def __init__(self, x_func: callable, y_func: callable):
        """
        Initialize a parametric curve defined by a functions x(t) and y(t).
        Uses a finite difference approximation to compute the tangent.

        Args:
            x_func (callable): Function x(t) where t is in [0, 1].
            y_func (callable): Function y(t) where t is in [0, 1].
        """
        self.x_func = x_func
        self.y_func = y_func

    def evaluate(self, t: np.ndarray) -> np.ndarray:
        x = self.x_func(t)
        y = self.y_func(t)
        return np.stack((x, y), axis=-1)

    def tangent(self, t: np.ndarray) -> np.ndarray:
        dt = 1e-5
        dx = (self.x_func(t + dt) - self.x_func(t)) / dt
        dy = (self.y_func(t + dt) - self.y_func(t)) / dt
        tangent = np.stack((dx, dy), axis=-1)
        return tangent

    def __repr__(self):
        return f"ParametricCurve2D(x_func={self.x_func}, y_func={self.y_func})"