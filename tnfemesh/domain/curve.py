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
        Compute the tangent vector to the curve at parameter values t.

        Args:
            t (np.ndarray): Array of parameter values in [0, 1].

        Returns:
            np.ndarray: Array of shape (len(t), 2) with tangent vectors.
        """
        pass


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
        return np.tile(self.end - self.start, (len(t), 1)) / np.linalg.norm(self.end - self.start)


class HalfCircle2D(Curve):
    def __init__(self, center: tuple[float, float], radius: float, top: bool = True):
        """
        Initialize a half-circle curve.

        Args:
            center (tuple): Coordinates of the center (x, y).
            radius (float): Radius of the half-circle.
            top (bool): True (default) for top half, False for bottom.
        """
        self.center = np.array(center)
        self.radius = radius
        self.top = top

    def evaluate(self, t: np.ndarray) -> np.ndarray:
        angle = t * np.pi
        if not self.top:
            angle = 2 * np.pi - angle
        x = self.center[0] + self.radius * np.cos(angle)
        y = self.center[1] + self.radius * np.sin(angle)
        return np.stack((x, y), axis=-1)

    def tangent(self, t: np.ndarray) -> np.ndarray:
        angle = t * np.pi
        if not self.top:
            angle = 2 * np.pi - angle
        dx = -np.sin(angle)
        dy = np.cos(angle)
        tangent = np.stack((dx, dy), axis=-1)
        return tangent / np.linalg.norm(tangent, axis=-1, keepdims=True)


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
        return tangent / np.linalg.norm(tangent, axis=-1, keepdims=True)

