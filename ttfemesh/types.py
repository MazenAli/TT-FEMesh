from enum import Enum, auto
from typing import TypeAlias

import torchtt

TensorTrain: TypeAlias = torchtt.TT


class BoundarySide2D(Enum):
    """
    Enum for the sides of a 2D domain.
    Note that the sides are considered to be ordered as follows:
    bottom (side 0), right (side 1), top (side 2), left (side 3).
    This is important for the boundary condition to work correctly.
    It may lead to confusion if, e.g., your side 0 is visually
    the right edge of the domain.
    """
    BOTTOM = 0
    RIGHT = auto()
    TOP = auto()
    LEFT = auto()


class BoundaryVertex2D(Enum):
    """
    Enum for the vertices of a 2D domain.
    Note that the vertices are considered to be ordered as follows:
    bottom_left (vertex 0), bottom_right (vertex 1), top_right (vertex 2), top_left (vertex 3).
    This is important for the boundary condition to work correctly.
    It may lead to confusion if, e.g., your vertex 0 is visually
    the bottom left corner of the domain.
    """
    BOTTOM_LEFT = 0
    BOTTOM_RIGHT = auto()
    TOP_RIGHT = auto()
    TOP_LEFT = auto()
