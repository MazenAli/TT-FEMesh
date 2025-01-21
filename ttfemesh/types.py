from typing import TypeAlias
from enum import Enum, auto
import torchtt

TensorTrain: TypeAlias = torchtt.TT

class BoundarySide2D(Enum):
    BOTTOM = 0
    RIGHT = auto()
    TOP = auto()
    LEFT = auto()

class BoundaryVertex2D(Enum):
    BOTTOM_LEFT = 0
    BOTTOM_RIGHT = auto()
    TOP_RIGHT = auto()
    TOP_LEFT = auto()
