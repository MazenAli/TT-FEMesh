from typing import Callable
import numpy as np
from ttfemesh.types import TensorTrain
from ttfemesh.tn_tools.meshgrid import range_meshgrid2d
from ttfemesh.tn_tools.operations import zorder_linfunc2d


def interpolate_linear2d(func: Callable[[np.ndarray], float], d) -> TensorTrain:
    """
    Interpolate a function on a 2D grid using linear interpolation.
    The function takes a quaternary argument index, corresponding to an index on the grid,
    arranged in the z-order, and returns a float value,
    corresponding to the function value at that point.

    Args:
        func (Callable[[ndarray], float]): Function to interpolate.
        d (int): Exponent of the 1D grid size.

    Returns:
        TensorTrain resulting from the linear interpolation of the function.
    """

    num_total = 2 ** d
    index0 = np.zeros(d, dtype=int)
    indexn0 = np.zeros(d, dtype=int)
    index0n = np.zeros(d, dtype=int)
    indexn0[::2] = 3
    index0n[1::2] = 3


    c = func(index0)
    cx = (func(indexn0) - c) / (num_total-1.)
    cy = (func(index0n) - c) / (num_total-1.)

    XX, YY = range_meshgrid2d(d)
    result = zorder_linfunc2d(c, cx, XX, cy, YY)

    return result
