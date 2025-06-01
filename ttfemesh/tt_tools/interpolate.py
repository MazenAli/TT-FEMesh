from typing import Callable

import numpy as np

from ttfemesh.tt_tools.meshgrid import range_meshgrid2d
from ttfemesh.tt_tools.operations import zorder_linfunc2d
from ttfemesh.types import TensorTrain


def interpolate_linear2d(
    func: Callable[[np.ndarray], float], mesh_size_exponent: int
) -> TensorTrain:
    """
    Interpolate a function on a 2D grid using linear interpolation.
    The function takes a quaternary argument index, corresponding to an index on the grid,
    arranged in the z-order, and returns a float value,
    corresponding to the function value at that point.

    Args:
        func (Callable[[ndarray], float]): Function to interpolate.
        mesh_size_exponent (int): Exponent of the 1D grid size.

    Returns:
        TensorTrain resulting from the linear interpolation of the function.
    """

    num_total = 2**mesh_size_exponent
    index0 = np.zeros(mesh_size_exponent, dtype=int)
    indexn0 = np.zeros(mesh_size_exponent, dtype=int)
    index0n = np.zeros(mesh_size_exponent, dtype=int)
    indexn0[::] = 1
    index0n[::] = 2

    c = func(index0)
    cx = (func(indexn0) - c) / (num_total - 1.0)
    cy = (func(index0n) - c) / (num_total - 1.0)

    XX, YY = range_meshgrid2d(mesh_size_exponent)
    result = zorder_linfunc2d(c, cx, XX, cy, YY)

    return result
