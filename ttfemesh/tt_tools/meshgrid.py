import numpy as np
import torchtt as tntt

from ttfemesh.tt_tools.operations import zorder_kron
from ttfemesh.types import TensorTrain


def zmeshgrid2d(X: TensorTrain, Y: TensorTrain) -> TensorTrain:
    """
    Compute the meshgrid of two TT-tensors using the Z-ordering.

    Args:
        X (TensorTrain): First TT-tensor.
        Y (TensorTrain): Second TT-tensor.

    Returns:
        TensorTrain resulting from the meshgrid of X and Y.
    """

    ones_x = tntt.ones(X.N)
    ones_y = tntt.ones(Y.N)
    XX = zorder_kron(X, ones_y)
    YY = zorder_kron(ones_x, Y)

    return XX, YY


def range_meshgrid2d(mesh_size_exponent: int) -> TensorTrain:
    """
    Compute the meshgrid corresponding to X and Y tensors counting from 0 to 2**d.

    Args:
        mesh_size_exponent (int): Exponent of 1D grid size.

    Returns:
        TensorTrain resulting from the meshgrid of two index tensors.
    """

    range = tntt._extras.xfun([2] * mesh_size_exponent)
    return zmeshgrid2d(range, range)


def map2canonical2d(mesh_size_exponent: int) -> np.ndarray:
    """
    Computes a vector where the i-th element is the index of the i-th element
    in the z-ordering of the meshgrid.
    The canonical ordering for a 2D grid indexed by (i, j) is computed as
    i + j * 2**d and does not correspond to the z-ordering of (i, j).
    When a TT-tensor in the z-ordering is reshaped to a vector,
    the order of the elements is given by the vector returned by this function.
    This effectively allows to map the z-ordering back to the canonical ordering.
    Useful for debugging and testing.

    Args:
        mesh_size_exponent (int): Exponent of 1D grid size.

    Returns:
        np.ndarray: Vector of indices mapping z-ordering to canonical ordering.

    Example:
        .. code-block:: python
            array = tt.full().flatten("F")
            zmap = map2canonical2d(3)
            canonical_array = np.empty_like(array)
            canonical_array[zmap] = array
    """

    meshgrid = range_meshgrid2d(mesh_size_exponent)
    XX, YY = meshgrid
    vecx = np.array(XX.full()).flatten("F")
    vecy = np.array(YY.full()).flatten("F")
    z2tuples = list(zip(vecx, vecy))
    xlen = 2**mesh_size_exponent

    def index_map(pair):
        i, j = pair
        idx = int(j * xlen + i)
        return idx

    zmap = [index_map(pair) for pair in z2tuples]

    return np.array(zmap)
