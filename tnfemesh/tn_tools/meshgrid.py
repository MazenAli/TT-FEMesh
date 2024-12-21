import torchtt as tntt
from tnfemesh.types import TensorTrain
from tnfemesh.tn_tools.operations import zorder_kron


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

def range_meshgrid2d(d: int) -> TensorTrain:
    """
    Compute the meshgrid corresponding to X and Y tensors counting from 0 to 2**d.

    Args:
        d (int): Exponent of 1D grid size.

    Returns:
        TensorTrain resulting from the meshgrid of two index tensors.
    """

    range = tntt._extras.xfun([2]*d)
    return zmeshgrid2d(range, range)
