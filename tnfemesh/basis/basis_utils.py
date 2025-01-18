from typing import Tuple
import numpy as np
import torch
from tnfemesh.types import TensorTrain


def left_corner2index_ttmap(d: int) -> TensorTrain:
    """
    Returns the TT-representation of the left corner element index to global basis index map (W_0).
    See arXiv:1802.02839, Section 5.4 for more details.
    Note the size of the TT alone already implies we are working with a linear 2D basis.

    Args:
        d (int): Exponent of the 1D mesh size (length of TT).

    Returns:
        TensorTrain: TT-representation of the left corner to global index map.
    """
    firstcore, middlecore, lastcore = left_corner2index_map_ttcores()
    cores = [firstcore] + [middlecore] * (d - 2) + [lastcore]
    torch_cores = [torch.tensor(core) for core in cores]

    return TensorTrain(torch_cores)

def right_corner2index_ttmap(d: int) -> TensorTrain:
    """
    Returns the TT-representation of the right corner element index to global basis index map (W_1).
    See arXiv:1802.02839, Section 5.4 for more details.
    Note the size of the TT alone already implies we are working with a linear 2D basis.

    Args:
        d (int): Exponent of the 1D mesh size (length of TT).

    Returns:
        TensorTrain: TT-representation of the right corner to global index map
    """
    firstcore, middlecore, lastcore = right_corner2index_map_ttcores()
    cores = [firstcore] + [middlecore] * (d - 2) + [lastcore]
    torch_cores = [torch.tensor(core) for core in cores]

    return TensorTrain(torch_cores)

def left_corner2index_map_ttcores() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns the TT-cores for the left corner element index to global basis index map (W_0).
    See arXiv:1802.02839, Section 5.4 for more details.
    We reverse the order of the cores for consistency with the little-endian
    convention in the QTT literature, i.e.,
    column-major or Fortran-style ordering.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: TT-cores of the left corner to global index map.
    """

    firstcore = np.zeros((1, 2, 2, 2))
    firstcore[0, :, :, 0] = np.array([[1.0, 0.0], [0.0, 1.0]])
    firstcore[0, :, :, 1] = np.array([[1.0, 0.0], [0.0, 0.0]])

    middlecore = np.zeros((2, 2, 2, 2))
    middlecore[0, :, :, 0] = np.array([[1.0, 0.0], [0.0, 1.0]])
    middlecore[0, :, :, 1] = np.array([[1.0, 0.0], [0.0, 0.0]])
    middlecore[1, :, :, 0] = np.array([[0.0, 0.0], [0.0, 0.0]])
    middlecore[1, :, :, 1] = np.array([[0.0, 0.0], [0.0, 1.0]])

    lastcore = np.zeros((2, 2, 2, 1))
    lastcore[0, :, :, 0] = np.array([[1.0, 0.0], [0.0, 0.0]])
    lastcore[1, :, :, 0] = np.array([[0.0, 0.0], [0.0, 1.0]])

    cores = firstcore, middlecore, lastcore
    return cores

def right_corner2index_map_ttcores() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns the TT-cores for the right corner element index to global basis index map (W_1).
    See arXiv:1802.02839, Section 5.4 for more details.
    We reverse the order of the cores for consistency with the little-endian
    convention in the QTT literature, i.e.,
    column-major or Fortran-style ordering.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: TT-cores of the right corner to global index map.
    """
    firstcore = np.zeros((1, 2, 2, 2))
    firstcore[0, :, :, 0] = np.array([[0.0, 1.0], [0.0, 0.0]])
    firstcore[0, :, :, 1] = np.array([[0.0, 0.0], [1.0, 0.0]])


    middlecore = np.zeros((2, 2, 2, 2))
    middlecore[0, :, :, 0] = np.array([[0.0, 1.0], [0.0, 0.0]])
    middlecore[0, :, :, 1] = np.array([[0.0, 0.0], [1.0, 0.0]])
    middlecore[1, :, :, 0] = np.array([[0.0, 0.0], [0.0, 0.0]])
    middlecore[1, :, :, 1] = np.array([[1.0, 0.0], [0.0, 1.0]])

    lastcore = np.zeros((2, 2, 2, 1))
    lastcore[0, :, :, 0] = np.array([[0.0, 0.0], [1.0, 0.0]])
    lastcore[1, :, :, 0] = np.array([[1.0, 0.0], [0.0, 1.0]])

    cores = firstcore, middlecore, lastcore
    return cores

