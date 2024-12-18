from typing import Tuple
import numpy as np

def bindex2dtuple(bindex: np.ndarray) -> Tuple[int, int]:
    """
    Convert a binary index to a 2D tuple.

    Args:
        bindex (np.ndarray): A binary index of shape (2 * num_bits1d,).

    Returns:
        Tuple[int, int]: A 2D tuple.

    Raises:
        ValueError: If bindex is not a 1D array, does not contain an even number of elements, or
            contains values other than 0 or 1.
    """

    shape = bindex.shape
    if len(shape) != 1:
        raise ValueError(f"Invalid shape ({shape}) for binary index. Expected 1D array.")

    if shape[0] % 2 != 0:
        raise ValueError("Binary index must have even number of elements.")

    if not np.all((bindex == 0) | (bindex == 1)):
        raise ValueError("Binary index must contain only 0s and 1s.")

    i_bits = bindex[::2]
    j_bits = bindex[1::2]

    i = np.sum(i_bits * (2 ** np.arange(len(i_bits))))
    j = np.sum(j_bits * (2 ** np.arange(len(j_bits))))

    return (i, j)

def qindex2dtuple(index: np.ndarray) -> Tuple[int, int]:
    """
    Convert a quaternary index to a 2D tuple.
    The ordering of the quaternary index is assumed to be
    (0, 1, 2, 3) -> ((0, 0), (1, 0), (0, 1), (1, 1)), i.e., column-major with
    index = i + 2*j.

    Args:
        index (np.ndarray): An index of shape (num_quats,) with values in {0, 1, 2, 3}.

    Returns:
        Tuple[int, int]: A 2D tuple.

    Raises:
        ValueError: If index is not a 1D array or contains values other than {0, 1, 2, 3}.
    """

    shape = index.shape
    if len(shape) != 1:
        raise ValueError(f"Invalid shape ({shape}) for index. Expected 1D array.")

    if not np.all((0 <= index) & (index <= 3)):
        raise ValueError("Index must contain only values in {0, 1, 2, 3}.")

    i = index % 2
    j = index // 2

    binary_index = np.empty(2 * len(index), dtype=int)
    binary_index[::2] = i
    binary_index[1::2] = j

    return bindex2dtuple(binary_index)
