from typing import Tuple
import numpy as np

def bindex2dtuple(bindex: np.ndarray) -> Tuple[int, int]:
    """
    Convert a binary index to a 2D tuple.

    Args:
        bindex (np.ndarray): A binary index of shape (2 * n,).

    Returns:
        Tuple[int, int]: A 2D tuple.
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