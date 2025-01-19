from typing import List
from ttfemesh.types import TensorTrain
import torch

def integer_to_little_endian(length: int, num: int, ) -> List[int]:
    """
    Convert an integer to its little-endian binary representation.

    Args:
        length (int): Number of bits to represent the integer.
        num (int): An integer.

    Returns:
        List[int]: A list of bits representing the integer in little-endian format.

    Raises:
        ValueError: If num is negative or greater than or equal to 2^length.
    """

    if num < 0:
        raise ValueError(f"Input number ({num}) must be non-negative.")
    if num >= 2**length:
        raise ValueError(f"Input number {num} must be less than 2^{length}.")

    return [(num >> i) & 1 for i in range(length)]


def unit_vector_binary_tt(length: int, index: int) -> TensorTrain:
    """
    Generate a unit vector with a 1 at the specified index for a binary quantized tensor train.

    Args:
        length (int): Length of the tensor train.
        index (int): Index of the position of the 1 in the unit vector.

    Returns:
        TensorTrain: A tensor train of dimension length representing the unit vector.

    Raises:
        ValueError: If index is negative or greater than or equal to 2^length.
    """

    if index < 0:
        raise ValueError(f"Index ({index}) must be non-negative.")
    if index >= 2**length:
        raise ValueError(f"Index ({index}) must be less than 2^{length}.")

    cores = [None] * length
    binidx = integer_to_little_endian(length, index)
    for j in range(length):
        core = torch.zeros(1, 2, 1)
        core[0, binidx[j], 0] = 1.
        cores[j] = core

    return TensorTrain(cores)
