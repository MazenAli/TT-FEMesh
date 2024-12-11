from typing import Union
import numpy as np


def ensure_1d(array_or_scalar: Union[np.ndarray, float]) -> np.ndarray:
    """
    Ensure the input is a 1D array.

    Args:
        array_or_scalar (Union[np.ndarray, float]): Input that can be a 1D array
            of shape (num_pts,) or a scalar.

    Returns:
        np.ndarray: A 1D array. If input is a scalar, it is converted to a 1D array
        with a single element.
    """

    if np.isscalar(array_or_scalar):
        return np.array([array_or_scalar], dtype=float)
    array_or_scalar = np.asarray(array_or_scalar)
    if array_or_scalar.ndim == 1:
        return array_or_scalar
    else:
        raise ValueError(f"Input must be a scalar or a 1D array, got {array_or_scalar.ndim} dimensions.")

def ensure_2d(array: np.ndarray) -> np.ndarray:
    """
    Ensure the input array has shape (num_pts, dim).

    Args:
        array (np.ndarray): Input array of shape (num_pts, dim) or (dim,).

    Returns:
        np.ndarray: A 2D array of shape (num_pts, dim).
    """

    array = np.asarray(array)
    if array.ndim == 1:
        return array[np.newaxis, :]
    elif array.ndim == 2:
        return array
    else:
        raise ValueError(f"Input array must have 1 or 2 dimensions, got {array.ndim}.")
