import torch
from tnfemesh.types import TT


def zorder_kron(left: TT, right: TT) -> TT:
    """
    Compute the Kronecker product of two TT-tensors using the Z-ordering.

    Args:
        left: Left TT-tensor.
        right: Right TT-tensor.

    Returns:
        TT-tensor resulting from the Kronecker product of left and right.

    Raises:
        ValueError: If the TT-length of left and right tensors are not equal.
    """

    cores_left = left.cores
    cores_right = right.cores

    if len(cores_left) != len(cores_right):
        raise ValueError(f"TT-length of left ({len(cores_left)})"
                         f" and right ({len(cores_right)}) tensors must be equal.")

    cores = [torch.kron(b, a) for a, b in zip(cores_left, cores_right)]

    return TT(cores)


# Aliases
transpose_kron = zorder_kron
levelwise_kron = zorder_kron